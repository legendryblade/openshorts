"""Facecam window extractor for gaming_split layout.

Detects the FULL facecam window (person + upper body + background) in gaming
videos, not just the face. The facecam is a persistent rectangular overlay
in one corner containing the streamer's webcam feed.

This replaces the face-centered tracking with window-based extraction.
"""
import os
import json
import subprocess
import tempfile
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

# Import existing utilities
import screencast_layout
import main as m

# Configuration
FACECAM_MIN_W = 0.10   # minimum width fraction
FACECAM_MAX_W = 0.40   # maximum width fraction
FACECAM_MIN_H = 0.15   # minimum height fraction
FACECAM_MAX_H = 0.50   # maximum height fraction

# Corner regions to search (as fractions of frame)
CORNER_REGIONS = {
    "top-left":      (0.0, 0.0, 0.5, 0.5),
    "top-right":     (0.5, 0.0, 1.0, 0.5),
    "bottom-left":   (0.0, 0.5, 0.5, 1.0),
    "bottom-right":  (0.5, 0.5, 1.0, 1.0),
}

# Minimum samples needed for stable detection
MIN_STABLE_SAMPLES = 3
MAX_CENTRE_SPREAD = 0.05  # max movement between frames as fraction of frame

# Cache file for facecam detection results
FACECAM_CACHE_DIR = os.path.join(tempfile.gettempdir(), "openshorts_facecam_cache")


def _ensure_cache_dir():
    os.makedirs(FACECAM_CACHE_DIR, exist_ok=True)


def _cache_key(video_path: str) -> str:
    """Generate cache key from video path and size."""
    import hashlib
    stat = os.stat(video_path)
    content = f"{video_path}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()


def _load_facecam_cache(video_path: str) -> Optional[Dict]:
    """Load cached facecam detection if valid."""
    _ensure_cache_dir()
    key = _cache_key(video_path)
    cache_file = os.path.join(FACECAM_CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            # Validate required fields
            required = ["x", "y", "w", "h", "corner", "confidence"]
            if all(k in data for k in required):
                return data
        except Exception:
            pass
    return None


def _save_facecam_cache(video_path: str, data: Dict):
    """Save facecam detection to cache."""
    _ensure_cache_dir()
    key = _cache_key(video_path)
    cache_file = os.path.join(FACECAM_CACHE_DIR, f"{key}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def detect_facecam_window(
    video_path: str,
    samples: int = 12,
    use_gemini: bool = True,
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.5-flash"
) -> Optional[Dict]:
    """
    Detect the full facecam window rectangle in a gaming video.
    
    Returns dict with: x, y, w, h (normalized 0-1), corner, confidence, is_stable
    Returns None if no facecam detected or detection unstable.
    """
    # Check cache first
    cached = _load_facecam_cache(video_path)
    if cached:
        print(f"   📦 Using cached facecam detection: {cached['corner']} ({cached['confidence']:.2f})")
        return cached
    
    # First try Gemini if available
    if use_gemini and gemini_api_key:
        gemini_result = _detect_facecam_gemini(video_path, gemini_api_key)
        if gemini_result and gemini_result.get("is_stable", False):
            # Cross-validate with local detection when it works; a Gemini
            # answer at high confidence is trusted even when the local
            # detector finds nothing (YOLO often misses webcam feeds on
            # busy gameplay backgrounds).
            local_result = _detect_facecam_local(video_path)
            if local_result and _rects_match(gemini_result, local_result):
                result = {**gemini_result, "source": "gemini+local"}
            elif gemini_result.get("confidence", 0) >= 0.5:
                result = {**gemini_result, "source": "gemini"}
            else:
                result = None
            if result:
                _save_facecam_cache(video_path, result)
                return result
    
    # Fallback to local detection only
    local_result = _detect_facecam_local(video_path)
    if local_result:
        result = {**local_result, "source": "local"}
        _save_facecam_cache(video_path, result)
        return result
    
    return None


def _detect_facecam_gemini(video_path: str, api_key: str) -> Optional[Dict]:
    """Use Gemini to detect facecam window from sampled frames."""
    try:
        from layout_picker import sample_frames
        from google import genai
        from google.genai import types as genai_types
        import gemini_worker
        
        frames = sample_frames(video_path, n=12, width=1024)
        if not frames:
            return None
        
        client = genai.Client(api_key=api_key)
        parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts + [gemini_worker.FACECAM_DETECTION_PROMPT],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_worker.FacecamDetectionResponse,
            ))
        gemini_worker.raise_if_blocked(response)
        answer = json.loads(response.text) or {}
        
        # Validate
        required = ["x", "y", "w", "h", "confidence", "corner", "is_stable"]
        if all(k in answer for k in required):
            if all(0 <= answer[k] <= 1 for k in ["x", "y", "w", "h"]):
                if 0 <= answer["confidence"] <= 1:
                    if answer["corner"] in ["top-left", "top-right", "bottom-left", "bottom-right"]:
                        return answer
        return None
    except Exception as e:
        print(f"   ⚠️ Gemini facecam detection failed: {e}")
        return None


def _detect_facecam_local(video_path: str) -> Optional[Dict]:
    """Local detection using YOLO + face detection + geometric constraints."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total <= 0 or not frame_w:
        cap.release()
        return None
    
    # Sample frames
    samples = min(12, max(3, total // 300))
    sample_indices = [int(i * total / 12) for i in range(12)]
    
    # Import YOLO model from main
    import main as m
    
    all_boxes = []
    corners_found = []
    
    try:
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            
            # Detect person using YOLO
            person_box = m.detect_person_yolo(frame)
            if not person_box:
                continue
            
            # Also try face detection for more precision
            small, scale = m._detection_frame(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            with m.DETECT_LOCK:
                results = m.face_detection.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            
            face_box = None
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * frame_w)
                    y = int(bbox.ymin * frame_h)
                    w = int(bbox.width * frame_w)
                    h = int(bbox.height * frame_h)
                    face_box = [x, y, w, h]
                    break
            
            # Use the larger of face_box or person_box to encompass full window
            if face_box and person_box:
                # Combine boxes to get full window
                x1 = min(face_box[0], person_box[0])
                y1 = min(face_box[1], person_box[1])
                x2 = max(face_box[0] + face_box[2], person_box[0] + person_box[2])
                y2 = max(face_box[1] + face_box[3], person_box[1] + person_box[3])
                combined = [x1, y1, x2 - x1, y2 - y1]
                box = combined
            elif person_box:
                box = person_box
            elif face_box:
                box = face_box
            else:
                continue
            
            # Check if box is in a corner and right size
            corner = _identify_corner(box, frame_w, frame_h)
            if not corner:
                continue
            
            if not _validate_facecam_size(box, frame_w, frame_h):
                continue
            
            all_boxes.append(box)
            corners_found.append(corner)
        
        cap.release()
        
        if len(all_boxes) < MIN_STABLE_SAMPLES:
            return None
        
        # Check corner consistency
        corner_counts = {}
        for c in corners_found:
            corner_counts[c] = corner_counts.get(c, 0) + 1
        dominant_corner = max(corner_counts, key=corner_counts.get)
        if corner_counts[dominant_corner] < MIN_STABLE_SAMPLES:
            return None
        
        # Filter boxes for dominant corner
        corner_boxes = [b for b, c in zip(all_boxes, corners_found) if c == dominant_corner]
        
        # Check stability (centre spread)
        centres = np.array([[b[0] + b[2]/2, b[1] + b[3]/2] for b in corner_boxes])
        mid = np.median(centres, axis=0)
        spread = np.max(np.abs(centres - mid), axis=0)
        frame_w = corner_boxes[0][2] * 10  # rough estimate, we have frame_w from earlier
        
        if spread[0] > frame_w * MAX_CENTRE_SPREAD or spread[1] > frame_h * MAX_CENTRE_SPREAD:
            return None
        
        # Compute median box
        median_box = tuple(int(v) for v in np.median(np.array(corner_boxes), axis=0))
        
        # Validate size
        if not _validate_facecam_size(median_box, frame_w, frame_h):
            return None
        
        # Normalize
        x, y, w, h = median_box
        return {
            "x": x / frame_w,
            "y": y / frame_h,
            "w": w / frame_w,
            "h": h / frame_h,
            "corner": dominant_corner,
            "confidence": 0.7,  # local detection confidence
            "is_stable": True,
            "source": "local"
        }
        
    finally:
        cap.release()


def _identify_corner(box: List[int], frame_w: int, frame_h: int) -> Optional[str]:
    """Identify which corner the box is in."""
    x, y, w, h = box
    cx = (x + w / 2) / frame_w
    cy = (y + h / 2) / frame_h
    
    # Must be in a corner region
    horizontal = "left" if cx < 0.5 else "right"
    vertical = "top" if cy < 0.5 else "bottom"
    return f"{vertical}-{horizontal}"


def _validate_facecam_size(box: List[int], frame_w: int, frame_h: int) -> bool:
    """Validate facecam box is reasonable size for a webcam window."""
    x, y, w, h = box
    w_frac = w / frame_w
    h_frac = h / frame_h
    
    if not (FACECAM_MIN_W <= w_frac <= FACECAM_MAX_W):
        return False
    if not (FACECAM_MIN_H <= h_frac <= FACECAM_MAX_H):
        return False
    return True


def _rects_match(r1: Dict, r2: Dict, threshold: float = 0.15) -> bool:
    """Check if two rectangles are similar (normalized coords)."""
    for k in ["x", "y", "w", "h"]:
        if abs(r1.get(k, 0) - r2.get(k, 0)) > threshold:
            return False
    return r1.get("corner") == r2.get("corner")


# Re-export for compatibility
ENABLED = True