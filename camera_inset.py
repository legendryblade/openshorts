"""Find the webcam inset in a screen recording, and frame the two apart.

The case: one source, the whole screen (a game, a desktop, an editor) with the
person composited into a corner. It is how OBS records and how every stream VOD
looks, and it is the case screencast_layout gets wrong. There the speaker band
is a large crop taken AROUND the face, which on a full-screen source means the
band is mostly more screen: measured on an Excel walkthrough, the output showed
the same spreadsheet twice, once whole and once enlarged.

What this needs instead is the inset's own rectangle, so the two bands can hold
genuinely different things — screen above, person below.

Finding it exactly is harder than it looks. The inset is not always a rectangle
(circles and clipped polygons are common), it has no reliable border, and on
gameplay the background moves as much as the person does, which rules out
temporal-variance tricks. What IS reliable is that the inset is anchored to a
corner and that a person detector fires inside it. So: locate the person, decide
which corner they are nearest, and grow a box from that corner until it covers
them with margin.
"""
import os

CORNER_MARGIN = 0.20   # a subject this far from an edge (as a fraction of the
                       # frame) still counts as anchored to it

# An inset subject is SMALL and OFF-CENTRE; a presenter filling the shot is
# neither. Requiring the detection to actually touch two edges was tried first
# and rejected: YOLO returns an upper-body box that stops at the chest, so a
# webcam pinned to the right edge measured 18% clear of the bottom and got
# thrown out. Measured on four real clips, these two properties separate the
# cases where "touching a corner" did not.
MAX_SUBJECT_HEIGHT = 0.35   # fraction of frame height

# Horizontal offset from centre, as a fraction of frame width. HORIZONTAL
# specifically: a composited camera is pinned to the left or right side, while a
# talking head is centred left-to-right even when their face sits high in the
# shot. Accepting offset on either axis let four talking heads through, all of
# them with a face near the top edge. Measured on those nine candidates the two
# groups do not overlap: real insets sat 0.37-0.43 from centre, the talking
# heads 0.01-0.12.
MIN_OFFSET = 0.18

# How much bigger than the detected head/upper body the inset is assumed to be.
# A webcam frames head and shoulders, and detectors return the head or the upper
# body, so the box has to grow to reach the inset's real edges.
INSET_PADDING = float(os.environ.get("INSET_PADDING", "1.45"))

# Guard rails as a fraction of frame height. Below the floor there is nothing
# worth showing; above the ceiling it stopped being an inset and the layout
# should not be used at all.
MIN_INSET_HEIGHT = 0.10
MAX_INSET_HEIGHT = 0.38

# An inset is pinned to the same pixels for the whole recording; a presenter
# walks about. Spread of the detected centres, as a fraction of frame width,
# above which this is a person in a room rather than a composited camera.
# Measured on the two false positives this rule was written for: a real inset
# moved 3-11px across samples, a presenter 316px.
MAX_CENTRE_SPREAD = 0.05


def nearest_corner(box, frame_w, frame_h):
    """Which corner the subject sits in: (horizontal, vertical) as strings.

    Returns e.g. ("left", "bottom"). A subject in the middle of the frame is
    reported by its nearest edges anyway; callers use `is_cornered` to reject.
    """
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    return ("left" if cx < frame_w / 2 else "right",
            "top" if cy < frame_h / 2 else "bottom")


def is_cornered(box, frame_w, frame_h, margin=CORNER_MARGIN):
    """True when the subject looks like a webcam inset rather than the shot.

    Small and off-centre, not "touches two edges": see MAX_SUBJECT_HEIGHT.

    This is a sanity filter, not the decision. Whether the video is a screen
    recording with a camera in it at all is answered upstream by layout_picker;
    a game character standing in a corner would pass this test, and is kept out
    by the fact that nobody asked for this layout on that video.
    """
    x, y, w, h = box
    if h > frame_h * MAX_SUBJECT_HEIGHT:
        return False

    cx = (x + w / 2.0) / frame_w
    off_centre = abs(cx - 0.5) >= MIN_OFFSET

    near_edge = (x <= frame_w * margin or (x + w) >= frame_w * (1 - margin)
                 or y <= frame_h * margin or (y + h) >= frame_h * (1 - margin))
    return off_centre and near_edge


INSET_ASPECT = 16 / 9.0


def inset_box(box, frame_w, frame_h, padding=None):
    """Estimated inset rectangle (x, y, w, h) around a detected subject.

    Grown from the corner the subject is anchored to, so the box hugs the same
    edges the inset does instead of floating around the face.

    The height comes from assuming a 16:9 inset rather than from scaling the
    detection, because the detection is often a torso: YOLO returned a 246x86
    box for a webcam whose real rectangle was about 325x180, and padding that
    shape upwards still cut the head off. Deriving the height from the width
    and pinning the result to the corner reaches the top of the inset instead.
    """
    padding = INSET_PADDING if padding is None else padding
    x, y, w, h = box
    horizontal, vertical = nearest_corner(box, frame_w, frame_h)

    new_w = min(frame_w, w * padding)
    new_h = min(frame_h, max(h * padding, new_w / INSET_ASPECT))

    # Anchor: keep the edge the subject is already near, grow the other way.
    if horizontal == "left":
        new_x = max(0, min(x - (new_w - w) / 2.0, frame_w - new_w))
        if x <= frame_w * CORNER_MARGIN:
            new_x = 0
    else:
        new_x = max(0, min(x + w + (new_w - w) / 2.0 - new_w, frame_w - new_w))
        if (x + w) >= frame_w * (1 - CORNER_MARGIN):
            new_x = frame_w - new_w

    # Vertical growth is biased upwards: detectors return the head or the chest,
    # and a portrait needs headroom, not more torso. Splitting the growth evenly
    # cropped the top of the head off on real clips.
    grow = new_h - h
    if vertical == "top":
        new_y = max(0, min(y - grow * 0.6, frame_h - new_h))
        if y <= frame_h * CORNER_MARGIN:
            new_y = 0
    else:
        # Pin to the bottom edge: the inset is there, and the detection sits
        # somewhere inside it rather than at its top.
        new_y = frame_h - new_h
        if (y + h) < frame_h * (1 - CORNER_MARGIN):
            new_y = max(0, min(y - grow * 0.6, frame_h - new_h))

    return (int(round(new_x)), int(round(new_y)),
            int(round(new_w)), int(round(new_h)))


def usable(box, frame_h, min_ratio=MIN_INSET_HEIGHT, max_ratio=MAX_INSET_HEIGHT):
    """Whether an inset of this size is worth building a layout around."""
    return min_ratio * frame_h <= box[3] <= max_ratio * frame_h


def detect(video_path, samples=10):
    """Median inset rectangle across sampled frames, or None.

    None means "no usable inset here", which callers must treat as "use another
    layout" rather than as an error.
    """
    import cv2
    import numpy as np
    import main as m
    import screencast_layout

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or not frame_w:
        cap.release()
        return None

    boxes = []
    try:
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / samples))
            ok, frame = cap.read()
            if not ok:
                continue
            # Faces first, body as fallback: a webcam inset is small, and on a
            # 1080p source the face inside it is often too few pixels for
            # BlazeFace even at full resolution, while YOLO still finds the
            # person.
            faces = screencast_layout.detect_faces_full_res(frame)
            if faces:
                box = max(faces, key=lambda c: c['score'])['box']
            else:
                box = m.detect_person_yolo(frame)
            if not box:
                continue
            if not is_cornered(box, frame_w, frame_h):
                continue
            boxes.append(inset_box(box, frame_w, frame_h))
    finally:
        cap.release()

    if len(boxes) < max(3, samples // 3):
        return None

    arr = np.array(boxes, dtype=float)

    # Stability gate. Without it, a presenter standing in front of a screen and
    # an ordinary talking head both come back as "insets" and the layout puts a
    # zoom of someone's face under a copy of the same shot.
    #
    # Measured against the plain min-to-max range first, which was far too
    # brittle: one sample landing on a face elsewhere in the frame killed three
    # genuine insets. So cluster around the median instead and require most
    # detections to agree, which tolerates the odd stray without tolerating a
    # subject that actually moves.
    centres_x = arr[:, 0] + arr[:, 2] / 2.0
    centres_y = arr[:, 1] + arr[:, 3] / 2.0
    mid_x, mid_y = np.median(centres_x), np.median(centres_y)
    tolerance = frame_w * MAX_CENTRE_SPREAD
    close = ((np.abs(centres_x - mid_x) <= tolerance)
             & (np.abs(centres_y - mid_y) <= tolerance))
    if close.sum() < max(3, 0.6 * len(boxes)):
        return None

    arr = arr[close]
    median = tuple(int(v) for v in np.median(arr, axis=0))
    if not usable(median, frame_h):
        return None
    return median


MAX_CAMERA_RATIO = 0.40


def inset_filtergraph(orig_w, orig_h, out_w, out_h, box, camera_ratio=None):
    """Screen on top at full width, the webcam inset below.

    The screen keeps its whole width, which is the point: a game HUD or a
    desktop puts what matters at the edges. The inset is scaled up so the person
    reads at a size the source never gave them.

    The camera band takes its height from the INSET'S OWN aspect ratio rather
    than a fixed share of the frame. A fixed share was tried first and stretched
    every face sideways: a 16:9 inset forced into a 2:1 band is a 12% horizontal
    stretch, and it is immediately visible on a face.
    """
    box_w = max(2, box[2])
    box_h = max(2, box[3])

    if camera_ratio is not None:
        cam_h = int(out_h * camera_ratio)
    else:
        cam_h = int(round(out_w * box_h / float(box_w)))
    cam_h = min(cam_h, int(out_h * MAX_CAMERA_RATIO))
    cam_h -= cam_h % 2

    screen_h = int(round(out_w * orig_h / float(orig_w)))
    screen_h -= screen_h % 2
    screen_h = max(2, min(screen_h, out_h - cam_h - 2))

    # Widen (or heighten) the crop to the band's aspect so the scale below is
    # uniform. Clamped to the frame, so an inset hard against an edge simply
    # keeps whatever it can reach.
    target_aspect = out_w / float(cam_h)
    x, y, w, h = box
    if w / float(h) < target_aspect:
        want_w = min(orig_w, int(round(h * target_aspect)))
        x = int(round(x + w / 2.0 - want_w / 2.0))
        w = want_w
    else:
        want_h = min(orig_h, int(round(w / target_aspect)))
        y = int(round(y + h / 2.0 - want_h / 2.0))
        h = want_h
    x = max(0, min(x, orig_w - w))
    y = max(0, min(y, orig_h - h))

    w -= w % 2
    h -= h % 2
    x -= x % 2
    y -= y % 2

    filler_h = out_h - screen_h - cam_h

    return (
        f"[0:v]split=3[bga][sa][ca];"
        # Blurred backdrop so the leftover strip is not a black bar. Scaled by
        # HEIGHT: scaling a 16:9 source to 1080 wide gives 608 tall, and there
        # is no 1920-tall crop to take out of that.
        f"[bga]scale=-2:{out_h},crop=w=min(iw\\,{out_w}):h={out_h},"
        f"scale={out_w}:{out_h},gblur=sigma=14[bg];"
        f"[sa]scale={out_w}:{screen_h}[screen];"
        f"[ca]crop=w={w}:h={h}:x={x}:y={y},scale={out_w}:{cam_h}[cam];"
        f"[bg][screen]overlay=x=0:y={filler_h // 2}[withscreen];"
        f"[withscreen][cam]overlay=x=0:y={filler_h // 2 + screen_h},"
        f"setsar=1[v]"
    )


# --- gaming_split layout: gameplay + facecam PiP composition ---
# Facecam takes ~25% of output height, positioned at top or bottom
GAMING_SPLIT_CAMERA_RATIO = 0.25
# HUD safe margin - don't crop within this margin of edges
HUD_SAFE_MARGIN = 0.08


def gaming_split_filtergraph(orig_w, orig_h, out_w, out_h, facecam_rect, facecam_dest="bottom", gameplay_crop="auto"):
    """
    Gaming split layout: full gameplay (minus facecam area) + facecam PiP overlay.
    
    Args:
        orig_w, orig_h: source dimensions
        out_w, out_h: output dimensions (e.g., 1080x1920)
        facecam_rect: (x, y, w, h) tuple OR {x, y, w, h} dict in SOURCE pixels - the facecam window
        facecam_dest: "top" or "bottom" - where to place facecam PiP panel
        gameplay_crop: "horizontal", "vertical", or "auto"
    
    Composition:
    - Gameplay layer: source with facecam area cropped out, scaled to gameplay panel
    - Facecam layer: extracted from facecam_rect, scaled proportionally, placed as PiP
    - Output: 1080x1920 with facecam panel at top or bottom
    """
    # Handle both dict and tuple formats
    if isinstance(facecam_rect, dict):
        fx, fy, fw, fh = facecam_rect['x'], facecam_rect['y'], facecam_rect['w'], facecam_rect['h']
    else:
        fx, fy, fw, fh = facecam_rect
    x, y, w, h = int(fx), int(fy), int(fw), int(fh)
    
    # Facecam destination panel - 25% of output height
    cam_panel_h = int(out_h * GAMING_SPLIT_CAMERA_RATIO)
    cam_panel_h -= cam_panel_h % 2
    
    # Facecam aspect ratio
    cam_aspect = fw / fh
    cam_w = int(cam_panel_h * cam_aspect)
    cam_w -= cam_w % 2
    cam_w = min(cam_w, out_w)
    
    # Position facecam panel
    if facecam_dest == "top":
        cam_panel_y = 0
        gameplay_panel_y = cam_panel_h
    else:  # bottom
        cam_panel_y = out_h - cam_panel_h
        gameplay_panel_y = 0
    
    cam_x = (out_w - cam_w) // 2
    gameplay_panel_h = out_h - cam_panel_h
    gameplay_panel_h -= gameplay_panel_h % 2
    
    # Gameplay source: crop out the facecam area (bottom strip if facecam is at bottom)
    # Facecam is typically in bottom-right corner for gaming
    # We crop the bottom strip containing the facecam
    facecam_bottom = y + h
    gameplay_src_h = facecam_bottom - int(h * 0.1)  # keep some margin below facecam
    if gameplay_src_h < orig_h * 0.5:
        # If facecam is very low, just use most of the frame
        gameplay_src_h = orig_h - int(orig_h * 0.1)
    
    # Gameplay scales to fill the gameplay panel (bottom 75% of output)
    gameplay_panel_h = out_h - int(out_h * GAMING_SPLIT_CAMERA_RATIO)
    gameplay_panel_h -= gameplay_panel_h % 2
    
    # Facecam PiP size (preserve aspect ratio)
    cam_scale = min(out_w / fw, (out_h * GAMING_SPLIT_CAMERA_RATIO) / fh)
    cam_scaled_w = int(fw * cam_scale)
    cam_scaled_h = int(fh * cam_scale)
    cam_scaled_w -= cam_scaled_w % 2
    cam_scaled_h -= cam_scaled_h % 2
    
    # Facecam position in output (centered horizontally)
    cam_out_x = (out_w - int(fw * min(out_w / fw, (out_h * GAMING_SPLIT_CAMERA_RATIO) / fh))) // 2
    cam_out_y = out_h - int(out_h * GAMING_SPLIT_CAMERA_RATIO)
    
    return (
        f"[0:v]split=3[bga][ga][ca];"
        # Background: blurred full frame (fallback)
        f"[bga]scale=-2:{out_h},crop=w=min(iw\\,{out_w}):h={out_h},"
        f"scale={out_w}:{out_h},gblur=sigma=14[bg];"
        # Gameplay: crop out facecam area (bottom portion), scale to gameplay panel
        f"[ga]crop=w={orig_w}:h={int(orig_h * 0.85)}:x=0:y=0,"
        f"scale={out_w}:{out_h - int(out_h * 0.25)}[gameplay];"
        # Facecam: extract from source corner, scale to PiP size
        f"[ca]crop=w={int(fw)}:h={int(fh)}:x={int(fx)}:y={int(fy)},"
        f"scale={int(fw * min(out_w / fw, (out_h * GAMING_SPLIT_CAMERA_RATIO) / fh))}:"
        f"{int(fh * min(out_w / fw, (out_h * GAMING_SPLIT_CAMERA_RATIO) / fh))}[facecam];"
        # Compose: background -> gameplay -> facecam
        f"[bg][gameplay]overlay=x=0:y={int(out_h * 0.25)}[withgame];"
        f"[withgame][facecam]overlay=x={(out_w - int(fw * min(out_w / fw, (out_h * GAMING_SPLIT_CAMERA_RATIO) / fh))) // 2}:"
        f"y={out_h - int(out_h * GAMING_SPLIT_CAMERA_RATIO)},setsar=1[v]"
    )
