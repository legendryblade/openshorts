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
# Below this kept fraction of the frame, strip-removal is not worth it —
# the gameplay keeps the full frame and the PiP is skipped instead.
GAMING_SPLIT_MIN_KEEP = float(os.environ.get("GAMING_SPLIT_MIN_KEEP", "0.35"))
# Bottom band reserved for burned captions (OpusClip pattern: cam top,
# gameplay middle, captions bottom). The gameplay panel ends above it so
# default bottom-center captions never cover the cam or the action.
GAMING_SPLIT_CAPTION_RATIO = float(os.environ.get("GAMING_SPLIT_CAPTION_RATIO", "0.18"))
# HUD safe margin - don't crop within this margin of edges
HUD_SAFE_MARGIN = 0.08


def _even_box(x, y, w, h, frame_w, frame_h, min_side=2):
    """Clamp a crop box inside the frame with even sides (encoder-safe)."""
    w = max(min_side, min(int(w) - (int(w) % 2), frame_w))
    h = max(min_side, min(int(h) - (int(h) % 2), frame_h))
    x = max(0, min(int(x), frame_w - w))
    y = max(0, min(int(y), frame_h - h))
    return x, y, w, h


def _zoom_sendcmd_lines(base_box, punches, fps, frame_w, frame_h, target):
    """sendcmd lines that drive a named crop through the punch-in curve.

    base_box: (x, y, w, h) the layer's resting crop, in source pixels.
    punches: [{'t','scale','duration'}] CLIP-RELATIVE seconds, pre-validated.
    Each punch eases to a zoomed box (base divided by scale, centred), holds,
    and eases back — 5 steps each way reads as a deliberate push, not a cut.

    Returns [] when punches is empty (no sendcmd filter is added then).
    """
    if not punches:
        return []
    lines = []
    bx, by, bw, bh = base_box
    STEPS = 5
    EASE = 0.25  # seconds of attack/release on each side of the hold
    for p in punches:
        try:
            t0 = max(0.0, float(p["t"]))
            scale = float(p.get("scale", 1.3))
            dur = float(p.get("duration", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        zw, zh = bw / scale, bh / scale
        zx = bx + (bw - zw) / 2.0
        zy = by + (bh - zh) / 2.0
        zx, zy, zw, zh = _even_box(zx, zy, zw, zh, frame_w, frame_h)

        def _emit(ts, x, y, w, h):
            x, y, w, h = _even_box(x, y, w, h, frame_w, frame_h)
            lines.append(f"{ts:.3f} {target} w {w}; {ts:.3f} {target} h {h}; "
                         f"{ts:.3f} {target} x {x}; {ts:.3f} {target} y {y};")

        # ease in (base -> zoomed), hold, ease out (zoomed -> base)
        for i in range(1, STEPS + 1):
            f = (i / STEPS) ** 2  # quadratic ease
            _emit(t0 - EASE + (EASE * i / STEPS),
                  bx + (zx - bx) * f, by + (zy - by) * f,
                  bw + (zw - bw) * f, bh + (zh - bh) * f)
        _emit(t0 + dur, zx, zy, zw, zh)
        for i in range(1, STEPS + 1):
            f = 1.0 - (i / STEPS) ** 2
            _emit(t0 + dur + (EASE * i / STEPS),
                  bx + (zx - bx) * f, by + (zy - by) * f,
                  bw + (zw - bw) * f, bh + (zh - bh) * f)
    # Deduped, chronological sendcmd expects sorted timestamps per target.
    lines.sort()
    return lines


def gaming_split_filtergraph(orig_w, orig_h, out_w, out_h, facecam_rect,
                             facecam_dest="top", gameplay_crop="auto",
                             punch_ins=None, fps=30.0, cmd_dir=None):
    """Gameplay panel + facecam PiP panel, both cropped from the same source.

    The gameplay crop EXCLUDES the facecam window — a corner cam forces the
    removal of a full strip (horizontal or vertical); the strip that retains
    more of the frame wins, so the HUD rows beside the cam survive. Both
    layers are scaled uniformly and centred in their panels: a direct
    scale-to-panel stretched a 4:3 keep into a 9:16 slot, which read as
    distortion on faces (17-sep-2026).

    If the cam window is so large that either strip-removal keeps under
    GAMING_SPLIT_MIN_KEEP of the frame, the gameplay keeps the FULL frame and
    the PiP is skipped — a small visible cam in the gameplay beats a stretched
    crop or a duplicated cam (the 16-sep GTA clips showed the cam twice: once
    in the 'cropped' gameplay, once in the PiP, because the crop ignored the
    rect).

    facecam_rect: normalized dict {x,y,w,h} (detector shape) or pixel tuple.
    facecam_dest: which output edge the PiP panel sits at.
    gameplay_crop: "horizontal" | "vertical" | "auto" (auto = larger keep wins).
    punch_ins: [{"t","scale","duration","layer"}] SEGMENT-relative seconds
        (the render seeks per scene; the caller re-bases). cam punches zoom
        the facecam crop, gameplay punches zoom the kept gameplay crop —
        both via sendcmd on a named crop, so the panel size stays fixed and
        the image inside it enlarges (reaction emphasis grows the person
        whole, never a tighter face crop).
    fps: source frame rate, for sendcmd timestamps.
    cmd_dir: where sendcmd command files are written (required when
        punch_ins is non-empty).
    """
    if isinstance(facecam_rect, dict):
        fx, fy, fw, fh = (facecam_rect['x'], facecam_rect['y'],
                          facecam_rect['w'], facecam_rect['h'])
    else:
        fx, fy, fw, fh = facecam_rect
    # Normalized input (detector output) is scaled to source pixels; a rect in
    # source pixels passes through. Above 1.0 can only be pixels, since a
    # fraction of a frame never exceeds 1.
    if 0.0 < fw <= 1.0:
        fx, fy, fw, fh = fx * orig_w, fy * orig_h, fw * orig_w, fh * orig_h

    def _even(v):
        return max(2, int(v) - (int(v) % 2))

    x = max(0, min(int(fx), orig_w - 2))
    y = max(0, min(int(fy), orig_h - 2))
    w = _even(min(max(fw, 2), orig_w - x))
    h = _even(min(max(fh, 2), orig_h - y))

    # Panels: cam band at the top, gameplay in the middle, bottom band kept
    # free for burned captions (see GAMING_SPLIT_CAPTION_RATIO).
    cam_panel_h = _even(out_h * GAMING_SPLIT_CAMERA_RATIO)
    caption_h = _even(out_h * GAMING_SPLIT_CAPTION_RATIO)
    gp_panel_y = cam_panel_h if facecam_dest == "top" else 0
    gp_panel_h = _even(out_h - cam_panel_h - caption_h)
    cam_panel_y = 0 if facecam_dest == "top" else (out_h - cam_panel_h)

    # Two strip-removal candidates for the gameplay keep-rect.
    keep_h = (0, 0, orig_w, y) if y > 0 else None                    # below-cam kept
    keep_h2 = (0, y + h, orig_w, orig_h - (y + h)) if y + h < orig_h else None
    keep_v = (0, 0, x, orig_h) if x > 0 else None                    # left of cam kept
    keep_v2 = (x + w, 0, orig_w - (x + w), orig_h) if x + w < orig_w else None

    horiz_opts = [k for k in (keep_h, keep_h2) if k and k[2] >= 4 and k[3] >= 4]
    vert_opts = [k for k in (keep_v, keep_v2) if k and k[2] >= 4 and k[3] >= 4]
    best_h = max(horiz_opts, key=lambda k: k[2] * k[3]) if horiz_opts else None
    best_v = max(vert_opts, key=lambda k: k[2] * k[3]) if vert_opts else None

    choice = (gameplay_crop or "auto").strip().lower()
    if choice == "horizontal" and best_h:
        keep = best_h
    elif choice == "vertical" and best_v:
        keep = best_v
    else:
        # Auto: PREFER removing the strip that contains the facecam.
        # The facecam is at (x,y,w,h), so it overlaps the right edge if x+w > orig_w*0.5
        # and the bottom edge if y+h > orig_h*0.5. We want to KEEP the side opposite the facecam.
        cands = [k for k in (best_h, best_v) if k]
        if not cands:
            keep = (0, 0, orig_w, orig_h)
        else:
            # Prefer the strip that DOES NOT contain the facecam center
            cx, cy = x + w/2, y + h/2
            cam_right = cx > orig_w * 0.5
            cam_bottom = cy > orig_h * 0.5
            
            # Score each candidate: prefer the one that keeps the side opposite the facecam
            def score_keep(k):
                kx, ky, kw, kh = k
                # Check if this keep-rect contains the facecam center
                contains_cam = (kx <= cx < kx + kw) and (ky <= cy < ky + kh)
                # Prefer keep-rects that DON'T contain the cam center
                score = 0 if contains_cam else 1
                # Among equal, prefer larger area
                return (score, k[2] * k[3])
            
            keep = max(cands, key=score_keep)

    # Fallback: if the best keep loses too much of the frame, keep the FULL
    # frame and drop the PiP — a small cam in the gameplay is less wrong than
    # a cropped-to-death gameplay or a duplicated cam.
    keep_frac = (keep[2] * keep[3]) / float(orig_w * orig_h)
    pip = keep_frac >= GAMING_SPLIT_MIN_KEEP

    if pip:
        gp_scale = min(out_w / float(keep[2]), gp_panel_h / float(keep[3]))
        gp_w = _even(keep[2] * gp_scale)
        gp_h = _even(keep[3] * gp_scale)
        gp_x = (out_w - gp_w) // 2
        gp_y = gp_panel_y + (gp_panel_h - gp_h) // 2

        cam_scale = min(out_w / float(w), cam_panel_h / float(h))
        cam_w = _even(w * cam_scale)
        cam_h = _even(h * cam_scale)
        cam_x = (out_w - cam_w) // 2
        cam_y = cam_panel_y + (cam_panel_h - cam_h) // 2

        # Punch-ins drive a NAMED crop through sendcmd: the panel sizes above
        # stay fixed and the image inside enlarges — zoom without re-layout.
        def _punch_branch(punches, base_box, name):
            if not punches or not cmd_dir:
                return "", _even_box(*base_box, orig_w, orig_h)
            import os as _os
            lines = _zoom_sendcmd_lines(base_box, punches, fps,
                                        orig_w, orig_h, f"crop@{name}")
            if not lines:
                return "", _even_box(*base_box, orig_w, orig_h)
            path = _os.path.join(cmd_dir, f"punch_{name}.txt")
            with open(path, "w") as f:
                f.write(chr(10).join(lines) + chr(10))
            from ffmpeg_utils import escape_filter_value as _efv
            return f"sendcmd=f='{_efv(path)}',", _even_box(*base_box, orig_w, orig_h)

        cam_punches = [p for p in (punch_ins or []) if p.get("layer") == "cam"]
        gp_punches = [p for p in (punch_ins or []) if p.get("layer") == "gameplay"]
        cam_pre, cam_base = _punch_branch(cam_punches, (x, y, w, h), "cp")
        gp_pre, gp_base = _punch_branch(gp_punches, keep, "gp")

        graph = (
            f"[0:v]split=3[bga][ga][ca];"
            f"[bga]scale=-2:{out_h},crop=w=min(iw\,{out_w}):h={out_h},"
            f"scale={out_w}:{out_h},gblur=sigma=14[bg];"
            f"[ga]{gp_pre}crop=w={gp_base[2]}:h={gp_base[3]}:x={gp_base[0]}:y={gp_base[1]},"
            f"scale={gp_w}:{gp_h}:flags=lanczos[gameplay];"
            f"[ca]{cam_pre}crop=w={cam_base[2]}:h={cam_base[3]}:x={cam_base[0]}:y={cam_base[1]},"
            f"scale={cam_w}:{cam_h}:flags=lanczos[facecam];"
            f"[bg][gameplay]overlay=x={gp_x}:y={gp_y}[withgame];"
            f"[withgame][facecam]overlay=x={cam_x}:y={cam_y},unsharp=5:5:0.8:5:5:0.4,setsar=1[v]"
        )
    else:
        gp_scale = min(out_w / float(orig_w), gp_panel_h / float(orig_h))
        gp_w = _even(orig_w * gp_scale)
        gp_h = _even(orig_h * gp_scale)
        gp_x = (out_w - gp_w) // 2
        gp_y = gp_panel_y + (gp_panel_h - gp_h) // 2
        graph = (
            f"[0:v]split=2[bga][ga];"
            f"[bga]scale=-2:{out_h},crop=w=min(iw\,{out_w}):h={out_h},"
            f"scale={out_w}:{out_h},gblur=sigma=14[bg];"
            f"[ga]scale={gp_w}:{gp_h}[gameplay];"
            f"[bg][gameplay]overlay=x={gp_x}:y={gp_y},setsar=1[v]"
        )
    return graph
