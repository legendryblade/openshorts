"""AI critique gate: judge a finished short before it is announced.

Renders fail in ways the render code cannot see — a facecam appearing twice,
gameplay squeezed into a letterbox sliver, captions covering the speaker's
face, frames that are mostly black or mostly blur. The layout picker and the
filtergraph guard check DECISIONS; this checks the RESULT.

One Gemini call per clip: six frames sampled evenly across the finished file
(captions already burned in), scored against a fixed checklist. The verdict
schema is categorical — measured on the layout work, "pick one of N" is stable
while numeric scores drift run to run.

Failures only count for OBJECTIVE, visible defects. "I would have zoomed
differently" is taste, and taste must not trigger a re-render.

Disabled by default (``CRITIQUE_GATE=1``). On fail the caller re-renders the
clip once with a simpler known-good layout and critiques again; the second
verdict is accepted whatever it says, so the gate can never loop or lose a
clip entirely.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

_MODE = os.environ.get("CRITIQUE_GATE", "0").strip().lower()
ENABLED = _MODE == "1"

FRAMES = int(os.environ.get("CRITIQUE_FRAMES", "6"))
FRAME_WIDTH = int(os.environ.get("CRITIQUE_FRAME_WIDTH", "1024"))

# The one model every vision task in this repo converges on.
MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"


class CritiqueResponse:
    """Structural duck-type of the Gemini response fields the gate consumes."""

    def __init__(self, verdict, facecam_count, issues, notes):
        self.verdict = str(verdict or "").strip().lower()
        self.facecam_count = int(facecam_count if facecam_count is not None else -1)
        self.issues = [str(i) for i in (issues or [])][:6]
        self.notes = str(notes or "")[:160]

    @property
    def passed(self):
        return self.verdict == "pass"


CRITIQUE_PROMPT = """These {n} frames are sampled evenly across ONE finished
vertical short (1080x1920) cut from a long-form video. Judge it as a quality
gate before publishing. It is FAIL only for the objective defects below —
anything else (framing taste, pacing, which moment was chosen) is PASS.

LAYOUT CONTEXT: {layout_context}
The checks below that reference a facecam or gameplay apply ONLY when the
layout context says those elements exist. If the video has no facecam or no
gameplay (talking heads, interviews, b-roll, podcasts), skip those checks
entirely — a clip with no facecam panel is CORRECT for those layouts, not a
defect.

UNIVERSAL CHECKS — fail the clip if ANY of these is visible in the frames:
1. Speaker invisible: if someone is speaking (or the video's point depends on
   a person), that person must be visible in most frames — not cropped out,
   not a blurred silhouette, not off-frame.
2. Content unreadable: the main content (gameplay, screen, speaker, action)
   is so small, stretched/squashed (faces visibly distorted), or buried under
   blur that a viewer cannot tell what is happening.
3. Captions block the view: burned-in text sits ON the speaker's face, or
   covers the central action the viewer needs to see.
4. Dead frame: a majority of the sampled frames are almost entirely black,
   or almost entirely blurred filler with nothing real visible.

GAMING-SPLIT CHECKS (only when layout context says gaming_split):
5. facecam_count != 1: the webcam feed must appear EXACTLY ONCE (a small
   panel, usually at the top). Two copies = FAIL, zero when the layout
   promises one = FAIL.
6. Gameplay unreadable: the game frame is too small, distorted, or buried.

Answer for EACH defect honestly against the frames. Do not invent problems:
a clip whose content is visible, its speaker on screen (if applicable), and
captions non-blocking is a PASS even if imperfect.

Return JSON exactly:
{{"verdict": "pass"|"fail",
  "facecam_count": <integer, how many webcam-feed copies you see (0 if none expected)>,
  "issues": ["<defect>: <which frames>", ...],
  "notes": "<max 20 words>"}}
"""


def critique_clip(clip_path, samples=None, layout="auto"):
    """Gemini's verdict on a finished short, or None on any failure.

    layout: the strategy the clip was rendered with ("GAMING_SPLIT",
    "SCREENCAST", "TRACK", "GENERAL", "SPLIT", etc). Adjusts which checks
    the prompt applies — facecam checks only fire for GAMING_SPLIT.

    None means "gate unavailable" — callers must treat that as PASS so a
    quota outage or SDK error can never block or lose a rendered clip.
    """
    if not ENABLED:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    layout_context = {
        "GAMING_SPLIT": "gaming video with a persistent facecam panel "
                        "(webcam feed should appear exactly once) and a "
                        "gameplay panel",
        "SCREENCAST": "screen recording — the screen content is the point, "
                      "no facecam expected",
        "SPLIT": "two speakers stacked vertically",
        "ALTERNATE": "speaker cuts between two people",
        "TRACK": "single speaker, centre-cropped from landscape",
        "GENERAL": "general footage — blurred background with content "
                   "centred",
        "WIDE": "full-width content (screen/slide) with blurred filler",
        "INSET": "screen on top, webcam below",
    }.get(str(layout or "auto").strip().upper(),
          "auto — judge by what you see; apply facecam/gameplay checks only "
          "if those elements are present")

    try:
        from layout_picker import sample_frames
        from google import genai
        from google.genai import types as genai_types

        frames = sample_frames(clip_path, n=samples or FRAMES,
                               width=FRAME_WIDTH)
        if not frames:
            return None

        client = genai.Client(api_key=api_key)
        parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg")
                 for b in frames]
        response = client.models.generate_content(
            model=MODEL,
            contents=parts + [CRITIQUE_PROMPT.format(
                n=len(frames), layout_context=layout_context)],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "verdict": {"type": "STRING",
                                    "enum": ["pass", "fail"]},
                        "facecam_count": {"type": "INTEGER"},
                        "issues": {"type": "ARRAY",
                                   "items": {"type": "STRING"}},
                        "notes": {"type": "STRING"},
                    },
                    "required": ["verdict", "facecam_count", "issues", "notes"],
                },
            ))
        answer = json.loads(response.text) or {}
        return CritiqueResponse(answer.get("verdict"),
                                answer.get("facecam_count"),
                                answer.get("issues"),
                                answer.get("notes"))
    except Exception as e:
        print(f"   ⚠️ Critique gate unavailable ({e}) — clip passes")
        return None


def gate(clip_path, redo_fn=None, samples=None, layout="auto"):
    """Critique a finished clip; re-render once through redo_fn on fail.

    layout: the strategy the clip was rendered with — adjusts which checks
    apply (facecam checks only for GAMING_SPLIT).
    redo_fn() must rebuild the clip at the same path (a simpler layout) and
    may be None, in which case a fail is only reported. The second verdict is
    final either way — the gate can never loop. Returns the CritiqueResponse
    that decided the outcome (or None when the gate is disabled/unavailable).
    """
    verdict = critique_clip(clip_path, samples=samples, layout=layout)
    if verdict is None or verdict.passed:
        if verdict is not None:
            print(f"   🧐 Critique: PASS ({verdict.notes or 'no defects'})")
        return verdict

    print(f"   🧐 Critique: FAIL — {', '.join(verdict.issues) or verdict.notes}")
    if redo_fn is None:
        return verdict

    print("   🔁 Re-rendering with the fallback layout…")
    try:
        redo_fn()
    except Exception as e:
        print(f"   ⚠️ Fallback re-render failed ({e}) — keeping the first render")
        return verdict

    second = critique_clip(clip_path, samples=samples)
    if second is None:
        print("   🧐 Critique unavailable after re-render — keeping the fallback clip")
        return verdict
    if second.passed:
        print(f"   🧐 Critique after re-render: PASS ({second.notes or 'fixed'})")
    else:
        print(f"   🧐 Critique after re-render: still FAIL "
              f"({', '.join(second.issues) or second.notes}) — accepted as-is")
    return second
