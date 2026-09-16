"""Pre-flight validation for the filtergraphs we hand to ffmpeg.

A zero dimension is a native crash. On 15-sep-2026 every clip in prod died with
ffmpeg exit code -1073741810 (STATUS_INVALID_USER_STACK, a Windows access
violation) and no Python traceback, because the failing graph was:

    [0:v]crop@c=w=0:h=0:x=0:y=0,scale=1080:1920,setsar=1[v]

The arithmetic upstream was at fault (a normalized facecam rect used as if it
were pixels), but nothing between that arithmetic and the subprocess read the
graph back, so it surfaced as a bare "ffmpeg failed" with no clue in it. This
module is that missing read: the graph is checked while it is still a string,
each problem names its filter and option, and the caller can retry a layout
that cannot produce the bad value instead of losing the clip.

``problems()`` never raises. ``validate()`` raises ``FiltergraphError`` with all
of them at once, so one pass reports everything instead of the first.

Checked: zero / negative / non-finite dimensions (negatives are legal only in
``scale``, where -1/-2 mean "keep the aspect ratio"), unbalanced or empty ``[]``
labels, empty option values, and — when asked — a graph that never produces a
labelled output to ``-map``. Crop offsets are deliberately left alone: ffmpeg
clamps a negative x/y rather than failing, so flagging them would reject graphs
that render fine. Not checked either: whether a filter exists, option spelling,
or anything else only ffmpeg can know. Expressions over frame properties
(``iw*0.42``) are left to ffmpeg; only pure literal arithmetic is evaluated.
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

# Options whose value is a pixel dimension. Zero and negative are invalid in
# every filter we build.
DIMENSION_KEYS = ("w", "h", "width", "height")

# Filters whose option-less syntax is positional, in this order. Both spellings
# are in use in this codebase: `crop=w=0:h=0` (named) and `scale=1080:1920`.
POSITIONAL = {
    "crop": ("w", "h", "x", "y"),
    "pad": ("w", "h", "x", "y"),
    "scale": ("w", "h"),
}

# Negative sizes mean "keep the aspect ratio" in `scale` only (``-1``; ``-2``
# also rounds to an even number). Everywhere else a negative is a bug in our
# arithmetic rather than an option: `crop=w=-4:h=16` fails with "Invalid too
# big or non positive size" on ffmpeg 9.0.1.
KEEP_ASPECT_FILTERS = ("scale",)

# Crop offsets (x/y) are deliberately NOT checked. Measured on this box:
# `crop=w=16:h=16:x=-5:y=0` exits 0 — ffmpeg clamps the window back into the
# frame — so a negative offset is not the crash class this module exists for,
# and rejecting it would fail graphs ffmpeg renders happily. Zero and negative
# *sizes* are another matter; see `_option_problems`.

# Only ever used as the second argument of -map/-vf style graphs; the name is
# free-form ("v", "out", "wm") so nothing is matched on it.
LABEL_RE = re.compile(r"\[([^\]]*)\]")

_LITERAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_ARITH_RE = re.compile(r"^[0-9eE+\-*/().\s]+$")
_NON_FINITE_RE = re.compile(r"(?<![A-Za-z0-9_])(nan|inf(?:inity)?)(?![A-Za-z0-9_])", re.I)

# A dimension written as a bound: `min(iw\,1080)`, `max(720\,ih)`. The inner
# group runs to the last ')' — nothing in this codebase nests a function
# inside a bound, and a greedy match keeps a trailing ')' out of the args.
_BOUND_RE = re.compile(r"^(min|max)\s*\((.*)\)$", re.S)

_QUOTES = ("'", '"')


class FiltergraphError(ValueError):
    """A graph ffmpeg would refuse, crash on, or silently render nothing for."""


def _split_outside_quotes(text: str, separators: str) -> List[str]:
    """Split on ``separators``, ignoring any that sit inside '...' or "...".

    ffmpeg's escaping rules are looser than this, but every graph in this
    codebase quotes its paths with single quotes, and the separators we split
    on (``;``, ``,``) appear inside Windows paths — the sendcmd file is written
    to a tempfile and passed as ``sendcmd=f='...'``. Splitting blindly is how a
    path ends up parsed as filter options.
    """
    parts: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch == "\\" and i + 1 < len(text) and text[i + 1] in separators:
            # ffmpeg escapes a comma inside a filter argument as `\,`, and
            # camera_inset builds exactly that: `crop=w=min(iw\,1080)`. A split
            # there cuts one filter into two nonsense tokens, whose second half
            # reads as a filter name — so the escape is honoured, not just the
            # quotes.
            buf.append(ch)
            buf.append(text[i + 1])
            i += 1
        elif ch in _QUOTES:
            quote = ch
            buf.append(ch)
        elif ch in separators:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _filter_chain(graph: str) -> List[List[str]]:
    """``[[filter, ...], ...]`` — one list of filter tokens per ``;`` chain."""
    return [[tok.strip() for tok in _split_outside_quotes(chain, ",") if tok.strip()]
            for chain in _split_outside_quotes(graph, ";")]


def _parse_filter(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """``('crop@c', [('w', '0'), ('h', '0'), ...])`` for one filter token.

    Handles the leading ``[0:v]`` input label, a trailing ``[v]`` output label,
    named options (``w=0``) and positional ones (``scale=1080:1920``), and
    leaves everything inside quotes alone — the sendcmd path is passed as
    ``sendcmd=f='C\\...\\cmd_000.txt'`` and contains both ``:`` and ``\\``.
    """
    while text.startswith("["):
        end = text.find("]")
        if end == -1:
            break
        text = text[end + 1:].lstrip()
    if text.endswith("]"):
        # A trailing output label such as `setsar=1[v]`. No filter token in
        # this codebase ends with `]` for any other reason — a quoted path
        # ends with a quote, an option value with a digit or a letter.
        text = text[:text.rfind("[")].rstrip()

    name, sep, rest = text.partition("=")
    name = name.strip()
    if not sep:
        return name, []

    positional = POSITIONAL.get(name)
    options: List[Tuple[str, str]] = []
    for index, token in enumerate(_split_outside_quotes(rest, ":")):
        token = token.strip()
        if not token:
            continue
        key, eq, value = token.partition("=")
        if eq:
            options.append((key.strip(), value.strip()))
        elif positional and index < len(positional):
            options.append((positional[index], token))
        else:
            options.append((f"arg{index}", token))
    return name, options


def _literal_value(text: str) -> Optional[float]:
    """The float a plain numeric literal denotes, else None."""
    if _LITERAL_RE.match(text):
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _arith_value(text: str) -> Optional[float]:
    """The value of a pure-arithmetic expression over literals, else None.

    ``iw*0.42`` and friends reference frame properties and are left to ffmpeg;
    ``(1080-1920)/2*-1`` is ours to evaluate, and is exactly the shape of
    arithmetic that produced the zero dimension in the first place.
    """
    if not text or not _ARITH_RE.match(text) or not any(ch.isdigit() for ch in text):
        return None
    try:
        return float(eval(text, {"__builtins__": {}}, {}))  # noqa: S307 - literals only
    except Exception:
        return None


def _output_labels(graph: str) -> List[str]:
    """Labels a later filter — or our own ``-map`` — can consume.

    The input side of a graph's first chain (``[0:v]``) is not an output, and
    neither is an empty pair of brackets.
    """
    labels = [m.group(1) for m in LABEL_RE.finditer(graph)]
    if graph.lstrip().startswith("[") and labels:
        labels = labels[1:]
    return [label for label in labels if label.strip()]


def _split_args(inner: str) -> List[str]:
    """Arguments of a nested expression, split on ffmpeg's escaped comma."""
    return [arg for arg in re.split(r"\\?,", inner) if arg.strip()]


def _zero_bound_reason(text: str) -> Optional[str]:
    """Why a dimension expression can only come out zero, or None.

    ``crop=w=min(iw\\,1080)`` is a real idiom in camera_inset: the literal
    bounds a frame property and ffmpeg resolves it at runtime, so the guard
    must not touch it. The same shape with a literal 0 is a zero-sized crop
    whatever the frame is — ``min(anything, 0)`` is 0 — and it fails exactly
    like the literal form, with "Invalid too big or non positive size".

    Only the provable cases are reported: a bound that merely *might* be zero
    (``max(iw\\,0)``, which is ``iw`` on any real frame) is left to ffmpeg.
    """
    match = _BOUND_RE.match(text.strip())
    if not match:
        return None
    func, inner = match.group(1), match.group(2)
    numbers = [_literal_value(arg.strip()) for arg in _split_args(inner)]
    if func == "min" and any(n is not None and n <= 0 for n in numbers):
        return "min() with a bound of 0 or less"
    if func == "max" and numbers and all(n is not None and n <= 0 for n in numbers):
        return "max() of bounds that are all 0 or less"
    return None


def _option_problems(name: str, key: str, raw: str) -> List[str]:
    """Problems with one filter option, judged by measurement, not assumption.

    Every rule here was checked against the ffmpeg on this box (9.0.1,
    16-sep-2026), because an over-eager guard would reject graphs ffmpeg
    renders without complaint:

      crop=w=0:h=0:x=0:y=0    exit -22  "Invalid too big or non positive size"
      crop=w=-4:h=16:x=0:y=0  exit -22  the same error class
      crop=w=16:h=16:x=-5     exit 0    window clamped into the frame
      scale=1080:-1           exit 0    "keep the aspect ratio"
      scale=-2:-1             exit 0    the same, rounded to an even number
      scale=0:-1              exit 0    and it encodes — but nothing is
                                        resized, so the intent is lost
    """
    text = raw.strip()
    if not text:
        return [f"{name}: empty value for '{key}'"]
    if key not in DIMENSION_KEYS:
        # Offsets (x/y), saturation, frame counts and paths are not this
        # module's business.
        return []

    value = _literal_value(text)
    if value is None:
        value = _arith_value(text)
    if value is None:
        bound = _zero_bound_reason(text)
        if bound:
            return [f"{name}: {key}={text} is provably zero — {bound}"]
        # Any other expression over frame properties (`min(iw\,1080)`,
        # `iw*0.42`) is ffmpeg's to evaluate; the arithmetic that produced the
        # original zero was caught above.
        return []
    if math.isnan(value) or math.isinf(value):
        return [f"{name}: {key}={text} is not a finite number"]
    if value == 0:
        return [f"{name}: {key}={text} is zero — crop refuses a zero-sized "
                f"frame and scale silently skips the resize"]
    if value < 0 and name not in KEEP_ASPECT_FILTERS:
        return [f"{name}: {key}={text} is negative — only scale accepts a "
                f"negative size (-1/-2 keep the aspect ratio)"]
    return []


def problems(graph: str, require_output_label: bool = False) -> List[str]:
    """Every reason ffmpeg would refuse, crash on, or mis-render ``graph``.

    Returns an empty list for a graph that looks sound. Never raises and never
    touches the filesystem, so it is safe to call on every clip.
    """
    if graph is None or not str(graph).strip():
        return ["filtergraph is empty"]
    text = str(graph)
    found: List[str] = []

    opens, closes = text.count("["), text.count("]")
    if opens != closes:
        found.append(f"unbalanced labels: {opens} '[' against {closes} ']'")
    if "[]" in text:
        found.append("empty label '[]' — nothing can map it")
    if _NON_FINITE_RE.search(text):
        found.append("non-finite value (nan/inf) in the graph")

    for chain in _filter_chain(text):
        for token in chain:
            name, options = _parse_filter(token)
            for key, raw in options:
                found.extend(_option_problems(name, key, raw))

    if require_output_label and not _output_labels(text):
        found.append("graph produces no labelled output to -map")
    return found


def validate(graph: str, require_output_label: bool = False) -> str:
    """``graph`` when it passes, else ``FiltergraphError`` with every reason.

    The all-at-once message matters for a crash that arrives as a return code:
    one render already cost the clip, so the log should name the whole
    problem rather than the first symptom.
    """
    found = problems(graph, require_output_label)
    if found:
        raise FiltergraphError("; ".join(found))
    return graph


def safe_graph(graph: str, fallback: Optional[str] = None,
               require_output_label: bool = False) -> Optional[str]:
    """``graph`` when it passes, else ``fallback``.

    For callers that would rather render a different layout than lose the
    clip — the router in ``reframe_v2`` uses this to drop a scene to the
    blurred-background layout instead of handing ffmpeg a graph that kills
    the whole render.
    """
    if not problems(graph, require_output_label):
        return graph
    return fallback
