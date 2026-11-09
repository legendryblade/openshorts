"""
Pure helpers for the Gemini clip-selection pipeline.

Standard-library only so both main.py and gemini_worker.py can import it and
the logic stays unit-testable without the heavy video dependencies.
"""

# USD per 1M tokens (input, output incl. thinking), from ai.google.dev pricing.
MODEL_PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),  # deprecated (shut down 2026-06-01)
}


def lookup_model_prices(model_name):
    """Longest-prefix match against MODEL_PRICES; None if unknown."""
    name = str(model_name or "").lower()
    best_key = None
    for key in MODEL_PRICES:
        if name.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    return MODEL_PRICES[best_key] if best_key else None


def clip_count_targets(n_windows):
    """How many clips to ask the detail pass for, given the shortlist size.

    Measured on prod 3-ago-2026: 408 of 429 jobs (95%) delivered 3 clips or
    fewer, the mode being ONE, while the prompt was free to return one per
    shortlisted window. Users who received 1-3 clips came back a second day
    0.4% of the time; those who received 4-9 came back 16.1% — so the clip
    count, not the clip quality, is what the retention curve hangs on.

    The old prompt biased hard the other way ("prefer one great clip per
    candidate window") and handed the model two unbounded licences to drop
    clips (the 2-second rule and STANDS ALONE both end in "or skip it"), with
    no floor to stop it collapsing to a single clip. This puts a floor and a
    realistic ceiling on it instead.

    ``CLIP_TARGET_MIN`` / ``CLIP_TARGET_MAX`` override both for A/B runs
    without a deploy (the reframe-testing harness drives them).
    """
    import os

    n = max(1, int(n_windows or 1))
    # Floor grows with the material: 3 windows -> 3, 5 -> 4, 10+ -> 6.
    low = max(2, min(6, n // 2 + 2))
    # Ceiling allows a rich window to yield more than one without inviting padding.
    high = min(12, max(4, n * 2))
    low = min(low, high)

    def _override(name, current):
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = _override("CLIP_TARGET_MIN", low)
    high = _override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


def trim_to_best(shorts, max_clips):
    """Cut an over-long detail-pass result down to ``max_clips`` BY SCORE.

    The detail pass hands its clips back in transcript order, batch after
    batch, so slicing the list keeps the EARLIEST clips rather than the best
    ones. On a 9-minute walkthrough that quietly threw away everything past
    minute three: the model proposed clips across the whole video, and the
    ones covering the demo, the MCP walkthrough and the close were the tail
    that got dropped. Worse, the failure scales the wrong way — the more
    generous the model is, the more of the video disappears.

    That sabotages the windowing: get_viral_clips builds scoring windows
    precisely because "a single call over the whole transcript clusters picks
    near the start", and a positional slice puts the clustering right back.

    Ranking is by ``predicted_score`` (the detail prompt already asks for it,
    and nothing else was reading it here). Ties keep transcript order, and the
    survivors come back in transcript order too, so clip numbering still runs
    front to back the way every caller downstream expects.
    """
    max_clips = max(1, int(max_clips or 1))
    if len(shorts) <= max_clips:
        return list(shorts)

    def score(item):
        try:
            return float(item[1].get("predicted_score") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    indexed = list(enumerate(shorts))
    best = sorted(indexed, key=score, reverse=True)[:max_clips]
    return [item for _, item in sorted(best, key=lambda pair: pair[0])]


def clip_duration_bounds():
    """The clip length band (seconds) the selection prompts and word-snapping
    enforce. ``CLIP_MIN_SECONDS`` / ``CLIP_MAX_SECONDS`` override the classic
    15-60 — set per job by /api/process when the user asks for a specific
    length, or by hand for A/B runs. Values are clamped to platform-sane
    limits and re-ordered so bad input degrades instead of breaking the job.
    """
    import os

    def _read(name, default):
        try:
            return float(os.environ.get(name, ""))
        except ValueError:
            return default

    lo = _read("CLIP_MIN_SECONDS", 15.0)
    hi = _read("CLIP_MAX_SECONDS", 60.0)
    lo = min(max(lo, 5.0), 175.0)
    hi = min(max(hi, 10.0), 180.0)
    if hi < lo + 5.0:  # keep a real band: degenerate ranges starve the model
        hi = min(180.0, lo + 5.0)
    return round(lo, 3), round(hi, 3)


def compact_words(words, precision=2):
    """Round word timestamps for prompts — full float precision wastes tokens."""
    return [
        {
            "w": w.get("w", ""),
            "s": round(float(w.get("s", 0)), precision),
            "e": round(float(w.get("e", 0)), precision),
        }
        for w in words
    ]


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """
    Build scoring windows aligned to Whisper segment boundaries, so a sentence
    (and usually a viral moment) is never cut in half mid-window. Windows grow
    segment by segment to roughly window_seconds (up to 1.25x for the closing
    segment) and the next window starts ~overlap_seconds before the previous
    end, also snapped to a segment start.
    """
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((float(segment.get("start", 0)), float(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    i = 0
    n = len(segments)
    while i < n:
        w_start = segments[i][0]
        j = i
        # Extend while the NEXT segment still fits within a tolerant cap, so the
        # window closes on a segment boundary near window_seconds.
        while j + 1 < n and segments[j + 1][1] - w_start <= window_seconds * 1.25:
            j += 1
            if segments[j][1] - w_start >= window_seconds:
                break
        w_end = segments[j][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(w_start, 3),
            "end": round(w_end, 3),
            "text": " ".join(seg[2] for seg in segments[i:j + 1]),
        })
        window_index += 1

        if j >= n - 1:
            break
        # Next window starts at the first segment beginning after (end - overlap),
        # but always makes progress.
        target = w_end - overlap_seconds
        k = i + 1
        while k <= j and segments[k][0] < target:
            k += 1
        i = max(k, i + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(float(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


def snap_clip_to_words(start, end, words, video_duration,
                       min_duration=15.0, max_duration=60.0,
                       search_window=1.5, max_lead=0.35, max_tail=0.45):
    """
    Snap Gemini-proposed clip boundaries onto real word boundaries plus a bit
    of the surrounding silence. LLMs are bad at millisecond arithmetic; the
    word-level timestamps are ground truth, so cuts land in pauses instead of
    mid-word.

    words: [{'w','s','e'}, ...] for the whole video, sorted by start.
    Returns (start, end); falls back to the input if no words are nearby or
    snapping cannot satisfy the duration bounds.
    """
    original = (round(float(start), 3), round(float(end), 3))
    if not words:
        return original

    starts = [float(w.get("s", 0)) for w in words]
    ends = [float(w.get("e", 0)) for w in words]

    # START: snap to the nearest word start, then lead into the silence before it.
    new_start = float(start)
    candidates = [s for s in starts if abs(s - new_start) <= search_window]
    if candidates:
        word_start = min(candidates, key=lambda s: abs(s - new_start))
        prev_ends = [e for e in ends if e <= word_start]
        if prev_ends:
            gap = max(0.0, word_start - max(prev_ends))
            lead = min(max_lead, gap / 2)
        else:
            lead = max_lead
        new_start = max(0.0, word_start - lead)

    # END: snap to the nearest word end, then trail into the silence after it.
    new_end = float(end)
    candidates = [e for e in ends if abs(e - new_end) <= search_window]
    if candidates:
        word_end = min(candidates, key=lambda e: abs(e - new_end))
        next_starts = [s for s in starts if s >= word_end]
        if next_starts:
            gap = max(0.0, min(next_starts) - word_end)
            tail = min(max_tail, gap / 2)
        else:
            tail = max_tail
        new_end = min(float(video_duration), word_end + tail)

    # Repair duration bounds while staying on word boundaries.
    if new_end - new_start < min_duration:
        target = new_start + min_duration
        later = sorted(e for e in ends if e >= target)
        if later and later[0] - new_start <= max_duration:
            new_end = min(float(video_duration), later[0] + 0.2)
        else:
            return original
    if new_end - new_start > max_duration:
        target = new_start + max_duration
        earlier = [e for e in ends if new_start < e <= target]
        new_end = (max(earlier) + 0.2) if earlier else target
        new_end = min(new_end, new_start + max_duration, float(video_duration))

    if new_end <= new_start or new_end - new_start < min_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))


def refine_clip_boundaries(start, end, words, video_duration,
                           min_duration=15.0, max_duration=60.0,
                           lead=0.4, tail=0.5, max_side_trim=0.4):
    """Trim dead air off a snapped clip without losing setup or payoff.

    snap_clip_to_words lands a cut in the pause AROUND the model's proposed
    boundary; it never pulls the boundary INWARD. The detail pass routinely
    proposes windows with 1.5-3s of silence before the first spoken word and
    after the reaction — dead weight at the exact edges a swipe-happy viewer
    judges. This keeps only a breath around the actual speech:

      - new_start = first word inside the window minus `lead` (visual beat
        before the hook), never earlier than the original start
      - new_end   = last word inside the window plus `tail` (the reaction
        beat), never later than the original end
    Guards: never trim more than `max_side_trim` of the clip from either
    side (word misdetection must not gut the clip), and never below
    min_duration — any violated guard returns the input unchanged.

    Returns (start, end) rounded to 3 decimals.
    """
    original = (round(float(start), 3), round(float(end), 3))
    s, e = original
    span = e - s
    if span < min_duration or not words:
        return original

    inside = [w for w in words
              if s - 0.5 <= float(w.get("s", 0)) and float(w.get("e", 0)) <= e + 0.5]
    if not inside:
        return original

    first = min(float(w.get("s", 0)) for w in inside)
    last = max(float(w.get("e", 0)) for w in inside)

    # Each side may move inward by at most max_side_trim of the span:
    # a word misdetection must not gut the clip from one edge.
    lo_floor = s + span * max_side_trim      # start cannot move later than this
    hi_ceiling = e - span * max_side_trim    # end cannot move earlier than this

    new_start = min(max(s, first - lead), lo_floor)
    new_end = max(min(e, last + tail), hi_ceiling)

    if new_end - new_start < min_duration:
        return original
    if new_end <= new_start or new_start < 0 or new_end > video_duration:
        return original
    return (round(new_start, 3), round(new_end, 3))


def validate_punch_ins(punch_ins, clip_start, clip_end, max_n=3,
                       min_gap=2.0, min_lead=0.6, min_tail=0.6):
    """Re-base the model's punch-in times to the trimmed clip and keep only
    the ones a renderer can actually use.

    The model times are ABSOLUTE source seconds; the renderer works on the
    clip-relative timeline after trimming. Rules enforced here, silently:
      - t inside (clip_start+min_lead, clip_end-min_tail), re-based to 0
      - layer in {"cam","gameplay"} (anything else -> "cam")
      - scale clamped 1.15-1.6; duration clamped 0.6-1.6
      - sorted by t, later entries closer than min_gap to the previous kept
        one are dropped, hard cap max_n
    Returns a list of dicts (possibly empty); never raises.
    """
    out = []
    lo, hi = clip_start + min_lead, clip_end - min_tail
    for p in (punch_ins or [])[:8]:
        try:
            t = float(p.get("t", -1))
        except (TypeError, ValueError):
            continue
        if not lo <= t <= hi:
            continue
        try:
            scale = min(1.6, max(1.15, float(p.get("scale", 1.3))))
            dur = min(1.6, max(0.6, float(p.get("duration", 1.0))))
        except (TypeError, ValueError):
            scale, dur = 1.3, 1.0
        layer = str(p.get("layer", "cam")).strip().lower()
        if layer not in ("cam", "gameplay"):
            layer = "cam"
        out.append({"t": round(t - clip_start, 3), "layer": layer,
                    "scale": scale, "duration": round(dur, 2)})
    out.sort(key=lambda p: p["t"])
    kept = []
    for p in out:
        if kept and p["t"] - kept[-1]["t"] < min_gap:
            continue
        kept.append(p)
        if len(kept) >= max_n:
            break
    return kept


# --- Rate-limit helpers (TPM pacing + 429 handling) --------------------------
# Groq's free tier allows 8,000 tokens/minute on gpt-oss-120b while a single
# scoring batch costs ~900 (measured 15-sep-2026: a 1,821-character scoring
# prompt billed 697 prompt + 165 completion tokens). Pass 1 fires its batches
# back to back, so a 39-minute video's ~34 windows all land inside ONE
# 60-second window, every call past the budget answers 429, and the run died
# because the retry policy gave up after 5s + 10s — far short of a 60s window.
# Pacing to the budget instead of hammering it is what lets an unattended run
# finish. These live here, not in main.py, for the same reason as the rest of
# this module: stdlib-only and unit-testable without the video dependencies.


_DAILY_QUOTA_TOKENS = (
    "per_day", "perday", "per day", "daily",
    "requests_per_model_per_day", "free_tier_requests",
)


def parse_go_duration(value):
    """Seconds from a Go duration string ("22.357s", "1m26.4s"), or None.

    Groq reports its reset windows as Go durations, which ``float()`` cannot
    read; the token window is the number that tells a caller how long a 429
    will actually last.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:  # some proxies send bare seconds
        return float(text)
    except ValueError:
        pass

    import re

    units = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3,
             "s": 1.0, "m": 60.0, "h": 3600.0}
    # Longest unit first so "ms" is not read as "m" and "us" not as "s".
    matches = re.findall(r"([0-9]*\.?[0-9]+)(ns|us|µs|ms|s|m|h)", text)
    if not matches:
        return None
    return round(sum(float(number) * units[unit] for number, unit in matches), 3)


def parse_retry_after(value, now=None):
    """Seconds to wait from a Retry-After header (numeric seconds or HTTP-date)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    import email.utils
    import time

    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        from datetime import timezone
        parsed = parsed.replace(tzinfo=timezone.utc)
    reference = time.time() if now is None else float(now)
    return max(0.0, round(parsed.timestamp() - reference, 3))


def estimate_prompt_tokens(prompt, chars_per_token=2.6):
    """Prompt tokens to reserve before a call is sent.

    A measured scoring call billed 697 prompt tokens for 1,821 characters —
    2.6 chars/token, not the usual 4: the prompt is dense JSON plus timestamped
    prose. Under-reserving is what trips a 429, so the default stays
    pessimistic. The caller adds its own expected-output allowance on top.
    """
    return max(1, int(len(str(prompt or "")) / max(0.5, float(chars_per_token))))


def trim_window_text(text, max_chars):
    """Keep a window's setup and payoff when its middle has to go.

    Verbose transcripts bury the moment: a 60-second window of repeated
    filler can carry 6k tokens, and the moment sits in the first and last
    exchanges. Head (the setup) + tail (the reaction) preserved, middle
    elided with a visible marker so the model knows text is missing. The
    marker counts toward ``max_chars`` — the result never exceeds the cap.
    """
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = " …[middle trimmed]… "
    body = max(1, max_chars - len(marker))
    head = max(1, int(body * 0.6))
    tail = max(1, body - head)
    if head + tail >= len(text):
        return text
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def fit_payload_to_budget(payload, budget_tokens, template_chars=0,
                          chars_per_token=2.6):
    """Shrink window texts until prompt+payload fits the token budget.

    A provider whose per-minute TPM limit is smaller than the REQUEST itself
    answers 413 — "Request too large … Limit 8000, Requested 8051" — and no
    amount of waiting, batching or key rotation helps: every key has the same
    ceiling and the request size never changes. The only lever is sending a
    smaller request, which is what this does: each window's text is scaled
    down proportionally (never below a floor) with setup/payoff kept.

    ``budget_tokens`` of 0/None disables fitting (Gemini's 1M context needs
    no such thing). ``template_chars`` is the prompt template's own fixed
    length; the JSON scaffolding around each window is charged against the
    budget too, so the estimate stays honest.
    Returns the same list, mutated — callers pass a freshly built payload.
    """
    if not payload or not budget_tokens:
        return payload
    # ~60 chars of keys/braces per window when the payload is json.dumps'd.
    per_window_overhead = 60
    budget_chars = (int(budget_tokens * chars_per_token)
                    - int(template_chars or 0)
                    - len(payload) * per_window_overhead)
    if budget_chars <= 0:
        return payload
    total = sum(len(str(w.get("text") or "")) for w in payload)
    if total <= budget_chars:
        return payload
    scale = budget_chars / float(total)
    for w in payload:
        text = str(w.get("text") or "")
        # Floor keeps a window usable; the proportional share is the target.
        cap = max(160, int(len(text) * scale))
        w["text"] = trim_window_text(text, cap)
    return payload


def is_daily_quota_error(message):
    """True when a 429 is a DAY quota rather than a per-minute one.

    Gemini's free tier allows 20 requests/day. Backing that off on a 60-second
    timer cannot help — it burns minutes per call and then fails with the same
    error — so the caller fails fast with something actionable instead.
    """
    text = str(message or "").lower()
    if not any(token in text for token in ("429", "resource_exhausted", "quota")):
        return False
    return any(token in text for token in _DAILY_QUOTA_TOKENS)


class TokenPacer:
    """Rolling-window token budget: what stops a TPM 429 happening at all.

    ``reserve`` waits until the call it is about to make fits in the budget and
    ``settle`` replaces that reservation with whatever the provider actually
    billed. Reserving BEFORE the request can be sent is what makes one pacer
    safe to share between scoring workers: two callers cannot spend the same
    tokens together.

    ``limit_tokens=0`` disables pacing, which is the right behaviour for a
    provider that advertises no limit; ``set_limit`` adopts the budget from the
    server's own ``x-ratelimit-limit-tokens`` header. ``clock`` and ``sleeper``
    are injectable so the pacing can be tested without real waiting.
    """

    def __init__(self, limit_tokens=0, window_seconds=60.0, clock=None, sleeper=None):
        import collections
        import threading
        import time

        self._limit = max(0, int(limit_tokens or 0))
        self._window = max(1.0, float(window_seconds))
        self._events = collections.deque()  # (monotonic timestamp, tokens)
        self._lock = threading.Lock()
        self._clock = clock or time.monotonic
        self._sleep = sleeper or time.sleep

    @property
    def limit_tokens(self):
        return self._limit

    @property
    def window_seconds(self):
        return self._window

    def set_limit(self, tokens):
        """Adopt a token budget learned from a response header."""
        try:
            value = int(tokens or 0)
        except (TypeError, ValueError):
            return
        if value > 0:
            with self._lock:
                self._limit = value

    def spent(self):
        """Tokens currently counted against the window."""
        with self._lock:
            return self._spent(self._clock())

    def _spent(self, now):
        cutoff = now - self._window
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()
        return sum(tokens for _, tokens in self._events)

    def wait_seconds(self, estimate):
        """Seconds a call costing ``estimate`` must wait before it is sent."""
        if self._limit <= 0:
            return 0.0
        estimate = max(1, int(estimate or 0))
        with self._lock:
            now = self._clock()
            spent = self._spent(now)
            if spent + estimate <= self._limit:
                return 0.0
            need = spent + estimate - self._limit
            freed = 0.0
            for timestamp, tokens in self._events:
                freed += tokens
                if freed >= need:
                    # Sleep past the moment that much has aged out.
                    return max(0.0, (timestamp + self._window) - now) + 0.05
            return self._window

    def reserve(self, estimate):
        """Claim budget for a call about to be sent; sleeps if it has to.

        Returns the seconds slept. A caller that cannot get its turn within a
        whole window proceeds anyway: the occasional 429 that the retry policy
        handles beats a pipeline that hangs, and a call larger than the entire
        budget would otherwise wait forever.
        """
        estimate = max(1, int(estimate or 0))
        slept = 0.0
        while slept < self._window:
            wait = self.wait_seconds(estimate)
            if wait <= 0:
                break
            wait = min(wait, self._window - slept)
            if wait <= 0:
                break
            self._sleep(wait)
            slept += wait
        with self._lock:
            self._events.append((self._clock(), estimate))
        return round(slept, 3)

    def settle(self, reserved, actual):
        """Replace a reservation with the tokens the provider actually billed.

        The difference is refunded, so a pessimistic up-front estimate does not
        silently cost throughput. Zeroed entries are left in place: they
        contribute nothing to the window and age out on their own.
        """
        reserved = max(0, int(reserved or 0))
        actual = max(0, int(actual or 0))
        with self._lock:
            if actual > reserved:
                self._events.append((self._clock(), actual - reserved))
                return
            refund = reserved - actual
            for index in range(len(self._events) - 1, -1, -1):
                if refund <= 0:
                    break
                timestamp, tokens = self._events[index]
                if tokens <= 0:
                    continue
                take = min(tokens, refund)
                self._events[index] = (timestamp, tokens - take)
                refund -= take
