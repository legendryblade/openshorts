"""Transcription backends: Groq cloud ASR, NVIDIA Parakeet (onnx-asr),
faster-whisper fallback.

Every caller goes through transcribe_media(), which returns the transcript
contract the whole pipeline depends on:

    {
      "text": str,          # full punctuated transcript
      "language": str,      # whisper-style short code ("es", "en", ...)
      "segments": [
        {"start": float, "end": float, "text": str,
         "words": [{"word": str, "start": float, "end": float}, ...]},
      ],
    }

Invariants the consumers rely on (clip cutting, karaoke subtitles, Remotion):
  - word["word"] carries a LEADING SPACE on true word starts; continuation
    fragments are merged into their base word (merge_continuation_words).
  - all numerics are native Python floats (json.dump of the transcript).
  - words sorted by start, segments chronological, absolute file timestamps.

TRANSCRIBE_BACKEND env: "whisper" (default) | "parakeet" | "groq".
The parakeet path falls back to whisper automatically when the model errors,
produces no usable words, or the detected language is outside its 25
supported European languages (e.g. Japanese/Chinese/Arabic uploads).
The groq path (cloud ASR, GROQ_API_KEY required) falls back to local whisper
on any API error — bad key, upload too large, rate limit, network drop.
GPU whisper in turn falls back to CPU whisper on CUDA errors (VRAM is shared
with other models on the host, so loads can OOM under load).
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time

from subtitles import (
    get_whisper_config,
    WHISPER_TRANSCRIBE_PARAMS,
    merge_continuation_words,
)

PARAKEET_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

# The 25 European languages parakeet-tdt-0.6b-v3 supports (ISO 639-1).
PARAKEET_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru",
    "uk",
}

# Serializes GPU transcription across concurrent jobs so N jobs can't stack
# N model contexts / decode batches in VRAM. CPU whisper stays ungated
# (CTranslate2 models are thread-safe and that matches the old behavior).
_ASR_GATE = threading.Semaphore(int(os.environ.get("ASR_GPU_CONCURRENCY", "1")))


class _NullGate:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_GATE = _NullGate()


class _TranscribeProgress:
    """Emits '🎙️ Transcribing… NN% (Xs)' lines at 25% steps.

    These are the only transcription lines cloud users see (log_view keeps
    them), so they must stay free of technical detail.
    """

    def __init__(self, total_seconds):
        self.total = max(float(total_seconds or 0), 0.0)
        self.started = time.time()
        self.next_pct = 25

    def update(self, position_seconds):
        if self.total <= 0:
            return
        pct = min(int(position_seconds / self.total * 100), 100)
        while pct >= self.next_pct and self.next_pct <= 100:
            elapsed = int(time.time() - self.started)
            print(f"🎙️ Transcribing… {self.next_pct}% ({elapsed}s)", flush=True)
            self.next_pct += 25

# --- whisper singleton ------------------------------------------------------

_whisper_model = None
_whisper_key = None
_whisper_lock = threading.Lock()
# Set after a CUDA failure (e.g. VRAM exhausted by other models on the GPU)
# so every later transcription goes straight to CPU instead of re-failing.
_whisper_force_cpu = False


def _get_whisper_model():
    """Process-wide WhisperModel singleton, rebuilt if the env config changes.

    Keeping the model resident avoids a full reload per transcription (which
    on GPU would also mean re-allocating a couple of GB of VRAM per job).
    """
    global _whisper_model, _whisper_key
    cfg = get_whisper_config()
    if _whisper_force_cpu:
        cfg["device"] = "cpu"
        cfg["compute_type"] = "int8"
    key = (cfg["model_size"], cfg["device"], cfg["compute_type"])
    with _whisper_lock:
        if _whisper_model is None or _whisper_key != key:
            if "turbo" in key[0]:
                # Everything downstream reads word timings: snap_clip_to_words
                # cuts on them and the karaoke ASS highlights them one word at
                # a time. Turbo is the fast checkpoint, not the accurate one,
                # and the repo's own troubleshooting table lists it as the
                # cause of desynced captions. Warn instead of silently
                # upgrading: the operator chose it for a reason (speed), and a
                # surprise 3x slower job is worse than a line in the log.
                print(f"   ⚠️ [ASR] WHISPER_MODEL={key[0]} is a turbo checkpoint. "
                      "Word timings will be looser than whisper-large-v3 and "
                      "karaoke captions can drift; set WHISPER_MODEL=large-v3 "
                      "when caption quality matters more than speed.")
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(key[0], device=key[1], compute_type=key[2])
            _whisper_key = key
    return _whisper_model, cfg["device"]


def _run_whisper_once(media_path, **params):
    model, device = _get_whisper_model()
    gate = _ASR_GATE if device != "cpu" else _NULL_GATE
    with gate:
        segments, info = model.transcribe(media_path, **params)
        progress = _TranscribeProgress(getattr(info, "duration", 0))
        materialized = []
        for segment in segments:
            materialized.append(segment)
            progress.update(segment.end)
        # VAD trims trailing silence, so the last segment can end short of the
        # media duration — force the 100% line.
        progress.update(progress.total)
        return materialized, info


def run_whisper_transcription(media_path, **params):
    """Transcribe and FULLY materialize the segments inside the GPU gate.

    faster-whisper returns a lazy generator — decoding happens while
    iterating, so the gate must wrap list(segments), not just transcribe().
    Returns (segments_list, info).

    A CUDA failure (model load OOM or mid-decode) retries once on CPU and
    pins CPU for the rest of the process — the GPU is shared with other
    models, so a job must degrade instead of dying when VRAM runs out.
    """
    global _whisper_model, _whisper_force_cpu
    try:
        return _run_whisper_once(media_path, **params)
    except RuntimeError as e:
        if _whisper_force_cpu or "cuda" not in str(e).lower():
            raise
        print(f"⚠️ [ASR] whisper GPU failed ({e}) — retrying on CPU", flush=True)
        _whisper_force_cpu = True
        with _whisper_lock:
            _whisper_model = None  # drop the GPU model to release its VRAM
        return _run_whisper_once(media_path, **params)


def _transcribe_with_whisper(media_path):
    try:
        segments, info = run_whisper_transcription(media_path, **WHISPER_TRANSCRIBE_PARAMS)
    except Exception as e:  # MemoryError included
        # faster-whisper's VAD allocates the whole audio track as one array;
        # a long video on a small box dies deep in numpy with
        # "_ArrayMemoryError: Unable to allocate 114 MiB" (observed on a
        # 25-min video after the Groq ASR path 413'd). That message helps no
        # one. Name the real options instead.
        name = type(e).__name__
        if "MemoryError" in name or "ArrayMemory" in str(e) or "allocate" in str(e):
            raise RuntimeError(
                "Local Whisper ran out of memory on this video (the whole "
                "audio track is decoded at once). Use a cloud backend "
                "(TRANSCRIBE_BACKEND=groq with a key), a shorter video, or a "
                "machine with more free RAM.") from e
        raise

    out_segments = []
    text_parts = []
    for segment in segments:
        words = [
            {"word": w.word, "start": float(w.start), "end": float(w.end)}
            for w in (segment.words or [])
        ]
        out_segments.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text,
            "words": merge_continuation_words(words),
        })
        text_parts.append(segment.text.strip())

    return {
        "text": " ".join(part for part in text_parts if part),
        "language": info.language,
        "segments": out_segments,
    }


# --- parakeet ---------------------------------------------------------------

_parakeet_model = None
_parakeet_lock = threading.Lock()


def _get_parakeet_model():
    global _parakeet_model
    with _parakeet_lock:
        if _parakeet_model is None:
            import onnx_asr
            model = onnx_asr.load_model(
                PARAKEET_MODEL_ID,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            vad = onnx_asr.load_vad("silero")
            _parakeet_model = model.with_vad(vad).with_timestamps()
    return _parakeet_model


def _extract_wav(media_path):
    """Parakeet wants 16kHz mono PCM wav; ffmpeg-extract to a temp file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", media_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=1800)
    return wav_path


def _words_from_tokens(tokens, timestamps, seg_start, seg_end):
    """Group parakeet BPE tokens into words with absolute timestamps.

    Verified on the prod model: tokens already carry the leading-space
    word-start convention (" T", "odo", " el", ...) and timestamps are token
    START times in seconds relative to the VAD segment. A token without a
    leading space (subword continuations, punctuation like ",") belongs to
    the previous word — same semantics merge_continuation_words expects.
    Word end is inferred: next word's start, capped near the word's last
    token so a long inter-word silence doesn't stretch the highlight.
    """
    words = []
    last_token_ts = []
    for token, ts in zip(tokens, timestamps):
        if not token:
            continue
        abs_ts = float(ts) + seg_start
        if token.startswith(" ") or not words:
            words.append({
                "word": token if token.startswith(" ") else " " + token,
                "start": abs_ts,
            })
            last_token_ts.append(abs_ts)
        else:
            words[-1]["word"] += token
            last_token_ts[-1] = abs_ts

    for i, word in enumerate(words):
        next_start = words[i + 1]["start"] if i + 1 < len(words) else seg_end
        cap = last_token_ts[i] + 0.6
        word["end"] = float(max(word["start"] + 0.05, min(next_start, cap)))

    return words


def _transcribe_with_parakeet(media_path):
    model = _get_parakeet_model()
    wav_path = _extract_wav(media_path)
    try:
        # 16kHz mono s16le wav -> 32000 bytes per second of audio.
        try:
            duration = os.path.getsize(wav_path) / 32000.0
        except OSError:
            duration = 0.0
        with _ASR_GATE:
            progress = _TranscribeProgress(duration)
            results = []
            for seg in model.recognize(wav_path):
                results.append(seg)
                progress.update(float(seg.end))
            progress.update(progress.total)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    out_segments = []
    text_parts = []
    for seg in results:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_text = str(seg.text or "").strip()
        if not seg_text:
            continue
        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": _words_from_tokens(
                list(seg.tokens or []), list(seg.timestamps or []),
                seg_start, seg_end,
            ),
        })
        text_parts.append(seg_text)

    text = " ".join(text_parts)
    return {
        "text": text,
        "language": _detect_language(text),
        "segments": out_segments,
    }


def _detect_language(text):
    """Parakeet doesn't report a language; classify the transcribed text.

    py3langid is pure-Python and returns ISO 639-1 codes compatible with the
    whisper codes the pipeline expects (thumbnail titles, Gemini prompts).
    """
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        import py3langid
        lang, _score = py3langid.classify(sample[:4000])
        return lang
    except Exception:
        return "en"


def _parakeet_fallback_reason(transcript, duration_hint=None):
    """Return why the parakeet result is untrustworthy, or None if it's fine."""
    segments = transcript.get("segments") or []
    total_words = sum(len(s.get("words") or []) for s in segments)
    if total_words == 0:
        return "no words recognized"
    language = transcript.get("language")
    if language not in PARAKEET_LANGS:
        return f"language '{language}' outside parakeet's supported set"
    duration = duration_hint or (segments[-1]["end"] if segments else 0)
    # Real speech averages >100 wpm; under ~12 wpm on a long video means the
    # audio was mostly not recognized (e.g. unsupported language or music).
    if duration > 60 and total_words < duration * 0.2:
        return f"only {total_words} words in {duration:.0f}s of audio"
    return None


# --- groq (cloud ASR) --------------------------------------------------------

def _groq_config():
    """Groq endpoint config, re-read per call so env changes apply live."""
    return {
        "api_key": os.environ.get("GROQ_API_KEY", ""),
        "base_url": os.environ.get(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        "model": os.environ.get("GROQ_ASR_MODEL", "whisper-large-v3"),
        "max_upload_mb": float(os.environ.get("GROQ_MAX_UPLOAD_MB", "24")),
    }


def _groq_transcribe_file(path, language=None):
    """POST one audio file to Groq; returns the verbose_json payload.

    Retries 429/5xx/transport errors with backoff — the free tier throttles
    aggressively and 503s while models cold-load. With a key pool
    (GROQ_API_KEYS) a 429 rotates to the next key immediately instead of
    sleeping, and only real 5xx/transport failures back off. Other statuses
    raise with the response body (bad key, unsupported model) so the caller
    can fall back instead of retrying forever.
    """
    import httpx

    cfg = _groq_config()
    data = {
        "model": cfg["model"],
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["segment", "word"],
    }
    if language:
        data["language"] = language

    # A key pool (GROQ_API_KEYS) multiplies the free tier: a 429 rides the next
    # key straight away instead of backing off against the same exhausted one.
    import groq_pool
    pool = groq_pool.get_pool()
    multi = len(pool) > 1

    last_err = None
    rotated_prev = False
    waited = 0  # per chunk, at most one window wait per key pool
    for attempt in range(8):
        if attempt and not rotated_prev:
            time.sleep(min(2 ** attempt * 2, 15))
        rotated_prev = False
        key = pool.pick() if multi else cfg["api_key"]
        try:
            with open(path, "rb") as f:
                response = httpx.post(
                    f"{cfg['base_url']}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files={"file": (os.path.basename(path), f, "audio/flac")},
                    timeout=httpx.Timeout(600.0, connect=30.0),
                )
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 and multi:
                pool.penalize(key)
                recovery = pool.soonest_recovery()
                if recovery is not None and waited < 1 and recovery <= 95:
                    # Every key is benched AND rotation is exhausted: waiting
                    # out one token window beats falling back to 15-minute
                    # local whisper for the whole video.
                    print(f"   ⏳ All Groq keys rate-limited — waiting "
                          f"{recovery:.0f}s for the window to refill")
                    time.sleep(recovery + 0.5)
                    waited += 1
                    rotated_prev = True  # the wait WAS the wait: no backoff too
                    continue
                rotated_prev = True
                last_err = RuntimeError(
                    f"groq HTTP 429 on {groq_pool._mask(key)} — rotating to next key")
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_err = RuntimeError(
                    f"groq HTTP {response.status_code}: {response.text[:200]}")
                continue
            raise RuntimeError(
                f"groq HTTP {response.status_code}: {response.text[:200]}")
        except (httpx.TransportError, TimeoutError) as e:
            last_err = e
    raise last_err


def _words_from_segment_text(text, seg_start, seg_end):
    """Evenly spaced words for payloads without word timestamps.

    whisper-large-v3-turbo on Groq doesn't support word-level granularity;
    karaoke captions need word entries, so distribute the segment's words
    evenly across its duration. Leading-space convention matches
    faster-whisper tokens.
    """
    tokens = [t for t in str(text or "").split() if t]
    if not tokens:
        return []
    step = max(seg_end - seg_start, 0.1 * len(tokens)) / len(tokens)
    return [
        {"word": " " + tok, "start": float(seg_start + i * step),
         "end": float(seg_start + (i + 1) * step)}
        for i, tok in enumerate(tokens)
    ]


def _groq_payload_to_transcript(payload, offset=0.0):
    """Map a Groq verbose_json payload into the transcribe_media contract.

    Groq reports one top-level `words` list of complete words ("Hello",
    "world,") while the pipeline expects faster-whisper tokens that carry a
    LEADING SPACE on word starts. Prepending a space to every word makes
    merge_continuation_words a no-op (Groq never splits a word) and keeps the
    karaoke highlight per real word. Each word joins the segment whose
    [start, end) window contains it.
    """
    words = payload.get("words") or []
    segments = payload.get("segments") or []
    if segments and not words:
        print("⚠️ [ASR] groq returned no word timestamps — "
              f"GROQ_ASR_MODEL={_groq_config()['model']} may not support word "
              "granularity (whisper-large-v3 does, -turbo does not)")

    out_segments = []
    text_parts = []
    wi = 0
    for seg in segments:
        seg_start = float(seg["start"]) + offset
        seg_end = float(seg["end"]) + offset
        seg_text = str(seg.get("text") or "").strip()
        seg_words = []
        while wi < len(words) and float(words[wi]["start"]) + offset < seg_end:
            w = words[wi]
            seg_words.append({
                "word": " " + str(w["word"]),
                "start": float(w["start"]) + offset,
                "end": float(w.get("end") if w.get("end") is not None
                             else w["start"]) + offset,
            })
            wi += 1
        if not seg_words:
            seg_words = _words_from_segment_text(seg_text, seg_start, seg_end)
        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": merge_continuation_words(seg_words),
        })
        text_parts.append(seg_text)

    # Word granularity without segments: synthesize one so the pipeline still
    # gets a usable transcript.
    if not out_segments and words:
        seg_words = [{
            "word": " " + str(w["word"]),
            "start": float(w["start"]) + offset,
            "end": float(w.get("end") if w.get("end") is not None
                         else w["start"]) + offset,
        } for w in words]
        text = " ".join(str(w["word"]).strip() for w in words)
        out_segments.append({
            "start": seg_words[0]["start"],
            "end": seg_words[-1]["end"],
            "text": text,
            "words": seg_words,
        })
        text_parts = [text]

    return {
        "text": " ".join(part for part in text_parts if part),
        "language": _normalize_language(payload.get("language")),
        "segments": out_segments,
    }


def _normalize_language(lang):
    """Convert Groq's full language names (e.g., "English") to ISO codes ("en")."""
    if not lang:
        return "en"
    lang_lower = str(lang).strip().lower()
    # Common full names -> ISO 639-1
    name_to_code = {
        "english": "en", "spanish": "es", "french": "fr", "german": "de",
        "italian": "it", "portuguese": "pt", "russian": "ru", "chinese": "zh",
        "japanese": "ja", "korean": "ko", "arabic": "ar", "hindi": "hi",
        "dutch": "nl", "polish": "pl", "turkish": "tr", "vietnamese": "vi",
    }
    return name_to_code.get(lang_lower, lang_lower[:2] if len(lang_lower) >= 2 else "en")


def _ffprobe(args, path, timeout=120):
    """Run ffprobe with these args and return its non-empty stdout lines."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", *args, str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return []
    return [line.strip() for line in (out.stdout or "").splitlines() if line.strip()]


def _probe_pts_range(path):
    """(first_packet_seconds, duration_seconds) read from the packet table.

    A demux-only read (no decode), and the only measurement that works on the
    FLAC parts the segment muxer writes: those carry no duration in their
    header, so ``format=duration`` answers ``N/A``. The muxer also leaves the
    timestamps stamped against the ORIGINAL media, which makes the first
    packet's time the part's true offset in the file.
    """
    lines = _ffprobe(["-select_streams", "a:0", "-show_entries",
                      "packet=pts_time,duration_time", "-of", "csv=p=0"], path)
    if not lines:
        return 0.0, 0.0

    def _head(line):
        # "60.022938,0.045437" -> (60.022938, 0.045437); N/A fields become 0.0.
        values = []
        for field in line.split(","):
            try:
                values.append(float(field))
            except ValueError:
                values.append(0.0)
        return values

    first = _head(lines[0])
    last = _head(lines[-1])
    start = first[0]
    end = (last[0] + last[1]) if len(last) > 1 else last[0]
    return start, max(0.0, end - start)


def _probe_duration(path):
    """Media/audio duration in seconds via ffprobe; 0.0 when unreadable.

    Falls back to the packet table when the header has no duration — without
    it, the muxer's FLAC parts measured 0.0 and every chunk offset silently
    collapsed onto the start of the file (2-nov-2026: a 4-hour video came back
    as 4 041 segments inside its first 9 minutes, which is what made the cuts
    land mid-sentence and the captions read late on every later clip).
    """
    for entries in ("format=duration", "stream=duration"):
        lines = _ffprobe(["-select_streams", "a:0", "-show_entries", entries,
                          "-of", "csv=p=0"], path, timeout=60)
        for line in lines:
            try:
                value = float(line)
            except ValueError:
                continue
            if value > 0:
                return value
    return _probe_pts_range(path)[1]


def _groq_part_plan(size_bytes, duration, max_bytes):
    """Seconds per upload part to stay under the cap, or None for one upload.

    Parts target 90% of the cap so a part can't land exactly on it. There is
    deliberately NO floor: dense audio (game footage, music, several voices)
    runs far above the ~1 MB/min speech figure, and a 300s floor on 12 MB/min
    audio produces 58 MB parts — a guaranteed 413 (that killed cloud ASR on
    the 1000-players Minecraft video, 17-oct-2026).

    An unreadable duration with an over-cap file still has to be split:
    returning ``None`` there used to hand the whole 4-hour FLAC to the API for
    a 413, so a conservative slice is planned instead of giving up.
    """
    if size_bytes <= max_bytes:
        return None
    if duration <= 0:
        return _UNKNOWN_DURATION_PART_SECS
    per_sec = size_bytes / duration
    part_secs = int((max_bytes * 0.9) / per_sec)
    return max(30, part_secs)


# Dense 16 kHz mono audio runs ~1 MB/min; 5 minutes stays under a 24 MB cap
# even for material that compresses poorly.
_UNKNOWN_DURATION_PART_SECS = 300


def _extract_groq_audio(media_path, tmpdir):
    """16kHz mono FLAC extraction — the upload format for every part."""
    full_path = os.path.join(tmpdir, "audio.flac")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(media_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", full_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=1800)
    return full_path


def _split_audio_for_groq(media_path, tmpdir, max_bytes):
    """Return [(part_path, offset_seconds)] covering the media's audio.

    Every offset is READ FROM THE PART (its first packet's timestamp), never
    accumulated from probed durations. The segment muxer leaves each part's
    timestamps stamped against the original media and writes no duration in
    the header, so the old running-offset total added 0.0 to every chunk: all
    23 parts of a 4-hour video were posted as if they each started at 0 s,
    their segments piled up inside the first 9 minutes, and clip cuts and
    karaoke captions inherited the drift (2-nov-2026).

    Two layers keep every part under Groq's upload cap: the plan targets 90%
    of the bytes, and anything that still overshoots — dense audio spikes far
    above speech's ~1 MB/min — is re-cut by TIME until it fits (``-fs`` is
    never used: a byte cap truncates mid-FLAC-frame and corrupts the file,
    which is how the 2-nov resplit's own sub_ files came back unreadable).
    Files are not deleted here: the caller's finally-block clears tmpdir, and
    deleting something still in the return value is how the old code broke its
    own parts (FileNotFoundError on its own sub_ files).
    """
    full_path = _extract_groq_audio(media_path, tmpdir)
    part_secs = _groq_part_plan(
        os.path.getsize(full_path), _probe_duration(full_path), max_bytes)
    if part_secs is None:
        return [(full_path, 0.0)]

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", full_path,
         "-f", "segment", "-segment_time", str(part_secs), "-c:a", "flac",
         os.path.join(tmpdir, "part_%05d.flac")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=1800,
    )
    names = sorted(name for name in os.listdir(tmpdir)
                   if name.startswith("part_") and name.endswith(".flac"))
    if not names:
        return [(full_path, 0.0)]

    fixed = []
    for slot, name in enumerate(names):
        path = os.path.join(tmpdir, name)
        offset, duration = _probe_pts_range(path)
        if duration <= 0:
            continue  # no audio packets: a boundary artefact, not content
        for part, part_offset in _fit_under_cap(
                path, offset, duration, max_bytes, tmpdir, f"p{slot:05d}"):
            fixed.append((part, part_offset))
    if fixed:
        return fixed

    # Nothing measured — ffprobe could not read any part. Numbering them by the
    # plan is approximate but monotonic, and that beats the old failure where
    # every chunk landed on 0 s (2-nov-2026).
    print("   ⚠️ [ASR] no chunk offsets readable; falling back to the planned "
          f"{part_secs}s cadence for {len(names)} parts")
    return [(os.path.join(tmpdir, name), slot * part_secs)
            for slot, name in enumerate(names)]


MAX_RESPLITS = 4


def _fit_under_cap(path, offset, duration, max_bytes, tmpdir, tag, depth=0):
    """[(path, absolute_offset)] for one part, halved until it fits the cap.

    Each cut gets its own file prefix: a shared ``sub_%05d`` let every ffmpeg
    invocation overwrite the previous part's sub files, so two oversized parts
    collected the same (last-written) files and their offsets drifted — the
    first form of the captions-don't-match bug (2-nov).

    Sub-part offsets stay absolute whatever the muxer does with the clock: the
    muxer normally keeps the parent's stamp, and a reset-to-zero sub file
    (``offset < parent``) is re-based on the parent instead of trusted.
    """
    size = os.path.getsize(path)
    if duration <= 0:
        return []  # no audio packets: a boundary artefact, not content
    if size <= max_bytes or depth >= MAX_RESPLITS or duration <= 1.0:
        if size > max_bytes:
            print(f"   ⚠️ [ASR] {os.path.basename(path)} still exceeds the upload "
                  f"cap after {depth} re-splits ({size / 1e6:.1f} MB); dropping "
                  "this slice — a gap beats a 413")
            return []
        return [(path, offset)]

    sub_secs = max(1, int(duration * (max_bytes / size) * 0.9))
    prefix = f"sub_{tag}_{depth}_"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-f", "segment", "-segment_time", str(sub_secs), "-c:a", "flac",
         os.path.join(tmpdir, prefix + "%05d.flac")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        timeout=1800,
    )
    out = []
    for sub_slot, name in enumerate(sorted(n for n in os.listdir(tmpdir)
                                           if n.startswith(prefix) and n.endswith(".flac"))):
        sub_path = os.path.join(tmpdir, name)
        sub_offset, sub_duration = _probe_pts_range(sub_path)
        if sub_offset < offset - 0.05:  # this muxer reset the clock
            sub_offset += offset
        out.extend(_fit_under_cap(sub_path, sub_offset, sub_duration, max_bytes,
                                  tmpdir, f"{tag}_{sub_slot:03d}", depth + 1))
    return out


def _free_gb(path):
    import shutil
    try:
        return shutil.disk_usage(path).free / 1024 ** 3
    except OSError:
        return -1.0


def _asr_tmpdir(min_free_gb=2.0):
    """Where ASR scratch lives (extraction + chunk splits) — the Groq path
    ffmpeg-extracts a large FLAC here BEFORE the API is ever called.

    The system temp default lives on a full OS drive: a 16 kHz mono FLAC of
    a ~1 h video is 300–500 MB, and with C: at 232 MB every Groq job died
    with ffmpeg exit -28 (ENOSPC) before reaching the API — the fallback
    then OOM'd on local whisper (observed 1-nov-2026 on the 10,000-players
    Minecraft video). ASR_TMPDIR wins when set; otherwise the candidate with
    the most free space among the fixed drives. Falls back to the system
    temp only when nothing else is writable.
    """
    explicit = (os.environ.get("ASR_TMPDIR") or "").strip()
    if explicit:
        # An explicitly configured path is honored unconditionally: the
        # operator chose it, and second-guessing them with a free-space check
        # is how a pinned volume gets silently ignored.
        try:
            os.makedirs(explicit, exist_ok=True)
            if os.access(explicit, os.W_OK):
                return explicit
        except OSError:
            pass
    candidates = []
    candidates.append(os.path.join(tempfile.gettempdir(), "openshorts_asr"))
    for letter in ("E", "D", "C"):
        candidates.append(f"{letter}:\\openshorts_asr")
    best, best_free = None, -1.0
    for cand in candidates:
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
            if not os.access(cand, os.W_OK):
                continue
        except OSError:
            continue
        free = _free_gb(cand)
        if free >= min_free_gb:
            return cand
        if free > best_free:
            best, best_free = cand, free
    return best or tempfile.mkdtemp(prefix="openshorts_asr_")


def _transcribe_with_groq(media_path):
    """Cloud ASR via Groq's OpenAI-compatible Whisper endpoint.

    Turns transcription — the heaviest local CPU stage — into an API call a
    weak machine can drive; no GPU gate because nothing runs locally. Audio
    is extracted to 16kHz mono FLAC, split under Groq's upload cap, and each
    part's timestamps are offset back to absolute file time. Language from
    the first part is passed as a hint to the rest for consistency. A missing
    key raises so transcribe_media can fall back to local whisper.
    """
    cfg = _groq_config()
    if not cfg["api_key"]:
        raise RuntimeError(
            "GROQ_API_KEY is not set — TRANSCRIBE_BACKEND=groq needs it "
            "(console.groq.com, free tier works)")

    tmpdir = tempfile.mkdtemp(prefix="groq_asr_", dir=_asr_tmpdir())
    try:
        parts = _split_audio_for_groq(
            media_path, tmpdir, cfg["max_upload_mb"] * 1024 * 1024)
        # Progress against the source length, not the sum of the parts: the FLAC
        # muxer copies the WHOLE file's sample count into each part's header, so
        # adding the parts up multiplies the total, and the parts' headers are
        # N/A anyway. Each part's offset is already "seconds of audio behind us".
        progress = _TranscribeProgress(_probe_duration(media_path))
        segments = []
        text_parts = []
        language = None
        for path, offset in parts:
            progress.update(offset)
            payload = _groq_transcribe_file(path, language=language)
            language = language or _normalize_language(payload.get("language"))
            transcript = _groq_payload_to_transcript(payload, offset=offset)
            segments.extend(transcript["segments"])
            text_parts.append(transcript["text"])
        progress.update(progress.total)
        return {
            "text": " ".join(part for part in text_parts if part),
            "language": str(language or "en"),
            "segments": segments,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- public entry point -----------------------------------------------------

class NoAudioError(Exception):
    """The media has no audio track — nothing to transcribe."""


def _has_audio_stream(media_path) -> bool:
    """True if the file has at least one audio stream (ffprobe)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=60,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True  # probe failed — don't block, let the backend try


def transcribe_media(media_path):
    """Transcribe with the configured backend, falling back to whisper."""
    # Silent videos (AI-generated clips, muted screen recordings) have no audio
    # stream; every ASR backend then crashes deep inside libav with an opaque
    # "tuple index out of range". Detect it up front and fail with a clear,
    # actionable reason instead.
    if not _has_audio_stream(media_path):
        raise NoAudioError(
            "This video has no audio track. OpenShorts finds viral moments from "
            "speech, so it needs a video with audio.")

    backend = os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()

    if backend == "parakeet":
        try:
            transcript = _transcribe_with_parakeet(media_path)
            reason = _parakeet_fallback_reason(transcript)
            if reason is None:
                print(f"🎙️ [ASR] parakeet ok: lang={transcript['language']} "
                      f"segments={len(transcript['segments'])}")
                return transcript
            print(f"⚠️ [ASR] parakeet result rejected ({reason}) — "
                  f"falling back to whisper")
        except Exception as e:
            print(f"⚠️ [ASR] parakeet failed ({type(e).__name__}: {e}) — "
                  f"falling back to whisper")

    if backend == "groq":
        try:
            transcript = _transcribe_with_groq(media_path)
            print(f"🎙️ [ASR] groq ok: lang={transcript['language']} "
                  f"segments={len(transcript['segments'])}")
            return transcript
        except Exception as e:
            print(f"⚠️ [ASR] groq failed ({type(e).__name__}: {e}) — "
                  f"falling back to whisper")

    return _transcribe_with_whisper(media_path)
