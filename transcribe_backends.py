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
    segments, info = run_whisper_transcription(media_path, **WHISPER_TRANSCRIBE_PARAMS)

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
    aggressively and 503s while models cold-load. Other statuses raise with
    the response body (bad key, unsupported model) so the caller can fall
    back instead of retrying forever.
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

    last_err = None
    for attempt in range(4):
        if attempt:
            time.sleep(min(2 ** attempt * 2, 15))
        try:
            with open(path, "rb") as f:
                response = httpx.post(
                    f"{cfg['base_url']}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                    data=data,
                    files={"file": (os.path.basename(path), f, "audio/flac")},
                    timeout=httpx.Timeout(600.0, connect=30.0),
                )
            if response.status_code == 200:
                return response.json()
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


def _probe_duration(path):
    """Media/audio duration in seconds via ffprobe; 0.0 when unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def _groq_part_plan(size_bytes, duration, max_bytes):
    """Seconds per upload part to stay under the cap, or None for one upload.

    The 300s floor assumes 16kHz mono FLAC speech (~1 MB/min worst case) —
    a cap that tight would mean the request itself is the problem.
    """
    if size_bytes <= max_bytes or duration <= 0:
        return None
    part_secs = int(max_bytes / (size_bytes / duration))
    return max(min(part_secs, 3600), 300)


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

    Offsets come from each part's actual probed duration, so a split landing
    on a stream discontinuity can't drift the transcript. Parts are named
    part_%05d.flac by the segment muxer.
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
    parts = []
    offset = 0.0
    for name in sorted(os.listdir(tmpdir)):
        if name.startswith("part_"):
            path = os.path.join(tmpdir, name)
            parts.append((path, offset))
            offset += _probe_duration(path)
    return parts or [(full_path, 0.0)]


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

    tmpdir = tempfile.mkdtemp(prefix="groq_asr_")
    try:
        parts = _split_audio_for_groq(
            media_path, tmpdir, cfg["max_upload_mb"] * 1024 * 1024)
        progress = _TranscribeProgress(
            sum(_probe_duration(p) for p, _ in parts))
        segments = []
        text_parts = []
        language = None
        done = 0.0
        for path, offset in parts:
            payload = _groq_transcribe_file(path, language=language)
            language = language or _normalize_language(payload.get("language"))
            transcript = _groq_payload_to_transcript(payload, offset=offset)
            segments.extend(transcript["segments"])
            text_parts.append(transcript["text"])
            done += _probe_duration(path)
            progress.update(done)
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
