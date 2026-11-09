# OpenShorts Pipeline — Handover Document

*Corrected against the code on 2026-10-16. Several claims from the previous
version of this document were wrong (see "Corrections vs. the old handover"
at the bottom); when in doubt, trust this file and the code, not memory.*

## 1. Project Overview

Automated long-video → 9:16 vertical shorts pipeline. Ingest a raw long video
(YouTube URL or local file), detect highlight moments with AI, cut them to
vertical format, burn in word-level karaoke captions, and output platform-ready
shorts with titles/descriptions/hashtags.

**Two operating modes exist in this repo — know which one you are in:**

- **Self-host mode (what you normally run):** `BILLING_ENABLED` unset. Groq
  for ASR and moment scoring, Gemini only for the frame-based stages (layout
  picking, facecam detection, critique gate). Zero local AI compute. This is
  what `.env` currently configures.
- **Cloud/billing mode:** `BILLING_ENABLED=1` enables `cloud/` (accounts,
  API keys, analytics, retention limits). The moment picker defaults back to
  Gemini (`gemini-3.1-flash-lite` per CLAUDE.md). The repo's `CLAUDE.md`
  describes this mode (port 8000 + Vite dashboard); don't confuse its
  architecture table with the self-host pipeline.

### Core stack

| Layer | Choice | Notes |
|---|---|---|
| Ingest | yt-dlp | YouTube download; `YOUTUBE_COOKIES` in `.env` bypasses bot detection |
| Transcription | faster-whisper (local) **or** Groq `whisper-large-v3` (cloud) | `TRANSCRIBE_BACKEND=whisper|parakeet|groq`; a third option, `parakeet` (onnx-asr, GPU image only), auto-falls back to whisper |
| Scene detection | PySceneDetect | Scene boundaries for scene-aware layout/crop windows |
| Moment picking | Groq `openai/gpt-oss-120b` (or any OpenAI-compatible LLM) **or** Gemini 3.x | Two-pass: transcript-window scoring, then a detail pass. **Text-only** — the scorer never sees frames |
| Layout / crop | Gemini 3.x + MediaPipe (BlazeFace) | This is where vision is actually used: pick 9:16 layout, detect facecam windows |
| Critique | Gemini vision on sampled frames of the finished clip (`CRITIQUE_GATE=1`) | Fail → re-render |
| Cutting | FFmpeg + `camera_inset.gaming_split_filtergraph` | Scene-wise crop windows; facecam PiP for gaming splits |
| Captions | ASS karaoke via libass burn-in | Word-level timestamps from Whisper; highlight + pop/box styles |
| Output | MP4 (libx264 CRF 18 / NVENC), ASS sidecar, `*_metadata.json` | 1080×1920; `loudnorm=I=-14:TP=-2.0:LRA=11` |

## 2. Repository layout (`D:\openshorts`)

| File | Purpose |
|---|---|
| `main.py` | CLI entry: ingest → transcribe → score → cut → render → captions. Argparse flags: `-i/--input`, `-u/--url`, `-o/--output`, `--format auto\|vertical\|horizontal\|square`, `--transcript`, `--max-concurrent`, `--keep-original`, `--skip-analysis` |
| `app.py` | FastAPI server. UI at `/` (serves `static_ui/index.html`); `/api/process`, `/api/status/{job_id}`, `/api/source/{job_id}`, `/api/clip/...`; static mounts `/videos`, `/thumbnails` |
| `static_ui/index.html` | The working web UI (submit → live logs → download clips) |
| `dashboard/` | React/Vite frontend for cloud mode (port 5173/5175) |
| `transcribe_backends.py` | ASR backends. `_groq_transcribe_file` (transcribe_backends.py:350), `_split_audio_for_groq` (:535), `_transcribe_with_groq` (:565), `_normalize_language` (:483, "English"→ISO) |
| `clip_selection.py` | Pure selection helpers: windowing, `snap_clip_to_words` (:199), `refine_clip_boundaries` (:264), `clip_count_targets`, `clip_duration_bounds` (:105, default 15–60 s) |
| `reframe_v2.py` | FFmpeg-native cut/reframe engine; calls `gaming_split_filtergraph` at reframe_v2.py:600 |
| `camera_inset.py` | `gaming_split_filtergraph` (camera_inset.py:367) — full-facecam PiP composition |
| `facecam_extractor.py` | Facecam window detection (Gemini + local MediaPipe fallback, cached per source) |
| `layout_picker.py` / `layout_ranges.py` | 9:16 layout choice with score-ranked layout ranges |
| `critique_gate.py` | Post-render visual QA: Gemini judges sampled frames; objective FAILs (e.g. duplicate facecam) trigger re-render |
| `hook_grounding.py` / `hooks.py` | Hook text generation (grounded in transcript) and emoji-aware text card rendering |
| `llm_backend.py` | OpenAI-compatible LLM client: TPM pacing, retry on 429/5xx |
| `ffmpeg_utils.py` | Encoder selection (x264 default, `nvenc`, `auto`), quality tiers (CRF 18/22), metadata scrub, `LOUDNORM_FILTER` (ffmpeg_utils.py:84) |
| `subtitles.py` | Word-level ASR, continuation-word merging, karaoke ASS generation |
| `cloud/` | Billing/account code (cloud mode only) |
| `.env` / `.env.example` | Runtime config; the example file is the authoritative variable reference |
| `CLAUDE.md` | Repo instructions — but describes the cloud/billing mode; read §1 here first |

## 3. Quick start

**Self-host web UI (the normal way):**

```bash
cd /d/openshorts
.venv311/Scripts/uvicorn.exe app:app --host 0.0.0.0 --port 8001
# open http://localhost:8001
```

**CLI:**

```bash
.venv311/Scripts/python.exe main.py -u "https://youtube.com/watch?v=ID" -o D:/output
.venv311/Scripts/python.exe main.py -i C:/path/video.mp4 -o D:/output --format vertical
```

**Portable EXE** (PyInstaller, self-contained; needs ffmpeg on PATH):

```cmd
D:\openshorts_portable\main.exe -i "C:\path\to\video.mp4" -o D:\output --format vertical
```

> `.env` is read at process start. Changing it requires restarting the server.

## 4. Configuration (`.env`)

Active values as of 2026-10-16:

| Variable | Active value | Purpose |
|---|---|---|
| `TRANSCRIBE_BACKEND` | `groq` | `whisper` (default) \| `parakeet` \| `groq` |
| `GROQ_API_KEY` | *(set)* | Groq ASR + LLM key |
| `GROQ_ASR_MODEL` | `whisper-large-v3` | **Required for word timestamps** — large-v3-turbo has no word granularity |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq ASR endpoint (separate from `LLM_BASE_URL`) |
| `GROQ_MAX_UPLOAD_MB` | `24` | Audio chunk split cap for upload |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint (Groq, Ollama, OpenRouter…) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Moment-scoring model |
| `LLM_API_KEY` | *(set)* | Bearer token for the scoring endpoint |
| `LLM_SCORE_BATCH` | `1` | Transcript windows per scoring call; 1 is safest on free-tier TPM |
| `LLM_TIMEOUT` | — | Default 600 s |
| `LLM_MAX_TOKENS` | — | Default 16384; matters for reasoning models that spend tokens thinking |
| `LLM_TPM_LIMIT` | — | Token/minute budget; learned from `x-ratelimit-limit-tokens` (Groq free = 8000) when unset |
| `GROQ_API_KEYS` | — | Comma-separated **key pool** for Groq (falls back to `GROQ_API_KEY`). Rotation on 429 covers BOTH the Groq LLM scoring and the Groq ASR transcription, so N keys ≈ N× the free-tier ceiling. See §7 |
| `LLM_RATE_LIMIT_ATTEMPTS` | — | Retries after waiting out the token window; default 4 |
| `CLIP_SCORE_CONCURRENCY` | — | Parallel scoring calls; default 1 |
| `CLIP_MIN_SECONDS` / `CLIP_MAX_SECONDS` | 15 / 60 | Clip length band (default 15–60 s, **not** 20–60) |
| `CLIP_TARGET_MIN` / `CLIP_TARGET_MAX` | — | Override the detail-pass clip-count floor/ceiling |
| `AUTO_LAYOUT` | `1` | Auto 9:16 layout pick |
| `CRITIQUE_GATE` | `1` | Post-render visual QA + re-render on fail |
| `CAPTION_MARGIN_V` | `32` | Caption vertical margin (frames) |
| `AUDIO_NORMALIZE` | on | `0` disables the loudnorm pass |
| `ASR_TMPDIR` | auto (roomiest drive) | Scratch dir for the Groq ASR extraction — **must be a drive with room**: the FLAC is 300–500 MB and the OS-drive temp fills up |
| `WHISPER_MODEL` | `small` | Local faster-whisper model; `small` > `base` on non-English audio |
| `FFMPEG_ENCODER` | `x264` | `x264` (default) \| `nvenc` \| `auto` (probes NVENC once, falls back) |
| `QUALITY_GATE_MIN_HEIGHT` | 720 | Pre-flight warning for low-res YouTube sources; 0 disables |
| `YOUTUBE_COOKIES` | — | Netscape-format cookies for bot-gated videos |

## 5. Pipeline (what actually happens)

```
INPUT (file / YouTube URL)
 │
 0. INGEST & PREP — yt-dlp download → 16 kHz mono audio for ASR → PySceneDetect scenes
 │
 1. TRANSCRIBE — faster-whisper | parakeet | Groq whisper-large-v3
    → transcript {text, language, segments[{start, end, text, words[{word, start, end}]}]}
    Groq: 16 kHz mono FLAC, split under GROQ_MAX_UPLOAD_MB, chunks offset-merged
 │
 2. HIGHLIGHT DETECTION (two-pass, text-only)
    Pass A  — transcript windowing (build_transcript_windows) → LLM scores each window
              (TPM-paced, LLM_SCORE_BATCH windows per call); top windows shortlisted
    Pass B  — detail pass on the shortlist: strict JSON per clip
              {start, end, title, predicted_score, reason, …}; ordered/scored by score
    NO frames and no audio energy are sent to the scorer. Vision enters only later.
 │
 3. CUT + REFRAME (FFmpeg + MediaPipe + Gemini layout picker)
    - snap_clip_to_words: cuts land in the pause around word boundaries
      (search ±1.5 s, lead ≤0.35 s / tail ≤0.45 s into silence) — no mid-word cuts
    - refine_clip_boundaries: trims 1.5–3 s of dead air at each edge, keeps
      0.4/0.5 s breaths, max 0.4 s trim per side, never below min_duration
    - Scene-wise crop windows (crop only moves at hard scene cuts — no per-frame jitter)
    - GAMING_SPLIT: facecam window (the full rectangle, not just the face) cut out
      of gameplay and re-laid as PiP via camera_inset.gaming_split_filtergraph
    - Re-encode to 1080×1920 (libx264 CRF 18 tier / NVENC cq≈crf+7)
 │
 4. CAPTIONS + POLISH
    - Word-level karaoke ASS (highlight color, "pop" scale / "box" outline styles)
      built from the Stage-1 transcript — no second ASR
    - Hook title card (emoji-aware rendered overlay, hooks.py)
    - loudnorm=I=-14:TP=-2.0:LRA=11 at the clip encode (TP=−2.0 because AAC adds
      inter-sample peaks on top; measured on 14 clips)
 │
 5. CRITIQUE GATE + OUTPUT
    - CRITIQUE_GATE: Gemini judges sampled frames of the finished clip; objective
      FAILs (e.g. facecam visible twice, layout promise broken) re-render
    - Output: clip_NN.mp4, .ass sidecar, {video_title}_metadata.json
      (titles, TikTok/Instagram descriptions, scores)
```

## 6. Run a video end-to-end

1. Start the server (see §3) and open `http://localhost:8001`.
2. Paste a YouTube URL or upload a file; pick format and max clips.
3. Click Generate; watch the live log stream.
4. Download clips from the results list.

CLI and portable-EXE equivalents are in §3.

## 7. Critical gotchas

| Issue | Reality / fix |
|---|---|
| Groq free-tier TPM | 8,000 tokens/min; `LLM_SCORE_BATCH=1` + `CLIP_SCORE_CONCURRENCY=1` + TPM pacing in `llm_backend.py`. 429s are retried after waiting out the window, not a fixed sleep |
| Groq 413 / payload too large | `LLM_MAX_TOKENS`, batch size; free tiers fail on oversized completions, JSON validator then rejects the empty generation |
| Word timestamps missing | Use `whisper-large-v3` via Groq; local turbo models lack word granularity → captions desync |
| Mid-word cuts | `snap_clip_to_words` + `refine_clip_boundaries` prevent them; if cuts look ragged, check the words array actually reached the cutter |
| Duplicate facecam in gaming splits | The critique gate catches this and re-renders; root cause is a crop that failed to exclude the facecam rectangle |
| Jittery reframing | Crop windows move only at hard scene cuts |
| `GROQ_BASE_URL` vs `LLM_BASE_URL` | Two different endpoints: ASR uses `GROQ_BASE_URL`, scoring uses `LLM_BASE_URL` |
| Output volume varies between clips | Loudnorm is applied to every clip; if a clip still sounds quiet, check `AUDIO_NORMALIZE` wasn't set to 0 |
| YouTube bot detection | Set `YOUTUBE_COOKIES` (Netscape format) in `.env`, or download manually and use file input |
| Repo `CLAUDE.md` disagrees with this doc | CLAUDE.md describes cloud/billing mode (Gemini picker, port 8000). Self-host is described here |
| CapCut PC automation | Not used — FFmpeg does everything |

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `GROQ_API_KEY required` | Missing key with `TRANSCRIBE_BACKEND=groq` | Set it in `.env`, restart |
| 429s during scoring | TPM budget exceeded | `LLM_SCORE_BATCH=1`; TPM pacing is automatic |
| yt-dlp 403 "Sign in to confirm you're not a bot" | Per-IP bot check — datacenter IPs get flagged after the first request; personal IPs get hit too after many downloads in a day | Set `YOUTUBE_COOKIES` in `.env` — either a **path** to an exported Netscape `cookies.txt` or the **contents** pasted into the var — then restart. `resolve_youtube_cookies()` in main.py handles both forms (pasted content lands in the system temp dir, never `/app/cookies.txt`, which only exists inside the Docker image and silently dropped cookies on self-hosted boxes). Export cookies with a "Get cookies.txt" browser extension or `yt-dlp --cookies-from-browser chrome`. Fastest fallback: download with any downloader and use the Upload zone (the UI says this in place now) |
| `ffmpeg not found` | Not on PATH | Gyan.FFmpeg via winget |
| Groq 413 "Request too large … Limit 8000, Requested 8051" | One request bigger than the entire per-minute TPM budget — waiting, batching and key pools canNOT help (every key has the same ceiling) | Fixed in code: `fit_payload_to_budget` (clip_selection.py) trims each window's text proportionally (setup/payoff kept, middle elided) so prompt+payload fits `TPM − 400 expected-output − 300 headroom`; `_run_stage_split` also splits the batch on a 413 as a backstop. Lower `LLM_SCORE_BATCH` still helps |
| Groq ASR 413 "Request Entity Too Large" on upload | An audio part bigger than the upload cap (24 MB) — the splitter had a 300 s floor that overshot on dense audio (game/music ≈ 12 MB/min → 58 MB parts) | Fixed: `_groq_part_plan` has no floor and targets 90% of the cap; the segment muxer gets a hard `-fs` byte limit; any part that still overshoots is re-split. Dense audio now plans ~111 s ≈ 21.5 MB parts |
| Every clip "failed" right after captions burned (`KeyError: 2`) | The critique-gate layout lookup indexed `layout_ranges[0][2]` while the sidecar rows are dicts — every clip died post-render, so the job reported "no clips" despite writing files | Fixed: `layout_ranges.first_layout()` (shape-safe) is used instead; regression-tested with dict rows, legacy tuples and junk |
| Captions out of sync | Missing word timestamps | Groq `whisper-large-v3` (not turbo) |
| Clips too short/long | Band config | `CLIP_MIN_SECONDS`/`CLIP_MAX_SECONDS` |
| UI changes after `.env` edit | Config read at startup | Restart uvicorn |
| Whole system sluggish / "No space left on device" | C: drive at 100% | Clear `%LOCALAPPDATA%\Temp` and package caches; outputs live on D: |

### Key pool (`GROQ_API_KEYS`) — multiplying the free tier

Groq's free limits (8K tokens/min, the daily request cap) are **per-account**, so N keys give N× the headroom on both AI stages. `groq_pool.py` (thread-safe) hands one key per request; a 429 cools that key for the server's own reset window (65 s floor, `retry-after` wins when longer) and the next request goes to the next key — no backoff sleep on rotation. **Smart rate-limit policy** (both LLM scoring and Groq ASR chunks): ① rotate through every key on 429 with no sleep between rotations; ② only when EVERY key is benched does it sleep — and the length is the server's own token window (from `retry-after`/`x-ratelimit-reset-tokens`, ≈65 s), not a guessed backoff; the ASR path waits one window before falling back to 15-minute local whisper; ③ after bounded cycles it raises `RateLimitError` carrying the real headers so the caller's pacer takes over. The pool covers `llm_backend` scoring **and** `transcribe_backends` Groq ASR. Manage it in the UI: Settings → Key pool has its own **Save keys** button and a live per-key status (ready / cooling with countdown) — it no longer hides behind the collapsed config panel. Caveats: pooling sits in a gray area against free-tier terms (ban risk, not just throttling), a friend revoking a key mid-job fails that key — keep a paid key in `.env` as the fallback of last resort, and the Groq pool never applies to a non-Groq endpoint (OpenRouter/Ollama keep using `LLM_API_KEY`).
| Test suite interrupted at collection | Billing deps missing | Cloud tests need `requirements-billing.txt` (sqlalchemy, alembic, jwt); `tests/conftest.py` skips them when absent — a self-host install needs nothing extra |
| Retry/llm_backend/subtitles tests fail on your box but pass in CI | Your `.env` leaking into the suite | `tests/conftest.py` neutralizes `LLM_BASE_URL`/`LLM_API_KEY`/`CAPTION_MARGIN_V` before imports (same principle as its `BILLING_ENABLED=0` guard); keep it that way — never let tests read the developer's personal `.env` |

## 9. Reliability notes

- **The server is watched.** `watchdog.ps1` (run detached: `powershell -ExecutionPolicy Bypass -File D:\openshorts\watchdog.ps1`) polls `/health` every 15 s, restarts uvicorn within ~15 s of death, and **redirects all server output to `server.out.log` / `server.err.log`** — the crash evidence that was invisible before (uvicorn died twice after job completion with no trace). Every restart is timestamped in `watchdog.log`. Kill-tested: confirmed down → back up in ≤15 s.
- **Clip detection retries once at the job level.** Provider flakes that survive the inner 3-attempt ladder (OpenRouter `:free` "empty response body" bursts) get one more full scoring pass after 60 s, classified by the shared `_is_transient_error()` token list (`main.py`). Non-transient failures (bad key, schema violations) still fail immediately.
- **Transcription failover ends in a readable error.** Groq ASR failing falls back to local Whisper; if that OOMs on a long video the job now says so with the three real options (cloud backend / shorter video / more RAM) instead of dying inside numpy.
- **Known remaining gaps** (design, not bugs): a YouTube bot-check failure still ends the job (the URL is retried through every client strategy; the fix is `YOUTUBE_COOKIES`); single-key Groq pacing adds ~30–60 s waits on long videos (the key pool divides that); the in-memory job table still dies with the process — finished work survives via `/api/jobs` and the Library.

## 10. Security & notes

- `GROQ_API_KEY` / `LLM_API_KEY` live in `.env` — never commit.
- Core is MIT; `cloud/` (billing) is source-available commercial — do not resell as a hosted service.
- No GPU required: Groq ASR+LLM and the layout/critique Gemini calls are all cloud; CPU-only encode at CRF 18 is the main local cost (~5–8 min per 8-min video for local whisper; Groq removes this).
- The web UI needs no build step — it's the static file `static_ui/index.html`.

## 10. Corrections vs. the old handover (keep these straight)

1. `gaming_split_filtergraph` is in **camera_inset.py**, not reframe_v2.py (reframe_v2.py:600 only calls it).
2. **No `/gallery` or `/video/{id}` routes.** UI = `/` → `static_ui/index.html`; static mounts are `/videos` and `/thumbnails`.
3. Portable EXE is **`D:\openshorts_portable\main.exe`**; there is no `D:\openshorts_dist`.
4. Active model is **`openai/gpt-oss-120b`**, not 20b.
5. Default clip band is **15–60 s**, not 20–60 s.
6. Moment selection **never sees frames or audio energy** — it's transcript text + windowing; vision starts at the layout pass.
7. Word-snap padding is **lead 0.35 / tail 0.45 s** (then refine's 0.4/0.5 s breaths), not ±0.3 s.
8. **No auto-editor step exists** — the "silence/jump-cut polish" row was fiction.
9. **No `FFMPEG_ENCODER_PRESET` variable** — presets are hardcoded tiers in ffmpeg_utils.py (medium/fast, CRF 18/22).
10. Groq ASR audio is 16 kHz mono **FLAC** split at 24 MB, not WAV at 25 MB.
11. The QA gate is the **critique gate** (visual re-render trigger), not a check-list of duration/peaks/face-presence.
12. No progress-bar feature found; the hook title card is real (hooks.py).
