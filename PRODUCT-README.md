# OpenShorts — Vertical Shorts Generator

Turn one long video into ready-to-post vertical shorts: paste a YouTube URL or
drop a file, and the pipeline finds the best moments, reframes them to 9:16,
burns in word-by-word karaoke captions, scores them, and serves them from a
local web UI.

## Run it (one command)

```bat
START.bat
```

Opens <http://localhost:8001>. `STOP.bat` shuts everything down; `STATUS.bat`
prints health, processes, restart history, and the last server errors.

What `START.bat` does: launches `watchdog.ps1`, which starts the server
(`uvicorn app:app`, port 8001), polls `/health` every 15 s, and restarts the
server within ~15 s if it ever dies. Server output is captured in
`server.out.log` / `server.err.log`; every restart is timestamped in
`watchdog.log`. Kill-tested: the UI survives crashes on its own.

## What you need

| Component | Where it comes from | Notes |
|---|---|---|
| Python 3.11 + the repo venv | `D:\openshorts\.venv311` | required for the web UI and CLI |
| ffmpeg on PATH | WinGet `Gyan.FFmpeg` | all rendering/cutting |
| Groq API key (`.env`) | console.groq.com (free) | transcription **and** clip scoring |
| Gemini API key (`.env`, optional) | Google AI Studio (free) | layout/facecam/critique — those stages degrade gracefully without it |
| YouTube cookies (optional) | browser export | fixes the "Sign in to confirm you're not a bot" wall; see HANDOVER.md |

## Use it

1. Open the UI, paste a YouTube URL or drop a video file (45 s minimum).
2. Pick a format (Vertical 9:16 recommended) and how many clips you want.
3. Generate. Watch the live log; finished clips appear in the **Library**
   with previews, per-clip downloads, and a ZIP button. History survives
   server restarts — it is rebuilt from disk.
4. Settings → **Key pool**: add several Groq keys (comma-separated) to
   multiply the free-tier budget — the pipeline rotates keys on rate limits
   and only waits for the token window when every key is exhausted.

## CLI (no server)

```bat
D:\openshorts\.venv311\Scripts\python.exe main.py -u "https://youtube.com/watch?v=ID" -o D:\out
D:\openshorts\.venv311\Scripts\python.exe main.py -i C:\video.mp4 -o D:\out --format vertical
```

## Portable (no Python needed)

`portable\main.exe` is a self-contained PyInstaller build — copy the folder to
any Windows box with ffmpeg and run `main.exe -u <url> -o <out>`. (It carries
the same pipeline as of the build date.)

## Ops commands

| Task | Command |
|---|---|
| Start | `START.bat` |
| Stop | `STOP.bat` |
| Health + last errors | `STATUS.bat` |
| Run the test suite | `D:\openshorts\.venv311\Scripts\python.exe -m pytest tests/ -q` |
| Point the pack at a different app dir | `set OPENSHORTS_APP_DIR=...` before `START.bat` |

## Scale

This machine runs one worker: 3 clips from a ~25-minute video takes ~15–30
minutes, dominated by CPU rendering. The scaling path (documented, not yet
built) is Docker + a shared job queue + object storage — the app already has
the two-instance handshake (heartbeat manifests, resume) that needs. Free-tier
API quotas, not the code, are the first ceiling you hit.

## Docs

- `HANDOVER.md` — full technical handover (architecture, env vars, every bug
  we fixed and why, troubleshooting)
- `.env.example` — every configuration variable
