"""Rotating API-key pool: GROQ_API_KEYS multiplies a free-tier ceiling by N.

Groq's free limits (8K tokens/min, the daily request cap) are per-ACCOUNT, so
N keys give N× the headroom for BOTH stages that touch Groq: the moment
picker's scoring calls (llm_backend) and the Groq ASR transcription chunks
(transcribe_backends). One request takes one key; a 429 cools that key for the
server's own reset window (plus a small margin) and the next request goes to
the next key, so a single exhausted account no longer stalls the job.

Thread-safe — scoring runs inside main.py's ThreadPoolExecutor. Status output
carries masked key material only (last four), never the full key.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import List, Optional

# Groq's token window is 60 seconds; 65s slightly over-covers it so a key never
# comes back one second too early.
DEFAULT_COOLDOWN = 65.0


def _mask(key: str) -> str:
    return "••••" + key[-4:] if len(key) > 4 else "••••"


def resolve_keys(env=None) -> List[str]:
    """The pool as an ordered, de-duplicated list of real keys.

    GROQ_API_KEYS (or LLM_API_KEYS for a non-Groq OpenAI-compatible endpoint)
    wins when set; otherwise the single classic GROQ_API_KEY / LLM_API_KEY.
    Keys never contain commas or whitespace, so both separate safely.
    """
    env = os.environ if env is None else env
    raw = (env.get("GROQ_API_KEYS") or env.get("LLM_API_KEYS") or "").strip()
    if not raw:
        single = (env.get("GROQ_API_KEY") or env.get("LLM_API_KEY") or "").strip()
        return [single] if single else []
    keys = [k for k in re.split(r"[,\s]+", raw) if k]
    return list(dict.fromkeys(keys))  # de-dupe, keep first-seen order


class KeyPool:
    """Round-robin keys with per-key cooldown after a rate-limit hit."""

    def __init__(self, keys, cooldown: float = DEFAULT_COOLDOWN):
        self._keys = list(dict.fromkeys(k for k in keys if k))
        self._cooldown = {}          # index -> time.monotonic() deadline
        self._cooldown_s = float(cooldown)
        self._lock = threading.Lock()
        self._rr = 0

    def __len__(self) -> int:
        return len(self._keys)

    def pick(self) -> Optional[str]:
        """The next ready key; when ALL are cooling, the one that recovers
        soonest (the caller's pacing waits anyway — failing outright would be
        worse than one more attempt)."""
        now = time.monotonic()
        with self._lock:
            if not self._keys:
                return None
            n = len(self._keys)
            for i in range(n):
                idx = (self._rr + i) % n
                if self._cooldown.get(idx, 0.0) <= now:
                    self._rr = (idx + 1) % n
                    return self._keys[idx]
            idx = min(range(n), key=lambda i: self._cooldown.get(i, 0.0))
            self._rr = (idx + 1) % n
            return self._keys[idx]

    def penalize(self, key: str, seconds: Optional[float] = None) -> None:
        """Cool ``key`` so subsequent picks skip it. ``seconds`` (the server's
        own retry-after/reset) wins when longer than the default window."""
        with self._lock:
            secs = max(self._cooldown_s, float(seconds or 0.0))
            for i, k in enumerate(self._keys):
                if k == key:
                    self._cooldown[i] = time.monotonic() + secs

    def soonest_recovery(self) -> Optional[float]:
        """Seconds until the next key leaves cooldown; ``None`` when all ready.

        This is the number a caller waits before retrying: the provider's own
        60-second token window, measured per key."""
        now = time.monotonic()
        with self._lock:
            deadlines = [d for d in self._cooldown.values() if d > now]
            if not deadlines:
                return None
            return max(0.0, min(deadlines) - now)

    def status(self) -> List[dict]:
        """Per-key readiness for the UI. Masked — never the full key."""
        now = time.monotonic()
        with self._lock:
            out = []
            for i, k in enumerate(self._keys):
                until = self._cooldown.get(i, 0.0)
                cooling = until > now
                out.append({
                    "masked": _mask(k),
                    "state": "cooling" if cooling else "ready",
                    "cooling_for": round(until - now, 1) if cooling else 0.0,
                })
            return out


# One process-wide pool, rebuilt when the key env changes: the provider-config
# endpoint applies env live, so a new key list must take effect without a
# server restart.
_pool: Optional[KeyPool] = None
_pool_src = None
_pool_lock = threading.Lock()


def _reset_for_tests():
    global _pool, _pool_src
    with _pool_lock:
        _pool, _pool_src = None, None


def get_pool() -> KeyPool:
    global _pool, _pool_src
    src = tuple((os.environ.get(v) or "") for v in
                ("GROQ_API_KEYS", "GROQ_API_KEY", "LLM_API_KEYS", "LLM_API_KEY"))
    with _pool_lock:
        if _pool is None or _pool_src != src:
            _pool = KeyPool(resolve_keys())
            _pool_src = src
        return _pool


def pool_status() -> List[dict]:
    return get_pool().status()


def ready_count() -> int:
    return sum(1 for s in pool_status() if s["state"] == "ready")
