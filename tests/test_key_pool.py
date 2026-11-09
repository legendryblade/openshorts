"""The rotating key pool (groq_pool.py): N keys buy N× the free-tier ceiling.

Groq's limits are per-account, so a pool of 10 friends' keys multiplies both
the scoring and the transcription budget — provided a 429 on one key rotates
to the next *without* burning backoff time, and a benched key stays benched
until the server's own reset window passes.
"""
import json

import httpx
import pytest

import gemini_worker
import groq_pool
import llm_backend


@pytest.fixture(autouse=True)
def _fresh_pool():
    groq_pool._reset_for_tests()
    yield
    groq_pool._reset_for_tests()


class _Clock:
    """Deterministic time.monotonic for cooldown tests."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, secs):
        self.now += secs


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(groq_pool.time, "monotonic", c)
    return c


# --- resolution -----------------------------------------------------------

class TestKeyResolution:
    def test_pool_env_splits_and_dedupes(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEYS", "keyA, keyA ,keyB\tkeyC\nkeyB")
        monkeypatch.setenv("GROQ_API_KEY", "legacy")
        assert groq_pool.resolve_keys() == ["keyA", "keyB", "keyC"]

    def test_llm_keys_are_a_valid_pool_source(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEYS", raising=False)
        monkeypatch.setenv("LLM_API_KEYS", "a,b")
        assert groq_pool.resolve_keys() == ["a", "b"]

    def test_single_key_fallback(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEYS", raising=False)
        monkeypatch.delenv("LLM_API_KEYS", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "solo")
        assert groq_pool.resolve_keys() == ["solo"]

    def test_no_keys_anywhere(self, monkeypatch):
        for v in ("GROQ_API_KEYS", "GROQ_API_KEY", "LLM_API_KEYS", "LLM_API_KEY"):
            monkeypatch.delenv(v, raising=False)
        assert groq_pool.resolve_keys() == []


# --- rotation ---------------------------------------------------------------

class TestRotation:
    def test_round_robin_covers_every_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEYS", "a,b,c")
        pool = groq_pool.get_pool()
        assert [pool.pick() for _ in range(4)] == ["a", "b", "c", "a"]

    def test_penalized_key_is_skipped_until_cooldown(self, monkeypatch, clock):
        monkeypatch.setenv("GROQ_API_KEYS", "a,b,c")
        pool = groq_pool.get_pool()
        pool.pick()                       # rr -> b
        pool.penalize("b", seconds=65)    # b benched until +65s
        assert [pool.pick() for _ in range(3)] == ["c", "a", "c"]
        clock.advance(64)                 # one second before recovery
        assert pool.pick() != "b"
        clock.advance(2)                  # past the window
        assert "b" in [pool.pick() for _ in range(3)]

    def test_short_retry_after_still_uses_the_full_window(self, monkeypatch, clock):
        monkeypatch.setenv("GROQ_API_KEYS", "a,b")
        pool = groq_pool.get_pool()
        pool.pick()
        pool.penalize("a", seconds=1)     # server asked 1s; we keep 65s floor
        clock.advance(2)
        assert pool.pick() != "a"

    def test_all_keys_cooling_returns_soonest_not_none(self, monkeypatch, clock):
        monkeypatch.setenv("GROQ_API_KEYS", "a,b")
        pool = groq_pool.get_pool()
        pool.penalize("a", seconds=70)    # recovers at +70
        clock.advance(1)
        pool.penalize("b", seconds=68)    # recovers at +69 — the sooner one
        assert pool.pick() == "b"

    def test_status_masks_keys_and_reports_state(self, monkeypatch, clock):
        monkeypatch.setenv("GROQ_API_KEYS", "gsk_secret_keyabcd,gsk_secret_keywxyz")
        pool = groq_pool.get_pool()
        pool.penalize("gsk_secret_keyabcd")
        status = pool.status()
        assert len(status) == 2
        assert [s["masked"] for s in status] == ["••••abcd", "••••wxyz"]
        assert status[0]["state"] == "cooling" and status[0]["cooling_for"] > 0
        assert status[1]["state"] == "ready" and status[1]["cooling_for"] == 0
        assert "secret" not in json.dumps(status)

    def test_pool_rebuilds_when_env_changes(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEYS", "a,b")
        assert len(groq_pool.get_pool()) == 2
        monkeypatch.setenv("GROQ_API_KEYS", "a,b,c")
        assert len(groq_pool.get_pool()) == 3


# --- llm_backend integration ----------------------------------------------

def _local(monkeypatch, **keys):
    # A groq.com endpoint: the pool guard only rotates keys against Groq's
    # own servers (an OpenRouter URL must keep using LLM_API_KEY).
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    for v in ("LLM_PROVIDER", "GROQ_API_KEYS", "GROQ_API_KEY",
              "LLM_API_KEYS", "LLM_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    for k, v in keys.items():
        monkeypatch.setenv(k, v)


def _serve(handler, monkeypatch):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40},
    })


class _Payload:
    @classmethod
    def model_validate(cls, parsed):
        return cls()

    def model_dump(self):
        return {"ok": True}


def _schema():
    class Resp:
        model_json_schema = staticmethod(lambda: {"type": "object"})
        model_validate = staticmethod(lambda parsed: _Payload())
        __name__ = "resp"
    return Resp


class TestBackendRotation:
    def test_single_key_unchanged(self, monkeypatch):
        _local(monkeypatch, GROQ_API_KEY="onlykey")
        seen = []
        def handler(request):
            seen.append(request.headers["authorization"])
            return _completion({"ok": True})
        _serve(handler, monkeypatch)
        monkeypatch.setattr(gemini_worker, "_parse_json_response_text",
                            lambda t: {"ok": True}, raising=False)
        llm_backend.generate_json("prompt", _schema())
        assert seen == ["Bearer onlykey"]

    def test_429_rotates_to_next_key_without_sleeping(self, monkeypatch):
        _local(monkeypatch, GROQ_API_KEYS="keyA,keyB")
        seen = []
        def handler(request):
            seen.append(request.headers["authorization"])
            if len(seen) == 1:
                return httpx.Response(429, json={"error": "tpm"},
                                     headers={"x-ratelimit-reset-tokens": "22.357s"})
            return _completion({"ok": True})
        _serve(handler, monkeypatch)
        monkeypatch.setattr(gemini_worker, "_parse_json_response_text",
                            lambda t: {"ok": True}, raising=False)
        parsed, _cost = llm_backend.generate_json("prompt", _schema())
        assert parsed == {"ok": True}
        assert seen == ["Bearer keyA", "Bearer keyB"], "first key benched, second used"
        # the benched key stays benched for the server's own window
        assert groq_pool.get_pool().status()[0]["state"] == "cooling"

    def test_all_keys_rate_limited_raises_after_trying_each(self, monkeypatch):
        _local(monkeypatch, GROQ_API_KEYS="keyA,keyB,keyC")
        monkeypatch.setattr(llm_backend.time, "sleep", lambda s: None)
        seen = []
        def handler(request):
            seen.append(request.headers["authorization"])
            return httpx.Response(429, json={"error": "tpm"})
        _serve(handler, monkeypatch)
        with pytest.raises(llm_backend.RateLimitError):
            llm_backend.generate_json("prompt", _schema())
        assert sorted(set(seen)) == ["Bearer keyA", "Bearer keyB", "Bearer keyC"]

    def test_groq_pool_never_leaks_to_another_endpoint(self, monkeypatch):
        # OpenRouter/Ollama pickers keep using LLM_API_KEY: a rotating Groq
        # key against someone else's server is a hard failure, not a retry.
        _local(monkeypatch, GROQ_API_KEYS="keyA,keyB", LLM_API_KEY="or-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        seen = []
        def handler(request):
            seen.append(request.headers["authorization"])
            return _completion({"ok": True})
        _serve(handler, monkeypatch)
        monkeypatch.setattr(gemini_worker, "_parse_json_response_text",
                            lambda t: {"ok": True}, raising=False)
        llm_backend.generate_json("prompt", _schema())
        assert seen == ["Bearer or-key"]


# --- rotate, then wait for the window --------------------------------------

class TestSoonestRecovery:
    def test_none_when_all_keys_ready(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEYS", "a,b")
        pool = groq_pool.get_pool()
        assert pool.soonest_recovery() is None

    def test_seconds_until_the_soonest_key_recovers(self, monkeypatch, clock):
        monkeypatch.setenv("GROQ_API_KEYS", "a,b")
        pool = groq_pool.get_pool()
        pool.penalize("a", seconds=70)   # recovers at +70
        clock.advance(5)
        pool.penalize("b", seconds=68)   # recovers at +73 — a is still sooner
        assert pool.soonest_recovery() == pytest.approx(65.0, abs=0.5)


class TestSendRotateThenWait:
    def _429(self, reset="22.357s"):
        return httpx.Response(429, json={"error": "tpm"},
                              headers={"x-ratelimit-reset-tokens": reset})

    def test_rotates_first_then_waits_for_the_window(self, monkeypatch):
        # Rotation is free (no sleep); the wait only fires when EVERY key is
        # benched, and its length comes from the pool's cooldown, not a guess.
        _local(monkeypatch, GROQ_API_KEYS="keyA,keyB")
        sleeps = []
        monkeypatch.setattr(llm_backend.time, "sleep", lambda s: sleeps.append(s))
        seen = []
        def handler(request):
            seen.append(request.headers["authorization"])
            if len(seen) <= 2:
                return self._429()
            return _completion({"ok": True})
        _serve(handler, monkeypatch)
        monkeypatch.setattr(gemini_worker, "_parse_json_response_text",
                            lambda t: {"ok": True}, raising=False)
        parsed, _cost = llm_backend.generate_json("prompt", _schema())
        assert parsed == {"ok": True}
        assert seen == ["Bearer keyA", "Bearer keyB", "Bearer keyA"], \
            "both keys tried before any waiting"
        assert len(sleeps) == 1 and 60 <= sleeps[0] <= 70, \
            f"one wait covering the token window, got {sleeps}"

    def test_all_keys_blocked_raises_after_bounded_waits(self, monkeypatch):
        _local(monkeypatch, GROQ_API_KEYS="keyA,keyB")
        sleeps = []
        monkeypatch.setattr(llm_backend.time, "sleep", lambda s: sleeps.append(s))
        def handler(request):
            return self._429()
        _serve(handler, monkeypatch)
        with pytest.raises(llm_backend.RateLimitError):
            llm_backend.generate_json("prompt", _schema())
        # Bounded: a few window waits, never an unbounded block.
        assert 1 <= len(sleeps) <= 3, sleeps

    def test_single_key_also_waits_before_giving_up(self, monkeypatch):
        _local(monkeypatch, GROQ_API_KEY="solo")
        sleeps = []
        monkeypatch.setattr(llm_backend.time, "sleep", lambda s: sleeps.append(s))
        seen = []
        def handler(request):
            seen.append(request.headers["authorization"])
            return self._429()
        _serve(handler, monkeypatch)
        with pytest.raises(llm_backend.RateLimitError):
            llm_backend.generate_json("prompt", _schema())
        assert seen == ["Bearer solo"] * len(seen)


class TestAsrWaitsInsteadOfWhisperFallback:
    """The ASR chunk path must wait out a fully-benched pool once rather than
    dropping to 15-minute local whisper for the whole video."""

    def test_all_keys_benched_waits_once_then_succeeds(self, tmp_path, monkeypatch):
        import transcribe_backends
        monkeypatch.setenv("GROQ_API_KEYS", "keyA,keyB")
        monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        chunk = tmp_path / "a.flac"
        chunk.write_bytes(b"\x00")
        calls = []
        sleeps = []
        monkeypatch.setattr(transcribe_backends.time, "sleep",
                            lambda s: sleeps.append(s))
        def fake_post(url, **kw):
            calls.append(kw["headers"]["Authorization"])
            if len(calls) <= 2:
                return httpx.Response(429, json={"error": "tpm"},
                                      headers={"x-ratelimit-reset-tokens": "60s"})
            return httpx.Response(200, json={"text": "ok", "segments": []})
        # _groq_transcribe_file imports httpx at function scope, so the
        # global module's post is the thing to swap.
        import httpx as real_httpx
        monkeypatch.setattr(real_httpx, "post", fake_post)
        payload = transcribe_backends._groq_transcribe_file(str(chunk))
        assert payload["text"] == "ok"
        assert calls[:2] == ["Bearer keyA", "Bearer keyB"], "both keys tried first"
        assert len(sleeps) == 1 and sleeps[0] > 60, "one window wait, not a fallback"


class TestGroqPartPlan:
    """The ASR upload splitter: a 300s floor sent 58 MB parts on dense audio
    and every chunk 413'd, killing cloud ASR for the whole video."""

    def test_dense_audio_gets_short_parts_that_fit_the_cap(self, monkeypatch):
        from transcribe_backends import _groq_part_plan
        CAP = 24 * 1024 * 1024
        # 291 MB of game/music audio over 25 min ≈ 12 MB/min
        part = _groq_part_plan(291 * 1024 * 1024, 1500, CAP)
        assert part is not None
        per_sec = 291 * 1024 * 1024 / 1500
        assert part * per_sec < CAP, "planned part must fit the upload cap"
        assert part * per_sec > CAP * 0.5, "and not waste the cap either"

    def test_no_floor_forces_oversized_parts(self, monkeypatch):
        from transcribe_backends import _groq_part_plan
        CAP = 24 * 1024 * 1024
        part = _groq_part_plan(291 * 1024 * 1024, 1500, CAP)
        assert part < 300, "the old 300s floor is exactly what 413'd"

    def test_small_file_is_one_upload(self, monkeypatch):
        from transcribe_backends import _groq_part_plan
        assert _groq_part_plan(5 * 1024 * 1024, 300, 24 * 1024 * 1024) is None


class TestReliabilityGuards:
    """The guards added after the reliability audit: job-level transient retry
    classification, and a readable error instead of a numpy OOM crash."""

    def test_transient_classifier_catches_free_tier_empty_bodies(self, monkeypatch):
        import main as _m
        err = RuntimeError("Gemini returned an empty response body")
        assert _m._is_transient_error(err, str(err))

    def test_non_transient_errors_are_not_retried(self, monkeypatch):
        import main as _m
        # A prose-instead-of-JSON answer IS retried (the 0348221 fix); a hard
        # rejection like a bad key is not.
        err = ValueError("401 unauthorized: invalid api key")
        assert not _m._is_transient_error(err, str(err))
        assert _m._is_transient_error(
            RuntimeError("did not contain a JSON object: sure! here you go"))


class TestAsrTmpdir:
    """The Groq ASR path extracts a large FLAC BEFORE calling the API; when
    that write lands on a full OS drive the job dies with ENOSCP (-28) and
    never reaches cloud Whisper (the 10,000-players video, 1-nov-2026)."""

    def test_explicit_env_wins(self, tmp_path, monkeypatch):
        import transcribe_backends as tb
        monkeypatch.setenv("ASR_TMPDIR", str(tmp_path))
        assert tb._asr_tmpdir() == str(tmp_path)

    def test_picks_the_drive_with_room(self, monkeypatch):
        import transcribe_backends as tb
        import shutil
        monkeypatch.delenv("ASR_TMPDIR", raising=False)
        # fake a nearly-full C: and a roomy E:
        real = shutil.disk_usage
        def fake(path):
            gb = {"E:": 200.0, "D:": 1.0, "C:": 0.2}.get(str(path)[:2], 3.0)
            return type("u", (), {"free": int(gb * 1024**3)})()
        # The system-temp candidate must look FULL too, or it wins as the
        # first >=2GB candidate before the fixed drives are considered.
        def _fake_free(p):
            sp = str(p).replace("\\", "/").upper()
            if sp.startswith("C:/") or sp.startswith("C:\\"):
                return 0.2           # the full OS drive (the bug this guards)
            return {"E": 200.0, "D": 1.0}.get(sp[:2], 3.0)
        monkeypatch.setattr(tb, "_free_gb", _fake_free)
        monkeypatch.setattr(tb.os, "makedirs", lambda *a, **k: None)
        monkeypatch.setattr(tb.os, "access", lambda p, m: True)
        monkeypatch.setattr(tb.os.path, "isdir", lambda p: True)
        monkeypatch.setattr(tb.os.path, "isfile", lambda p: False)
        # ASR_TMPDIR unset -> system temp candidate first (patch it to look full)
        monkeypatch.setattr(tb.tempfile, "gettempdir", lambda: "C:/dummy")
        assert tb._asr_tmpdir().startswith("E:")


class TestAsrResplitSafety:
    """The resplit used to delete its own sub_ files mid-collection (they were
    still in the return value) and cut with -fs, which truncates mid-frame and
    corrupts the part (FileNotFoundError + unreadable sub_ files, 2-nov)."""

    def test_plan_targets_under_the_cap(self):
        from transcribe_backends import _groq_part_plan
        CAP = 24 * 1024 * 1024
        part = _groq_part_plan(291 * 1024 * 1024, 1500, CAP)
        assert part * (291 * 1024 * 1024 / 1500) < CAP * 0.95

    def test_no_delete_of_collected_parts(self):
        # The old bug: os.remove(sub_path) inside the collection loop.
        import inspect
        from transcribe_backends import _fit_under_cap, _split_audio_for_groq
        src = inspect.getsource(_split_audio_for_groq) + inspect.getsource(_fit_under_cap)
        assert "os.remove(sub_path)" not in src, "must not delete files it returns"
        assert '"-fs"' not in src, "byte-cap segmenting corrupts FLAC frames"

    def test_offsets_are_read_from_the_part_not_accumulated(self):
        # The 2-nov-2026 collapse: segment-muxer FLAC parts have no duration in
        # their header (ffprobe answers N/A -> 0.0), so `offset +=
        # _probe_duration(path)` left every chunk at 0.0 and a 4-hour video was
        # transcribed as its first 9 minutes. Offsets must come from the part.
        import inspect
        from transcribe_backends import _split_audio_for_groq
        src = inspect.getsource(_split_audio_for_groq)
        assert "_probe_pts_range(" in src, "each part's offset must be measured"
        assert "offset += _probe_duration" not in src, "accumulated offsets drift silently"

    def test_sub_prefix_is_unique_per_part(self):
        # The 2-nov-2026 "captions don't match / cuts mid-sentence" bug: a
        # shared sub_%05d prefix let each ffmpeg invocation overwrite the
        # previous part's sub files, so two oversized parts collected the
        # same files and every later chunk's offset drifted.
        import inspect
        from transcribe_backends import _fit_under_cap
        src = inspect.getsource(_fit_under_cap)
        assert "prefix = f\"sub_{tag}_{depth}_\"" in src, "each part needs its own sub-file prefix"
        assert 'n.startswith(prefix)' in src, "collection must match only this part's prefix"
        assert '"sub_%05d.flac"' not in src, "the shared prefix is the bug"

    def test_oversized_sub_parts_are_dropped_not_deleted(self):
        # Dropping keeps the part list consistent (offsets from the original
        # plan stay correct) instead of raising FileNotFoundError later.
        from transcribe_backends import _groq_part_plan
        assert _groq_part_plan(10 * 1024 * 1024, 300, 24 * 1024 * 1024) is None
