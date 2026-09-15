"""Verify the clip-picking rate-limit fix: TPM pacing + real 429 handling.

Run from the repo root:  .venv311/Scripts/python verify_llm_rate_limit.py

Checks, in order:
  1. Go-duration / Retry-After parsing against the values Groq really sent
     (measured 15-sep-2026: x-ratelimit-reset-tokens: 22.357s / 1m26.4s / 2m52.8s).
  2. A per-minute 429 is NOT mistaken for a daily quota, and Gemini's real
     day-quota message IS — the two need opposite policies.
  3. The pacer holds the 8,000 TPM budget across a whole 34-window video, and
     shows what the old unpaced behaviour would have spent.
  4. A 429 from the transport arrives as llm_backend.RateLimitError carrying the
     server's own wait, and a success carries the advertised budget.

Exit code is 0 only when every check passes.
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
# main.py loads YOLO("yolov8n.pt") by a RELATIVE path at import time, and
# ultralytics downloads missing weights into the CURRENT directory. Run this
# harness from anywhere else and the download lands there (a 30s-capped shell
# killed one mid-write on 16-sep-2026, leaving a truncated .pt that broke every
# later import). Pinning the CWD makes the harness independent of where — and
# by which shell — it was launched.
os.chdir(REPO)

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"   ✅ {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"   ❌ {label}" + (f" — {detail}" if detail else ""))
        FAILURES.append(label)


class _Clock:
    """Fake monotonic clock; sleeping advances it instead of the process."""

    def __init__(self):
        self.now = 0.0
        self.slept = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.slept += seconds


def verify_duration_parsing():
    """Groq's reset windows are Go durations; they gate how long a 429 lasts."""
    print("\n🧪 Go durations / Retry-After")
    from clip_selection import parse_go_duration, parse_retry_after

    # The exact strings Groq returned on 15-sep-2026.
    for raw, expected in (("22.357s", 22.357), ("1m26.4s", 86.4), ("2m52.8s", 172.8),
                          ("500ms", 0.5), ("1h2m3s", 3723.0), ("30", 30.0)):
        got = parse_go_duration(raw)
        check(f"{raw!r} -> {expected}s", got == expected, f"got {got}")
    for raw in (None, "", "soon"):
        check(f"{raw!r} -> None", parse_go_duration(raw) is None,
              f"got {parse_go_duration(raw)}")

    check("retry-after '37' -> 37.0s", parse_retry_after("37") == 37.0)
    # A parsed HTTP-date is measured against the supplied clock, not the wall.
    # 2026-10-21 is a Wednesday, and the epoch comes from datetime so the test
    # cannot drift the way a hand-written constant does.
    import datetime as _dt

    target = _dt.datetime(2026, 10, 21, 7, 28, 0, tzinfo=_dt.timezone.utc).timestamp()
    check("retry-after date -> 120.0s",
          parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT", now=target - 120) == 120.0,
          f"got {parse_retry_after('Wed, 21 Oct 2026 07:28:00 GMT', now=target - 120)}")
    check("retry-after garbage -> None", parse_retry_after("nope") is None)


def verify_daily_quota_detection():
    """A per-minute 429 is waitable; a per-day one is not."""
    print("\n Daily vs per-minute quota")
    from clip_selection import is_daily_quota_error

    gemini_daily = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
                    "generate_requests_per_model_per_day, limit: 20")
    groq_tpm = ("LLM server 429: Rate limit reached for openai/gpt-oss-120b in "
                "organization org_x on requests per minute (TPM): Limit 8000")
    check("Gemini RPD -> daily", is_daily_quota_error(gemini_daily))
    check("Groq TPM -> not daily", not is_daily_quota_error(groq_tpm))
    check("empty -> not daily", not is_daily_quota_error(""))


def verify_pacer_budget():
    """The pacer must hold the budget across a whole video's scoring calls."""
    print("\n TokenPacer against the real 8,000 TPM budget")
    from clip_selection import TokenPacer

    # Same shape as the measured call: ~697 prompt + 165 completion tokens.
    estimate, actual = 700 + 400, 862
    batches = 12  # 34 windows at LLM_SCORE_BATCH=3
    clock = _Clock()
    pacer = TokenPacer(limit_tokens=8000, window_seconds=60.0,
                       clock=clock.time, sleeper=clock.sleep)

    over_budget = []
    waits = []
    for _ in range(batches):
        before = pacer.spent()
        waited = pacer.reserve(estimate)
        waits.append(waited)
        if pacer.spent() > pacer.limit_tokens:
            over_budget.append((before, waited, pacer.spent()))
        clock.now += 0.4  # the call itself returns in ~0.4s
        pacer.settle(estimate, actual)

    check("no call ever pushed the window over 8,000", not over_budget,
          f"{over_budget}" if over_budget else f"spent now {pacer.spent()}")
    check("early calls need no pacing", waits[0] == 0 and waits[1] == 0,
          f"first two waits {waits[0]}s / {waits[1]}s")
    check("later calls wait for the window to refill", max(waits) > 0,
          f"longest wait {max(waits):.1f}s")
    check("whole video stays under the budget per minute",
          clock.slept < 180, f"total paced wait {clock.slept:.1f}s")

    # What the pipeline did BEFORE the fix: nothing paced it, and 12 back-to-back
    # calls spend 10,344 tokens inside one 60s window against an 8,000 budget —
    # the 10th+ calls are exactly the 429s that killed clip selection.
    unpaced = TokenPacer(limit_tokens=0, window_seconds=60.0,
                         clock=clock.time, sleeper=clock.sleep)
    for _ in range(batches):
        unpaced.reserve(estimate)
        unpaced.settle(estimate, actual)
    print(f"      (unpaced, the old behaviour: {unpaced.spent():,} tokens in one "
          f"60s window vs an 8,000 budget = {unpaced.spent() / 8000:.0%} over)")
    check("unpaced run reproduces the old overspend",
          unpaced.spent() > 8000 and unpaced.limit_tokens == 0)


def verify_rate_limit_error_surfaces_wait():
    """A 429 must arrive as RateLimitError carrying the server's own wait.

    Uses httpx.MockTransport — the hook ``llm_backend._client`` documents for
    exactly this — so no quota is spent proving the policy.
    """
    print("\n 429 -> RateLimitError, and headers learned on success")
    import httpx

    import gemini_worker
    import llm_backend

    os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"
    os.environ["LLM_API_KEY"] = "test-key-not-used"

    def client_with(responses):
        queue = list(responses)

        def handler(request):
            return queue.pop(0)

        transport = httpx.MockTransport(handler)
        return lambda **kwargs: httpx.Client(transport=transport, **kwargs)

    original = llm_backend._client
    try:
        llm_backend._client = client_with([
            httpx.Response(429, headers={
                "retry-after": "37",
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "22.357s",
            }, json={"error": {"message": "Rate limit reached (TPM)"}}),
        ])
        try:
            llm_backend.generate_json("score these windows",
                                      gemini_worker.ScoreResponse,
                                      model="openai/gpt-oss-120b")
            check("429 raises RateLimitError", False, "no exception raised")
        except llm_backend.RateLimitError as e:
            check("429 raises RateLimitError", True, str(e)[:70])
            check("the server's Retry-After wins (37s)", e.retry_after == 37.0,
                  f"got {e.retry_after}")
            check("429 is detectable by status", e.status_code == 429)
            check("the budget travels with the error",
                  e.rate_limit.get("limit_tokens") == 8000, f"{e.rate_limit}")
            check("text still contains 429 for text-matching callers", "429" in str(e))
        except Exception as e:  # noqa: BLE001 - the whole point is to classify it
            check("429 raises RateLimitError", False, f"{type(e).__name__}: {e}")

        llm_backend._client = client_with([
            httpx.Response(200, headers={"x-ratelimit-limit-tokens": "8000",
                                         "x-ratelimit-remaining-tokens": "4523"},
                           json={"choices": [{"message": {"content": '{"windows": []}'}}],
                                 "usage": {"prompt_tokens": 528, "completion_tokens": 194}}),
        ])
        _parsed, cost = llm_backend.generate_json("score these windows",
                                                 gemini_worker.ScoreResponse,
                                                 model="openai/gpt-oss-120b")
        check("a success reports what it billed",
              cost["input_tokens"] == 528 and cost["output_tokens"] == 194)
        check("a success advertises the budget the pacer adopts",
              (cost.get("rate_limit") or {}).get("limit_tokens") == 8000,
              f"{cost.get('rate_limit')}")
    finally:
        llm_backend._client = original


def verify_repo_wiring():
    """The scoring stage must actually consult the new knobs.

    Asserted against the source rather than by importing main.py: that import
    pulls torch and ultralytics, a needless minute for a check whose failure
    mode is a missing call site. ``main.py --help`` covers the CLI side.
    """
    print("\n Wiring in main.py")
    with open(os.path.join(REPO, "main.py"), encoding="utf-8") as f:
        source = f.read()
    for needle, label in (
        ("_PACER.reserve(reserved)", "calls are paced before they are sent"),
        ("_PACER.settle(reserved, _billed_tokens(cost))", "the real bill is settled back"),
        ('_PACER.set_limit(advertised["limit_tokens"])', "the budget is learned from headers"),
        ("if is_daily_quota_error(msg):", "a day quota fails fast"),
        ("if _is_rate_limit_error(e, msg):", "429s take the rate-limit path"),
        ("workers = max(1, min(score_concurrency(), len(batches)))",
         "scoring honours the concurrency knob"),
        ('os.environ["CLIP_SCORE_CONCURRENCY"] = str(max(1, args.max_concurrent))',
         "--max-concurrent reaches the stage"),
        ('LLM_RATE_LIMIT_ATTEMPTS', "the rate-limit attempt budget is configurable"),
    ):
        check(label, needle in source)


def verify_stage_integration():
    """Import main and run the REAL stage against a mocked server.

    This is the check a source-level scan cannot do: the pacing and retry rewrite
    lives inside ``_run_gemini_stage``, and an indentation slip there is
    invisible until something actually calls it (15-sep-2026: a mis-indented
    cost line made the OpenAI-compatible branch reference an unbound local, and
    every check above still passed). Importing main.py pulls torch and
    ultralytics, so it runs last and is skippable with --fast.
    """
    print("\n Integration: main._run_gemini_stage against a mocked server")
    os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"
    os.environ["LLM_API_KEY"] = "test-key-not-used"
    os.environ["LLM_PROVIDER"] = "openai"

    try:
        import httpx

        import gemini_worker
        import llm_backend
        import main
    except Exception as e:  # noqa: BLE001 - report, do not abort the harness
        check("main imports", False, f"{type(e).__name__}: {e}")
        return

    replies = [
        # A 429 that names its own (short) wait, then a good answer.
        httpx.Response(429, headers={"retry-after": "1",
                                     "x-ratelimit-limit-tokens": "8000"},
                       json={"error": {"message": "Rate limit reached (TPM)"}}),
        httpx.Response(200, headers={"x-ratelimit-limit-tokens": "8000",
                                     "x-ratelimit-remaining-tokens": "4000"},
                       json={"choices": [{"message": {"content": '{"windows": []}'}}],
                             "usage": {"prompt_tokens": 697, "completion_tokens": 165}}),
    ]
    queue = list(replies)
    transport = httpx.MockTransport(lambda request: queue.pop(0))
    original_client = llm_backend._client
    original_sleep = main.time.sleep
    llm_backend._client = lambda **kwargs: httpx.Client(transport=transport, **kwargs)
    slept = []
    main.time.sleep = lambda seconds: slept.append(seconds)  # never really wait
    try:
        parsed, cost = main._run_gemini_stage(None, "openai/gpt-oss-120b",
                                              "score these windows",
                                              gemini_worker.ScoreResponse)
    finally:
        main.time.sleep = original_sleep
        llm_backend._client = original_client

    check("the OpenAI-compatible path runs end to end", parsed == {"windows": []},
          f"parsed={parsed}")
    check("a 429 is retried after the wait the server asked for", slept == [1.0],
          f"slept={slept}")
    check("the retry's answer is billed", cost["input_tokens"] == 697)
    check("the stage adopts the budget the server advertised",
          main._PACER.limit_tokens == 8000, f"limit={main._PACER.limit_tokens}")

    # A transient answer (a score the schema refuses) must retry AND survive the
    # settle afterwards: the rewrite's first cut fell through the retry branch
    # into the success path, where `cost` was still unbound — that killed a real
    # run on 16-sep-2026 right after a clean 429 recovery proved the other path.
    queue2 = [
        httpx.Response(200, headers={"x-ratelimit-limit-tokens": "8000"},
                       json={"choices": [{"message": {"content":
                            '{"windows": [{"id": 0, "start": 0, "end": 90, "score": "high"}]}'}}],
                            "usage": {"prompt_tokens": 700, "completion_tokens": 200}}),
        httpx.Response(200, headers={"x-ratelimit-limit-tokens": "8000"},
                       json={"choices": [{"message": {"content": '{"windows": []}'}}],
                             "usage": {"prompt_tokens": 701, "completion_tokens": 201}}),
    ]
    transport2 = httpx.MockTransport(lambda request: queue2.pop(0))
    llm_backend._client = lambda **kwargs: httpx.Client(transport=transport2, **kwargs)
    slept2 = []
    main.time.sleep = lambda seconds: slept2.append(seconds)
    try:
        parsed2, cost2 = main._run_gemini_stage(None, "openai/gpt-oss-120b",
                                                "score these windows",
                                                gemini_worker.ScoreResponse)
    finally:
        main.time.sleep = original_sleep
        llm_backend._client = original_client

    check("a schema-violating answer is retried as transient",
          parsed2 == {"windows": []}, f"parsed={parsed2}")
    check("the transient backoff is the 5s first step", slept2 == [5.0],
          f"slept={slept2}")
    check("the retry's bill is settled without touching a bad cost",
          cost2["input_tokens"] == 701)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    print("🧪 Verifying clip-picking rate-limit handling...")
    verify_duration_parsing()
    verify_daily_quota_detection()
    verify_pacer_budget()
    verify_rate_limit_error_surfaces_wait()
    verify_repo_wiring()
    if "--fast" in argv:
        print("\n ⏭️  Skipping the integration check (--fast).")
    else:
        verify_stage_integration()
    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("✨ All rate-limit checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())