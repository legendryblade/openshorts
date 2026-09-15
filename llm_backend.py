"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def max_completion_tokens() -> int:
    """Cap for one completion, from ``LLM_MAX_TOKENS``; 0 lets the server decide.

    A reasoning model spends real tokens thinking before it answers — gpt-oss
    burned 100-200 tokens on trivial replies in testing — and the detail pass
    then still has to print a whole clip list. Under a low server-side default
    the thinking can eat the budget and leave an empty generation, which fails
    the JSON validator with nothing to show for it (15-sep-2026: that is how the
    clip-selection detail pass died). The cap only costs anything when the
    tokens are actually generated.
    """
    try:
        return max(0, int(os.environ.get("LLM_MAX_TOKENS", "")))
    except ValueError:
        return 0


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    if "json_validate_failed" in body or "failed to validate json" in body:
        # The server's OWN validator rejecting the model's output is a format
        # problem too, and the looser mode is the fix: json_schema is the strict
        # one, while json_object lets the model through and leaves validation to
        # pydantic — which is what the caller does anyway. 15-sep-2026: the
        # detail pass died here with a 400 and an empty failed_generation.
        return True
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


class RateLimitError(RuntimeError):
    """A 429 from the server, carrying the wait the server asked for.

    The retry policy lives in the caller (``main._run_gemini_stage``), so the
    wait has to travel with the error: Groq answers a TPM breach with
    ``retry-after`` / ``x-ratelimit-reset-tokens`` against a 60-second window,
    and a blind 5s + 10s backoff can never outlast that. Subclasses
    RuntimeError, and the message still contains "429", so callers that only
    classify by text keep working.
    """

    def __init__(self, message, status_code=429, retry_after=None, rate_limit=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.rate_limit = rate_limit or {}


def parse_rate_limit_headers(headers) -> dict:
    """The rate-limit numbers a response advertises, as plain values.

    Groq sends them on every completion (measured 15-sep-2026, free tier:
    ``x-ratelimit-limit-tokens: 8000``, ``x-ratelimit-remaining-tokens: 5019``,
    ``x-ratelimit-reset-tokens: 22.357s``), so the pacer learns the real budget
    off the wire instead of trusting a hardcoded constant.
    """
    import clip_selection

    def _int(name):
        raw = headers.get(name)
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return None

    info = {}
    for key, header in (
        ("limit_tokens", "x-ratelimit-limit-tokens"),
        ("remaining_tokens", "x-ratelimit-remaining-tokens"),
        ("limit_requests", "x-ratelimit-limit-requests"),
        ("remaining_requests", "x-ratelimit-remaining-requests"),
    ):
        value = _int(header)
        if value is not None:
            info[key] = value
    reset_tokens = clip_selection.parse_go_duration(headers.get("x-ratelimit-reset-tokens"))
    if reset_tokens is not None:
        info["reset_tokens_seconds"] = reset_tokens
    retry_after = clip_selection.parse_retry_after(headers.get("retry-after"))
    if retry_after is not None:
        info["retry_after"] = retry_after
    return info


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if max_completion_tokens():
                body["max_completion_tokens"] = max_completion_tokens()
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                rate_limit = parse_rate_limit_headers(resp.headers)
                message = f"LLM server {resp.status_code} from {url}: {resp.text[:300]}"
                if resp.status_code == 429:
                    raise RateLimitError(
                        message, rate_limit=rate_limit,
                        retry_after=rate_limit.get("retry_after")
                        or rate_limit.get("reset_tokens_seconds"))
                raise RuntimeError(message)
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
        # The server's own rate-limit numbers, learned on every call: the pacer
        # in main.py adopts x-ratelimit-limit-tokens from here instead of
        # hardcoding a provider's budget.
        "rate_limit": parse_rate_limit_headers(resp.headers),
    }
    return validated, cost
