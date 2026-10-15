"""Minimal LLM provider adapter (no SDKs, just HTTPS via httpx).

* ``anthropic``         -> Anthropic Messages API
* ``openai_compatible`` -> any /chat/completions endpoint: OpenAI, Groq,
                           Google Gemini (OpenAI-compatible endpoint),
                           OpenRouter, Together, a local vLLM/Ollama, ...

The model is never given tools: it receives text and returns text. Retries
cover rate limits, 5xx and timeouts; anything else surfaces as
``LLMError`` with a message that is safe to show (no keys, no raw bodies).
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Protocol

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Model provider failure. ``str(err)`` is safe to show to users."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str: ...


_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class _HTTPClient:
    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise LLMError("The language model is not configured (missing LLM_API_KEY).")
        if not settings.llm_model:
            raise LLMError("The language model is not configured (missing LLM_MODEL).")
        self.settings = settings
        self._http = httpx.Client(timeout=httpx.Timeout(settings.llm_timeout_s, connect=10.0))

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        # Rate limits (429) get more patience than other transient errors: free
        # tiers enforce per-minute token budgets and say how long to wait.
        attempts = self.settings.llm_max_retries + 1
        rate_limit_attempts = max(attempts, 5)
        attempt, rate_limited = 0, False
        while True:
            wait = None
            try:
                resp = self._http.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                reason = "timeout"
            except httpx.HTTPError as exc:
                reason = type(exc).__name__
            else:
                if resp.status_code == 200:
                    return resp.json()
                reason = f"HTTP {resp.status_code}"
                if resp.status_code not in _RETRYABLE:
                    log.error("LLM request rejected: %s", reason)
                    if resp.status_code in (401, 403):
                        raise LLMError("The language model rejected the configured credentials.")
                    raise LLMError(f"The language model request failed ({reason}).")
                if resp.status_code == 429:
                    rate_limited = True
                    wait = _retry_after_seconds(resp)
            attempt += 1
            limit = rate_limit_attempts if rate_limited else attempts
            log.warning("LLM attempt %d/%d failed: %s", attempt, limit, reason)
            if attempt >= limit:
                break
            time.sleep(min(wait if wait is not None else 2 ** attempt, 30))
        if rate_limited:
            raise LLMError("The language model's rate limit was reached. Please wait a few seconds and try again.")
        raise LLMError("The language model is temporarily unavailable. Please try again in a moment.")


class AnthropicClient(_HTTPClient):
    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        base = (self.settings.llm_base_url or "https://api.anthropic.com").rstrip("/")
        data = self._post(
            f"{base}/v1/messages",
            {
                "x-api-key": self.settings.llm_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.settings.llm_model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        try:
            return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError):
            raise LLMError("The language model returned an unexpected response.")


class OpenAICompatibleClient(_HTTPClient):
    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if not self.settings.llm_base_url:
            raise LLMError("The language model is not configured (missing LLM_BASE_URL).")
        payload = {
            "model": self.settings.llm_model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.settings.llm_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.settings.llm_reasoning_effort:
            # Reasoning models spend completion tokens on hidden reasoning first;
            # keep effort low and leave room so the JSON answer is not truncated.
            payload["reasoning_effort"] = self.settings.llm_reasoning_effort
            payload["max_tokens"] = max(max_tokens, 2048)
        data = self._post(
            f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {self.settings.llm_api_key}", "Content-Type": "application/json"},
            payload,
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError("The language model returned an unexpected response.")


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Seconds to wait from Retry-After or x-ratelimit-reset-* headers (e.g. '7.5s', '1m2s')."""
    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        value = resp.headers.get(name)
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            m = re.fullmatch(r"(?:(\d+)m)?(?:([\d.]+)s)?", value.strip())
            if m and (m.group(1) or m.group(2)):
                return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    return None


def make_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        return AnthropicClient(settings)
    if settings.llm_provider in ("openai_compatible", "openai", "groq", "gemini"):
        return OpenAICompatibleClient(settings)
    raise LLMError(f"Unsupported LLM_PROVIDER: {settings.llm_provider!r}")


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences / prose)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found in model output")
