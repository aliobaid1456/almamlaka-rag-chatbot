"""Model-provider failures and timeouts (category 18) using httpx mock transports."""
import httpx
import pytest

from backend.config import Settings
from backend.llm import AnthropicClient, LLMError, OpenAICompatibleClient, make_llm, parse_json_object
from conftest import FakeLLM


def client(cls, handler, **kw):
    s = Settings(llm_api_key="test-key", llm_model="m", llm_base_url=kw.get("base", "https://example.test/v1"),
                 llm_max_retries=kw.get("retries", 2))
    c = cls(s)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("backend.llm.time.sleep", lambda s: None)


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "rate"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": 1}"}}]})

    assert client(OpenAICompatibleClient, handler).complete("s", "u") == '{"ok": 1}'
    assert calls["n"] == 3


def test_persistent_5xx_raises_safe_error():
    c = client(AnthropicClient, lambda r: httpx.Response(503, text="overloaded"))
    with pytest.raises(LLMError, match="temporarily unavailable"):
        c.complete("s", "u")


def test_timeout_raises_safe_error():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(LLMError, match="temporarily unavailable"):
        client(OpenAICompatibleClient, handler, retries=1).complete("s", "u")


def test_bad_credentials_not_retried_and_key_not_leaked():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, text="invalid x-api-key test-key")

    with pytest.raises(LLMError) as e:
        client(AnthropicClient, handler).complete("s", "u")
    assert calls["n"] == 1 and "test-key" not in str(e.value)


def test_anthropic_request_shape():
    seen = {}

    def handler(req):
        seen["url"], seen["headers"], seen["body"] = str(req.url), req.headers, req.read()
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}]})

    out = client(AnthropicClient, handler, base="https://api.anthropic.com").complete("SYS", "USER")
    assert out == "hi" and seen["url"].endswith("/v1/messages") and seen["headers"]["x-api-key"] == "test-key"
    assert b'"tools"' not in seen["body"]  # the model is never given tools


def test_missing_configuration():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        make_llm(Settings(llm_api_key="", llm_model="m"))
    with pytest.raises(LLMError, match="LLM_BASE_URL"):
        make_llm(Settings(llm_provider="openai_compatible", llm_api_key="k", llm_model="m")).complete("s", "u")


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": {"b": "}"}} thanks') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


class FailingLLM(FakeLLM):
    def complete(self, system, user, max_tokens=1200):
        raise LLMError("The language model is temporarily unavailable. Please try again in a moment.")


def test_pipeline_reports_provider_failure(make_pipeline):
    r = make_pipeline(FailingLLM()).answer("What is the budget?")
    assert r.kind == "error" and "temporarily unavailable" in r.text


def test_pipeline_without_configured_model(settings, store):
    from backend.rag import RAGPipeline
    from backend.retrieval import HybridRetriever

    p = RAGPipeline(settings, HybridRetriever(store, None), None, {}, "missing LLM_API_KEY")
    r = p.answer("What is the budget?")
    assert r.kind == "error" and "LLM_API_KEY" in r.text


def test_reasoning_effort_is_sent_with_room_for_output():
    seen = {}

    def handler(req):
        import json as _j
        seen.update(_j.loads(req.read()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    c = client(OpenAICompatibleClient, handler)
    c.settings.llm_reasoning_effort = "low"
    c.complete("s", "u", max_tokens=400)
    assert seen["reasoning_effort"] == "low" and seen["max_tokens"] >= 2048


def test_rate_limit_honours_retry_after_and_reports_clearly(monkeypatch):
    waits = []
    monkeypatch.setattr("backend.llm.time.sleep", lambda s: waits.append(s))
    c = client(OpenAICompatibleClient, lambda r: httpx.Response(429, headers={"retry-after": "7"}))
    with pytest.raises(LLMError, match="rate limit"):
        c.complete("s", "u")
    assert waits and all(w == 7 for w in waits) and len(waits) == 4  # 5 attempts for 429


def test_retry_after_header_formats():
    from backend.llm import _retry_after_seconds
    assert _retry_after_seconds(httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1m2.5s"})) == 62.5
    assert _retry_after_seconds(httpx.Response(429, headers={"x-ratelimit-reset-tokens": "450ms"})) is None
    assert _retry_after_seconds(httpx.Response(429)) is None
