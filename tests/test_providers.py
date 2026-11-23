from __future__ import annotations

import json

import httpx
import pytest
import respx

from llmverdict.providers import (
    FunctionProvider,
    GenerationParams,
    MockProvider,
    MockRule,
    OpenAICompatProvider,
    Pricing,
    ProviderConfig,
    ProviderError,
    build_provider,
)
from llmverdict.types import Message, Usage

MSGS = [Message(role="system", content="sys"), Message(role="user", content="What is 2+2?")]
URL = "http://llm.test/v1/chat/completions"


def ok_body(text: str = "4") -> dict:
    return {
        "model": "served-model",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2},
    }


@respx.mock
async def test_openai_compat_success_and_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-test")
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=ok_body()))
    provider = OpenAICompatProvider(
        "m", base_url="http://llm.test/v1/", api_key_env="TEST_KEY", extra_body={"top_k": 5}
    )
    resp = await provider.generate(MSGS, GenerationParams(max_tokens=7, seed=3, stop=["\n"]))
    await provider.aclose()

    assert resp.text == "4"
    assert resp.model == "served-model"
    assert resp.usage == Usage(prompt_tokens=11, completion_tokens=2)
    assert resp.finish_reason == "stop"
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["max_tokens"] == 7
    assert body["seed"] == 3
    assert body["stop"] == ["\n"]
    assert body["top_k"] == 5
    assert "top_p" not in body


@respx.mock
@pytest.mark.parametrize(
    ("status", "retryable", "retry_after"),
    [(429, True, 2.0), (503, True, None), (400, False, None), (401, False, None)],
)
async def test_openai_compat_error_classification(
    status: int, retryable: bool, retry_after: float | None
) -> None:
    headers = (
        {"retry-after": "2"} if retry_after else {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}
    )
    respx.post(URL).mock(return_value=httpx.Response(status, text="nope", headers=headers))
    provider = OpenAICompatProvider("m", base_url="http://llm.test/v1", api_key_env=None)
    with pytest.raises(ProviderError) as info:
        await provider.generate(MSGS, GenerationParams())
    assert info.value.retryable is retryable
    assert info.value.status == status
    assert info.value.retry_after == retry_after


@respx.mock
async def test_openai_compat_transport_and_malformed() -> None:
    provider = OpenAICompatProvider("m", base_url="http://llm.test/v1", api_key_env=None)
    respx.post(URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(ProviderError) as info:
        await provider.generate(MSGS, GenerationParams())
    assert info.value.retryable
    respx.post(URL).mock(return_value=httpx.Response(200, json={"choices": []}))
    with pytest.raises(ProviderError) as info:
        await provider.generate(MSGS, GenerationParams())
    assert not info.value.retryable


@respx.mock
async def test_openai_compat_null_content_and_injected_client() -> None:
    body = ok_body()
    body["choices"][0]["message"]["content"] = None
    del body["usage"]
    respx.post(URL).mock(return_value=httpx.Response(200, json=body))
    client = httpx.AsyncClient()
    provider = OpenAICompatProvider("m", base_url="http://llm.test/v1", api_key="k", client=client)
    resp = await provider.generate(MSGS, GenerationParams())
    assert resp.text == ""
    assert resp.usage.total_tokens == 0
    assert client.headers["authorization"] == "Bearer k"
    await provider.aclose()


async def test_mock_provider_rules_echo_and_failures() -> None:
    p = MockProvider(rules=[MockRule(pattern=r"2\+2", response="4")], default="?", fail_first=1)
    with pytest.raises(ProviderError):
        await p.generate(MSGS, GenerationParams())
    assert (await p.generate(MSGS, GenerationParams())).text == "4"
    assert (await p.generate([Message(role="user", content="x")], GenerationParams())).text == "?"
    echo = MockProvider(echo=True, latency_ms=1)
    resp = await echo.generate(MSGS, GenerationParams())
    assert resp.text == "What is 2+2?"
    assert resp.usage.completion_tokens > 0
    assert p.calls == 3


async def test_function_provider_sync_async_and_errors() -> None:
    def sync_app(messages: list[dict[str, str]]) -> str:
        return messages[-1]["content"].upper()

    async def async_app(messages: list[dict[str, str]]) -> int:
        return len(messages)

    def broken(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("bug in app")

    assert (
        await FunctionProvider(sync_app).generate(MSGS, GenerationParams())
    ).text == "WHAT IS 2+2?"
    assert (await FunctionProvider(async_app).generate(MSGS, GenerationParams())).text == "2"
    with pytest.raises(ProviderError) as info:
        await FunctionProvider(broken).generate(MSGS, GenerationParams())
    assert not info.value.retryable
    assert "bug in app" in str(info.value)
    by_path = FunctionProvider("examples.apps.support_bot:naive_bot")
    assert by_path.model == "examples.apps.support_bot:naive_bot"
    with pytest.raises(ValueError, match=r"module\.path:callable"):
        FunctionProvider("no_colon")


def test_cache_key_is_stable_and_sensitive() -> None:
    p = MockProvider(model="a")
    k1 = p.cache_key(MSGS, GenerationParams())
    assert k1 == MockProvider(model="a").cache_key(MSGS, GenerationParams())
    assert k1 != MockProvider(model="b").cache_key(MSGS, GenerationParams())
    assert k1 != p.cache_key(MSGS, GenerationParams(max_tokens=1))
    assert k1 != p.cache_key(MSGS[:1], GenerationParams())


def test_pricing() -> None:
    cost = Pricing(input_per_mtok=1.0, output_per_mtok=2.0).cost(
        Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    )
    assert cost == pytest.approx(2.0)


def test_build_provider() -> None:
    assert isinstance(build_provider(ProviderConfig(provider="mock")), MockProvider)
    assert isinstance(
        build_provider(ProviderConfig(model="x", api_key_env=None)), OpenAICompatProvider
    )
    fn = build_provider(
        ProviderConfig(provider="function", target="examples.apps.support_bot:guarded_bot")
    )
    assert isinstance(fn, FunctionProvider)
    with pytest.raises(ValueError, match="requires 'target"):
        build_provider(ProviderConfig(provider="function"))
