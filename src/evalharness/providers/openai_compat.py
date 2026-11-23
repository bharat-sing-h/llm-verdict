"""Client for any OpenAI-compatible ``/chat/completions`` endpoint.

One implementation covers OpenAI, vLLM, Ollama, llama.cpp server, TGI, LM Studio, Together,
Groq, OpenRouter and most hosted inference gateways.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from evalharness.providers.base import GenerationParams, Pricing, Provider, ProviderError
from evalharness.types import Message, ModelResponse, Usage

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form; fall back to exponential backoff


class OpenAICompatProvider(Provider):
    kind = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        pricing: Pricing | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(model, pricing)
        self.base_url = base_url.rstrip("/")
        key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self.extra_body = extra_body or {}
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)
        if client is not None:
            self._client.headers.update(headers)

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), "base_url": self.base_url, "extra_body": self.extra_body}

    def _body(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        for key in ("top_p", "stop", "seed"):
            value = getattr(params, key)
            if value is not None:
                body[key] = value
        body.update(self.extra_body)
        return body

    async def generate(self, messages: list[Message], params: GenerationParams) -> ModelResponse:
        url = f"{self.base_url}/chat/completions"
        started = time.perf_counter()
        try:
            resp = await self._client.post(url, json=self._body(messages, params))
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise ProviderError(f"transport error: {exc!r}", retryable=True) from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if resp.status_code != 200:
            raise ProviderError(
                f"HTTP {resp.status_code}: {resp.text[:500]}",
                retryable=resp.status_code in RETRYABLE_STATUS,
                status=resp.status_code,
                retry_after=_parse_retry_after(resp.headers.get("retry-after")),
            )

        try:
            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed response: {resp.text[:500]}", retryable=False) from exc

        usage_raw = data.get("usage") or {}
        return ModelResponse(
            text=text,
            model=data.get("model", self.model),
            usage=Usage(
                prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
                completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            ),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
