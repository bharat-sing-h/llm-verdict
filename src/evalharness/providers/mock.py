"""Deterministic offline providers for tests, CI smoke runs and evaluating arbitrary Python apps."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import re
import sys
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from evalharness.providers.base import GenerationParams, Pricing, Provider, ProviderError
from evalharness.types import Message, ModelResponse, Usage


def _approx_tokens(text: str) -> int:
    # ~4 chars/token is the usual rule of thumb for English BPE vocabularies.
    return max(1, len(text) // 4)


class MockRule(BaseModel):
    pattern: str
    response: str


class MockProvider(Provider):
    """Returns canned responses chosen by regex over the conversation.

    ``fail_first`` makes the first N calls raise a retryable error, which is how the retry
    path is exercised in tests without a network.
    """

    kind = "mock"

    def __init__(
        self,
        model: str = "mock-model",
        rules: list[MockRule] | None = None,
        default: str = "",
        echo: bool = False,
        latency_ms: float = 0.0,
        fail_first: int = 0,
        pricing: Pricing | None = None,
    ) -> None:
        super().__init__(model, pricing)
        self.rules = [(re.compile(r.pattern, re.S | re.I), r.response) for r in rules or []]
        self.default = default
        self.echo = echo
        self.latency_ms = latency_ms
        self._failures_left = fail_first
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        return {
            **super().identity(),
            "rules": [(p.pattern, r) for p, r in self.rules],
            "default": self.default,
            "echo": self.echo,
        }

    async def generate(self, messages: list[Message], params: GenerationParams) -> ModelResponse:
        self.calls += 1
        if self._failures_left > 0:
            self._failures_left -= 1
            raise ProviderError("simulated overload", retryable=True, status=529)
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)

        conversation = "\n".join(m.content for m in messages)
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = last_user if self.echo else self.default
        for pattern, response in self.rules:
            if pattern.search(conversation):
                text = response
                break
        return ModelResponse(
            text=text,
            model=self.model,
            usage=Usage(
                prompt_tokens=_approx_tokens(conversation), completion_tokens=_approx_tokens(text)
            ),
            latency_ms=self.latency_ms,
            finish_reason="stop",
        )


AppFn = Callable[[list[dict[str, str]]], Any]


class FunctionProvider(Provider):
    """Evaluate *any* Python callable – a RAG pipeline, an agent, a guarded chatbot.

    The callable receives ``[{"role": ..., "content": ...}, ...]`` and returns a string
    (sync or async). Referenced from config as ``"package.module:function"``.
    This is what lets the harness test a whole LLM *application*, not just a raw model.
    """

    kind = "function"

    def __init__(self, target: str | AppFn, model: str | None = None) -> None:
        if isinstance(target, str):
            module_name, _, attr = target.partition(":")
            if not attr:
                raise ValueError("function target must look like 'module.path:callable'")
            # Console scripts don't put the working directory on sys.path; user apps live there.
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())
            fn = getattr(importlib.import_module(module_name), attr)
            name = target
        else:
            fn = target
            name = getattr(target, "__qualname__", "function")
        super().__init__(model or name)
        self.fn: AppFn = fn

    async def generate(self, messages: list[Message], params: GenerationParams) -> ModelResponse:
        payload = [m.model_dump() for m in messages]
        started = time.perf_counter()
        try:
            result = self.fn(payload)
            if inspect.isawaitable(result):
                result = await result
        except ProviderError:
            raise
        except Exception as exc:  # the app under test crashed: a real, non-retryable failure
            raise ProviderError(f"application error: {exc!r}", retryable=False) from exc
        text = str(result)
        conversation = "\n".join(m.content for m in messages)
        return ModelResponse(
            text=text,
            model=self.model,
            usage=Usage(
                prompt_tokens=_approx_tokens(conversation), completion_tokens=_approx_tokens(text)
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason="stop",
        )
