"""Provider abstraction: anything that turns a list of messages into text."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from evalharness.types import Message, ModelResponse, Usage


class GenerationParams(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float | None = None
    stop: list[str] | None = None
    seed: int | None = None


class Pricing(BaseModel):
    """USD per one million tokens."""

    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, usage: Usage) -> float:
        return (
            usage.prompt_tokens * self.input_per_mtok
            + usage.completion_tokens * self.output_per_mtok
        ) / 1_000_000


class ProviderError(Exception):
    """Raised by providers. ``retryable`` tells the engine whether backing off can help."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


class Provider(ABC):
    """Base class for model backends.

    Subclasses implement :meth:`generate`. They must raise :class:`ProviderError` (never raw
    transport exceptions) so the engine can make a uniform retry decision.
    """

    #: short identifier used in run metadata, e.g. ``"openai_compat"``
    kind: str = "base"

    def __init__(self, model: str, pricing: Pricing | None = None) -> None:
        self.model = model
        self.pricing = pricing or Pricing()

    @abstractmethod
    async def generate(self, messages: list[Message], params: GenerationParams) -> ModelResponse:
        """Return a completion for ``messages``."""

    def identity(self) -> dict[str, Any]:
        """Everything that changes the output distribution; used for cache keys."""
        return {"kind": self.kind, "model": self.model}

    def cache_key(self, messages: list[Message], params: GenerationParams) -> str:
        payload = {
            "provider": self.identity(),
            "messages": [m.model_dump() for m in messages],
            "params": params.model_dump(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources."""
