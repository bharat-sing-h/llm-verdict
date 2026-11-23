from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from evalharness.providers.base import (
    GenerationParams,
    Pricing,
    Provider,
    ProviderError,
)
from evalharness.providers.mock import FunctionProvider, MockProvider, MockRule
from evalharness.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "FunctionProvider",
    "GenerationParams",
    "MockProvider",
    "MockRule",
    "OpenAICompatProvider",
    "Pricing",
    "Provider",
    "ProviderConfig",
    "ProviderError",
    "build_provider",
]


class ProviderConfig(BaseModel):
    """Declarative provider spec, as written in a run config's ``model:`` / ``judge:`` block."""

    provider: Literal["openai_compat", "mock", "function"] = "openai_compat"
    model: str = "mock-model"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = "OPENAI_API_KEY"
    timeout_s: float = 60.0
    extra_body: dict[str, Any] = Field(default_factory=dict)
    pricing: Pricing = Field(default_factory=Pricing)
    # mock-only
    rules: list[MockRule] = Field(default_factory=list)
    default: str = ""
    echo: bool = False
    # function-only
    target: str | None = None


def build_provider(cfg: ProviderConfig) -> Provider:
    if cfg.provider == "openai_compat":
        return OpenAICompatProvider(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            timeout_s=cfg.timeout_s,
            extra_body=cfg.extra_body,
            pricing=cfg.pricing,
        )
    if cfg.provider == "mock":
        return MockProvider(
            model=cfg.model,
            rules=cfg.rules,
            default=cfg.default,
            echo=cfg.echo,
            pricing=cfg.pricing,
        )
    if cfg.provider == "function":
        if not cfg.target:
            raise ValueError("provider 'function' requires 'target: module:callable'")
        return FunctionProvider(cfg.target, model=cfg.model if cfg.model != "mock-model" else None)
    raise ValueError(f"unknown provider {cfg.provider!r}")  # pragma: no cover - Literal guards
