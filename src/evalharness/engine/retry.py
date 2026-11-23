"""Exponential backoff with full jitter, honouring server ``Retry-After`` hints."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from evalharness.providers.base import ProviderError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_retries: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    rng: random.Random = field(default_factory=random.Random)

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before retry number ``attempt`` (0-based).

        Full jitter (AWS architecture blog, "Exponential Backoff and Jitter") spreads retries
        from many concurrent workers so they don't hammer the server in lock-step.
        """
        if retry_after is not None:
            return min(retry_after, self.max_delay_s)
        ceiling = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return self.rng.uniform(0, ceiling)


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[int, ProviderError], None] | None = None,
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except ProviderError as exc:
            if not exc.retryable or attempt >= policy.max_retries:
                raise
            if on_retry is not None:
                on_retry(attempt, exc)
            await sleep(policy.delay(attempt, exc.retry_after))
            attempt += 1
