"""Async token-bucket rate limiter.

Providers enforce requests-per-minute quotas; exceeding them produces 429s, which waste retries
and skew latency measurements. A client-side bucket keeps us just under the quota instead.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class AsyncTokenBucket:
    def __init__(
        self,
        rate_per_s: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self.rate = rate_per_s
        # Default burst = one second of traffic (at least one request).
        self.capacity = capacity if capacity is not None else max(1.0, rate_per_s)
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        # The lock makes waiters queue FIFO, so no request starves under contention.
        self._lock = asyncio.Lock()

    @classmethod
    def per_minute(cls, rpm: float, **kwargs: object) -> AsyncTokenBucket:
        return cls(rpm / 60.0, **kwargs)  # type: ignore[arg-type]

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens > self.capacity:
            raise ValueError("requested more tokens than bucket capacity")
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await self._sleep((tokens - self._tokens) / self.rate)
