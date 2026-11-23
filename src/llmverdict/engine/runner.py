"""The evaluation engine.

Concurrency model: one asyncio task per sample, bounded by a semaphore (max in-flight requests)
and a token bucket (requests per minute). Each request is wrapped in a timeout and retried with
jittered exponential backoff on retryable errors only. A failure on one sample is recorded on
that sample and never aborts the run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from llmverdict.engine.cache import ResponseCache
from llmverdict.engine.ratelimit import AsyncTokenBucket
from llmverdict.engine.retry import RetryPolicy, call_with_retry
from llmverdict.providers.base import GenerationParams, Provider, ProviderError
from llmverdict.scorers.base import Scorer
from llmverdict.store import ResultStore
from llmverdict.summary import RunSummary, summarize
from llmverdict.tasks.base import TaskSpec
from llmverdict.types import ModelResponse, Sample, SampleResult, Score


class ExecutionConfig(BaseModel):
    concurrency: int = 8
    rpm: float | None = None
    max_retries: int = 4
    timeout_s: float = 120.0
    cache: bool = True


ProgressFn = Callable[[SampleResult], None]


class Runner:
    def __init__(
        self,
        provider: Provider,
        scorers: list[Scorer],
        store: ResultStore,
        *,
        params: GenerationParams | None = None,
        execution: ExecutionConfig | None = None,
        cache: ResponseCache | None = None,
        on_result: ProgressFn | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.provider = provider
        self.scorers = scorers
        self.store = store
        self.params = params or GenerationParams()
        self.execution = execution or ExecutionConfig()
        # Sampled (temperature > 0) outputs are never cached: see engine/cache.py.
        self.cache = cache if self.execution.cache and self.params.temperature == 0 else None
        self.on_result = on_result
        self._sleep = sleep
        self._limiter = (
            AsyncTokenBucket.per_minute(self.execution.rpm) if self.execution.rpm else None
        )
        self._retry = RetryPolicy(max_retries=self.execution.max_retries)
        self.retries = 0

    async def run(
        self,
        task: TaskSpec,
        run_id: str,
        *,
        config_snapshot: dict[str, Any] | None = None,
        resume: bool = True,
    ) -> RunSummary:
        self.store.create_run(
            run_id,
            task=task.name,
            model=self.provider.model,
            provider=self.provider.kind,
            config=config_snapshot or {},
            primary_metric=task.primary_metric,
        )
        done = self.store.completed_ids(run_id) if resume else set()
        todo = [s for s in task.samples if s.id not in done]
        semaphore = asyncio.Semaphore(self.execution.concurrency)

        async def worker(sample: Sample) -> None:
            async with semaphore:
                result = await self.evaluate(sample, run_id)
            self.store.save_result(result)
            if self.on_result:
                self.on_result(result)

        started = time.perf_counter()
        try:
            await asyncio.gather(*(worker(s) for s in todo))
        except BaseException:
            # Completed samples are already persisted; mark the run so `resume` picks it up.
            self.store.finish_run(run_id, {}, status="interrupted")
            raise
        wall = time.perf_counter() - started

        valid_ids = {s.id for s in task.samples}
        results = [r for r in self.store.get_results(run_id) if r.sample_id in valid_ids]
        summary = summarize(results, pricing=self.provider.pricing, wall_time_s=wall)
        self.store.finish_run(run_id, summary.model_dump(mode="json"))
        return summary

    async def generate(self, sample: Sample) -> ModelResponse:
        # NB: `is not None`, never truthiness - ResponseCache defines __len__, so an empty cache
        # is falsy and `if self.cache:` would silently disable caching.
        cache = self.cache
        key = self.provider.cache_key(sample.messages, self.params) if cache is not None else None
        if cache is not None and key:
            hit = cache.get(key)
            if hit is not None:
                return hit

        async def attempt() -> ModelResponse:
            if self._limiter:
                await self._limiter.acquire()
            try:
                return await asyncio.wait_for(
                    self.provider.generate(sample.messages, self.params),
                    timeout=self.execution.timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise ProviderError(
                    f"timed out after {self.execution.timeout_s}s", retryable=True
                ) from exc

        def count_retry(_attempt: int, _exc: ProviderError) -> None:
            self.retries += 1

        response = await call_with_retry(
            attempt, self._retry, sleep=self._sleep, on_retry=count_retry
        )
        if cache is not None and key:
            cache.put(key, response)
        return response

    async def evaluate(self, sample: Sample, run_id: str) -> SampleResult:
        try:
            response = await self.generate(sample)
        except ProviderError as exc:
            return SampleResult(
                run_id=run_id, sample_id=sample.id, error=str(exc), tags=sample.tags
            )

        scores: list[Score] = []
        for scorer in self.scorers:
            if not scorer.applies_to(sample):
                continue
            try:
                scores.append(await scorer.score(sample, response.text))
            except Exception as exc:  # a buggy scorer must not take the run down
                scores.append(
                    Score(name=scorer.name, value=None, explanation=f"scorer crashed: {exc!r}")
                )
        return SampleResult(
            run_id=run_id, sample_id=sample.id, response=response, scores=scores, tags=sample.tags
        )
