from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest
from tests.conftest import make_sample

from llmverdict.engine import (
    AsyncTokenBucket,
    ExecutionConfig,
    ResponseCache,
    RetryPolicy,
    Runner,
    call_with_retry,
)
from llmverdict.providers import GenerationParams, MockProvider, MockRule, Provider, ProviderError
from llmverdict.scorers import build_scorer
from llmverdict.scorers.base import Scorer
from llmverdict.store import ResultStore
from llmverdict.tasks import TaskSpec
from llmverdict.types import Message, ModelResponse, Sample, Score


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# --- rate limiter -------------------------------------------------------------------------


async def test_token_bucket_bursts_then_throttles() -> None:
    clock = FakeClock()
    bucket = AsyncTokenBucket(rate_per_s=2, capacity=2, clock=clock, sleep=clock.sleep)
    for _ in range(2):
        await bucket.acquire()
    assert clock.sleeps == []  # burst capacity is free
    for _ in range(4):
        await bucket.acquire()
    assert clock.now == pytest.approx(2.0)  # 4 more tokens at 2/s


async def test_token_bucket_validation_and_per_minute() -> None:
    with pytest.raises(ValueError, match="positive"):
        AsyncTokenBucket(0)
    bucket = AsyncTokenBucket.per_minute(120)
    assert bucket.rate == 2
    with pytest.raises(ValueError, match="capacity"):
        await bucket.acquire(10)


# --- retry --------------------------------------------------------------------------------


def test_retry_policy_delays() -> None:
    policy = RetryPolicy(base_delay_s=1, max_delay_s=5, rng=random.Random(0))
    for attempt in range(6):
        assert 0 <= policy.delay(attempt) <= min(5, 2**attempt)
    assert policy.delay(0, retry_after=3) == 3
    assert policy.delay(0, retry_after=99) == 5


async def test_call_with_retry_recovers_and_gives_up() -> None:
    clock = FakeClock()
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("busy", retryable=True)
        return "ok"

    seen: list[int] = []
    result = await call_with_retry(
        flaky, RetryPolicy(max_retries=5), sleep=clock.sleep, on_retry=lambda a, e: seen.append(a)
    )
    assert result == "ok"
    assert seen == [0, 1]

    async def always_busy() -> str:
        raise ProviderError("busy", retryable=True)

    with pytest.raises(ProviderError):
        await call_with_retry(always_busy, RetryPolicy(max_retries=2), sleep=clock.sleep)

    attempts = {"n": 0}

    async def fatal() -> str:
        attempts["n"] += 1
        raise ProviderError("bad request", retryable=False)

    with pytest.raises(ProviderError):
        await call_with_retry(fatal, RetryPolicy(max_retries=5), sleep=clock.sleep)
    assert attempts["n"] == 1  # non-retryable errors fail fast


# --- cache --------------------------------------------------------------------------------


def test_response_cache_roundtrip(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "sub" / "c.db")
    assert len(cache) == 0  # an empty cache is falsy: callers must use `is not None`
    assert cache.get("k") is None
    cache.put("k", ModelResponse(text="hi", model="m", cached=True))
    hit = cache.get("k")
    assert hit is not None
    assert (hit.text, hit.cached) == ("hi", True)
    assert (cache.hits, cache.misses, len(cache)) == (1, 1, 1)
    cache.close()


# --- runner -------------------------------------------------------------------------------


def math_task(n: int = 6) -> TaskSpec:
    samples = [
        make_sample(f"s{i}", user=f"What is {i}+{i}?", reference=2 * i, tags=[f"p:{i % 2}"])
        for i in range(n)
    ]
    return TaskSpec(
        name="addition",
        samples=samples,
        default_scorers=["final_answer"],
        primary_metric="final_answer",
    )


def adder() -> MockProvider:
    rules = [
        MockRule(pattern=rf"What is {i}\+{i}\?", response=f"Answer: {2 * i}") for i in range(4)
    ]
    return MockProvider(rules=rules, default="Answer: -1")


async def test_runner_end_to_end(store: ResultStore) -> None:
    task = math_task(6)
    seen: list[str] = []
    runner = Runner(
        adder(), [build_scorer("final_answer")], store, on_result=lambda r: seen.append(r.sample_id)
    )
    summary = await runner.run(task, "run1")
    assert sorted(seen) == [f"s{i}" for i in range(6)]
    metric = summary.metrics["final_answer"]
    assert metric.n == 6
    assert metric.mean == pytest.approx(4 / 6)  # mock knows 0..3, wrong on 4 and 5
    assert summary.by_tag["p:0"]["final_answer"].n == 3
    record = store.get_run("run1")
    assert record.status == "completed"
    assert record.primary_metric == "final_answer"


async def test_runner_records_errors_and_resumes(store: ResultStore) -> None:
    class Flaky(Provider):
        kind = "flaky"

        def __init__(self, broken: set[str]) -> None:
            super().__init__("flaky")
            self.broken = broken
            self.calls = 0

        async def generate(
            self, messages: list[Message], params: GenerationParams
        ) -> ModelResponse:
            self.calls += 1
            if any(b in messages[-1].content for b in self.broken):
                raise ProviderError("400 bad", retryable=False)
            return ModelResponse(text="Answer: 0", model=self.model)

    task = math_task(4)
    first = Flaky(broken={"3+3"})
    summary = await Runner(first, [build_scorer("final_answer")], store).run(task, "r")
    assert summary.n_errors == 1
    assert "s3" in summary.error_examples[0]

    second = Flaky(broken=set())
    summary = await Runner(second, [build_scorer("final_answer")], store).run(task, "r")
    assert second.calls == 1  # only the previously failed sample is re-run
    assert summary.n_errors == 0
    assert summary.n_samples == 4


async def test_runner_retries_transient_errors(store: ResultStore) -> None:
    clock = FakeClock()
    provider = MockProvider(default="Answer: 0", fail_first=2)
    runner = Runner(
        provider,
        [build_scorer("final_answer")],
        store,
        sleep=clock.sleep,
        execution=ExecutionConfig(concurrency=1, cache=False),
    )
    summary = await runner.run(math_task(1), "r")
    assert runner.retries == 2
    assert summary.n_errors == 0
    assert len(clock.sleeps) == 2


async def test_runner_cache_hits_on_rerun(store: ResultStore, tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.db")
    provider = adder()
    await Runner(provider, [build_scorer("final_answer")], store, cache=cache).run(math_task(), "a")
    calls_after_first = provider.calls
    summary = await Runner(provider, [build_scorer("exact_match")], store, cache=cache).run(
        math_task(), "b"
    )
    assert provider.calls == calls_after_first  # re-scoring is free
    assert summary.cache_hits == 6
    assert summary.cost_usd == 0
    # sampled requests bypass the cache entirely
    sampled = Runner(provider, [], store, cache=cache, params=GenerationParams(temperature=0.7))
    assert sampled.cache is None
    cache.close()


async def test_runner_timeout_concurrency_and_rate_limit(store: ResultStore) -> None:
    class Slow(Provider):
        kind = "slow"
        in_flight = 0
        peak = 0

        async def generate(
            self, messages: list[Message], params: GenerationParams
        ) -> ModelResponse:
            Slow.in_flight += 1
            Slow.peak = max(Slow.peak, Slow.in_flight)
            try:
                await asyncio.sleep(0.2 if "5+5" in messages[-1].content else 0.01)
            finally:
                Slow.in_flight -= 1
            return ModelResponse(text="Answer: 0", model=self.model)

    runner = Runner(
        Slow("slow"),
        [build_scorer("final_answer")],
        store,
        execution=ExecutionConfig(
            concurrency=2, timeout_s=0.1, max_retries=0, rpm=60_000, cache=False
        ),
    )
    summary = await runner.run(math_task(6), "r")
    assert Slow.peak <= 2
    assert summary.n_errors == 1
    assert "timed out" in summary.error_examples[0]


async def test_runner_survives_crashing_scorer_and_skips_tagged(store: ResultStore) -> None:
    class Crashy(Scorer):
        type_name = "crashy"

        async def score(self, sample: Sample, output: str) -> Score:
            raise RuntimeError("scorer bug")

    tagged = build_scorer({"type": "final_answer", "name": "even_only", "only_tags": ["p:0"]})
    summary = await Runner(adder(), [Crashy(), tagged], store).run(math_task(4), "r")
    assert summary.metrics["crashy"].not_applicable == 4
    assert summary.metrics["even_only"].n == 2
    result = store.get_results("r")[0]
    assert "scorer crashed" in (result.score("crashy").explanation or "")


async def test_runner_marks_interrupted_runs(store: ResultStore) -> None:
    class Abort(BaseException):  # stands in for Ctrl-C / task cancellation
        pass

    class Boom(Provider):
        kind = "boom"

        async def generate(
            self, messages: list[Message], params: GenerationParams
        ) -> ModelResponse:
            raise Abort

    with pytest.raises(Abort):
        await Runner(Boom("b"), [], store).run(math_task(1), "r")
    assert store.get_run("r").status == "interrupted"
