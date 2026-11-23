"""Aggregate per-sample results into run-level metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, Field

from evalharness.providers.base import Pricing
from evalharness.stats import bootstrap_ci, percentile, wilson_interval
from evalharness.types import SampleResult, Usage


class MetricSummary(BaseModel):
    name: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    pass_rate: float | None = None
    not_applicable: int = 0


class LatencySummary(BaseModel):
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


class RunSummary(BaseModel):
    n_samples: int
    n_errors: int
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    by_tag: dict[str, dict[str, MetricSummary]] = Field(default_factory=dict)
    latency: LatencySummary = Field(default_factory=LatencySummary)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0
    wall_time_s: float = 0.0
    error_examples: list[str] = Field(default_factory=list)


def summarize_metric(
    name: str, values: list[float], passed: list[bool], n_na: int
) -> MetricSummary:
    if not values:
        nan = float("nan")
        return MetricSummary(name=name, n=0, mean=nan, ci_low=nan, ci_high=nan, not_applicable=n_na)
    if all(v in (0.0, 1.0) for v in values):
        ci = wilson_interval(int(sum(values)), len(values))
    else:
        ci = bootstrap_ci(values)
    return MetricSummary(
        name=name,
        n=len(values),
        mean=ci.estimate,
        ci_low=ci.low,
        ci_high=ci.high,
        pass_rate=sum(passed) / len(passed) if passed else None,
        not_applicable=n_na,
    )


def _collect(
    results: Iterable[SampleResult],
) -> dict[str, tuple[list[float], list[bool], int]]:
    acc: dict[str, tuple[list[float], list[bool], int]] = {}
    for result in results:
        if result.error:
            continue
        for score in result.scores:
            values, passed, na = acc.get(score.name, ([], [], 0))
            if score.value is None:
                acc[score.name] = (values, passed, na + 1)
                continue
            values.append(score.value)
            if score.passed is not None:
                passed.append(score.passed)
            acc[score.name] = (values, passed, na)
    return acc


def summarize(
    results: list[SampleResult], *, pricing: Pricing | None = None, wall_time_s: float = 0.0
) -> RunSummary:
    """Errors count as failures of the *system* but are excluded from quality metrics.

    Mixing them in would conflate "the model answered wrong" with "the API was down"; instead
    ``n_errors`` is reported separately and gates can fail on it independently.
    """
    pricing = pricing or Pricing()
    ok = [r for r in results if not r.error]
    metrics = {name: summarize_metric(name, v, p, na) for name, (v, p, na) in _collect(ok).items()}

    by_tag_results: dict[str, list[SampleResult]] = defaultdict(list)
    for r in ok:
        for tag in r.tags:
            by_tag_results[tag].append(r)
    by_tag = {
        tag: {name: summarize_metric(name, v, p, na) for name, (v, p, na) in _collect(rs).items()}
        for tag, rs in sorted(by_tag_results.items())
    }

    fresh = [r.response for r in ok if r.response and not r.response.cached]
    latencies = [resp.latency_ms for resp in fresh]
    usage = Usage(
        prompt_tokens=sum(r.response.usage.prompt_tokens for r in ok if r.response),
        completion_tokens=sum(r.response.usage.completion_tokens for r in ok if r.response),
    )
    return RunSummary(
        n_samples=len(results),
        n_errors=len(results) - len(ok),
        metrics=metrics,
        by_tag=by_tag,
        latency=LatencySummary(
            mean_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_ms=percentile(latencies, 50) if latencies else 0.0,
            p95_ms=percentile(latencies, 95) if latencies else 0.0,
            p99_ms=percentile(latencies, 99) if latencies else 0.0,
        ),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        # Spend reflects only calls actually made; cache hits are free.
        cost_usd=round(
            pricing.cost(
                Usage(
                    prompt_tokens=sum(r.usage.prompt_tokens for r in fresh),
                    completion_tokens=sum(r.usage.completion_tokens for r in fresh),
                )
            ),
            6,
        ),
        cache_hits=sum(1 for r in ok if r.response and r.response.cached),
        wall_time_s=round(wall_time_s, 3),
        error_examples=[f"{r.sample_id}: {r.error}" for r in results if r.error][:5],
    )
