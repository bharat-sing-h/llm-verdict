"""Paired comparison of two runs and the CI regression gate."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from llmverdict.stats import mean, paired_test
from llmverdict.store import ResultStore
from llmverdict.types import SampleResult

Verdict = Literal["improved", "regressed", "no_significant_change"]


class MetricComparison(BaseModel):
    metric: str
    n: int
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    verdict: Verdict


class SampleDiff(BaseModel):
    sample_id: str
    metric: str
    baseline: float
    candidate: float
    tags: list[str] = Field(default_factory=list)


class TagComparison(BaseModel):
    tag: str
    metric: str
    n: int
    baseline: float
    candidate: float
    delta: float


class ComparisonReport(BaseModel):
    baseline_run: str
    candidate_run: str
    baseline_model: str
    candidate_model: str
    task: str
    alpha: float
    metrics: list[MetricComparison]
    regressions: list[SampleDiff]
    improvements: list[SampleDiff]
    by_tag: list[TagComparison]
    unpaired_samples: int

    def metric(self, name: str) -> MetricComparison | None:
        return next((m for m in self.metrics if m.metric == name), None)


def _scores(results: list[SampleResult]) -> dict[str, dict[str, float]]:
    """sample_id -> metric -> value, for successfully scored samples only."""
    table: dict[str, dict[str, float]] = {}
    for r in results:
        if r.error:
            continue
        table[r.sample_id] = {s.name: s.value for s in r.scores if s.value is not None}
    return table


def compare_runs(
    store: ResultStore, baseline_id: str, candidate_id: str, *, alpha: float = 0.05
) -> ComparisonReport:
    base_run, cand_run = store.get_run(baseline_id), store.get_run(candidate_id)
    base_results, cand_results = store.get_results(baseline_id), store.get_results(candidate_id)
    base, cand = _scores(base_results), _scores(cand_results)
    tags = {r.sample_id: r.tags for r in cand_results}
    shared = sorted(base.keys() & cand.keys())
    metric_names = sorted({m for sid in shared for m in base[sid].keys() & cand[sid].keys()})

    metrics: list[MetricComparison] = []
    regressions: list[SampleDiff] = []
    improvements: list[SampleDiff] = []
    by_tag: list[TagComparison] = []
    for metric in metric_names:
        ids = [sid for sid in shared if metric in base[sid] and metric in cand[sid]]
        b = [base[sid][metric] for sid in ids]
        c = [cand[sid][metric] for sid in ids]
        test = paired_test(b, c, alpha=alpha)
        verdict: Verdict = "no_significant_change"
        if test.p_value < alpha:
            verdict = "improved" if test.delta > 0 else "regressed"
        metrics.append(
            MetricComparison(
                metric=metric,
                n=test.n,
                baseline=test.baseline_mean,
                candidate=test.candidate_mean,
                delta=test.delta,
                ci_low=test.ci_low,
                ci_high=test.ci_high,
                p_value=test.p_value,
                method=test.method,
                verdict=verdict,
            )
        )
        tag_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for sid, bv, cv in zip(ids, b, c, strict=True):
            if cv < bv:
                regressions.append(
                    SampleDiff(
                        sample_id=sid,
                        metric=metric,
                        baseline=bv,
                        candidate=cv,
                        tags=tags.get(sid, []),
                    )
                )
            elif cv > bv:
                improvements.append(
                    SampleDiff(
                        sample_id=sid,
                        metric=metric,
                        baseline=bv,
                        candidate=cv,
                        tags=tags.get(sid, []),
                    )
                )
            for tag in tags.get(sid, []):
                tag_pairs[tag].append((bv, cv))
        for tag, pairs in sorted(tag_pairs.items()):
            tb, tc = mean([p[0] for p in pairs]), mean([p[1] for p in pairs])
            by_tag.append(
                TagComparison(
                    tag=tag, metric=metric, n=len(pairs), baseline=tb, candidate=tc, delta=tc - tb
                )
            )

    return ComparisonReport(
        baseline_run=baseline_id,
        candidate_run=candidate_id,
        baseline_model=base_run.model,
        candidate_model=cand_run.model,
        task=cand_run.task,
        alpha=alpha,
        metrics=metrics,
        regressions=regressions,
        improvements=improvements,
        by_tag=by_tag,
        unpaired_samples=len(base.keys() ^ cand.keys()),
    )


class GatePolicy(BaseModel):
    metric: str | None = None  # defaults to the task's primary metric
    max_drop: float = 0.0
    alpha: float = 0.05
    min_score: float | None = None
    max_error_rate: float = 0.05
    mode: Literal["significant", "non_inferiority"] = "significant"


class GateResult(BaseModel):
    passed: bool
    metric: str
    candidate_score: float | None
    reasons: list[str]
    comparison: MetricComparison | None = None


def evaluate_gate(
    store: ResultStore, candidate_id: str, policy: GatePolicy, *, baseline_id: str | None = None
) -> GateResult:
    """Decide whether a candidate run is good enough to merge / deploy.

    * ``significant`` (default): fail only on a statistically significant drop larger than
      ``max_drop``. Robust to noise; won't flake on small benchmarks.
    * ``non_inferiority``: pass only if we're confident the drop is *smaller* than ``max_drop``
      (the lower CI bound of the delta is above ``-max_drop``). Stricter; needs more samples.
    """
    run = store.get_run(candidate_id)
    metric = policy.metric or run.primary_metric
    if metric is None:
        raise ValueError("no metric given and the run has no primary metric")
    summary = run.summary or {}
    reasons: list[str] = []

    n, n_err = summary.get("n_samples", 0), summary.get("n_errors", 0)
    if n and n_err / n > policy.max_error_rate:
        reasons.append(f"error rate {n_err / n:.1%} exceeds {policy.max_error_rate:.1%}")

    metric_summary = summary.get("metrics", {}).get(metric)
    score = metric_summary["mean"] if metric_summary else None
    if score is None:
        reasons.append(f"metric {metric!r} missing from run {candidate_id}")
    elif policy.min_score is not None and score < policy.min_score:
        reasons.append(f"{metric} = {score:.4f} is below the minimum {policy.min_score:.4f}")

    comparison = None
    if baseline_id is not None:
        comparison = compare_runs(store, baseline_id, candidate_id, alpha=policy.alpha).metric(
            metric
        )
        if comparison is None:
            reasons.append(f"metric {metric!r} could not be paired between runs")
        elif policy.mode == "significant":
            if comparison.delta < -policy.max_drop and comparison.p_value < policy.alpha:
                reasons.append(
                    f"significant regression on {metric}: {comparison.delta:+.4f} "
                    f"(95% CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}], "
                    f"p={comparison.p_value:.4f})"
                )
        elif comparison.ci_low < -policy.max_drop:
            reasons.append(
                f"cannot rule out a drop larger than {policy.max_drop} on {metric}: "
                f"CI lower bound {comparison.ci_low:+.4f}"
            )

    return GateResult(
        passed=not reasons,
        metric=metric,
        candidate_score=score,
        reasons=reasons,
        comparison=comparison,
    )
