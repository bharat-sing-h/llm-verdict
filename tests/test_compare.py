from __future__ import annotations

import pytest
from tests.conftest import add_run

from evalharness.compare import GatePolicy, compare_runs, evaluate_gate
from evalharness.report import render_comparison, render_gate, render_run
from evalharness.store import ResultStore, RunNotFoundError

N = 80


def ids(n: int = N) -> list[str]:
    return [f"s{i:03d}" for i in range(n)]


def scores(n_correct: int, n: int = N) -> dict[str, float | None]:
    return {sid: 1.0 if i < n_correct else 0.0 for i, sid in enumerate(ids(n))}


def test_compare_detects_significant_regression(store: ResultStore) -> None:
    tags = {sid: ["easy" if i % 2 else "hard"] for i, sid in enumerate(ids())}
    add_run(store, "base", scores(70), tags=tags)
    add_run(store, "cand", scores(50), tags=tags, model="m2")
    report = compare_runs(store, "base", "cand")
    m = report.metric("acc")
    assert m is not None
    assert m.verdict == "regressed"
    assert m.delta == pytest.approx(-0.25)
    assert m.method == "mcnemar_exact"
    assert m.ci_high < 0
    assert len(report.regressions) == 20
    assert {t.tag for t in report.by_tag} == {"easy", "hard"}
    assert report.candidate_model == "m2"
    md = render_comparison(report)
    assert "regressed" in md
    assert "Regressed samples (20)" in md


def test_compare_small_difference_is_not_significant(store: ResultStore) -> None:
    add_run(store, "base", scores(60))
    add_run(store, "cand", scores(58))
    m = compare_runs(store, "base", "cand").metric("acc")
    assert m is not None
    assert m.verdict == "no_significant_change"


def test_compare_pairs_only_shared_successful_samples(store: ResultStore) -> None:
    add_run(store, "base", scores(10, n=10))
    cand = scores(10, n=12)
    add_run(store, "cand", cand, errors={"s000"})
    report = compare_runs(store, "base", "cand")
    m = report.metric("acc")
    assert m is not None
    assert m.n == 9
    assert report.unpaired_samples == 3  # s000 errored, s010/s011 missing from baseline
    with pytest.raises(RunNotFoundError):
        compare_runs(store, "base", "missing")


def test_compare_continuous_metric_uses_bootstrap(store: ResultStore) -> None:
    base = {sid: 0.5 for sid in ids(40)}
    cand = {sid: 0.7 for sid in ids(40)}
    add_run(store, "base", base)
    add_run(store, "cand", cand)
    m = compare_runs(store, "base", "cand").metric("acc")
    assert m is not None
    assert m.method == "paired_bootstrap"
    assert m.verdict == "improved"


def test_gate_policies(store: ResultStore) -> None:
    add_run(store, "base", scores(70))
    add_run(store, "worse", scores(50))
    add_run(store, "slightly_worse", scores(68))
    add_run(store, "same", scores(70))

    failed = evaluate_gate(store, "worse", GatePolicy(), baseline_id="base")
    assert not failed.passed
    assert "significant regression" in failed.reasons[0]
    assert "❌" in render_gate(failed)

    # noise-level dip passes the default (significance) policy...
    assert evaluate_gate(store, "slightly_worse", GatePolicy(), baseline_id="base").passed
    # ...but the stricter non-inferiority policy can't rule out a >1pp drop with n=80
    strict = evaluate_gate(
        store,
        "slightly_worse",
        GatePolicy(mode="non_inferiority", max_drop=0.01),
        baseline_id="base",
    )
    assert not strict.passed
    assert "cannot rule out" in strict.reasons[0]
    lenient = evaluate_gate(
        store, "same", GatePolicy(mode="non_inferiority", max_drop=0.5), baseline_id="base"
    )
    assert lenient.passed
    assert "✅" in render_gate(lenient)

    # absolute floor without a baseline
    floor = evaluate_gate(store, "worse", GatePolicy(min_score=0.8))
    assert not floor.passed
    assert "below the minimum" in floor.reasons[0]


def test_gate_error_rate_and_missing_metric(store: ResultStore) -> None:
    add_run(store, "flaky", scores(10, n=10), errors={"s000", "s001"})
    result = evaluate_gate(store, "flaky", GatePolicy(max_error_rate=0.1))
    assert not result.passed
    assert "error rate" in result.reasons[0]
    missing = evaluate_gate(store, "flaky", GatePolicy(metric="nope", max_error_rate=1.0))
    assert "missing" in missing.reasons[0]
    add_run(store, "other", scores(5, n=10), metric="other_metric")
    unpaired = evaluate_gate(store, "flaky", GatePolicy(max_error_rate=1.0), baseline_id="other")
    assert "could not be paired" in unpaired.reasons[0]


def test_gate_requires_metric(store: ResultStore) -> None:
    store.create_run("r", task="t", model="m", provider="p", config={}, primary_metric=None)
    with pytest.raises(ValueError, match="no metric"):
        evaluate_gate(store, "r", GatePolicy())


def test_render_run_without_summary(store: ResultStore) -> None:
    store.create_run("r", task="t", model="m", provider="p", config={})
    assert "no summary yet" in render_run(store.get_run("r"))
    add_run(store, "done", scores(5, n=10), errors={"s009"})
    md = render_run(store.get_run("done"), store)
    assert "## Metrics" in md
    assert "Sample failures" in md
    assert "## Errors" in md
