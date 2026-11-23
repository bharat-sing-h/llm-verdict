from __future__ import annotations

import math

import pytest

from llmverdict import stats


def test_mean_and_percentile() -> None:
    assert stats.mean([1, 2, 3]) == 2
    assert math.isnan(stats.mean([]))
    data = [1.0, 2.0, 3.0, 4.0]
    assert stats.percentile(data, 50) == 2.5  # numpy linear interpolation
    assert stats.percentile(data, 0) == 1.0
    assert stats.percentile(data, 100) == 4.0
    assert math.isnan(stats.percentile([], 50))


def test_wilson_interval_matches_reference_values() -> None:
    # Reference values from statsmodels.stats.proportion.proportion_confint(8, 10, method="wilson")
    ci = stats.wilson_interval(8, 10)
    assert ci.estimate == pytest.approx(0.8)
    assert ci.low == pytest.approx(0.4902, abs=1e-4)
    assert ci.high == pytest.approx(0.9433, abs=1e-4)
    edge = stats.wilson_interval(0, 20)
    assert edge.low == 0.0
    assert edge.high > 0.1  # never a degenerate zero-width interval at the boundary
    assert math.isnan(stats.wilson_interval(0, 0).estimate)


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    values = [0.1, 0.5, 0.9, 0.4, 0.7, 0.3, 0.8, 0.2]
    a = stats.bootstrap_ci(values, seed=3)
    b = stats.bootstrap_ci(values, seed=3)
    assert a == b
    assert a.low < a.estimate < a.high
    assert a.half_width > 0
    assert math.isnan(stats.bootstrap_ci([]).estimate)


def test_mcnemar_exact() -> None:
    # 10 discordant pairs all favouring the candidate: p = 2 * 0.5**10
    base = [False] * 10 + [True] * 5
    cand = [True] * 10 + [True] * 5
    assert stats.mcnemar_exact(base, cand) == pytest.approx(2 / 1024)
    assert stats.mcnemar_exact([True, False], [True, False]) == 1.0
    with pytest.raises(ValueError, match="equal length"):
        stats.mcnemar_exact([True], [True, False])


def test_paired_test_binary_uses_mcnemar() -> None:
    base = [0.0] * 30 + [1.0] * 30
    cand = [1.0] * 30 + [1.0] * 30
    result = stats.paired_test(base, cand)
    assert result.method == "mcnemar_exact"
    assert result.delta == pytest.approx(0.5)
    assert result.p_value < 1e-6
    assert result.ci_low > 0


def test_paired_test_continuous_and_identical() -> None:
    base = [0.2, 0.4, 0.6, 0.8, 0.5, 0.3]
    same = stats.paired_test(base, base)
    assert same.method == "paired_bootstrap"
    assert same.p_value == 1.0
    assert same.delta == 0
    shifted = stats.paired_test(base * 5, [v + 0.15 for v in base * 5])
    assert shifted.delta == pytest.approx(0.15)
    assert shifted.p_value < 0.01
    empty = stats.paired_test([], [])
    assert empty.n == 0
    assert empty.p_value == 1.0
    with pytest.raises(ValueError, match="equal length"):
        stats.paired_test([1.0], [])


def test_paired_test_noise_is_not_significant() -> None:
    base = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    cand = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    assert stats.paired_test(base, cand).p_value > 0.5


def test_cohen_kappa_hand_computed() -> None:
    # po = 0.75, pe = 0.5*0.75 + 0.5*0.25 = 0.5  ->  kappa = 0.5
    assert stats.cohen_kappa([0, 1, 0, 1], [0, 1, 1, 1]) == pytest.approx(0.5)
    assert stats.cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert stats.cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert math.isnan(stats.cohen_kappa([], []))
    with pytest.raises(ValueError, match="same items"):
        stats.cohen_kappa([1], [1, 2])


def test_weighted_kappa_penalises_far_misses_more() -> None:
    labels = [1, 2, 3, 4, 5]
    human = [1, 2, 3, 4, 5, 3, 2, 4]
    near = [1, 2, 3, 5, 4, 3, 2, 4]  # two off-by-one errors
    far = [5, 2, 3, 4, 1, 3, 2, 4]  # two off-by-four errors
    q_near = stats.cohen_kappa(human, near, weights="quadratic", labels=labels)
    q_far = stats.cohen_kappa(human, far, weights="quadratic", labels=labels)
    assert q_near > 0.8 > q_far
    lin = stats.cohen_kappa(human, near, weights="linear", labels=labels)
    assert 0 < lin < 1
    # unweighted kappa can't tell the two apart
    assert stats.cohen_kappa(human, near, labels=labels) == pytest.approx(
        stats.cohen_kappa(human, far, labels=labels)
    )


def test_spearman_and_pearson() -> None:
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert stats.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)  # ties
    assert math.isnan(stats.pearson([1, 1, 1], [1, 2, 3]))
    assert math.isnan(stats.pearson([1], [1]))
