"""Statistics for evaluation results, in dependency-free Python.

A single accuracy number hides two questions practitioners actually care about:

1. *How sure are we?* -> bootstrap confidence intervals.
2. *Is model B really better than model A on the same items?* -> paired tests. Pairing matters:
   two models' scores on the same benchmark are strongly correlated, so an unpaired comparison
   of two CIs is badly under-powered and frequently calls real differences "noise".
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method), ``q`` in [0, 100]."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Percentile bootstrap CI for the mean."""
    n = len(values)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    return Interval(
        mean(values),
        percentile(means, 100 * alpha / 2),
        percentile(means, 100 * (1 - alpha / 2)),
    )


def wilson_interval(successes: int, n: int, *, z: float = 1.959964) -> Interval:
    """Wilson score interval for a binomial proportion; well behaved near 0 and 1."""
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class PairedTest:
    n: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    method: Literal["mcnemar_exact", "paired_bootstrap"]


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value for k successes out of n at p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    # exact integer arithmetic, then a single division: no float overflow for large n
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_exact(baseline: Sequence[bool], candidate: Sequence[bool]) -> float:
    """Exact McNemar test on paired binary outcomes; only discordant pairs carry information."""
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have equal length")
    b = sum(1 for x, y in zip(baseline, candidate, strict=True) if x and not y)
    c = sum(1 for x, y in zip(baseline, candidate, strict=True) if y and not x)
    return _binom_two_sided(b, b + c)


def paired_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedTest:
    """Compare two systems on the same items.

    The CI on the delta always comes from a paired bootstrap. The p-value uses the exact McNemar
    test when every score is binary (the textbook test for paired accuracy), otherwise a
    two-sided paired-bootstrap p-value.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired samples must have equal length")
    n = len(baseline)
    if n == 0:
        nan = float("nan")
        return PairedTest(0, nan, nan, nan, nan, nan, 1.0, "paired_bootstrap")
    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    rng = random.Random(seed)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    delta = mean(diffs)

    binary = all(v in (0.0, 1.0) for v in (*baseline, *candidate))
    if binary:
        p_value = mcnemar_exact([v == 1.0 for v in baseline], [v == 1.0 for v in candidate])
        method: Literal["mcnemar_exact", "paired_bootstrap"] = "mcnemar_exact"
    else:
        if all(d == 0 for d in diffs):
            p_value = 1.0
        else:
            # Shift the bootstrap distribution to be centred on 0 (the null) and count how often
            # it is at least as extreme as the observed delta.
            extreme = sum(1 for m in boot if abs(m - delta) >= abs(delta))
            p_value = min(1.0, (extreme + 1) / (n_resamples + 1))
        method = "paired_bootstrap"

    return PairedTest(
        n=n,
        baseline_mean=mean(baseline),
        candidate_mean=mean(candidate),
        delta=delta,
        ci_low=percentile(boot, 100 * alpha / 2),
        ci_high=percentile(boot, 100 * (1 - alpha / 2)),
        p_value=p_value,
        method=method,
    )


def cohen_kappa(
    rater_a: Sequence[Hashable],
    rater_b: Sequence[Hashable],
    *,
    weights: Literal[None, "linear", "quadratic"] = None,
    labels: Sequence[Hashable] | None = None,
) -> float:
    """Cohen's kappa: agreement corrected for chance.

    Use ``weights="quadratic"`` for ordinal scales (e.g. 1-5 judge scores) so that a 4-vs-5
    disagreement costs less than a 1-vs-5 one.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same items")
    if not rater_a:
        return float("nan")
    cats = list(labels) if labels is not None else sorted(set(rater_a) | set(rater_b), key=str)
    index = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(rater_a)
    if k == 1:
        return 1.0

    observed = [[0.0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b, strict=True):
        observed[index[a]][index[b]] += 1
    count_a = Counter(index[a] for a in rater_a)
    count_b = Counter(index[b] for b in rater_b)

    def weight(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        dist = abs(i - j) / (k - 1)
        return dist if weights == "linear" else dist**2

    obs = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k)) / n
    exp = sum(weight(i, j) * count_a[i] * count_b[j] for i in range(k) for j in range(k)) / n**2
    if exp == 0:
        return 1.0
    return 1 - obs / exp


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for idx in order[i : j + 1]:
            ranks[idx] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks; handles ties)."""
    return pearson(_ranks(x), _ranks(y))
