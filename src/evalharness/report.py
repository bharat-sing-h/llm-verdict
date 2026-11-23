"""Markdown rendering for runs and comparisons (GitHub renders these natively in PR comments
and job summaries, which is where eval results are most often read)."""

from __future__ import annotations

import math
from typing import Any

from evalharness.calibration import CalibrationReport
from evalharness.compare import ComparisonReport, GateResult
from evalharness.store import ResultStore, RunRecord
from evalharness.summary import RunSummary

_VERDICT_ICON = {
    "improved": "🟢 improved",
    "regressed": "🔴 regressed",
    "no_significant_change": "⚪ no sig. change",
}


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.1f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_run(run: RunRecord, store: ResultStore | None = None, *, max_failures: int = 5) -> str:
    if run.summary is None:
        return f"# Run `{run.run_id}`\n\nStatus: **{run.status}** (no summary yet)\n"
    s = RunSummary.model_validate(run.summary)
    lines = [
        f"# Eval run `{run.run_id}`",
        "",
        f"**Task:** {run.task} · **Model:** `{run.model}` ({run.provider}) · "
        f"**Samples:** {s.n_samples} · **Errors:** {s.n_errors} · **Status:** {run.status}",
        "",
        "## Metrics",
        "",
        _table(
            ["Metric", "Score", "95% CI", "n", "N/A"],
            [
                [
                    f"**{m.name}**" if m.name == run.primary_metric else m.name,
                    _pct(m.mean),
                    f"[{_pct(m.ci_low)}, {_pct(m.ci_high)}]",
                    str(m.n),
                    str(m.not_applicable),
                ]
                for m in s.metrics.values()
            ],
        ),
    ]
    primary = run.primary_metric
    if primary and s.by_tag:
        rows = [
            [f"`{tag}`", _pct(ms[primary].mean), str(ms[primary].n)]
            for tag, ms in s.by_tag.items()
            if primary in ms and ms[primary].n
        ]
        if rows:
            lines += [
                "",
                f"## Breakdown by tag ({primary})",
                "",
                _table(["Tag", "Score", "n"], rows),
            ]
    lines += [
        "",
        "## Performance",
        "",
        _table(
            [
                "p50",
                "p95",
                "p99",
                "Prompt tok",
                "Completion tok",
                "Cost",
                "Cache hits",
                "Wall time",
            ],
            [
                [
                    f"{s.latency.p50_ms:.0f} ms",
                    f"{s.latency.p95_ms:.0f} ms",
                    f"{s.latency.p99_ms:.0f} ms",
                    f"{s.prompt_tokens:,}",
                    f"{s.completion_tokens:,}",
                    f"${s.cost_usd:.4f}",
                    str(s.cache_hits),
                    f"{s.wall_time_s:.1f} s",
                ]
            ],
        ),
    ]
    if store is not None and primary:
        failures = [
            r
            for r in store.get_results(run.run_id)
            if (sc := r.score(primary)) is not None and sc.passed is False
        ][:max_failures]
        if failures:
            lines += ["", f"## Sample failures ({primary})", ""]
            for r in failures:
                sc = r.score(primary)
                text = (r.response.text if r.response else "").strip().replace("\n", " ")
                lines.append(
                    f"- `{r.sample_id}` {' '.join(f'`{t}`' for t in r.tags)} — "
                    f"{sc.explanation if sc and sc.explanation else 'failed'}; "
                    f"output: _{text[:160]}{'…' if len(text) > 160 else ''}_"
                )
    if s.error_examples:
        lines += ["", "## Errors", ""] + [f"- {e}" for e in s.error_examples]
    return "\n".join(lines) + "\n"


def render_comparison(report: ComparisonReport, *, max_samples: int = 10) -> str:
    lines = [
        f"# Comparison on `{report.task}`",
        "",
        f"**Baseline:** `{report.baseline_run}` (`{report.baseline_model}`)  ",
        f"**Candidate:** `{report.candidate_run}` (`{report.candidate_model}`)",
        "",
        _table(
            ["Metric", "Baseline", "Candidate", "Δ", "95% CI (Δ)", "p-value", "Test", "Verdict"],
            [
                [
                    m.metric,
                    _pct(m.baseline),
                    _pct(m.candidate),
                    f"{100 * m.delta:+.1f} pp",
                    f"[{100 * m.ci_low:+.1f}, {100 * m.ci_high:+.1f}]",
                    f"{m.p_value:.4f}",
                    m.method,
                    _VERDICT_ICON[m.verdict],
                ]
                for m in report.metrics
            ],
        ),
        "",
        f"Paired on shared samples; α = {report.alpha}. "
        f"Unpaired samples excluded: {report.unpaired_samples}.",
    ]
    movers = sorted(report.by_tag, key=lambda t: t.delta)
    movers = [t for t in movers if abs(t.delta) > 1e-9]
    if movers:
        lines += ["", "## Biggest per-tag movers", ""]
        top = movers[:5] + [t for t in movers[-5:] if t not in movers[:5]]
        lines.append(
            _table(
                ["Tag", "Metric", "n", "Baseline", "Candidate", "Δ"],
                [
                    [
                        f"`{t.tag}`",
                        t.metric,
                        str(t.n),
                        _pct(t.baseline),
                        _pct(t.candidate),
                        f"{100 * t.delta:+.1f} pp",
                    ]
                    for t in top
                ],
            )
        )
    if report.regressions:
        lines += ["", f"## Regressed samples ({len(report.regressions)})", ""]
        lines += [
            f"- `{d.sample_id}` [{d.metric}] {d.baseline:.2f} → {d.candidate:.2f}"
            for d in report.regressions[:max_samples]
        ]
    return "\n".join(lines) + "\n"


def render_gate(result: GateResult) -> str:
    head = "## ✅ Eval gate passed" if result.passed else "## ❌ Eval gate failed"
    lines = [head, "", f"Metric: `{result.metric}` = {_pct(result.candidate_score)}"]
    if result.comparison:
        c = result.comparison
        lines.append(
            f"vs baseline: {100 * c.delta:+.1f} pp (95% CI [{100 * c.ci_low:+.1f}, "
            f"{100 * c.ci_high:+.1f}], p={c.p_value:.4f}, {c.method})"
        )
    lines += [f"- {r}" for r in result.reasons]
    return "\n".join(lines) + "\n"


def render_calibration(report: CalibrationReport) -> str:
    def f(x: float) -> str:
        return "–" if math.isnan(x) else f"{x:.3f}"

    rows: list[list[Any]] = [
        ["Examples (parsed / failed)", f"{report.n} / {report.parse_failures}"],
        ["Exact agreement", _pct(report.exact_agreement)],
        ["Cohen's κ", f(report.kappa)],
        ["Quadratic-weighted κ", f(report.quadratic_kappa)],
        ["Spearman ρ", f(report.spearman)],
        ["Pass/fail accuracy", _pct(report.binary_accuracy)],
        ["Pass/fail κ", f(report.binary_kappa)],
        [
            "Judge precision / recall (pass)",
            f"{f(report.judge_precision)} / {f(report.judge_recall)}",
        ],
        ["Leniency bias (judge − human)", f"{report.leniency_bias:+.2f}"],
    ]
    lines = [
        "# Judge calibration",
        "",
        _table(["Statistic", "Value"], [[str(a), str(b)] for a, b in rows]),
    ]
    labels = list(report.confusion)
    if labels:
        lines += [
            "",
            "## Confusion matrix (rows = human, cols = judge)",
            "",
            _table(
                ["", *labels], [[h, *[str(report.confusion[h][j]) for j in labels]] for h in labels]
            ),
        ]
    if report.worst_disagreements:
        lines += ["", "## Largest disagreements", ""]
        lines += [
            f"- `{d.id}` human={d.human} judge={d.judge}: _{d.reasoning[:200]}_"
            for d in report.worst_disagreements
        ]
    return "\n".join(lines) + "\n"
