"""Streamlit dashboard over the results database.

    pip install -e ".[dashboard]"
    streamlit run dashboard/app.py -- --db .evalharness/results.db
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from evalharness.compare import compare_runs
from evalharness.store import ResultStore
from evalharness.summary import RunSummary

st.set_page_config(page_title="LLM Eval Harness", layout="wide")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=".evalharness/results.db")
    return parser.parse_known_args()[0]


def pct(x: float | None) -> str:
    return "–" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


db_path = st.sidebar.text_input("Results DB", _args().db)
if not Path(db_path).exists():
    st.error(f"No database at `{db_path}`. Run `evalharness run <config>` first.")
    st.stop()
# One connection per script run: Streamlit re-runs scripts on different threads, and sqlite3
# connections must not cross threads. Opening SQLite is cheap.
store = ResultStore(db_path)
runs = [r for r in store.list_runs() if r.summary]
if not runs:
    st.info("No completed runs yet.")
    st.stop()

tasks = sorted({r.task for r in runs})
task = st.sidebar.selectbox("Task", tasks)
task_runs = [r for r in runs if r.task == task]

st.title(f"📊 {task}")

# --- leaderboard ----------------------------------------------------------------------------
rows = []
for r in task_runs:
    s = RunSummary.model_validate(r.summary)
    row: dict[str, object] = {"run": r.run_id, "model": r.model, "samples": s.n_samples,
                              "errors": s.n_errors}
    for name, m in s.metrics.items():
        row[name] = 100 * m.mean
    row |= {"p50 ms": s.latency.p50_ms, "p95 ms": s.latency.p95_ms, "cost $": s.cost_usd}
    rows.append(row)
board = pd.DataFrame(rows)
primary = task_runs[0].primary_metric
if primary and primary in board:
    board = board.sort_values(primary, ascending=False)
metric_names = {n for r in task_runs for n in (r.summary or {}).get("metrics", {})}
st.subheader("Leaderboard")
st.dataframe(
    board,
    use_container_width=True,
    hide_index=True,
    column_config={
        **{n: st.column_config.NumberColumn(n, format="%.1f%%") for n in metric_names},
        "p50 ms": st.column_config.NumberColumn(format="%.0f"),
        "p95 ms": st.column_config.NumberColumn(format="%.0f"),
        "cost $": st.column_config.NumberColumn(format="$%.4f"),
    },
)

# --- run drill-down -------------------------------------------------------------------------
st.subheader("Run details")
# Default to the weakest run on the leaderboard: that's where the failures worth reading are.
ordered = list(board["run"])
run_id = st.selectbox("Run", ordered, index=len(ordered) - 1)
record = store.get_run(run_id)
summary = RunSummary.model_validate(record.summary)
cols = st.columns(max(1, len(summary.metrics)))
for col, m in zip(cols, summary.metrics.values(), strict=False):
    col.metric(m.name, pct(m.mean), help=f"95% CI [{pct(m.ci_low)}, {pct(m.ci_high)}], n={m.n}")

metric = st.selectbox("Metric for breakdown", list(summary.metrics), index=0)
tag_rows = [
    {"tag": tag, "score": ms[metric].mean, "n": ms[metric].n}
    for tag, ms in summary.by_tag.items()
    if metric in ms and ms[metric].n
]
if tag_rows:
    st.bar_chart(pd.DataFrame(tag_rows).set_index("tag")["score"], horizontal=True)

results = store.get_results(run_id)
only_failures = st.toggle("Only failures", value=True)
sample_rows = []
for res in results:
    sc = res.score(metric)
    if only_failures and (sc is None or sc.passed is not False):
        continue
    sample_rows.append({
        "sample": res.sample_id,
        "tags": ", ".join(res.tags),
        "score": None if sc is None else sc.value,
        "explanation": None if sc is None else sc.explanation,
        "output": res.response.text if res.response else res.error,
    })
st.dataframe(pd.DataFrame(sample_rows), use_container_width=True, hide_index=True)

# --- comparison -----------------------------------------------------------------------------
if len(task_runs) >= 2:
    st.subheader("Compare two runs")
    ids = [r.run_id for r in task_runs]
    c1, c2 = st.columns(2)
    base = c1.selectbox("Baseline", ids, index=1)
    cand = c2.selectbox("Candidate", ids, index=0)
    if base != cand:
        report = compare_runs(store, base, cand)
        st.dataframe(
            pd.DataFrame([m.model_dump() for m in report.metrics]),
            use_container_width=True, hide_index=True,
        )
