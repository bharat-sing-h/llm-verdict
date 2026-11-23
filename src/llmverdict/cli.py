"""Command-line interface.

``llmverdict run | report | compare | gate | calibrate | runs | tasks | export``
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from llmverdict import __version__
from llmverdict.calibration import calibrate as run_calibration
from llmverdict.calibration import load_labels
from llmverdict.compare import GatePolicy, compare_runs, evaluate_gate
from llmverdict.config import load_config
from llmverdict.harness import CompletedRun, run_config
from llmverdict.providers import build_provider
from llmverdict.report import render_calibration, render_comparison, render_gate, render_run
from llmverdict.scorers.judge import LLMJudge
from llmverdict.store import ResultStore, RunNotFoundError
from llmverdict.tasks import BUILTIN_TASKS, TaskSpec, load_builtin
from llmverdict.types import SampleResult

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Evaluate and red-team LLMs and LLM applications with statistically sound comparisons.",
)
console = Console()

DbOption = Annotated[
    Path,
    typer.Option("--db", envvar="LLMVERDICT_DB", help="Results database."),
]
DEFAULT_DB = Path(".llmverdict/results.db")


def _emit(markdown: str, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        console.print(f"[dim]wrote {output}[/dim]")
    else:
        console.print(Markdown(markdown))
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")


@contextmanager
def _open_store(db: Path) -> Iterator[ResultStore]:
    if not db.exists():
        console.print(f"[red]No results database at {db}. Run `llmverdict run` first.[/red]")
        raise typer.Exit(2)
    store = ResultStore(db)
    try:
        yield store
    except RunNotFoundError as exc:
        console.print(f"[red]Unknown run id {exc}[/red]")
        raise typer.Exit(2) from None
    finally:
        store.close()


@app.callback(invoke_without_command=True)
def _main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        console.print(f"llmverdict {__version__}")
        raise typer.Exit()


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Run config YAML.")],
    task: Annotated[
        list[str] | None, typer.Option("--task", "-t", help="Only these tasks.")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Max samples per task (smoke tests).")] = None,
    label: Annotated[
        str | None,
        typer.Option(help="Stable run label, e.g. 'baseline' -> run id 'baseline__<task>'."),
    ] = None,
    resume: Annotated[
        bool, typer.Option(help="Skip samples already completed under this run id.")
    ] = True,
    report_dir: Annotated[
        Path | None, typer.Option(help="Write a Markdown report per run here.")
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option("--db", envvar="LLMVERDICT_DB", help="Override the config's store_path."),
    ] = None,
) -> None:
    """Run every task in CONFIG against the configured model."""
    cfg = load_config(config)
    if db is not None:
        cfg.store_path = str(db)
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[red]{task.fields[errors]} err"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bars: dict[str, TaskID] = {}
        errors: dict[str, int] = {}

        def on_start(run_id: str, spec: TaskSpec) -> None:
            bars[run_id] = progress.add_task(spec.name, total=len(spec.samples), errors=0)
            errors[run_id] = 0

        def on_result(result: SampleResult) -> None:
            bar = bars[result.run_id]
            progress.advance(bar)
            if result.error:
                errors[result.run_id] += 1
                progress.update(bar, errors=errors[result.run_id])

        completed = asyncio.run(
            run_config(
                cfg,
                only_tasks=task,
                limit=limit,
                label=label,
                resume=resume,
                on_task_start=on_start,
                on_result=on_result,
            )
        )
        # resumed runs skip already-completed samples; show their bars as full
        for spec_total, bar in ((len(c.task.samples), bars[c.run_id]) for c in completed):
            progress.update(bar, completed=spec_total)

    if not completed:
        console.print("[yellow]No tasks matched.[/yellow]")
        raise typer.Exit(1)

    table = Table(title="Results", show_lines=False)
    for col in ("run id", "primary metric", "score", "95% CI", "errors", "p50 latency", "cost"):
        table.add_column(col, overflow="fold")
    store = ResultStore(cfg.store_path)
    try:
        _summarise_runs(completed, store, table, report_dir)
    finally:
        store.close()
    console.print(table)


def _summarise_runs(
    completed: list[CompletedRun], store: ResultStore, table: Table, report_dir: Path | None
) -> None:
    for done in completed:
        metric = done.task.primary_metric or ""
        m = done.summary.metrics.get(metric)
        score = "–" if m is None or math.isnan(m.mean) else f"{100 * m.mean:.1f}%"
        ci = (
            "–"
            if m is None or math.isnan(m.mean)
            else f"[{100 * m.ci_low:.1f}, {100 * m.ci_high:.1f}]"
        )
        table.add_row(
            done.run_id,
            metric,
            score,
            ci,
            str(done.summary.n_errors),
            f"{done.summary.latency.p50_ms:.0f} ms",
            f"${done.summary.cost_usd:.4f}",
        )
        if report_dir:
            path = report_dir / f"{done.run_id}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_run(store.get_run(done.run_id), store), encoding="utf-8")


@app.command()
def report(
    run_id: str,
    db: DbOption = DEFAULT_DB,
    fmt: Annotated[Literal["md", "json"], typer.Option("--format")] = "md",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Show the report for a run."""
    with _open_store(db) as store:
        record = store.get_run(run_id)
        if fmt == "json":
            text = json.dumps(record.model_dump(mode="json"), indent=2)
            if output:
                output.write_text(text, encoding="utf-8")
            else:
                typer.echo(text)
            return
        _emit(render_run(record, store), output)


@app.command()
def compare(
    baseline: str,
    candidate: str,
    db: DbOption = DEFAULT_DB,
    alpha: Annotated[float, typer.Option(help="Significance level.")] = 0.05,
    fmt: Annotated[Literal["md", "json"], typer.Option("--format")] = "md",
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Paired statistical comparison of two runs on the same task."""
    with _open_store(db) as store:
        result = compare_runs(store, baseline, candidate, alpha=alpha)
    if fmt == "json":
        typer.echo(result.model_dump_json(indent=2))
        return
    _emit(render_comparison(result), output)


@app.command()
def gate(
    candidate: str,
    baseline: Annotated[
        str | None, typer.Option(help="Baseline run id to compare against.")
    ] = None,
    metric: Annotated[
        str | None, typer.Option(help="Defaults to the task's primary metric.")
    ] = None,
    max_drop: Annotated[
        float, typer.Option(help="Tolerated absolute drop, e.g. 0.02 = 2 pp.")
    ] = 0.0,
    min_score: Annotated[float | None, typer.Option(help="Absolute floor for the metric.")] = None,
    max_error_rate: Annotated[float, typer.Option(help="Max fraction of errored samples.")] = 0.05,
    alpha: float = 0.05,
    mode: Annotated[Literal["significant", "non_inferiority"], typer.Option()] = "significant",
    db: DbOption = DEFAULT_DB,
) -> None:
    """Exit non-zero if CANDIDATE fails the quality policy. Built for CI."""
    policy = GatePolicy(
        metric=metric,
        max_drop=max_drop,
        alpha=alpha,
        min_score=min_score,
        max_error_rate=max_error_rate,
        mode=mode,
    )
    with _open_store(db) as store:
        result = evaluate_gate(store, candidate, policy, baseline_id=baseline)
    _emit(render_gate(result), None)
    raise typer.Exit(0 if result.passed else 1)


@app.command()
def calibrate(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    labels: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Human-labelled JSONL.")
    ],
    scorer: Annotated[
        str | None, typer.Option(help="Name of the llm_judge scorer whose rubric to use.")
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Measure LLM-judge agreement with human labels (kappa, Spearman, bias)."""
    cfg = load_config(config)
    if cfg.judge is None:
        console.print("[red]Config has no 'judge:' block.[/red]")
        raise typer.Exit(2)
    options: dict[str, object] = {}
    for task_cfg in cfg.tasks:
        for spec in task_cfg.scorers or []:
            if spec.type == "llm_judge" and (scorer is None or spec.name == scorer):
                options = spec.options
                break
    judge_provider = build_provider(cfg.judge)
    judge = LLMJudge("llm_judge", judge=judge_provider, **options)

    async def go() -> str:
        try:
            result = await run_calibration(
                judge, load_labels(labels), concurrency=cfg.execution.concurrency
            )
        finally:
            await judge_provider.aclose()
        return render_calibration(result)

    _emit(asyncio.run(go()), output)


@app.command()
def runs(
    db: DbOption = DEFAULT_DB,
    task: Annotated[str | None, typer.Option(help="Filter by task.")] = None,
) -> None:
    """List recorded runs."""
    table = Table()
    for col in ("run id", "task", "model", "status", "primary", "score", "n"):
        table.add_column(col, overflow="fold")
    with _open_store(db) as store:
        records = store.list_runs(task)
    for r in records:
        metrics = (r.summary or {}).get("metrics", {})
        primary = metrics.get(r.primary_metric or "", {})
        mean = primary.get("mean")
        table.add_row(
            r.run_id,
            r.task,
            r.model,
            r.status,
            r.primary_metric or "–",
            "–" if mean is None or math.isnan(mean) else f"{100 * mean:.1f}%",
            str((r.summary or {}).get("n_samples", "–")),
        )
    console.print(table)


@app.command()
def tasks() -> None:
    """List built-in benchmark tasks."""
    table = Table()
    for col in ("task", "samples (default)", "primary metric", "description"):
        table.add_column(col)
    for name in BUILTIN_TASKS:
        spec = load_builtin(name)
        table.add_row(name, str(len(spec.samples)), spec.primary_metric or "–", spec.description)
    console.print(table)


@app.command()
def export(run_id: str, output: Path, db: DbOption = DEFAULT_DB) -> None:
    """Export a run's per-sample results as JSONL."""
    with _open_store(db) as store:
        results = store.get_results(run_id)
    with output.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(r.model_dump_json() + "\n")
    console.print(f"exported {len(results)} results to {output}")


if __name__ == "__main__":  # pragma: no cover
    app()
