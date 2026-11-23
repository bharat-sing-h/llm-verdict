"""High-level orchestration: config in, persisted runs out. The CLI is a thin layer over this."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from llmverdict.config import RunConfig
from llmverdict.engine.cache import ResponseCache
from llmverdict.engine.runner import Runner
from llmverdict.providers import build_provider
from llmverdict.scorers import build_scorer
from llmverdict.store import ResultStore
from llmverdict.summary import RunSummary
from llmverdict.tasks import TaskSpec
from llmverdict.types import SampleResult


@dataclass
class CompletedRun:
    run_id: str
    task: TaskSpec
    summary: RunSummary


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower() or "run"


def make_run_id(task: str, model: str, label: str | None = None) -> str:
    """Labelled runs get a stable id (so they resume / can be referenced from CI);
    unlabelled runs are timestamped so they never collide."""
    if label:
        return f"{_slug(label)}__{_slug(task)}"
    return f"{_slug(model)}__{_slug(task)}__{time.strftime('%Y%m%d-%H%M%S')}"


async def run_config(
    cfg: RunConfig,
    *,
    only_tasks: list[str] | None = None,
    limit: int | None = None,
    label: str | None = None,
    resume: bool = True,
    on_task_start: Callable[[str, TaskSpec], None] | None = None,
    on_result: Callable[[SampleResult], None] | None = None,
) -> list[CompletedRun]:
    provider = build_provider(cfg.model)
    judge = build_provider(cfg.judge) if cfg.judge else None
    store = ResultStore(cfg.store_path)
    cache = ResponseCache(cfg.cache_path) if cfg.execution.cache else None
    completed: list[CompletedRun] = []
    try:
        for task_cfg in cfg.tasks:
            if only_tasks and task_cfg.display_name not in only_tasks:
                continue
            task = task_cfg.load(cfg.base_dir if cfg.base_dir else Path.cwd())
            if limit is not None:
                task = task.model_copy(update={"samples": task.samples[:limit]})
            scorers = [build_scorer(s, judge=judge) for s in task.default_scorers]
            run_id = make_run_id(task.name, provider.model, label)
            if on_task_start:
                on_task_start(run_id, task)
            runner = Runner(
                provider,
                scorers,
                store,
                params=cfg.params,
                execution=cfg.execution,
                cache=cache,
                on_result=on_result,
            )
            snapshot = cfg.model_dump(mode="json")
            snapshot["model"].pop("rules", None)  # mock rules are noise in run metadata
            summary = await runner.run(task, run_id, config_snapshot=snapshot, resume=resume)
            completed.append(CompletedRun(run_id, task, summary))
    finally:
        await provider.aclose()
        if judge:
            await judge.aclose()
        if cache is not None:
            cache.close()
        store.close()
    return completed
