from __future__ import annotations

from typing import Any

from llmverdict.tasks import json_extraction, math_reasoning, prompt_injection
from llmverdict.tasks.base import TaskFactory, TaskSpec, load_jsonl

BUILTIN_TASKS: dict[str, TaskFactory] = {
    "math_reasoning": math_reasoning.build,
    "json_extraction": json_extraction.build,
    "prompt_injection": prompt_injection.build,
}


def load_builtin(name: str, **options: Any) -> TaskSpec:
    try:
        factory = BUILTIN_TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown builtin task {name!r}; available: {sorted(BUILTIN_TASKS)}"
        ) from None
    return factory(**options)


__all__ = ["BUILTIN_TASKS", "TaskSpec", "load_builtin", "load_jsonl"]
