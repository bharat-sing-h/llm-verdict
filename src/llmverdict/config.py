"""Declarative run configuration (YAML) with ``${ENV_VAR}`` / ``${ENV_VAR:-default}`` expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from llmverdict.engine.runner import ExecutionConfig
from llmverdict.providers import ProviderConfig
from llmverdict.providers.base import GenerationParams
from llmverdict.scorers import ScorerSpec
from llmverdict.tasks import TaskSpec, load_builtin, load_jsonl

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class TaskConfig(BaseModel):
    builtin: str | None = None
    path: str | None = None
    name: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    limit: int | None = None
    scorers: list[ScorerSpec] | None = None

    @model_validator(mode="after")
    def _one_source(self) -> TaskConfig:
        if (self.builtin is None) == (self.path is None):
            raise ValueError("each task needs exactly one of 'builtin' or 'path'")
        return self

    @property
    def display_name(self) -> str:
        return self.name or self.builtin or Path(self.path or "task").stem

    def load(self, base_dir: Path | None = None) -> TaskSpec:
        if self.builtin:
            spec = load_builtin(self.builtin, **self.options)
        else:
            path = Path(self.path or "")
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            spec = load_jsonl(path)
        updates: dict[str, Any] = {"name": self.display_name}
        if self.limit is not None:
            updates["samples"] = spec.samples[: self.limit]
        if self.scorers is not None:
            updates["default_scorers"] = [s.model_dump() for s in self.scorers]
            if spec.primary_metric not in {s.name or s.type for s in self.scorers}:
                updates["primary_metric"] = self.scorers[0].name or self.scorers[0].type
        elif spec.primary_metric is None and spec.default_scorers:
            first = ScorerSpec.model_validate(spec.default_scorers[0])
            updates["primary_metric"] = first.name or first.type
        return spec.model_copy(update=updates)


class RunConfig(BaseModel):
    model: ProviderConfig
    judge: ProviderConfig | None = None
    params: GenerationParams = Field(default_factory=GenerationParams)
    tasks: list[TaskConfig]
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    store_path: str = ".llmverdict/results.db"
    cache_path: str = ".llmverdict/cache.db"
    #: modules imported before scorers are built; they register custom scorers
    plugins: list[str] = Field(default_factory=list)

    #: directory relative dataset paths are resolved against (the config file's directory)
    base_dir: Path | None = Field(default=None, exclude=True)


def expand_env(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is None:
            if default is None:
                raise KeyError(f"environment variable {name} is not set (use ${{{name}:-default}})")
            return default
        return value

    return _ENV.sub(replace, text)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = yaml.safe_load(expand_env(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a mapping")
    cfg = RunConfig.model_validate(raw)
    cfg.base_dir = path.parent
    return cfg
