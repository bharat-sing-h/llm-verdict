from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llmverdict.config import load_config
from llmverdict.harness import run_config
from llmverdict.plugins import PluginError, load_plugins
from llmverdict.scorers import SCORERS, Scorer, register_scorer
from llmverdict.types import Sample, Score


def test_plugin_scorer_is_usable_from_config(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text('{"id": "a", "input": "hi"}\n', encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
model: {{provider: mock, default: "HELLO"}}
plugins: [tests.plugins.shouty]
tasks:
  - path: data.jsonl
    scorers: [shouty]
store_path: {tmp_path}/r.db
cache_path: {tmp_path}/c.db
""",
        encoding="utf-8",
    )
    [done] = asyncio.run(run_config(load_config(cfg_path), label="p"))
    assert done.summary.metrics["shouty"].mean == 1.0
    assert "shouty" in SCORERS


def test_plugin_errors() -> None:
    with pytest.raises(PluginError, match="no_such_module"):
        load_plugins(["no_such_module"])
    assert load_plugins([]) == []

    class Clash(Scorer):
        type_name = "exact_match"

        async def score(self, sample: Sample, output: str) -> Score:
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        register_scorer(Clash)
