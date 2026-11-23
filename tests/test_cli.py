from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmverdict.cli import app
from llmverdict.config import expand_env, load_config
from llmverdict.harness import make_run_id

runner = CliRunner(env={"COLUMNS": "220"})

CONFIG = """
model:
  provider: mock
  model: ${MOCK_MODEL:-calc-v1}
  rules:
    - pattern: "What is 1\\\\+1"
      response: "Answer: 2"
  default: "Answer: 0"
  pricing: {input_per_mtok: 1.0, output_per_mtok: 2.0}
judge:
  provider: mock
  model: judge
  default: '{"score": 4, "reasoning": "fine"}'
tasks:
  - path: data.jsonl
    name: arith
    scorers:
      - final_answer
      - {type: llm_judge, name: quality, options: {rubric: "Be strict."}}
  - builtin: prompt_injection
    options: {n_variants: 1}
    limit: 5
store_path: STORE/results.db
cache_path: STORE/cache.db
"""


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [
        {"id": "a", "input": "What is 1+1?", "reference": 2},
        {"id": "b", "input": "What is 2+2?", "reference": 4},
    ]
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "cfg.yaml").write_text(
        CONFIG.replace("STORE", str(tmp_path / "store")), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLMVERDICT_DB", str(tmp_path / "store" / "results.db"))
    return tmp_path


def test_full_cli_workflow(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(app, ["run", "cfg.yaml", "--label", "base", "--report-dir", "reports"])
    assert result.exit_code == 0, result.output
    assert (workdir / "reports" / "base__arith.md").exists()

    monkeypatch.setenv("MOCK_MODEL", "calc-v2")
    result = runner.invoke(app, ["run", "cfg.yaml", "--label", "cand", "-t", "arith"])
    assert result.exit_code == 0, result.output

    listing = runner.invoke(app, ["runs"])
    assert "base__prompt_injection" in listing.output
    assert "cand__arith" in listing.output

    md = runner.invoke(app, ["report", "base__arith"])
    assert md.exit_code == 0
    assert "final_answer" in md.output

    as_json = runner.invoke(app, ["report", "base__arith", "--format", "json"])
    data = json.loads(as_json.output)
    assert data["summary"]["metrics"]["final_answer"]["mean"] == 0.5
    assert data["summary"]["metrics"]["quality"]["mean"] == 0.75

    out = workdir / "cmp.md"
    cmp = runner.invoke(app, ["compare", "base__arith", "cand__arith", "-o", str(out)])
    assert cmp.exit_code == 0
    assert "Comparison on `arith`" in out.read_text(encoding="utf-8")
    cmp_json = runner.invoke(app, ["compare", "base__arith", "cand__arith", "--format", "json"])
    assert json.loads(cmp_json.output)["task"] == "arith"

    summary_file = workdir / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    ok = runner.invoke(app, ["gate", "cand__arith", "--baseline", "base__arith"])
    assert ok.exit_code == 0, ok.output
    assert "Eval gate passed" in summary_file.read_text(encoding="utf-8")
    bad = runner.invoke(app, ["gate", "cand__arith", "--min-score", "0.9"])
    assert bad.exit_code == 1

    export_path = workdir / "export.jsonl"
    exp = runner.invoke(app, ["export", "base__arith", str(export_path)])
    assert exp.exit_code == 0
    assert len(export_path.read_text(encoding="utf-8").splitlines()) == 2


def test_cli_calibrate(workdir: Path) -> None:
    labels = workdir / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps({"question": "q", "response": r, "human_score": h})
            for r, h in [("a", 4), ("b", 5), ("c", 2)]
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["calibrate", "cfg.yaml", str(labels), "--scorer", "quality"])
    assert result.exit_code == 0, result.output
    assert "Judge calibration" in result.output


def test_cli_errors(workdir: Path, tmp_path: Path) -> None:
    assert runner.invoke(app, ["report", "nope"]).exit_code == 2
    runner.invoke(app, ["run", "cfg.yaml", "--label", "x", "-t", "arith"])
    assert runner.invoke(app, ["report", "nope"]).exit_code == 2
    assert runner.invoke(app, ["compare", "x__arith", "nope"]).exit_code == 2
    assert runner.invoke(app, ["gate", "nope"]).exit_code == 2
    assert runner.invoke(app, ["run", "cfg.yaml", "-t", "does-not-exist"]).exit_code == 1
    assert runner.invoke(app, ["runs", "--db", str(tmp_path / "missing.db")]).exit_code == 2
    no_judge = workdir / "nojudge.yaml"
    no_judge.write_text(
        "model: {provider: mock}\ntasks: [{builtin: math_reasoning}]\n", encoding="utf-8"
    )
    assert runner.invoke(app, ["calibrate", str(no_judge), "cfg.yaml"]).exit_code == 2


def test_cli_misc() -> None:
    assert "llmverdict" in runner.invoke(app, ["--version"]).output
    tasks = runner.invoke(app, ["tasks"])
    assert "prompt_injection" in tasks.output


def test_offline_demo_config_runs_red_team_against_toy_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    db = tmp_path / "r.db"
    monkeypatch.setenv("LLMVERDICT_DB", str(db))
    cfg = (root / "examples" / "offline_demo.yaml").read_text(encoding="utf-8")
    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text(cfg + f"\nstore_path: {db}\n", encoding="utf-8")
    assert runner.invoke(app, ["run", str(cfg_path), "--label", "naive"]).exit_code == 0
    monkeypatch.setenv("APP", "guarded_bot")
    assert runner.invoke(app, ["run", str(cfg_path), "--label", "guarded"]).exit_code == 0
    gate = runner.invoke(
        app, ["gate", "naive__prompt_injection", "--baseline", "guarded__prompt_injection"]
    )
    assert gate.exit_code == 1  # the unguarded bot is a significant security regression
    assert "significant regression" in gate.output


def test_config_env_expansion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FOO", "bar")
    assert expand_env("${FOO} ${MISSING:-dflt} ${EMPTY:-}") == "bar dflt "
    with pytest.raises(KeyError, match="MISSING"):
        expand_env("${MISSING}")
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)
    both = tmp_path / "both.yaml"
    both.write_text("model: {}\ntasks: [{builtin: x, path: y}]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_config(both)


def test_make_run_id() -> None:
    assert make_run_id("My Task", "org/model:7b", "Baseline v1") == "baseline-v1__my-task"
    auto = make_run_id("t", "org/Model:7B")
    assert auto.startswith("org-model-7b__t__")
