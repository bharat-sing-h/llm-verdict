from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from llmverdict.calibration import LabeledExample, agreement_report, calibrate, load_labels
from llmverdict.providers import MockProvider, MockRule
from llmverdict.report import render_calibration
from llmverdict.scorers.judge import LLMJudge


def ex(i: int, human: int, response: str = "r") -> LabeledExample:
    return LabeledExample(id=f"e{i}", question="q", response=response, human_score=human)


def test_agreement_report_perfect_and_lenient() -> None:
    perfect = [(ex(i, h), h, "") for i, h in enumerate([1, 2, 3, 4, 5, 5, 4, 2])]
    report = agreement_report(perfect, low=1, high=5, threshold=4)
    assert report.exact_agreement == 1.0
    assert report.quadratic_kappa == pytest.approx(1.0)
    assert report.binary_kappa == pytest.approx(1.0)
    assert report.leniency_bias == 0
    assert report.worst_disagreements == []

    # A judge that always adds one point: still ranks perfectly, but is lenient and its
    # pass/fail decisions are wrong at the threshold - exactly what calibration should expose.
    lenient = [
        (ex(i, h), min(5, h + 1), "too nice") for i, h in enumerate([1, 2, 3, 3, 2, 1, 3, 2])
    ]
    report = agreement_report(lenient, low=1, high=5, threshold=4)
    assert report.spearman == pytest.approx(1.0)
    assert report.leniency_bias == pytest.approx(1.0)
    assert report.exact_agreement == 0
    assert report.judge_precision == 0
    assert math.isnan(report.judge_recall)  # humans passed nothing
    assert report.confusion["3"]["4"] == 3
    assert len(report.worst_disagreements) == 5


async def test_calibrate_with_mock_judge(tmp_path: Path) -> None:
    rows = [
        {"question": "2+2?", "reference": "4", "response": "4", "human_score": 5},
        {"question": "2+2?", "reference": "4", "response": "5", "human_score": 1},
        {"question": "2+2?", "reference": "4", "response": "four-ish", "human_score": 3},
        {"question": "2+2?", "reference": "4", "response": "unparseable", "human_score": 2},
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    examples = load_labels(path)
    assert [e.id for e in examples] == ["0", "1", "2", "3"]

    judge_model = MockProvider(
        rules=[
            MockRule(pattern=r"evaluate\n4\n", response='{"score": 5, "reasoning": "right"}'),
            MockRule(pattern=r"evaluate\n5\n", response='{"score": 2, "reasoning": "wrong"}'),
            MockRule(pattern=r"evaluate\nfour-ish", response='{"score": 3, "reasoning": "meh"}'),
        ],
        default="I cannot decide",
    )
    report = await calibrate(LLMJudge(judge=judge_model), examples)
    assert report.n == 3
    assert report.parse_failures == 1
    assert report.binary_accuracy == 1.0
    assert report.spearman == pytest.approx(1.0)
    md = render_calibration(report)
    assert "Quadratic-weighted" in md
    assert "Largest disagreements" in md
