"""Measure how well an LLM judge agrees with human labels before trusting it.

Reported statistics and why each one matters:

* **quadratic-weighted kappa** – headline agreement on the ordinal scale, chance-corrected and
  lenient on near misses (4 vs 5) but harsh on far misses (1 vs 5).
* **binary kappa / precision / recall at the pass threshold** – what actually matters if the
  judge feeds a pass/fail metric or a CI gate.
* **Spearman** – does the judge *rank* responses like humans do (enough for A/B comparisons)?
* **leniency bias** – mean(judge) - mean(human); judges are typically too generous.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmverdict.scorers.judge import JudgeParseError, LLMJudge
from llmverdict.stats import cohen_kappa, mean, spearman


class LabeledExample(BaseModel):
    id: str
    question: str
    response: str
    human_score: int
    reference: Any = None


class Disagreement(BaseModel):
    id: str
    human: int
    judge: int
    reasoning: str


class CalibrationReport(BaseModel):
    n: int
    parse_failures: int
    exact_agreement: float
    kappa: float
    quadratic_kappa: float
    spearman: float
    binary_accuracy: float
    binary_kappa: float
    judge_precision: float
    judge_recall: float
    human_mean: float
    judge_mean: float
    leniency_bias: float
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    worst_disagreements: list[Disagreement] = Field(default_factory=list)


def load_labels(path: str | Path) -> list[LabeledExample]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if line.strip():
                data = json.loads(line)
                data.setdefault("id", str(i))
                rows.append(LabeledExample.model_validate(data))
    return rows


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def agreement_report(
    pairs: list[tuple[LabeledExample, int, str]],
    *,
    low: int,
    high: int,
    threshold: float,
    parse_failures: int = 0,
) -> CalibrationReport:
    human = [ex.human_score for ex, _, _ in pairs]
    judge = [score for _, score, _ in pairs]
    labels = list(range(low, high + 1))
    h_pass = [h >= threshold for h in human]
    j_pass = [j >= threshold for j in judge]
    tp = sum(1 for h, j in zip(h_pass, j_pass, strict=True) if h and j)
    confusion = {
        str(h): {
            str(j): sum(1 for a, b in zip(human, judge, strict=True) if a == h and b == j)
            for j in labels
        }
        for h in labels
    }
    worst = sorted(pairs, key=lambda p: abs(p[0].human_score - p[1]), reverse=True)
    return CalibrationReport(
        n=len(pairs),
        parse_failures=parse_failures,
        exact_agreement=_safe_div(
            sum(h == j for h, j in zip(human, judge, strict=True)), len(pairs)
        ),
        kappa=cohen_kappa(human, judge, labels=labels),
        quadratic_kappa=cohen_kappa(human, judge, labels=labels, weights="quadratic"),
        spearman=spearman([float(h) for h in human], [float(j) for j in judge]),
        binary_accuracy=_safe_div(
            sum(a == b for a, b in zip(h_pass, j_pass, strict=True)), len(pairs)
        ),
        binary_kappa=cohen_kappa(h_pass, j_pass, labels=[False, True]),
        judge_precision=_safe_div(tp, sum(j_pass)),
        judge_recall=_safe_div(tp, sum(h_pass)),
        human_mean=mean([float(h) for h in human]),
        judge_mean=mean([float(j) for j in judge]),
        leniency_bias=mean([float(j) for j in judge]) - mean([float(h) for h in human]),
        confusion=confusion,
        worst_disagreements=[
            Disagreement(id=ex.id, human=ex.human_score, judge=score, reasoning=why)
            for ex, score, why in worst[:5]
            if ex.human_score != score
        ],
    )


async def calibrate(
    judge: LLMJudge, examples: list[LabeledExample], *, concurrency: int = 8
) -> CalibrationReport:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(ex: LabeledExample) -> tuple[LabeledExample, int, str] | None:
        async with semaphore:
            try:
                verdict = await judge.judge_one(ex.question, ex.reference, ex.response)
            except JudgeParseError:
                return None
        return ex, verdict.score, verdict.reasoning

    outcomes = await asyncio.gather(*(one(ex) for ex in examples))
    pairs = [o for o in outcomes if o is not None]
    return agreement_report(
        pairs,
        low=judge.low,
        high=judge.high,
        threshold=judge.pass_threshold,
        parse_failures=len(outcomes) - len(pairs),
    )
