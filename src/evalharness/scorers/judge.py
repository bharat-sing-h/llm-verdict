"""LLM-as-judge scorer.

Model-graded evaluation scales to open-ended outputs that string matching can't handle, but a
judge is itself a model with biases (leniency, verbosity preference, self-preference). Never trust
one blindly: calibrate it against human labels first (``evalharness calibrate``) and report the
agreement next to any judge-based metric.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from evalharness.engine.retry import RetryPolicy, call_with_retry
from evalharness.providers.base import GenerationParams, Provider
from evalharness.scorers.base import Scorer
from evalharness.scorers.structured import extract_json
from evalharness.types import Message, Sample, Score

DEFAULT_RUBRIC = (
    "Judge whether the response correctly and completely answers the question. "
    "Use the reference answer as ground truth when provided. Ignore style and length; "
    "penalise factual errors, missing key information and unsupported claims."
)

JUDGE_SYSTEM = (
    "You are a meticulous, impartial evaluator of AI assistant responses. You follow the rubric "
    "exactly, do not reward verbosity, and do not let the response's confidence sway you."
)

JUDGE_TEMPLATE = """\
## Question
{question}

## Reference answer
{reference}

## Response to evaluate
{response}

## Rubric
{rubric}

Rate the response on an integer scale from {low} (worst) to {high} (best).
Reply with ONLY a JSON object: {{"reasoning": "<at most two sentences>", "score": <integer>}}"""

_SCORE_FALLBACK = re.compile(r"score\W{0,5}(\d+)", re.I)


@dataclass(frozen=True)
class Verdict:
    score: int
    reasoning: str


class JudgeParseError(ValueError):
    pass


def parse_verdict(text: str, low: int, high: int) -> Verdict:
    try:
        data = extract_json(text)
        if not isinstance(data, dict):
            raise JudgeParseError("judge output is not an object")
        raw = data["score"]
        reasoning = str(data.get("reasoning", ""))
    except (ValueError, KeyError) as exc:
        match = _SCORE_FALLBACK.search(text)
        if not match:
            raise JudgeParseError(f"could not parse judge output: {text[:200]!r}") from exc
        raw, reasoning = match.group(1), text.strip()
    try:
        score = round(float(raw))
    except (TypeError, ValueError) as exc:
        raise JudgeParseError(f"non-numeric judge score: {raw!r}") from exc
    if not low <= score <= high:
        raise JudgeParseError(f"judge score {score} outside [{low}, {high}]")
    return Verdict(score, reasoning)


class LLMJudge(Scorer):
    type_name = "llm_judge"

    def __init__(
        self,
        name: str | None = None,
        *,
        judge: Provider | None = None,
        **options: Any,
    ) -> None:
        super().__init__(name, **options)
        if judge is None:
            raise ValueError("llm_judge scorer needs a judge provider (add a 'judge:' block)")
        self.judge = judge
        self.low = int(options.get("scale_min", 1))
        self.high = int(options.get("scale_max", 5))
        self.pass_threshold = float(options.get("pass_threshold", 4))
        self.rubric = str(options.get("rubric", DEFAULT_RUBRIC))
        self.params = GenerationParams(
            temperature=0.0, max_tokens=int(options.get("max_tokens", 300))
        )
        self.retry = RetryPolicy(max_retries=int(options.get("max_retries", 3)))

    def build_messages(self, question: str, reference: Any, response: str) -> list[Message]:
        ref = reference if isinstance(reference, str) else json.dumps(reference, default=str)
        prompt = JUDGE_TEMPLATE.format(
            question=question,
            reference=ref if reference is not None else "(none provided)",
            response=response,
            rubric=self.rubric,
            low=self.low,
            high=self.high,
        )
        return [Message(role="system", content=JUDGE_SYSTEM), Message(role="user", content=prompt)]

    async def judge_one(self, question: str, reference: Any, response: str) -> Verdict:
        messages = self.build_messages(question, reference, response)
        reply = await call_with_retry(
            lambda: self.judge.generate(messages, self.params), self.retry
        )
        return parse_verdict(reply.text, self.low, self.high)

    async def score(self, sample: Sample, output: str) -> Score:
        try:
            verdict = await self.judge_one(sample.prompt_text, sample.reference, output)
        except JudgeParseError as exc:
            return Score(name=self.name, value=None, explanation=f"judge error: {exc}")
        value = (verdict.score - self.low) / (self.high - self.low)
        return Score(
            name=self.name,
            value=value,
            passed=verdict.score >= self.pass_threshold,
            explanation=verdict.reasoning,
            metadata={"raw_score": verdict.score},
        )
