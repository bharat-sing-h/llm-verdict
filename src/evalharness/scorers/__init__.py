from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from evalharness.providers.base import Provider
from evalharness.scorers.base import Scorer
from evalharness.scorers.judge import LLMJudge
from evalharness.scorers.safety import CanaryLeak, Refusal
from evalharness.scorers.structured import JSONFieldAccuracy, JSONSchemaScorer
from evalharness.scorers.text import ExactMatch, FinalAnswer, Includes, RegexMatch, TokenF1

SCORERS: dict[str, type[Scorer]] = {
    cls.type_name: cls
    for cls in (
        ExactMatch,
        Includes,
        RegexMatch,
        FinalAnswer,
        TokenF1,
        JSONSchemaScorer,
        JSONFieldAccuracy,
        CanaryLeak,
        Refusal,
        LLMJudge,
    )
}


class ScorerSpec(BaseModel):
    """``"exact_match"`` or ``{type: llm_judge, name: helpfulness, options: {...}}``."""

    type: str
    name: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    only_tags: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        return {"type": value} if isinstance(value, str) else value


def build_scorer(spec: ScorerSpec | str, *, judge: Provider | None = None) -> Scorer:
    spec = ScorerSpec.model_validate(spec)
    try:
        cls = SCORERS[spec.type]
    except KeyError:
        raise ValueError(f"unknown scorer {spec.type!r}; available: {sorted(SCORERS)}") from None
    scorer = (
        LLMJudge(spec.name, judge=judge, **spec.options)
        if cls is LLMJudge
        else cls(spec.name, **spec.options)
    )
    scorer.only_tags = spec.only_tags
    return scorer


__all__ = ["SCORERS", "Scorer", "ScorerSpec", "build_scorer"]
