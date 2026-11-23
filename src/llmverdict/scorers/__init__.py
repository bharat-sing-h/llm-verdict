from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field, model_validator

from llmverdict.providers.base import Provider
from llmverdict.scorers.base import Scorer
from llmverdict.scorers.judge import LLMJudge
from llmverdict.scorers.safety import CanaryLeak, Refusal
from llmverdict.scorers.structured import JSONFieldAccuracy, JSONSchemaScorer
from llmverdict.scorers.text import ExactMatch, FinalAnswer, Includes, RegexMatch, TokenF1

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


ScorerT = TypeVar("ScorerT", bound=type[Scorer])


def register_scorer(cls: ScorerT) -> ScorerT:
    """Class decorator that makes a custom scorer available to configs by its ``type_name``."""
    if cls.type_name in SCORERS and SCORERS[cls.type_name] is not cls:
        raise ValueError(f"scorer type {cls.type_name!r} is already registered")
    SCORERS[cls.type_name] = cls
    return cls


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


__all__ = ["SCORERS", "Scorer", "ScorerSpec", "build_scorer", "register_scorer"]
