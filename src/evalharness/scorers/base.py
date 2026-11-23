from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evalharness.types import Sample, Score


class Scorer(ABC):
    """Turns (sample, model output) into a :class:`Score`.

    Scorers are async so model-graded scorers (LLM-as-judge) share the same interface as cheap
    deterministic ones, and the runner can schedule them uniformly.
    """

    #: registry key, e.g. ``"exact_match"``
    type_name: str = "base"

    def __init__(self, name: str | None = None, **options: Any) -> None:
        self.name = name or self.type_name
        self.options = options
        #: if set, the scorer only runs on samples carrying at least one of these tags
        self.only_tags: list[str] | None = None

    def applies_to(self, sample: Sample) -> bool:
        return self.only_tags is None or any(t in sample.tags for t in self.only_tags)

    @abstractmethod
    async def score(self, sample: Sample, output: str) -> Score:
        """Score one output. Return ``Score(value=None)`` when not applicable."""

    def _na(self, why: str) -> Score:
        return Score(name=self.name, value=None, passed=None, explanation=why)

    def _bool(self, ok: bool, explanation: str | None = None, **metadata: Any) -> Score:
        return Score(
            name=self.name,
            value=1.0 if ok else 0.0,
            passed=ok,
            explanation=explanation,
            metadata=metadata,
        )
