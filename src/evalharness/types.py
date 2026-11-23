"""Core data model shared by every component of the harness.

Everything that crosses a component boundary (dataset -> runner -> provider -> scorer -> store)
is a pydantic model, so data is validated once at the edge and serialises losslessly to SQLite
and JSONL.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class Sample(BaseModel):
    """A single evaluation item.

    ``reference`` is intentionally untyped: a scorer decides how to interpret it (a string,
    a number, a JSON schema, a canary token, ...). ``tags`` drive per-slice breakdowns in reports.
    """

    id: str
    messages: list[Message]
    reference: Any = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        """Last user turn; handy for judges and reports."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelResponse(BaseModel):
    text: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    cached: bool = False


class Score(BaseModel):
    """Outcome of one scorer on one sample.

    ``value`` is normalised to [0, 1]. ``None`` means "not applicable" (e.g. a canary-leak scorer
    on a benign control sample) and is excluded from aggregation rather than counted as zero.
    """

    name: str
    value: float | None
    passed: bool | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleResult(BaseModel):
    run_id: str
    sample_id: str
    response: ModelResponse | None = None
    scores: list[Score] = Field(default_factory=list)
    error: str | None = None
    tags: list[str] = Field(default_factory=list)

    def score(self, name: str) -> Score | None:
        return next((s for s in self.scores if s.name == name), None)
