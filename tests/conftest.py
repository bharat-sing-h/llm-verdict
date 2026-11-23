from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from llmverdict.store import ResultStore
from llmverdict.types import Message, ModelResponse, Sample, SampleResult, Score


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ResultStore]:
    s = ResultStore(tmp_path / "results.db")
    yield s
    s.close()


def make_sample(
    sid: str,
    user: str = "hi",
    reference: object = None,
    tags: list[str] | None = None,
    system: str | None = None,
) -> Sample:
    messages = [Message(role="system", content=system)] if system else []
    messages.append(Message(role="user", content=user))
    return Sample(id=sid, messages=messages, reference=reference, tags=tags or [])


def add_run(
    store: ResultStore,
    run_id: str,
    values: dict[str, float | None],
    *,
    metric: str = "acc",
    task: str = "t",
    model: str = "m",
    errors: set[str] | None = None,
    tags: dict[str, list[str]] | None = None,
) -> None:
    """Insert a finished run whose samples have the given metric values."""
    from llmverdict.summary import summarize

    store.create_run(
        run_id, task=task, model=model, provider="mock", config={}, primary_metric=metric
    )
    results = []
    for sid, value in values.items():
        if errors and sid in errors:
            r = SampleResult(run_id=run_id, sample_id=sid, error="boom")
        else:
            r = SampleResult(
                run_id=run_id,
                sample_id=sid,
                response=ModelResponse(text="x", model=model, latency_ms=10.0),
                scores=[
                    Score(name=metric, value=value, passed=None if value is None else value >= 0.5)
                ],
                tags=(tags or {}).get(sid, []),
            )
        store.save_result(r)
        results.append(r)
    store.finish_run(run_id, summarize(results).model_dump(mode="json"))
