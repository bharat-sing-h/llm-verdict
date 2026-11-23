from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evalharness.types import Message, Sample


class TaskSpec(BaseModel):
    name: str
    description: str = ""
    samples: list[Sample]
    default_scorers: list[Any] = Field(default_factory=list)
    #: scorer whose pass rate is the task's headline number in reports and gates
    primary_metric: str | None = None


TaskFactory = Callable[..., TaskSpec]


def load_jsonl(path: str | Path, *, name: str | None = None) -> TaskSpec:
    """Load a custom dataset.

    Each line is an object with either ``input`` (a user string, optionally with ``system``) or a
    full ``messages`` list, plus optional ``id``, ``reference``, ``tags`` and ``metadata``.
    Missing ids are derived from content, so they stay stable across file re-orderings – which
    matters because paired comparisons align samples by id.
    """
    path = Path(path)
    samples: list[Sample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if "messages" in row:
                messages = [Message.model_validate(m) for m in row["messages"]]
            elif "input" in row:
                messages = []
                if row.get("system"):
                    messages.append(Message(role="system", content=row["system"]))
                messages.append(Message(role="user", content=str(row["input"])))
            else:
                raise ValueError(f"{path}:{lineno}: row needs 'input' or 'messages'")
            sample_id = str(row.get("id") or _content_id(messages))
            if sample_id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate sample id {sample_id!r}")
            seen.add(sample_id)
            samples.append(
                Sample(
                    id=sample_id,
                    messages=messages,
                    reference=row.get("reference"),
                    tags=list(row.get("tags", [])),
                    metadata=dict(row.get("metadata", {})),
                )
            )
    return TaskSpec(
        name=name or path.stem, description=f"custom dataset {path.name}", samples=samples
    )


def _content_id(messages: list[Message]) -> str:
    blob = json.dumps([m.model_dump() for m in messages], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
