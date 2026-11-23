from __future__ import annotations

import datetime as dt
import json
import random
import re
from pathlib import Path

import jsonschema
import pytest

from llmverdict.tasks import BUILTIN_TASKS, load_builtin, load_jsonl
from llmverdict.tasks import math_reasoning as mr
from llmverdict.tasks.json_extraction import SCHEMA
from llmverdict.tasks.prompt_injection import ATTACKS


@pytest.mark.parametrize("name", sorted(BUILTIN_TASKS))
def test_builtin_tasks_are_deterministic_and_have_unique_ids(name: str) -> None:
    a, b = load_builtin(name, seed=11), load_builtin(name, seed=11)
    assert [s.model_dump() for s in a.samples] == [s.model_dump() for s in b.samples]
    ids = [s.id for s in a.samples]
    assert len(ids) == len(set(ids))
    assert a.primary_metric
    assert a.default_scorers
    c = load_builtin(name, seed=12)
    assert [s.messages for s in a.samples] != [s.messages for s in c.samples]


def test_unknown_builtin() -> None:
    with pytest.raises(ValueError, match="unknown builtin"):
        load_builtin("nope")


def test_math_task_covers_all_generators_and_levels() -> None:
    spec = mr.build(n=36, seed=1)
    tags = {t for s in spec.samples for t in s.tags}
    assert {f"type:{k}" for k in mr.GENERATORS} <= tags
    assert {"level:1", "level:2", "level:3"} <= tags


def test_weekday_generator_ground_truth() -> None:
    rng = random.Random(5)
    for _ in range(50):
        text, answer = mr.weekday_offset(rng, 2)
        m = re.search(r"Today is \w+, (\w+) (\d+), (\d+)\. .* (\d+) days", text)
        assert m
        start = dt.datetime.strptime(f"{m[1]} {m[2]} {m[3]}", "%B %d %Y").date()
        target = start + dt.timedelta(days=int(m[4]))
        assert answer == target.strftime("%A")
        assert text.startswith(f"Today is {start.strftime('%A')}")


def test_leftover_and_work_rate_ground_truth() -> None:
    rng = random.Random(9)
    for _ in range(50):
        text, answer = mr.leftover(rng, 3)
        students, group = map(int, re.findall(r"\d+", text)[:2])
        assert answer == students % group
        assert 0 < answer < group
        text, answer = mr.work_rate(rng, 2)
        nums = list(map(int, re.findall(r"\d+", text)))
        rate, hours, minutes = nums[:3]
        downtime = nums[3] if len(nums) > 3 else 0
        assert answer == rate * (hours * 60 + minutes - downtime)


def test_json_extraction_references_are_self_consistent() -> None:
    spec = load_builtin("json_extraction", n=40, seed=3)
    for s in spec.samples:
        expected = s.reference["expected"]
        jsonschema.validate(expected, SCHEMA)
        assert expected["total"] == pytest.approx(
            expected["quantity"] * expected["unit_price"], abs=0.01
        )
        user = s.messages[-1].content
        assert expected["customer"] in user
        assert expected["product"] in user
        assert ("express shipping" in user) == expected["express_shipping"]


def test_prompt_injection_structure() -> None:
    spec = load_builtin("prompt_injection", n_variants=2, seed=0)
    attacks = [s for s in spec.samples if "kind:attack" in s.tags]
    controls = [s for s in spec.samples if "kind:control" in s.tags]
    assert len(attacks) == 2 * len(ATTACKS)
    assert len(controls) == 4
    canaries = set()
    for s in spec.samples:
        canary = s.reference["canary"]
        assert canary in s.messages[0].content  # secret is planted in the system prompt
        assert canary not in s.messages[-1].content  # ...and never in the attack itself
        canaries.add(canary)
    assert len(canaries) == len(spec.samples)  # fresh canary per sample
    assert all(s.reference["expected_fact"] in s.messages[0].content for s in controls)


def test_load_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [
        {"id": "a", "input": "2+2?", "reference": 4, "tags": ["math"], "system": "be brief"},
        {"messages": [{"role": "user", "content": "hello"}], "reference": "hi"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    spec = load_jsonl(path)
    assert spec.name == "data"
    assert [m.role for m in spec.samples[0].messages] == ["system", "user"]
    assert spec.samples[0].tags == ["math"]
    assert len(spec.samples[1].id) == 12  # content-derived id
    assert load_jsonl(path).samples[1].id == spec.samples[1].id


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("{not json}\n", "invalid JSON"),
        ('{"reference": 1}\n', "needs 'input' or 'messages'"),
        ('{"id": "x", "input": "a"}\n{"id": "x", "input": "b"}\n', "duplicate sample id"),
    ],
)
def test_load_jsonl_errors(tmp_path: Path, content: str, error: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_jsonl(path)
