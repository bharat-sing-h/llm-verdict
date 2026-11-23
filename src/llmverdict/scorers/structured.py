"""Scorers for structured (JSON) output: format validity and field-level correctness."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import jsonschema

from llmverdict.scorers.base import Scorer
from llmverdict.scorers.text import normalize_text
from llmverdict.types import Sample, Score

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


class JSONExtractionError(ValueError):
    pass


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response.

    Order: whole string -> fenced code block -> first balanced object/array. The balanced scan is
    string-aware, so braces inside string literals don't confuse it.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for block in _FENCE.findall(text):
        try:
            return json.loads(block.strip())
        except ValueError:
            continue
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        closing = "}" if ch == "{" else "]"
        depth, in_str, escaped = 0, False, False
        for end in range(start, len(text)):
            c = text[end]
            if in_str:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == ch:
                depth += 1
            elif c == closing:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : end + 1])
                    except ValueError:
                        break
    raise JSONExtractionError("no valid JSON found in output")


def _schema_for(sample: Sample, options: dict[str, Any]) -> Any:
    if "schema" in options:
        return options["schema"]
    ref = sample.reference
    if isinstance(ref, dict) and "schema" in ref:
        return ref["schema"]
    return None


class JSONSchemaScorer(Scorer):
    """1.0 iff the output contains JSON that validates against the schema.

    ``strict: true`` additionally requires the *entire* output to be JSON (no prose, no fences),
    which is what a downstream parser in production would demand.
    """

    type_name = "json_schema"

    async def score(self, sample: Sample, output: str) -> Score:
        schema = _schema_for(sample, self.options)
        try:
            data = json.loads(output) if self.options.get("strict") else extract_json(output)
        except ValueError:
            return self._bool(False, "output is not valid JSON", failure="parse")
        if schema is None:
            return self._bool(True)
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            return self._bool(False, f"schema violation at {path}: {exc.message}", failure="schema")
        return self._bool(True)


def _values_equal(expected: Any, got: Any, tol: float) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected is got
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(float(expected), float(got), rel_tol=0.0, abs_tol=tol)
    if isinstance(expected, str) and isinstance(got, str):
        return normalize_text(expected) == normalize_text(got)
    if isinstance(expected, list) and isinstance(got, list):
        return len(expected) == len(got) and all(
            _values_equal(e, g, tol) for e, g in zip(expected, got, strict=True)
        )
    if isinstance(expected, dict) and isinstance(got, dict):
        return expected.keys() == got.keys() and all(
            _values_equal(v, got[k], tol) for k, v in expected.items()
        )
    return bool(expected == got)


class JSONFieldAccuracy(Scorer):
    """Fraction of expected top-level fields whose values are correct (partial credit)."""

    type_name = "json_fields"

    async def score(self, sample: Sample, output: str) -> Score:
        ref = sample.reference
        expected = ref.get("expected") if isinstance(ref, dict) else None
        if not isinstance(expected, dict) or not expected:
            return self._na("no expected fields")
        try:
            data = extract_json(output)
        except ValueError:
            return Score(name=self.name, value=0.0, passed=False, explanation="unparseable JSON")
        if not isinstance(data, dict):
            return Score(name=self.name, value=0.0, passed=False, explanation="not a JSON object")
        tol = float(self.options.get("tol", 1e-6))
        wrong = [
            k for k, v in expected.items() if k not in data or not _values_equal(v, data[k], tol)
        ]
        value = 1 - len(wrong) / len(expected)
        return Score(
            name=self.name,
            value=value,
            passed=not wrong,
            explanation=f"incorrect fields: {', '.join(wrong)}" if wrong else None,
            metadata={"incorrect_fields": wrong},
        )
