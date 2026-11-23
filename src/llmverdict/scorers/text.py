"""Deterministic text scorers."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any

from llmverdict.scorers.base import Scorer
from llmverdict.types import Sample, Score

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_ANSWER_LINE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*(.+)", re.I)


def normalize_text(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, squash spaces."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _as_list(reference: Any) -> list[str]:
    if reference is None:
        return []
    if isinstance(reference, (list, tuple)):
        return [str(r) for r in reference]
    return [str(reference)]


def extract_final_answer(text: str) -> str:
    """Pull the final answer out of a (possibly chain-of-thought) response.

    Prefers the *last* ``Answer: ...`` line; falls back to the last non-empty line.
    """
    matches: list[str] = _ANSWER_LINE.findall(text)
    candidate: str
    if matches:
        candidate = matches[-1]
    else:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        candidate = lines[-1] if lines else ""
    # Peel markdown emphasis and trailing full stops in any order: "**Tuesday**." -> "Tuesday"
    previous = None
    while previous != candidate:
        previous = candidate
        candidate = candidate.strip().strip("*_`").rstrip(".")
    return candidate


def parse_number(text: str) -> float | None:
    """Last number in ``text``; tolerant of thousands separators and currency symbols."""
    matches = _NUMBER.findall(text.replace("$", "").replace("−", "-"))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:  # pragma: no cover - regex guarantees a parseable token
        return None


class ExactMatch(Scorer):
    type_name = "exact_match"

    async def score(self, sample: Sample, output: str) -> Score:
        refs = _as_list(sample.reference)
        if not refs:
            return self._na("no reference")
        norm = self.options.get("normalize", True)
        got = normalize_text(output) if norm else output.strip()
        ok = any(got == (normalize_text(r) if norm else r.strip()) for r in refs)
        return self._bool(ok)


class Includes(Scorer):
    """Passes if any reference string appears in the output (after normalisation)."""

    type_name = "includes"

    async def score(self, sample: Sample, output: str) -> Score:
        reference = sample.reference
        if isinstance(reference, dict):
            reference = reference.get(self.options.get("field", "expected_fact"))
        refs = _as_list(reference)
        if not refs:
            return self._na("no reference")
        got = normalize_text(output)
        ok = any(normalize_text(r) in got for r in refs)
        return self._bool(ok)


class RegexMatch(Scorer):
    type_name = "regex"

    async def score(self, sample: Sample, output: str) -> Score:
        pattern = self.options.get("pattern") or sample.reference
        if not pattern:
            return self._na("no pattern")
        ok = re.search(str(pattern), output, re.I | re.S) is not None
        return self._bool(ok)


class FinalAnswer(Scorer):
    """Extract ``Answer: <x>`` and compare numerically (with tolerance) or as normalised text."""

    type_name = "final_answer"

    async def score(self, sample: Sample, output: str) -> Score:
        ref = sample.reference
        if ref is None:
            return self._na("no reference")
        extracted = extract_final_answer(output)
        if isinstance(ref, (int, float)) and not isinstance(ref, bool):
            got = parse_number(extracted)
            if got is None:
                return self._bool(False, "no number found in final answer", extracted=extracted)
            ok = math.isclose(
                got,
                float(ref),
                rel_tol=float(self.options.get("rel_tol", 1e-6)),
                abs_tol=float(self.options.get("abs_tol", 1e-6)),
            )
            return self._bool(ok, None if ok else f"expected {ref}, got {got}", extracted=got)
        ok = normalize_text(extracted) == normalize_text(str(ref))
        return self._bool(ok, None if ok else f"expected {ref!r}", extracted=extracted)


class TokenF1(Scorer):
    """Token-overlap F1 (SQuAD). Graded credit for free-form short answers."""

    type_name = "token_f1"

    async def score(self, sample: Sample, output: str) -> Score:
        refs = _as_list(sample.reference)
        if not refs:
            return self._na("no reference")
        pred = normalize_text(output).split()
        best = 0.0
        for ref in refs:
            gold = normalize_text(ref).split()
            common = Counter(pred) & Counter(gold)
            overlap = sum(common.values())
            if overlap == 0:
                continue
            precision, recall = overlap / len(pred), overlap / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
        threshold = float(self.options.get("threshold", 0.5))
        return Score(name=self.name, value=best, passed=best >= threshold)
