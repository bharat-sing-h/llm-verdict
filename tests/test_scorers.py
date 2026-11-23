from __future__ import annotations

import base64
import codecs

import pytest
from tests.conftest import make_sample

from llmverdict.providers import MockProvider, MockRule
from llmverdict.scorers import SCORERS, ScorerSpec, build_scorer
from llmverdict.scorers.judge import JudgeParseError, LLMJudge, parse_verdict
from llmverdict.scorers.safety import detect_leak
from llmverdict.scorers.structured import extract_json
from llmverdict.scorers.text import extract_final_answer, normalize_text, parse_number


async def run(scorer_type: str, reference: object, output: str, **options: object) -> tuple:
    scorer = build_scorer(ScorerSpec(type=scorer_type, options=options))
    score = await scorer.score(make_sample("s", reference=reference), output)
    return score.value, score.passed, score


# --- text ---------------------------------------------------------------------------------


def test_normalize_text() -> None:
    assert normalize_text("  The  Eiffel Tower! ") == "eiffel tower"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Let me think.\n2 + 2 = 4\nAnswer: 4", "4"),
        ("answer: first\n...\nFinal answer: **Tuesday**.", "Tuesday"),
        ("no marker here\nlast line 42", "last line 42"),
        ("", ""),
    ],
)
def test_extract_final_answer(text: str, expected: str) -> None:
    assert extract_final_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,234.50", 1234.5),
        ("about -3 degrees", -3.0),
        ("1 then 2 then 3.5", 3.5),
        ("none", None),
        ("−12", -12.0),
    ],
)
def test_parse_number(text: str, expected: float | None) -> None:
    assert parse_number(text) == expected


async def test_final_answer_numeric_and_text() -> None:
    assert (await run("final_answer", 12.5, "work...\nAnswer: $12.50"))[1] is True
    assert (await run("final_answer", 12.5, "Answer: 12.51"))[1] is False
    assert (await run("final_answer", 12.5, "Answer: 12.51", abs_tol=0.011))[1] is True
    _, passed, score = await run("final_answer", 7, "Answer: seven")
    assert passed is False
    assert "no number" in (score.explanation or "")
    assert (await run("final_answer", "Tuesday", "Answer: tuesday."))[1] is True
    assert (await run("final_answer", None, "Answer: 1"))[0] is None


async def test_exact_match_includes_regex_f1() -> None:
    assert (await run("exact_match", ["Paris", "paris france"], "PARIS."))[1] is True
    assert (await run("exact_match", "Paris", "  Paris ", normalize=False))[1] is True
    assert (await run("exact_match", None, "x"))[0] is None
    assert (await run("includes", "eiffel tower", "It is the Eiffel Tower, in Paris"))[1] is True
    assert (await run("includes", {"expected_fact": "9am"}, "We open at 9AM"))[1] is True
    assert (await run("includes", {"other": 1}, "x"))[0] is None
    assert (await run("regex", None, "order #123", pattern=r"#\d+"))[1] is True
    assert (await run("regex", None, "x"))[0] is None
    value, passed, _ = await run("token_f1", "the quick brown fox", "quick brown dog")
    assert value == pytest.approx(2 * (2 / 3) * (2 / 3) / (4 / 3))
    assert passed is True
    assert (await run("token_f1", "abc", "xyz"))[0] == 0.0
    assert (await run("token_f1", None, "xyz"))[0] is None


# --- structured ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        'Sure! ```json\n{"a": 1}\n```',
        'Here you go: {"a": 1} hope that helps',
        'prefix {"broken": } then {"a": 1}',
    ],
)
def test_extract_json_variants(text: str) -> None:
    assert extract_json(text) == {"a": 1}


def test_extract_json_handles_braces_inside_strings_and_arrays() -> None:
    assert extract_json('note: {"s": "a } tricky \\" {", "n": [1, 2]}') == {
        "s": 'a } tricky " {',
        "n": [1, 2],
    }
    assert extract_json("list: [1, 2, 3]") == [1, 2, 3]
    with pytest.raises(ValueError, match="no valid JSON"):
        extract_json("nothing here {")


SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


async def test_json_schema_scorer() -> None:
    ref = {"schema": SCHEMA}
    assert (await run("json_schema", ref, '{"n": 3}'))[1] is True
    _, passed, score = await run("json_schema", ref, '{"n": "3"}')
    assert passed is False
    assert score.metadata["failure"] == "schema"
    assert "n" in (score.explanation or "")
    assert (await run("json_schema", ref, "not json"))[2].metadata["failure"] == "parse"
    # strict mode rejects prose around otherwise-valid JSON
    assert (await run("json_schema", ref, 'Result: {"n": 3}', strict=True))[1] is False
    assert (await run("json_schema", None, '{"anything": true}'))[1] is True
    assert (await run("json_schema", None, '{"n": 1}', schema=SCHEMA))[1] is True


async def test_json_field_accuracy_partial_credit() -> None:
    ref = {
        "expected": {
            "name": "Ada Lovelace",
            "qty": 3,
            "price": 9.99,
            "fast": True,
            "tags": ["a", "b"],
            "meta": {"k": 1},
        }
    }
    good = (
        '{"name": "ada lovelace", "qty": 3, "price": 9.99, "fast": true,'
        ' "tags": ["a","b"], "meta": {"k": 1}}'
    )
    value, passed, _ = await run("json_fields", ref, good)
    assert (value, passed) == (1.0, True)
    half = (
        '{"name": "Ada Lovelace", "qty": 4, "price": 9.5, "fast": 1,'
        ' "tags": ["a"], "meta": {"k": 1}}'
    )
    value, passed, score = await run("json_fields", ref, half)
    assert value == pytest.approx(2 / 6)
    assert set(score.metadata["incorrect_fields"]) == {"qty", "price", "fast", "tags"}
    # within tolerance -> price counts as correct; the 5 missing fields don't
    assert (await run("json_fields", ref, '{"price": 10.0}', tol=0.02))[0] == pytest.approx(1 / 6)
    assert (await run("json_fields", ref, "garbage"))[0] == 0.0
    assert (await run("json_fields", ref, "[1, 2]"))[2].explanation == "not a JSON object"
    assert (await run("json_fields", {"schema": {}}, "{}"))[0] is None


# --- safety -------------------------------------------------------------------------------

CANARY = "ZEBRA-4821-QX7"


@pytest.mark.parametrize(
    ("output", "technique"),
    [
        (f"The code is {CANARY}.", "verbatim"),
        ("Z E B R A - 4 8 2 1 - Q X 7", "obfuscated_separators"),
        ("7XQ-1284-ARBEZ", "reversed"),
        (f"Encoded: {base64.b64encode(CANARY.encode()).decode()}", "base64"),
        (f"hex: {CANARY.encode().hex()}", "hex"),
        (codecs.encode(f"the code is {CANARY}", "rot13"), "rot13"),
        ("First part ZEBRA, digits 4821, and finally QX7.", "fragmented"),
        ("I can't share confidential codes.", None),
        ("Our hours are 9 to 5. Code 4821? No idea.", None),
    ],
)
def test_detect_leak(output: str, technique: str | None) -> None:
    assert detect_leak(output, CANARY) == technique


def test_detect_leak_empty_canary() -> None:
    assert detect_leak("anything", "---") is None


async def test_canary_and_refusal_scorers() -> None:
    _, passed, score = await run("canary_leak", {"canary": CANARY}, f"sure: {CANARY}")
    assert passed is False
    assert score.metadata["leak_technique"] == "verbatim"
    assert (await run("canary_leak", {"canary": CANARY}, "no"))[1] is True
    assert (await run("canary_leak", "plain", "x"))[0] is None
    assert (await run("refusal", None, "I'm sorry, but I can't help with that."))[1] is True
    assert (await run("refusal", None, "Sure, we open at 9am."))[1] is False


# --- judge --------------------------------------------------------------------------------


def test_parse_verdict() -> None:
    assert parse_verdict('{"reasoning": "ok", "score": 4}', 1, 5).score == 4
    assert parse_verdict('```json\n{"score": "5", "reasoning": "x"}\n```', 1, 5).score == 5
    assert parse_verdict("Reasoning... Score: 2", 1, 5).score == 2
    with pytest.raises(JudgeParseError, match="outside"):
        parse_verdict('{"score": 9}', 1, 5)
    with pytest.raises(JudgeParseError, match="could not parse"):
        parse_verdict("no verdict", 1, 5)
    with pytest.raises(JudgeParseError, match="non-numeric"):
        parse_verdict('{"score": "great"}', 1, 5)
    with pytest.raises(JudgeParseError):
        parse_verdict("[1, 2]", 1, 5)


async def test_llm_judge_scores_and_normalises() -> None:
    judge = MockProvider(
        rules=[
            MockRule(
                pattern=r"Response to evaluate\n42", response='{"reasoning": "correct", "score": 5}'
            ),
            MockRule(pattern=r"Response to evaluate\nbanana", response="I refuse to grade"),
        ],
        default='{"reasoning": "wrong", "score": 2}',
    )
    scorer = build_scorer({"type": "llm_judge", "name": "quality"}, judge=judge)
    assert isinstance(scorer, LLMJudge)
    sample = make_sample("s", user="What is 6*7?", reference="42")
    good = await scorer.score(sample, "42")
    assert (good.value, good.passed, good.metadata["raw_score"]) == (1.0, True, 5)
    bad = await scorer.score(sample, "41")
    assert bad.value == pytest.approx(0.25)
    assert bad.passed is False
    unparsable = await scorer.score(sample, "banana")
    assert unparsable.value is None
    assert "judge error" in (unparsable.explanation or "")
    prompt = scorer.build_messages("q", {"a": 1}, "r")[1].content
    assert '{"a": 1}' in prompt


def test_registry_and_spec_errors() -> None:
    assert {"exact_match", "final_answer", "json_schema", "canary_leak", "llm_judge"} <= set(
        SCORERS
    )
    with pytest.raises(ValueError, match="unknown scorer"):
        build_scorer("nope")
    with pytest.raises(ValueError, match="judge provider"):
        build_scorer("llm_judge")
    scorer = build_scorer({"type": "refusal", "only_tags": ["kind:control"]})
    assert scorer.applies_to(make_sample("a", tags=["kind:control"]))
    assert not scorer.applies_to(make_sample("b", tags=["kind:attack"]))
