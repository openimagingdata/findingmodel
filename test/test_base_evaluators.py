"""Unit tests for base evaluator classes.

This test suite verifies the scoring logic, configuration options, and edge case
handling in the 5 base evaluator classes. Per pydantic_ai_testing_best_practices,
these tests focus on OUR code behavior (scoring calculations, partial credit,
edge cases) rather than framework functionality.

All tests use simple Pydantic models and execute quickly without external dependencies.
"""

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_evals.evaluators import EvaluatorContext

from evals.base import (
    ContainsEvaluator,
    ErrorHandlingEvaluator,
    ExactMatchEvaluator,
    KeywordMatchEvaluator,
    StructuralValidityEvaluator,
)


def _create_ctx(
    inputs: Any,
    expected_output: Any,
    output: Any,
    name: str = "test",
) -> EvaluatorContext:
    """Helper to create minimal EvaluatorContext for unit tests."""
    return EvaluatorContext(
        name=name,
        inputs=inputs,
        metadata=None,
        expected_output=expected_output,
        output=output,
        duration=0.0,
        _span_tree=None,  # type: ignore[arg-type]
        attributes={},
        metrics={},
    )


# Test models for ExactMatchEvaluator
class ExactMatchInput(BaseModel):
    query: str


class ExactMatchExpected(BaseModel):
    text: str


# ============================================================================
# ExactMatchEvaluator Tests (4 tests)
# ============================================================================


def test_exact_match_returns_one_on_match() -> None:
    """Verify exact string match returns 1.0."""
    evaluator = ExactMatchEvaluator[ExactMatchInput]()

    ctx = _create_ctx(
        inputs=ExactMatchInput(query="test"),
        expected_output=ExactMatchExpected(text="expected text"),
        output="expected text",
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_exact_match_returns_zero_on_mismatch() -> None:
    """Verify non-matching strings return 0.0."""
    evaluator = ExactMatchEvaluator[ExactMatchInput]()

    ctx = _create_ctx(
        inputs=ExactMatchInput(query="test"),
        expected_output=ExactMatchExpected(text="expected text"),
        output="different text",
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_exact_match_is_case_sensitive() -> None:
    """Verify matching is case-sensitive (exact means exact)."""
    evaluator = ExactMatchEvaluator[ExactMatchInput]()

    ctx = _create_ctx(
        inputs=ExactMatchInput(query="test"),
        expected_output=ExactMatchExpected(text="Expected Text"),
        output="expected text",
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0  # Case mismatch


def test_exact_match_handles_empty_strings() -> None:
    """Verify empty string matching behavior."""
    evaluator = ExactMatchEvaluator[ExactMatchInput]()

    # Empty matches empty
    ctx = _create_ctx(
        inputs=ExactMatchInput(query="test"),
        expected_output=ExactMatchExpected(text=""),
        output="",
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


# Test models for ContainsEvaluator
class ContainsInput(BaseModel):
    query: str


class ContainsExpected(BaseModel):
    substring: str


# ============================================================================
# ContainsEvaluator Tests (5 tests)
# ============================================================================


def test_contains_case_insensitive_default() -> None:
    """Verify default case-insensitive matching."""
    evaluator = ContainsEvaluator[ContainsInput](case_sensitive=False)

    ctx = _create_ctx(
        inputs=ContainsInput(query="test"),
        expected_output=ContainsExpected(substring="KEYWORD"),
        output="This contains keyword text",
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_contains_case_sensitive_option() -> None:
    """Verify case-sensitive matching when enabled."""
    evaluator = ContainsEvaluator[ContainsInput](case_sensitive=True)

    ctx = _create_ctx(
        inputs=ContainsInput(query="test"),
        expected_output=ContainsExpected(substring="KEYWORD"),
        output="This contains keyword text",
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0  # Case mismatch


def test_contains_returns_zero_when_not_found() -> None:
    """Verify returns 0.0 when substring absent."""
    evaluator = ContainsEvaluator[ContainsInput](case_sensitive=False)

    ctx = _create_ctx(
        inputs=ContainsInput(query="test"),
        expected_output=ContainsExpected(substring="missing"),
        output="This text has no match",
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_contains_handles_empty_substring() -> None:
    """Verify behavior with empty expected substring."""
    evaluator = ContainsEvaluator[ContainsInput](case_sensitive=False)

    ctx = _create_ctx(
        inputs=ContainsInput(query="test"),
        expected_output=ContainsExpected(substring=""),
        output="any text",
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # Empty string is contained in any string


def test_contains_handles_empty_output() -> None:
    """Verify behavior with empty output string."""
    evaluator = ContainsEvaluator[ContainsInput](case_sensitive=False)

    ctx = _create_ctx(
        inputs=ContainsInput(query="test"),
        expected_output=ContainsExpected(substring="keyword"),
        output="",
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0  # Keyword not in empty string


# Test models for KeywordMatchEvaluator
class KeywordInput(BaseModel):
    query: str


class KeywordExpected(BaseModel):
    expected_keywords: list[str]


class KeywordOutput(BaseModel):
    text: str


# ============================================================================
# KeywordMatchEvaluator Tests (7 tests)
# ============================================================================


def test_keyword_all_found_partial_credit() -> None:
    """Verify 3/3 keywords with partial_credit=True returns 1.0."""
    evaluator = KeywordMatchEvaluator[KeywordInput, KeywordOutput](
        keyword_field="expected_keywords",
        text_extractor=lambda x: x.text,
        partial_credit=True,
    )

    ctx = _create_ctx(
        inputs=KeywordInput(query="test"),
        expected_output=KeywordExpected(expected_keywords=["alpha", "beta", "gamma"]),
        output=KeywordOutput(text="Found alpha and beta and gamma"),
    )

    score = evaluator.evaluate(ctx)
    assert score == pytest.approx(1.0)


def test_keyword_some_found_partial_credit() -> None:
    """Verify 2/3 keywords with partial_credit=True returns 0.67."""
    evaluator = KeywordMatchEvaluator[KeywordInput, KeywordOutput](
        keyword_field="expected_keywords",
        text_extractor=lambda x: x.text,
        partial_credit=True,
    )

    ctx = _create_ctx(
        inputs=KeywordInput(query="test"),
        expected_output=KeywordExpected(expected_keywords=["alpha", "beta", "gamma"]),
        output=KeywordOutput(text="Found alpha and beta but not the third"),
    )

    score = evaluator.evaluate(ctx)
    assert score == pytest.approx(0.666, abs=0.01)  # 2/3


def test_keyword_none_found_returns_zero() -> None:
    """Verify 0/3 keywords returns 0.0."""
    evaluator = KeywordMatchEvaluator[KeywordInput, KeywordOutput](
        keyword_field="expected_keywords",
        text_extractor=lambda x: x.text,
        partial_credit=True,
    )

    ctx = _create_ctx(
        inputs=KeywordInput(query="test"),
        expected_output=KeywordExpected(expected_keywords=["alpha", "beta", "gamma"]),
        output=KeywordOutput(text="No matching words here"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_keyword_all_found_no_partial_credit() -> None:
    """Verify 3/3 keywords with partial_credit=False returns 1.0."""
    evaluator = KeywordMatchEvaluator[KeywordInput, KeywordOutput](
        keyword_field="expected_keywords",
        text_extractor=lambda x: x.text,
        partial_credit=False,
    )

    ctx = _create_ctx(
        inputs=KeywordInput(query="test"),
        expected_output=KeywordExpected(expected_keywords=["alpha", "beta", "gamma"]),
        output=KeywordOutput(text="Found alpha and beta and gamma"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_keyword_some_found_no_partial_credit() -> None:
    """Verify 2/3 keywords with partial_credit=False returns 0.0."""
    evaluator = KeywordMatchEvaluator[KeywordInput, KeywordOutput](
        keyword_field="expected_keywords",
        text_extractor=lambda x: x.text,
        partial_credit=False,
    )

    ctx = _create_ctx(
        inputs=KeywordInput(query="test"),
        expected_output=KeywordExpected(expected_keywords=["alpha", "beta", "gamma"]),
        output=KeywordOutput(text="Found alpha and beta but not the third"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0  # Not all keywords found


def test_keyword_empty_list_returns_one() -> None:
    """Verify empty keyword list returns 1.0 (N/A case)."""
    evaluator = KeywordMatchEvaluator[KeywordInput, KeywordOutput](
        keyword_field="expected_keywords",
        text_extractor=lambda x: x.text,
        partial_credit=True,
    )

    ctx = _create_ctx(
        inputs=KeywordInput(query="test"),
        expected_output=KeywordExpected(expected_keywords=[]),
        output=KeywordOutput(text="Any text here"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_keyword_case_insensitive_matching() -> None:
    """Verify keywords match regardless of case."""
    evaluator = KeywordMatchEvaluator[KeywordInput, KeywordOutput](
        keyword_field="expected_keywords",
        text_extractor=lambda x: x.text,
        partial_credit=True,
    )

    ctx = _create_ctx(
        inputs=KeywordInput(query="test"),
        expected_output=KeywordExpected(expected_keywords=["ALPHA", "Beta", "gamma"]),
        output=KeywordOutput(text="found alpha and BETA and GaMmA"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


# Test models for StructuralValidityEvaluator
class StructuralInput(BaseModel):
    query: str


class StructuralExpected(BaseModel):
    pass  # No specific expected output


class StructuralOutput(BaseModel):
    model_id: str = ""
    attributes: list[Any] = []


# ============================================================================
# StructuralValidityEvaluator Tests (5 tests)
# ============================================================================


def test_structural_all_fields_present() -> None:
    """Verify 2/2 required fields returns 1.0."""
    evaluator = StructuralValidityEvaluator[StructuralInput](required_fields=["model_id", "attributes"])

    ctx = _create_ctx(
        inputs=StructuralInput(query="test"),
        expected_output=StructuralExpected(),
        output=StructuralOutput(model_id="FM123", attributes=["a", "b"]),
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_structural_some_fields_present() -> None:
    """Verify 1/2 required fields returns 0.5."""
    evaluator = StructuralValidityEvaluator[StructuralInput](required_fields=["model_id", "missing_field"])

    ctx = _create_ctx(
        inputs=StructuralInput(query="test"),
        expected_output=StructuralExpected(),
        output=StructuralOutput(model_id="FM123"),
    )

    score = evaluator.evaluate(ctx)
    assert score == pytest.approx(0.5)  # 1/2


def test_structural_no_fields_present() -> None:
    """Verify 0/2 required fields returns 0.0."""
    evaluator = StructuralValidityEvaluator[StructuralInput](required_fields=["missing1", "missing2"])

    ctx = _create_ctx(
        inputs=StructuralInput(query="test"),
        expected_output=StructuralExpected(),
        output=StructuralOutput(model_id="FM123"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_structural_empty_required_list_returns_one() -> None:
    """Verify no required fields specified returns 1.0 (N/A case)."""
    evaluator = StructuralValidityEvaluator[StructuralInput](required_fields=[])

    ctx = _create_ctx(
        inputs=StructuralInput(query="test"),
        expected_output=StructuralExpected(),
        output=StructuralOutput(model_id="FM123"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_structural_non_basemodel_returns_zero() -> None:
    """Verify non-Pydantic output returns 0.0."""
    evaluator = StructuralValidityEvaluator[StructuralInput](required_fields=["field"])

    # Using a non-BaseModel output (string)
    ctx = _create_ctx(
        inputs=StructuralInput(query="test"),
        expected_output=StructuralExpected(),
        output="not a pydantic model",
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


# Test models for ErrorHandlingEvaluator
class ErrorInput(BaseModel):
    query: str


class ErrorExpected(BaseModel):
    should_succeed: bool


class ErrorOutput(BaseModel):
    result: str
    error: str | None = None


# ============================================================================
# ErrorHandlingEvaluator Tests (4 tests - truth table)
# ============================================================================


def test_error_should_succeed_and_does() -> None:
    """Verify should_succeed=True with no error returns 1.0."""
    evaluator = ErrorHandlingEvaluator[ErrorInput, ErrorOutput](error_field="error")

    ctx = _create_ctx(
        inputs=ErrorInput(query="test"),
        expected_output=ErrorExpected(should_succeed=True),
        output=ErrorOutput(result="success", error=None),
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_error_should_succeed_but_fails() -> None:
    """Verify should_succeed=True with error returns 0.0."""
    evaluator = ErrorHandlingEvaluator[ErrorInput, ErrorOutput](error_field="error")

    ctx = _create_ctx(
        inputs=ErrorInput(query="test"),
        expected_output=ErrorExpected(should_succeed=True),
        output=ErrorOutput(result="", error="Something went wrong"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_error_should_fail_and_does() -> None:
    """Verify should_succeed=False with error returns 1.0."""
    evaluator = ErrorHandlingEvaluator[ErrorInput, ErrorOutput](error_field="error")

    ctx = _create_ctx(
        inputs=ErrorInput(query="test"),
        expected_output=ErrorExpected(should_succeed=False),
        output=ErrorOutput(result="", error="Rejected as expected"),
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_error_should_fail_but_succeeds() -> None:
    """Verify should_succeed=False with no error returns 0.0."""
    evaluator = ErrorHandlingEvaluator[ErrorInput, ErrorOutput](error_field="error")

    ctx = _create_ctx(
        inputs=ErrorInput(query="test"),
        expected_output=ErrorExpected(should_succeed=False),
        output=ErrorOutput(result="success", error=None),
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0
