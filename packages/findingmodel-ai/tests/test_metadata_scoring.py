"""Tests for metadata eval scoring helpers."""

from __future__ import annotations

from types import SimpleNamespace

from evals.metadata_assignment import (
    _quality_field_applicable,
    metadata_gate_failures,
    metadata_quality_case_score,
)
from evals.metadata_scoring import (
    score_code_selection,
    score_commission_sensitive_set_similarity,
    score_expected_time_course,
    score_optional_field_match,
    score_set_similarity,
)
from findingmodel import ExpectedDuration, ExpectedTimeCourse, TimeCourseModifier


def test_optional_field_match_gives_abstention_credit_for_blank() -> None:
    assert score_optional_field_match(None, ["chest"]) == 0.25


def test_set_similarity_gives_configured_missing_credit() -> None:
    assert score_set_similarity(None, ["CT"], missing_expected_credit=0.25) == 0.25


def test_commission_sensitive_set_similarity_penalizes_extra_more_than_missing() -> None:
    missing_score = score_commission_sensitive_set_similarity(["expected"], ["expected", "missed"])
    extra_score = score_commission_sensitive_set_similarity(["expected", "extra"], ["expected"])

    assert missing_score > extra_score


def test_commission_sensitive_set_similarity_penalizes_extra_on_expected_blank() -> None:
    omission_score = score_commission_sensitive_set_similarity(None, ["expected"])
    commission_score = score_commission_sensitive_set_similarity(["extra"], None)

    assert commission_score < omission_score


def test_expected_time_course_penalizes_commission_more_than_omission() -> None:
    time_course = ExpectedTimeCourse(
        duration=ExpectedDuration.MONTHS,
        modifiers=[TimeCourseModifier.RESOLVING],
    )

    omission_score = score_expected_time_course(None, time_course)
    commission_score = score_expected_time_course(time_course, None)

    assert commission_score < omission_score


def test_code_selection_does_not_penalize_existing_extra() -> None:
    score = score_code_selection(
        actual=["SNOMEDCT:1", "RADLEX:extra"],
        expected=["SNOMEDCT:1"],
        existing=["RADLEX:extra"],
    )

    assert score == 1.0


def test_code_selection_moderately_penalizes_new_extra() -> None:
    score = score_code_selection(
        actual=["SNOMEDCT:1", "RADLEX:new"],
        expected=["SNOMEDCT:1"],
        existing=[],
    )

    assert 0.8 < score < 1.0


def test_metadata_quality_case_score_excludes_gate_scores() -> None:
    case = SimpleNamespace(
        scores={
            "quality.entity_type": SimpleNamespace(value=0.5),
            "ExecutionSuccessEvaluator": SimpleNamespace(value=1.0),
        }
    )

    assert metadata_quality_case_score(case) == 0.5


def test_metadata_gate_failures_read_assertions_not_scores() -> None:
    report = SimpleNamespace(
        cases=[
            SimpleNamespace(
                name="case_a",
                assertions={
                    "ExecutionSuccessEvaluator": SimpleNamespace(value=True, reason="ok"),
                    "CandidateIntegrityEvaluator": SimpleNamespace(value=False, reason="bad candidate"),
                },
            )
        ]
    )

    assert metadata_gate_failures(report) == [("case_a", "CandidateIntegrityEvaluator", "bad candidate")]


def test_optional_both_null_non_core_field_is_not_quality_applicable() -> None:
    actual = SimpleNamespace(age_profile=None)
    gold = SimpleNamespace(age_profile=None)

    assert not _quality_field_applicable(actual, gold, "age_profile", {})
