"""Tests for etiology/time-course component eval infrastructure."""

from __future__ import annotations

from evals.metadata_etiology_tempo_decision import (
    EtiologyTempoDecisionActualOutput,
    EtiologyTempoDecisionExpectedOutput,
    miss_labels,
    select_etiology_tempo_cases,
)
from findingmodel import EtiologyCode, ExpectedDuration, ExpectedTimeCourse, TimeCourseModifier


def test_etiology_tempo_case_sets_cover_reviewed_corpus() -> None:
    pilot_cases = select_etiology_tempo_cases("pilot")
    gold_cases = select_etiology_tempo_cases("gold")
    reviewed_cases = select_etiology_tempo_cases("reviewed")
    expanded_cases = select_etiology_tempo_cases("expanded")
    all_cases = select_etiology_tempo_cases("all")

    assert len(pilot_cases) == 20
    assert len(gold_cases) == 35
    assert len(reviewed_cases) == 71
    assert len(expanded_cases) == 106
    assert len(all_cases) == 126


def test_etiology_tempo_miss_labels_are_field_specific() -> None:
    expected = EtiologyTempoDecisionExpectedOutput(
        etiologies={EtiologyCode.INFLAMMATORY},
        expected_time_course=ExpectedTimeCourse(
            duration=ExpectedDuration.DAYS,
            modifiers=[TimeCourseModifier.PROGRESSIVE],
        ),
    )
    output = EtiologyTempoDecisionActualOutput(
        etiologies={EtiologyCode.INFLAMMATORY_INFECTIOUS},
        expected_time_course=ExpectedTimeCourse(
            duration=ExpectedDuration.WEEKS,
            modifiers=[TimeCourseModifier.RESOLVING],
        ),
    )

    assert miss_labels(expected, output) == [
        "missing_expected_etiology",
        "extra_unsupported_etiology",
        "wrong_etiology_family_or_subtype",
        "wrong_duration",
        "wrong_modifier",
    ]
