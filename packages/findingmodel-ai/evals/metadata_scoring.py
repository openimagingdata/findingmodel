"""Shared scoring helpers for metadata enrichment evals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from findingmodel import EntityType, ExpectedDuration, ExpectedTimeCourse
from pydantic_evals.evaluators import EvaluationReason

T = TypeVar("T")


DURATION_ORDER: tuple[ExpectedDuration, ...] = (
    ExpectedDuration.HOURS,
    ExpectedDuration.DAYS,
    ExpectedDuration.WEEKS,
    ExpectedDuration.MONTHS,
    ExpectedDuration.YEARS,
    ExpectedDuration.PERMANENT,
)


@dataclass(frozen=True)
class WeightedComponent:
    """Named score component for a weighted case result."""

    name: str
    score: float
    weight: float
    reason: str = ""


def clamp_score(value: float) -> float:
    """Clamp a numeric score into the eval score range."""

    return max(0.0, min(1.0, value))


def score_set_similarity(
    actual: Iterable[T] | None,
    expected: Iterable[T] | None,
    *,
    recall_weight: float = 0.65,
    missing_expected_credit: float = 0.0,
    extra_actual_credit: float = 0.35,
) -> float:
    """Score set agreement with recall weighted above precision by default."""

    actual_set = set(actual or [])
    expected_set = set(expected or [])
    if not actual_set and not expected_set:
        return 1.0
    if not actual_set:
        return missing_expected_credit if expected_set else 1.0
    if not expected_set:
        return extra_actual_credit if actual_set else 1.0

    true_positive = len(actual_set & expected_set)
    recall = true_positive / len(expected_set)
    precision = true_positive / len(actual_set)
    return clamp_score((recall_weight * recall) + ((1.0 - recall_weight) * precision))


def score_commission_sensitive_set_similarity(
    actual: Iterable[T] | None,
    expected: Iterable[T] | None,
    *,
    recall_weight: float = 0.35,
    missing_expected_credit: float = 0.40,
    extra_actual_credit: float = 0.10,
) -> float:
    """Score set agreement when unsupported extras are costlier than omissions."""

    return score_set_similarity(
        actual,
        expected,
        recall_weight=recall_weight,
        missing_expected_credit=missing_expected_credit,
        extra_actual_credit=extra_actual_credit,
    )


def score_optional_field_match(actual: Any, expected: Any, *, abstention_credit: float = 0.25) -> float:
    """Score optional metadata with conservative credit for justified blanks."""

    if actual == expected:
        return 1.0
    if actual is None and expected is not None:
        return abstention_credit
    if isinstance(actual, list) or isinstance(expected, list):
        return score_set_similarity(
            actual if isinstance(actual, list) else None,
            expected if isinstance(expected, list) else None,
            missing_expected_credit=abstention_credit,
        )
    return 0.0


def score_code_selection(
    actual: Iterable[T] | None,
    expected: Iterable[T] | None,
    *,
    existing: Iterable[T] | None = None,
    recall_weight: float = 0.70,
) -> float:
    """Score code selections while not penalizing carried-forward existing extras.

    Missing expected codes are costly. Existing extra codes are treated as acceptable review
    carry-forward. Newly added extras reduce precision moderately rather than zeroing the case.
    """

    actual_set = set(actual or [])
    expected_set = set(expected or [])
    existing_set = set(existing or [])
    if not actual_set and not expected_set:
        return 1.0
    if not actual_set:
        return 0.0 if expected_set else 1.0

    expected_or_existing = expected_set | existing_set
    true_positive = len(actual_set & expected_set)
    recall = true_positive / len(expected_set) if expected_set else 1.0
    precision = len(actual_set & expected_or_existing) / len(actual_set)
    return clamp_score((recall_weight * recall) + ((1.0 - recall_weight) * precision))


def score_required_forbidden_allowed(
    actual: Iterable[T] | None,
    *,
    required: Iterable[T] | None = None,
    forbidden: Iterable[T] | None = None,
    allowed: Iterable[T] | None = None,
    recall_weight: float = 0.70,
) -> float:
    """Score required/forbidden/allowed set expectations.

    Forbidden hits are treated as severe enough to zero the component. Missing required
    values hurt recall. Values outside an allowed set reduce precision without zeroing the case.
    """

    actual_set = set(actual or [])
    required_set = set(required or [])
    forbidden_set = set(forbidden or [])
    allowed_set = set(allowed) if allowed is not None else None

    if actual_set & forbidden_set:
        return 0.0

    recall = 1.0 if not required_set else len(actual_set & required_set) / len(required_set)
    precision = 1.0 if allowed_set is None or not actual_set else len(actual_set & allowed_set) / len(actual_set)
    return clamp_score((recall_weight * recall) + ((1.0 - recall_weight) * precision))


def score_entity_type(actual: EntityType | None, expected: EntityType | None) -> float:
    """Score entity type with limited partial credit for nearby clinical distinctions."""

    if actual == expected:
        return 1.0
    if actual is None or expected is None:
        return 0.0
    near_pairs = {
        frozenset({EntityType.FINDING, EntityType.DIAGNOSIS}): 0.65,
        frozenset({EntityType.MEASUREMENT, EntityType.ASSESSMENT}): 0.50,
        frozenset({EntityType.ASSESSMENT, EntityType.RECOMMENDATION}): 0.40,
        frozenset({EntityType.FINDING, EntityType.MEASUREMENT}): 0.30,
        frozenset({EntityType.FINDING, EntityType.TECHNIQUE_ISSUE}): 0.25,
    }
    return near_pairs.get(frozenset({actual, expected}), 0.0)


def score_expected_time_course(
    actual: ExpectedTimeCourse | None,
    expected: ExpectedTimeCourse | None,
    *,
    duration_weight: float = 0.80,
    missing_expected_credit: float = 0.35,
    extra_actual_credit: float = 0.0,
) -> float:
    """Score observable-persistence agreement with ordinal duration distance."""

    if actual is None and expected is None:
        return 1.0
    if actual is None:
        return missing_expected_credit
    if expected is None:
        return extra_actual_credit

    if actual.duration is None and expected.duration is None:
        duration_score = 1.0
    elif actual.duration is None:
        duration_score = missing_expected_credit
    elif expected.duration is None:
        duration_score = extra_actual_credit
    else:
        duration_score = _score_duration(actual.duration, expected.duration)
    modifier_score = score_commission_sensitive_set_similarity(
        actual.modifiers or [],
        expected.modifiers or [],
        missing_expected_credit=0.25,
    )
    return clamp_score((duration_weight * duration_score) + ((1.0 - duration_weight) * modifier_score))


def weighted_score(components: Sequence[WeightedComponent]) -> EvaluationReason:
    """Return a weighted numeric score with a compact diagnostic reason."""

    active = [component for component in components if component.weight > 0]
    if not active:
        return EvaluationReason(value=0.0, reason="no weighted components")
    total_weight = sum(component.weight for component in active)
    value = sum(component.score * component.weight for component in active) / total_weight
    reason = "; ".join(
        f"{component.name}={component.score:.2f}" + (f" ({component.reason})" if component.reason else "")
        for component in active
    )
    return EvaluationReason(value=clamp_score(value), reason=reason)


def weighted_case_score(case: Any, weights: Mapping[str, float]) -> float:
    """Compute a weighted score from a Pydantic Evals report case."""

    total = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        score = case.scores.get(name)
        if score is None:
            continue
        total += score.value * weight
        total_weight += weight
    return total / total_weight if total_weight else 0.0


def print_weighted_summary(report: Any, weights: Mapping[str, float], *, title: str) -> None:
    """Print a compact weighted summary for a metadata eval report."""

    if not report.cases:
        print(f"{title}: no cases")
        return
    case_scores = [(case.name or "<unnamed>", weighted_case_score(case, weights)) for case in report.cases]
    overall = sum(score for _, score in case_scores) / len(case_scores)
    print(f"\n{title} weighted overall: {overall:.2f}")
    for evaluator_name in weights:
        values = [case.scores[evaluator_name].value for case in report.cases if evaluator_name in case.scores]
        if values:
            print(f"{evaluator_name}: {sum(values) / len(values):.2f}")
    print("Lowest scoring cases:")
    for name, score in sorted(case_scores, key=lambda item: item[1])[:5]:
        print(f"- {name}: {score:.2f}")


def _score_duration(actual: ExpectedDuration, expected: ExpectedDuration) -> float:
    if actual == expected:
        return 1.0
    actual_index = DURATION_ORDER.index(actual)
    expected_index = DURATION_ORDER.index(expected)
    distance = abs(actual_index - expected_index)
    return clamp_score(1.0 - (distance * 0.25))
