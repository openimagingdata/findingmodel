"""Confidence parsing and threshold helpers for metadata assignment."""

from __future__ import annotations

from findingmodel_ai.metadata.types import (
    ConfidenceFieldKey,
    FieldConfidence,
    FieldConfidenceScore,
)
from findingmodel_ai.metadata.types import (
    coerce_field_confidence_map as _coerce_field_confidence_map,
)

LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.85


def coerce_field_confidence_map(value: object) -> dict[ConfidenceFieldKey, FieldConfidenceScore]:
    return _coerce_field_confidence_map(value)


def confidence_level(score: float | None) -> FieldConfidence | None:
    if score is None:
        return None
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return FieldConfidence.HIGH
    if score >= LOW_CONFIDENCE_THRESHOLD:
        return FieldConfidence.MEDIUM
    return FieldConfidence.LOW


def field_confidence_level(
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore],
    field_name: ConfidenceFieldKey,
) -> FieldConfidence | None:
    return confidence_level(field_confidence.get(field_name))


def is_low_confidence(
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore],
    field_name: ConfidenceFieldKey,
) -> bool:
    return field_confidence_level(field_confidence, field_name) == FieldConfidence.LOW


def is_high_confidence(
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore],
    field_name: ConfidenceFieldKey,
) -> bool:
    return field_confidence_level(field_confidence, field_name) == FieldConfidence.HIGH


def is_low_or_medium_confidence(
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore],
    field_name: ConfidenceFieldKey,
) -> bool:
    return field_confidence_level(field_confidence, field_name) in {
        FieldConfidence.LOW,
        FieldConfidence.MEDIUM,
    }
