"""Metadata-assignment result and review types for structured finding model workflows."""

from datetime import datetime
from enum import Enum
from typing import Literal, get_args

from findingmodel import FindingModelFull
from oidm_common.models import IndexCode
from pydantic import BaseModel, Field, field_validator

ConfidenceFieldKey = Literal[
    "body_regions",
    "subspecialties",
    "etiologies",
    "entity_type",
    "applicable_modalities",
    "expected_time_course",
    "age_profile",
    "sex_specificity",
    "anatomic_locations",
    "index_codes",
]


class FieldConfidence(str, Enum):
    """Confidence level for an assigned field value."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


FieldConfidenceScore = float
CONFIDENCE_FIELD_KEYS = set(get_args(ConfidenceFieldKey))

_CONFIDENCE_LABEL_TO_SCORE = {
    FieldConfidence.HIGH.value: 0.9,
    FieldConfidence.MEDIUM.value: 0.6,
    FieldConfidence.LOW.value: 0.2,
}


def coerce_field_confidence_map(value: object) -> dict[ConfidenceFieldKey, FieldConfidenceScore]:
    """Convert model confidence output into numeric 0-1 scores.

    Invalid keys and values are ignored because confidence is optional review metadata, not a reason
    to retry or reject an otherwise usable assignment.
    """
    if not isinstance(value, dict):
        return {}

    coerced: dict[ConfidenceFieldKey, FieldConfidenceScore] = {}
    for raw_key, raw_value in value.items():
        if raw_key not in CONFIDENCE_FIELD_KEYS:
            continue

        score = _coerce_confidence_score(raw_value)
        if score is None:
            continue
        coerced[raw_key] = score  # type: ignore[literal-required]

    return coerced


def _coerce_confidence_score(value: object) -> float | None:
    score: float | None = None
    if isinstance(value, FieldConfidence):
        score = _CONFIDENCE_LABEL_TO_SCORE[value.value]
    elif isinstance(value, str):
        score = _CONFIDENCE_LABEL_TO_SCORE.get(value.strip().lower())
        if score is None:
            try:
                score = float(value)
            except ValueError:
                return None
    elif isinstance(value, int | float):
        score = float(value)

    if score is None:
        return None
    if 1 < score <= 100:
        score = score / 100
    if 0 <= score <= 1:
        return score
    return None


class OntologyCandidateRelationship(str, Enum):
    """Relationship between a finding and an ontology concept."""

    EXACT_MATCH = "exact_match"
    CLINICALLY_SUBSTITUTABLE = "clinically_substitutable"
    NARROWER = "narrower"
    BROADER = "broader"
    RELATED = "related"
    COMPLICATION = "complication"


class OntologyCandidateRejectionReason(str, Enum):
    """Why a reviewed ontology candidate was not accepted as canonical."""

    TOO_SPECIFIC = "too_specific"
    TOO_BROAD = "too_broad"
    WRONG_CONCEPT = "wrong_concept"
    DEFINITION_MISMATCH = "definition_mismatch"
    OVERLAPPING_SCOPE = "overlapping_scope"


class OntologyCandidate(BaseModel):
    """A candidate ontology code with its relationship to the finding."""

    code: IndexCode
    relationship: OntologyCandidateRelationship
    rationale: str | None = None
    rejection_reason: OntologyCandidateRejectionReason | None = None


class OntologyCandidateReport(BaseModel):
    """Report of ontology candidates split into canonical codes and review candidates."""

    canonical_codes: list[OntologyCandidate] = Field(default_factory=list)
    review_candidates: list[OntologyCandidate] = Field(default_factory=list)


class AnatomicCandidate(BaseModel):
    """A candidate anatomic location for the finding."""

    location: IndexCode
    selected: bool
    rationale: str | None = None
    support_level: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    broader_candidate_ids: list[str] = Field(default_factory=list)


class MetadataAssignmentReview(BaseModel):
    """Review/provenance data from a metadata-assignment run."""

    oifm_id: str | None = None
    finding_name: str
    assignment_timestamp: datetime
    model_used: str
    assignment_mode: str = "reassess"  # "reassess" or "fill_blanks_only"
    logfire_trace_id: str | None = None
    ontology_candidates: OntologyCandidateReport = Field(default_factory=OntologyCandidateReport)
    anatomic_candidates: list[AnatomicCandidate] = Field(default_factory=list)
    classification_rationale: str = ""
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _coerce_field_confidence(cls, value: object) -> object:
        return coerce_field_confidence_map(value)


class MetadataAssignmentResult(BaseModel):
    """Combined result of a metadata-assignment run."""

    model: FindingModelFull
    review: MetadataAssignmentReview
