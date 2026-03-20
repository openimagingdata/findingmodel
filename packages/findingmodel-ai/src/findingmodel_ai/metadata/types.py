"""Metadata-assignment result and review types for structured finding model workflows."""

from datetime import datetime
from enum import Enum

from findingmodel.finding_model import FindingModelFull
from oidm_common.models import IndexCode
from pydantic import BaseModel, Field


class FieldConfidence(str, Enum):
    """Confidence level for an assigned field value."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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


class MetadataAssignmentReview(BaseModel):
    """Review/provenance data from a metadata-assignment run."""

    oifm_id: str | None = None
    finding_name: str
    assignment_timestamp: datetime
    model_used: str
    logfire_trace_id: str | None = None
    ontology_candidates: OntologyCandidateReport = Field(default_factory=OntologyCandidateReport)
    anatomic_candidates: list[AnatomicCandidate] = Field(default_factory=list)
    classification_rationale: str = ""
    field_confidence: dict[str, FieldConfidence] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class MetadataAssignmentResult(BaseModel):
    """Combined result of a metadata-assignment run."""

    model: FindingModelFull
    review: MetadataAssignmentReview
