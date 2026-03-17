"""Enrichment result and review types for finding model enrichment workflows."""

from datetime import datetime
from enum import Enum
from typing import Literal

from findingmodel.finding_model import FindingModelFull
from oidm_common.models import IndexCode
from pydantic import BaseModel, Field


class FieldConfidence(str, Enum):
    """Confidence level for an enriched field value."""

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


class OntologyCandidate(BaseModel):
    """A candidate ontology code with its relationship to the finding."""

    code: IndexCode
    relationship: OntologyCandidateRelationship
    rationale: str | None = None


class OntologyCandidateReport(BaseModel):
    """Report of ontology candidates split into canonical codes and review candidates."""

    canonical_codes: list[OntologyCandidate] = Field(default_factory=list)
    review_candidates: list[OntologyCandidate] = Field(default_factory=list)


class AnatomicCandidate(BaseModel):
    """A candidate anatomic location for the finding."""

    location: IndexCode
    selected: bool
    rationale: str | None = None


class EnrichModelReview(BaseModel):
    """Review/provenance data from an enrichment run — kept separate from the canonical model."""

    oifm_id: str | None = None
    finding_name: str
    enrichment_timestamp: datetime
    model_used: str
    model_tier: Literal["base", "small", "full"]
    ontology_candidates: OntologyCandidateReport = Field(default_factory=OntologyCandidateReport)
    anatomic_candidates: list[AnatomicCandidate] = Field(default_factory=list)
    classification_rationale: str = ""
    field_confidence: dict[str, FieldConfidence] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class EnrichModelResult(BaseModel):
    """Combined result of an enrichment run: the canonical model plus its review sidecar."""

    model: FindingModelFull
    review: EnrichModelReview
