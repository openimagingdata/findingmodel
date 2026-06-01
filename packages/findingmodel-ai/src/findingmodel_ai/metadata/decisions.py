"""Pydantic output models for metadata-assignment agents."""

from __future__ import annotations

from findingmodel import (
    AgeProfile,
    BodyRegion,
    EntityType,
    EtiologyCode,
    ExpectedTimeCourse,
    Modality,
    SexSpecificity,
    Subspecialty,
)
from pydantic import BaseModel, Field, field_validator

from findingmodel_ai.metadata.confidence import coerce_field_confidence_map
from findingmodel_ai.metadata.types import (
    ConfidenceFieldKey,
    FieldConfidenceScore,
    OntologyCandidateRejectionReason,
    OntologyCandidateRelationship,
)


class OntologyCandidateDecision(BaseModel):
    """Selection decision for one ontology candidate."""

    candidate_id: str
    relationship: OntologyCandidateRelationship
    selected_as_canonical: bool = False
    rejection_reason: OntologyCandidateRejectionReason | None = None


class AnatomicCandidateDecision(BaseModel):
    """Selection decision for one anatomic candidate."""

    candidate_id: str
    selected: bool
    rationale: str | None = None
    rejection_reason: str | None = None


class MetadataAssignmentDecision(BaseModel):
    """Combined focused-agent output used by metadata assignment assembly."""

    body_regions: list[BodyRegion] | None = Field(
        default=None,
        description=(
            "Top-level affected anatomy: head, neck, chest, breast, abdomen, pelvis, spine, "
            "upper_extremity, lower_extremity, or whole_body."
        ),
    )
    subspecialties: list[Subspecialty] | None = Field(
        default=None,
        description=(
            "Radiology domain labels: BR breast, CA cardiac, CH chest, ER emergency, GI "
            "gastrointestinal, GU genitourinary, HN head/neck, IR interventional, MI molecular/PET, "
            "MK musculoskeletal, NM nuclear medicine, NR neuroradiology, OB obstetric/gynecologic, "
            "OI oncologic imaging, PD pediatric, SQ quality/safety/technique, VA vascular."
        ),
    )
    etiologies: list[EtiologyCode] | None = Field(
        default=None,
        description=(
            "Broad cause class when intrinsic to the modeled finding, such as inflammatory, "
            "neoplastic, traumatic, vascular, cardiac, degenerative, metabolic, congenital, "
            "developmental, autoimmune, toxic, mechanical, iatrogenic, idiopathic, or normal-variant."
        ),
    )
    entity_type: EntityType | None = Field(
        default=None,
        description=(
            "Kind of modeled concept: finding, diagnosis, grouping, measurement, assessment, "
            "recommendation, or technique_issue."
        ),
    )
    applicable_modalities: list[Modality] | None = Field(
        default=None,
        description="Routine direct imaging modalities for the modeled finding: XR, CT, MR, US, PET, NM, MG, RF, or DSA.",
    )
    expected_time_course: ExpectedTimeCourse | None = Field(
        default=None,
        description="Expected observable imaging duration and behavior for the modeled finding itself.",
    )
    age_profile: AgeProfile | None = Field(
        default=None, description="Patient age applicability and optional commonness."
    )
    sex_specificity: SexSpecificity | None = Field(
        default=None,
        description="Patient sex applicability: male-specific, female-specific, or sex-neutral.",
    )
    ontology_decisions: list[OntologyCandidateDecision] = Field(default_factory=list)
    anatomic_decisions: list[AnatomicCandidateDecision] = Field(default_factory=list)
    classification_rationale: str = ""
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore] = Field(default_factory=dict)

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _coerce_field_confidence(cls, value: object) -> object:
        return coerce_field_confidence_map(value)


class EntityTypeDecision(BaseModel):
    """Focused entity-type output."""

    entity_type: EntityType | None = Field(
        default=None,
        description=(
            "Kind of modeled concept: finding, diagnosis, grouping, measurement, assessment, "
            "recommendation, or technique_issue."
        ),
    )
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore] = Field(default_factory=dict)

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _coerce_field_confidence(cls, value: object) -> object:
        return coerce_field_confidence_map(value)


class EtiologyTempoDecision(BaseModel):
    """Pilot focused broad-cause and observable-persistence output."""

    etiologies: list[EtiologyCode] | None = Field(
        default=None,
        description=(
            "Why the finding exists: the common disease process, injury, treatment/device effect, "
            "or normal variant that produces the finding or diagnosis. Null when no cause label is "
            "supported."
        ),
    )
    expected_time_course: ExpectedTimeCourse | None = Field(
        default=None,
        description=(
            "How long the finding commonly remains visible on imaging, using the long end of common "
            "persistence rather than symptom duration or rare outliers."
        ),
    )
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore] = Field(
        default_factory=dict,
        description="Optional confidence scores from 0 to 1 for metadata fields assigned by this agent.",
    )

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _coerce_field_confidence(cls, value: object) -> object:
        return coerce_field_confidence_map(value)


class PatientApplicabilityDecision(BaseModel):
    """Focused patient age and sex applicability output."""

    age_profile: AgeProfile | None = Field(
        default=None,
        description="Patient age applicability: all_ages or supported MeSH-derived age stages, with optional commonness.",
    )
    sex_specificity: SexSpecificity | None = Field(
        default=None,
        description="Patient sex applicability: male-specific, female-specific, or sex-neutral.",
    )
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore] = Field(default_factory=dict)

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _coerce_field_confidence(cls, value: object) -> object:
        return coerce_field_confidence_map(value)


class SubspecialtyDomainDecision(BaseModel):
    """Focused radiology subspecialty-domain output."""

    subspecialties: list[Subspecialty] | None = Field(
        default=None,
        description=(
            "Multi-label radiology domain tags: BR breast, CA cardiac, CH chest, ER emergency, "
            "GI gastrointestinal/hepatobiliary, GU genitourinary, HN head/neck, IR interventional, "
            "MI molecular/PET, MK musculoskeletal, NM nuclear medicine, NR neuroradiology, "
            "OB obstetric/gynecologic, OI oncologic, PD pediatric, SQ quality/safety/technique, "
            "VA vascular."
        ),
    )
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore] = Field(default_factory=dict)

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _coerce_field_confidence(cls, value: object) -> object:
        return coerce_field_confidence_map(value)


class ModalityApplicabilityDecision(BaseModel):
    """Focused imaging-modality applicability output."""

    applicable_modalities: list[Modality] | None = Field(
        default=None,
        description="Routine direct imaging modalities for the modeled finding: XR, CT, MR, US, PET, NM, MG, RF, or DSA.",
    )
    field_confidence: dict[ConfidenceFieldKey, FieldConfidenceScore] = Field(default_factory=dict)

    @field_validator("field_confidence", mode="before")
    @classmethod
    def _coerce_field_confidence(cls, value: object) -> object:
        return coerce_field_confidence_map(value)


class OntologyDecision(BaseModel):
    """Focused ontology-candidate output."""

    ontology_decisions: list[OntologyCandidateDecision] = Field(default_factory=list)


class AnatomyDecision(BaseModel):
    """Focused anatomy and body-region output."""

    body_regions: list[BodyRegion] | None = Field(
        default=None,
        description=(
            "Top-level affected anatomy: head, neck, chest, breast, abdomen, pelvis, spine, "
            "upper_extremity, lower_extremity, or whole_body."
        ),
    )
    anatomic_decisions: list[AnatomicCandidateDecision] = Field(default_factory=list)
