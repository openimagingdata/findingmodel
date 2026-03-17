"""Canonical structured metadata types for finding models.

These types are shared across findingmodel and findingmodel-ai packages.
"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator


class BodyRegion(str, Enum):
    HEAD = "head"
    NECK = "neck"
    CHEST = "chest"
    BREAST = "breast"
    ABDOMEN = "abdomen"
    PELVIS = "pelvis"
    SPINE = "spine"
    UPPER_EXTREMITY = "upper_extremity"
    LOWER_EXTREMITY = "lower_extremity"
    WHOLE_BODY = "whole_body"


class Subspecialty(str, Enum):
    AB = "AB"  # Abdominal
    BR = "BR"  # Breast
    CA = "CA"  # Cardiac
    CH = "CH"  # Chest
    ER = "ER"  # Emergency
    GI = "GI"  # Gastrointestinal
    GU = "GU"  # Genitourinary
    HN = "HN"  # Head and Neck
    IR = "IR"  # Interventional
    MI = "MI"  # Molecular Imaging
    MK = "MK"  # Musculoskeletal
    NR = "NR"  # Neuroradiology
    OB = "OB"  # OB/GYN
    OI = "OI"  # Oncologic Imaging
    PD = "PD"  # Pediatric
    VI = "VI"  # Vascular/Interventional


class Modality(str, Enum):
    XR = "XR"
    CT = "CT"
    MR = "MR"
    US = "US"
    PET = "PET"
    NM = "NM"
    MG = "MG"
    RF = "RF"
    DSA = "DSA"


class EntityType(str, Enum):
    FINDING = "finding"
    DIAGNOSIS = "diagnosis"
    GROUPING = "grouping"
    MEASUREMENT = "measurement"
    ASSESSMENT = "assessment"
    RECOMMENDATION = "recommendation"
    TECHNIQUE_ISSUE = "technique_issue"


class SexSpecificity(str, Enum):
    MALE_SPECIFIC = "male-specific"
    FEMALE_SPECIFIC = "female-specific"
    SEX_NEUTRAL = "sex-neutral"


class AgeStage(str, Enum):
    """Disjoint, MeSH-derived age bins for faceting."""

    NEWBORN = "newborn"
    INFANT = "infant"
    PRESCHOOL_CHILD = "preschool_child"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    MIDDLE_AGED = "middle_aged"
    AGED = "aged"


class AgeProfile(BaseModel):
    """Age applicability profile for a finding model."""

    applicability: Literal["all_ages"] | list[AgeStage] = Field(
        description="Either 'all_ages' or a list of applicable age stages."
    )
    more_common_in: list[AgeStage] | None = Field(
        default=None,
        description="Age stages where this finding is more commonly seen.",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_applicability(cls, data: object) -> object:
        app = data.get("applicability") if isinstance(data, dict) else getattr(data, "applicability", None)
        if isinstance(app, str) and app != "all_ages":
            raise ValueError("applicability must be 'all_ages' or a list of AgeStage values")
        if isinstance(app, list) and len(app) == 0:
            raise ValueError("applicability must be 'all_ages' or a non-empty list of AgeStage values")
        return data


class ExpectedDuration(str, Enum):
    """Upper bound on how long the finding typically remains visible on imaging."""

    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"
    YEARS = "years"
    PERMANENT = "permanent"


class TimeCourseModifier(str, Enum):
    """Behavioral pattern of the finding over time."""

    PROGRESSIVE = "progressive"  # SNOMED 255314001, RadLex RID39162
    STABLE = "stable"  # SNOMED 702322003, RadLex RID5734
    EVOLVING = "evolving"  # SNOMED 59381007
    RESOLVING = "resolving"  # common radiology usage
    INTERMITTENT = "intermittent"  # SNOMED 7087005
    FLUCTUATING = "fluctuating"  # SNOMED 255341006
    RECURRENT = "recurrent"  # SNOMED 255227004


class ExpectedTimeCourse(BaseModel):
    """Expected temporal behavior of a finding on imaging."""

    duration: ExpectedDuration | None = None
    modifiers: list[TimeCourseModifier] = Field(default_factory=list)


class EtiologyCode(str, Enum):
    INFLAMMATORY = "inflammatory"
    INFLAMMATORY_INFECTIOUS = "inflammatory:infectious"
    NEOPLASTIC_BENIGN = "neoplastic:benign"
    NEOPLASTIC_MALIGNANT = "neoplastic:malignant"
    NEOPLASTIC_METASTATIC = "neoplastic:metastatic"
    NEOPLASTIC_POTENTIAL = "neoplastic:potential"
    TRAUMATIC_ACUTE = "traumatic:acute"
    TRAUMATIC_SEQUELA = "traumatic:sequela"
    VASCULAR_ISCHEMIC = "vascular:ischemic"
    VASCULAR_HEMORRHAGIC = "vascular:hemorrhagic"
    VASCULAR_THROMBOTIC = "vascular:thrombotic"
    VASCULAR_ANEURYSMAL = "vascular:aneurysmal"
    DEGENERATIVE = "degenerative"
    METABOLIC = "metabolic"
    CONGENITAL = "congenital"
    DEVELOPMENTAL = "developmental"
    AUTOIMMUNE = "autoimmune"
    TOXIC = "toxic"
    MECHANICAL = "mechanical"
    IATROGENIC_POST_OPERATIVE = "iatrogenic:post-operative"
    IATROGENIC_POST_RADIATION = "iatrogenic:post-radiation"
    IATROGENIC_DEVICE = "iatrogenic:device"
    IATROGENIC_MEDICATION_RELATED = "iatrogenic:medication-related"
    IDIOPATHIC = "idiopathic"
    NORMAL_VARIANT = "normal-variant"


# ============================================================================
# Legacy Value Normalization
# ============================================================================

# Body region normalization: title-cased legacy -> lowercase canonical
BODY_REGION_LEGACY_MAP: dict[str, BodyRegion] = {
    "Head": BodyRegion.HEAD,
    "Neck": BodyRegion.NECK,
    "Chest": BodyRegion.CHEST,
    "Breast": BodyRegion.BREAST,
    "Abdomen": BodyRegion.ABDOMEN,
    "Pelvis": BodyRegion.PELVIS,
    "Spine": BodyRegion.SPINE,
    "Arm": BodyRegion.UPPER_EXTREMITY,
    "Leg": BodyRegion.LOWER_EXTREMITY,
    "ALL": BodyRegion.WHOLE_BODY,
    "Upper Extremity": BodyRegion.UPPER_EXTREMITY,
    "Lower Extremity": BodyRegion.LOWER_EXTREMITY,
    "Whole Body": BodyRegion.WHOLE_BODY,
}


def normalize_body_region(value: str) -> BodyRegion:
    """Normalize a body region string to a canonical BodyRegion value.

    Accepts canonical lowercase values, legacy title-cased values, and special aliases.
    """
    try:
        return BodyRegion(value)
    except ValueError:
        pass
    if value in BODY_REGION_LEGACY_MAP:
        return BODY_REGION_LEGACY_MAP[value]
    raise ValueError(f"Unknown body region: {value!r}")


# Etiology normalization: legacy shorthand -> canonical
ETIOLOGY_LEGACY_MAP: dict[str, EtiologyCode] = {
    "traumatic": EtiologyCode.TRAUMATIC_ACUTE,
    "post-traumatic": EtiologyCode.TRAUMATIC_SEQUELA,
    "iatrogenic:device-related": EtiologyCode.IATROGENIC_DEVICE,
}


def normalize_etiology(value: str) -> EtiologyCode:
    """Normalize an etiology string to a canonical EtiologyCode value."""
    try:
        return EtiologyCode(value)
    except ValueError:
        pass
    if value in ETIOLOGY_LEGACY_MAP:
        return ETIOLOGY_LEGACY_MAP[value]
    raise ValueError(f"Unknown etiology code: {value!r}")


# Modality normalization: CR and DX -> XR
MODALITY_LEGACY_MAP: dict[str, Modality] = {
    "CR": Modality.XR,
    "DX": Modality.XR,
}


def normalize_modality(value: str) -> Modality:
    """Normalize a modality string to a canonical Modality value."""
    try:
        return Modality(value)
    except ValueError:
        pass
    if value in MODALITY_LEGACY_MAP:
        return MODALITY_LEGACY_MAP[value]
    raise ValueError(f"Unknown modality: {value!r}")


# Age label normalization: legacy free-text -> canonical AgeProfile
AGE_LABEL_LEGACY_MAP: dict[str, AgeProfile] = {
    "any age": AgeProfile(applicability="all_ages"),
    "all ages": AgeProfile(applicability="all_ages"),
    "any": AgeProfile(applicability="all_ages"),
    "pediatric": AgeProfile(
        applicability=[AgeStage.NEWBORN, AgeStage.INFANT, AgeStage.PRESCHOOL_CHILD, AgeStage.CHILD, AgeStage.ADOLESCENT]
    ),
    "adult": AgeProfile(applicability=[AgeStage.ADULT, AgeStage.MIDDLE_AGED, AgeStage.AGED]),
    "elderly": AgeProfile(applicability=[AgeStage.AGED]),
    "neonatal": AgeProfile(applicability=[AgeStage.NEWBORN]),
}


def normalize_age_label(value: str) -> AgeProfile:
    """Normalize a legacy age label string to a canonical AgeProfile."""
    normalized = value.strip().lower()
    if normalized in AGE_LABEL_LEGACY_MAP:
        return AGE_LABEL_LEGACY_MAP[normalized]
    raise ValueError(f"Unknown age label: {value!r}")


# ============================================================================
# Normalizing Validators for Pydantic Field Use
# ============================================================================


def _normalize_body_region_value(v: object) -> object:
    """BeforeValidator that normalizes a single body region string."""
    if isinstance(v, str):
        try:
            return normalize_body_region(v)
        except ValueError:
            return v  # let Pydantic's own enum validation produce the error
    return v


def _normalize_body_region_list(v: object) -> object:
    """BeforeValidator that normalizes a list of body region strings."""
    if isinstance(v, list):
        return [_normalize_body_region_value(item) for item in v]
    return v


def _normalize_etiology_value(v: object) -> object:
    if isinstance(v, str):
        try:
            return normalize_etiology(v)
        except ValueError:
            return v
    return v


def _normalize_etiology_list(v: object) -> object:
    if isinstance(v, list):
        return [_normalize_etiology_value(item) for item in v]
    return v


def _normalize_modality_value(v: object) -> object:
    if isinstance(v, str):
        try:
            return normalize_modality(v)
        except ValueError:
            return v
    return v


def _normalize_modality_list(v: object) -> object:
    if isinstance(v, list):
        return [_normalize_modality_value(item) for item in v]
    return v


# Annotated types that auto-normalize legacy values during Pydantic validation.
# Use these on model fields instead of raw `list[BodyRegion]` etc.
NormalizedBodyRegionList = Annotated[list[BodyRegion], BeforeValidator(_normalize_body_region_list)]
NormalizedEtiologyList = Annotated[list[EtiologyCode], BeforeValidator(_normalize_etiology_list)]
NormalizedModalityList = Annotated[list[Modality], BeforeValidator(_normalize_modality_list)]
