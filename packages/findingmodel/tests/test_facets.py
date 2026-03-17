"""Tests for canonical structured metadata types defined in findingmodel.facets."""

import json
from typing import Any

import pytest
from findingmodel.facets import (
    AgeProfile,
    AgeStage,
    BodyRegion,
    EntityType,
    EtiologyCode,
    ExpectedDuration,
    ExpectedTimeCourse,
    Modality,
    SexSpecificity,
    Subspecialty,
    TimeCourseModifier,
    normalize_age_label,
    normalize_body_region,
    normalize_etiology,
    normalize_modality,
)
from findingmodel.finding_model import (
    ChoiceAttribute,
    ChoiceAttributeIded,
    ChoiceValue,
    ChoiceValueIded,
    FindingModelBase,
    FindingModelFull,
    NumericAttributeIded,
)
from pydantic import ValidationError

# ============================================================================
# BodyRegion Enum Tests
# ============================================================================


def test_body_region_from_canonical_values() -> None:
    assert BodyRegion("head") == BodyRegion.HEAD
    assert BodyRegion("neck") == BodyRegion.NECK
    assert BodyRegion("chest") == BodyRegion.CHEST
    assert BodyRegion("breast") == BodyRegion.BREAST
    assert BodyRegion("abdomen") == BodyRegion.ABDOMEN
    assert BodyRegion("pelvis") == BodyRegion.PELVIS
    assert BodyRegion("spine") == BodyRegion.SPINE
    assert BodyRegion("upper_extremity") == BodyRegion.UPPER_EXTREMITY
    assert BodyRegion("lower_extremity") == BodyRegion.LOWER_EXTREMITY
    assert BodyRegion("whole_body") == BodyRegion.WHOLE_BODY


def test_body_region_all_members() -> None:
    expected = {
        "head",
        "neck",
        "chest",
        "breast",
        "abdomen",
        "pelvis",
        "spine",
        "upper_extremity",
        "lower_extremity",
        "whole_body",
    }
    actual = {member.value for member in BodyRegion}
    assert actual == expected


def test_body_region_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        BodyRegion("Head")  # title case not directly valid
    with pytest.raises(ValueError):
        BodyRegion("unknown_region")


# ============================================================================
# Subspecialty Enum Tests
# ============================================================================


def test_subspecialty_from_canonical_values() -> None:
    assert Subspecialty("AB") == Subspecialty.AB
    assert Subspecialty("BR") == Subspecialty.BR
    assert Subspecialty("CA") == Subspecialty.CA
    assert Subspecialty("CH") == Subspecialty.CH
    assert Subspecialty("NR") == Subspecialty.NR
    assert Subspecialty("MK") == Subspecialty.MK


def test_subspecialty_all_members() -> None:
    expected = {"AB", "BR", "CA", "CH", "ER", "GI", "GU", "HN", "IR", "MI", "MK", "NR", "OB", "OI", "PD", "VI"}
    actual = {member.value for member in Subspecialty}
    assert actual == expected


def test_subspecialty_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        Subspecialty("UNKNOWN")
    with pytest.raises(ValueError):
        Subspecialty("ab")  # lowercase not valid


# ============================================================================
# Modality Enum Tests
# ============================================================================


def test_modality_from_canonical_values() -> None:
    assert Modality("XR") == Modality.XR
    assert Modality("CT") == Modality.CT
    assert Modality("MR") == Modality.MR
    assert Modality("US") == Modality.US
    assert Modality("PET") == Modality.PET
    assert Modality("NM") == Modality.NM
    assert Modality("MG") == Modality.MG
    assert Modality("RF") == Modality.RF
    assert Modality("DSA") == Modality.DSA


def test_modality_all_members() -> None:
    expected = {"XR", "CT", "MR", "US", "PET", "NM", "MG", "RF", "DSA"}
    actual = {member.value for member in Modality}
    assert actual == expected


def test_modality_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        Modality("CR")  # legacy alias, not canonical
    with pytest.raises(ValueError):
        Modality("XRAY")


# ============================================================================
# EntityType Enum Tests
# ============================================================================


def test_entity_type_from_canonical_values() -> None:
    assert EntityType("finding") == EntityType.FINDING
    assert EntityType("diagnosis") == EntityType.DIAGNOSIS
    assert EntityType("grouping") == EntityType.GROUPING
    assert EntityType("measurement") == EntityType.MEASUREMENT
    assert EntityType("assessment") == EntityType.ASSESSMENT
    assert EntityType("recommendation") == EntityType.RECOMMENDATION


def test_entity_type_all_members() -> None:
    expected = {"finding", "diagnosis", "grouping", "measurement", "assessment", "recommendation", "technique_issue"}
    actual = {member.value for member in EntityType}
    assert actual == expected


def test_entity_type_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        EntityType("unknown")
    with pytest.raises(ValueError):
        EntityType("Finding")  # title case not valid


# ============================================================================
# SexSpecificity Enum Tests
# ============================================================================


def test_sex_specificity_from_canonical_values() -> None:
    assert SexSpecificity("male-specific") == SexSpecificity.MALE_SPECIFIC
    assert SexSpecificity("female-specific") == SexSpecificity.FEMALE_SPECIFIC
    assert SexSpecificity("sex-neutral") == SexSpecificity.SEX_NEUTRAL


def test_sex_specificity_all_members() -> None:
    expected = {"male-specific", "female-specific", "sex-neutral"}
    actual = {member.value for member in SexSpecificity}
    assert actual == expected


def test_sex_specificity_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        SexSpecificity("neutral")
    with pytest.raises(ValueError):
        SexSpecificity("male")


# ============================================================================
# AgeStage Enum Tests
# ============================================================================


def test_age_stage_from_canonical_values() -> None:
    assert AgeStage("newborn") == AgeStage.NEWBORN
    assert AgeStage("infant") == AgeStage.INFANT
    assert AgeStage("preschool_child") == AgeStage.PRESCHOOL_CHILD
    assert AgeStage("child") == AgeStage.CHILD
    assert AgeStage("adolescent") == AgeStage.ADOLESCENT
    assert AgeStage("young_adult") == AgeStage.YOUNG_ADULT
    assert AgeStage("adult") == AgeStage.ADULT
    assert AgeStage("middle_aged") == AgeStage.MIDDLE_AGED
    assert AgeStage("aged") == AgeStage.AGED


def test_age_stage_all_members() -> None:
    expected = {
        "newborn",
        "infant",
        "preschool_child",
        "child",
        "adolescent",
        "young_adult",
        "adult",
        "middle_aged",
        "aged",
    }
    actual = {member.value for member in AgeStage}
    assert actual == expected


def test_age_stage_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        AgeStage("elderly")
    with pytest.raises(ValueError):
        AgeStage("Adult")  # title case not valid


# ============================================================================
# ExpectedDuration Enum Tests
# ============================================================================


def test_expected_duration_from_canonical_values() -> None:
    assert ExpectedDuration("hours") == ExpectedDuration.HOURS
    assert ExpectedDuration("days") == ExpectedDuration.DAYS
    assert ExpectedDuration("weeks") == ExpectedDuration.WEEKS
    assert ExpectedDuration("months") == ExpectedDuration.MONTHS
    assert ExpectedDuration("years") == ExpectedDuration.YEARS
    assert ExpectedDuration("permanent") == ExpectedDuration.PERMANENT


def test_expected_duration_all_members() -> None:
    expected = {"hours", "days", "weeks", "months", "years", "permanent"}
    actual = {member.value for member in ExpectedDuration}
    assert actual == expected


def test_expected_duration_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        ExpectedDuration("acute")
    with pytest.raises(ValueError):
        ExpectedDuration("Hours")  # title case not valid


# ============================================================================
# TimeCourseModifier Enum Tests
# ============================================================================


def test_time_course_modifier_from_canonical_values() -> None:
    assert TimeCourseModifier("progressive") == TimeCourseModifier.PROGRESSIVE
    assert TimeCourseModifier("stable") == TimeCourseModifier.STABLE
    assert TimeCourseModifier("evolving") == TimeCourseModifier.EVOLVING
    assert TimeCourseModifier("resolving") == TimeCourseModifier.RESOLVING
    assert TimeCourseModifier("intermittent") == TimeCourseModifier.INTERMITTENT
    assert TimeCourseModifier("fluctuating") == TimeCourseModifier.FLUCTUATING
    assert TimeCourseModifier("recurrent") == TimeCourseModifier.RECURRENT


def test_time_course_modifier_all_members() -> None:
    expected = {"progressive", "stable", "evolving", "resolving", "intermittent", "fluctuating", "recurrent"}
    actual = {member.value for member in TimeCourseModifier}
    assert actual == expected


def test_time_course_modifier_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        TimeCourseModifier("chronic")
    with pytest.raises(ValueError):
        TimeCourseModifier("Stable")  # title case not valid


# ============================================================================
# EtiologyCode Enum Tests
# ============================================================================


def test_etiology_code_from_canonical_values() -> None:
    assert EtiologyCode("inflammatory") == EtiologyCode.INFLAMMATORY
    assert EtiologyCode("inflammatory:infectious") == EtiologyCode.INFLAMMATORY_INFECTIOUS
    assert EtiologyCode("neoplastic:benign") == EtiologyCode.NEOPLASTIC_BENIGN
    assert EtiologyCode("neoplastic:malignant") == EtiologyCode.NEOPLASTIC_MALIGNANT
    assert EtiologyCode("neoplastic:metastatic") == EtiologyCode.NEOPLASTIC_METASTATIC
    assert EtiologyCode("neoplastic:potential") == EtiologyCode.NEOPLASTIC_POTENTIAL
    assert EtiologyCode("traumatic:acute") == EtiologyCode.TRAUMATIC_ACUTE
    assert EtiologyCode("traumatic:sequela") == EtiologyCode.TRAUMATIC_SEQUELA
    assert EtiologyCode("vascular:ischemic") == EtiologyCode.VASCULAR_ISCHEMIC
    assert EtiologyCode("vascular:hemorrhagic") == EtiologyCode.VASCULAR_HEMORRHAGIC
    assert EtiologyCode("vascular:thrombotic") == EtiologyCode.VASCULAR_THROMBOTIC
    assert EtiologyCode("vascular:aneurysmal") == EtiologyCode.VASCULAR_ANEURYSMAL
    assert EtiologyCode("degenerative") == EtiologyCode.DEGENERATIVE
    assert EtiologyCode("metabolic") == EtiologyCode.METABOLIC
    assert EtiologyCode("congenital") == EtiologyCode.CONGENITAL
    assert EtiologyCode("developmental") == EtiologyCode.DEVELOPMENTAL
    assert EtiologyCode("autoimmune") == EtiologyCode.AUTOIMMUNE
    assert EtiologyCode("toxic") == EtiologyCode.TOXIC
    assert EtiologyCode("mechanical") == EtiologyCode.MECHANICAL
    assert EtiologyCode("iatrogenic:post-operative") == EtiologyCode.IATROGENIC_POST_OPERATIVE
    assert EtiologyCode("iatrogenic:post-radiation") == EtiologyCode.IATROGENIC_POST_RADIATION
    assert EtiologyCode("iatrogenic:device") == EtiologyCode.IATROGENIC_DEVICE
    assert EtiologyCode("iatrogenic:medication-related") == EtiologyCode.IATROGENIC_MEDICATION_RELATED
    assert EtiologyCode("idiopathic") == EtiologyCode.IDIOPATHIC
    assert EtiologyCode("normal-variant") == EtiologyCode.NORMAL_VARIANT


def test_etiology_code_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        EtiologyCode("traumatic")  # legacy alias, not canonical
    with pytest.raises(ValueError):
        EtiologyCode("unknown")


# ============================================================================
# AgeProfile Model Tests
# ============================================================================


def test_age_profile_all_ages() -> None:
    profile = AgeProfile(applicability="all_ages")
    assert profile.applicability == "all_ages"
    assert profile.more_common_in is None


def test_age_profile_with_list_of_stages() -> None:
    profile = AgeProfile(applicability=[AgeStage.ADULT, AgeStage.AGED])
    assert profile.applicability == [AgeStage.ADULT, AgeStage.AGED]
    assert profile.more_common_in is None


def test_age_profile_with_more_common_in() -> None:
    profile = AgeProfile(
        applicability=[AgeStage.ADULT, AgeStage.MIDDLE_AGED, AgeStage.AGED],
        more_common_in=[AgeStage.AGED],
    )
    assert profile.applicability == [AgeStage.ADULT, AgeStage.MIDDLE_AGED, AgeStage.AGED]
    assert profile.more_common_in == [AgeStage.AGED]


def test_age_profile_invalid_string_applicability_raises() -> None:
    """Only 'all_ages' is valid as a string; other strings (even plausible ones) must be rejected."""
    with pytest.raises(ValidationError):
        AgeProfile(applicability="none")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AgeProfile(applicability="some_ages")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AgeProfile(applicability="banana")  # type: ignore[arg-type]


def test_age_profile_json_roundtrip_all_ages() -> None:
    profile = AgeProfile(applicability="all_ages")
    serialized = profile.model_dump_json()
    loaded = AgeProfile.model_validate_json(serialized)
    assert loaded.applicability == "all_ages"
    assert loaded.more_common_in is None


def test_age_profile_json_roundtrip_list_form() -> None:
    profile = AgeProfile(
        applicability=[AgeStage.NEWBORN, AgeStage.INFANT],
        more_common_in=[AgeStage.NEWBORN],
    )
    serialized = profile.model_dump_json()
    loaded = AgeProfile.model_validate_json(serialized)
    assert loaded.applicability == [AgeStage.NEWBORN, AgeStage.INFANT]
    assert loaded.more_common_in == [AgeStage.NEWBORN]


def test_age_profile_json_roundtrip_excludes_none() -> None:
    profile = AgeProfile(applicability="all_ages")
    data = json.loads(profile.model_dump_json(exclude_none=True))
    assert "more_common_in" not in data
    assert data["applicability"] == "all_ages"


# ============================================================================
# ExpectedTimeCourse Model Tests
# ============================================================================


def test_expected_time_course_defaults() -> None:
    tc = ExpectedTimeCourse()
    assert tc.duration is None
    assert tc.modifiers == []


def test_expected_time_course_with_duration() -> None:
    tc = ExpectedTimeCourse(duration=ExpectedDuration.WEEKS)
    assert tc.duration == ExpectedDuration.WEEKS
    assert tc.modifiers == []


def test_expected_time_course_with_modifiers() -> None:
    tc = ExpectedTimeCourse(
        duration=ExpectedDuration.PERMANENT,
        modifiers=[TimeCourseModifier.PROGRESSIVE, TimeCourseModifier.RECURRENT],
    )
    assert tc.duration == ExpectedDuration.PERMANENT
    assert tc.modifiers == [TimeCourseModifier.PROGRESSIVE, TimeCourseModifier.RECURRENT]


def test_expected_time_course_json_roundtrip() -> None:
    tc = ExpectedTimeCourse(
        duration=ExpectedDuration.MONTHS,
        modifiers=[TimeCourseModifier.STABLE],
    )
    serialized = tc.model_dump_json()
    loaded = ExpectedTimeCourse.model_validate_json(serialized)
    assert loaded.duration == ExpectedDuration.MONTHS
    assert loaded.modifiers == [TimeCourseModifier.STABLE]


def test_expected_time_course_json_roundtrip_defaults() -> None:
    tc = ExpectedTimeCourse()
    serialized = tc.model_dump_json()
    loaded = ExpectedTimeCourse.model_validate_json(serialized)
    assert loaded.duration is None
    assert loaded.modifiers == []


# ============================================================================
# normalize_body_region Tests
# ============================================================================


def test_normalize_body_region_canonical_passthrough() -> None:
    assert normalize_body_region("head") == BodyRegion.HEAD
    assert normalize_body_region("neck") == BodyRegion.NECK
    assert normalize_body_region("chest") == BodyRegion.CHEST
    assert normalize_body_region("breast") == BodyRegion.BREAST
    assert normalize_body_region("abdomen") == BodyRegion.ABDOMEN
    assert normalize_body_region("pelvis") == BodyRegion.PELVIS
    assert normalize_body_region("spine") == BodyRegion.SPINE
    assert normalize_body_region("upper_extremity") == BodyRegion.UPPER_EXTREMITY
    assert normalize_body_region("lower_extremity") == BodyRegion.LOWER_EXTREMITY
    assert normalize_body_region("whole_body") == BodyRegion.WHOLE_BODY


def test_normalize_body_region_legacy_title_case() -> None:
    assert normalize_body_region("Head") == BodyRegion.HEAD
    assert normalize_body_region("Neck") == BodyRegion.NECK
    assert normalize_body_region("Chest") == BodyRegion.CHEST
    assert normalize_body_region("Breast") == BodyRegion.BREAST
    assert normalize_body_region("Abdomen") == BodyRegion.ABDOMEN
    assert normalize_body_region("Pelvis") == BodyRegion.PELVIS
    assert normalize_body_region("Spine") == BodyRegion.SPINE


def test_normalize_body_region_legacy_aliases() -> None:
    assert normalize_body_region("ALL") == BodyRegion.WHOLE_BODY
    assert normalize_body_region("Arm") == BodyRegion.UPPER_EXTREMITY
    assert normalize_body_region("Leg") == BodyRegion.LOWER_EXTREMITY
    assert normalize_body_region("Upper Extremity") == BodyRegion.UPPER_EXTREMITY
    assert normalize_body_region("Lower Extremity") == BodyRegion.LOWER_EXTREMITY
    assert normalize_body_region("Whole Body") == BodyRegion.WHOLE_BODY


def test_normalize_body_region_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown body region"):
        normalize_body_region("torso")
    with pytest.raises(ValueError, match="Unknown body region"):
        normalize_body_region("arm")  # lowercase alias not in legacy map
    with pytest.raises(ValueError, match="Unknown body region"):
        normalize_body_region("")


# ============================================================================
# normalize_etiology Tests
# ============================================================================


def test_normalize_etiology_canonical_passthrough() -> None:
    assert normalize_etiology("inflammatory") == EtiologyCode.INFLAMMATORY
    assert normalize_etiology("neoplastic:malignant") == EtiologyCode.NEOPLASTIC_MALIGNANT
    assert normalize_etiology("traumatic:acute") == EtiologyCode.TRAUMATIC_ACUTE
    assert normalize_etiology("traumatic:sequela") == EtiologyCode.TRAUMATIC_SEQUELA
    assert normalize_etiology("iatrogenic:device") == EtiologyCode.IATROGENIC_DEVICE
    assert normalize_etiology("idiopathic") == EtiologyCode.IDIOPATHIC
    assert normalize_etiology("normal-variant") == EtiologyCode.NORMAL_VARIANT


def test_normalize_etiology_legacy_aliases() -> None:
    assert normalize_etiology("traumatic") == EtiologyCode.TRAUMATIC_ACUTE
    assert normalize_etiology("post-traumatic") == EtiologyCode.TRAUMATIC_SEQUELA
    assert normalize_etiology("iatrogenic:device-related") == EtiologyCode.IATROGENIC_DEVICE


def test_normalize_etiology_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown etiology code"):
        normalize_etiology("unknown")
    with pytest.raises(ValueError, match="Unknown etiology code"):
        normalize_etiology("Inflammatory")  # title case not valid
    with pytest.raises(ValueError, match="Unknown etiology code"):
        normalize_etiology("")


# ============================================================================
# normalize_modality Tests
# ============================================================================


def test_normalize_modality_canonical_passthrough() -> None:
    assert normalize_modality("XR") == Modality.XR
    assert normalize_modality("CT") == Modality.CT
    assert normalize_modality("MR") == Modality.MR
    assert normalize_modality("US") == Modality.US
    assert normalize_modality("PET") == Modality.PET
    assert normalize_modality("NM") == Modality.NM
    assert normalize_modality("MG") == Modality.MG
    assert normalize_modality("RF") == Modality.RF
    assert normalize_modality("DSA") == Modality.DSA


def test_normalize_modality_legacy_aliases() -> None:
    assert normalize_modality("CR") == Modality.XR
    assert normalize_modality("DX") == Modality.XR


def test_normalize_modality_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown modality"):
        normalize_modality("XRAY")
    with pytest.raises(ValueError, match="Unknown modality"):
        normalize_modality("ct")  # lowercase not valid
    with pytest.raises(ValueError, match="Unknown modality"):
        normalize_modality("")


# ============================================================================
# normalize_age_label Tests
# ============================================================================


def test_normalize_age_label_any_age() -> None:
    profile = normalize_age_label("any age")
    assert profile.applicability == "all_ages"


def test_normalize_age_label_all_ages() -> None:
    profile = normalize_age_label("all ages")
    assert profile.applicability == "all_ages"


def test_normalize_age_label_any() -> None:
    profile = normalize_age_label("any")
    assert profile.applicability == "all_ages"


def test_normalize_age_label_pediatric() -> None:
    profile = normalize_age_label("pediatric")
    assert isinstance(profile.applicability, list)
    assert AgeStage.NEWBORN in profile.applicability
    assert AgeStage.INFANT in profile.applicability
    assert AgeStage.PRESCHOOL_CHILD in profile.applicability
    assert AgeStage.CHILD in profile.applicability
    assert AgeStage.ADOLESCENT in profile.applicability
    assert AgeStage.ADULT not in profile.applicability


def test_normalize_age_label_adult() -> None:
    profile = normalize_age_label("adult")
    assert isinstance(profile.applicability, list)
    assert AgeStage.ADULT in profile.applicability
    assert AgeStage.MIDDLE_AGED in profile.applicability
    assert AgeStage.AGED in profile.applicability
    assert AgeStage.CHILD not in profile.applicability


def test_normalize_age_label_elderly() -> None:
    profile = normalize_age_label("elderly")
    assert isinstance(profile.applicability, list)
    assert profile.applicability == [AgeStage.AGED]


def test_normalize_age_label_neonatal() -> None:
    profile = normalize_age_label("neonatal")
    assert isinstance(profile.applicability, list)
    assert profile.applicability == [AgeStage.NEWBORN]


def test_normalize_age_label_case_insensitive() -> None:
    assert normalize_age_label("ANY AGE").applicability == "all_ages"
    assert normalize_age_label("Pediatric").applicability == normalize_age_label("pediatric").applicability
    assert normalize_age_label("ADULT").applicability == normalize_age_label("adult").applicability


def test_normalize_age_label_strips_whitespace() -> None:
    assert normalize_age_label("  any age  ").applicability == "all_ages"
    assert normalize_age_label("\tpediatric\t").applicability == normalize_age_label("pediatric").applicability


def test_normalize_age_label_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown age label"):
        normalize_age_label("teenager")
    with pytest.raises(ValueError, match="Unknown age label"):
        normalize_age_label("middle aged")
    with pytest.raises(ValueError, match="Unknown age label"):
        normalize_age_label("")


# ============================================================================
# FindingModelBase Integration Tests
# ============================================================================


def _make_base_model(**extra: Any) -> FindingModelBase:
    return FindingModelBase(
        name="Test Finding",
        description="A test finding for facet tests.",
        attributes=[
            ChoiceAttribute(
                name="Presence",
                values=[ChoiceValue(name="Present"), ChoiceValue(name="Absent")],
            )
        ],
        **extra,
    )


def test_finding_model_base_without_new_fields() -> None:
    model = _make_base_model()
    assert model.body_regions is None
    assert model.subspecialties is None
    assert model.etiologies is None
    assert model.entity_type is None
    assert model.applicable_modalities is None
    assert model.expected_time_course is None
    assert model.age_profile is None
    assert model.sex_specificity is None


def test_finding_model_base_with_body_regions() -> None:
    model = _make_base_model(body_regions=[BodyRegion.HEAD, BodyRegion.NECK])
    assert model.body_regions == [BodyRegion.HEAD, BodyRegion.NECK]


def test_finding_model_base_with_subspecialties() -> None:
    model = _make_base_model(subspecialties=[Subspecialty.NR, Subspecialty.HN])
    assert model.subspecialties == [Subspecialty.NR, Subspecialty.HN]


def test_finding_model_base_with_etiologies() -> None:
    model = _make_base_model(etiologies=[EtiologyCode.NEOPLASTIC_MALIGNANT, EtiologyCode.INFLAMMATORY])
    assert model.etiologies == [EtiologyCode.NEOPLASTIC_MALIGNANT, EtiologyCode.INFLAMMATORY]


def test_finding_model_base_with_entity_type() -> None:
    model = _make_base_model(entity_type=EntityType.FINDING)
    assert model.entity_type == EntityType.FINDING


def test_finding_model_base_with_applicable_modalities() -> None:
    model = _make_base_model(applicable_modalities=[Modality.CT, Modality.MR])
    assert model.applicable_modalities == [Modality.CT, Modality.MR]


def test_finding_model_base_with_expected_time_course() -> None:
    tc = ExpectedTimeCourse(duration=ExpectedDuration.YEARS, modifiers=[TimeCourseModifier.PROGRESSIVE])
    model = _make_base_model(expected_time_course=tc)
    assert model.expected_time_course is not None
    assert model.expected_time_course.duration == ExpectedDuration.YEARS
    assert model.expected_time_course.modifiers == [TimeCourseModifier.PROGRESSIVE]


def test_finding_model_base_with_age_profile() -> None:
    profile = AgeProfile(applicability=[AgeStage.ADULT, AgeStage.MIDDLE_AGED])
    model = _make_base_model(age_profile=profile)
    assert model.age_profile is not None
    assert model.age_profile.applicability == [AgeStage.ADULT, AgeStage.MIDDLE_AGED]


def test_finding_model_base_with_sex_specificity() -> None:
    model = _make_base_model(sex_specificity=SexSpecificity.FEMALE_SPECIFIC)
    assert model.sex_specificity == SexSpecificity.FEMALE_SPECIFIC


def test_finding_model_base_with_all_new_fields() -> None:
    tc = ExpectedTimeCourse(duration=ExpectedDuration.YEARS, modifiers=[TimeCourseModifier.PROGRESSIVE])
    profile = AgeProfile(applicability=[AgeStage.ADULT, AgeStage.AGED], more_common_in=[AgeStage.AGED])
    model = _make_base_model(
        body_regions=[BodyRegion.CHEST],
        subspecialties=[Subspecialty.CH],
        etiologies=[EtiologyCode.NEOPLASTIC_MALIGNANT],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT, Modality.PET],
        expected_time_course=tc,
        age_profile=profile,
        sex_specificity=SexSpecificity.SEX_NEUTRAL,
    )
    assert model.body_regions == [BodyRegion.CHEST]
    assert model.subspecialties == [Subspecialty.CH]
    assert model.etiologies == [EtiologyCode.NEOPLASTIC_MALIGNANT]
    assert model.entity_type == EntityType.FINDING
    assert model.applicable_modalities == [Modality.CT, Modality.PET]
    assert model.expected_time_course == tc
    assert model.age_profile == profile
    assert model.sex_specificity == SexSpecificity.SEX_NEUTRAL


def test_finding_model_base_json_roundtrip_with_new_fields() -> None:
    tc = ExpectedTimeCourse(duration=ExpectedDuration.DAYS, modifiers=[TimeCourseModifier.RESOLVING])
    profile = AgeProfile(applicability="all_ages")
    model = _make_base_model(
        body_regions=[BodyRegion.ABDOMEN],
        etiologies=[EtiologyCode.INFLAMMATORY],
        entity_type=EntityType.DIAGNOSIS,
        applicable_modalities=[Modality.CT],
        expected_time_course=tc,
        age_profile=profile,
        sex_specificity=SexSpecificity.SEX_NEUTRAL,
    )
    serialized = model.model_dump_json()
    loaded = FindingModelBase.model_validate_json(serialized)
    assert loaded.body_regions == [BodyRegion.ABDOMEN]
    assert loaded.etiologies == [EtiologyCode.INFLAMMATORY]
    assert loaded.entity_type == EntityType.DIAGNOSIS
    assert loaded.applicable_modalities == [Modality.CT]
    assert loaded.expected_time_course is not None
    assert loaded.expected_time_course.duration == ExpectedDuration.DAYS
    assert loaded.age_profile is not None
    assert loaded.age_profile.applicability == "all_ages"
    assert loaded.sex_specificity == SexSpecificity.SEX_NEUTRAL


def test_finding_model_base_json_exclude_none_omits_absent_fields() -> None:
    model = _make_base_model(entity_type=EntityType.FINDING)
    data = json.loads(model.model_dump_json(exclude_none=True))
    # Present field survives
    assert data["entity_type"] == "finding"
    # Absent optional fields are omitted
    assert "body_regions" not in data
    assert "age_profile" not in data
    assert "sex_specificity" not in data
    assert "expected_time_course" not in data


def test_finding_model_base_json_exclude_none_with_populated_fields() -> None:
    profile = AgeProfile(applicability=[AgeStage.ADULT])
    model = _make_base_model(
        body_regions=[BodyRegion.HEAD],
        age_profile=profile,
    )
    data = json.loads(model.model_dump_json(exclude_none=True))
    assert "body_regions" in data
    assert data["body_regions"] == ["head"]
    assert "age_profile" in data
    assert data["age_profile"]["applicability"] == ["adult"]


# ============================================================================
# FindingModelFull Integration Tests
# ============================================================================


def _make_full_model(**extra: Any) -> FindingModelFull:
    return FindingModelFull(
        oifm_id="OIFM_TEST_000001",
        name="Test Finding Full",
        description="A test finding model with IDs for facet tests.",
        attributes=[
            ChoiceAttributeIded(
                oifma_id="OIFMA_TEST_000001",
                name="Presence",
                values=[
                    ChoiceValueIded(value_code="OIFMA_TEST_000001.0", name="Present"),
                    ChoiceValueIded(value_code="OIFMA_TEST_000001.1", name="Absent"),
                ],
            )
        ],
        **extra,
    )


def test_finding_model_full_without_new_fields() -> None:
    model = _make_full_model()
    assert model.body_regions is None
    assert model.subspecialties is None
    assert model.etiologies is None
    assert model.entity_type is None
    assert model.applicable_modalities is None
    assert model.expected_time_course is None
    assert model.age_profile is None
    assert model.sex_specificity is None


def test_finding_model_full_with_new_fields() -> None:
    tc = ExpectedTimeCourse(duration=ExpectedDuration.PERMANENT)
    profile = AgeProfile(applicability=[AgeStage.ADULT, AgeStage.MIDDLE_AGED, AgeStage.AGED])
    model = _make_full_model(
        body_regions=[BodyRegion.SPINE],
        subspecialties=[Subspecialty.MK],
        etiologies=[EtiologyCode.DEGENERATIVE],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.MR, Modality.XR],
        expected_time_course=tc,
        age_profile=profile,
        sex_specificity=SexSpecificity.SEX_NEUTRAL,
    )
    assert model.body_regions == [BodyRegion.SPINE]
    assert model.subspecialties == [Subspecialty.MK]
    assert model.etiologies == [EtiologyCode.DEGENERATIVE]
    assert model.entity_type == EntityType.FINDING
    assert model.applicable_modalities == [Modality.MR, Modality.XR]
    assert model.expected_time_course is not None
    assert model.expected_time_course.duration == ExpectedDuration.PERMANENT
    assert model.age_profile is not None
    assert model.age_profile.applicability == [AgeStage.ADULT, AgeStage.MIDDLE_AGED, AgeStage.AGED]
    assert model.sex_specificity == SexSpecificity.SEX_NEUTRAL


def test_finding_model_full_json_roundtrip_with_new_fields() -> None:
    tc = ExpectedTimeCourse(duration=ExpectedDuration.WEEKS, modifiers=[TimeCourseModifier.FLUCTUATING])
    profile = AgeProfile(applicability=[AgeStage.YOUNG_ADULT, AgeStage.ADULT])
    model = _make_full_model(
        body_regions=[BodyRegion.PELVIS, BodyRegion.ABDOMEN],
        subspecialties=[Subspecialty.GU, Subspecialty.AB],
        etiologies=[EtiologyCode.NEOPLASTIC_BENIGN],
        entity_type=EntityType.DIAGNOSIS,
        applicable_modalities=[Modality.CT, Modality.MR, Modality.US],
        expected_time_course=tc,
        age_profile=profile,
        sex_specificity=SexSpecificity.FEMALE_SPECIFIC,
    )
    serialized = model.model_dump_json()
    loaded = FindingModelFull.model_validate_json(serialized)
    assert loaded.oifm_id == "OIFM_TEST_000001"
    assert loaded.body_regions == [BodyRegion.PELVIS, BodyRegion.ABDOMEN]
    assert loaded.subspecialties == [Subspecialty.GU, Subspecialty.AB]
    assert loaded.etiologies == [EtiologyCode.NEOPLASTIC_BENIGN]
    assert loaded.entity_type == EntityType.DIAGNOSIS
    assert loaded.applicable_modalities == [Modality.CT, Modality.MR, Modality.US]
    assert loaded.expected_time_course is not None
    assert loaded.expected_time_course.duration == ExpectedDuration.WEEKS
    assert loaded.expected_time_course.modifiers == [TimeCourseModifier.FLUCTUATING]
    assert loaded.age_profile is not None
    assert loaded.age_profile.applicability == [AgeStage.YOUNG_ADULT, AgeStage.ADULT]
    assert loaded.sex_specificity == SexSpecificity.FEMALE_SPECIFIC


def test_finding_model_full_json_exclude_none_omits_absent_fields() -> None:
    model = _make_full_model(entity_type=EntityType.GROUPING)
    data = json.loads(model.model_dump_json(exclude_none=True))
    # Present field survives
    assert data["entity_type"] == "grouping"
    # Absent optional fields are omitted
    assert "body_regions" not in data
    assert "age_profile" not in data
    assert "sex_specificity" not in data
    assert "expected_time_course" not in data
    assert "etiologies" not in data


def test_finding_model_full_json_exclude_none_with_populated_fields() -> None:
    profile = AgeProfile(applicability="all_ages")
    model = _make_full_model(
        applicable_modalities=[Modality.CT],
        age_profile=profile,
    )
    data = json.loads(model.model_dump_json(exclude_none=True))
    assert "applicable_modalities" in data
    assert data["applicable_modalities"] == ["CT"]
    assert "age_profile" in data
    assert data["age_profile"]["applicability"] == "all_ages"
    # None-valued more_common_in is excluded within the nested model too
    assert "more_common_in" not in data["age_profile"]


def test_finding_model_full_age_profile_numeric_attribute() -> None:
    """Test FindingModelFull with a NumericAttributeIded and age_profile."""
    profile = AgeProfile(applicability=[AgeStage.CHILD, AgeStage.ADOLESCENT])
    model = FindingModelFull(
        oifm_id="OIFM_TEST_000002",
        name="Pediatric Size Measurement",
        description="A measurement finding applicable to children and adolescents.",
        age_profile=profile,
        entity_type=EntityType.MEASUREMENT,
        applicable_modalities=[Modality.US],
        attributes=[
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_000002",
                name="Diameter",
                minimum=0,
                maximum=20,
                unit="cm",
            )
        ],
    )
    assert model.age_profile is not None
    assert model.entity_type == EntityType.MEASUREMENT
    assert isinstance(model.age_profile.applicability, list)
    assert AgeStage.CHILD in model.age_profile.applicability
    assert AgeStage.ADOLESCENT in model.age_profile.applicability


# ============================================================================
# AgeProfile Empty List Rejection
# ============================================================================


def test_age_profile_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        AgeProfile(applicability=[])


# ============================================================================
# Legacy Normalization on Model Load
# ============================================================================


def test_base_model_normalizes_legacy_body_regions() -> None:
    """Legacy title-cased body regions should auto-normalize during validation."""
    model = _make_base_model(body_regions=["Head", "Chest", "Arm", "ALL"])
    assert model.body_regions == [BodyRegion.HEAD, BodyRegion.CHEST, BodyRegion.UPPER_EXTREMITY, BodyRegion.WHOLE_BODY]


def test_base_model_normalizes_legacy_modalities() -> None:
    """Legacy CR/DX should auto-normalize to XR during validation."""
    model = _make_base_model(applicable_modalities=["CR", "DX", "CT"])
    assert model.applicable_modalities == [Modality.XR, Modality.XR, Modality.CT]


def test_base_model_normalizes_legacy_etiologies() -> None:
    """Legacy etiology shorthands should auto-normalize during validation."""
    model = _make_base_model(etiologies=["traumatic", "post-traumatic", "iatrogenic:device-related"])
    assert model.etiologies == [
        EtiologyCode.TRAUMATIC_ACUTE,
        EtiologyCode.TRAUMATIC_SEQUELA,
        EtiologyCode.IATROGENIC_DEVICE,
    ]


def test_full_model_normalizes_legacy_body_regions() -> None:
    model = _make_full_model(body_regions=["Leg", "ALL"])
    assert model.body_regions == [BodyRegion.LOWER_EXTREMITY, BodyRegion.WHOLE_BODY]


def test_full_model_normalizes_legacy_modalities() -> None:
    model = _make_full_model(applicable_modalities=["DX", "MR"])
    assert model.applicable_modalities == [Modality.XR, Modality.MR]


def test_full_model_normalizes_legacy_etiologies() -> None:
    model = _make_full_model(etiologies=["traumatic", "degenerative"])
    assert model.etiologies == [EtiologyCode.TRAUMATIC_ACUTE, EtiologyCode.DEGENERATIVE]


def test_json_load_normalizes_legacy_values() -> None:
    """JSON with legacy values should normalize on model_validate_json."""
    import json

    data = {
        "name": "Test Finding",
        "description": "Test finding for normalization.",
        "body_regions": ["Head", "Arm"],
        "applicable_modalities": ["CR"],
        "etiologies": ["traumatic"],
        "attributes": [{"name": "Presence", "type": "choice", "values": [{"name": "Present"}, {"name": "Absent"}]}],
    }
    model = FindingModelBase.model_validate_json(json.dumps(data))
    assert model.body_regions == [BodyRegion.HEAD, BodyRegion.UPPER_EXTREMITY]
    assert model.applicable_modalities == [Modality.XR]
    assert model.etiologies == [EtiologyCode.TRAUMATIC_ACUTE]


# ============================================================================
# NormalizedAgeProfile on Model Load
# ============================================================================


def test_age_profile_from_legacy_string_on_model_load() -> None:
    """A plain string like 'pediatric' should auto-normalize to AgeProfile on load."""
    data = {
        "name": "Test Finding",
        "description": "Test finding for age normalization.",
        "age_profile": "pediatric",
        "attributes": [{"name": "Presence", "type": "choice", "values": [{"name": "Present"}, {"name": "Absent"}]}],
    }
    model = FindingModelBase.model_validate_json(json.dumps(data))
    assert model.age_profile is not None
    assert isinstance(model.age_profile.applicability, list)
    assert AgeStage.CHILD in model.age_profile.applicability
    assert AgeStage.ADOLESCENT in model.age_profile.applicability


def test_age_profile_from_legacy_string_all_ages() -> None:
    data = {
        "name": "Test Finding",
        "description": "Test finding for all ages.",
        "age_profile": "any age",
        "attributes": [{"name": "Presence", "type": "choice", "values": [{"name": "Present"}, {"name": "Absent"}]}],
    }
    model = FindingModelBase.model_validate_json(json.dumps(data))
    assert model.age_profile is not None
    assert model.age_profile.applicability == "all_ages"


# ============================================================================
# Formatting Helpers
# ============================================================================


def test_format_age_profile_all_ages() -> None:
    from findingmodel.facets import format_age_profile

    ap = AgeProfile(applicability="all_ages")
    assert format_age_profile(ap) == "all ages"


def test_format_age_profile_with_stages() -> None:
    from findingmodel.facets import format_age_profile

    ap = AgeProfile(applicability=[AgeStage.ADULT, AgeStage.MIDDLE_AGED])
    assert format_age_profile(ap) == "applicable: adult, middle_aged"


def test_format_age_profile_with_more_common_in() -> None:
    from findingmodel.facets import format_age_profile

    ap = AgeProfile(applicability=[AgeStage.ADULT, AgeStage.MIDDLE_AGED, AgeStage.AGED], more_common_in=[AgeStage.AGED])
    assert format_age_profile(ap) == "applicable: adult, middle_aged, aged; more common in: aged"


def test_format_time_course_duration_only() -> None:
    from findingmodel.facets import format_time_course

    tc = ExpectedTimeCourse(duration=ExpectedDuration.WEEKS)
    assert format_time_course(tc) == "duration: weeks"


def test_format_time_course_modifiers_only() -> None:
    from findingmodel.facets import format_time_course

    tc = ExpectedTimeCourse(modifiers=[TimeCourseModifier.RESOLVING, TimeCourseModifier.EVOLVING])
    assert format_time_course(tc) == "modifiers: resolving, evolving"


def test_format_time_course_both() -> None:
    from findingmodel.facets import format_time_course

    tc = ExpectedTimeCourse(duration=ExpectedDuration.DAYS, modifiers=[TimeCourseModifier.RESOLVING])
    assert format_time_course(tc) == "duration: days; modifiers: resolving"


def test_format_time_course_empty() -> None:
    from findingmodel.facets import format_time_course

    tc = ExpectedTimeCourse()
    assert format_time_course(tc) == ""
