"""Tests verifying the type restructure preserves public API and behavior."""

from __future__ import annotations

import pytest
from findingmodel import (
    AgeProfile,
    AgeStage,
    AttributeType,
    BodyRegion,
    ChoiceAttribute,
    ChoiceAttributeIded,
    ChoiceValue,
    ChoiceValueIded,
    EmbeddingCache,
    EntityType,
    EtiologyCode,
    ExpectedDuration,
    ExpectedTimeCourse,
    FindingInfo,
    FindingModelBase,
    FindingModelFull,
    Index,
    IndexCode,
    IndexCodeList,
    Modality,
    NumericAttribute,
    NumericAttributeIded,
    RelatedModelWeights,
    SexSpecificity,
    Subspecialty,
    TimeCourseModifier,
    WebReference,
    create_model_stub_from_info,
    format_age_profile,
    format_time_course,
    get_settings,
    logger,
    tools,
)

# ---------------------------------------------------------------------------
# 1. Top-level imports
# ---------------------------------------------------------------------------


def test_top_level_imports() -> None:
    """All documented public symbols are importable from `findingmodel`."""
    symbols = [
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
        format_age_profile,
        format_time_course,
        AttributeType,
        ChoiceValue,
        ChoiceValueIded,
        ChoiceAttribute,
        ChoiceAttributeIded,
        NumericAttribute,
        NumericAttributeIded,
        FindingModelBase,
        FindingModelFull,
        IndexCodeList,
        FindingInfo,
        Index,
        RelatedModelWeights,
        IndexCode,
        WebReference,
        EmbeddingCache,
        create_model_stub_from_info,
        get_settings,
        logger,
        tools,
    ]
    assert all(x is not None for x in symbols)


# ---------------------------------------------------------------------------
# 2. Class identity preserved
# ---------------------------------------------------------------------------


def test_class_identity_finding_model_full() -> None:
    """Top-level FindingModelFull is the same class object as the internal path."""
    from findingmodel import FindingModelFull as TopLevel
    from findingmodel.types.models import FindingModelFull as Internal

    assert TopLevel is Internal


def test_class_identity_finding_model_base() -> None:
    from findingmodel import FindingModelBase as TopLevel
    from findingmodel.types.models import FindingModelBase as Internal

    assert TopLevel is Internal


def test_class_identity_metadata_enums() -> None:
    from findingmodel import BodyRegion as TopLevel
    from findingmodel.types.metadata import BodyRegion as Internal

    assert TopLevel is Internal

    from findingmodel import EntityType as TopLevel2
    from findingmodel.types.metadata import EntityType as Internal2

    assert TopLevel2 is Internal2


def test_class_identity_attributes() -> None:
    from findingmodel import ChoiceAttribute as TopLevel
    from findingmodel.types.attributes import ChoiceAttribute as Internal

    assert TopLevel is Internal

    from findingmodel import NumericAttributeIded as TopLevel2
    from findingmodel.types.attributes import NumericAttributeIded as Internal2

    assert TopLevel2 is Internal2


# ---------------------------------------------------------------------------
# 3. Round-trip serialization
# ---------------------------------------------------------------------------


def test_finding_model_full_round_trip(full_model: FindingModelFull) -> None:
    """Serialize to JSON and deserialize back — model should be equal."""
    json_data = full_model.model_dump(mode="json")
    restored = FindingModelFull.model_validate(json_data)
    assert restored == full_model


def test_finding_model_full_round_trip_json_string(full_model: FindingModelFull) -> None:
    """Serialize to JSON string and deserialize — model should be equal."""
    json_str = full_model.model_dump_json()
    restored = FindingModelFull.model_validate_json(json_str)
    assert restored == full_model


def test_finding_model_base_round_trip(base_model: FindingModelBase) -> None:
    json_data = base_model.model_dump(mode="json")
    restored = FindingModelBase.model_validate(json_data)
    assert restored == base_model


def test_real_model_round_trip(real_model: FindingModelFull) -> None:
    """A real-world model (pulmonary embolism) survives round-trip."""
    json_str = real_model.model_dump_json()
    restored = FindingModelFull.model_validate_json(json_str)
    assert restored == real_model


# ---------------------------------------------------------------------------
# 4. ID generation from new location
# ---------------------------------------------------------------------------


def test_generate_oifm_id() -> None:
    from findingmodel._id_gen import generate_oifm_id

    oifm = generate_oifm_id("TEST")
    assert oifm.startswith("OIFM_TEST_")
    assert len(oifm) == len("OIFM_TEST_") + 6
    assert oifm[len("OIFM_TEST_") :].isdigit()


def test_generate_oifma_id() -> None:
    from findingmodel._id_gen import generate_oifma_id

    oifma = generate_oifma_id("TEST")
    assert oifma.startswith("OIFMA_TEST_")
    assert len(oifma) == len("OIFMA_TEST_") + 6
    assert oifma[len("OIFMA_TEST_") :].isdigit()


def test_random_digits_length() -> None:
    from findingmodel._id_gen import _random_digits

    for length in [1, 6, 10]:
        result = _random_digits(length)
        assert len(result) == length
        assert result.isdigit()


def test_generate_oifm_id_uppercase_source() -> None:
    from findingmodel._id_gen import generate_oifm_id

    oifm = generate_oifm_id("lower")
    assert oifm.startswith("OIFM_LOWER_")


def test_generated_ids_are_unique() -> None:
    """IDs should be random — two calls should (almost certainly) differ."""
    from findingmodel._id_gen import generate_oifm_id

    ids = {generate_oifm_id("TEST") for _ in range(20)}
    # With 6 random digits, 20 unique IDs is overwhelmingly likely
    assert len(ids) >= 15


# ---------------------------------------------------------------------------
# 5. Markdown rendering
# ---------------------------------------------------------------------------


def test_markdown_rendering_unchanged(full_model: FindingModelFull, base_model: FindingModelBase) -> None:
    """as_markdown() produces expected content for both FindingModelFull and FindingModelBase."""
    # FindingModelFull markdown
    full_md = full_model.as_markdown()
    assert full_model.name.lower() in full_md.lower()
    assert full_model.description in full_md
    # Attribute names should appear in the markdown
    for attr in full_model.attributes:
        assert attr.name in full_md

    # FindingModelBase markdown
    base_md = base_model.as_markdown()
    assert base_model.name.lower() in base_md.lower()
    assert base_model.description in base_md
    for attr in base_model.attributes:
        assert attr.name in base_md


# ---------------------------------------------------------------------------
# 6. Attribute validation
# ---------------------------------------------------------------------------


def test_attribute_validation_unchanged() -> None:
    """ChoiceAttribute validators and NumericAttribute round-trip work correctly."""
    # max_selected="all" should be fixed to len(values)
    attr = ChoiceAttribute(
        name="Test Choice",
        description="A choice attribute.",
        values=[ChoiceValue(name="A"), ChoiceValue(name="B"), ChoiceValue(name="C")],
        max_selected="all",  # type: ignore[arg-type]
    )
    assert attr.max_selected == 3

    # values with fewer than 2 items should raise ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChoiceAttribute(
            name="Bad Choice",
            description="Not enough values.",
            values=[ChoiceValue(name="Only")],
        )

    # NumericAttribute round-trip
    num = NumericAttribute(
        name="Length",
        description="Measurement in mm.",
        minimum=0,
        maximum=100,
        unit="mm",
        required=True,
    )
    data = num.model_dump(mode="json")
    restored = NumericAttribute.model_validate(data)
    assert restored == num
    assert restored.unit == "mm"
    assert restored.minimum == 0
    assert restored.maximum == 100
