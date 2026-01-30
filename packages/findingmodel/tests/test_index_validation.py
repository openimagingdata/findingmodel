"""Tests for shared index validation logic.

This module tests the protocol-based validation functions that work with any index backend
(DuckDB, MongoDB, etc.) without tight coupling.
"""

import pytest
from findingmodel.finding_model import (
    ChoiceAttributeIded,
    ChoiceValueIded,
    FindingModelFull,
    NumericAttributeIded,
)
from findingmodel.index_validation import (
    ValidationContext,
    check_attribute_id_conflict,
    check_name_conflict,
    check_oifm_id_conflict,
    validate_finding_model,
)


# Mock ValidationContext implementation for testing
class MockValidationContext:
    """Simple dict-based ValidationContext for testing without a database."""

    def __init__(
        self,
        existing_ids: set[str] | None = None,
        existing_names: set[str] | None = None,
        attribute_ids_by_model: dict[str, str] | None = None,
    ) -> None:
        self.existing_ids = existing_ids or set()
        self.existing_names = existing_names or set()
        self.attribute_ids_by_model = attribute_ids_by_model or {}

    async def get_existing_oifm_ids(self) -> set[str]:
        return self.existing_ids

    async def get_existing_names(self) -> set[str]:
        return self.existing_names

    async def get_attribute_ids_by_model(self) -> dict[str, str]:
        return self.attribute_ids_by_model


@pytest.fixture
def another_model() -> FindingModelFull:
    """Create another model with different IDs for conflict testing."""
    return FindingModelFull(
        oifm_id="OIFM_TEST_999999",
        name="Another Test Model",
        description="Another test model.",
        attributes=[
            ChoiceAttributeIded(
                oifma_id="OIFMA_TEST_999001",
                name="Status",
                description="Status of the finding",
                values=[
                    ChoiceValueIded(value_code="OIFMA_TEST_999001.0", name="Active"),
                    ChoiceValueIded(value_code="OIFMA_TEST_999001.1", name="Resolved"),
                ],
                required=True,
                max_selected=1,
            ),
        ],
    )


# ValidationContext Protocol Tests
@pytest.mark.asyncio
async def test_mock_validation_context_implements_protocol() -> None:
    """Test that MockValidationContext correctly implements the ValidationContext protocol."""
    context = MockValidationContext(
        existing_ids={"OIFM_TEST_000001"},
        existing_names={"test model"},
        attribute_ids_by_model={"OIFMA_TEST_000001": "OIFM_TEST_000001"},
    )

    # Test that all protocol methods work
    ids = await context.get_existing_oifm_ids()
    assert ids == {"OIFM_TEST_000001"}

    names = await context.get_existing_names()
    assert names == {"test model"}

    attrs = await context.get_attribute_ids_by_model()
    assert attrs == {"OIFMA_TEST_000001": "OIFM_TEST_000001"}


@pytest.mark.asyncio
async def test_validation_context_accepts_different_implementations() -> None:
    """Test that ValidationContext protocol works with multiple implementations."""

    # Create another implementation
    class AlternativeContext:
        async def get_existing_oifm_ids(self) -> set[str]:
            return {"ID1", "ID2"}

        async def get_existing_names(self) -> set[str]:
            return {"name1", "name2"}

        async def get_attribute_ids_by_model(self) -> dict[str, str]:
            return {"ATTR1": "ID1"}

    # Both implementations should work with validation functions
    mock_ctx: ValidationContext = MockValidationContext()
    alt_ctx: ValidationContext = AlternativeContext()

    assert await mock_ctx.get_existing_oifm_ids() == set()
    assert await alt_ctx.get_existing_oifm_ids() == {"ID1", "ID2"}


# check_oifm_id_conflict() Tests
def test_check_oifm_id_conflict_no_conflict(full_model: FindingModelFull) -> None:
    """Test that no conflict is detected when ID is unique."""
    existing_ids = {"OIFM_DIFF_000001", "OIFM_DIFF_000002"}
    errors = check_oifm_id_conflict(full_model, existing_ids)
    assert errors == []


def test_check_oifm_id_conflict_detects_duplicate(full_model: FindingModelFull) -> None:
    """Test that conflict is detected when ID already exists."""
    existing_ids = {"OIFM_TEST_123456", "OIFM_DIFF_000001"}
    errors = check_oifm_id_conflict(full_model, existing_ids)
    assert len(errors) == 1
    assert "OIFM_TEST_123456" in errors[0]
    assert "already exists" in errors[0]


def test_check_oifm_id_conflict_allow_self_permits_duplicate(full_model: FindingModelFull) -> None:
    """Test that allow_self=True permits the model's own ID (for updates)."""
    existing_ids = {"OIFM_TEST_123456", "OIFM_DIFF_000001"}
    errors = check_oifm_id_conflict(full_model, existing_ids, allow_self=True)
    assert errors == []


def test_check_oifm_id_conflict_allow_self_false_blocks_duplicate(full_model: FindingModelFull) -> None:
    """Test that allow_self=False blocks the model's own ID."""
    existing_ids = {"OIFM_TEST_123456"}
    errors = check_oifm_id_conflict(full_model, existing_ids, allow_self=False)
    assert len(errors) == 1
    assert "OIFM_TEST_123456" in errors[0]


def test_check_oifm_id_conflict_empty_existing_set(full_model: FindingModelFull) -> None:
    """Test with empty existing IDs set."""
    existing_ids: set[str] = set()
    errors = check_oifm_id_conflict(full_model, existing_ids)
    assert errors == []


# check_name_conflict() Tests
def test_check_name_conflict_no_conflict(full_model: FindingModelFull) -> None:
    """Test that no conflict is detected when name is unique."""
    existing_names = {"different model", "another model"}
    errors = check_name_conflict(full_model, existing_names)
    assert errors == []


def test_check_name_conflict_detects_case_insensitive_duplicate(full_model: FindingModelFull) -> None:
    """Test that conflict is detected with case-insensitive matching."""
    # Model name is "Test Model", should match "test model" (case-folded)
    existing_names = {"test model", "other model"}
    errors = check_name_conflict(full_model, existing_names)
    assert len(errors) == 1
    assert "Test Model" in errors[0]
    assert "already in use" in errors[0]


def test_check_name_conflict_detects_exact_case_match(full_model: FindingModelFull) -> None:
    """Test detection of exact case match."""
    # Existing names should be case-folded in the set
    existing_names = {"test model", "other model"}
    errors = check_name_conflict(full_model, existing_names)
    assert len(errors) == 1
    assert "Test Model" in errors[0]


def test_check_name_conflict_detects_normalized_slug(full_model: FindingModelFull) -> None:
    """Test that normalized slug names are checked."""
    # "Test Model" normalizes to "test_model"
    existing_names = {"test_model", "other_model"}
    errors = check_name_conflict(full_model, existing_names)
    assert len(errors) >= 1
    # Should detect the conflict
    assert any("test_model" in error.lower() or "test model" in error.lower() for error in errors)


def test_check_name_conflict_allow_self_permits_duplicate(full_model: FindingModelFull) -> None:
    """Test that allow_self=True permits the model's own name (for updates)."""
    existing_names = {"test model"}
    errors = check_name_conflict(full_model, existing_names, allow_self=True)
    assert errors == []


def test_check_name_conflict_allow_self_false_blocks_duplicate(full_model: FindingModelFull) -> None:
    """Test that allow_self=False blocks the model's own name."""
    existing_names = {"test model"}
    errors = check_name_conflict(full_model, existing_names, allow_self=False)
    assert len(errors) == 1


def test_check_name_conflict_empty_existing_set(full_model: FindingModelFull) -> None:
    """Test with empty existing names set."""
    existing_names: set[str] = set()
    errors = check_name_conflict(full_model, existing_names)
    assert errors == []


def test_check_name_conflict_various_cases(full_model: FindingModelFull) -> None:
    """Test case-insensitive matching with various case combinations."""
    test_cases = [
        "test model",  # lowercase
        "TEST MODEL",  # uppercase
        "Test Model",  # titlecase
        "TeSt MoDeL",  # mixed case
    ]

    for existing_name in test_cases:
        existing_names = {existing_name.casefold()}  # Store case-folded in set
        errors = check_name_conflict(full_model, existing_names)
        assert len(errors) >= 1, f"Should detect conflict for '{existing_name}'"


# check_attribute_id_conflict() Tests
def test_check_attribute_id_conflict_no_conflict(full_model: FindingModelFull) -> None:
    """Test that no conflict is detected when all attribute IDs are unique."""
    attribute_map = {
        "OIFMA_DIFF_000001": "OIFM_OTHER_000001",
        "OIFMA_DIFF_000002": "OIFM_OTHER_000002",
    }
    errors = check_attribute_id_conflict(full_model, attribute_map)
    assert errors == []


def test_check_attribute_id_conflict_detects_duplicate_in_different_model(full_model: FindingModelFull) -> None:
    """Test that conflict is detected when attribute ID is used by another model."""
    # Model has OIFMA_TEST_123456, claim it's owned by a different model
    attribute_map = {
        "OIFMA_TEST_123456": "OIFM_OTHER_999999",  # Different model owns this attribute
    }
    errors = check_attribute_id_conflict(full_model, attribute_map)
    assert len(errors) == 1
    assert "OIFMA_TEST_123456" in errors[0]
    assert "OIFM_OTHER_999999" in errors[0]
    assert "already used by model" in errors[0]


def test_check_attribute_id_conflict_allow_self_permits_own_attributes(full_model: FindingModelFull) -> None:
    """Test that allow_self permits the same model's attributes (for updates)."""
    # Map attributes to the model's own ID
    attribute_map = {
        "OIFMA_TEST_123456": "OIFM_TEST_123456",  # Same model owns these
        "OIFMA_TEST_654321": "OIFM_TEST_123456",
    }
    errors = check_attribute_id_conflict(full_model, attribute_map, allow_self=True)
    assert errors == []


def test_check_attribute_id_conflict_detects_multiple_conflicts(full_model: FindingModelFull) -> None:
    """Test detection of conflicts in multiple attributes."""
    # Both attributes conflict with different models
    attribute_map = {
        "OIFMA_TEST_123456": "OIFM_OTHER_111111",
        "OIFMA_TEST_654321": "OIFM_OTHER_222222",
    }
    errors = check_attribute_id_conflict(full_model, attribute_map)
    assert len(errors) == 2
    assert any("OIFMA_TEST_123456" in error for error in errors)
    assert any("OIFMA_TEST_654321" in error for error in errors)


def test_check_attribute_id_conflict_some_conflicting_some_not(full_model: FindingModelFull) -> None:
    """Test with some attributes conflicting and others not."""
    # Only the first attribute conflicts
    attribute_map = {
        "OIFMA_TEST_123456": "OIFM_OTHER_999999",
        # OIFMA_TEST_654321 is not in the map, so no conflict
    }
    errors = check_attribute_id_conflict(full_model, attribute_map)
    assert len(errors) == 1
    assert "OIFMA_TEST_123456" in errors[0]
    assert "OIFMA_TEST_654321" not in errors[0]


def test_check_attribute_id_conflict_empty_attribute_map(full_model: FindingModelFull) -> None:
    """Test with empty attribute map."""
    attribute_map: dict[str, str] = {}
    errors = check_attribute_id_conflict(full_model, attribute_map)
    assert errors == []


def test_check_attribute_id_conflict_model_with_no_attributes() -> None:
    """Test with a model that has no attributes (edge case, though models require at least 1)."""
    # Note: FindingModelFull requires at least 1 attribute, so we use a minimal valid model
    model = FindingModelFull(
        oifm_id="OIFM_TEST_000001",
        name="Single Attribute Model",
        description="Model with single attribute",
        attributes=[
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_888888",
                name="Size",
                description="Size attribute",
                minimum=0,
                maximum=10,
                unit="cm",
                required=False,
            )
        ],
    )
    # Check against an attribute map that doesn't include this model's attributes
    attribute_map = {"OIFMA_TEST_777777": "OIFM_TEST_666666"}
    errors = check_attribute_id_conflict(model, attribute_map)
    assert errors == []


# validate_finding_model() Tests
@pytest.mark.asyncio
async def test_validate_finding_model_no_conflicts(full_model: FindingModelFull) -> None:
    """Test validation passes when there are no conflicts."""
    context = MockValidationContext(
        existing_ids={"OIFM_OTHER_000001"},
        existing_names={"other model"},
        attribute_ids_by_model={"OIFMA_OTHER_000001": "OIFM_OTHER_000001"},
    )
    errors = await validate_finding_model(full_model, context)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_finding_model_detects_id_conflict(full_model: FindingModelFull) -> None:
    """Test validation detects OIFM ID conflict."""
    context = MockValidationContext(
        existing_ids={"OIFM_TEST_123456"},  # Conflicts with sample model
        existing_names=set(),
        attribute_ids_by_model={},
    )
    errors = await validate_finding_model(full_model, context)
    assert len(errors) == 1
    assert "OIFM_TEST_123456" in errors[0]
    assert "already exists" in errors[0]


@pytest.mark.asyncio
async def test_validate_finding_model_detects_name_conflict(full_model: FindingModelFull) -> None:
    """Test validation detects name conflict."""
    context = MockValidationContext(
        existing_ids=set(),
        existing_names={"test model"},  # Conflicts with sample model
        attribute_ids_by_model={},
    )
    errors = await validate_finding_model(full_model, context)
    assert len(errors) == 1
    assert "Test Model" in errors[0]
    assert "already in use" in errors[0]


@pytest.mark.asyncio
async def test_validate_finding_model_detects_attribute_conflict(full_model: FindingModelFull) -> None:
    """Test validation detects attribute ID conflict."""
    context = MockValidationContext(
        existing_ids=set(),
        existing_names=set(),
        attribute_ids_by_model={"OIFMA_TEST_123456": "OIFM_OTHER_999999"},
    )
    errors = await validate_finding_model(full_model, context)
    assert len(errors) == 1
    assert "OIFMA_TEST_123456" in errors[0]
    assert "already used by model" in errors[0]


@pytest.mark.asyncio
async def test_validate_finding_model_detects_multiple_conflicts(full_model: FindingModelFull) -> None:
    """Test validation combines all error messages when multiple conflicts exist."""
    context = MockValidationContext(
        existing_ids={"OIFM_TEST_123456"},
        existing_names={"test model"},
        attribute_ids_by_model={"OIFMA_TEST_123456": "OIFM_OTHER_999999"},
    )
    errors = await validate_finding_model(full_model, context)
    assert len(errors) == 3, "Should detect ID, name, and attribute conflicts"
    # Check that each type of error is present
    error_text = " ".join(errors)
    assert "OIFM_TEST_123456" in error_text
    assert "Test Model" in error_text or "test model" in error_text
    assert "OIFMA_TEST_123456" in error_text


@pytest.mark.asyncio
async def test_validate_finding_model_allow_self_permits_all_conflicts(full_model: FindingModelFull) -> None:
    """Test that allow_self=True permits all conflicts (for updates)."""
    context = MockValidationContext(
        existing_ids={"OIFM_TEST_123456"},
        existing_names={"test model"},
        attribute_ids_by_model={
            "OIFMA_TEST_123456": "OIFM_TEST_123456",
            "OIFMA_TEST_654321": "OIFM_TEST_123456",
        },
    )
    errors = await validate_finding_model(full_model, context, allow_self=True)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_finding_model_allow_self_false_detects_conflicts(full_model: FindingModelFull) -> None:
    """Test that allow_self=False detects conflicts with the model's own data."""
    context = MockValidationContext(
        existing_ids={"OIFM_TEST_123456"},
        existing_names={"test model"},
        attribute_ids_by_model={
            "OIFMA_TEST_123456": "OIFM_TEST_123456",
            "OIFMA_TEST_654321": "OIFM_TEST_123456",
        },
    )
    errors = await validate_finding_model(full_model, context, allow_self=False)
    # Should detect ID and name conflicts (attribute conflicts won't be flagged since they're owned by same model)
    assert len(errors) >= 2


@pytest.mark.asyncio
async def test_validate_finding_model_empty_context(full_model: FindingModelFull) -> None:
    """Test validation with completely empty context."""
    context = MockValidationContext()
    errors = await validate_finding_model(full_model, context)
    assert errors == []


# Integration Tests with Real Model Fixtures
@pytest.mark.asyncio
async def test_validate_finding_model_with_full_model_fixture(full_model: FindingModelFull) -> None:
    """Test validation with the full_model fixture from conftest.py."""
    context = MockValidationContext()
    errors = await validate_finding_model(full_model, context)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_finding_model_with_real_model_data(real_model: FindingModelFull) -> None:
    """Test validation with real model structure from test data."""
    context = MockValidationContext()
    errors = await validate_finding_model(real_model, context)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_finding_model_single_attribute() -> None:
    """Test validation with model that has a single attribute (minimum required)."""
    model = FindingModelFull(
        oifm_id="OIFM_TEST_111111",
        name="Single Attribute Model",
        description="Model with single attribute",
        attributes=[
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_111111",
                name="Size",
                description="Single attribute",
                minimum=0,
                maximum=10,
                unit="cm",
                required=False,
            )
        ],
    )
    context = MockValidationContext()
    errors = await validate_finding_model(model, context)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_finding_model_minimal_synonyms() -> None:
    """Test validation with model that has minimal synonyms (1 required)."""
    model = FindingModelFull(
        oifm_id="OIFM_TEST_222222",
        name="Minimal Synonyms Model",
        description="Model with minimal synonyms for testing validation",
        synonyms=["synonym"],  # At least 1 synonym required
        attributes=[
            ChoiceAttributeIded(
                oifma_id="OIFMA_TEST_222222",
                name="Test Attribute",
                description="Test attribute for validation",  # At least 5 chars
                values=[
                    ChoiceValueIded(value_code="OIFMA_TEST_222222.0", name="Value 1"),
                    ChoiceValueIded(value_code="OIFMA_TEST_222222.1", name="Value 2"),
                ],  # At least 2 values required
                required=True,
                max_selected=1,
            )
        ],
    )
    context = MockValidationContext()
    errors = await validate_finding_model(model, context)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_finding_model_minimal_tags() -> None:
    """Test validation with model that has minimal tags (1 required)."""
    model = FindingModelFull(
        oifm_id="OIFM_TEST_333333",
        name="Minimal Tags Model",
        description="Model with minimal tags for testing validation",
        tags=["test"],  # At least 1 tag required
        attributes=[
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_333333",
                name="Size",
                description="Test size attribute for validation",
                minimum=0,
                maximum=100,
                unit="mm",
                required=False,
            )
        ],
    )
    context = MockValidationContext()
    errors = await validate_finding_model(model, context)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_finding_model_cross_model_attribute_conflict() -> None:
    """Test validation detects attribute conflicts across different models."""
    # First model is already in the index
    context = MockValidationContext(
        existing_ids={"OIFM_TEST_123456"},
        existing_names={"test model"},
        attribute_ids_by_model={
            "OIFMA_TEST_123456": "OIFM_TEST_123456",
            "OIFMA_TEST_654321": "OIFM_TEST_123456",
        },
    )

    # Create another model with a conflicting attribute ID
    conflicting_model = FindingModelFull(
        oifm_id="OIFM_TEST_999999",
        name="Another Test Model",
        description="Another test model with conflicting attribute.",
        attributes=[
            ChoiceAttributeIded(
                oifma_id="OIFMA_TEST_999001",
                name="Status",
                description="Status of the finding",
                values=[
                    ChoiceValueIded(value_code="OIFMA_TEST_999001.0", name="Active"),
                    ChoiceValueIded(value_code="OIFMA_TEST_999001.1", name="Resolved"),
                ],
                required=True,
                max_selected=1,
            ),
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_123456",  # Conflicts with full_model
                name="Conflicting Size",
                description="This attribute ID already exists",
                minimum=1,
                maximum=100,
                unit="cm",
                required=False,
            ),
        ],
    )

    errors = await validate_finding_model(conflicting_model, context)
    # Should detect attribute conflict but not ID/name conflicts
    assert any("OIFMA_TEST_123456" in error for error in errors)
    assert any("OIFM_TEST_123456" in error for error in errors)


@pytest.mark.asyncio
async def test_validate_finding_model_concurrent_validation_safety() -> None:
    """Test that multiple validations can run concurrently without interference."""
    import asyncio

    model1 = FindingModelFull(
        oifm_id="OIFM_TEST_444444",
        name="Concurrent Model 1",
        description="Test concurrent validation",
        attributes=[
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_444444",
                name="Attr1",
                description="Attribute 1",
                minimum=0,
                maximum=10,
                unit="cm",
                required=False,
            )
        ],
    )

    model2 = FindingModelFull(
        oifm_id="OIFM_TEST_555555",
        name="Concurrent Model 2",
        description="Test concurrent validation",
        attributes=[
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_555555",
                name="Attr2",
                description="Attribute 2",
                minimum=0,
                maximum=20,
                unit="mm",
                required=False,
            )
        ],
    )

    context = MockValidationContext()

    # Run validations concurrently
    results = await asyncio.gather(
        validate_finding_model(model1, context),
        validate_finding_model(model2, context),
    )

    # Both should pass
    assert results[0] == []
    assert results[1] == []
