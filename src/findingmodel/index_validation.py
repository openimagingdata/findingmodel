"""Shared validation logic for finding model indexes.

This module provides protocol-based validation that works with any index backend
(DuckDB, MongoDB, etc.) without tight coupling.
"""

from typing import Protocol

from findingmodel.common import normalize_name
from findingmodel.finding_model import FindingModelFull


class ValidationContext(Protocol):
    """Protocol for index implementations to provide validation data.

    Any index backend can implement this protocol to use shared validation logic.
    This enables DRY validation across different storage implementations.
    """

    async def get_existing_oifm_ids(self) -> set[str]:
        """Get all existing OIFM IDs in the index.

        Returns:
            Set of OIFM IDs currently in the index
        """
        ...

    async def get_existing_names(self) -> set[str]:
        """Get all existing model names (case-folded) in the index.

        Returns:
            Set of lowercase model names currently in the index
        """
        ...

    async def get_attribute_ids_by_model(self) -> dict[str, str]:
        """Get mapping of attribute IDs to the models that own them.

        Returns:
            Dictionary mapping attribute_id -> oifm_id
        """
        ...


def check_oifm_id_conflict(
    model: FindingModelFull,
    existing_ids: set[str],
    *,
    allow_self: bool = False,
) -> list[str]:
    """Check for OIFM ID conflicts.

    Args:
        model: The finding model to validate
        existing_ids: Set of existing OIFM IDs in the index
        allow_self: If True, allow the model's own ID (for updates)

    Returns:
        List of error messages (empty if no conflicts)
    """
    errors: list[str] = []

    if model.oifm_id in existing_ids and not allow_self:
        errors.append(f"OIFM ID '{model.oifm_id}' already exists")

    return errors


def check_name_conflict(
    model: FindingModelFull,
    existing_names: set[str],
    *,
    allow_self: bool = False,
) -> list[str]:
    """Check for name and slug_name conflicts.

    Compares both the case-folded name and the normalized slug_name to ensure
    uniqueness across the index.

    Args:
        model: The finding model to validate
        existing_names: Set of existing case-folded names in the index
        allow_self: If True, allow the model's own name (for updates)

    Returns:
        List of error messages (empty if no conflicts)
    """
    errors: list[str] = []

    # Check case-insensitive name match
    name_lower = model.name.casefold()
    if name_lower in existing_names and not allow_self:
        errors.append(f"Name '{model.name}' already in use")

    # Check normalized slug_name match
    slug_name = normalize_name(model.name)
    # Note: existing_names should already contain normalized versions
    # but we check the slug explicitly for clarity
    if slug_name in existing_names and not allow_self and name_lower not in existing_names:
        # Only add this error if we didn't already flag the name conflict
        errors.append(f"Slug name '{slug_name}' already in use")

    return errors


def check_attribute_id_conflict(
    model: FindingModelFull,
    attribute_ids_by_model: dict[str, str],
    *,
    allow_self: bool = False,
) -> list[str]:
    """Check for attribute ID conflicts across models.

    Ensures that attribute IDs are unique across the entire index, preventing
    different models from using the same attribute ID.

    Args:
        model: The finding model to validate
        attribute_ids_by_model: Mapping of attribute_id -> oifm_id for all attributes in the index
        allow_self: If True, allow conflicts with the model's own attributes (for updates)

    Returns:
        List of error messages (empty if no conflicts)
    """
    errors: list[str] = []

    for attr in model.attributes:
        attr_id = attr.oifma_id
        if attr_id in attribute_ids_by_model:
            existing_model_id = attribute_ids_by_model[attr_id]
            # Only flag as error if it's owned by a different model
            if existing_model_id != model.oifm_id:
                errors.append(f"Attribute ID '{attr_id}' already used by model '{existing_model_id}'")

    return errors


async def validate_finding_model(
    model: FindingModelFull,
    context: ValidationContext,
    *,
    allow_self: bool = False,
) -> list[str]:
    """Complete validation of a finding model using protocol-based context.

    Performs all validation checks in parallel and returns a combined list of errors.
    This function works with any index backend that implements the ValidationContext protocol.

    Args:
        model: The finding model to validate
        context: Index backend providing validation data
        allow_self: If True, allow the model to match itself (for updates)

    Returns:
        List of validation error messages (empty list means valid)
    """
    errors: list[str] = []

    # Gather validation data from the index backend
    existing_ids = await context.get_existing_oifm_ids()
    existing_names = await context.get_existing_names()
    attribute_map = await context.get_attribute_ids_by_model()

    # Run all validation checks
    errors.extend(check_oifm_id_conflict(model, existing_ids, allow_self=allow_self))
    errors.extend(check_name_conflict(model, existing_names, allow_self=allow_self))
    errors.extend(check_attribute_id_conflict(model, attribute_map, allow_self=allow_self))

    return errors


__all__ = [
    "ValidationContext",
    "check_attribute_id_conflict",
    "check_name_conflict",
    "check_oifm_id_conflict",
    "validate_finding_model",
]
