"""Shared utilities for evaluation suites.

This module provides common helper functions used across all agent
evaluation suites in the findingmodel project.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from findingmodel.finding_model import FindingModelFull

if TYPE_CHECKING:
    from findingmodel.tools import model_editor


def load_fm_json(filename: str) -> str:
    """Load a finding model JSON file from test data directory.

    Args:
        filename: Name of the .fm.json file (e.g., "pulmonary_embolism.fm.json")

    Returns:
        JSON content as a string

    Raises:
        FileNotFoundError: If the specified file does not exist

    Example:
        >>> json_str = load_fm_json("pulmonary_embolism.fm.json")
        >>> model = FindingModelFull.model_validate_json(json_str)
    """
    test_data_dir = Path(__file__).parent.parent / "test" / "data" / "defs"
    file_path = test_data_dir / filename
    return file_path.read_text(encoding="utf-8")


def create_mock_edit_result(
    model: FindingModelFull,
    changes: list[str] | None = None,
    rejections: list[str] | None = None,
) -> "model_editor.EditResult":
    """Create a mock EditResult for testing without API calls.

    This utility standardizes the creation of mock EditResult objects
    for use with TestModel in evaluation tests.

    Args:
        model: The finding model (modified or original)
        changes: List of change descriptions (default: empty list)
        rejections: List of rejection reasons (default: empty list)

    Returns:
        EditResult suitable for testing

    Example:
        >>> model = FindingModelFull.model_validate_json(json_str)
        >>> result = create_mock_edit_result(
        ...     model=model,
        ...     changes=["Added severity attribute"]
        ... )
        >>> assert result.model == model
        >>> assert "severity" in result.changes[0]
    """
    from findingmodel.tools import model_editor

    return model_editor.EditResult(
        model=model,
        changes=changes or [],
        rejections=rejections or [],
    )


def compare_models(model1: FindingModelFull, model2: FindingModelFull) -> bool:
    """Compare two finding models for equality.

    Uses model_dump_json() to perform a deep comparison of all model fields,
    including nested structures like attributes and choice values.

    Args:
        model1: First finding model
        model2: Second finding model

    Returns:
        True if models are identical, False otherwise

    Example:
        >>> model1 = FindingModelFull.model_validate_json(json1)
        >>> model2 = FindingModelFull.model_validate_json(json2)
        >>> are_equal = compare_models(model1, model2)
    """
    return model1.model_dump_json() == model2.model_dump_json()


def extract_text_for_keywords(changes: list[str], rejections: list[str]) -> str:
    """Extract and combine text from changes and rejections for keyword matching.

    Combines all change descriptions and rejection reasons into a single
    lowercase string suitable for case-insensitive keyword searching.

    Args:
        changes: List of change description strings
        rejections: List of rejection reason strings

    Returns:
        Combined lowercase text suitable for keyword searching

    Example:
        >>> text = extract_text_for_keywords(
        ...     changes=["Added severity attribute"],
        ...     rejections=[]
        ... )
        >>> "severity" in text
        True
    """
    all_text = changes + rejections
    combined = " ".join(all_text)
    return combined.lower()


def get_attribute_names(model: FindingModelFull) -> set[str]:
    """Extract attribute names from a finding model.

    Args:
        model: Finding model to extract attributes from

    Returns:
        Set of attribute name strings

    Example:
        >>> model = FindingModelFull.model_validate_json(json_str)
        >>> attrs = get_attribute_names(model)
        >>> "presence" in attrs
        True
    """
    return {attr.name for attr in model.attributes}
