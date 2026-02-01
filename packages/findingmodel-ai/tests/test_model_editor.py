from collections.abc import Iterator

import pytest
from findingmodel import Index
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID
from findingmodel_ai.authoring import editor as model_editor
from pydantic_ai import models
from pydantic_ai.models.test import TestModel


@pytest.fixture(autouse=True)
def _disable_model_requests() -> Iterator[None]:
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    try:
        yield
    finally:
        models.ALLOW_MODEL_REQUESTS = original


@pytest.mark.asyncio
async def test_edit_model_natural_language_add_attribute(real_model: FindingModelFull) -> None:
    """Test adding an attribute via natural language command."""
    from unittest.mock import patch

    from findingmodel_ai.config import FindingModelAIConfig

    # Use provided real model fixture
    model = real_model

    # Build a modified model that includes a new 'severity' attribute
    base_data = model.model_dump()
    base_data["attributes"].append({
        "oifma_id": "OIFMA_XXXX_000000",
        "name": "severity",
        "type": "choice",
        "values": [{"name": "mild"}, {"name": "moderate"}, {"name": "severe"}],
        "required": False,
    })
    modified_model = FindingModelFull.model_validate(base_data)

    # Use TestModel to override the real model and supply a controlled output (EditResult)
    # Need to mock get_model to avoid API key check
    with patch.object(FindingModelAIConfig, "get_model", return_value="test"):
        agent = model_editor.create_edit_agent()

    command = "Add attribute 'severity' with values mild, moderate, severe."
    mock_output = model_editor.EditResult(
        model=modified_model,
        rejections=[],
        changes=["Added severity attribute with values mild, moderate, severe"],
    )
    # TestModel requires custom_output_args to be a dict, not a Pydantic model
    with agent.override(model=TestModel(custom_output_args=mock_output.model_dump())):
        result = await model_editor.edit_model_natural_language(model, command, agent=agent)
    assert hasattr(result, "model")
    assert isinstance(result.model, FindingModelFull)
    # ID preserved
    assert result.model.oifm_id == model.oifm_id
    # New attribute added
    names = [a.name for a in result.model.attributes]
    assert "severity" in names
    # Values propagated
    severity_attr = next(a for a in result.model.attributes if a.name == "severity")
    value_names = [v.name for v in getattr(severity_attr, "values", [])]
    assert value_names == ["mild", "moderate", "severe"]
    # Changes summarised
    assert result.changes == ["Added severity attribute with values mild, moderate, severe"]
    # No 'source' on EditResult; provenance should be handled by caller logs if needed


def test_export_model_for_editing_roundtrip(real_model: FindingModelFull) -> None:
    model = real_model
    md = model_editor.export_model_for_editing(model)
    assert isinstance(md, str)
    assert md.strip().startswith("# ")


def test_export_model_for_editing_structure_full(real_model: FindingModelFull) -> None:
    model = real_model
    md = model_editor.export_model_for_editing(model)

    # Top-level header is capitalized
    assert md.startswith("# Pulmonary embolism")
    # Synonyms included when present
    assert "Synonyms: PE" in md
    # Attributes section exists
    assert "\n## Attributes\n" in md
    # Attribute appears after Attributes section
    assert md.index("## Attributes") < md.index("### presence")

    # Choice attribute with descriptions includes value descriptions
    assert "### presence" in md
    assert "Presence or absence of pulmonary embolism" in md
    assert "- absent: Pulmonary embolism is absent" in md
    assert "- present: Pulmonary embolism is present" in md

    # Choice attribute with missing value descriptions should not add a colon
    assert "### other presence" in md
    section_start = md.index("### other presence")
    next_header = md.find("### ", section_start + 1)
    section = md[section_start : next_header if next_header != -1 else len(md)]
    assert "- absent\n" in section or "- absent\r\n" in section
    assert "- absent:" not in section

    # Numeric attribute renders constraints succinctly
    assert "### size" in md
    assert "- min 0; max 10; unit cm" in md


def test_export_model_for_editing_attributes_only(real_model: FindingModelFull) -> None:
    model = real_model
    md = model_editor.export_model_for_editing(model, attributes_only=True)

    # No top metadata when attributes_only
    assert "# Pulmonary embolism" not in md
    assert "Synonyms: PE" not in md
    assert "## Attributes" not in md

    # Starts directly with an attribute header
    assert "### presence" in md


@pytest.mark.callout
@pytest.mark.asyncio
async def test_edit_model_natural_language_callout_real_api(real_model: FindingModelFull) -> None:
    """Sanity check that real API integration works.
    Behavior validation is handled by evals; this just verifies we get a valid result.
    """
    # Temporarily enable model requests for this test only
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        # Use fast model for integration test
        agent = model_editor.create_edit_agent(model_tier="small")

        command = "Add a new attribute named 'severity' of type choice with values: mild, moderate, severe."
        result = await model_editor.edit_model_natural_language(real_model, command, agent=agent)

        # Verify we got a valid result structure back
        assert isinstance(result.model, FindingModelFull)
        assert isinstance(result.changes, list)
        assert isinstance(result.rejections, list)
    finally:
        models.ALLOW_MODEL_REQUESTS = original


def test_assign_real_attribute_ids_infers_source(real_model: FindingModelFull, index_with_test_db: Index) -> None:
    """Test that assign_real_attribute_ids infers source from model's OIFM ID."""
    base_data = real_model.model_dump()
    base_data["attributes"].append({
        "oifma_id": PLACEHOLDER_ATTRIBUTE_ID,
        "name": "severity",
        "type": "choice",
        "values": [{"name": "mild"}, {"name": "moderate"}, {"name": "severe"}],
        "required": False,
    })
    with_placeholder = FindingModelFull.model_validate(base_data)

    updated = model_editor.assign_real_attribute_ids(with_placeholder, index=index_with_test_db)

    attr = next(a for a in updated.attributes if a.name == "severity")
    assert getattr(attr, "type", None) == "choice"
    values = list(getattr(attr, "values", []))
    # Should infer MSFT from model's OIFM_MSFT_932618
    assert attr.oifma_id.startswith("OIFMA_MSFT_")
    assert [v.value_code for v in values] == [f"{attr.oifma_id}.{i}" for i in range(3)]
    assert PLACEHOLDER_ATTRIBUTE_ID not in {a.oifma_id for a in updated.attributes}


def test_assign_real_attribute_ids_uses_explicit_source(
    real_model: FindingModelFull,
    index_with_test_db: Index,
) -> None:
    """Test that assign_real_attribute_ids uses explicitly provided source code."""
    base_data = real_model.model_dump()
    base_data["attributes"].append({
        "oifma_id": PLACEHOLDER_ATTRIBUTE_ID,
        "name": "pattern",
        "type": "choice",
        "values": [{"name": "filling"}, {"name": "wall-adherent"}],
        "required": False,
    })
    with_placeholder = FindingModelFull.model_validate(base_data)

    updated = model_editor.assign_real_attribute_ids(with_placeholder, source="ABC", index=index_with_test_db)

    attr = next(a for a in updated.attributes if a.name == "pattern")
    assert getattr(attr, "type", None) == "choice"
    values = list(getattr(attr, "values", []))
    # Should use explicit source ABC instead of inferring MSFT
    assert attr.oifma_id.startswith("OIFMA_ABC_")
    assert [v.value_code for v in values] == [f"{attr.oifma_id}.{i}" for i in range(2)]


def test_assign_real_attribute_ids_no_placeholders_returns_same_object(
    real_model: FindingModelFull,
    index_with_test_db: Index,
) -> None:
    """Test that when no placeholders exist, the original model is returned unchanged."""
    result = model_editor.assign_real_attribute_ids(real_model, index=index_with_test_db)
    assert result is real_model
