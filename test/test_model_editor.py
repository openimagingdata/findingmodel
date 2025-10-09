from collections.abc import Iterator

import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from findingmodel.finding_model import FindingModelFull
from findingmodel.tools import model_editor
from findingmodel.tools.add_ids import PLACEHOLDER_ATTRIBUTE_ID, IdManager


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
    agent = model_editor.create_edit_agent()
    command = "Add attribute 'severity' with values mild, moderate, severe."
    mock_output = model_editor.EditResult(
        model=modified_model,
        rejections=[],
        changes=["Added severity attribute with values mild, moderate, severe"],
    )
    with agent.override(model=TestModel(custom_output_args=mock_output)):
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
    """Exercise a real LLM call to demonstrate an actual edit.
    This test is expected to run only in callout mode and will error if OpenAI is not configured.
    """

    # Temporarily enable model requests for this test only
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        model = real_model
        base_count = len(model.attributes)
        command = (
            "Add a new attribute named 'severity' of type choice with values: mild, moderate, severe. "
            "Do not modify or remove any existing attributes or values. Preserve all existing IDs."
        )
        result = await model_editor.edit_model_natural_language(model, command)

        # Basic validations
        assert isinstance(result.model, FindingModelFull)
        assert result.model.oifm_id == model.oifm_id  # ID preservation
        # Ideally, the attribute count increases and we find the new attribute
        assert len(result.model.attributes) >= base_count
        names = [a.name.lower() for a in result.model.attributes]
        # The model should add 'severity'; if not, this indicates the LLM ignored the instruction
        assert "severity" in names
        sev = next(a for a in result.model.attributes if a.name.lower() == "severity")
        vnames = [getattr(v, "name", "").lower() for v in getattr(sev, "values", [])]
        assert {"mild", "moderate", "severe"}.issubset(set(vnames))
        assert isinstance(result.changes, list)
    finally:
        models.ALLOW_MODEL_REQUESTS = original


@pytest.mark.callout
@pytest.mark.asyncio
async def test_edit_model_markdown_callout_real_api(real_model: FindingModelFull) -> None:
    """Exercise a real LLM call for the Markdown-based edit path.
    This test runs only in callout mode and will error if OpenAI is not configured.
    """

    # Temporarily enable model requests for this test only
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        model = real_model
        base_count = len(model.attributes)

        # Start from exported markdown and append a new attribute section
        md = model_editor.export_model_for_editing(model)
        md += "\n".join([
            "### severity",
            "- mild",
            "- moderate",
            "- severe",
            "",
        ])

        result = await model_editor.edit_model_markdown(model, md)

        # Basic validations
        assert isinstance(result.model, FindingModelFull)
        assert result.model.oifm_id == model.oifm_id  # ID preservation

        # Ideally, the attribute count increases and we find the new attribute
        assert len(result.model.attributes) >= base_count
        names = [a.name.lower() for a in result.model.attributes]
        assert "severity" in names
        sev = next(a for a in result.model.attributes if a.name.lower() == "severity")
        vnames = [getattr(v, "name", "").lower() for v in getattr(sev, "values", [])]
        assert {"mild", "moderate", "severe"}.issubset(set(vnames))
        assert isinstance(result.changes, list)
    finally:
        models.ALLOW_MODEL_REQUESTS = original


@pytest.mark.callout
@pytest.mark.asyncio
async def test_forbidden_change_nl_callout_real_api(real_model: FindingModelFull) -> None:
    """Ask the agent to perform a forbidden change via natural language and verify it's rejected."""
    original_flag = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        model = real_model
        # Attempt to rename an existing attribute (forbidden)
        existing_name = model.attributes[0].name
        command = f"Rename attribute '{existing_name}' to 'renamed_attr'."
        result = await model_editor.edit_model_natural_language(model, command)

        # Model should be unchanged and at least one rejection present
        assert result.model.model_dump_json() == model.model_dump_json()
        assert result.rejections and isinstance(result.rejections[0], str)
        assert result.changes == []
    finally:
        models.ALLOW_MODEL_REQUESTS = original_flag


@pytest.mark.callout
@pytest.mark.asyncio
async def test_forbidden_change_markdown_callout_real_api(real_model: FindingModelFull) -> None:
    """Ask the agent to perform a forbidden change via Markdown and verify it's rejected."""
    original_flag = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        model = real_model
        md = model_editor.export_model_for_editing(model)
        # Attempt to remove the first attribute section entirely by truncating markdown after header
        first_attr = model.attributes[0].name
        # Remove the '### first_attr' section and its bullets/description
        lines = md.splitlines()
        out_lines: list[str] = []
        skip = False
        for line in lines:
            if line.strip() == f"### {first_attr}":
                skip = True
                continue
            if skip and line.startswith("### "):
                skip = False
            if not skip:
                out_lines.append(line)
        edited_md = "\n".join(out_lines) + "\n"

        result = await model_editor.edit_model_markdown(model, edited_md)

        # Model should be unchanged and at least one rejection present
        assert result.model.model_dump_json() == model.model_dump_json()
        assert result.rejections and isinstance(result.rejections[0], str)
        assert result.changes == []
    finally:
        models.ALLOW_MODEL_REQUESTS = original_flag


def test_assign_real_attribute_ids_infers_source(real_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch) -> None:
    counters: dict[str, int] = {}

    def _fake_generate(src: str, existing: set[str]) -> str:
        counters[src] = counters.get(src, 0) + 1
        return f"OIFMA_{src}_{counters[src]:06d}"

    monkeypatch.setattr(
        IdManager,
        "_generate_unique_oifma",
        staticmethod(_fake_generate),
    )
    monkeypatch.setattr(
        IdManager,
        "load_used_ids_from_github",
        lambda self, refresh_cache=False: None,
    )

    base_data = real_model.model_dump()
    base_data["attributes"].append({
        "oifma_id": PLACEHOLDER_ATTRIBUTE_ID,
        "name": "severity",
        "type": "choice",
        "values": [{"name": "mild"}, {"name": "moderate"}, {"name": "severe"}],
        "required": False,
    })
    with_placeholder = FindingModelFull.model_validate(base_data)

    manager = IdManager()
    updated = model_editor.assign_real_attribute_ids(with_placeholder, manager=manager)

    attr = next(a for a in updated.attributes if a.name == "severity")
    assert getattr(attr, "type", None) == "choice"
    values = list(getattr(attr, "values", []))
    assert attr.oifma_id.startswith("OIFMA_MSFT_")
    assert [v.value_code for v in values] == [f"{attr.oifma_id}.{i}" for i in range(3)]
    assert PLACEHOLDER_ATTRIBUTE_ID not in {a.oifma_id for a in updated.attributes}


def test_assign_real_attribute_ids_uses_explicit_source(
    real_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    counters: dict[str, int] = {}

    def _fake_generate(src: str, existing: set[str]) -> str:
        counters[src] = counters.get(src, 0) + 1
        return f"OIFMA_{src}_{counters[src]:06d}"

    monkeypatch.setattr(
        IdManager,
        "_generate_unique_oifma",
        staticmethod(_fake_generate),
    )
    monkeypatch.setattr(
        IdManager,
        "load_used_ids_from_github",
        lambda self, refresh_cache=False: None,
    )

    base_data = real_model.model_dump()
    base_data["attributes"].append({
        "oifma_id": PLACEHOLDER_ATTRIBUTE_ID,
        "name": "pattern",
        "type": "choice",
        "values": [{"name": "filling"}, {"name": "wall-adherent"}],
        "required": False,
    })
    with_placeholder = FindingModelFull.model_validate(base_data)

    manager = IdManager()
    updated = model_editor.assign_real_attribute_ids(with_placeholder, source="abc", manager=manager)

    attr = next(a for a in updated.attributes if a.name == "pattern")
    assert getattr(attr, "type", None) == "choice"
    values = list(getattr(attr, "values", []))
    assert attr.oifma_id.startswith("OIFMA_ABC_")
    assert [v.value_code for v in values] == [f"{attr.oifma_id}.{i}" for i in range(2)]


def test_assign_real_attribute_ids_no_placeholders_returns_same_object(
    real_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        IdManager,
        "load_used_ids_from_github",
        lambda self, refresh_cache=False: None,
    )
    manager = IdManager()
    result = model_editor.assign_real_attribute_ids(real_model, manager=manager)
    assert result is real_model
