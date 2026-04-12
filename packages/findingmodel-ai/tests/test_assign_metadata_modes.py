"""Tests for assign_metadata modes: reassess vs fill_blanks_only, SYSTEM:CODE format, and validators."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from findingmodel import (
    AgeProfile,
    AgeStage,
    BodyRegion,
    EntityType,
    EtiologyCode,
    ExpectedDuration,
    ExpectedTimeCourse,
    FindingModelFull,
    IndexCode,
    Modality,
    SexSpecificity,
    Subspecialty,
)
from findingmodel.protocols import OntologySearchResult
from findingmodel_ai.metadata.assignment import (
    MetadataAssignmentDecision,
    OntologyCandidateDecision,
    assign_metadata,
    create_metadata_assignment_agent,
)
from findingmodel_ai.metadata.types import (
    FieldConfidence,
    OntologyCandidateRelationship,
)
from findingmodel_ai.search.anatomic import LocationSearchResponse
from findingmodel_ai.search.ontology import CategorizedOntologyConcepts
from pydantic_ai import models
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_model(finding_model: FindingModelFull) -> FindingModelFull:
    """A fully-populated model with all metadata fields set."""
    return finding_model.model_copy(
        update={
            "body_regions": [BodyRegion.CHEST],
            "subspecialties": [Subspecialty.CH],
            "etiologies": [EtiologyCode.INFLAMMATORY_INFECTIOUS],
            "entity_type": EntityType.FINDING,
            "applicable_modalities": [Modality.XR, Modality.CT],
            "expected_time_course": ExpectedTimeCourse(duration=ExpectedDuration.WEEKS),
            "age_profile": AgeProfile(applicability=[AgeStage.ADULT]),
            "sex_specificity": SexSpecificity.SEX_NEUTRAL,
            "index_codes": [IndexCode(system="SNOMEDCT", code="233604007", display="Pneumonia")],
            "anatomic_locations": [IndexCode(system="ANATOMICLOCATIONS", code="RID1301", display="lung")],
        }
    )


def _ontology_results() -> CategorizedOntologyConcepts:
    return CategorizedOntologyConcepts(
        exact_matches=[
            OntologySearchResult(
                concept_id="233604007",
                concept_text="Pneumonia",
                score=0.99,
                table_name="snomedct",
            )
        ],
        should_include=[
            OntologySearchResult(
                concept_id="RID5350",
                concept_text="pneumonia",
                score=0.95,
                table_name="radlex",
            )
        ],
        marginal_concepts=[],
        search_summary="Test ontology summary",
        excluded_anatomical=[],
    )


def _anatomic_results() -> LocationSearchResponse:
    return LocationSearchResponse(
        primary_location=OntologySearchResult(
            concept_id="RID1301",
            concept_text="lung",
            score=0.0,
            table_name="anatomic_locations",
        ),
        alternate_locations=[],
        reasoning="Lung is the primary site of pneumonia.",
    )


def _mock_gathering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ontology and anatomic gathering with standard results."""
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment._get_trace_id", lambda: None)


# ---------------------------------------------------------------------------
# 1. Reassess mode runs classifier on populated model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assign_metadata_reassesses_populated_model(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In default (reassess) mode, even a fully-populated model gets gathering + classifier."""
    _mock_gathering(monkeypatch)

    classifier_called = False

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal classifier_called
        classifier_called = True
        decision = MetadataAssignmentDecision(
            body_regions=[BodyRegion.CHEST, BodyRegion.ABDOMEN],
            entity_type=EntityType.FINDING,
            classification_rationale="Reassessed and added abdomen.",
            field_confidence={"body_regions": FieldConfidence.HIGH},
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    decision.model_dump(mode="json"),
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model)

    assert classifier_called, "Classifier should be called in reassess mode even for populated model"
    # The classifier's body_regions override the existing ones in reassess mode
    assert result.model.body_regions == [BodyRegion.CHEST, BodyRegion.ABDOMEN]
    assert result.review.assignment_mode == "reassess"


# ---------------------------------------------------------------------------
# 2. Fill blanks only preserves populated fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_blanks_only_preserves_populated_fields(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fill_blanks_only should not overwrite fields that already have values."""
    # Model with some fields populated, some blank
    partial_model = finding_model.model_copy(
        update={
            "body_regions": [BodyRegion.CHEST],
            "entity_type": EntityType.FINDING,
            # subspecialties, etiologies, modalities, etc. left blank
        }
    )
    _mock_gathering(monkeypatch)

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.ABDOMEN],  # Tries to change existing field
        entity_type=EntityType.DIAGNOSIS,  # Tries to change existing field
        subspecialties=[Subspecialty.CH],  # Fills blank field
        applicable_modalities=[Modality.CT],  # Fills blank field
        classification_rationale="Classifier tried to change everything.",
        field_confidence={"body_regions": FieldConfidence.HIGH, "subspecialties": FieldConfidence.HIGH},
    )
    agent = create_metadata_assignment_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(partial_model, fill_blanks_only=True)

    # Existing fields preserved
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.entity_type == EntityType.FINDING
    # Blank fields filled
    assert result.model.subspecialties == [Subspecialty.CH]
    assert result.model.applicable_modalities == [Modality.CT]
    assert result.review.assignment_mode == "fill_blanks_only"


# ---------------------------------------------------------------------------
# 3. Fill blanks only preserves index codes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_blanks_only_preserves_index_codes(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fill_blanks_only should not overwrite existing index_codes."""
    existing_codes = [IndexCode(system="SNOMEDCT", code="233604007", display="Pneumonia")]
    model_with_codes = finding_model.model_copy(update={"index_codes": existing_codes})
    _mock_gathering(monkeypatch)

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="RADLEX:RID5350",
                relationship=OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE,
                selected_as_canonical=True,
                rationale="RadLex equivalent.",
            ),
        ],
        classification_rationale="Would add new ontology codes.",
        field_confidence={},
    )
    agent = create_metadata_assignment_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(model_with_codes, fill_blanks_only=True)

    # Existing index_codes should be preserved
    assert result.model.index_codes == existing_codes
    # Blank field should be filled
    assert result.model.entity_type == EntityType.FINDING


# ---------------------------------------------------------------------------
# 4. Fill blanks only ignores clear_fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_blanks_only_ignores_clear_fields(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fill_blanks_only should warn and ignore clear_fields from the classifier."""
    _mock_gathering(monkeypatch)

    decision = MetadataAssignmentDecision(
        clear_fields=["body_regions"],
        classification_rationale="Tried to clear body_regions.",
        field_confidence={},
    )
    agent = create_metadata_assignment_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model, fill_blanks_only=True)

    # body_regions should NOT be cleared
    assert result.model.body_regions == [BodyRegion.CHEST]
    # Warning should be present
    assert any("clear_fields ignored" in w for w in result.review.warnings)


# ---------------------------------------------------------------------------
# 5. SYSTEM:CODE format for candidate IDs in prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_code_candidate_ids(finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt payload should use SYSTEM:CODE format for candidate IDs."""
    _mock_gathering(monkeypatch)

    captured: dict[str, str] = {}

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        prompt_parts: list[str] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        prompt_parts.append(content)
        captured["prompt"] = "\n".join(prompt_parts)
        return ModelResponse(
            parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {
                            "classification_rationale": "Test.",
                            "body_regions": ["chest"],
                            "entity_type": "finding",
                            "applicable_modalities": ["CT"],
                        },
                        tool_call_id="pyd_ai_tool_call_id__output",
                    )
            ]
        )

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    await assign_metadata(finding_model)

    prompt = captured["prompt"]
    # The ontology candidates should use SYSTEM:CODE format
    assert "SNOMEDCT:233604007" in prompt
    assert "RADLEX:RID5350" in prompt
    # The anatomic candidates should also use SYSTEM:CODE format
    assert "ANATOMICLOCATIONS:RID1301" in prompt


# ---------------------------------------------------------------------------
# 6. Fill-blanks prompt includes explicit mode guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_blanks_prompt_includes_assignment_mode(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt payload should tell the classifier when it is in fill_blanks_only mode."""
    _mock_gathering(monkeypatch)
    partial_model = finding_model.model_copy(
        update={
            "body_regions": [BodyRegion.CHEST],
            "entity_type": EntityType.FINDING,
        }
    )

    captured: dict[str, str] = {}

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        prompt_parts: list[str] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        prompt_parts.append(content)
        captured["prompt"] = "\n".join(prompt_parts)
        return ModelResponse(
            parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {
                            "classification_rationale": "Test.",
                            "entity_type": "finding",
                            "applicable_modalities": ["CT"],
                        },
                        tool_call_id="pyd_ai_tool_call_id__output",
                    )
                ]
        )

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    await assign_metadata(partial_model, fill_blanks_only=True)

    prompt = captured["prompt"]
    assert '"assignment_mode": "fill_blanks_only"' in prompt
    assert '"mode_context"' in prompt
    assert '"blank_structured_fields"' in prompt
    assert '"locked_structured_fields"' in prompt
    assert '"blank_required_fields": [' in prompt
    assert '"required_structured_fields": [' in prompt
    assert '"body_regions"' in prompt
    assert '"entity_type"' in prompt


# ---------------------------------------------------------------------------
# 7. Unknown candidate ID triggers retry via output validator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_candidate_id_triggers_retry(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output validator should reject hallucinated candidate IDs, then accept valid output."""
    _mock_gathering(monkeypatch)

    call_count = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: return a hallucinated candidate ID
            decision = MetadataAssignmentDecision(
                body_regions=[BodyRegion.CHEST],
                ontology_decisions=[
                    OntologyCandidateDecision(
                        candidate_id="FAKE:HALLUCINATED_999",
                        relationship=OntologyCandidateRelationship.EXACT_MATCH,
                        selected_as_canonical=True,
                        rationale="Made-up code.",
                    ),
                ],
                classification_rationale="First attempt with bad ID.",
                field_confidence={"body_regions": FieldConfidence.HIGH},
            )
        else:
            # Second call: return valid output
            decision = MetadataAssignmentDecision(
                body_regions=[BodyRegion.CHEST],
                entity_type=EntityType.FINDING,
                applicable_modalities=[Modality.CT],
                classification_rationale="Second attempt, valid.",
                field_confidence={
                    "body_regions": FieldConfidence.HIGH,
                    "entity_type": FieldConfidence.HIGH,
                    "applicable_modalities": FieldConfidence.HIGH,
                },
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    decision.model_dump(mode="json"),
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    # _run_classifier calls create_metadata_assignment_agent() with no model arg,
    # then registers an output_validator internally. We need the validator to fire,
    # so we can't replace create_metadata_assignment_agent. Instead, replace the
    # settings object's methods that _run_classifier uses for model resolution.
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.get_agent_model.return_value = FunctionModel(model_function)
    mock_settings.get_effective_model_string.return_value = "function:test"
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.settings", mock_settings)

    result = await assign_metadata(finding_model)

    # Should have retried: first call had hallucinated ID, second was valid
    assert call_count == 2
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.review.classification_rationale == "Second attempt, valid."


# ---------------------------------------------------------------------------
# 7. Missing required fields triggers retry in reassess mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_required_fields_triggers_retry_in_reassess_mode(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reassess mode should retry until all required fields are populated in the projected result."""
    _mock_gathering(monkeypatch)

    call_count = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: no required fields set → should trigger ModelRetry
            decision = MetadataAssignmentDecision(
                classification_rationale="First attempt, no required fields.",
                field_confidence={},
            )
        else:
            # Second call: provide all required fields → should succeed
            decision = MetadataAssignmentDecision(
                body_regions=[BodyRegion.CHEST],
                entity_type=EntityType.FINDING,
                applicable_modalities=[Modality.CT],
                classification_rationale="Second attempt, required fields set.",
                field_confidence={
                    "body_regions": FieldConfidence.HIGH,
                    "entity_type": FieldConfidence.HIGH,
                    "applicable_modalities": FieldConfidence.HIGH,
                },
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    decision.model_dump(mode="json"),
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.get_agent_model.return_value = FunctionModel(model_function)
    mock_settings.get_effective_model_string.return_value = "function:test"
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.settings", mock_settings)

    result = await assign_metadata(finding_model)

    assert call_count == 2
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.entity_type == EntityType.FINDING
    assert result.model.applicable_modalities == [Modality.CT]
    assert result.review.classification_rationale == "Second attempt, required fields set."


# ---------------------------------------------------------------------------
# 8. Missing blank required fields triggers retry in fill_blanks mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_blank_required_fields_triggers_retry_in_fill_blanks_mode(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In fill_blanks_only mode, all blank required fields must be filled before validation passes."""
    _mock_gathering(monkeypatch)

    call_count = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: all required fields still None → retry
            decision = MetadataAssignmentDecision(
                classification_rationale="First attempt, nothing filled.",
                field_confidence={},
            )
        else:
            # Second call: provide all blank required fields → should succeed
            decision = MetadataAssignmentDecision(
                body_regions=[BodyRegion.CHEST],
                entity_type=EntityType.FINDING,
                applicable_modalities=[Modality.CT],
                classification_rationale="Second attempt, required blanks filled.",
                field_confidence={
                    "body_regions": FieldConfidence.HIGH,
                    "entity_type": FieldConfidence.HIGH,
                    "applicable_modalities": FieldConfidence.HIGH,
                },
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    decision.model_dump(mode="json"),
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.get_agent_model.return_value = FunctionModel(model_function)
    mock_settings.get_effective_model_string.return_value = "function:test"
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.settings", mock_settings)

    result = await assign_metadata(finding_model, fill_blanks_only=True)

    assert call_count == 2
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.entity_type == EntityType.FINDING
    assert result.model.applicable_modalities == [Modality.CT]
    assert result.review.assignment_mode == "fill_blanks_only"


# ---------------------------------------------------------------------------
# 9. Reassess mode rejects clearing required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reassess_clearing_required_field_triggers_retry(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reassess mode should reject clear_fields on required metadata and retry."""
    _mock_gathering(monkeypatch)

    call_count = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            decision = MetadataAssignmentDecision(
                clear_fields=["body_regions"],
                classification_rationale="First attempt cleared a required field.",
                field_confidence={},
            )
        else:
            decision = MetadataAssignmentDecision(
                body_regions=[BodyRegion.CHEST],
                entity_type=EntityType.FINDING,
                applicable_modalities=[Modality.CT],
                classification_rationale="Second attempt replaced required fields correctly.",
                field_confidence={
                    "body_regions": FieldConfidence.HIGH,
                    "entity_type": FieldConfidence.HIGH,
                    "applicable_modalities": FieldConfidence.HIGH,
                },
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    decision.model_dump(mode="json"),
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.get_agent_model.return_value = FunctionModel(model_function)
    mock_settings.get_effective_model_string.return_value = "function:test"
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.settings", mock_settings)

    result = await assign_metadata(populated_model)

    assert call_count == 2
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.review.classification_rationale == "Second attempt replaced required fields correctly."


# ---------------------------------------------------------------------------
# 10. SYSTEM:CODE format applies to pre-existing candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_code_applies_to_pre_existing_candidates(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-existing index_codes appear in prompt as SYSTEM:CODE and can be referenced by the classifier."""
    _mock_gathering(monkeypatch)

    captured: dict[str, str] = {}

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        prompt_parts: list[str] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    content = getattr(part, "content", None)
                    if isinstance(content, str):
                        prompt_parts.append(content)
        captured["prompt"] = "\n".join(prompt_parts)

        # Classifier references the pre-existing code using SYSTEM:CODE format
        decision = MetadataAssignmentDecision(
            entity_type=EntityType.FINDING,
            body_regions=[BodyRegion.CHEST],
            ontology_decisions=[
                OntologyCandidateDecision(
                    candidate_id="SNOMEDCT:233604007",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH,
                    selected_as_canonical=True,
                    rationale="Pre-existing SNOMED code for pneumonia.",
                ),
            ],
            classification_rationale="Confirmed pre-existing codes.",
            field_confidence={"entity_type": FieldConfidence.HIGH},
        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    decision.model_dump(mode="json"),
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.get_agent_model.return_value = FunctionModel(model_function)
    mock_settings.get_effective_model_string.return_value = "function:test"
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.settings", mock_settings)

    result = await assign_metadata(populated_model)

    # Pre-existing index_codes should appear in the prompt with SYSTEM:CODE format
    prompt = captured["prompt"]
    assert "SNOMEDCT:233604007" in prompt

    # Result should contain the pre-existing index code
    assert any(c.system == "SNOMEDCT" and c.code == "233604007" for c in (result.model.index_codes or []))
