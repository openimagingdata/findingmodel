"""Tests for assign_metadata modes: reassess vs fill_blanks_only, SYSTEM:CODE format, and validators."""

from __future__ import annotations

import json
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
    AnatomicCandidateDecision,
    AnatomyDecision,
    EtiologyDecision,
    IdentityDecision,
    ImagingWorkflowDecision,
    MetadataAssignmentDecision,
    OntologyCandidateDecision,
    OntologyDecision,
    PatientApplicabilityDecision,
    _anatomic_candidate_prompt_states,
    _AnatomicCandidateState,
    _append_source_support_level_consistency_warnings,
    _combine_focused_decisions,
    _etiology_decision_prompt,
    _ontology_candidate_prompt_states,
    _OntologyCandidateState,
    _validate_anatomy_decision,
    assign_metadata,
    create_anatomy_decision_agent,
    create_etiology_assignment_agent,
    create_identity_assignment_agent,
    create_imaging_workflow_agent,
    create_metadata_assignment_agent,
    create_ontology_decision_agent,
    create_patient_applicability_agent,
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


def test_source_support_level_consistency_warns_for_selected_supported_child() -> None:
    warnings: list[str] = []
    states = {
        "ANATOMICLOCATIONS:RID480": _AnatomicCandidateState(
            result=OntologySearchResult(
                concept_id="RID480",
                concept_text="aorta",
                score=0.0,
                table_name="anatomic_locations",
            ),
            source_bucket="candidate",
            support_level="source_inferred_query",
        ),
        "ANATOMICLOCATIONS:RID879": _AnatomicCandidateState(
            result=OntologySearchResult(
                concept_id="RID879",
                concept_text="thoracic aorta",
                score=0.0,
                table_name="anatomic_locations",
            ),
            selected=True,
            source_bucket="candidate",
            support_level="child_of_supported",
            broader_candidate_ids=["RID480"],
        ),
    }

    _append_source_support_level_consistency_warnings(states, warnings)

    assert warnings == [
        "source support level consistency check: selected 'thoracic aorta' (child_of_supported) "
        "while broader candidate 'aorta' had support_level=source_inferred_query"
    ]


def test_anatomy_validation_does_not_retry_named_component_ending_in_system() -> None:
    """Named components like renal collecting system should not be forced to whole-body anatomy."""
    states = {
        "ANATOMICLOCATIONS:RID225": _AnatomicCandidateState(
            result=OntologySearchResult(
                concept_id="RID225",
                concept_text="calyx of renal collecting system",
                score=0.0,
                table_name="anatomic_locations",
            ),
            selected=True,
            source_bucket="candidate",
            support_level="source_inferred_query",
        )
    }
    decision = AnatomyDecision(
        body_regions=[BodyRegion.ABDOMEN],
        anatomic_decisions=[AnatomicCandidateDecision(candidate_id="ANATOMICLOCATIONS:RID225", selected=True)],
    )

    assert _validate_anatomy_decision(decision, anatomic_states=states) == decision


def test_anatomic_candidate_prompt_states_limits_by_evidence() -> None:
    states: dict[str, _AnatomicCandidateState] = {}
    for idx in range(20):
        states[f"ANATOMICLOCATIONS:RID{idx}"] = _AnatomicCandidateState(
            result=OntologySearchResult(
                concept_id=f"RID{idx}",
                concept_text=f"search-only candidate {idx}",
                score=0.0,
                table_name="anatomic_locations",
            ),
            source_bucket="candidate",
            support_level="search_only",
        )
    states["ANATOMICLOCATIONS:RID-selected"] = _AnatomicCandidateState(
        result=OntologySearchResult(
            concept_id="RID-selected",
            concept_text="selected existing candidate",
            score=0.0,
            table_name="anatomic_locations",
        ),
        selected=True,
        source_bucket="existing_anatomic_locations",
        support_level="current_metadata",
    )
    states["ANATOMICLOCATIONS:RID-direct"] = _AnatomicCandidateState(
        result=OntologySearchResult(
            concept_id="RID-direct",
            concept_text="direct source candidate",
            score=0.0,
            table_name="anatomic_locations",
        ),
        source_bucket="candidate",
        support_level="direct_source",
    )

    prompt_states = _anatomic_candidate_prompt_states(states, limit=15)

    assert len(prompt_states) == 15
    assert "ANATOMICLOCATIONS:RID-selected" in prompt_states
    assert "ANATOMICLOCATIONS:RID-direct" in prompt_states


def test_ontology_candidate_prompt_states_limits_by_evidence() -> None:
    states: dict[str, _OntologyCandidateState] = {}
    for idx in range(20):
        states[f"RADLEX:RID{idx}"] = _OntologyCandidateState(
            result=OntologySearchResult(
                concept_id=f"RID{idx}",
                concept_text=f"marginal candidate {idx}",
                score=0.0,
                table_name="radlex",
            ),
            relationship=OntologyCandidateRelationship.RELATED,
            selected_as_canonical=False,
            source_bucket="marginal",
        )
    states["SNOMEDCT:233604007"] = _OntologyCandidateState(
        result=OntologySearchResult(
            concept_id="233604007",
            concept_text="Pneumonia",
            score=0.0,
            table_name="snomedct",
        ),
        relationship=OntologyCandidateRelationship.EXACT_MATCH,
        selected_as_canonical=True,
        source_bucket="exact_matches",
    )

    prompt_states = _ontology_candidate_prompt_states(states, limit=15)

    assert len(prompt_states) == 15
    assert "SNOMEDCT:233604007" in prompt_states


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


def test_field_confidence_ignores_non_metadata_keys() -> None:
    """Confidence output should ignore bookkeeping keys rather than retrying."""
    decision = MetadataAssignmentDecision.model_validate(
        {
            "classification_rationale": "Invalid confidence-key test.",
            "field_confidence": {"ontology_decisions": "high", "entity_type": 0.92},
        }
    )

    assert decision.field_confidence == {"entity_type": 0.92}


def test_combine_focused_decisions_preserves_agent_ownership() -> None:
    """Focused decisions should assemble without moving field ownership into orchestration."""
    decision = _combine_focused_decisions(
        ontology=OntologyDecision(
            ontology_decisions=[
                OntologyCandidateDecision(
                    candidate_id="SNOMEDCT:233604007",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH,
                    selected_as_canonical=True,
                    rationale="Exact pneumonia concept.",
                )
            ],
        ),
        anatomy=AnatomyDecision(
            body_regions=[BodyRegion.CHEST],
            anatomic_decisions=[
                AnatomicCandidateDecision(
                    candidate_id="ANATOMICLOCATIONS:RID1301",
                    selected=True,
                    rationale="Lung is selected.",
                )
            ],
        ),
        identity=IdentityDecision(
            entity_type=EntityType.FINDING,
            field_confidence={"entity_type": 0.9},
        ),
        etiology=EtiologyDecision(
            etiologies=[EtiologyCode.INFLAMMATORY_INFECTIOUS],
            field_confidence={"etiologies": 0.6},
        ),
        patient=PatientApplicabilityDecision(
            age_profile=AgeProfile(applicability=[AgeStage.ADULT]),
            sex_specificity=SexSpecificity.SEX_NEUTRAL,
            field_confidence={"age_profile": 0.9, "sex_specificity": 0.9},
        ),
        imaging_workflow=ImagingWorkflowDecision(
            subspecialties=[Subspecialty.CH],
            applicable_modalities=[Modality.CT],
            field_confidence={"subspecialties": 0.9, "applicable_modalities": 0.9},
        ),
    )

    assert decision.body_regions == [BodyRegion.CHEST]
    assert decision.anatomic_decisions[0].candidate_id == "ANATOMICLOCATIONS:RID1301"
    assert decision.ontology_decisions[0].candidate_id == "SNOMEDCT:233604007"
    assert decision.entity_type == EntityType.FINDING
    assert decision.etiologies == [EtiologyCode.INFLAMMATORY_INFECTIOUS]
    assert decision.age_profile == AgeProfile(applicability=[AgeStage.ADULT])
    assert decision.sex_specificity == SexSpecificity.SEX_NEUTRAL
    assert decision.subspecialties == [Subspecialty.CH]
    assert decision.applicable_modalities == [Modality.CT]
    assert "body_regions" not in decision.field_confidence
    assert decision.field_confidence["applicable_modalities"] == 0.9


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
        field_confidence={
            "subspecialties": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
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


@pytest.mark.asyncio
async def test_focused_assignment_path_splits_patient_and_workflow_agents(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path should collect patient and imaging workflow fields from separate agents."""
    _mock_gathering(monkeypatch)

    ontology_agent = create_ontology_decision_agent(
        model=TestModel(
            custom_output_args=OntologyDecision(
                ontology_decisions=[
                    OntologyCandidateDecision(
                        candidate_id="SNOMEDCT:233604007",
                        relationship=OntologyCandidateRelationship.EXACT_MATCH,
                        selected_as_canonical=True,
                    )
                ],
                field_confidence={"index_codes": FieldConfidence.HIGH},
            ).model_dump(mode="json")
        )
    )
    anatomy_agent = create_anatomy_decision_agent(
        model=TestModel(
            custom_output_args=AnatomyDecision(
                body_regions=[BodyRegion.CHEST],
                anatomic_decisions=[
                    AnatomicCandidateDecision(candidate_id="ANATOMICLOCATIONS:RID1301", selected=True)
                ],
                field_confidence={"body_regions": FieldConfidence.HIGH, "anatomic_locations": FieldConfidence.HIGH},
            ).model_dump(mode="json")
        )
    )
    identity_agent = create_identity_assignment_agent(
        model=TestModel(
            custom_output_args=IdentityDecision(
                entity_type=EntityType.FINDING,
                field_confidence={"entity_type": FieldConfidence.HIGH},
            ).model_dump(mode="json")
        )
    )
    etiology_agent = create_etiology_assignment_agent(
        model=TestModel(
            custom_output_args=EtiologyDecision(
                etiologies=[EtiologyCode.INFLAMMATORY_INFECTIOUS],
                field_confidence={"etiologies": FieldConfidence.HIGH},
            ).model_dump(mode="json")
        )
    )
    patient_agent = create_patient_applicability_agent(
        model=TestModel(
            custom_output_args=PatientApplicabilityDecision(
                age_profile=AgeProfile(applicability=[AgeStage.ADULT]),
                sex_specificity=SexSpecificity.SEX_NEUTRAL,
                field_confidence={"age_profile": FieldConfidence.HIGH, "sex_specificity": FieldConfidence.HIGH},
            ).model_dump(mode="json")
        )
    )
    imaging_agent = create_imaging_workflow_agent(
        model=TestModel(
            custom_output_args=ImagingWorkflowDecision(
                subspecialties=[Subspecialty.CH],
                applicable_modalities=[Modality.CT],
                field_confidence={
                    "subspecialties": FieldConfidence.HIGH,
                    "applicable_modalities": FieldConfidence.HIGH,
                },
            ).model_dump(mode="json")
        )
    )

    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_ontology_decision_agent", lambda **_: ontology_agent)
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_anatomy_decision_agent", lambda **_: anatomy_agent)
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_identity_assignment_agent", lambda **_: identity_agent)
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_etiology_assignment_agent", lambda **_: etiology_agent)
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_patient_applicability_agent",
        lambda **_: patient_agent,
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_imaging_workflow_agent", lambda **_: imaging_agent)

    result = await assign_metadata(finding_model)

    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.entity_type == EntityType.FINDING
    assert result.model.etiologies == [EtiologyCode.INFLAMMATORY_INFECTIOUS]
    assert result.model.age_profile == AgeProfile(applicability=[AgeStage.ADULT])
    assert result.model.sex_specificity == SexSpecificity.SEX_NEUTRAL
    assert result.model.subspecialties == [Subspecialty.CH]
    assert result.model.applicable_modalities == [Modality.CT]
    assert result.review.field_confidence["age_profile"] == 0.9
    assert result.review.field_confidence["applicable_modalities"] == 0.9


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
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    agent = create_metadata_assignment_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(model_with_codes, fill_blanks_only=True)

    # Existing index_codes should be preserved
    assert result.model.index_codes == existing_codes
    assert "Missing field_confidence entry for changed field: index_codes" not in result.review.warnings
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
                            "field_confidence": {
                                "body_regions": "high",
                                "entity_type": "high",
                                "applicable_modalities": "high",
                                "index_codes": "high",
                                "anatomic_locations": "high",
                            },
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
                            "field_confidence": {
                                "applicable_modalities": "high",
                                "index_codes": "high",
                                "anatomic_locations": "high",
                            },
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
                    "index_codes": FieldConfidence.HIGH,
                    "anatomic_locations": FieldConfidence.HIGH,
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model)

    # Should have retried: first call had hallucinated ID, second was valid
    assert call_count == 2
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.review.classification_rationale == "Second attempt, valid."


@pytest.mark.asyncio
async def test_changed_candidate_fields_do_not_require_confidence(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting candidates without confidence should not warn or retry."""
    _mock_gathering(monkeypatch)

    call_count = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        shared_output = {
            "body_regions": ["chest"],
            "entity_type": "finding",
            "applicable_modalities": ["CT"],
            "ontology_decisions": [
                {
                    "candidate_id": "RADLEX:RID5350",
                    "relationship": "clinically_substitutable",
                    "selected_as_canonical": True,
                }
            ],
            "anatomic_decisions": [
                {
                    "candidate_id": "ANATOMICLOCATIONS:RID1301",
                    "selected": True,
                }
            ],
            "classification_rationale": "Candidate confidence retry test.",
        }
        if call_count == 1:
            output = {
                **shared_output,
                "field_confidence": {
                    "body_regions": "high",
                    "entity_type": "high",
                    "applicable_modalities": "high",
                },
            }
        else:
            output = {
                **shared_output,
                "field_confidence": {
                    "body_regions": "high",
                    "entity_type": "high",
                    "applicable_modalities": "high",
                    "index_codes": "high",
                    "anatomic_locations": "high",
                },
            }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    output,
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model)

    assert call_count == 1
    assert "index_codes" not in result.review.field_confidence
    assert "anatomic_locations" not in result.review.field_confidence
    assert not any("Missing field_confidence entry" in warning for warning in result.review.warnings)


@pytest.mark.asyncio
async def test_reassess_sanitizes_non_disease_entity_with_etiologies(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assessment/measurement outputs should not retain disease-like etiologies."""
    _mock_gathering(monkeypatch)

    call_count = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        output = {
            "body_regions": ["spine"],
            "entity_type": "assessment",
            "etiologies": ["traumatic:acute"],
            "applicable_modalities": ["CT"],
            "classification_rationale": "Assessment output incorrectly kept etiology.",
            "field_confidence": {
                "body_regions": "high",
                "entity_type": "high",
                "etiologies": "high",
                "applicable_modalities": "high",
                "index_codes": "high",
                "anatomic_locations": "high",
            },
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    output,
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model)

    assert call_count == 1
    assert result.model.entity_type == EntityType.ASSESSMENT
    assert result.model.etiologies is None
    assert any("etiologies ignored for non-disease entity_type" in warning for warning in result.review.warnings)


@pytest.mark.asyncio
async def test_reassess_retains_medium_confidence_etiologies(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Medium-confidence etiology values are acceptable; only low confidence is dropped."""
    _mock_gathering(monkeypatch)

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        _ = messages
        output = {
            "body_regions": ["chest"],
            "entity_type": "diagnosis",
            "etiologies": ["vascular:thrombotic"],
            "applicable_modalities": ["CT"],
            "classification_rationale": "Pulmonary embolism is a thrombotic vascular diagnosis.",
            "field_confidence": {
                "body_regions": "high",
                "entity_type": "high",
                "etiologies": "medium",
                "applicable_modalities": "high",
                "index_codes": "high",
                "anatomic_locations": "high",
            },
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    output,
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model)

    assert result.model.etiologies == [EtiologyCode.VASCULAR_THROMBOTIC]
    assert not any("Optional field 'etiologies' ignored without high confidence" in warning for warning in result.review.warnings)


@pytest.mark.asyncio
async def test_reassess_drops_low_confidence_etiologies(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low-confidence etiology values are still treated as unsupported optional metadata."""
    _mock_gathering(monkeypatch)

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        _ = messages
        output = {
            "body_regions": ["chest"],
            "entity_type": "finding",
            "etiologies": ["neoplastic:malignant"],
            "applicable_modalities": ["CT"],
            "classification_rationale": "Generic finding with uncertain malignant cause.",
            "field_confidence": {
                "body_regions": "high",
                "entity_type": "high",
                "etiologies": "low",
                "applicable_modalities": "high",
                "index_codes": "high",
                "anatomic_locations": "high",
            },
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    output,
                    tool_call_id="pyd_ai_tool_call_id__output",
                )
            ]
        )

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model)

    assert result.model.etiologies is None
    assert any("Low-confidence optional field 'etiologies' ignored" in warning for warning in result.review.warnings)


def test_etiology_prompt_serializes_identity_context(finding_model: FindingModelFull) -> None:
    """Structured identity context should be JSON-serializable for the etiology agent."""
    prompt = _etiology_decision_prompt(
        finding_model,
        {},
        {},
        identity=IdentityDecision(
            entity_type=EntityType.FINDING,
            expected_time_course=ExpectedTimeCourse(duration=ExpectedDuration.WEEKS),
        ),
        fill_blanks_only=False,
    )

    payload = json.loads(prompt)

    assert payload["identity_context"] == {
        "entity_type": "finding",
        "expected_time_course": {"duration": "weeks", "modifiers": []},
    }


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
                    "index_codes": FieldConfidence.HIGH,
                    "anatomic_locations": FieldConfidence.HIGH,
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

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
                    "index_codes": FieldConfidence.HIGH,
                    "anatomic_locations": FieldConfidence.HIGH,
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model, fill_blanks_only=True)

    assert call_count == 2
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.entity_type == EntityType.FINDING
    assert result.model.applicable_modalities == [Modality.CT]
    assert result.review.assignment_mode == "fill_blanks_only"


# ---------------------------------------------------------------------------
# 9. Reassess mode ignores clear_fields for required fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reassess_clearing_required_field_triggers_retry(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reassess mode should ignore required-field clears and warn instead of failing."""
    _mock_gathering(monkeypatch)

    call_count = 0

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        decision = MetadataAssignmentDecision(
            clear_fields=["body_regions"],
            classification_rationale="Invalid required-field clear should be ignored.",
            field_confidence={},
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

    assert call_count == 1
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.review.classification_rationale == "Invalid required-field clear should be ignored."
    assert any("clear_fields: required field 'body_regions' ignored" in warning for warning in result.review.warnings)


@pytest.mark.asyncio
async def test_reassess_candidate_selections_override_clear_fields(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ontology/anatomy selections should not be cleared by unrelated clear_fields output."""
    _mock_gathering(monkeypatch)

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        _ = messages
        decision = MetadataAssignmentDecision(
            ontology_decisions=[
                OntologyCandidateDecision(
                    candidate_id="SNOMEDCT:233604007",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH,
                    selected_as_canonical=True,
                    rationale="Selected candidate should win.",
                )
            ],
            anatomic_decisions=[
                AnatomicCandidateDecision(
                    candidate_id="ANATOMICLOCATIONS:RID1301",
                    selected=True,
                    rationale="Selected candidate should win.",
                )
            ],
            clear_fields=["index_codes", "anatomic_locations"],
            classification_rationale="Candidate selections should override clear_fields.",
            field_confidence={
                "index_codes": FieldConfidence.HIGH,
                "anatomic_locations": FieldConfidence.HIGH,
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model)

    assert result.model.index_codes == [IndexCode(system="SNOMEDCT", code="233604007", display="Pneumonia")]
    assert result.model.anatomic_locations == [
        IndexCode(system="ANATOMICLOCATIONS", code="RID1301", display="lung")
    ]


@pytest.mark.asyncio
async def test_reassess_high_confidence_optional_clear_is_applied(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A high-confidence optional clear should remove unsupported existing identity metadata."""
    _mock_gathering(monkeypatch)

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        _ = messages
        decision = MetadataAssignmentDecision(
            clear_fields=["etiologies", "expected_time_course"],
            classification_rationale="Etiology is unsupported, while time course remains uncertain.",
            field_confidence={
                "etiologies": FieldConfidence.HIGH,
                "expected_time_course": FieldConfidence.MEDIUM,
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model)

    assert result.model.etiologies is None
    assert result.model.expected_time_course == ExpectedTimeCourse(duration=ExpectedDuration.WEEKS)
    assert not any(
        "clear_fields: existing identity field 'etiologies' ignored" in warning for warning in result.review.warnings
    )
    assert any(
        "clear_fields: existing identity field 'expected_time_course' ignored" in warning
        for warning in result.review.warnings
    )


@pytest.mark.asyncio
async def test_reassess_existing_identity_clear_is_applied_for_finding(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit clear_fields decisions should remove existing optional identity metadata in reassess mode."""
    _mock_gathering(monkeypatch)

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        _ = messages
        decision = MetadataAssignmentDecision(
            entity_type=EntityType.FINDING,
            clear_fields=["etiologies", "expected_time_course"],
            classification_rationale="Existing identity fields should be preserved without a supported replacement.",
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model)

    assert result.model.etiologies is None
    assert result.model.expected_time_course is None
    assert not any("existing identity field" in warning for warning in result.review.warnings)


@pytest.mark.asyncio
async def test_reassess_measurement_entity_clears_existing_etiologies(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measurements should not preserve existing etiologies when entity_type changes."""
    _mock_gathering(monkeypatch)

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        _ = messages
        decision = MetadataAssignmentDecision(
            entity_type=EntityType.MEASUREMENT,
            classification_rationale="A measurement has no intrinsic etiology.",
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model)

    assert result.model.entity_type == EntityType.MEASUREMENT
    assert result.model.etiologies is None


@pytest.mark.asyncio
async def test_reassess_low_confidence_patient_and_workflow_clear_is_ignored(
    populated_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low-confidence optional clears should not remove existing patient or workflow fields."""
    _mock_gathering(monkeypatch)

    def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        _ = messages
        decision = MetadataAssignmentDecision(
            clear_fields=["subspecialties", "age_profile", "sex_specificity"],
            classification_rationale="Uncertain optional clear should be ignored.",
            field_confidence={
                "subspecialties": FieldConfidence.LOW,
                "age_profile": FieldConfidence.LOW,
                "sex_specificity": FieldConfidence.LOW,
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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model)

    assert result.model.subspecialties == [Subspecialty.CH]
    assert result.model.age_profile == AgeProfile(applicability=[AgeStage.ADULT])
    assert result.model.sex_specificity == SexSpecificity.SEX_NEUTRAL
    assert any(
        "Low-confidence optional field 'subspecialties' ignored" in warning for warning in result.review.warnings
    )
    assert any("Low-confidence optional field 'age_profile' ignored" in warning for warning in result.review.warnings)
    assert any("Low-confidence optional field 'sex_specificity' ignored" in warning for warning in result.review.warnings)


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

    agent = create_metadata_assignment_agent(model=FunctionModel(model_function))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(populated_model)

    # Pre-existing index_codes should appear in the prompt with SYSTEM:CODE format
    prompt = captured["prompt"]
    assert "SNOMEDCT:233604007" in prompt

    # Result should contain the pre-existing index code
    assert any(c.system == "SNOMEDCT" and c.code == "233604007" for c in (result.model.index_codes or []))
