"""Tests for the canonical assign_metadata pipeline (Slice 7)."""

from __future__ import annotations

import asyncio
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
    OntologyCandidateRejectionReason,
    OntologyCandidateRelationship,
)
from findingmodel_ai.search.anatomic import LocationSearchResponse
from findingmodel_ai.search.ontology import CategorizedOntologyConcepts
from pydantic_ai import models
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False


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
        marginal_concepts=[
            OntologySearchResult(
                concept_id="RID9999",
                concept_text="lung opacity",
                score=0.60,
                table_name="radlex",
            )
        ],
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
        alternate_locations=[
            OntologySearchResult(
                concept_id="RID2848",
                concept_text="lower respiratory tract",
                score=0.0,
                table_name="anatomic_locations",
            )
        ],
        reasoning="Lung is the primary site of pneumonia.",
    )


@pytest.mark.asyncio
async def test_assign_metadata_assembles_canonical_result(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment._get_trace_id", lambda: "trace-123")

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT, Modality.XR],
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="RADLEX:RID5350",
                relationship=OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE,
                selected_as_canonical=True,
                rationale="RadLex near-equivalent for the same finding.",
            ),
            OntologyCandidateDecision(
                candidate_id="RADLEX:RID9999",
                relationship=OntologyCandidateRelationship.RELATED,
                selected_as_canonical=False,
                rationale="Describes a broader imaging appearance.",
                rejection_reason=OntologyCandidateRejectionReason.OVERLAPPING_SCOPE,
            ),
        ],
        classification_rationale="Pneumonia is a chest finding usually seen on CT and radiography.",
        field_confidence={"body_regions": FieldConfidence.HIGH, "entity_type": FieldConfidence.HIGH},
    )
    agent = create_metadata_assignment_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model)

    assert result.model.name == "pneumonia"
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.entity_type == EntityType.FINDING
    assert result.model.applicable_modalities == [Modality.CT, Modality.XR]
    assert result.model.index_codes is not None
    assert [(code.system, code.code) for code in result.model.index_codes] == [
        ("SNOMEDCT", "233604007"),
        ("RADLEX", "RID5350"),
    ]
    assert [code.display for code in result.model.index_codes] == ["Pneumonia", "pneumonia"]
    assert result.model.anatomic_locations is not None
    assert [(code.system, code.code) for code in result.model.anatomic_locations] == [("ANATOMICLOCATIONS", "RID1301")]
    assert [code.display for code in result.model.anatomic_locations] == ["lung"]

    assert result.review.logfire_trace_id == "trace-123"
    assert result.review.field_confidence["body_regions"] == FieldConfidence.HIGH
    assert len(result.review.ontology_candidates.canonical_codes) == 2
    assert len(result.review.ontology_candidates.review_candidates) == 1
    assert result.review.ontology_candidates.canonical_codes[0].code.display == "Pneumonia"
    assert result.review.ontology_candidates.canonical_codes[1].code.display == "pneumonia"
    assert result.review.ontology_candidates.review_candidates[0].code.code == "RID9999"
    assert result.review.ontology_candidates.review_candidates[0].code.display == "lung opacity"
    assert (
        result.review.ontology_candidates.review_candidates[0].rejection_reason
        == OntologyCandidateRejectionReason.OVERLAPPING_SCOPE
    )
    assert len(result.review.anatomic_candidates) == 2
    assert result.review.anatomic_candidates[0].location.display == "lung"
    assert result.review.anatomic_candidates[1].location.display == "lower respiratory tract"
    assert result.review.classification_rationale


@pytest.mark.asyncio
async def test_assign_metadata_function_model_receives_candidate_context(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment._get_trace_id", lambda: None)

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
                        "classification_rationale": "No metadata changes required.",
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

    result = await assign_metadata(finding_model)

    assert "RID5350" in captured["prompt"]
    assert "RID1301" in captured["prompt"]
    assert "lung infection" in captured["prompt"]
    assert '"display": "Pneumonia"' in captured["prompt"]
    assert '"display": "lung"' in captured["prompt"]
    assert result.model.entity_type == EntityType.FINDING
    assert result.model.index_codes is not None
    assert [(code.system, code.code) for code in result.model.index_codes] == [("SNOMEDCT", "233604007")]
    assert result.review.classification_rationale == "No metadata changes required."


@pytest.mark.asyncio
async def test_assign_metadata_reassesses_populated_model(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    complete_model = finding_model.model_copy(
        update={
            "body_regions": [BodyRegion.CHEST],
            "subspecialties": [Subspecialty.CH],
            "etiologies": [EtiologyCode.INFLAMMATORY_INFECTIOUS],
            "entity_type": EntityType.FINDING,
            "applicable_modalities": [Modality.XR],
            "expected_time_course": ExpectedTimeCourse(duration=ExpectedDuration.WEEKS),
            "age_profile": AgeProfile(applicability=[AgeStage.ADULT]),
            "sex_specificity": SexSpecificity.SEX_NEUTRAL,
            "index_codes": [IndexCode(system="SNOMEDCT", code="233604007", display="Pneumonia")],
            "anatomic_locations": [IndexCode(system="ANATOMICLOCATIONS", code="RID1301", display="lung")],
        }
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment._get_trace_id", lambda: None)

    classifier_called = False

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        subspecialties=[Subspecialty.CH],
        etiologies=[EtiologyCode.INFLAMMATORY_INFECTIOUS],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.XR],
        sex_specificity=SexSpecificity.SEX_NEUTRAL,
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="SNOMEDCT:233604007",
                relationship=OntologyCandidateRelationship.EXACT_MATCH,
                selected_as_canonical=True,
                rationale="Existing SNOMED code confirmed.",
            ),
        ],
        classification_rationale="Confirmed existing metadata is correct.",
        field_confidence={"body_regions": FieldConfidence.HIGH, "entity_type": FieldConfidence.HIGH},
    )

    original_create = create_metadata_assignment_agent

    def tracking_create(**kwargs: Any) -> Any:
        nonlocal classifier_called
        classifier_called = True
        return original_create(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", tracking_create)

    result = await assign_metadata(complete_model)

    assert classifier_called, "Classifier should always be called (no fast-path skip)"
    assert result.review.assignment_mode == "reassess"
    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.entity_type == EntityType.FINDING
    assert result.review.ontology_candidates.canonical_codes[0].code.code == "233604007"


@pytest.mark.asyncio
async def test_assign_metadata_surfaces_gathering_failures_as_warnings(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(side_effect=RuntimeError("ontology exploded")),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment._get_trace_id", lambda: None)

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        classification_rationale="Applied metadata despite missing ontology candidates.",
        field_confidence={
            "body_regions": FieldConfidence.MEDIUM,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
        },
    )
    agent = create_metadata_assignment_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    result = await assign_metadata(finding_model)

    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.anatomic_locations is not None
    assert [(code.system, code.code) for code in result.model.anatomic_locations] == [("ANATOMICLOCATIONS", "RID1301")]
    assert result.model.index_codes is None
    assert result.review.warnings == ["Ontology candidate gathering failed: ontology exploded"]
    assert result.review.ontology_candidates.canonical_codes == []
    assert result.review.classification_rationale == "Applied metadata despite missing ontology candidates."


@pytest.mark.asyncio
async def test_assign_metadata_gathers_ontology_and_anatomic_in_parallel(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    ontology_started = asyncio.Event()
    anatomic_started = asyncio.Event()
    release_both = asyncio.Event()
    starts: list[str] = []

    async def fake_match_ontology_concepts(**kwargs: Any) -> CategorizedOntologyConcepts:
        _ = kwargs
        starts.append("ontology")
        ontology_started.set()
        await release_both.wait()
        return _ontology_results()

    async def fake_find_anatomic_locations(**kwargs: Any) -> LocationSearchResponse:
        _ = kwargs
        starts.append("anatomic")
        anatomic_started.set()
        await release_both.wait()
        return _anatomic_results()

    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        fake_match_ontology_concepts,
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        fake_find_anatomic_locations,
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment._get_trace_id", lambda: None)

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        classification_rationale="Parallel gather test.",
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
        },
    )
    agent = create_metadata_assignment_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))
    monkeypatch.setattr("findingmodel_ai.metadata.assignment.create_metadata_assignment_agent", lambda **_: agent)

    task = asyncio.create_task(assign_metadata(finding_model))

    await asyncio.wait_for(ontology_started.wait(), timeout=1)
    await asyncio.wait_for(anatomic_started.wait(), timeout=1)
    assert starts == ["ontology", "anatomic"] or starts == ["anatomic", "ontology"]

    release_both.set()
    result = await task

    assert result.model.body_regions == [BodyRegion.CHEST]
