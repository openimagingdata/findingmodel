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
    create_anatomy_decision_agent,
    create_entity_type_agent,
    create_etiology_tempo_agent,
    create_modality_applicability_agent,
    create_ontology_decision_agent,
    create_patient_applicability_agent,
    create_subspecialty_domain_agent,
)
from findingmodel_ai.metadata.decisions import (
    AnatomicCandidateDecision,
    AnatomyDecision,
    EntityTypeDecision,
    EtiologyTempoDecision,
    ModalityApplicabilityDecision,
    OntologyDecision,
    PatientApplicabilityDecision,
    SubspecialtyDomainDecision,
)
from findingmodel_ai.metadata.ontology_cache import OntologyLookupCache
from findingmodel_ai.metadata.types import (
    FieldConfidence,
    OntologyCandidateRejectionReason,
    OntologyCandidateRelationship,
)
from findingmodel_ai.search.anatomic import LocationSearchResponse
from findingmodel_ai.search.ontology import CategorizedOntologyConcepts
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False


def _patch_split_agents(monkeypatch: pytest.MonkeyPatch, decision: MetadataAssignmentDecision) -> None:
    """Patch split metadata agents from one expected combined decision."""

    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_ontology_decision_agent",
        lambda **_: create_ontology_decision_agent(
            model=TestModel(
                custom_output_args=OntologyDecision(
                    ontology_decisions=decision.ontology_decisions,
                ).model_dump(mode="json")
            )
        ),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_anatomy_decision_agent",
        lambda **_: create_anatomy_decision_agent(
            model=TestModel(
                custom_output_args=AnatomyDecision(
                    body_regions=decision.body_regions,
                    anatomic_decisions=decision.anatomic_decisions,
                ).model_dump(mode="json")
            )
        ),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_entity_type_agent",
        lambda **_: create_entity_type_agent(
            model=TestModel(
                custom_output_args=EntityTypeDecision(
                    entity_type=decision.entity_type,
                    field_confidence=decision.field_confidence,
                ).model_dump(mode="json")
            )
        ),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_patient_applicability_agent",
        lambda **_: create_patient_applicability_agent(
            model=TestModel(
                custom_output_args=PatientApplicabilityDecision(
                    age_profile=decision.age_profile,
                    sex_specificity=decision.sex_specificity,
                    field_confidence=decision.field_confidence,
                ).model_dump(mode="json")
            )
        ),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_subspecialty_domain_agent",
        lambda **_: create_subspecialty_domain_agent(
            model=TestModel(
                custom_output_args=SubspecialtyDomainDecision(
                    subspecialties=decision.subspecialties,
                    field_confidence=decision.field_confidence,
                ).model_dump(mode="json")
            )
        ),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_modality_applicability_agent",
        lambda **_: create_modality_applicability_agent(
            model=TestModel(
                custom_output_args=ModalityApplicabilityDecision(
                    applicable_modalities=decision.applicable_modalities,
                    field_confidence=decision.field_confidence,
                ).model_dump(mode="json")
            )
        ),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.create_etiology_tempo_agent",
        lambda **_: create_etiology_tempo_agent(
            model=TestModel(
                custom_output_args=EtiologyTempoDecision(
                    etiologies=decision.etiologies,
                    expected_time_course=decision.expected_time_course,
                    field_confidence=decision.field_confidence,
                ).model_dump(mode="json")
            )
        ),
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


def _air_in_esophagus_ontology_results() -> CategorizedOntologyConcepts:
    return CategorizedOntologyConcepts(
        exact_matches=[
            OntologySearchResult(
                concept_id="RID95",
                concept_text="esophagus",
                score=0.99,
                table_name="radlex",
            ),
            OntologySearchResult(
                concept_id="056",
                concept_text="air in esophagus",
                score=0.99,
                table_name="gamuts",
            ),
        ],
        should_include=[],
        marginal_concepts=[],
        search_summary="Test anatomy-filtering ontology summary",
        excluded_anatomical=[],
    )


def _esophagus_anatomic_results() -> LocationSearchResponse:
    return LocationSearchResponse(
        primary_location=OntologySearchResult(
            concept_id="RID95",
            concept_text="esophagus",
            score=0.0,
            table_name="anatomic_locations",
        ),
        alternate_locations=[],
        reasoning="Esophagus is the anatomic site.",
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
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

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
    assert result.review.field_confidence["body_regions"] == 0.9
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
    assert result.review.classification_rationale == ""


@pytest.mark.asyncio
async def test_assign_metadata_filters_ontology_candidates_that_duplicate_anatomy(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    air_model = finding_model.model_copy(
        update={
            "name": "air in esophagus",
            "description": "Air visible in the esophagus.",
            "anatomic_locations": None,
        }
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_air_in_esophagus_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_esophagus_anatomic_results()),
    )
    monkeypatch.setattr("findingmodel_ai.metadata.assignment._get_trace_id", lambda: None)

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.XR],
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="GAMUTS:056",
                relationship=OntologyCandidateRelationship.EXACT_MATCH,
                selected_as_canonical=True,
            )
        ],
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(air_model)

    assert result.model.index_codes is not None
    assert [(code.system, code.code) for code in result.model.index_codes] == [("GAMUTS", "056")]
    assert result.model.anatomic_locations is not None
    assert [(code.system, code.code) for code in result.model.anatomic_locations] == [("ANATOMICLOCATIONS", "RID95")]


@pytest.mark.asyncio
async def test_assign_metadata_records_ontology_cache(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Assignment can write durable ontology evidence without changing the model output."""
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="RADLEX:RID9999",
                relationship=OntologyCandidateRelationship.RELATED,
                selected_as_canonical=False,
                rejection_reason=OntologyCandidateRejectionReason.OVERLAPPING_SCOPE,
            ),
        ],
        classification_rationale="Cache evidence test.",
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    cache_path = tmp_path / "ontology-cache.duckdb"
    result = await assign_metadata(finding_model, ontology_cache=cache_path)

    assert result.model.index_codes is not None
    cache = OntologyLookupCache(cache_path)
    exact = cache.get("SNOMEDCT", "233604007")
    rejected = cache.get("RADLEX", "RID9999")
    assert exact is not None
    assert exact.preferred_display == "Pneumonia"
    assert exact.usage == "canonical_selected"
    assert rejected is not None
    assert rejected.preferred_display == "lung opacity"
    assert rejected.usage == "rejected_candidate"
    assert rejected.relationship == "related"
    assert rejected.rejection_reason == "overlapping_scope"


@pytest.mark.asyncio
async def test_assign_metadata_does_not_promote_related_ontology_candidate(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Related ontology candidates should stay review evidence, not canonical index codes."""
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="RADLEX:RID9999",
                relationship=OntologyCandidateRelationship.RELATED,
                selected_as_canonical=True,
                rationale="Incorrectly tried to promote a related broader appearance.",
            )
        ],
        classification_rationale="Related ontology candidate guardrail test.",
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(finding_model)

    assert result.model.index_codes is not None
    assert ("RADLEX", "RID9999") not in {(code.system, code.code) for code in result.model.index_codes}
    assert any(
        "Ignoring canonical ontology selection for RADLEX:RID9999" in warning for warning in result.review.warnings
    )
    assert any(
        candidate.code.system == "RADLEX" and candidate.code.code == "RID9999"
        for candidate in result.review.ontology_candidates.review_candidates
    )


@pytest.mark.asyncio
async def test_assign_metadata_promotes_canonical_relationship_without_boolean(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact/substitutable relationship should be canonical unless a rejection reason is present."""
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="RADLEX:RID5350",
                relationship=OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE,
                selected_as_canonical=False,
            ),
            OntologyCandidateDecision(
                candidate_id="RADLEX:RID9999",
                relationship=OntologyCandidateRelationship.EXACT_MATCH,
                selected_as_canonical=False,
                rejection_reason=OntologyCandidateRejectionReason.WRONG_CONCEPT,
            ),
        ],
        classification_rationale="Canonical relationship consistency test.",
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(finding_model)

    assert result.model.index_codes is not None
    assert ("RADLEX", "RID5350") in [(code.system, code.code) for code in result.model.index_codes]
    assert ("RADLEX", "RID9999") not in [(code.system, code.code) for code in result.model.index_codes]


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

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(finding_model)

    assert result.model.entity_type == EntityType.FINDING
    assert result.model.index_codes is not None
    assert [(code.system, code.code) for code in result.model.index_codes] == [("SNOMEDCT", "233604007")]
    assert result.review.classification_rationale == ""


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
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "subspecialties": FieldConfidence.HIGH,
            "etiologies": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "sex_specificity": FieldConfidence.HIGH,
        },
    )

    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(complete_model)

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
            "anatomic_locations": FieldConfidence.MEDIUM,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(finding_model)

    assert result.model.body_regions == [BodyRegion.CHEST]
    assert result.model.anatomic_locations is not None
    assert [(code.system, code.code) for code in result.model.anatomic_locations] == [("ANATOMICLOCATIONS", "RID1301")]
    assert result.model.index_codes is None
    assert result.review.warnings == ["Ontology candidate gathering failed: ontology exploded"]
    assert result.review.ontology_candidates.canonical_codes == []
    assert result.review.classification_rationale == ""


@pytest.mark.asyncio
async def test_assign_metadata_passes_ontology_labels_to_anatomic_search(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[str] = []
    captured_locality_labels: list[str] = []

    async def fake_match_ontology_concepts(**kwargs: Any) -> CategorizedOntologyConcepts:
        _ = kwargs
        await asyncio.sleep(0)
        starts.append("ontology")
        return _ontology_results()

    async def fake_find_anatomic_locations(**kwargs: Any) -> LocationSearchResponse:
        await asyncio.sleep(0)
        starts.append("anatomic")
        captured_locality_labels.extend(kwargs.get("locality_labels") or [])
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
        classification_rationale="Ontology-informed anatomy gather test.",
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "index_codes": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(finding_model)

    assert starts == ["ontology", "anatomic"]
    assert "Pneumonia" in captured_locality_labels
    assert result.model.body_regions == [BodyRegion.CHEST]


@pytest.mark.asyncio
async def test_reassess_allows_optional_null_outputs(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only entity_type is required; optional nulls clear reviewed existing metadata in reassess mode."""
    populated_model = finding_model.model_copy(
        update={
            "body_regions": [BodyRegion.CHEST],
            "subspecialties": [Subspecialty.CH],
            "etiologies": [EtiologyCode.INFLAMMATORY_INFECTIOUS],
            "entity_type": EntityType.FINDING,
            "applicable_modalities": [Modality.CT],
            "expected_time_course": ExpectedTimeCourse(duration=ExpectedDuration.WEEKS),
            "age_profile": AgeProfile(applicability=[AgeStage.ADULT]),
            "sex_specificity": SexSpecificity.SEX_NEUTRAL,
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

    decision = MetadataAssignmentDecision(
        entity_type=EntityType.FINDING,
        field_confidence={"entity_type": FieldConfidence.HIGH},
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(populated_model)

    assert result.model.entity_type == EntityType.FINDING
    assert result.model.body_regions is None
    assert result.model.applicable_modalities is None
    assert any(warning == "metadata cleared: body_regions" for warning in result.review.warnings)
    assert any(warning == "metadata cleared: applicable_modalities" for warning in result.review.warnings)


@pytest.mark.asyncio
async def test_reassess_requires_only_entity_type(
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

    decision = MetadataAssignmentDecision(
        entity_type=EntityType.FINDING,
        field_confidence={"entity_type": FieldConfidence.HIGH},
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(
        finding_model.model_copy(update={"body_regions": None, "applicable_modalities": None})
    )

    assert result.model.entity_type == EntityType.FINDING
    assert result.model.body_regions is None
    assert result.model.applicable_modalities is None


@pytest.mark.asyncio
async def test_existing_codes_and_anatomy_are_kept_on_silence(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_model = finding_model.model_copy(
        update={
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

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(existing_model)

    assert [(code.system, code.code) for code in result.model.index_codes or []] == [("SNOMEDCT", "233604007")]
    assert [(code.system, code.code) for code in result.model.anatomic_locations or []] == [
        ("ANATOMICLOCATIONS", "RID1301")
    ]


@pytest.mark.asyncio
async def test_existing_anatomic_location_requires_reason_to_remove(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_model = finding_model.model_copy(
        update={"anatomic_locations": [IndexCode(system="ANATOMICLOCATIONS", code="RID1301", display="lung")]}
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        anatomic_decisions=[AnatomicCandidateDecision(candidate_id="ANATOMICLOCATIONS:RID1301", selected=False)],
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(existing_model)

    assert [(code.system, code.code) for code in result.model.anatomic_locations or []] == [
        ("ANATOMICLOCATIONS", "RID1301")
    ]
    assert any("Existing anatomic location kept" in warning for warning in result.review.warnings)


@pytest.mark.asyncio
async def test_existing_anatomic_location_can_be_removed_with_reason(
    finding_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_model = finding_model.model_copy(
        update={"anatomic_locations": [IndexCode(system="ANATOMICLOCATIONS", code="RID1301", display="lung")]}
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.match_ontology_concepts",
        AsyncMock(return_value=_ontology_results()),
    )
    monkeypatch.setattr(
        "findingmodel_ai.metadata.assignment.find_anatomic_locations",
        AsyncMock(return_value=_anatomic_results()),
    )

    decision = MetadataAssignmentDecision(
        body_regions=[BodyRegion.CHEST],
        entity_type=EntityType.FINDING,
        applicable_modalities=[Modality.CT],
        anatomic_decisions=[
            AnatomicCandidateDecision(
                candidate_id="ANATOMICLOCATIONS:RID1301",
                selected=False,
                rejection_reason="direct contradiction",
            )
        ],
        field_confidence={
            "body_regions": FieldConfidence.HIGH,
            "entity_type": FieldConfidence.HIGH,
            "applicable_modalities": FieldConfidence.HIGH,
            "anatomic_locations": FieldConfidence.HIGH,
        },
    )
    _patch_split_agents(monkeypatch, decision)

    result = await assign_metadata(existing_model)

    assert result.model.anatomic_locations is None


def test_metadata_assignment_decision_ignores_invalid_confidence_key() -> None:
    decision = MetadataAssignmentDecision.model_validate({
        "classification_rationale": "Invalid confidence key.",
        "field_confidence": {"ontology_decisions": "high", "body_regions": "95"},
    })

    assert decision.field_confidence == {"body_regions": 0.95}
