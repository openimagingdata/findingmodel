"""Tests for the metadata prompt-repair pilot infrastructure."""

from __future__ import annotations

import pytest
from findingmodel import BodyRegion, EntityType, EtiologyCode, Modality, SexSpecificity, Subspecialty
from findingmodel_ai.metadata.assignment import (
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
    OntologyCandidateDecision,
    OntologyDecision,
    PatientApplicabilityDecision,
    SubspecialtyDomainDecision,
)
from findingmodel_ai.metadata.prompt_loader import load_metadata_prompt
from findingmodel_ai.metadata.types import OntologyCandidateRelationship
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False

PROMPT_IDS = (
    "entity_type",
    "patient_applicability",
    "subspecialty_domain",
    "modality_applicability",
    "etiology_tempo",
    "ontology_decision",
    "anatomy_decision",
)


@pytest.mark.parametrize("prompt_id", PROMPT_IDS)
def test_load_metadata_prompts_are_externalized(prompt_id: str) -> None:
    prompt = load_metadata_prompt(prompt_id)

    assert prompt.strip()


def test_load_metadata_prompt_rejects_path_like_ids() -> None:
    with pytest.raises(ValueError, match="Invalid metadata prompt id"):
        load_metadata_prompt("../subspecialty_domain")


async def test_subspecialty_domain_agent_uses_focused_decision_model() -> None:
    decision = SubspecialtyDomainDecision(subspecialties=[Subspecialty.VA])
    agent = create_subspecialty_domain_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    result = await agent.run('{"task": "Assign only subspecialties."}')

    assert result.output.subspecialties == [Subspecialty.VA]


async def test_modality_applicability_agent_uses_focused_decision_model() -> None:
    decision = ModalityApplicabilityDecision(applicable_modalities=[Modality.CT])
    agent = create_modality_applicability_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    result = await agent.run('{"task": "Assign only applicable_modalities."}')

    assert result.output.applicable_modalities == [Modality.CT]


async def test_etiology_tempo_agent_uses_focused_decision_model() -> None:
    decision = EtiologyTempoDecision(etiologies=[EtiologyCode.VASCULAR_THROMBOTIC])
    agent = create_etiology_tempo_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    result = await agent.run('{"task": "Assign only etiologies and expected_time_course."}')

    assert result.output.etiologies == [EtiologyCode.VASCULAR_THROMBOTIC]


async def test_entity_type_agent_uses_focused_decision_model() -> None:
    decision = EntityTypeDecision(entity_type=EntityType.FINDING)
    agent = create_entity_type_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    result = await agent.run('{"task": "Assign only entity_type."}')

    assert result.output.entity_type == EntityType.FINDING


async def test_patient_applicability_agent_uses_focused_decision_model() -> None:
    decision = PatientApplicabilityDecision(sex_specificity=SexSpecificity.SEX_NEUTRAL)
    agent = create_patient_applicability_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    result = await agent.run('{"task": "Assign only patient applicability."}')

    assert result.output.sex_specificity == SexSpecificity.SEX_NEUTRAL


async def test_ontology_decision_agent_uses_focused_decision_model() -> None:
    decision = OntologyDecision(
        ontology_decisions=[
            OntologyCandidateDecision(
                candidate_id="SNOMEDCT:123",
                relationship=OntologyCandidateRelationship.EXACT_MATCH,
                selected_as_canonical=True,
            )
        ]
    )
    agent = create_ontology_decision_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    result = await agent.run('{"task": "Decide ontology candidates."}')

    assert result.output.ontology_decisions[0].candidate_id == "SNOMEDCT:123"


async def test_anatomy_decision_agent_uses_focused_decision_model() -> None:
    decision = AnatomyDecision(
        body_regions=[BodyRegion.CHEST],
        anatomic_decisions=[AnatomicCandidateDecision(candidate_id="ANATOMICLOCATIONS:RID1301", selected=True)],
    )
    agent = create_anatomy_decision_agent(model=TestModel(custom_output_args=decision.model_dump(mode="json")))

    result = await agent.run('{"task": "Decide anatomy candidates."}')

    assert result.output.body_regions == [BodyRegion.CHEST]
    assert result.output.anatomic_decisions[0].selected is True
