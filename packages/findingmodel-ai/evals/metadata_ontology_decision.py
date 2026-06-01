"""Focused evals for the metadata ontology decision sub-agent."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from findingmodel_ai.metadata.assignment import create_ontology_decision_agent
from findingmodel_ai.metadata.decisions import OntologyDecision
from findingmodel_ai.metadata.types import OntologyCandidateRelationship
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from evals.metadata_scoring import print_weighted_summary, score_set_similarity

EVAL_MAX_CONCURRENCY = 2
EVALUATOR_WEIGHTS: dict[str, float] = {"OntologyEvaluator": 1.0}


class OntologyDecisionInput(BaseModel):
    """Input payload for one ontology-agent replay case."""

    payload: dict[str, Any]


class OntologyDecisionExpectedOutput(BaseModel):
    """Expected canonical ontology IDs for one replay case."""

    selected_candidate_ids: set[str] = Field(default_factory=set)


class OntologyDecisionActualOutput(BaseModel):
    """Observed ontology-agent output for one replay case."""

    selected_candidate_ids: set[str] = Field(default_factory=set)
    raw_output: OntologyDecision | None = None
    error: str | None = None


def _ontology_candidate(
    candidate_id: str,
    text: str,
    *,
    source_bucket: str,
    relationship: str,
    selected: bool,
) -> dict[str, Any]:
    system, _, _code = candidate_id.partition(":")
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "table_name": system.lower(),
        "system": system,
        "source_bucket": source_bucket,
        "default_relationship": relationship,
        "default_selected_as_canonical": selected,
    }


def _payload(
    *,
    name: str,
    description: str,
    synonyms: list[str] | None = None,
    existing_index_codes: list[dict[str, str]] | None = None,
    ontology_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "assignment_mode": "reassess",
        "finding_model": {
            "oifm_id": f"EVAL_{name.upper().replace(' ', '_')}",
            "name": name,
            "description": description,
            "synonyms": synonyms or [],
            "tags": [],
            "existing_structured_metadata": {
                "body_regions": None,
                "subspecialties": None,
                "etiologies": None,
                "entity_type": None,
                "applicable_modalities": None,
                "expected_time_course": None,
                "age_profile": None,
                "sex_specificity": None,
                "index_codes": existing_index_codes or [],
                "anatomic_locations": [],
            },
            "attributes": [],
        },
        "ontology_candidates": ontology_candidates,
        "anatomic_candidates": [],
        "task": "Decide ontology candidate relationships and canonical index-code selection.",
    }


def _case(
    name: str,
    *,
    payload: dict[str, Any],
    selected_candidate_ids: set[str],
) -> Case[OntologyDecisionInput, OntologyDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=OntologyDecisionInput(payload=payload),
        expected_output=OntologyDecisionExpectedOutput(selected_candidate_ids=selected_candidate_ids),
    )


CASES: list[Case[OntologyDecisionInput, OntologyDecisionExpectedOutput]] = [
    _case(
        "unqualified_tumor_rejects_malignant_only_candidate",
        payload=_payload(
            name="primary brain tumor",
            description="Tumor originating within the brain",
            synonyms=["intracranial neoplasm"],
            existing_index_codes=[
                {"system": "GAMUTS", "code": "6882", "display": "primary brain tumor"},
            ],
            ontology_candidates=[
                _ontology_candidate(
                    "GAMUTS:6882",
                    "primary brain tumor",
                    source_bucket="existing_index_codes",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH.value,
                    selected=True,
                ),
                _ontology_candidate(
                    "SNOMEDCT:93727008",
                    "Primary malignant neoplasm of brain",
                    source_bucket="exact_matches",
                    relationship=OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE.value,
                    selected=True,
                ),
            ],
        ),
        selected_candidate_ids={"GAMUTS:6882"},
    ),
    _case(
        "exact_matches_across_systems_can_both_be_canonical",
        payload=_payload(
            name="pulmonary consolidation",
            description="Air-space opacity representing consolidation in lung parenchyma.",
            ontology_candidates=[
                _ontology_candidate(
                    "RADLEX:RID28473",
                    "pulmonary consolidation",
                    source_bucket="exact_matches",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH.value,
                    selected=True,
                ),
                _ontology_candidate(
                    "SNOMEDCT:95436008",
                    "pulmonary consolidation",
                    source_bucket="exact_matches",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH.value,
                    selected=True,
                ),
                _ontology_candidate(
                    "SNOMEDCT:233604007",
                    "pneumonia",
                    source_bucket="marginal",
                    relationship=OntologyCandidateRelationship.RELATED.value,
                    selected=False,
                ),
            ],
        ),
        selected_candidate_ids={"RADLEX:RID28473", "SNOMEDCT:95436008"},
    ),
    _case(
        "dropped_qualifier_is_not_canonical",
        payload=_payload(
            name="subacute infarct",
            description="Infarct with subacute imaging appearance.",
            ontology_candidates=[
                _ontology_candidate(
                    "RADLEX:SUBACUTE_INFARCT",
                    "subacute infarct",
                    source_bucket="exact_matches",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH.value,
                    selected=True,
                ),
                _ontology_candidate(
                    "SNOMEDCT:INFARCT",
                    "infarct",
                    source_bucket="should_include",
                    relationship=OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE.value,
                    selected=True,
                ),
            ],
        ),
        selected_candidate_ids={"RADLEX:SUBACUTE_INFARCT"},
    ),
    _case(
        "anatomy_only_candidate_is_review_evidence",
        payload=_payload(
            name="gas in esophagus",
            description="Gas visible within the esophageal lumen.",
            ontology_candidates=[
                _ontology_candidate(
                    "RADLEX:GAS_IN_ESOPHAGUS",
                    "gas in esophagus",
                    source_bucket="exact_matches",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH.value,
                    selected=True,
                ),
                _ontology_candidate(
                    "RADLEX:ESOPHAGUS",
                    "esophagus",
                    source_bucket="should_include",
                    relationship=OntologyCandidateRelationship.RELATED.value,
                    selected=False,
                ),
            ],
        ),
        selected_candidate_ids={"RADLEX:GAS_IN_ESOPHAGUS"},
    ),
    _case(
        "assessment_target_disease_is_not_canonical",
        payload=_payload(
            name="parenchymal volume score",
            description="Structured score summarizing parenchymal volume loss severity.",
            ontology_candidates=[
                _ontology_candidate(
                    "LOINC:VOLUME_SCORE",
                    "parenchymal volume score",
                    source_bucket="exact_matches",
                    relationship=OntologyCandidateRelationship.EXACT_MATCH.value,
                    selected=True,
                ),
                _ontology_candidate(
                    "SNOMEDCT:ATROPHY",
                    "parenchymal atrophy",
                    source_bucket="should_include",
                    relationship=OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE.value,
                    selected=True,
                ),
            ],
        ),
        selected_candidate_ids={"LOINC:VOLUME_SCORE"},
    ),
]


class OntologyEvaluator(Evaluator[OntologyDecisionInput, OntologyDecisionActualOutput, OntologyDecisionExpectedOutput]):
    """Score canonical candidate agreement with partial credit."""

    def evaluate(
        self, ctx: EvaluatorContext[OntologyDecisionInput, OntologyDecisionActualOutput, OntologyDecisionExpectedOutput]
    ) -> float:
        if ctx.output.error is not None:
            return 0.0
        return score_set_similarity(ctx.output.selected_candidate_ids, ctx.expected_output.selected_candidate_ids)


async def run_ontology_decision_task(case_input: OntologyDecisionInput) -> OntologyDecisionActualOutput:
    """Run only the ontology decision agent for one replay payload."""

    agent = create_ontology_decision_agent()
    try:
        result = await agent.run(json.dumps(case_input.payload, indent=2))
    except Exception as exc:  # pragma: no cover - useful in eval reports
        return OntologyDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")

    selected = {
        decision.candidate_id for decision in result.output.ontology_decisions if decision.selected_as_canonical
    }
    return OntologyDecisionActualOutput(selected_candidate_ids=selected, raw_output=result.output)


metadata_ontology_decision_dataset: Dataset[
    OntologyDecisionInput, OntologyDecisionActualOutput, OntologyDecisionExpectedOutput
] = Dataset(
    cases=CASES,
    evaluators=[OntologyEvaluator()],
)


async def run_metadata_ontology_decision_evals() -> EvaluationReport[
    OntologyDecisionInput, OntologyDecisionActualOutput, OntologyDecisionExpectedOutput
]:
    """Run the focused ontology-decision replay suite."""

    return await metadata_ontology_decision_dataset.evaluate(
        run_ontology_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )


if __name__ == "__main__":
    try:
        print("\nRunning focused ontology decision evaluation suite...")
        print("=" * 80)
        report = asyncio.run(run_metadata_ontology_decision_evals())
        report.print(include_input=False, include_expected_output=False, include_durations=True)
        print_weighted_summary(report, EVALUATOR_WEIGHTS, title="Ontology decision")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
