"""Focused evals for the metadata entity-type decision sub-agent."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from findingmodel import EntityType
from findingmodel_ai.metadata.assignment import create_entity_type_agent
from findingmodel_ai.metadata.decisions import EntityTypeDecision
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from evals.metadata_scoring import print_weighted_summary, score_entity_type

EVAL_MAX_CONCURRENCY = 2
EVALUATOR_WEIGHTS: dict[str, float] = {"EntityTypeEvaluator": 1.0}


class EntityTypeDecisionInput(BaseModel):
    """Input payload for one entity-type replay case."""

    payload: dict[str, Any]


class EntityTypeDecisionExpectedOutput(BaseModel):
    """Expected entity-type output for one replay case."""

    entity_type: EntityType


class EntityTypeDecisionActualOutput(BaseModel):
    """Observed entity-type output for one replay case."""

    entity_type: EntityType | None = None
    raw_output: EntityTypeDecision | None = None
    error: str | None = None


def _payload(name: str, description: str, *, ontology_text: str | None = None) -> dict[str, Any]:
    ontology = []
    if ontology_text is not None:
        ontology = [
            {
                "candidate_id": f"EVAL:{name.upper().replace(' ', '_')}",
                "text": ontology_text,
                "display": ontology_text,
                "table_name": "eval",
                "system": "EVAL",
                "source_bucket": "existing_index_codes",
                "default_relationship": "exact_match",
                "default_selected_as_canonical": True,
            }
        ]
    return {
        "assignment_mode": "reassess",
        "finding_model": {
            "oifm_id": f"EVAL_{name.upper().replace(' ', '_')}",
            "name": name,
            "description": description,
            "synonyms": [],
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
                "index_codes": [],
                "anatomic_locations": [],
            },
            "attributes": [],
        },
        "ontology_candidates": ontology,
        "anatomic_candidates": [],
        "task": "Assign only entity_type.",
    }


def _case(
    name: str, description: str, expected: EntityType, *, ontology_text: str | None = None
) -> Case[EntityTypeDecisionInput, EntityTypeDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=EntityTypeDecisionInput(payload=_payload(name, description, ontology_text=ontology_text)),
        expected_output=EntityTypeDecisionExpectedOutput(entity_type=expected),
    )


CASES: list[Case[EntityTypeDecisionInput, EntityTypeDecisionExpectedOutput]] = [
    _case("arterial aneurysm", "Dilation of an artery representing a disease entity.", EntityType.DIAGNOSIS),
    _case(
        "abnormal enhancement pattern", "Descriptive imaging appearance with a broad differential.", EntityType.FINDING
    ),
    _case("organ volume", "Quantitative measured organ volume.", EntityType.MEASUREMENT),
    _case("structured severity scale", "Named grading scale used to categorize severity.", EntityType.ASSESSMENT),
    _case("follow-up recommendation", "Recommendation for additional imaging follow-up.", EntityType.RECOMMENDATION),
    _case("motion artifact", "Image degradation from patient motion.", EntityType.TECHNIQUE_ISSUE),
    _case("finding group", "Container concept grouping related imaging observations.", EntityType.GROUPING),
]


class EntityTypeEvaluator(
    Evaluator[EntityTypeDecisionInput, EntityTypeDecisionActualOutput, EntityTypeDecisionExpectedOutput]
):
    """Score entity-type agreement with limited partial credit."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            EntityTypeDecisionInput, EntityTypeDecisionActualOutput, EntityTypeDecisionExpectedOutput
        ],
    ) -> float:
        if ctx.output.error is not None:
            return 0.0
        return score_entity_type(ctx.output.entity_type, ctx.expected_output.entity_type)


async def run_entity_type_decision_task(case_input: EntityTypeDecisionInput) -> EntityTypeDecisionActualOutput:
    """Run only the entity-type decision agent for one replay payload."""

    agent = create_entity_type_agent()
    try:
        result = await agent.run(json.dumps(case_input.payload, indent=2))
    except Exception as exc:
        return EntityTypeDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")
    return EntityTypeDecisionActualOutput(entity_type=result.output.entity_type, raw_output=result.output)


metadata_entity_type_decision_dataset: Dataset[
    EntityTypeDecisionInput, EntityTypeDecisionActualOutput, EntityTypeDecisionExpectedOutput
] = Dataset(cases=CASES, evaluators=[EntityTypeEvaluator()])


async def run_metadata_entity_type_decision_evals() -> EvaluationReport[
    EntityTypeDecisionInput, EntityTypeDecisionActualOutput, EntityTypeDecisionExpectedOutput
]:
    """Run the focused entity-type replay suite."""

    return await metadata_entity_type_decision_dataset.evaluate(
        run_entity_type_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )


if __name__ == "__main__":
    try:
        print("\nRunning focused entity-type decision evaluation suite...")
        print("=" * 80)
        report = asyncio.run(run_metadata_entity_type_decision_evals())
        report.print(include_input=False, include_expected_output=False, include_durations=True)
        print_weighted_summary(report, EVALUATOR_WEIGHTS, title="Entity type decision")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
