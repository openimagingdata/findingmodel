"""Focused evals for the metadata anatomy decision sub-agent.

These cases replay only the anatomy-agent payload shape. They do not run ontology
search, anatomy search, assembly, audit, or the other focused metadata agents.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from findingmodel import BodyRegion
from findingmodel_ai.metadata.assignment import create_anatomy_decision_agent
from findingmodel_ai.metadata.decisions import AnatomyDecision
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

EVAL_MAX_CONCURRENCY = 2


class AnatomyDecisionInput(BaseModel):
    """Input payload for one anatomy-agent replay case."""

    payload: dict[str, Any]


class AnatomyDecisionExpectedOutput(BaseModel):
    """Expected anatomy-agent behavior for one replay case."""

    selected_candidate_ids: list[str] = Field(default_factory=list)
    body_regions: list[BodyRegion] = Field(default_factory=list)


class AnatomyDecisionActualOutput(BaseModel):
    """Observed anatomy-agent output for one replay case."""

    selected_candidate_ids: list[str] = Field(default_factory=list)
    body_regions: list[BodyRegion] | None = None
    raw_output: AnatomyDecision | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def _candidate(
    candidate_id: str,
    text: str,
    support_level: str,
    *,
    default_selected: bool = False,
    broader_candidate_ids: list[str] | None = None,
    source_bucket: str = "candidate",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "source_bucket": source_bucket,
        "support_level": support_level,
        "matched_terms": [text] if support_level in {"direct_source", "source_inferred_query"} else [],
        "broader_candidate_ids": broader_candidate_ids or [],
        "default_selected": default_selected,
    }


def _payload(
    *,
    name: str,
    description: str,
    attributes: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    existing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "assignment_mode": "reassess",
        "finding_model": {
            "oifm_id": f"EVAL_{name.upper().replace(' ', '_')}",
            "name": name,
            "description": description,
            "synonyms": [],
            "tags": [],
            "existing_structured_metadata": existing_metadata
            or {
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
            "attributes": attributes,
        },
        "ontology_candidates": [],
        "anatomic_candidates": candidates,
        "task": "Decide anatomic candidate selection and body_regions.",
    }


def _case(
    name: str,
    *,
    payload: dict[str, Any],
    selected_candidate_ids: list[str],
    body_regions: list[BodyRegion],
) -> Case[AnatomyDecisionInput, AnatomyDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=AnatomyDecisionInput(payload=payload),
        expected_output=AnatomyDecisionExpectedOutput(
            selected_candidate_ids=selected_candidate_ids,
            body_regions=body_regions,
        ),
    )


CASES: list[Case[AnatomyDecisionInput, AnatomyDecisionExpectedOutput]] = [
    _case(
        "arterial_system_parent_over_attribute_options",
        payload=_payload(
            name="Arterial Stent",
            description="An arterial stent is a tubular device inserted into an artery to improve blood flow.",
            attributes=[
                {
                    "name": "Location",
                    "description": "Indicates the location of the arterial stent.",
                    "required": True,
                    "type": "choice",
                    "values": ["Carotid artery", "Subclavian artery", "Brachial artery", "Cephalic artery"],
                },
                {
                    "name": "Presence of Arterial Stent",
                    "description": "Indicates whether an arterial stent is present.",
                    "required": True,
                    "type": "choice",
                    "values": ["Present", "Absent"],
                },
            ],
            candidates=[
                _candidate(
                    "ANATOMICLOCATIONS:RID13183",
                    "arterial system",
                    "source_inferred_query",
                    broader_candidate_ids=["RID39569"],
                ),
                _candidate(
                    "ANATOMICLOCATIONS:RID584",
                    "common carotid artery",
                    "source_inferred_query",
                    broader_candidate_ids=["RID7488", "RID39569"],
                ),
                _candidate(
                    "ANATOMICLOCATIONS:RID773",
                    "subclavian artery",
                    "source_inferred_query",
                    broader_candidate_ids=["RID1243", "RID39569"],
                ),
                _candidate(
                    "ANATOMICLOCATIONS:RID865",
                    "brachial artery",
                    "source_inferred_query",
                    broader_candidate_ids=["RID1968", "RID1850", "RID39569"],
                ),
                _candidate("ANATOMICLOCATIONS:RID39569", "whole body", "parent_of_supported"),
            ],
        ),
        selected_candidate_ids=["ANATOMICLOCATIONS:RID13183"],
        body_regions=[BodyRegion.WHOLE_BODY],
    ),
    _case(
        "breast_scope_without_unqualified_breast_candidate",
        payload=_payload(
            name="Breast density",
            description="Breast density refers to the proportion of fatty tissue to fibroglandular tissue in the breast as seen on a mammogram.",
            attributes=[
                {
                    "name": "density category",
                    "description": "Categorical classification of breast density based on mammographic appearance.",
                    "required": False,
                    "type": "choice",
                    "values": [
                        "a: The breasts are almost entirely fatty.",
                        "b: There are scattered areas of fibroglandular density.",
                        "c: The breasts are heterogeneously dense.",
                        "d: The breasts are extremely dense.",
                    ],
                }
            ],
            candidates=[
                _candidate("ANATOMICLOCATIONS:RID34263", "Intramammary lymph node", "search_only"),
                _candidate("ANATOMICLOCATIONS:RID29955", "accessory breast", "search_only"),
                _candidate("ANATOMICLOCATIONS:RID29917", "areola of female breast", "search_only"),
                _candidate("ANATOMICLOCATIONS:RID29914", "areola of male breast", "search_only"),
                _candidate("ANATOMICLOCATIONS:RID29949", "central region of breast", "search_only"),
            ],
        ),
        selected_candidate_ids=[],
        body_regions=[BodyRegion.BREAST],
    ),
    _case(
        "urinary_tract_parent_covers_current_children",
        payload=_payload(
            name="radiolucent urinary calculus",
            description="A kidney stone that does not appear on standard radiography but may be seen on ultrasound or CT.",
            attributes=[],
            candidates=[
                _candidate("ANATOMICLOCATIONS:RID225", "calyx of renal collecting system", "current_metadata", default_selected=True),
                _candidate("ANATOMICLOCATIONS:RID205", "kidney", "current_metadata", default_selected=True),
                _candidate("ANATOMICLOCATIONS:RID228", "renal pelvis", "current_metadata", default_selected=True),
                _candidate("ANATOMICLOCATIONS:RID229", "ureter", "current_metadata", default_selected=True),
                _candidate("ANATOMICLOCATIONS:RID204", "urinary tract", "direct_source"),
                _candidate("ANATOMICLOCATIONS:RID56", "abdomen", "parent_of_supported"),
            ],
        ),
        selected_candidate_ids=["ANATOMICLOCATIONS:RID204"],
        body_regions=[BodyRegion.ABDOMEN],
    ),
    _case(
        "tunneled_catheter_supported_course_location",
        payload=_payload(
            name="tunneled catheter",
            description="Central venous catheter with a subcutaneous tunnel, visible as a radiopaque line with cuff.",
            attributes=[],
            candidates=[
                _candidate(
                    "ANATOMICLOCATIONS:RID1243",
                    "thorax",
                    "parent_of_supported",
                    default_selected=True,
                    source_bucket="existing_anatomic_locations",
                ),
                _candidate(
                    "ANATOMICLOCATIONS:RID29859",
                    "anterior chest wall",
                    "source_inferred_query",
                    broader_candidate_ids=["RID2468", "RID1243", "RID39569"],
                ),
                _candidate(
                    "ANATOMICLOCATIONS:RID2468",
                    "chest wall",
                    "source_inferred_query",
                    broader_candidate_ids=["RID1243", "RID39569"],
                ),
                _candidate("ANATOMICLOCATIONS:RID39569", "whole body", "parent_of_supported"),
                _candidate(
                    "ANATOMICLOCATIONS:RID30049",
                    "back of thorax",
                    "child_of_supported",
                    broader_candidate_ids=["RID2468"],
                ),
            ],
        ),
        selected_candidate_ids=["ANATOMICLOCATIONS:RID29859"],
        body_regions=[BodyRegion.CHEST],
    ),
    _case(
        "generic_axillary_mass_does_not_narrow_to_lymph_node",
        payload=_payload(
            name="axillary mass",
            description="Abnormal soft tissue mass located in the axillary (armpit) region.",
            attributes=[],
            candidates=[
                _candidate(
                    "ANATOMICLOCATIONS:RID1850",
                    "upper extremity",
                    "current_metadata",
                    default_selected=True,
                    source_bucket="existing_anatomic_locations",
                ),
                _candidate(
                    "ANATOMICLOCATIONS:RID1517",
                    "axillary lymph node",
                    "source_inferred_query",
                    broader_candidate_ids=["RID1243", "RID39569"],
                ),
                _candidate("ANATOMICLOCATIONS:RID1243", "thorax", "parent_of_supported"),
                _candidate("ANATOMICLOCATIONS:RID39569", "whole body", "parent_of_supported"),
                _candidate("ANATOMICLOCATIONS:RID1014", "axillary vein", "search_only"),
                _candidate("ANATOMICLOCATIONS:RID860", "axillary artery", "search_only"),
            ],
        ),
        selected_candidate_ids=["ANATOMICLOCATIONS:RID1850"],
        body_regions=[BodyRegion.UPPER_EXTREMITY],
    ),
]


class SelectedCandidateEvaluator(Evaluator[AnatomyDecisionInput, AnatomyDecisionActualOutput]):
    """Score exact selected-candidate agreement."""

    def evaluate(self, ctx: EvaluatorContext[AnatomyDecisionInput, AnatomyDecisionActualOutput]) -> float:
        if ctx.expected_output is None or ctx.output.error:
            return 0.0
        return float(set(ctx.output.selected_candidate_ids) == set(ctx.expected_output.selected_candidate_ids))


class BodyRegionEvaluator(Evaluator[AnatomyDecisionInput, AnatomyDecisionActualOutput]):
    """Score exact body-region agreement."""

    def evaluate(self, ctx: EvaluatorContext[AnatomyDecisionInput, AnatomyDecisionActualOutput]) -> float:
        if ctx.expected_output is None or ctx.output.error:
            return 0.0
        actual = set(ctx.output.body_regions or [])
        expected = set(ctx.expected_output.body_regions)
        return float(actual == expected)


async def run_anatomy_decision_task(case_input: AnatomyDecisionInput) -> AnatomyDecisionActualOutput:
    """Run only the anatomy decision agent for one replay payload."""

    candidate_ids = {candidate["candidate_id"] for candidate in case_input.payload.get("anatomic_candidates", [])}
    agent = create_anatomy_decision_agent()

    @agent.output_validator
    def validate_candidate_ids(ctx: RunContext[None], output: AnatomyDecision) -> AnatomyDecision:
        _ = ctx
        output.anatomic_decisions = [
            decision for decision in output.anatomic_decisions if decision.candidate_id in candidate_ids
        ]
        return output

    try:
        result = await agent.run(json.dumps(case_input.payload, indent=2))
    except Exception as exc:
        return AnatomyDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")

    selected = [
        decision.candidate_id
        for decision in result.output.anatomic_decisions
        if decision.selected and decision.candidate_id in candidate_ids
    ]
    usage = result.usage().model_dump(mode="json") if hasattr(result.usage(), "model_dump") else {}
    return AnatomyDecisionActualOutput(
        selected_candidate_ids=selected,
        body_regions=result.output.body_regions,
        raw_output=result.output,
        usage=usage,
    )


metadata_anatomy_decision_dataset: Dataset[
    AnatomyDecisionInput, AnatomyDecisionActualOutput, AnatomyDecisionExpectedOutput
] = Dataset(
    cases=CASES,
    evaluators=[
        SelectedCandidateEvaluator(),
        BodyRegionEvaluator(),
    ],
)


async def run_metadata_anatomy_decision_evals() -> EvaluationReport[
    AnatomyDecisionInput, AnatomyDecisionActualOutput, AnatomyDecisionExpectedOutput
]:
    """Run the focused anatomy-decision replay suite."""

    return await metadata_anatomy_decision_dataset.evaluate(
        run_anatomy_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
        progress=False,
    )


if __name__ == "__main__":
    from evals import ensure_instrumented

    ensure_instrumented()

    async def main() -> None:
        print("\nRunning focused anatomy decision evaluation suite...")
        print("=" * 80)
        report = await run_metadata_anatomy_decision_evals()
        print("\n" + "=" * 80)
        print("FOCUSED ANATOMY DECISION RESULTS")
        print("=" * 80 + "\n")
        report.print(include_input=False, include_output=True, include_durations=True, width=120)

    asyncio.run(main())
