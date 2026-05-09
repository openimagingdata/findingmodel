"""Focused evals for the metadata imaging-workflow decision sub-agent."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from findingmodel import BodyRegion, Modality, Subspecialty
from findingmodel_ai.metadata.assignment import create_imaging_workflow_agent
from findingmodel_ai.metadata.decisions import ImagingWorkflowDecision
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

EVAL_MAX_CONCURRENCY = 2


class ImagingWorkflowDecisionInput(BaseModel):
    """Input payload for one imaging-workflow replay case."""

    payload: dict[str, Any]


class ImagingWorkflowDecisionExpectedOutput(BaseModel):
    """Expected imaging-workflow output for one replay case."""

    required_subspecialties: set[Subspecialty] = Field(default_factory=set)
    forbidden_subspecialties: set[Subspecialty] = Field(default_factory=set)
    allowed_subspecialties: set[Subspecialty] | None = None
    required_modalities: set[Modality] = Field(default_factory=set)
    forbidden_modalities: set[Modality] = Field(default_factory=set)
    allowed_modalities: set[Modality] | None = None


class ImagingWorkflowDecisionActualOutput(BaseModel):
    """Observed imaging-workflow output for one replay case."""

    subspecialties: set[Subspecialty] = Field(default_factory=set)
    applicable_modalities: set[Modality] = Field(default_factory=set)
    raw_output: ImagingWorkflowDecision | None = None
    error: str | None = None


def _ontology_candidate(
    candidate_id: str,
    text: str,
    *,
    selected: bool = True,
    source_bucket: str = "existing_index_codes",
) -> dict[str, Any]:
    system, _, _code = candidate_id.partition(":")
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "table_name": system.lower(),
        "system": system,
        "source_bucket": source_bucket,
        "default_relationship": "exact_match" if selected else "related",
        "default_selected_as_canonical": selected,
    }


def _anatomic_candidate(
    candidate_id: str,
    text: str,
    *,
    selected: bool = True,
    support_level: str = "direct_source",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "source_bucket": "candidate",
        "support_level": support_level,
        "matched_terms": [text] if support_level == "direct_source" else [],
        "broader_candidate_ids": [],
        "default_selected": selected,
    }


def _payload(
    *,
    name: str,
    description: str,
    tags: list[str],
    ontology: list[dict[str, Any]],
    anatomy: list[dict[str, Any]],
    anatomy_body_regions: list[BodyRegion],
    synonyms: list[str] | None = None,
    attributes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "assignment_mode": "reassess",
        "finding_model": {
            "oifm_id": f"EVAL_{name.upper().replace(' ', '_')}",
            "name": name,
            "description": description,
            "synonyms": synonyms or [],
            "tags": tags,
            "existing_structured_metadata": {
                "body_regions": [region.value for region in anatomy_body_regions],
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
            "attributes": attributes or [],
        },
        "ontology_candidates": ontology,
        "anatomic_candidates": anatomy,
        "task": "Assign only subspecialties and applicable_modalities.",
        "workflow_values_under_review": {
            "subspecialties": None,
            "applicable_modalities": None,
        },
        "anatomy_context": {
            "body_regions": [region.value for region in anatomy_body_regions],
        },
    }


def _case(
    name: str,
    *,
    payload: dict[str, Any],
    required_subspecialties: set[Subspecialty],
    required_modalities: set[Modality],
    forbidden_subspecialties: set[Subspecialty] | None = None,
    forbidden_modalities: set[Modality] | None = None,
    allowed_subspecialties: set[Subspecialty] | None = None,
    allowed_modalities: set[Modality] | None = None,
) -> Case[ImagingWorkflowDecisionInput, ImagingWorkflowDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=ImagingWorkflowDecisionInput(payload=payload),
        expected_output=ImagingWorkflowDecisionExpectedOutput(
            required_subspecialties=required_subspecialties,
            forbidden_subspecialties=forbidden_subspecialties or set(),
            allowed_subspecialties=allowed_subspecialties,
            required_modalities=required_modalities,
            forbidden_modalities=forbidden_modalities or set(),
            allowed_modalities=allowed_modalities,
        ),
    )


CASES: list[Case[ImagingWorkflowDecisionInput, ImagingWorkflowDecisionExpectedOutput]] = [
    _case(
        "ordinary_presence_attributes_do_not_support_quality_workflow",
        payload=_payload(
            name="antegonial notching of the mandible",
            description="Indentation near the anterior angle of the mandible.",
            tags=["head_neck", "CT", "XR", "jaw", "congenital anomaly", "finding"],
            ontology=[_ontology_candidate("GAMUTS:25564", "antegonial notching of the mandible")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID9082", "mandible")],
            anatomy_body_regions=[BodyRegion.HEAD],
            attributes=[
                {
                    "name": "presence",
                    "description": "Presence or absence of antegonial notching of the mandible",
                    "required": False,
                    "type": "choice",
                    "values": ["absent", "present", "indeterminate", "unknown"],
                },
                {
                    "name": "change from prior",
                    "description": "Whether and how an antegonial notching of the mandible changed",
                    "required": False,
                    "type": "choice",
                    "values": ["unchanged", "stable", "new", "resolved"],
                },
            ],
        ),
        required_subspecialties={Subspecialty.HN},
        forbidden_subspecialties={Subspecialty.SQ, Subspecialty.MK},
        allowed_subspecialties={Subspecialty.HN},
        required_modalities={Modality.XR},
        forbidden_modalities={Modality.MG, Modality.MR, Modality.PET, Modality.NM, Modality.US, Modality.RF, Modality.DSA},
        allowed_modalities={Modality.CT, Modality.XR},
    ),
    _case(
        "indirect_modalities_do_not_make_pulmonary_embolism_xr_or_us",
        payload=_payload(
            name="pulmonary embolism",
            description=(
                "Blockage of an artery in the lungs by a substance that has moved from elsewhere "
                "in the body through the bloodstream."
            ),
            synonyms=["pulmonary artery thromboembolism", "PE", "pulmonary thromboembolism"],
            tags=["vascular", "CT", "XR", "US", "pulmonary artery", "embolism", "diagnosis"],
            ontology=[
                _ontology_candidate("GAMUTS:22548", "pulmonary embolism"),
                _ontology_candidate("SNOMEDCT:59282003", "Pulmonary embolism", source_bucket="exact_matches"),
            ],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID974", "pulmonary artery")],
            anatomy_body_regions=[BodyRegion.CHEST],
        ),
        required_subspecialties={Subspecialty.VA},
        forbidden_subspecialties={Subspecialty.CH, Subspecialty.ER},
        allowed_subspecialties={Subspecialty.VA},
        required_modalities={Modality.CT},
        forbidden_modalities={Modality.XR, Modality.US, Modality.MR, Modality.NM, Modality.DSA},
        allowed_modalities={Modality.CT},
    ),
    _case(
        "regional_soft_tissue_mass_does_not_inherit_nearby_organ_workflow_or_xray",
        payload=_payload(
            name="axillary mass",
            description="Abnormal soft tissue mass located in the axillary region.",
            tags=["chest", "XR", "CT", "MR", "US", "chest wall", "lymphatic", "finding"],
            ontology=[
                _ontology_candidate("RADLEX:RID35025", "axillary mass sign"),
                _ontology_candidate("GAMUTS:16069", "axillary mass"),
            ],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1850", "upper extremity")],
            anatomy_body_regions=[BodyRegion.UPPER_EXTREMITY],
        ),
        required_subspecialties=set(),
        forbidden_subspecialties={Subspecialty.BR, Subspecialty.CH, Subspecialty.MK},
        allowed_subspecialties=set(),
        required_modalities={Modality.US},
        forbidden_modalities={Modality.XR, Modality.MG, Modality.NM, Modality.PET, Modality.DSA, Modality.RF},
        allowed_modalities={Modality.US, Modality.CT, Modality.MR},
    ),
]


class ImagingWorkflowEvaluator(Evaluator[ImagingWorkflowDecisionInput, ImagingWorkflowDecisionActualOutput]):
    """Score exact workflow and modality agreement."""

    def evaluate(
        self,
        ctx: EvaluatorContext[ImagingWorkflowDecisionInput, ImagingWorkflowDecisionActualOutput],
    ) -> float:
        if ctx.output.error is not None:
            return 0.0
        expected = ctx.expected_output
        subspecialties = ctx.output.subspecialties
        modalities = ctx.output.applicable_modalities
        if not expected.required_subspecialties.issubset(subspecialties):
            return 0.0
        if expected.forbidden_subspecialties.intersection(subspecialties):
            return 0.0
        if expected.allowed_subspecialties is not None and not subspecialties.issubset(expected.allowed_subspecialties):
            return 0.0
        if not expected.required_modalities.issubset(modalities):
            return 0.0
        if expected.forbidden_modalities.intersection(modalities):
            return 0.0
        if expected.allowed_modalities is not None and not modalities.issubset(expected.allowed_modalities):
            return 0.0
        return 1.0


async def run_imaging_workflow_decision_task(
    case_input: ImagingWorkflowDecisionInput,
) -> ImagingWorkflowDecisionActualOutput:
    """Run only the imaging-workflow decision agent for one replay payload."""

    agent = create_imaging_workflow_agent()
    payload = copy.deepcopy(case_input.payload)
    payload["finding_model"]["tags"] = []
    payload["ontology_candidates"] = [
        candidate for candidate in payload["ontology_candidates"] if candidate.get("default_selected_as_canonical") is True
    ]
    payload["anatomic_candidates"] = [
        candidate for candidate in payload["anatomic_candidates"] if candidate.get("default_selected") is True
    ]
    try:
        result = await agent.run(json.dumps(payload, indent=2))
    except Exception as exc:  # pragma: no cover - useful in eval reports
        return ImagingWorkflowDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")

    return ImagingWorkflowDecisionActualOutput(
        subspecialties=set(result.output.subspecialties or []),
        applicable_modalities=set(result.output.applicable_modalities or []),
        raw_output=result.output,
    )


metadata_imaging_workflow_decision_dataset: Dataset[
    ImagingWorkflowDecisionInput, ImagingWorkflowDecisionActualOutput, ImagingWorkflowDecisionExpectedOutput
] = Dataset(
    cases=CASES,
    evaluators=[ImagingWorkflowEvaluator()],
)


async def run_metadata_imaging_workflow_decision_evals() -> EvaluationReport[
    ImagingWorkflowDecisionInput, ImagingWorkflowDecisionActualOutput, ImagingWorkflowDecisionExpectedOutput
]:
    """Run the focused imaging-workflow replay suite."""

    return await metadata_imaging_workflow_decision_dataset.evaluate(
        run_imaging_workflow_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )


if __name__ == "__main__":
    try:
        print("\nRunning focused imaging-workflow decision evaluation suite...")
        print("=" * 80)
        report = asyncio.run(run_metadata_imaging_workflow_decision_evals())
        report.print(include_input=False, include_expected_output=False, include_durations=True)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
