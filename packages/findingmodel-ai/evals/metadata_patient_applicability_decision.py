"""Focused evals for the metadata patient-applicability decision sub-agent."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from findingmodel import AgeProfile, AgeStage, SexSpecificity
from findingmodel_ai.metadata.assignment import create_patient_applicability_agent
from findingmodel_ai.metadata.decisions import PatientApplicabilityDecision
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from evals.metadata_scoring import WeightedComponent, print_weighted_summary, score_optional_field_match, weighted_score

EVAL_MAX_CONCURRENCY = 2
EVALUATOR_WEIGHTS: dict[str, float] = {"PatientApplicabilityEvaluator": 1.0}


class PatientApplicabilityInput(BaseModel):
    """Input payload for one patient-applicability replay case."""

    payload: dict[str, Any]


class PatientApplicabilityExpectedOutput(BaseModel):
    """Expected age/sex applicability for one replay case."""

    age_profile: AgeProfile | None = None
    sex_specificity: SexSpecificity | None = None


class PatientApplicabilityActualOutput(BaseModel):
    """Observed patient-applicability output for one replay case."""

    age_profile: AgeProfile | None = None
    sex_specificity: SexSpecificity | None = None
    raw_output: PatientApplicabilityDecision | None = None
    error: str | None = None


def _payload(
    name: str,
    description: str,
    *,
    anatomy: list[str] | None = None,
    tags: list[str] | None = None,
    existing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "assignment_mode": "reassess",
        "finding_model": {
            "oifm_id": f"EVAL_{name.upper().replace(' ', '_')}",
            "name": name,
            "description": description,
            "synonyms": [],
            "tags": tags or [],
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
            "attributes": [],
        },
        "ontology_candidates": [],
        "anatomic_candidates": [
            {
                "candidate_id": f"ANATOMICLOCATIONS:EVAL_{index}",
                "text": label,
                "display": label,
                "source_bucket": "direct_source",
                "support_level": "direct_source",
                "default_selected": True,
            }
            for index, label in enumerate(anatomy or [], start=1)
        ],
        "task": "Assign only age_profile and sex_specificity.",
    }


def _case(
    name: str,
    description: str,
    *,
    expected_age: AgeProfile | None = None,
    expected_sex: SexSpecificity | None = None,
    anatomy: list[str] | None = None,
    tags: list[str] | None = None,
    existing_metadata: dict[str, Any] | None = None,
) -> Case[PatientApplicabilityInput, PatientApplicabilityExpectedOutput]:
    return Case(
        name=name,
        inputs=PatientApplicabilityInput(
            payload=_payload(
                name,
                description,
                anatomy=anatomy,
                tags=tags,
                existing_metadata=existing_metadata,
            )
        ),
        expected_output=PatientApplicabilityExpectedOutput(
            age_profile=expected_age,
            sex_specificity=expected_sex,
        ),
    )


CASES: list[Case[PatientApplicabilityInput, PatientApplicabilityExpectedOutput]] = [
    _case(
        "prostate_anatomy_supports_male_specific",
        "Abnormal lesion centered in the prostate gland.",
        anatomy=["prostate"],
        expected_sex=SexSpecificity.MALE_SPECIFIC,
    ),
    _case(
        "ovarian_anatomy_supports_female_specific",
        "Abnormal ovarian vascular compromise.",
        anatomy=["ovary"],
        expected_sex=SexSpecificity.FEMALE_SPECIFIC,
    ),
    _case(
        "breast_context_remains_sex_neutral",
        "Calcifications seen in breast tissue on mammography.",
        anatomy=["breast"],
        expected_sex=SexSpecificity.SEX_NEUTRAL,
    ),
    _case(
        "pediatric_identity_supports_child_age",
        "An inflammatory abdominal condition of childhood.",
        tags=["pediatric"],
        expected_age=AgeProfile(applicability=[AgeStage.CHILD, AgeStage.ADOLESCENT]),
        expected_sex=SexSpecificity.SEX_NEUTRAL,
    ),
    _case(
        "unsupported_existing_age_is_cleared",
        "Generic pulmonary opacity without intrinsic age limitation.",
        anatomy=["lung"],
        existing_metadata={
            "body_regions": ["chest"],
            "subspecialties": None,
            "etiologies": None,
            "entity_type": None,
            "applicable_modalities": None,
            "expected_time_course": None,
            "age_profile": {"applicability": ["aged"], "more_common_in": ["aged"]},
            "sex_specificity": "female-specific",
            "index_codes": [],
            "anatomic_locations": [],
        },
        expected_age=None,
        expected_sex=SexSpecificity.SEX_NEUTRAL,
    ),
]


class PatientApplicabilityEvaluator(
    Evaluator[PatientApplicabilityInput, PatientApplicabilityActualOutput, PatientApplicabilityExpectedOutput]
):
    """Score age and sex applicability with separate weighted components."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            PatientApplicabilityInput, PatientApplicabilityActualOutput, PatientApplicabilityExpectedOutput
        ],
    ) -> float:
        if ctx.output.error is not None:
            return 0.0
        return weighted_score([
            WeightedComponent(
                "age_profile",
                score_optional_field_match(ctx.output.age_profile, ctx.expected_output.age_profile),
                0.40,
            ),
            WeightedComponent(
                "sex_specificity",
                score_optional_field_match(ctx.output.sex_specificity, ctx.expected_output.sex_specificity),
                0.60,
            ),
        ])


async def run_patient_applicability_task(
    case_input: PatientApplicabilityInput,
) -> PatientApplicabilityActualOutput:
    """Run only the patient-applicability decision agent for one replay payload."""

    agent = create_patient_applicability_agent()
    try:
        result = await agent.run(json.dumps(case_input.payload, indent=2))
    except Exception as exc:  # pragma: no cover - useful in eval reports
        return PatientApplicabilityActualOutput(error=f"{type(exc).__name__}: {exc}")
    return PatientApplicabilityActualOutput(
        age_profile=result.output.age_profile,
        sex_specificity=result.output.sex_specificity,
        raw_output=result.output,
    )


metadata_patient_applicability_dataset: Dataset[
    PatientApplicabilityInput,
    PatientApplicabilityActualOutput,
    PatientApplicabilityExpectedOutput,
] = Dataset(cases=CASES, evaluators=[PatientApplicabilityEvaluator()])


async def run_metadata_patient_applicability_evals() -> EvaluationReport[
    PatientApplicabilityInput,
    PatientApplicabilityActualOutput,
    PatientApplicabilityExpectedOutput,
]:
    """Run the focused patient-applicability replay suite."""

    return await metadata_patient_applicability_dataset.evaluate(
        run_patient_applicability_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )


if __name__ == "__main__":
    try:
        print("\nRunning focused patient-applicability decision evaluation suite...")
        print("=" * 80)
        report = asyncio.run(run_metadata_patient_applicability_evals())
        report.print(include_input=False, include_expected_output=False, include_durations=True)
        print_weighted_summary(report, EVALUATOR_WEIGHTS, title="Patient applicability decision")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
