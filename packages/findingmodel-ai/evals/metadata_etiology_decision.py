"""Focused evals for the metadata etiology decision sub-agent.

These cases replay only the etiology-agent payload shape. They do not run ontology
search, anatomy search, assembly, audit, or the other focused metadata agents.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from findingmodel import EntityType, EtiologyCode
from findingmodel_ai.metadata.assignment import create_etiology_assignment_agent
from findingmodel_ai.metadata.decisions import EtiologyDecision
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

EVAL_MAX_CONCURRENCY = 2


class EtiologyDecisionInput(BaseModel):
    """Input payload for one etiology-agent replay case."""

    payload: dict[str, Any]


class EtiologyDecisionExpectedOutput(BaseModel):
    """Expected etiology-agent behavior for one replay case."""

    etiologies: list[EtiologyCode] | None = None


class EtiologyDecisionActualOutput(BaseModel):
    """Observed etiology-agent output for one replay case."""

    etiologies: list[EtiologyCode] | None = None
    raw_output: EtiologyDecision | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def _ontology_candidate(
    candidate_id: str,
    text: str,
    *,
    selected: bool = True,
    source_bucket: str = "exact_matches",
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


def _anatomic_candidate(candidate_id: str, text: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "source_bucket": "candidate",
        "support_level": "direct_source",
        "matched_terms": [text],
        "broader_candidate_ids": [],
        "default_selected": True,
    }


def _payload(
    *,
    name: str,
    description: str,
    entity_type: EntityType,
    ontology: list[dict[str, Any]],
    anatomy: list[dict[str, Any]],
    synonyms: list[str] | None = None,
    tags: list[str] | None = None,
    attributes: list[dict[str, Any]] | None = None,
    existing_etiologies: list[str] | None = None,
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
                "index_codes": [],
                "anatomic_locations": [],
            },
            "attributes": attributes or [],
        },
        "ontology_candidates": [
            candidate
            for candidate in ontology
            if candidate.get("default_selected_as_canonical") is True
            and candidate.get("source_bucket") in {"existing_index_codes", "exact_matches"}
        ],
        "anatomic_candidates": [],
        "task": "Assign only etiologies.",
        "identity_context": {
            "entity_type": entity_type.value,
            "expected_time_course": None,
        },
    }


def _case(
    name: str,
    *,
    payload: dict[str, Any],
    etiologies: list[EtiologyCode] | None,
) -> Case[EtiologyDecisionInput, EtiologyDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=EtiologyDecisionInput(payload=payload),
        expected_output=EtiologyDecisionExpectedOutput(etiologies=etiologies),
    )


CASES: list[Case[EtiologyDecisionInput, EtiologyDecisionExpectedOutput]] = [
    _case(
        "generic_axillary_mass_stays_null",
        payload=_payload(
            name="axillary mass",
            description="Abnormal soft tissue mass located in the axillary region.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("GAMUTS:16069", "axillary mass")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID199", "axilla")],
            synonyms=["axilla mass", "axillary swelling", "axillary lymphadenopathy (if lymph node-related)"],
            existing_etiologies=["inflammatory", "neoplastic:benign", "neoplastic:malignant"],
        ),
        etiologies=None,
    ),
    _case(
        "description_differential_does_not_create_etiology",
        payload=_payload(
            name="widening of rib interspaces",
            description="Increased distance between adjacent ribs, suggesting hyperinflation, mass effect, or prior surgery.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("OIDM:594364", "widening of rib interspaces")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1243", "thorax")],
        ),
        etiologies=None,
    ),
    _case(
        "fluid_collection_with_many_causes_stays_null",
        payload=_payload(
            name="pericardial effusion",
            description="Accumulation of fluid in the pericardial cavity.",
            entity_type=EntityType.FINDING,
            ontology=[
                _ontology_candidate("SNOMEDCT:373945007", "Pericardial effusion"),
                _ontology_candidate("SNOMEDCT:1269348003", "Malignant pericardial effusion", selected=False),
                _ontology_candidate("SNOMEDCT:1230405007", "Postoperative pericardial effusion", selected=False),
                _ontology_candidate(
                    "SNOMEDCT:460445000",
                    "Viral pericarditis with pericardial effusion",
                    selected=False,
                ),
            ],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1248", "pericardium")],
            existing_etiologies=["inflammatory", "neoplastic:malignant", "iatrogenic:post-operative"],
        ),
        etiologies=None,
    ),
    _case(
        "morphology_with_only_legacy_tag_stays_null",
        payload=_payload(
            name="antegonial notching of the mandible",
            description="Indentation near the anterior angle of the mandible.",
            entity_type=EntityType.FINDING,
            ontology=[
                _ontology_candidate("GAMUTS:25564", "antegonial notching of the mandible"),
                _ontology_candidate("SNOMEDCT:708685001", "Mandibular plane angle", selected=False),
                _ontology_candidate("SNOMEDCT:710243000", "Flat mandibular plane angle", selected=False),
            ],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID9113", "mandible")],
            tags=["head_neck", "CT", "XR", "jaw", "congenital anomaly"],
        ),
        etiologies=None,
    ),
    _case(
        "urinary_calculus_is_metabolic",
        payload=_payload(
            name="radiolucent urinary calculus",
            description="A kidney stone that does not appear on standard radiography but may be seen on ultrasound or CT.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:304543009", "Radiolucent calculus of urinary tract")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID204", "urinary tract")],
            synonyms=["radiolucent kidney stone"],
        ),
        etiologies=[EtiologyCode.METABOLIC],
    ),
    _case(
        "calcification_cluster_stays_null",
        payload=_payload(
            name="breast calcification cluster",
            description=(
                "Breast calcification clusters are typically a sign of benign changes in breast "
                "tissue but can sometimes indicate malignancy."
            ),
            entity_type=EntityType.FINDING,
            ontology=[
                _ontology_candidate("SNOMEDCT:129769006", "Calcification cluster", source_bucket="existing_index_codes"),
                _ontology_candidate(
                    "SNOMEDCT:697944008",
                    "Mammographic calcification of breast",
                    source_bucket="existing_index_codes",
                ),
                _ontology_candidate("SNOMEDCT:44771000", "Microcalcifications of the breast"),
            ],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28728", "breast")],
            synonyms=["breast calcifications"],
        ),
        etiologies=None,
    ),
    _case(
        "pulmonary_embolism_is_thrombotic",
        payload=_payload(
            name="pulmonary embolism",
            description="Blockage of an artery in the lungs by material that moved through the bloodstream.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:59282003", "Pulmonary embolism")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            synonyms=["pulmonary artery thromboembolism"],
        ),
        etiologies=[EtiologyCode.VASCULAR_THROMBOTIC],
    ),
    _case(
        "device_presence_is_iatrogenic_device",
        payload=_payload(
            name="aortic stent",
            description="Endovascular stent graft within the aorta, visible as metallic mesh on radiograph.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:70512008", "Aortic stent")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID480", "aorta")],
            synonyms=["endovascular aortic repair", "aortic stent graft"],
        ),
        etiologies=[EtiologyCode.IATROGENIC_DEVICE],
    ),
    _case(
        "primary_brain_tumor_is_neoplastic_without_metastatic",
        payload=_payload(
            name="primary brain tumor",
            description="Tumor originating within the brain.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:428061005", "Primary neoplasm of brain")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6434", "brain")],
            synonyms=["intracranial neoplasm"],
            existing_etiologies=["neoplastic:benign", "neoplastic:malignant", "neoplastic:metastatic"],
        ),
        etiologies=[EtiologyCode.NEOPLASTIC_BENIGN, EtiologyCode.NEOPLASTIC_MALIGNANT],
    ),
    _case(
        "pneumonia_is_infectious_not_extra_parent_bucket",
        payload=_payload(
            name="Pneumonia",
            description="Common Data Elements and macros for pneumonia.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:233604007", "Pneumonia")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            synonyms=["pneumonitis"],
            existing_etiologies=["inflammatory", "inflammatory:infectious"],
        ),
        etiologies=[EtiologyCode.INFLAMMATORY_INFECTIOUS],
    ),
]


class EtiologyEvaluator(Evaluator[EtiologyDecisionInput, EtiologyDecisionActualOutput]):
    """Score exact etiology agreement."""

    def evaluate(self, ctx: EvaluatorContext[EtiologyDecisionInput, EtiologyDecisionActualOutput]) -> float:
        if ctx.expected_output is None or ctx.output.error:
            return 0.0
        actual = set(ctx.output.etiologies or [])
        expected = set(ctx.expected_output.etiologies or [])
        return float(actual == expected)


async def run_etiology_decision_task(case_input: EtiologyDecisionInput) -> EtiologyDecisionActualOutput:
    """Run only the etiology decision agent for one replay payload."""

    agent = create_etiology_assignment_agent()
    try:
        result = await agent.run(json.dumps(case_input.payload, indent=2))
    except Exception as exc:
        return EtiologyDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")

    usage = result.usage().model_dump(mode="json") if hasattr(result.usage(), "model_dump") else {}
    return EtiologyDecisionActualOutput(
        etiologies=result.output.etiologies,
        raw_output=result.output,
        usage=usage,
    )


metadata_etiology_decision_dataset: Dataset[
    EtiologyDecisionInput, EtiologyDecisionActualOutput, EtiologyDecisionExpectedOutput
] = Dataset(
    cases=CASES,
    evaluators=[EtiologyEvaluator()],
)


async def run_metadata_etiology_decision_evals() -> EvaluationReport[
    EtiologyDecisionInput, EtiologyDecisionActualOutput, EtiologyDecisionExpectedOutput
]:
    """Run the focused etiology-decision replay suite."""

    return await metadata_etiology_decision_dataset.evaluate(
        run_etiology_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
        progress=False,
    )


if __name__ == "__main__":
    from evals import ensure_instrumented

    ensure_instrumented()

    async def main() -> None:
        print("\nRunning focused etiology decision evaluation suite...")
        print("=" * 80)
        report = await run_metadata_etiology_decision_evals()
        print("\n" + "=" * 80)
        print("FOCUSED ETIOLOGY DECISION RESULTS")
        print("=" * 80 + "\n")
        report.print(include_input=False, include_output=True, include_durations=True, width=120)

    asyncio.run(main())
