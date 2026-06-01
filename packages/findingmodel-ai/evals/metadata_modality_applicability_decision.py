"""Focused evals for the metadata modality-applicability decision sub-agent."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from collections import Counter
from typing import Any

from findingmodel import BodyRegion, Modality
from findingmodel_ai.metadata.assignment import create_modality_applicability_agent
from findingmodel_ai.metadata.decisions import ModalityApplicabilityDecision
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from evals.metadata_scoring import (
    print_weighted_summary,
    score_required_forbidden_allowed,
)

EVAL_MAX_CONCURRENCY = 2
EVALUATOR_WEIGHTS: dict[str, float] = {"ModalityApplicabilityEvaluator": 1.0}


class ModalityApplicabilityDecisionInput(BaseModel):
    """Input payload for one modality-applicability replay case."""

    payload: dict[str, Any]


class ModalityApplicabilityDecisionExpectedOutput(BaseModel):
    """Expected modality-applicability output for one replay case."""

    required_modalities: set[Modality] = Field(default_factory=set)
    forbidden_modalities: set[Modality] = Field(default_factory=set)
    allowed_modalities: set[Modality] | None = None


class ModalityApplicabilityDecisionActualOutput(BaseModel):
    """Observed modality-applicability output for one replay case."""

    applicable_modalities: set[Modality] = Field(default_factory=set)
    raw_output: ModalityApplicabilityDecision | None = None
    error: str | None = None


def _ontology_candidate(candidate_id: str, text: str, *, selected: bool = True) -> dict[str, Any]:
    system, _, _code = candidate_id.partition(":")
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "table_name": system.lower(),
        "system": system,
        "source_bucket": "existing_index_codes",
        "default_relationship": "exact_match" if selected else "related",
        "default_selected_as_canonical": selected,
    }


def _anatomic_candidate(candidate_id: str, text: str, *, selected: bool = True) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "source_bucket": "candidate",
        "support_level": "direct_source",
        "matched_terms": [text],
        "broader_candidate_ids": [],
        "default_selected": selected,
    }


def _payload(
    *,
    name: str,
    description: str,
    ontology: list[dict[str, Any]],
    anatomy: list[dict[str, Any]],
    body_regions: list[BodyRegion],
    tags: list[str] | None = None,
    synonyms: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "assignment_mode": "reassess",
        "finding_model": {
            "oifm_id": f"EVAL_{name.upper().replace(' ', '_')}",
            "name": name,
            "description": description,
            "synonyms": synonyms or [],
            "tags": tags or [],
            "existing_structured_metadata": {
                "body_regions": [region.value for region in body_regions],
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
        "anatomic_candidates": anatomy,
        "task": "Assign only applicable_modalities.",
        "modality_values_under_review": None,
        "anatomy_context": {"body_regions": [region.value for region in body_regions]},
    }


def _case(
    name: str,
    *,
    payload: dict[str, Any],
    required: set[Modality],
    forbidden: set[Modality] | None = None,
    allowed: set[Modality] | None = None,
) -> Case[ModalityApplicabilityDecisionInput, ModalityApplicabilityDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=ModalityApplicabilityDecisionInput(payload=payload),
        expected_output=ModalityApplicabilityDecisionExpectedOutput(
            required_modalities=required,
            forbidden_modalities=forbidden or set(),
            allowed_modalities=allowed,
        ),
    )


CASES: list[Case[ModalityApplicabilityDecisionInput, ModalityApplicabilityDecisionExpectedOutput]] = [
    _case(
        "pneumothorax_supports_xr_and_ct",
        payload=_payload(
            name="pneumothorax",
            description="Air in the pleural space, commonly evaluated on chest radiography and CT.",
            ontology=[_ontology_candidate("SNOMEDCT:36118008", "Pneumothorax")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1340", "pleural space")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.XR},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.XR, Modality.CT},
    ),
    _case(
        "pulmonary_embolism_supports_ct_not_xr_or_us",
        payload=_payload(
            name="pulmonary embolism",
            description="Embolic occlusion of a pulmonary artery directly evaluated by CT pulmonary angiography.",
            ontology=[_ontology_candidate("SNOMEDCT:59282003", "Pulmonary embolism")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1310", "pulmonary artery")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.CT},
        forbidden={Modality.XR, Modality.US, Modality.MG},
        allowed={Modality.CT, Modality.NM, Modality.DSA},
    ),
    _case(
        "acute_aortic_dissection_supports_ct",
        payload=_payload(
            name="acute aortic dissection",
            description="Acute dissection flap involving the thoracic aorta.",
            ontology=[_ontology_candidate("SNOMEDCT:308546005", "Aortic dissection")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID486", "thoracic aorta")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.CT},
        forbidden={Modality.XR, Modality.MG},
        allowed={Modality.CT, Modality.MR, Modality.DSA},
    ),
    _case(
        "coronary_calcium_score_supports_ct",
        payload=_payload(
            name="coronary calcified plaque burden",
            description="Quantification of calcified plaque in the coronary arteries.",
            ontology=[_ontology_candidate("RADLEX:RID49701", "coronary artery calcification")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID34755", "coronary artery")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.CT},
        forbidden={Modality.XR, Modality.US, Modality.MG},
        allowed={Modality.CT},
    ),
    _case(
        "cerebral_infarction_supports_ct_and_mr",
        payload=_payload(
            name="cerebral infarction",
            description="Acute infarction involving brain parenchyma, evaluated with CT and MRI.",
            ontology=[_ontology_candidate("SNOMEDCT:432504007", "Cerebral infarction")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6434", "brain")],
            body_regions=[BodyRegion.HEAD],
        ),
        required={Modality.CT, Modality.MR},
        forbidden={Modality.MG, Modality.RF},
        allowed={Modality.CT, Modality.MR},
    ),
    _case(
        "spinal_cord_lesion_supports_mr",
        payload=_payload(
            name="spinal cord lesion",
            description="Focal abnormal signal within the spinal cord.",
            ontology=[_ontology_candidate("GAMUTS:17100", "spinal cord lesion")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6449", "spinal cord")],
            body_regions=[BodyRegion.SPINE],
        ),
        required={Modality.MR},
        forbidden={Modality.MG, Modality.RF, Modality.PET},
        allowed={Modality.MR, Modality.CT},
    ),
    _case(
        "adnexal_t2_mass_supports_mr_and_us",
        payload=_payload(
            name="T2 hypointense adnexal mass",
            description="Adnexal mass that appears hypointense on T2-weighted MRI.",
            ontology=[_ontology_candidate("GAMUTS:17752", "T2-hypointense adnexal mass")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28668", "uterine adnexa")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Modality.MR},
        forbidden={Modality.XR, Modality.MG},
        allowed={Modality.MR, Modality.US},
    ),
    _case(
        "ovarian_torsion_supports_us",
        payload=_payload(
            name="ovarian torsion",
            description="Twisting of the ovarian vascular pedicle with compromised flow.",
            ontology=[_ontology_candidate("SNOMEDCT:76571007", "Torsion of ovary")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28702", "ovary")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Modality.US},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.US, Modality.CT, Modality.MR},
    ),
    _case(
        "hydronephrosis_supports_us_and_ct",
        payload=_payload(
            name="hydronephrosis",
            description="Dilation of the renal collecting system.",
            ontology=[_ontology_candidate("SNOMEDCT:43064006", "Hydronephrosis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID205", "kidney")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Modality.US},
        forbidden={Modality.MG, Modality.PET},
        allowed={Modality.US, Modality.CT},
    ),
    _case(
        "radiolucent_urinary_calculus_excludes_xr",
        payload=_payload(
            name="radiolucent urinary calculus",
            description="Urinary tract stone that is not visible on plain radiography.",
            ontology=[_ontology_candidate("SNOMEDCT:95570007", "Urinary calculus")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID39343", "urinary tract")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Modality.CT, Modality.US},
        forbidden={Modality.XR, Modality.MG},
        allowed={Modality.CT, Modality.US},
    ),
    _case(
        "thyroid_nodule_supports_us",
        payload=_payload(
            name="thyroid nodule",
            description="Discrete nodule arising in the thyroid gland.",
            ontology=[_ontology_candidate("SNOMEDCT:237495005", "Thyroid nodule")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28550", "thyroid gland")],
            body_regions=[BodyRegion.NECK],
        ),
        required={Modality.US},
        forbidden={Modality.XR, Modality.MG, Modality.RF},
        allowed={Modality.US, Modality.CT, Modality.MR},
    ),
    _case(
        "testicular_torsion_supports_us",
        payload=_payload(
            name="testicular torsion",
            description="Twisting of the spermatic cord with compromised testicular blood flow.",
            ontology=[_ontology_candidate("SNOMEDCT:81996005", "Torsion of testis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID301", "testis")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Modality.US},
        forbidden={Modality.MG, Modality.RF},
        allowed={Modality.US},
    ),
    _case(
        "breast_calcification_cluster_supports_mg",
        payload=_payload(
            name="breast calcification cluster",
            description="Clustered calcifications in breast tissue on mammography.",
            ontology=[_ontology_candidate("RADLEX:RID34218", "clustered calcifications")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28749", "breast")],
            body_regions=[BodyRegion.BREAST],
        ),
        required={Modality.MG},
        forbidden={Modality.XR, Modality.NM, Modality.DSA},
        allowed={Modality.MG},
    ),
    _case(
        "breast_density_supports_mg",
        payload=_payload(
            name="breast density",
            description="Mammographic assessment of fibroglandular breast density.",
            ontology=[_ontology_candidate("RADLEX:RID34245", "breast density")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28749", "breast")],
            body_regions=[BodyRegion.BREAST],
        ),
        required={Modality.MG},
        forbidden={Modality.XR, Modality.CT, Modality.NM},
        allowed={Modality.MG},
    ),
    _case(
        "pet_avid_pulmonary_nodule_supports_pet",
        payload=_payload(
            name="PET-avid pulmonary nodule",
            description="Pulmonary nodule with increased radiotracer uptake on PET.",
            ontology=[_ontology_candidate("GAMUTS:18901", "PET-avid pulmonary nodule")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.PET},
        forbidden={Modality.MG, Modality.RF, Modality.DSA},
        allowed={Modality.PET, Modality.CT},
    ),
    _case(
        "tumor_treatment_response_supports_pet_and_ct",
        payload=_payload(
            name="tumor treatment response",
            description="Assessment of interval response of known tumor burden after therapy.",
            ontology=[_ontology_candidate("RADLEX:RID49462", "tumor response")],
            anatomy=[],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required={Modality.CT},
        forbidden={Modality.XR, Modality.MG, Modality.RF},
        allowed={Modality.CT, Modality.MR, Modality.PET},
    ),
    _case(
        "bone_scan_uptake_supports_nm",
        payload=_payload(
            name="bone scan uptake abnormality",
            description="Abnormal focal radiotracer uptake on skeletal scintigraphy.",
            ontology=[_ontology_candidate("RADLEX:RID10340", "abnormal bone scan uptake")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID13295", "skeleton")],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required={Modality.NM},
        forbidden={Modality.MG, Modality.RF},
        allowed={Modality.NM, Modality.PET},
    ),
    _case(
        "vq_mismatch_supports_nm",
        payload=_payload(
            name="ventilation-perfusion mismatch",
            description="Mismatch between ventilation and perfusion on lung scintigraphy.",
            ontology=[_ontology_candidate("RADLEX:RID10349", "ventilation perfusion mismatch")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.NM},
        forbidden={Modality.MG, Modality.RF},
        allowed={Modality.NM},
    ),
    _case(
        "hida_gallbladder_nonvisualization_supports_nm",
        payload=_payload(
            name="HIDA gallbladder nonvisualization",
            description="Nonvisualization of the gallbladder on hepatobiliary scintigraphy.",
            ontology=[_ontology_candidate("RADLEX:RID10351", "HIDA scan abnormality")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID187", "gallbladder")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Modality.NM},
        forbidden={Modality.MG, Modality.PET},
        allowed={Modality.NM},
    ),
    _case(
        "uterine_artery_embolization_supports_dsa",
        payload=_payload(
            name="uterine artery embolization",
            description="Endovascular embolization of the uterine arteries.",
            ontology=[_ontology_candidate("SNOMEDCT:17514000", "Embolization of uterine artery")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28672", "uterine artery")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Modality.DSA},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.DSA, Modality.RF, Modality.US},
    ),
    _case(
        "cerebral_aneurysm_coiling_supports_dsa",
        payload=_payload(
            name="cerebral aneurysm coiling",
            description="Endovascular coil treatment of an intracranial aneurysm.",
            ontology=[_ontology_candidate("SNOMEDCT:432119003", "Endovascular coiling of aneurysm")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6434", "brain")],
            body_regions=[BodyRegion.HEAD],
        ),
        required={Modality.DSA},
        forbidden={Modality.MG},
        allowed={Modality.DSA, Modality.RF, Modality.CT, Modality.MR},
    ),
    _case(
        "swallow_study_supports_rf",
        payload=_payload(
            name="videofluoroscopic swallow study aspiration",
            description="Aspiration observed during fluoroscopic swallow evaluation.",
            ontology=[_ontology_candidate("RADLEX:RID10370", "swallow study aspiration")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28561", "pharynx")],
            body_regions=[BodyRegion.NECK],
        ),
        required={Modality.RF},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.RF},
    ),
    _case(
        "esophagram_stricture_supports_rf",
        payload=_payload(
            name="esophageal stricture on esophagram",
            description="Narrowing of the esophagus demonstrated on contrast esophagram.",
            ontology=[_ontology_candidate("SNOMEDCT:63305008", "Esophageal stricture")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID13227", "esophagus")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.RF},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.RF, Modality.CT},
    ),
    _case(
        "acl_tear_supports_mr",
        payload=_payload(
            name="anterior cruciate ligament tear",
            description="Disruption of the anterior cruciate ligament of the knee.",
            ontology=[_ontology_candidate("SNOMEDCT:239725005", "Rupture of anterior cruciate ligament")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID26077", "anterior cruciate ligament")],
            body_regions=[BodyRegion.LOWER_EXTREMITY],
        ),
        required={Modality.MR},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.MR, Modality.XR},
    ),
    _case(
        "rib_fracture_supports_xr_and_ct",
        payload=_payload(
            name="rib fracture",
            description="Fracture involving a rib.",
            ontology=[_ontology_candidate("SNOMEDCT:33737001", "Fracture of rib")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID2507", "rib")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Modality.XR},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.XR, Modality.CT},
    ),
    _case(
        "motion_artifact_has_no_specific_modality",
        payload=_payload(
            name="motion artifact",
            description="Image degradation caused by patient motion during acquisition.",
            ontology=[_ontology_candidate("RADLEX:RID10312", "motion artifact")],
            anatomy=[],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required=set(),
        forbidden={Modality.MG, Modality.PET, Modality.NM, Modality.DSA},
        allowed=None,
    ),
    _case(
        "ct_dose_alert_supports_ct",
        payload=_payload(
            name="CT dose alert",
            description="Dose or safety issue related to CT acquisition.",
            ontology=[_ontology_candidate("RADLEX:RID10443", "radiation dose")],
            anatomy=[],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required={Modality.CT},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.CT},
    ),
    _case(
        "placenta_previa_supports_us",
        payload=_payload(
            name="placenta previa",
            description="Placental tissue extends over or near the internal cervical os.",
            ontology=[_ontology_candidate("SNOMEDCT:36813001", "Placenta previa")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28670", "placenta")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Modality.US},
        forbidden={Modality.MG, Modality.PET, Modality.NM},
        allowed={Modality.US, Modality.MR},
    ),
    _case(
        "brain_hemorrhage_supports_ct",
        payload=_payload(
            name="quantified intracranial hemorrhage",
            description="Quantitative assessment of blood within the intracranial compartment.",
            ontology=[_ontology_candidate("SNOMEDCT:1386000", "Intracranial hemorrhage")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6434", "brain")],
            body_regions=[BodyRegion.HEAD],
        ),
        required={Modality.CT},
        forbidden={Modality.MG, Modality.RF},
        allowed={Modality.CT, Modality.MR},
    ),
]


class ModalityApplicabilityEvaluator(
    Evaluator[
        ModalityApplicabilityDecisionInput,
        ModalityApplicabilityDecisionActualOutput,
        ModalityApplicabilityDecisionExpectedOutput,
    ]
):
    """Score required, forbidden, and allowed modality agreement."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            ModalityApplicabilityDecisionInput,
            ModalityApplicabilityDecisionActualOutput,
            ModalityApplicabilityDecisionExpectedOutput,
        ],
    ) -> float:
        if ctx.output.error is not None:
            return 0.0
        return score_required_forbidden_allowed(
            ctx.output.applicable_modalities,
            required=ctx.expected_output.required_modalities,
            forbidden=ctx.expected_output.forbidden_modalities,
            allowed=ctx.expected_output.allowed_modalities,
        )


def score_modality_applicability_output(
    expected: ModalityApplicabilityDecisionExpectedOutput,
    output: ModalityApplicabilityDecisionActualOutput,
) -> bool:
    """Return whether one actual output satisfies the fixture expectation."""

    if output.error is not None:
        return False
    modalities = output.applicable_modalities
    if not expected.required_modalities.issubset(modalities):
        return False
    if expected.forbidden_modalities.intersection(modalities):
        return False
    return expected.allowed_modalities is None or modalities.issubset(expected.allowed_modalities)


def weighted_modality_applicability_score(
    expected: ModalityApplicabilityDecisionExpectedOutput,
    output: ModalityApplicabilityDecisionActualOutput,
) -> float:
    """Return the weighted numeric score for one actual output."""

    if output.error is not None:
        return 0.0
    return score_required_forbidden_allowed(
        output.applicable_modalities,
        required=expected.required_modalities,
        forbidden=expected.forbidden_modalities,
        allowed=expected.allowed_modalities,
    )


async def run_modality_applicability_decision_task(
    case_input: ModalityApplicabilityDecisionInput,
) -> ModalityApplicabilityDecisionActualOutput:
    """Run only the modality-applicability decision agent for one replay payload."""

    agent = create_modality_applicability_agent()
    payload = copy.deepcopy(case_input.payload)
    try:
        result = await agent.run(json.dumps(payload, indent=2))
    except Exception as exc:  # pragma: no cover - useful in eval reports
        return ModalityApplicabilityDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")

    return ModalityApplicabilityDecisionActualOutput(
        applicable_modalities=set(result.output.applicable_modalities or []),
        raw_output=result.output,
    )


def build_modality_applicability_dataset() -> Dataset[
    ModalityApplicabilityDecisionInput,
    ModalityApplicabilityDecisionActualOutput,
    ModalityApplicabilityDecisionExpectedOutput,
]:
    """Build the pilot modality-applicability dataset."""

    return Dataset(cases=CASES, evaluators=[ModalityApplicabilityEvaluator()])


async def run_metadata_modality_applicability_decision_evals() -> EvaluationReport[
    ModalityApplicabilityDecisionInput,
    ModalityApplicabilityDecisionActualOutput,
    ModalityApplicabilityDecisionExpectedOutput,
]:
    """Run the focused modality-applicability replay suite."""

    return await build_modality_applicability_dataset().evaluate(
        run_modality_applicability_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )


async def sample_modality_applicability_cases(case_names: list[str], *, repeats: int) -> None:
    """Run selected cases repeatedly and print pass rates plus output variants."""

    case_by_name = {case.name: case for case in CASES}
    missing_names = [name for name in case_names if name not in case_by_name]
    if missing_names:
        raise SystemExit(f"Unknown case name(s): {', '.join(missing_names)}")

    for case_name in case_names:
        case = case_by_name[case_name]
        variants: Counter[tuple[str, ...]] = Counter()
        errors: Counter[str] = Counter()
        pass_count = 0
        for _ in range(repeats):
            output = await run_modality_applicability_decision_task(case.inputs)
            if output.error is not None:
                errors[output.error] += 1
            else:
                variants[tuple(sorted(value.value for value in output.applicable_modalities))] += 1
            if score_modality_applicability_output(case.expected_output, output):
                pass_count += 1

        print(f"\n{case_name}: {pass_count}/{repeats} passed")
        if variants:
            print("  variants:")
            for variant, count in variants.most_common():
                label = ", ".join(variant) if variant else "<empty>"
                print(f"  - {count}x {label}")
        if errors:
            print("  errors:")
            for error, count in errors.most_common():
                print(f"  - {count}x {error}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--sample-case",
            action="append",
            default=[],
            help="Run one named case repeatedly and summarize output variants. May be passed more than once.",
        )
        parser.add_argument(
            "--repeats",
            type=int,
            default=5,
            help="Number of repetitions for each --sample-case.",
        )
        args = parser.parse_args()
        if args.sample_case:
            asyncio.run(sample_modality_applicability_cases(args.sample_case, repeats=args.repeats))
        else:
            print("\nRunning focused modality-applicability decision evaluation suite...")
            print("=" * 80)
            report = asyncio.run(run_metadata_modality_applicability_decision_evals())
            report.print(include_input=False, include_expected_output=False, include_durations=True)
            print_weighted_summary(report, EVALUATOR_WEIGHTS, title="Modality applicability")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
