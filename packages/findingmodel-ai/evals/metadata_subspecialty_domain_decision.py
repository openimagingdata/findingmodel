"""Focused evals for the metadata subspecialty-domain decision sub-agent."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from collections import Counter
from typing import Any

from findingmodel import BodyRegion, Subspecialty
from findingmodel_ai.metadata.assignment import create_subspecialty_domain_agent
from findingmodel_ai.metadata.decisions import SubspecialtyDomainDecision
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from evals.metadata_scoring import (
    print_weighted_summary,
    score_required_forbidden_allowed,
)

EVAL_MAX_CONCURRENCY = 2
EVALUATOR_WEIGHTS: dict[str, float] = {"SubspecialtyDomainEvaluator": 1.0}


class SubspecialtyDomainDecisionInput(BaseModel):
    """Input payload for one subspecialty-domain replay case."""

    payload: dict[str, Any]


class SubspecialtyDomainDecisionExpectedOutput(BaseModel):
    """Expected subspecialty-domain output for one replay case."""

    required_subspecialties: set[Subspecialty] = Field(default_factory=set)
    forbidden_subspecialties: set[Subspecialty] = Field(default_factory=set)
    allowed_subspecialties: set[Subspecialty] | None = None


class SubspecialtyDomainDecisionActualOutput(BaseModel):
    """Observed subspecialty-domain output for one replay case."""

    subspecialties: set[Subspecialty] = Field(default_factory=set)
    raw_output: SubspecialtyDomainDecision | None = None
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
        "task": "Assign only subspecialties.",
        "subspecialty_values_under_review": None,
        "anatomy_context": {"body_regions": [region.value for region in body_regions]},
    }


def _case(
    name: str,
    *,
    payload: dict[str, Any],
    required: set[Subspecialty],
    forbidden: set[Subspecialty] | None = None,
    allowed: set[Subspecialty] | None = None,
) -> Case[SubspecialtyDomainDecisionInput, SubspecialtyDomainDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=SubspecialtyDomainDecisionInput(payload=payload),
        expected_output=SubspecialtyDomainDecisionExpectedOutput(
            required_subspecialties=required,
            forbidden_subspecialties=forbidden or set(),
            allowed_subspecialties=allowed,
        ),
    )


CASES: list[Case[SubspecialtyDomainDecisionInput, SubspecialtyDomainDecisionExpectedOutput]] = [
    _case(
        "aortic_aneurysm_is_vascular",
        payload=_payload(
            name="abdominal aortic aneurysm",
            description="Focal abnormal dilation of the abdominal aorta.",
            ontology=[_ontology_candidate("SNOMEDCT:233985008", "Aneurysm of abdominal aorta")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID480", "abdominal aorta")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.VA},
        forbidden={Subspecialty.GI, Subspecialty.GU},
        allowed={Subspecialty.VA},
    ),
    _case(
        "cervical_lymphoid_tissue_is_head_neck",
        payload=_payload(
            name="cervical lymphoid tissue",
            description="Lymphoid tissue in the cervical neck region.",
            ontology=[_ontology_candidate("SNOMEDCT:59441001", "Lymphoid tissue")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28504", "cervical lymph node")],
            body_regions=[BodyRegion.NECK],
        ),
        required={Subspecialty.HN},
        forbidden={Subspecialty.CH, Subspecialty.MK},
        allowed={Subspecialty.HN, Subspecialty.OI},
    ),
    _case(
        "pulmonary_mass_is_chest_not_cardiac",
        payload=_payload(
            name="pulmonary mass",
            description="Mass-like opacity located in the lung.",
            ontology=[_ontology_candidate("GAMUTS:18470", "pulmonary mass")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.CH},
        forbidden={Subspecialty.CA, Subspecialty.VA},
        allowed={Subspecialty.CH, Subspecialty.OI},
    ),
    _case(
        "coronary_calcified_plaque_burden_is_cardiac",
        payload=_payload(
            name="coronary calcified plaque burden",
            description="Quantification of calcified plaque in the coronary arteries.",
            ontology=[_ontology_candidate("RADLEX:RID49701", "coronary artery calcification")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID34755", "coronary artery")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.CA},
        allowed={Subspecialty.CA, Subspecialty.CH, Subspecialty.VA},
    ),
    _case(
        "renal_calculus_is_gu_not_vascular",
        payload=_payload(
            name="radiolucent urinary calculus",
            description="Urinary tract stone that is not visible on plain radiography.",
            ontology=[_ontology_candidate("SNOMEDCT:95570007", "Urinary calculus")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID39343", "urinary tract")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.GU},
        forbidden={Subspecialty.VA, Subspecialty.GI},
        allowed={Subspecialty.GU, Subspecialty.ER},
    ),
    _case(
        "breast_calcification_cluster_is_breast",
        payload=_payload(
            name="breast calcification cluster",
            description="Clustered calcifications in breast tissue on mammography.",
            ontology=[_ontology_candidate("RADLEX:RID34218", "clustered calcifications")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28749", "breast")],
            body_regions=[BodyRegion.BREAST],
        ),
        required={Subspecialty.BR},
        forbidden={Subspecialty.CH},
        allowed={Subspecialty.BR, Subspecialty.OI},
    ),
    _case(
        "brain_hemorrhage_is_neuro_emergency",
        payload=_payload(
            name="quantified intracranial hemorrhage",
            description="Quantitative assessment of blood within the intracranial compartment.",
            ontology=[_ontology_candidate("SNOMEDCT:1386000", "Intracranial hemorrhage")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6434", "brain")],
            body_regions=[BodyRegion.HEAD],
        ),
        required={Subspecialty.NR},
        forbidden={Subspecialty.HN},
        allowed={Subspecialty.NR, Subspecialty.ER},
    ),
    _case(
        "uterine_adnexal_mass_is_ob_gyn",
        payload=_payload(
            name="T2 hypointense adnexal mass",
            description="Adnexal mass that appears hypointense on T2-weighted MRI.",
            ontology=[_ontology_candidate("GAMUTS:17752", "T2-hypointense adnexal mass")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28668", "uterine adnexa")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Subspecialty.OB},
        forbidden={Subspecialty.GI},
        allowed={Subspecialty.OB, Subspecialty.GU, Subspecialty.OI},
    ),
    _case(
        "carotid_artery_stenosis_is_vascular_head_neck_overlay",
        payload=_payload(
            name="carotid artery stenosis",
            description="Narrowing of the cervical carotid artery.",
            ontology=[_ontology_candidate("SNOMEDCT:64586002", "Carotid artery stenosis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID5809", "carotid artery")],
            body_regions=[BodyRegion.NECK],
        ),
        required={Subspecialty.VA},
        forbidden={Subspecialty.CH, Subspecialty.GI},
        allowed={Subspecialty.VA, Subspecialty.HN, Subspecialty.NR},
    ),
    _case(
        "pulmonary_embolism_is_vascular_chest_emergency_overlay",
        payload=_payload(
            name="pulmonary embolism",
            description="Embolic occlusion of a pulmonary artery.",
            ontology=[_ontology_candidate("SNOMEDCT:59282003", "Pulmonary embolism")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1310", "pulmonary artery")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.VA},
        forbidden={Subspecialty.CA, Subspecialty.GI},
        allowed={Subspecialty.VA, Subspecialty.CH, Subspecialty.ER},
    ),
    _case(
        "portal_vein_thrombosis_is_vascular_gi_overlay",
        payload=_payload(
            name="portal vein thrombosis",
            description="Thrombus within the portal vein.",
            ontology=[_ontology_candidate("SNOMEDCT:17920008", "Portal vein thrombosis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID494", "portal vein")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.VA},
        forbidden={Subspecialty.GU, Subspecialty.CA},
        allowed={Subspecialty.VA, Subspecialty.GI, Subspecialty.ER},
    ),
    _case(
        "hepatic_mass_is_gi_oncologic_possible",
        payload=_payload(
            name="hepatic mass",
            description="Mass lesion centered in the liver.",
            ontology=[_ontology_candidate("GAMUTS:14309", "hepatic mass")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID58", "liver")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.GI},
        forbidden={Subspecialty.GU, Subspecialty.VA},
        allowed={Subspecialty.GI, Subspecialty.OI},
    ),
    _case(
        "small_bowel_obstruction_is_gi_emergency_possible",
        payload=_payload(
            name="small bowel obstruction",
            description="Dilated small bowel loops with obstructive transition point.",
            ontology=[_ontology_candidate("SNOMEDCT:281255004", "Small bowel obstruction")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID13238", "small intestine")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.GI},
        forbidden={Subspecialty.GU, Subspecialty.VA},
        allowed={Subspecialty.GI, Subspecialty.ER},
    ),
    _case(
        "pancreatic_duct_dilation_is_gi_not_gu",
        payload=_payload(
            name="pancreatic duct dilation",
            description="Abnormal caliber of the pancreatic duct.",
            ontology=[_ontology_candidate("RADLEX:RID48963", "pancreatic duct dilatation")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID170", "pancreatic duct")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.GI},
        forbidden={Subspecialty.GU, Subspecialty.VA},
        allowed={Subspecialty.GI, Subspecialty.OI},
    ),
    _case(
        "hydronephrosis_is_gu",
        payload=_payload(
            name="hydronephrosis",
            description="Dilation of the renal collecting system.",
            ontology=[_ontology_candidate("SNOMEDCT:43064006", "Hydronephrosis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID205", "kidney")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.GU},
        forbidden={Subspecialty.GI, Subspecialty.VA},
        allowed={Subspecialty.GU, Subspecialty.ER},
    ),
    _case(
        "prostate_lesion_is_gu_oncologic_possible",
        payload=_payload(
            name="prostate lesion",
            description="Focal lesion in the prostate gland.",
            ontology=[_ontology_candidate("GAMUTS:16614", "prostate lesion")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID344", "prostate")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Subspecialty.GU},
        forbidden={Subspecialty.GI, Subspecialty.CH},
        allowed={Subspecialty.GU, Subspecialty.OI},
    ),
    _case(
        "ovarian_torsion_is_ob_emergency",
        payload=_payload(
            name="ovarian torsion",
            description="Twisting of the ovarian vascular pedicle with compromised flow.",
            ontology=[_ontology_candidate("SNOMEDCT:76571007", "Torsion of ovary")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28702", "ovary")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Subspecialty.OB, Subspecialty.ER},
        forbidden={Subspecialty.GI, Subspecialty.CH},
        allowed={Subspecialty.OB, Subspecialty.GU, Subspecialty.VA, Subspecialty.ER},
    ),
    _case(
        "placenta_previa_is_ob",
        payload=_payload(
            name="placenta previa",
            description="Placental tissue extends over or near the internal cervical os.",
            ontology=[_ontology_candidate("SNOMEDCT:36813001", "Placenta previa")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28670", "placenta")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Subspecialty.OB},
        forbidden={Subspecialty.GI, Subspecialty.GU},
        allowed={Subspecialty.OB, Subspecialty.ER},
    ),
    _case(
        "vertebral_compression_fracture_is_msk_emergency_possible",
        payload=_payload(
            name="acute vertebral compression fracture",
            description="Acute compression deformity of a thoracic vertebral body.",
            ontology=[_ontology_candidate("SNOMEDCT:698077007", "Compression fracture of vertebral column")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID320", "vertebra")],
            body_regions=[BodyRegion.SPINE],
        ),
        required={Subspecialty.MK},
        forbidden={Subspecialty.CH, Subspecialty.GI},
        allowed={Subspecialty.MK, Subspecialty.ER, Subspecialty.NR},
    ),
    _case(
        "spinal_cord_lesion_is_neuro_not_msk_only",
        payload=_payload(
            name="spinal cord lesion",
            description="Focal abnormal signal within the spinal cord.",
            ontology=[_ontology_candidate("GAMUTS:17100", "spinal cord lesion")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6449", "spinal cord")],
            body_regions=[BodyRegion.SPINE],
        ),
        required={Subspecialty.NR},
        forbidden={Subspecialty.CH, Subspecialty.GI},
        allowed={Subspecialty.NR, Subspecialty.MK, Subspecialty.OI},
    ),
    _case(
        "anterior_cruciate_ligament_tear_is_msk",
        payload=_payload(
            name="anterior cruciate ligament tear",
            description="Disruption of the anterior cruciate ligament of the knee.",
            ontology=[_ontology_candidate("SNOMEDCT:239725005", "Rupture of anterior cruciate ligament")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID26077", "anterior cruciate ligament")],
            body_regions=[BodyRegion.LOWER_EXTREMITY],
        ),
        required={Subspecialty.MK},
        forbidden={Subspecialty.VA, Subspecialty.NR},
        allowed={Subspecialty.MK, Subspecialty.ER},
    ),
    _case(
        "rib_fracture_is_chest_msk_emergency_possible",
        payload=_payload(
            name="rib fracture",
            description="Fracture involving a rib.",
            ontology=[_ontology_candidate("SNOMEDCT:33737001", "Fracture of rib")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID2507", "rib")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.CH, Subspecialty.MK},
        forbidden={Subspecialty.CA, Subspecialty.GI},
        allowed={Subspecialty.CH, Subspecialty.MK, Subspecialty.ER},
    ),
    _case(
        "thyroid_nodule_is_head_neck_oncologic_possible",
        payload=_payload(
            name="thyroid nodule",
            description="Discrete nodule arising in the thyroid gland.",
            ontology=[_ontology_candidate("SNOMEDCT:237495005", "Thyroid nodule")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28550", "thyroid gland")],
            body_regions=[BodyRegion.NECK],
        ),
        required={Subspecialty.HN},
        forbidden={Subspecialty.CH, Subspecialty.MK},
        allowed={Subspecialty.HN, Subspecialty.OI},
    ),
    _case(
        "orbital_floor_fracture_is_head_neck_msk_emergency_possible",
        payload=_payload(
            name="orbital floor fracture",
            description="Fracture of the orbital floor.",
            ontology=[_ontology_candidate("SNOMEDCT:71642004", "Fracture of orbital floor")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28569", "orbit")],
            body_regions=[BodyRegion.HEAD],
        ),
        required={Subspecialty.HN},
        forbidden={Subspecialty.NR, Subspecialty.CH},
        allowed={Subspecialty.HN, Subspecialty.MK, Subspecialty.ER},
    ),
    _case(
        "cerebral_infarction_is_neuro_vascular_emergency",
        payload=_payload(
            name="cerebral infarction",
            description="Acute infarction involving brain parenchyma.",
            ontology=[_ontology_candidate("SNOMEDCT:432504007", "Cerebral infarction")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6434", "brain")],
            body_regions=[BodyRegion.HEAD],
        ),
        required={Subspecialty.NR},
        forbidden={Subspecialty.HN, Subspecialty.CH},
        allowed={Subspecialty.NR, Subspecialty.VA, Subspecialty.ER},
    ),
    _case(
        "dural_based_mass_is_neuro_oncologic_possible",
        payload=_payload(
            name="dural-based mass",
            description="Extra-axial mass arising from or abutting the dura.",
            ontology=[_ontology_candidate("GAMUTS:11671", "dural-based mass")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6456", "dura mater")],
            body_regions=[BodyRegion.HEAD],
        ),
        required={Subspecialty.NR},
        forbidden={Subspecialty.HN, Subspecialty.CH},
        allowed={Subspecialty.NR, Subspecialty.OI},
    ),
    _case(
        "pet_avid_pulmonary_nodule_is_molecular_chest_oncologic",
        payload=_payload(
            name="PET-avid pulmonary nodule",
            description="Pulmonary nodule with increased radiotracer uptake on PET.",
            ontology=[_ontology_candidate("GAMUTS:18901", "PET-avid pulmonary nodule")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.MI, Subspecialty.CH},
        forbidden={Subspecialty.CA, Subspecialty.VA},
        allowed={Subspecialty.MI, Subspecialty.CH, Subspecialty.OI},
    ),
    _case(
        "bone_scan_uptake_abnormality_is_nuclear_medicine",
        payload=_payload(
            name="bone scan uptake abnormality",
            description="Abnormal focal radiotracer uptake on skeletal scintigraphy.",
            ontology=[_ontology_candidate("RADLEX:RID10340", "abnormal bone scan uptake")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID13295", "skeleton")],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required={Subspecialty.NM},
        forbidden={Subspecialty.CH, Subspecialty.GI},
        allowed={Subspecialty.NM, Subspecialty.MK, Subspecialty.OI},
    ),
    _case(
        "vq_mismatch_is_nuclear_vascular_chest",
        payload=_payload(
            name="ventilation-perfusion mismatch",
            description="Mismatch between ventilation and perfusion on lung scintigraphy.",
            ontology=[_ontology_candidate("RADLEX:RID10349", "ventilation perfusion mismatch")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.NM, Subspecialty.CH},
        forbidden={Subspecialty.CA, Subspecialty.GI},
        allowed={Subspecialty.NM, Subspecialty.CH, Subspecialty.VA, Subspecialty.ER},
    ),
    _case(
        "image_guided_liver_biopsy_is_ir_gi_oncologic_possible",
        payload=_payload(
            name="image-guided liver biopsy",
            description="Percutaneous image-guided biopsy of a liver lesion.",
            ontology=[_ontology_candidate("SNOMEDCT:274331003", "Biopsy of liver")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID58", "liver")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.IR, Subspecialty.GI},
        forbidden={Subspecialty.GU, Subspecialty.CH},
        allowed={Subspecialty.IR, Subspecialty.GI, Subspecialty.OI},
    ),
    _case(
        "uterine_artery_embolization_is_ir_ob_vascular",
        payload=_payload(
            name="uterine artery embolization",
            description="Endovascular embolization of the uterine arteries.",
            ontology=[_ontology_candidate("SNOMEDCT:17514000", "Embolization of uterine artery")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID28672", "uterine artery")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Subspecialty.IR, Subspecialty.VA, Subspecialty.OB},
        forbidden={Subspecialty.GI, Subspecialty.CH},
        allowed={Subspecialty.IR, Subspecialty.VA, Subspecialty.OB, Subspecialty.GU},
    ),
    _case(
        "central_venous_catheter_malposition_is_ir_vascular_chest_possible",
        payload=_payload(
            name="central venous catheter malposition",
            description="Malpositioned central venous catheter tip in the thorax.",
            ontology=[_ontology_candidate("RADLEX:RID5765", "central venous catheter malposition")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID6045", "central vein")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.IR, Subspecialty.VA},
        forbidden={Subspecialty.GI, Subspecialty.GU},
    ),
    _case(
        "motion_artifact_is_quality_safety_not_anatomic",
        payload=_payload(
            name="motion artifact",
            description="Image degradation caused by patient motion during acquisition.",
            ontology=[_ontology_candidate("RADLEX:RID10312", "motion artifact")],
            anatomy=[],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required={Subspecialty.SQ},
        forbidden={Subspecialty.CH, Subspecialty.GI, Subspecialty.GU, Subspecialty.MK},
        allowed={Subspecialty.SQ},
    ),
    _case(
        "ct_dose_alert_is_quality_safety",
        payload=_payload(
            name="CT dose alert",
            description="Dose or safety issue related to CT acquisition.",
            ontology=[_ontology_candidate("RADLEX:RID10443", "radiation dose")],
            anatomy=[],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required={Subspecialty.SQ},
        forbidden={Subspecialty.CH, Subspecialty.GI, Subspecialty.NR},
        allowed={Subspecialty.SQ},
    ),
    _case(
        "pediatric_necrotizing_enterocolitis_is_pediatric_gi_emergency",
        payload=_payload(
            name="pediatric necrotizing enterocolitis",
            description="Necrotizing enterocolitis in a neonate or infant.",
            ontology=[_ontology_candidate("SNOMEDCT:206999008", "Necrotizing enterocolitis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID13238", "intestine")],
            body_regions=[BodyRegion.ABDOMEN],
            tags=["pediatric"],
        ),
        required={Subspecialty.PD, Subspecialty.GI},
        forbidden={Subspecialty.GU, Subspecialty.CH},
        allowed={Subspecialty.PD, Subspecialty.GI, Subspecialty.ER},
    ),
    _case(
        "congenital_hip_dysplasia_is_pediatric_msk",
        payload=_payload(
            name="developmental dysplasia of the hip",
            description="Pediatric developmental abnormality of the acetabulum and femoral head.",
            ontology=[_ontology_candidate("SNOMEDCT:52781008", "Developmental dysplasia of hip")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID29632", "hip joint")],
            body_regions=[BodyRegion.LOWER_EXTREMITY],
            tags=["pediatric"],
        ),
        required={Subspecialty.PD, Subspecialty.MK},
        forbidden={Subspecialty.GI, Subspecialty.VA},
        allowed={Subspecialty.PD, Subspecialty.MK},
    ),
    _case(
        "testicular_torsion_is_gu_emergency_vascular_possible",
        payload=_payload(
            name="testicular torsion",
            description="Twisting of the spermatic cord with compromised testicular blood flow.",
            ontology=[_ontology_candidate("SNOMEDCT:81996005", "Torsion of testis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID301", "testis")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Subspecialty.GU, Subspecialty.ER},
        forbidden={Subspecialty.GI, Subspecialty.OB},
        allowed={Subspecialty.GU, Subspecialty.ER, Subspecialty.VA, Subspecialty.PD},
    ),
    _case(
        "aortic_dissection_is_vascular_emergency_chest_possible",
        payload=_payload(
            name="acute aortic dissection",
            description="Acute dissection flap involving the thoracic aorta.",
            ontology=[_ontology_candidate("SNOMEDCT:308546005", "Aortic dissection")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID486", "thoracic aorta")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.VA, Subspecialty.ER},
        forbidden={Subspecialty.GI, Subspecialty.GU},
        allowed={Subspecialty.VA, Subspecialty.ER, Subspecialty.CH, Subspecialty.CA},
    ),
    _case(
        "appendicitis_is_gi_emergency_not_gu",
        payload=_payload(
            name="acute appendicitis",
            description="Acute inflammation of the appendix.",
            ontology=[_ontology_candidate("SNOMEDCT:85189001", "Acute appendicitis")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID13241", "appendix")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.GI, Subspecialty.ER},
        forbidden={Subspecialty.GU, Subspecialty.VA},
        allowed={Subspecialty.GI, Subspecialty.ER},
    ),
    _case(
        "pneumothorax_is_chest_emergency_possible",
        payload=_payload(
            name="pneumothorax",
            description="Air in the pleural space.",
            ontology=[_ontology_candidate("SNOMEDCT:36118008", "Pneumothorax")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1340", "pleural space")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.CH},
        forbidden={Subspecialty.CA, Subspecialty.GI},
        allowed={Subspecialty.CH, Subspecialty.ER},
    ),
    _case(
        "pericardial_effusion_is_cardiac_chest_possible",
        payload=_payload(
            name="pericardial effusion",
            description="Fluid accumulation in the pericardial space often seen on chest imaging.",
            ontology=[_ontology_candidate("SNOMEDCT:373945007", "Pericardial effusion")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID341", "pericardium")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.CA},
        forbidden={Subspecialty.GI, Subspecialty.GU},
        allowed={Subspecialty.CA, Subspecialty.CH, Subspecialty.ER},
    ),
    _case(
        "scrotal_mass_is_gu_oncologic_possible_not_ob",
        payload=_payload(
            name="scrotal mass",
            description="Mass lesion arising in the scrotum.",
            ontology=[_ontology_candidate("GAMUTS:16880", "scrotal mass")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID300", "scrotum")],
            body_regions=[BodyRegion.PELVIS],
        ),
        required={Subspecialty.GU},
        forbidden={Subspecialty.OB, Subspecialty.GI},
        allowed={Subspecialty.GU, Subspecialty.OI},
    ),
    _case(
        "pleural_effusion_is_chest_not_cardiac_by_default",
        payload=_payload(
            name="pleural effusion",
            description="Fluid collection in the pleural space.",
            ontology=[_ontology_candidate("SNOMEDCT:60046008", "Pleural effusion")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1340", "pleural space")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.CH},
        forbidden={Subspecialty.CA, Subspecialty.GI},
        allowed={Subspecialty.CH, Subspecialty.OI},
    ),
    _case(
        "metastatic_liver_lesions_are_oncologic_gi",
        payload=_payload(
            name="metastatic liver lesions",
            description="Multiple hepatic lesions representing metastatic disease.",
            ontology=[_ontology_candidate("SNOMEDCT:94381002", "Metastatic malignant neoplasm to liver")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID58", "liver")],
            body_regions=[BodyRegion.ABDOMEN],
        ),
        required={Subspecialty.OI, Subspecialty.GI},
        forbidden={Subspecialty.GU, Subspecialty.VA},
        allowed={Subspecialty.OI, Subspecialty.GI},
    ),
    _case(
        "lung_cancer_staging_is_oncologic_chest",
        payload=_payload(
            name="lung cancer staging",
            description="Imaging assessment for staging known lung cancer.",
            ontology=[_ontology_candidate("SNOMEDCT:93880001", "Primary malignant neoplasm of lung")],
            anatomy=[_anatomic_candidate("ANATOMICLOCATIONS:RID1301", "lung")],
            body_regions=[BodyRegion.CHEST],
        ),
        required={Subspecialty.OI, Subspecialty.CH},
        forbidden={Subspecialty.CA, Subspecialty.GU},
        allowed={Subspecialty.OI, Subspecialty.CH, Subspecialty.MI},
    ),
    _case(
        "tumor_treatment_response_is_oncologic",
        payload=_payload(
            name="tumor treatment response",
            description="Assessment of interval response of known tumor burden after therapy.",
            ontology=[_ontology_candidate("RADLEX:RID49462", "tumor response")],
            anatomy=[],
            body_regions=[BodyRegion.WHOLE_BODY],
        ),
        required={Subspecialty.OI},
        forbidden={Subspecialty.GI, Subspecialty.GU, Subspecialty.CH},
        allowed={Subspecialty.OI, Subspecialty.MI},
    ),
]


class SubspecialtyDomainEvaluator(
    Evaluator[
        SubspecialtyDomainDecisionInput,
        SubspecialtyDomainDecisionActualOutput,
        SubspecialtyDomainDecisionExpectedOutput,
    ]
):
    """Score required, forbidden, and allowed subspecialty agreement."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            SubspecialtyDomainDecisionInput,
            SubspecialtyDomainDecisionActualOutput,
            SubspecialtyDomainDecisionExpectedOutput,
        ],
    ) -> float:
        if ctx.output.error is not None:
            return 0.0
        expected = ctx.expected_output
        return score_required_forbidden_allowed(
            ctx.output.subspecialties,
            required=expected.required_subspecialties,
            forbidden=expected.forbidden_subspecialties,
            allowed=expected.allowed_subspecialties,
        )


def score_subspecialty_domain_output(
    expected: SubspecialtyDomainDecisionExpectedOutput,
    output: SubspecialtyDomainDecisionActualOutput,
) -> bool:
    """Return whether one actual output satisfies the fixture expectation."""

    if output.error is not None:
        return False
    subspecialties = output.subspecialties
    if not expected.required_subspecialties.issubset(subspecialties):
        return False
    if expected.forbidden_subspecialties.intersection(subspecialties):
        return False
    return expected.allowed_subspecialties is None or subspecialties.issubset(expected.allowed_subspecialties)


def weighted_subspecialty_domain_score(
    expected: SubspecialtyDomainDecisionExpectedOutput,
    output: SubspecialtyDomainDecisionActualOutput,
) -> float:
    """Return the weighted numeric score for one actual output."""

    if output.error is not None:
        return 0.0
    return score_required_forbidden_allowed(
        output.subspecialties,
        required=expected.required_subspecialties,
        forbidden=expected.forbidden_subspecialties,
        allowed=expected.allowed_subspecialties,
    )


async def run_subspecialty_domain_decision_task(
    case_input: SubspecialtyDomainDecisionInput,
) -> SubspecialtyDomainDecisionActualOutput:
    """Run only the subspecialty-domain decision agent for one replay payload."""

    agent = create_subspecialty_domain_agent()
    payload = copy.deepcopy(case_input.payload)
    try:
        result = await agent.run(json.dumps(payload, indent=2))
    except Exception as exc:  # pragma: no cover - useful in eval reports
        return SubspecialtyDomainDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")

    return SubspecialtyDomainDecisionActualOutput(
        subspecialties=set(result.output.subspecialties or []),
        raw_output=result.output,
    )


def build_subspecialty_domain_dataset() -> Dataset[
    SubspecialtyDomainDecisionInput,
    SubspecialtyDomainDecisionActualOutput,
    SubspecialtyDomainDecisionExpectedOutput,
]:
    """Build the pilot subspecialty-domain dataset."""

    return Dataset(cases=CASES, evaluators=[SubspecialtyDomainEvaluator()])


async def run_metadata_subspecialty_domain_decision_evals() -> EvaluationReport[
    SubspecialtyDomainDecisionInput,
    SubspecialtyDomainDecisionActualOutput,
    SubspecialtyDomainDecisionExpectedOutput,
]:
    """Run the focused subspecialty-domain replay suite."""

    return await build_subspecialty_domain_dataset().evaluate(
        run_subspecialty_domain_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )


async def sample_subspecialty_domain_cases(case_names: list[str], *, repeats: int) -> None:
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
            output = await run_subspecialty_domain_decision_task(case.inputs)
            if output.error is not None:
                errors[output.error] += 1
            else:
                variants[tuple(sorted(value.value for value in output.subspecialties))] += 1
            if score_subspecialty_domain_output(case.expected_output, output):
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
            asyncio.run(sample_subspecialty_domain_cases(args.sample_case, repeats=args.repeats))
        else:
            print("\nRunning focused subspecialty-domain decision evaluation suite...")
            print("=" * 80)
            report = asyncio.run(run_metadata_subspecialty_domain_decision_evals())
            report.print(include_input=False, include_expected_output=False, include_durations=True)
            print_weighted_summary(report, EVALUATOR_WEIGHTS, title="Subspecialty domain")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
