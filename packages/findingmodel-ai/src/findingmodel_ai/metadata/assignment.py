"""Canonical metadata-assignment pipeline for structured finding model metadata."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import logfire
from anatomic_locations import AnatomicLocationIndex
from findingmodel import EntityType, FindingModelFull
from findingmodel.protocols import OntologySearchResult
from oidm_common.models import IndexCode
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from findingmodel_ai import logger
from findingmodel_ai.config import settings
from findingmodel_ai.metadata.confidence import is_high_confidence as _is_high_confidence
from findingmodel_ai.metadata.confidence import is_low_confidence as _is_low_confidence
from findingmodel_ai.metadata.confidence import is_low_or_medium_confidence as _is_low_or_medium_confidence
from findingmodel_ai.metadata.decisions import (
    AnatomicCandidateDecision,
    AnatomyDecision,
    EtiologyDecision,
    IdentityDecision,
    ImagingWorkflowDecision,
    MetadataAssignmentDecision,
    OntologyCandidateDecision,
    OntologyDecision,
    PatientApplicabilityDecision,
)
from findingmodel_ai.metadata.ontology_cache import OntologyEvidenceUsage, OntologyLookupCache
from findingmodel_ai.metadata.types import (
    AnatomicCandidate,
    ConfidenceFieldKey,
    FieldConfidenceScore,
    MetadataAssignmentResult,
    MetadataAssignmentReview,
    OntologyCandidate,
    OntologyCandidateRejectionReason,
    OntologyCandidateRelationship,
    OntologyCandidateReport,
)
from findingmodel_ai.search.anatomic import (
    AnatomicCandidateSearchResponse,
    LocationSearchResponse,
    find_anatomic_locations,
)
from findingmodel_ai.search.ontology import CategorizedOntologyConcepts, match_ontology_concepts


class _OntologyCandidateState(BaseModel):
    result: OntologySearchResult
    relationship: OntologyCandidateRelationship
    selected_as_canonical: bool = False
    rationale: str | None = None
    rejection_reason: OntologyCandidateRejectionReason | None = None
    source_bucket: str


class _AnatomicCandidateState(BaseModel):
    result: OntologySearchResult
    selected: bool = False
    rationale: str | None = None
    source_bucket: str
    support_level: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    broader_candidate_ids: list[str] = Field(default_factory=list)


_ANATOMIC_SUPPORT_ORDER: dict[str, int] = {
    "direct_source": 0,
    "source_inferred_query": 1,
    "parent_of_supported": 2,
    "ontology_context": 3,
    "child_of_supported": 4,
    "current_metadata": 5,
    "primary": 6,
    "alternate": 7,
    "search_only": 8,
}
_ONTOLOGY_SOURCE_ORDER: dict[str, int] = {
    "existing_index_codes": 0,
    "exact_matches": 1,
    "should_include": 2,
    "marginal": 3,
}
def _metadata_classification_settings() -> ModelSettings:
    return ModelSettings(temperature=0)


def _get_trace_id() -> str | None:
    """Return the current OpenTelemetry trace ID as a hex string if available."""
    span_context = otel_trace.get_current_span().get_span_context()
    if not span_context or not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"


def create_metadata_assignment_agent(
    model: Model | None = None,
) -> Agent[None, MetadataAssignmentDecision]:
    """Create the legacy all-field classifier agent.

    The model parameter exists for test injection (TestModel/FunctionModel).
    Production code should not pass it — model selection goes through agent tags.

    New production assignment uses the focused anatomy, ontology, identity, etiology, patient
    applicability, and imaging workflow agents below.
    This constructor is retained for compatibility with callers/tests that instantiate the legacy
    aggregate decision model directly.
    """
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, MetadataAssignmentDecision](
        model=resolved_model,
        output_type=MetadataAssignmentDecision,
        model_settings=_metadata_classification_settings(),
        retries=3,
        instructions="""You assign canonical structured metadata for a radiology finding model.

## Objective

You are not rewriting the authored finding definition. Preserve the finding's identity, name,
description, synonyms, attributes, and tags. Your job is only to decide:
- canonical structured metadata fields
- which ontology candidates are exact / clinically substitutable / related
- which anatomic candidates should be selected

The structured fields are the decision. `classification_rationale` explains the decision, but it
does not replace the decision. If you say an existing field is wrong, the structured output must
contain the corrected value when the evidence supports one.

## Assignment Mode Contract

- `reassess` means existing structured metadata is provisional context only and may be wrong.
  Non-empty existing fields are NOT locked. If the finding and candidate evidence support a better
  value, output the corrected replacement value.
- `fill_blanks_only` means existing non-empty structured metadata is locked context. Do not
  overwrite those populated fields, but do use them to infer and fill complementary blank fields.
- In `fill_blanks_only`, populate every blank structured field that is clearly supported. Do not
  stop after filling only one blank field if the evidence supports additional blank fields.
- In `reassess`, do not describe the correction only in prose while leaving the actual structured
  field null or effectively unchanged.
- Use `clear_fields` only when the best supported answer is truly unknown after review. Do not use
  `clear_fields` to remove a wrong required field when the corrected replacement value is supported.

## Field Rules

- `entity_type=diagnosis` for named diseases, disorders, injuries, complications, or syndromes
  such as embolism, aneurysm, dissection, pneumonia, fracture, or abscess.
- `entity_type=finding` for descriptive imaging observations or broad abnormality labels that do
  not themselves commit to one fully specified disease entity.
- `entity_type=measurement` for quantitative or graded measurements such as density, size, volume,
  score, or percentage, including cases like breast density.
- `entity_type=assessment`, `recommendation`, `technique_issue`, and `grouping` should be used
  only when the finding clearly matches those categories.
- Descriptive imaging states such as enlargement, effusions, uptake abnormalities, and image-quality
  problems usually remain `finding` or `technique_issue`, not `diagnosis`.
- Broad radiologic abnormalities that are commonly further typed by cause, subtype, compartment, or
  severity usually remain `finding` even when clinically important, for example air collections,
  hemorrhage patterns, cystic lesions, nodal enlargement, or metabolically avid nodules.
- Broad abnormal accumulations or collections of air, fluid, or blood usually remain `finding`
  unless the name itself commits to a specific disease mechanism or narrower diagnostic subtype.
- `body_regions` should reflect the primary affected imaged anatomy, not every symptom location or
  every place mentioned in the narrative.
- Use the most specific canonical body region available. `breast` is distinct from `chest`.
- Use the body region implied by the affected anatomy, not a neighboring compartment. Shoulder maps
  to `upper_extremity`, ribs/chest wall map to `chest`, and ovary/uterus/adnexa/prostate map to
  `pelvis`.
- Do not emit multiple body regions unless the finding itself clearly spans multiple primary regions
  or the selected anatomic evidence directly supports more than one region.
- Do not widen to adjacent or alternate regions just because the description mentions them as
  possible variants. Only include the regions actually supported by the named finding and selected
  anatomy.
- If the name, synonyms, ontology matches, and selected anatomic candidates all point to one
  dominant site, keep that single site even if the prose definition mentions less-common alternate
  sites.
- If an anatomic candidate is an organ or vessel inside a larger body region, still assign the
  corresponding body region (for example lung -> chest, abdominal aorta -> abdomen).
- Generalized technique issues or artifacts that are not localized to one region may use
  `body_regions=["whole_body"]`.
- `applicable_modalities` should include modalities where the finding is routinely demonstrated or
  evaluated, not every modality that could theoretically show it once in a while.
- Include a modality only when it is a routine, direct way to demonstrate or evaluate the named
  finding itself. Do not add a modality based only on indirect clues, rare use, or general
  possibility.
- For generalized artifacts or technique issues, include only modalities clearly supported by the
  finding definition and routine radiology use. Do not automatically include every modality enum
  just because motion or artifact can happen anywhere.
- Do not infer `US`, `PET`, `NM`, `RF`, or `DSA` for a generalized artifact unless the authored
  finding explicitly supports those modalities.
- Do not add `MR` to a thoracic finding by default just because MR can depict it. Include `MR` only
  when it is routine for the named finding itself.
- `etiologies` should capture intrinsic/common etiologic categories for the named finding, not an
  exhaustive differential of every possible cause. If no short, high-confidence set is justified,
  leave `etiologies` null.
- For base findings with many possible causes, keep `etiologies` short and high-confidence,
  usually zero to two broad codes unless the finding definition clearly supports more.
- `technique_issue`, `assessment`, and `measurement` usually should not receive etiologies.
- Use `vascular` for vessel-centered, perfusion, flow, or blood-supply mechanisms when the finding
  itself is not specifically cardiac. Use `cardiac` for heart-failure, cardiac-pressure,
  valvular, or fluid-overload mechanisms. Do not collapse these into an ambiguous combined label.
- `expected_time_course` should reflect how long the imaging finding itself typically remains
  observable, not the duration of the underlying clinical process and not every possible clinical
  variant.
- When evidence offers a range such as weeks/months, months/years, or years/permanent, choose the
  value that best matches the typical observable imaging window. Do not default to the upper bound.
- Congenital, fixed developmental, chronic structural anomaly, and calcification findings are
  usually `duration=permanent`; add `stable` or `progressive` only when the named finding supports
  that behavior.
- Chronic lesions, nodules, neoplasms, masses, and soft-tissue tumors usually persist for years
  rather than becoming permanent unless the imaging trace is fixed; add `progressive` when the
  modeled finding is biologically expected to grow or worsen.
- Acute injuries, acute inflammatory findings, contusions, pneumatocele, and similar healing
  abnormalities are usually `duration=weeks` or `duration=months` with `resolving` or `evolving`.
  Do not call fractures resolved merely because the acute episode resolves; persistent deformity or
  healing change can justify months, years, or permanent.
- Devices, tubes, lines, catheters, clips, and leads should not be marked permanent just because
  they are hardware. Choose weeks/months or months/years based on the device class; surgically
  retained clips may be permanent.
- Measurements, classifications, scores, and assessments usually have no intrinsic time course.
  Output `expected_time_course=null` unless the modeled finding itself has temporal biology.
- Age applicability defaults to all ages unless the finding identity truly excludes age groups.
  Use `more_common_in` only for supported commonness, not as a substitute for applicability.
- Sex specificity defaults to `sex-neutral` unless the anatomy or finding identity is intrinsically
  sex-specific. Fetal and pregnancy-related findings do not make the fetus itself female-specific;
  distinguish fetal applicability from patient sex specificity.

## Subspecialty Rules

- `subspecialties` is a fully multi-label field. All codes are non-exclusive. There is no primary
  or preferred single code. Return every applicable code supported by the finding.
- In `fill_blanks_only`, a blank `subspecialties` field still requires full reasoning from the
  finding identity, description, and candidate evidence. Do not under-fill `subspecialties` just
  because `body_regions` and `entity_type` are already locked.
- Choose radiology divisions that would reasonably read/report the finding, not every clinical
  specialty that could claim the anatomy.
- Do not describe only some codes as "additive." Co-occurrence is the default rule for this field.
- Generalized technique issues, artifacts, QA/QI findings, and imaging-safety problems usually
  should include `SQ`.
- Regional, organ-system, vascular, and emergency subspecialties can co-occur when justified.
- Use a narrower specialty in addition to a broader regional specialty when both are relevant.
- Do not treat one justified subspecialty tag as replacing another unless the evidence clearly rules
  the other out.
- Organ membership alone is not enough to add a narrower specialty.
- There is no generic abdominal fallback code in this schema. Do not invent `AB`.
- `GI` covers gastrointestinal, hepatobiliary, pancreatic, and other digestive-abdominal findings.
- Do not add `GI` to abdominal findings that are primarily GU, vascular, gynecologic, or
  non-digestive in nature.
- `CA` is for cardiac, coronary, and pericardial findings.
- `CH` is for pulmonary, pleural, mediastinal, rib, and chest-wall findings; many cardiac findings
  justify both `CA` and `CH`.
- Do not replace `CA` with `CH` for cardiac or pericardial conditions.
- Do not add `CH` just because an anatomy lies in the thorax when the finding is primarily cardiac,
  coronary, pericardial, or vascular. Thoracic location or chest-pain presentation alone is not
  enough.
- Thoracic vascular disease can still justify `CH` when thoracic/chest interpretation is a core
  part of the reading problem.
- Do not add `CA` to lung nodules, lung malignancy, pleural findings, or other noncardiac thoracic
  lesions.
- `GU` is for kidney, ureter, bladder, prostate, and female pelvic GU findings. `GU` can co-occur
  with `OB` for gynecologic findings.
- `OB` applies to ovarian, uterine, adnexal, and obstetric findings. Do not let `OB` replace `GU`
  when both are justified.
- `OI` applies to malignant, staging, surveillance, oncologic-workup, or malignant-pattern
  uptake problems, and for lymph-node findings where cancer/staging interpretation is a core part
  of the imaging problem. Do not add `OI` to every benign neoplasm by default.
- `MI` applies when PET/FDG or broader molecular/functional imaging interpretation is routine for
  the named finding.
- `NM` is for conventional nuclear medicine interpretation such as planar scintigraphy or
  SPECT/SPECT-CT centered findings.
- If PET/FDG is one of the routine modalities you selected for a malignant or tracer-driven problem,
  `MI` should usually also be present in `subspecialties`.
- If the finding is routine on planar scintigraphy or SPECT/SPECT-CT rather than PET-centered
  molecular imaging, prefer `NM`.
- If both PET-centered molecular imaging and conventional nuclear medicine are genuinely central,
  both `MI` and `NM` can be present.
- `MK` is for bones, joints, tendons, ligaments, fractures, and degenerative/traumatic spine
  findings. Shoulder findings are usually `MK`, not `CH`. Spine findings may require both `MK` and
  `NR`.
- Do not add `MK` unless the lesion itself is centered in musculoskeletal structures or in a
  degenerative/traumatic spine process. CNS and meningeal neoplasms remain `NR` unless the modeled
  site is actually musculoskeletal.
- `PD` applies to pediatric-specific entities and can co-occur with organ-system specialties such
  as `GI`.
- `VA` applies to vessel-centered vascular findings such as embolism, dissection, aneurysm,
  thrombosis of a named vessel, or direct arterial/venous injury.
- Use `VA`, not `VI`.
- Do not add `VA` just because the etiology is vascular or the lesion has vascular biology.
  Parenchymal endpoint diagnoses such as cerebral infarction and non-vessel mass lesions such as
  hemangioma do not automatically get `VA`.
- Do not add `VA` to nonvascular abdominal findings such as kidney stones.
- `ER` applies to acute or urgent findings. It should not replace the core organ-system or
  regional specialty.
- `ER` can also apply to high-risk conditions that are frequently worked up in emergency imaging,
  even if the finding can also be chronic outside the emergency setting.
- Acute traumatic musculoskeletal injuries and high-pain urgent GU diagnoses such as kidney stone
  commonly justify `ER` in addition to the organ-system specialty.

## Synthetic Contrast Examples

- `thoracic aortic injury` -> `body_regions=["chest"]`, `subspecialties=["CA","CH","VA","ER"]`,
  `entity_type=diagnosis`, `applicable_modalities=["CT","MR"]`.
- `cardiac silhouette enlargement` -> `body_regions=["chest"]`, `subspecialties=["CA","CH"]`,
  `entity_type=finding`, `applicable_modalities=["XR","CT"]`.
- `coronary calcified plaque burden` -> `body_regions=["chest"]`, `subspecialties=["CA"]`,
  `entity_type=finding`, `applicable_modalities=["CT"]`. Do not add `CH` by default.
- `brain hemorrhagic focus` -> `body_regions=["head"]`, `subspecialties=["NR","ER"]`,
  `entity_type=finding`, `applicable_modalities=["CT","MR"]`. A broad hemorrhage pattern remains a
  `finding`; a named compartment or lesion subtype would be narrower.
- `global motion-degradation artifact on cross-sectional imaging` ->
  `body_regions=["whole_body"]`, `subspecialties=["SQ"]`, `entity_type=technique_issue`,
  `applicable_modalities=["CT","MR"]`. Do not automatically include every modality in the enum.
- `pelvic adnexal cystic lesion` -> `body_regions=["pelvis"]`, `subspecialties=["GU","OB"]`,
  `entity_type=finding`, `applicable_modalities=["US","CT","MR"]`.
- `bronchogenic carcinoma` -> `body_regions=["chest"]`,
  `subspecialties=["CH","OI","MI"]`, `entity_type=diagnosis`,
  `applicable_modalities=["CT","PET","XR"]`. Do not add `CA` or `ER` by default.
- `PET-avid pulmonary mass` -> `body_regions=["chest"]`, `subspecialties=["CH","OI","MI"]`,
  `entity_type=finding`, `applicable_modalities=["PET","CT"]`.
- `scintigraphic thyroid uptake abnormality` -> `body_regions=["neck"]`, `subspecialties=["NM","HN"]`,
  `entity_type=finding`, `applicable_modalities=["NM"]`.
- `shoulder tendon tear` -> `body_regions=["upper_extremity"]`, `subspecialties=["MK"]`,
  `entity_type=diagnosis`, `applicable_modalities=["MR","US"]`. Shoulder maps to
  `upper_extremity`, not `chest`.
- `anterior cruciate ligament tear` -> `body_regions=["lower_extremity"]`,
  `subspecialties=["MK","ER"]`, `entity_type=diagnosis`, `applicable_modalities=["MR"]`.
- `kidney stone` -> `body_regions=["abdomen"]`, `subspecialties=["GU","ER"]`,
  `entity_type=diagnosis`, `applicable_modalities=["CT","US","XR"]`. Do not add `VA`.
- `cerebral infarction` -> `body_regions=["head"]`, `subspecialties=["NR","ER"]`,
  `entity_type=diagnosis`, `applicable_modalities=["CT","MR"]`. Do not add `VA` merely because
  the pathophysiology is vascular.
- `liver hemangioma` -> `body_regions=["abdomen"]`, `subspecialties=["GI"]`,
  `entity_type=diagnosis`, `applicable_modalities=["US","CT","MR"]`. Do not add `VA` just
  because the lesion is vascular in composition.
- `pulmonary embolism` -> `body_regions=["chest"]`, `subspecialties=["VA"]`,
  `entity_type=diagnosis`, `applicable_modalities=["CT"]`.
- `infant gastric outlet obstruction` -> `body_regions=["abdomen"]`,
  `subspecialties=["PD","GI","ER"]`, `entity_type=diagnosis`, `applicable_modalities=["US"]`.
- `benign dural-based extra-axial tumor` -> `body_regions=["head"]`, `subspecialties=["NR"]`,
  `entity_type=diagnosis`, `applicable_modalities=["MR","CT"]`. Do not widen to spine, add `OI`,
  or add `XR` just because related variants can involve the spine or skull.
- `thoracic vertebral collapse fracture` -> `body_regions=["spine"]`,
  `subspecialties=["MK","NR","ER"]`, `entity_type=diagnosis`,
  `applicable_modalities=["XR","CT","MR"]`.

## Candidate Rules

- Be conservative. If the evidence is weak, leave a field null/omitted rather than guessing.
- A slightly broader ontology concept can be acceptable when it preserves grouping of equivalent
  findings.
- A narrower ontology concept is not acceptable.
- Only mark ontology candidates as canonical when they are true equivalents for the finding.
- Do not store broader, narrower, merely related, procedure, exam, or modality-specific concepts as
  canonical `index_codes` unless they are true equivalents for the full modeled finding. Preserve
  useful non-canonical candidates in review output instead.
- For measurement, classification, score, or assessment models, prefer exact measurement or
  assessment concepts over disease, procedure, or classification-system bucket concepts.
- For non-canonical ontology candidates, provide a rejection reason whenever the evidence supports
  one.
- When a candidate is accepted as canonical or selected as anatomy, preserve its preferred term as
  the resulting `IndexCode.display`.
- Ontology labels can support `diagnosis`, but do not let an exact ontology match force
  `diagnosis` when the model name is still a broad radiologic abnormality or umbrella observation.
  In this schema, exact ontology matches can still map to `finding`.
- Use the provided candidate IDs exactly as given.
- Do not invent candidate IDs that are not in the prompt.

## Partial Field Teaching Snippets

These snippets are intentionally partial. Only the fields shown are the teaching target. Omitted
fields are not implied to be null and must still be decided from the actual finding and candidate
evidence.

- `mediastinal lymphadenopathy` -> `etiologies=["inflammatory:infectious","inflammatory","neoplastic:malignant"]`.
  Broad mediastinal nodal enlargement commonly supports infectious, inflammatory, and malignant
  etiologic labels together.
- `bone island (enostosis)` -> `etiologies=["normal-variant"]`.
- `post-radiation enteritis` -> `etiologies=["iatrogenic:post-radiation"]`.
- `necrotizing enterocolitis` -> `age_profile={"applicability":["newborn","infant"],"more_common_in":["newborn"]}`.
- `slipped capital femoral epiphysis` ->
  `age_profile={"applicability":["child","adolescent"],"more_common_in":["adolescent"]}`.
- `degenerative lumbar facet arthropathy` ->
  `age_profile={"applicability":"all_ages","more_common_in":["middle_aged","aged"]}`.
- `prostate abscess` -> `sex_specificity="male-specific"`.
- `endometrial polyp` -> `sex_specificity="female-specific"`.
- `renal cyst` -> `sex_specificity="sex-neutral"`.
- `pulmonary contusion` -> `expected_time_course={"duration":"weeks","modifiers":["resolving"]}`.
- `atheromatous plaque burden` ->
  `expected_time_course={"duration":"permanent","modifiers":["progressive"]}`.
- `bone mineral density T-score` -> `expected_time_course=null`.
  Measurement findings do not automatically get an intrinsic temporal trajectory.

## Index Code Teaching Snippets

- Canonical `index_codes` should keep the selected candidate's `system`, `code`, and preferred
  `display`.
- `pulmonary nodule`: exact or clinically substitutable pulmonary nodule codes, including SNOMED
  and RadLex equivalents, can be canonical. Broader labels such as `lung lesion` and narrower
  labels such as `spiculated pulmonary nodule` cannot be canonical for the broader modeled finding.
- `bone mineral density T-score`: if offered an exact `LOINC` measurement code and osteoporosis
  diagnosis codes, prefer the exact `LOINC` measurement code as canonical for the measurement
  finding. Do not substitute the disease code unless the modeled finding is osteoporosis itself.
- `mammographic architectural distortion`: accept finding codes; reject exam, procedure, or study
  codes.
- `developmental venous anomaly`: reject neighboring but non-equivalent vascular malformation
  codes.

## Output Discipline

- Keep `classification_rationale` concise and specific.
- `field_confidence` should only include fields you actually set or override.
- `field_confidence` values should be numeric scores from 0 to 1.
- `field_confidence` keys must be actual metadata field names only.

Return only the structured output.""",
    )


def create_identity_assignment_agent(model: Model | None = None) -> Agent[None, IdentityDecision]:
    """Create the focused identity and natural-history assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, IdentityDecision](
        model=resolved_model,
        output_type=IdentityDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions="""You assign only identity metadata for a radiology finding model.

Output only these fields when supported: `entity_type`, `expected_time_course`, `clear_fields`,
and optional `field_confidence`.

Rules:
- Preserve the authored finding identity; do not rewrite the model.
- For each optional field, first decide whether the source directly supports a value, directly
  contradicts a value, or leaves it unclear. Output a value only in the first case. Output `null` in
  the unclear case.
- Only output `expected_time_course` when source evidence supports the finding's intrinsic
  observable duration. If support is unclear, output `null` or omit it.
- In reassess mode, omitting a field preserves the existing value. Use `clear_fields` only when an
  existing identity field is unsupported and should be removed. If the evidence supports the existing
  time course, output that value instead of clearing it.
- Use selected canonical ontology only as identity support. Rejected ontology candidates, broader
  candidates, narrower candidates, related candidates, and review-only candidates do not support
  `entity_type` or `expected_time_course`.
- `diagnosis` is for named diseases, disorders, injuries, complications, or syndromes.
- Named aneurysms, thromboses, dissections, carcinomas, tumors, infections, and other disease
  entities should usually be `diagnosis` even when they can also appear as imaging observations.
- Broad imaging observations, descriptive lesions, enhancement patterns, effusions, uptake
  abnormalities, and umbrella abnormality labels usually remain `finding`.
- Exact ontology matches can support identity, but they do not automatically force `diagnosis`.
- Source tags such as `diagnosis`, `finding`, `congenital anomaly`, or organ/system tags are weak
  context. Use them to understand the source, but do not let tags alone assign `entity_type` or
  `expected_time_course`.
- `measurement`, `assessment`, `recommendation`, `technique_issue`, and `grouping` require clear
  support from the finding itself. Named scoring systems, grading scales, classifications, and
  structured reporting categories are `assessment`, not `finding`.
- Quantitative volume, size, density, ratio, percentile, and other numeric metric models are
  `measurement` when they represent one metric and `assessment` when they represent a grouped
  measurement/interpretation package. They are not `finding` merely because they are used to evaluate
  disease.
- Do not decide etiologies in this agent. A separate etiology agent handles them. Do not put
  `etiologies` in `clear_fields`.
- `expected_time_course` is how long the imaging finding remains observable, not the duration of the
  underlying clinical process. Do not default to the upper bound of a range.
- Because `expected_time_course` is optional, leave it null unless the modeled finding itself has a
  strongly supported, intrinsic natural history. Do not assign permanent/stable merely because a
  finding may reflect a congenital variant, anomaly, chronic disease, or structural difference.
- Generic morphology/abnormality findings usually have `expected_time_course=null` because the
  time course depends on the underlying cause.
- Use long or permanent time courses only for findings whose definition itself includes durable
  persistence, such as retained surgical material, healed fixed deformity, established
  calcification, or a named chronic/progressive disease.
- Acute injury/inflammation often lasts weeks or months with resolving/evolving modifiers.
- Devices, tubes, lines, and catheters are not permanent just because they are hardware; choose by
  expected dwell or leave null if the finding has no intrinsic time course.
- Generic vascular connections, shunts, fistulas, and malformations are usually imaging findings in
  this metadata context unless the authored model is clearly a named disease/diagnosis.
- Aneurysm is a disease entity, even when generic by vessel class. `arterial aneurysm` is a
  diagnosis, not a descriptive imaging finding.
- Named morphology terms and descriptive abnormality names remain `finding` when they describe an
  imaging appearance or structure, even if source tags include `diagnosis` or possible congenital
  causes. Use `diagnosis` only when the modeled concept is itself a disease, disorder, injury,
  complication, or syndrome.
- Negative findings, technique/artifact rows, pure assessments, and recommendations should not get
  invented chronicity.
- Examples:
  - `abnormal intracranial enhancement` with a description saying common causes include neoplasm,
    infection, inflammation, infarction, hemorrhage, and postoperative change: `entity_type=finding`,
    `expected_time_course=null`. The cause list is differential context.
  - `anisospondyly` with a description saying it is possibly due to growth disorders or congenital
    anomalies: `entity_type=finding`, `expected_time_course=null`. The possible
    causes are not the modeled finding itself.
  - `aortic atherosclerosis`: `entity_type=finding`; do not assign time course unless you can
    support it with high confidence from the modeled finding itself.
  - `epidural spinal cord compression scale`: `entity_type=assessment`, not `finding`, because it
    is a named grading scale.
- `clear_fields` entries must be metadata field names such as `expected_time_course`, never enum
  values or candidate IDs.
- Never put `field_confidence` or `clear_fields` inside `clear_fields`; those are response
  bookkeeping fields, not metadata fields that can be removed from the model.
- If evidence is weak, leave optional fields null rather than guessing.
- If `field_confidence` is present, keys must be real metadata fields and values must be
  numeric scores from 0 to 1.""",
    )


def create_etiology_assignment_agent(model: Model | None = None) -> Agent[None, EtiologyDecision]:
    """Create the focused etiology assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, EtiologyDecision](
        model=resolved_model,
        output_type=EtiologyDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions="""You assign only etiology metadata for a radiology finding model.

Output only `etiologies`, `clear_fields`, and optional `field_confidence`.

Core rule:
Assign an etiology only when the modeled concept itself says or conventionally means a causal
mechanism or disease class. If you are using possible causes, related variants, workup concerns,
tags, or existing metadata to infer the answer, leave `etiologies` null.

Use as evidence:
- finding name, description, synonyms, attributes;
- selected canonical ontology candidates where `default_selected_as_canonical` is true;
- decided `entity_type`.

Do not use as evidence:
- tags;
- existing metadata;
- selected anatomy;
- noncanonical ontology candidates, including `should_include`, related, broader, narrower,
  marginal, or search-only candidates.

Null by default:
- generic mass, lesion, opacity, calcification, fluid collection, soft-tissue abnormality, broad
  imaging sign, measurement, assessment, recommendation, technique issue, or grouping;
- morphology-only findings such as shape, contour, angle, notch, density, size, or spacing, unless
  the named morphology is conventionally tied to a causal class;
- descriptions that say "suggesting", "may represent", "can indicate", "possibly due to", or list
  possible causes.
- Do not use `mechanical` for a shape, notch, contour, indentation, deformity, measurement, or
  positional description merely because it involves physical form. Use `mechanical` only when the
  modeled concept itself names a physical mechanism such as obstruction, compression, traction,
  pressure, displacement, malposition, impaction, or hardware-related mechanics.

Non-null when identity supports it:
- urinary/biliary calculus or stone -> `metabolic`;
- calcification, mineralization, microcalcification, or calcification cluster is an imaging
  appearance/material description, not a metabolic etiology by itself;
- embolism or thrombus -> `vascular:thrombotic`;
- ischemia or infarct -> `vascular:ischemic`;
- pneumonia -> `inflammatory:infectious`;
- benign, malignant, or metastatic neoplasm -> the matching `neoplastic:*` bucket;
- established tumor or neoplasm with no benign/malignant/metastatic qualifier -> both
  `neoplastic:benign` and `neoplastic:malignant`, not `neoplastic:potential`;
- In tumor terminology, `primary` means originating at that site; it does not mean malignant.
  `primary brain tumor` or another unqualified primary tumor should still get both benign and
  malignant buckets unless the modeled source says malignant, benign, metastatic, or premalignant;
- use `neoplastic:potential` only for explicitly suspected, premalignant, at-risk, or possible
  future neoplasm concepts, not for an established tumor. Example: `renal tumor` -> benign and
  malignant; `premalignant colon polyp` -> potential.
- device, line, tube, stent, graft, or postoperative state -> matching `iatrogenic:*`.
- Use `normal-variant` only when the modeled concept itself is explicitly a normal variant or
  normal anatomic/developmental variant. Do not use `normal-variant` for a morphology, notch,
  calcification pattern, density, mass, lesion, or other observation merely because it can be benign
  or incidental.

Prefer the most specific supported bucket. Do not output both a child and its broad parent, such as
`inflammatory:infectious` plus `inflammatory`.

Use `clear_fields=["etiologies"]` only when `etiologies` is null. If you output a replacement
etiology value, leave `clear_fields` empty.

`clear_fields` entries must be metadata field names, never enum values or candidate IDs. Never put
`field_confidence` or `clear_fields` inside `clear_fields`.

If `field_confidence` is present, keys must be real metadata fields and values must be numeric scores
from 0 to 1.""",
    )


def create_patient_applicability_agent(model: Model | None = None) -> Agent[None, PatientApplicabilityDecision]:
    """Create the focused age/sex applicability assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, PatientApplicabilityDecision](
        model=resolved_model,
        output_type=PatientApplicabilityDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions="""You assign only patient applicability metadata for a radiology finding model.

Output only these fields when supported: `age_profile`, `sex_specificity`, `clear_fields`,
and optional `field_confidence`.

Rules:
- Use the finding name, description, selected anatomy, selected canonical ontology, attributes, and
  source tags as context. Rejected candidates and search-only candidates do not support age or sex.
- For each field, first decide whether the source directly supports a value, directly contradicts a
  value, or leaves it unclear. Output a value only when it is directly supported.
- Age defaults to all ages unless finding identity truly constrains applicability. Use
  `more_common_in` only for supported commonness, not as a substitute for applicability.
- Do not output age commonness unless the source directly supports commonness. If support is unclear,
  leave `age_profile` null.
- Pregnancy and fetal/gestational findings are not all-ages findings. Do not use maternal
  reproductive-age bins to represent fetal anatomy; if the available age enum cannot represent the
  fetal patient/applicability cleanly, leave `age_profile` null rather than inventing an adult
  applicability profile.
- Sex defaults to `sex-neutral` unless anatomy or finding identity is intrinsically sex-specific.
- Fetal/pregnancy applicability is not patient female specificity by itself.
- Breast tissue and mammography workflow do not automatically make a finding female-specific; use
  female-specific or male-specific only when the modeled finding or selected anatomy is explicitly
  sex-limited.
- Male genital anatomy such as testis, scrotum, penis, prostate, and seminal vesicle supports
  male-specific sex applicability.
- Uterus, ovary, adnexa, endometrium, cervix, pregnancy, and fetal-gestational findings support
  female-specific sex applicability only when the modeled patient applicability is female.
- Every non-null `age_profile` and `sex_specificity` value must have direct support from the modeled
  finding. Do not keep values because they are plausible.
- In reassess mode, existing patient metadata is reviewed context. Preserve existing values that are
  still supported by the finding; remove or omit them when the evidence contradicts them.
- `clear_fields` entries must be metadata field names such as `age_profile` or `sex_specificity`,
  never enum values or candidate IDs.
- Never put `field_confidence` or `clear_fields` inside `clear_fields`.
- If `field_confidence` is present, keys must be real metadata fields and values must be
  numeric scores from 0 to 1.""",
    )


def create_imaging_workflow_agent(model: Model | None = None) -> Agent[None, ImagingWorkflowDecision]:
    """Create the focused modality and subspecialty workflow assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, ImagingWorkflowDecision](
        model=resolved_model,
        output_type=ImagingWorkflowDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions="""You assign only imaging workflow metadata for a radiology finding model.

Output only these fields when supported: `subspecialties`, `applicable_modalities`,
`clear_fields`, and optional `field_confidence`.

Rules:
- Evidence standard:
  - Direct support: source name/description/synonyms that apply to the modeled finding; selected
    canonical ontology whose meaning itself implies a workflow or modality; selected anatomy only
    when the modeled finding is normally read in that anatomy-specific workflow.
  - Context only, not support: existing metadata, attribute names/values, rejected or review-only
    candidates, search-only anatomy, adjacent anatomy, possible complications, downstream workup,
    source tags, and statements that a modality could show the finding. Source tags are not supplied
    as workflow evidence because they are often broad/noisy.
  - For each output value, ask: if existing metadata, attributes, and rejected/review candidates were
    removed, would the source text, selected canonical ontology, and selected anatomy still directly
    support this value? If not, remove it.
- Existing workflow metadata is what you are reassessing, not source evidence. In reassess mode,
  omitting a field preserves it; output a corrected list or `clear_fields` when an existing value
  fails the support tests below. Existing workflow values are provided separately as
  `workflow_values_under_review`.
- For list fields, output the complete corrected list when some existing values should be removed.
  Use `clear_fields` only when no value for that optional field is supported.
- `applicable_modalities` is required. Choose the smallest set of routine direct modalities for the
  modeled finding itself. Source modality tags can help you understand context, but they are not
  enough by themselves; the modeled finding itself must support each modality.
- "Applicable" means the default routine imaging workflow for this finding model, not every
  modality that can sometimes evaluate it. Do not enumerate second-line, historical, confirmatory,
  problem-solving, or institution-dependent alternatives unless the modeled finding itself is
  modality-specific or the source explicitly requires them.
- Prefer the diagnostic use of a modality over mere detectability. If a modality can only show
  consequences, risk factors, screening context, or indirect clues for the named finding, exclude it
  even when that modality appears in tags.
- Do not infer a breast, chest, musculoskeletal, or other specialty workflow from a regional
  adjective alone. A generic regional soft-tissue mass or lesion supports only the modality choices
  directly suitable for that modeled mass, not a nearby organ workflow.
- An upper-extremity, axillary, chest-wall, abdominal-wall, or other regional soft-tissue location
  is not enough for `MK`; use `MK` only when the modeled finding is about bone, joint, muscle,
  tendon, ligament, spine, or a musculoskeletal disease/injury.
- Soft-tissue mass/lesion models do not get radiography merely because radiographs might show an
  indirect contour, opacity, calcification, or nearby bony effect. Include `XR` only when the source
  text, selected ontology, or selected anatomy makes radiography a routine direct modality for the
  modeled finding.
- Do not include modalities that only show indirect signs, screen for related disease, evaluate an
  adjacent cause, or belong to downstream workup. A modality tag is not enough when the named
  finding itself is not directly and routinely assessed with that modality.
- For embolic or thrombotic diagnoses, imaging that evaluates an upstream source clot or associated
  complication is context, not an applicable modality for the named embolus/thrombus unless it
  directly images that modeled finding.
- `subspecialties` are reading-workflow labels. Body region, organ membership, possible cause,
  patient population, and downstream workup are not enough. Leave `subspecialties=null` when no
  workflow is directly supported.
- Prefer the workflow that best matches the modeled finding. Do not add a regional subspecialty
  merely because the anatomy lies in that region when a more specific workflow, such as vascular,
  cardiac, oncologic, pediatric, quality/safety, or molecular imaging, is the actual reading
  workflow. Do not add `ER` only because the condition can be urgent; use `ER` when the modeled
  finding is explicitly acute, traumatic, emergency-care focused, or normally handled as an
  emergency imaging workflow.
- Vessel-centered findings use `VA` when the modeled finding is primarily about a vessel, flow,
  aneurysm, thrombus, embolus, dissection, stenosis, or vascular device. Do not also add `CH`,
  `GI`, `GU`, `HN`, `MK`, or another regional label solely because that vessel is located in the
  chest, abdomen, pelvis, head/neck, or extremities. An organ phrase in the definition, such as
  "artery in the lungs" or "vessel in the abdomen", is anatomy context, not regional workflow
  support.
- For named-vessel concepts, the vessel controls workflow: pulmonary artery, aorta, coronary,
  carotid, renal artery, portal vein, peripheral artery, and peripheral vein findings are vascular
  workflow unless the modeled finding also names a separate nonvascular thoracic, abdominal,
  neuro, or musculoskeletal disease being evaluated.
- Subspecialty code meanings:
  `BR` breast; `CA` cardiac; `CH` chest; `ER` emergency/acute care; `GI` gastrointestinal;
  `GU` genitourinary; `HN` head/neck; `IR` interventional; `MI` molecular/PET; `MK`
  musculoskeletal; `NM` nuclear medicine; `NR` neuroradiology; `OB` obstetric/gynecologic; `OI`
  oncologic imaging; `PD` pediatric; `SQ` quality/safety/technique; `VA` vascular.
  Use a code only when that meaning matches the modeled finding workflow.
- `MK` is not a default for every bone. Craniofacial, skull-base, jaw, orbit, or other head/neck
  bony findings are `HN` or `NR` workflow when the modeled finding is mainly head/neck or
  neuroanatomic rather than appendicular/axial musculoskeletal imaging.
- Use `SQ` only for acquisition, artifact, quality, safety, dose, report-quality, or technique
  problems. Ordinary presence, absent, indeterminate, unknown, and change-from-prior attributes do
  not support `SQ`.
- Compact examples: malignancy/staging/PET can support `OI`/`MI`; artifact or acquisition problem
  can support `SQ`; fetal/neonatal/child-specific findings can support `PD`; generic trauma
  fracture supports `ER`/`MK`; mammographic breast calcification supports `BR`/`MG`; a generic
  postoperative-state model with a "surgical clips present" attribute does not by itself support
  every modality that could see clips; `radiolucent urinary calculus` supports `CT`/`US`, not `XR`;
  a chest radiograph can show indirect signs of some vascular diagnoses, but indirect signs do not
  make `XR` an applicable modality for the diagnosis itself; ultrasound for an upstream clot source
  does not make the downstream embolus diagnosis a US finding; a pulmonary-artery embolus is a
  vascular workflow finding with `VA` and `CT`, not `CH`, `XR`, `US`, `MR`, `NM`, or `DSA` by
  default.
- Craniofacial or head/neck bony morphology findings are usually `HN` workflow with `XR` and/or
  `CT`. Do not add `MG`, `MR`, `PET`, `NM`, `US`, `RF`, or `DSA` unless the source explicitly
  makes that modality part of the modeled finding.
- For embolus/thrombus models, include only modalities that directly demonstrate the embolus or
  thrombus at the modeled site. Do not include radiography for secondary signs, or ultrasound for a
  possible source clot, when the modeled finding is not the source clot itself. Ultrasound supports
  a thrombus/embolus model only when the named clot itself is directly evaluated by ultrasound at
  the modeled site.
- For embolus/thrombus models, choose the dominant routine direct imaging modality for the named
  site. Do not list every possible vascular test.
- Every non-null `subspecialties` and `applicable_modalities` value must have direct support from
  the modeled finding.
- Final exclusion checklist before output:
  - remove any modality supported only by a tag, indirect sign, screening context, possible source
    lesion, adjacent cause, complication, or downstream workup;
  - remove any regional subspecialty supported only by the body region or containing organ;
  - for vessel-centered models, keep vascular workflow and direct clot/vessel imaging modalities;
    remove regional workflow labels and indirect/source-workup modalities unless the modeled finding
    separately names that nonvascular workflow.
- `clear_fields` entries must be metadata field names such as `subspecialties` or
  `applicable_modalities`, never enum values or candidate IDs.
- Never put `field_confidence` or `clear_fields` inside `clear_fields`.
- If `field_confidence` is present, keys must be real metadata fields and values must be
  numeric scores from 0 to 1.""",
    )


def create_ontology_decision_agent(model: Model | None = None) -> Agent[None, OntologyDecision]:
    """Create the focused ontology-candidate decision agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, OntologyDecision](
        model=resolved_model,
        output_type=OntologyDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions="""You decide ontology-code applicability for one radiology finding model.

Output ontology candidate decisions only. Do not assign anatomy or other metadata fields.

Rules:
- Use only offered candidate IDs.
- Mark a candidate canonical only when it is an exact match or clinically substitutable for the
  modeled finding.
- If multiple candidates from different ontology systems are exact matches or clinically
  substitutable for the modeled finding, they may all be canonical. Do not reject an exact candidate
  solely because another exact candidate already represents the same concept.
- Do not mark a candidate canonical when it adds detail not present in the modeled finding, including
  material, device subtype, graft, drug-eluting, named-vessel, location, pattern, or disease-context
  qualifiers.
- Do not mark a candidate canonical when it drops an explicit patient, fetal/neonatal/pediatric,
  pregnancy, timing, stage, severity, location, or modality qualifier from the modeled finding. If an
  exact authored/source code exists with that qualifier, broader codes without the qualifier are not
  clinically substitutable.
- A candidate whose display is a strict, broader subset of the modeled finding text is not
  canonical when the omitted words change the clinical scope.
- A specific pathologic subtype is narrower than a generic lesion, mass, tumor-like growth, or
  abnormality. Do not select a subtype as canonical merely because it can be one example of the
  broader modeled finding.
- For a generic lesion or tumor-like finding, reject candidates that name one subtype, morphology, or
  cystic form unless the modeled source itself includes that subtype.
- For an unqualified tumor or neoplasm concept, reject candidates that add benign, malignant,
  metastatic, premalignant, histologic-subtype, or grade qualifiers unless the modeled source itself
  includes that qualifier. An unqualified tumor concept can include both benign and malignant
  diseases; a malignant-only or benign-only candidate is narrower.
- If the model already has an authored exact code for a broad generic concept, do not add narrower
  examples, subtypes, complications, or diseases merely because they are clinically plausible forms
  of that broad concept. Keep those as review evidence.
- A broader device concept is not clinically substitutable when it drops a meaningful qualifier. For
  example, `vascular stent` is broader than `arterial stent` because it drops the arterial qualifier.
- Preserve explicit stage, temporal, severity, laterality, modality, and composition qualifiers in
  the modeled finding. A candidate that drops a meaningful qualifier such as `early`, `acute`,
  `chronic`, `radiolucent`, or `left/right` is usually broader, not clinically substitutable.
- Preserve focality/pattern qualifiers such as `cluster`, `diffuse`, `focal`, and `multiple`. A
  candidate that drops the focality/pattern qualifier is usually broader even when it names the same
  organ and material.
- A candidate that is only an anatomy, body part, organ, vessel group, imaging view, measurement
  component, or observation target is not canonical for an abnormality/finding model, even if the
  words overlap exactly. Keep it as review evidence instead.
- For measurement, score, classification, or assessment models, a candidate that names only the
  target anatomy is not canonical unless it also represents the measurement, score, classification,
  or assessment itself. Anatomy-only target concepts belong in review evidence and anatomy
  selection, not model-level `index_codes`.
- For measurement, score, classification, or assessment models, do not select a disease or
  abnormality concept that can be assessed by the model unless it represents the measurement,
  score, classification, or assessment itself.
- A measurement or assessment used to evaluate a disease is not the disease. For example, a global
  brain-volume assessment is not canonicalized as brain atrophy unless the model itself is brain
  atrophy.
- Do not mark a candidate canonical if it does not capture the modeled measurement/finding.
- `clinically_substitutable` is only for concepts that can stand in for the modeled finding itself.
  Do not use it for an interpretive counterpart, downstream diagnosis, clinical correlate, target
  condition, possible result, or disease that the model can help evaluate.
- Broader, narrower, merely related, complication, history, procedure, exam, and modality-specific
  concepts should remain review candidates unless they truly match the whole modeled finding.
- Do not prefer SNOMEDCT, RadLex, LOINC, GAMUTS, or any other ontology system by name. Judge concept
  meaning and evidence quality.
- Preserve useful non-canonical concepts as review evidence with a relationship and rejection
  reason when supported.
- Implanted device findings are not procedure concepts; postoperative states are not procedure
  concepts unless the authored finding is the procedure itself.
- Examples:
  - `early intrauterine pregnancy`: select an exact `early intrauterine pregnancy` concept; reject
    `intrauterine pregnancy` and `early stage of pregnancy` as broader because each drops a qualifier.
  - `breast calcification cluster`: select an exact `calcification cluster` concept; reject generic
    `mammographic calcification of breast` and `microcalcifications of the breast` unless the modeled
    finding is generic breast calcification/microcalcification rather than a cluster. Do not treat
    generic breast microcalcification terms as canonical for a cluster finding because they drop the
    cluster qualifier.
  - `benign synovial lesion`: keep an exact generic lesion code if available; reject `synovial cyst`
    unless the source says the lesion is a cyst.
  - `renal tumor`: select an exact generic tumor concept; reject `malignant renal neoplasm` as
    narrower because the source does not say malignant.
  - `arterial stent`: select exact `arterial stent`; reject `vascular stent` as broader and reject
    material/subtype concepts unless the source states the material/subtype.
  - Generic `arterial stent`: select exact generic `arterial stent`; reject `metal arterial stent`,
    `bioabsorbable arterial stent`, `drug-eluting arterial stent`, and stent-graft concepts unless
    the modeled finding explicitly states that material/subtype.
  - `fetal hepatosplenomegaly`: keep the exact fetal concept; reject generic `hepatosplenomegaly`
    concepts as broader because they drop the fetal qualifier.
- If no candidate is good enough for canonical selection, return no canonical candidate decisions.""",
    )


def create_anatomy_decision_agent(model: Model | None = None) -> Agent[None, AnatomyDecision]:
    """Create the focused anatomic-candidate decision agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, AnatomyDecision](
        model=resolved_model,
        output_type=AnatomyDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions="""You decide the top-level anatomic scope for one radiology finding model.

Output only `anatomic_decisions` and `body_regions`.

Selection contract:
- Use only offered candidate IDs.
- Select the smallest candidate set that covers the modeled anatomic scope. Do not select every
  possible site for one instance of the finding.
- The model name, description, exact ontology labels, and attributes define scope. Candidate
  `support_level`, `current_metadata`, and `default_selected` are evidence, not commands.
- A candidate is selectable only when its label preserves the modeled scope. Reject labels that add
  unsupported sex, accessory/variant, quadrant/segment, named-vessel/branch, side, lobe, endpoint,
  or other locality specificity.
- Reject candidates that narrow a generic regional mass, lesion, swelling, or soft-tissue
  abnormality to one tissue or structure such as a lymph node, vessel, nerve, or muscle unless the
  modeled finding itself commits to that structure. Conditional phrases such as "if lymph nodes" or
  "if vascular" are not scope.
- Location attribute values are choices within the model, not separate top-level anatomy. If one
  parent, tract, organ set, or system candidate covers those values, select that parent alone and
  mark the child/example candidates false.
- If no offered candidate covers the supported scope, select no anatomic candidate rather than using
  a narrower child, variant, landmark, or search hit as a proxy. Still assign `body_regions`.
- Device, tube, and catheter models are about placement or course anatomy when that is the modeled
  finding. Prefer a supported placement/course candidate over a broader containing region or endpoint
  organ. For placement/course candidates, support can come from the device/course concept and
  candidate evidence; do not reject a more specific course location merely because every locality
  word is not repeated verbatim in the model description. Existing/default parent metadata does not
  outrank supported placement/course anatomy.
- When a device/course candidate and its broader parent both have source support, choose the more
  specific course candidate if it remains a normal course/placement scope for the modeled device.
  The broader parent is context.
- If a device/course candidate is source-supported and lists an existing broader location in
  `broader_candidate_ids`, select the source-supported course candidate and reject the broader
  existing parent. The parent is body-region context, not the final anatomic selection.
- Select both a parent and child only when they are two distinct modeled scopes.

Examples:
- `diffuse breast parenchymal enhancement` supports breast anatomy. If the offered candidates are
  `female breast`, `accessory breast`, and `upper outer quadrant of breast`, select no anatomic
  candidate because each label adds unsupported specificity; set `body_regions` to `breast`.
- `peripheral arterial calcification` with a Location attribute listing femoral, popliteal, and
  tibial arteries supports the arterial system as the model scope. If `arterial system` is offered,
  select it alone; do not also select the named arteries.

Body regions:
- Assign `body_regions` from the selected anatomy and modeled scope, not from every possible
  per-instance site.
- Breast maps to `breast`; shoulder to `upper_extremity`; ovary/uterus/adnexa/prostate to `pelvis`;
  ribs/chest wall to `chest`; orbit/eye/lacrimal anatomy to `head`.
- Urinary tract calculus models centered on kidney, ureter, renal pelvis, or collecting system map
  to `abdomen` unless the modeled scope is bladder, urethra, pelvic organ, or explicitly lower
  pelvic anatomy.
- True system-level or nonlocalized anatomy such as `arterial system`, `vascular system`, `whole
  body`, or generic nonlocalized device/artifact maps to `whole_body`.
- If the selected anatomy is `arterial system`, `vascular system`, or another whole-system anatomy,
  body_regions must be `["whole_body"]`, not a list of the possible regions where examples can
  occur. This remains true even if rejected child/example candidates name vessels in the head,
  chest, abdomen, or extremities.
- If no anatomy is selected because the finding is generic, unlocalized, or site-variable, use
  `whole_body` unless the source clearly supports a narrower body region.
- Do not emit multiple body regions solely because candidate labels cross boundaries. Use multiple
  regions only when the modeled finding itself spans multiple primary regions.""",
    )


_DEFAULT_CREATE_METADATA_ASSIGNMENT_AGENT = create_metadata_assignment_agent


def _attribute_summary(model: FindingModelFull) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for attribute in model.attributes:
        item: dict[str, Any] = {
            "name": attribute.name,
            "description": attribute.description,
            "required": attribute.required,
            "type": attribute.type.value if hasattr(attribute.type, "value") else str(attribute.type),
        }
        values = getattr(attribute, "values", None)
        if values is not None:
            item["values"] = [value.name for value in values]
        summary.append(item)
    return summary


def _attribute_labels(model: FindingModelFull) -> list[str]:
    labels: list[str] = []
    for attribute in model.attributes:
        labels.append(attribute.name)
        values = getattr(attribute, "values", None)
        if values is not None:
            labels.extend(value.name for value in values)
    return labels


def _compact_model_context(model: FindingModelFull) -> dict[str, Any]:
    return {
        "oifm_id": model.oifm_id,
        "name": model.name,
        "description": model.description,
        "synonyms": list(model.synonyms or []),
        "tags": list(model.tags or []),
        "existing_structured_metadata": {
            "body_regions": [value.value for value in model.body_regions] if model.body_regions else None,
            "subspecialties": [value.value for value in model.subspecialties] if model.subspecialties else None,
            "etiologies": [value.value for value in model.etiologies] if model.etiologies else None,
            "entity_type": model.entity_type.value if model.entity_type else None,
            "applicable_modalities": (
                [value.value for value in model.applicable_modalities] if model.applicable_modalities else None
            ),
            "expected_time_course": (
                model.expected_time_course.model_dump(mode="json") if model.expected_time_course else None
            ),
            "age_profile": model.age_profile.model_dump(mode="json") if model.age_profile else None,
            "sex_specificity": model.sex_specificity.value if model.sex_specificity else None,
            "index_codes": [code.model_dump(mode="json") for code in model.index_codes or []],
            "anatomic_locations": [code.model_dump(mode="json") for code in model.anatomic_locations or []],
        },
        "attributes": _attribute_summary(model),
    }


def _ontology_candidate_states(result: CategorizedOntologyConcepts) -> dict[str, _OntologyCandidateState]:
    states: dict[str, _OntologyCandidateState] = {}

    def add_candidates(
        candidates: list[OntologySearchResult],
        *,
        relationship: OntologyCandidateRelationship,
        selected_as_canonical: bool,
        source_bucket: str,
    ) -> None:
        for candidate in candidates:
            idx_code = candidate.as_index_code()
            state_key = f"{idx_code.system}:{candidate.concept_id}"
            # Keep the first-seen bucket/relationship when duplicates appear across categories.
            if state_key not in states:
                states[state_key] = _OntologyCandidateState(
                    result=candidate,
                    relationship=relationship,
                    selected_as_canonical=selected_as_canonical,
                    source_bucket=source_bucket,
                )

    add_candidates(
        result.exact_matches,
        relationship=OntologyCandidateRelationship.EXACT_MATCH,
        selected_as_canonical=True,
        source_bucket="exact_matches",
    )
    add_candidates(
        result.should_include,
        relationship=OntologyCandidateRelationship.RELATED,
        selected_as_canonical=False,
        source_bucket="should_include",
    )
    add_candidates(
        result.marginal_concepts,
        relationship=OntologyCandidateRelationship.RELATED,
        selected_as_canonical=False,
        source_bucket="marginal",
    )
    return states


def _anatomic_candidate_states(
    result: AnatomicCandidateSearchResponse | LocationSearchResponse | None,
) -> dict[str, _AnatomicCandidateState]:
    states: dict[str, _AnatomicCandidateState] = {}
    if result is None:
        return states

    if isinstance(result, AnatomicCandidateSearchResponse):
        for candidate in result.candidates:
            idx_code = candidate.location.as_index_code()
            state_key = f"{idx_code.system}:{candidate.location.concept_id}"
            if state_key not in states:
                states[state_key] = _AnatomicCandidateState(
                    result=candidate.location,
                    selected=False,
                    source_bucket="candidate",
                    support_level=candidate.support_level,
                    matched_terms=candidate.matched_terms,
                    broader_candidate_ids=candidate.broader_candidate_ids,
                )
        return states

    candidates = [("primary", result.primary_location, result.primary_location.concept_id != "NO_RESULTS")]
    candidates.extend(("alternate", candidate, False) for candidate in result.alternate_locations)
    for source_bucket, candidate, selected in candidates:
        if candidate.concept_id == "NO_RESULTS":
            continue
        idx_code = candidate.as_index_code()
        state_key = f"{idx_code.system}:{candidate.concept_id}"
        if state_key not in states:
            states[state_key] = _AnatomicCandidateState(
                result=candidate,
                selected=selected,
                source_bucket=source_bucket,
                support_level=source_bucket,
            )
    return states


def _filter_anatomic_ontology_states(
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
) -> None:
    anatomic_displays = {
        state.result.as_index_code().display.strip().lower()
        for state in anatomic_states.values()
        if state.result.as_index_code().display
    }
    for key, state in list(ontology_states.items()):
        display = (state.result.as_index_code().display or "").strip().lower()
        if display and display in anatomic_displays:
            del ontology_states[key]


def _compact_ontology_candidates(states: dict[str, _OntologyCandidateState]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": key,
            "text": state.result.concept_text,
            "display": state.result.as_index_code().display,
            "table_name": state.result.table_name,
            "system": state.result.as_index_code().system,
            "source_bucket": state.source_bucket,
            "default_relationship": state.relationship.value,
            "default_selected_as_canonical": state.selected_as_canonical,
        }
        for key, state in states.items()
    ]


def _ontology_candidate_prompt_states(
    states: dict[str, _OntologyCandidateState],
    *,
    limit: int | None = None,
) -> dict[str, _OntologyCandidateState]:
    """Return the highest-evidence ontology candidates to keep LLM decision prompts bounded."""
    limit = limit or settings.metadata_candidate_decision_limit
    ranked = sorted(
        states.items(),
        key=lambda item: (
            0 if item[1].selected_as_canonical else 1,
            _ONTOLOGY_SOURCE_ORDER.get(item[1].source_bucket, 99),
            item[1].result.as_index_code().display or item[1].result.concept_text,
            item[0],
        ),
    )
    return dict(ranked[:limit])


def _compact_anatomic_candidates(states: dict[str, _AnatomicCandidateState]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": key,
            "text": state.result.concept_text,
            "display": state.result.as_index_code().display,
            "source_bucket": state.source_bucket,
            "support_level": state.support_level,
            "matched_terms": state.matched_terms,
            "broader_candidate_ids": state.broader_candidate_ids,
            "default_selected": state.selected,
        }
        for key, state in states.items()
    ]


def _anatomic_candidate_prompt_states(
    states: dict[str, _AnatomicCandidateState],
    *,
    limit: int | None = None,
) -> dict[str, _AnatomicCandidateState]:
    """Return the highest-evidence anatomy candidates to keep LLM decision prompts bounded."""
    limit = limit or settings.metadata_candidate_decision_limit
    ranked = sorted(
        states.items(),
        key=lambda item: (
            _ANATOMIC_SUPPORT_ORDER.get(item[1].support_level or "search_only", 99),
            0 if item[1].selected else 1,
            item[1].result.as_index_code().display or item[1].result.concept_text,
            item[0],
        ),
    )
    return dict(ranked[:limit])


def _bounded_candidate_prompt_payload(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    fill_blanks_only: bool,
) -> dict[str, Any]:
    return _agent_payload(
        model,
        _ontology_candidate_prompt_states(ontology_states),
        _anatomic_candidate_prompt_states(anatomic_states),
        fill_blanks_only=fill_blanks_only,
    )


def _decision_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    fill_blanks_only: bool,
) -> str:
    assignment_mode = "fill_blanks_only" if fill_blanks_only else "reassess"
    blank_structured_fields = [
        field_name for field_name in STRUCTURED_METADATA_FIELDS if getattr(model, field_name) is None
    ]
    locked_structured_fields = [
        field_name for field_name in STRUCTURED_METADATA_FIELDS if getattr(model, field_name) is not None
    ]
    blank_required_fields = [
        field_name for field_name in ("body_regions", "entity_type", "applicable_modalities")
        if getattr(model, field_name) is None
    ]
    mode_guidance = (
        "Only populate fields that are currently blank or empty. Do not try to clear or overwrite "
        "already-populated fields. Use the locked fields as context, and fill every blank field that is "
        "clearly supported by the finding and candidate evidence."
        if fill_blanks_only
        else "Reassess all structured metadata fields. If the existing value is wrong or incomplete, "
        "replace it with the best supported value."
    )
    payload = {
        "assignment_mode": assignment_mode,
        "mode_context": {
            "blank_structured_fields": blank_structured_fields,
            "locked_structured_fields": locked_structured_fields,
            "blank_required_fields": blank_required_fields,
            "required_structured_fields": list(REQUIRED_METADATA_FIELDS),
        },
        "finding_model": _compact_model_context(model),
        "ontology_candidates": _compact_ontology_candidates(ontology_states),
        "anatomic_candidates": _compact_anatomic_candidates(anatomic_states),
    }
    return (
        "Review this finding model and candidate evidence. Decide only the canonical structured metadata "
        "and candidate selections that are justified.\n"
        f"Mode: {assignment_mode}. {mode_guidance}\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _agent_payload(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    fill_blanks_only: bool,
) -> dict[str, Any]:
    assignment_mode = "fill_blanks_only" if fill_blanks_only else "reassess"
    return {
        "assignment_mode": assignment_mode,
        "finding_model": _compact_model_context(model),
        "ontology_candidates": _compact_ontology_candidates(ontology_states),
        "anatomic_candidates": _compact_anatomic_candidates(anatomic_states),
    }


def _ontology_decision_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    *,
    fill_blanks_only: bool,
) -> str:
    payload = _agent_payload(
        model,
        _ontology_candidate_prompt_states(ontology_states),
        {},
        fill_blanks_only=fill_blanks_only,
    )
    payload["task"] = "Decide ontology candidate relationships and canonical index-code selection."
    return json.dumps(payload, indent=2)


def _anatomy_decision_prompt(
    model: FindingModelFull,
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    fill_blanks_only: bool,
) -> str:
    payload = _agent_payload(model, {}, _anatomic_candidate_prompt_states(anatomic_states), fill_blanks_only=fill_blanks_only)
    payload["task"] = "Decide anatomic candidate selection and body_regions."
    return json.dumps(payload, indent=2)


def _identity_decision_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    fill_blanks_only: bool,
) -> str:
    payload = _bounded_candidate_prompt_payload(
        model,
        ontology_states,
        anatomic_states,
        fill_blanks_only=fill_blanks_only,
    )
    payload["task"] = "Assign only entity_type and expected_time_course."
    return json.dumps(payload, indent=2)


def _etiology_decision_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    identity: IdentityDecision,
    fill_blanks_only: bool,
) -> str:
    payload = _bounded_candidate_prompt_payload(
        model,
        ontology_states,
        anatomic_states,
        fill_blanks_only=fill_blanks_only,
    )
    payload["finding_model"]["tags"] = []
    payload["finding_model"]["existing_structured_metadata"] = {
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
    }
    payload["ontology_candidates"] = [
        candidate
        for candidate in payload["ontology_candidates"]
        if candidate.get("default_selected_as_canonical") is True
        and candidate.get("source_bucket") in {"existing_index_codes", "exact_matches"}
    ]
    payload["anatomic_candidates"] = []
    payload["task"] = "Assign only etiologies."
    payload["identity_context"] = {
        "entity_type": identity.entity_type.value if identity.entity_type is not None else None,
        "expected_time_course": (
            identity.expected_time_course.model_dump(mode="json")
            if identity.expected_time_course is not None
            else None
        ),
    }
    return json.dumps(payload, indent=2)


def _patient_applicability_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    fill_blanks_only: bool,
) -> str:
    payload = _bounded_candidate_prompt_payload(
        model,
        ontology_states,
        anatomic_states,
        fill_blanks_only=fill_blanks_only,
    )
    payload["task"] = "Assign only age_profile and sex_specificity."
    return json.dumps(payload, indent=2)


def _imaging_workflow_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    anatomy: AnatomyDecision,
    fill_blanks_only: bool,
) -> str:
    payload = _bounded_candidate_prompt_payload(
        model,
        ontology_states,
        anatomic_states,
        fill_blanks_only=fill_blanks_only,
    )
    payload["task"] = "Assign only subspecialties and applicable_modalities."
    payload["finding_model"]["tags"] = []
    payload["ontology_candidates"] = [
        candidate
        for candidate in payload["ontology_candidates"]
        if candidate.get("default_selected_as_canonical") is True
    ]
    payload["anatomic_candidates"] = [
        candidate for candidate in payload["anatomic_candidates"] if candidate.get("default_selected") is True
    ]
    existing_metadata = payload["finding_model"]["existing_structured_metadata"]
    payload["workflow_values_under_review"] = {
        "subspecialties": existing_metadata["subspecialties"],
        "applicable_modalities": existing_metadata["applicable_modalities"],
    }
    existing_metadata["subspecialties"] = None
    existing_metadata["applicable_modalities"] = None
    payload["anatomy_context"] = {
        "body_regions": [value.value for value in anatomy.body_regions] if anatomy.body_regions else None,
    }
    return json.dumps(payload, indent=2)


def _apply_ontology_decisions(
    states: dict[str, _OntologyCandidateState],
    decisions: list[OntologyCandidateDecision],
    warnings: list[str],
) -> None:
    for decision in decisions:
        state = states.get(decision.candidate_id)
        if state is None:
            warnings.append(f"Classifier referenced unknown ontology candidate: {decision.candidate_id}")
            continue
        state.relationship = decision.relationship
        state.selected_as_canonical = decision.selected_as_canonical
        state.rationale = getattr(decision, "rationale", None)
        state.rejection_reason = decision.rejection_reason
        if (
            not state.selected_as_canonical
            and state.relationship
            in {
                OntologyCandidateRelationship.EXACT_MATCH,
                OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE,
            }
            and state.rejection_reason is None
        ):
            state.selected_as_canonical = True
        if state.selected_as_canonical and state.relationship not in {
            OntologyCandidateRelationship.EXACT_MATCH,
            OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE,
        }:
            warnings.append(
                "Ignoring canonical ontology selection for "
                f"{decision.candidate_id} because relationship {state.relationship.value} is not canonical"
            )
            state.selected_as_canonical = False
        if decision.rejection_reason is not None and state.selected_as_canonical:
            warnings.append(f"Ignoring rejection reason for canonical ontology candidate {decision.candidate_id}")
            state.rejection_reason = None


def _apply_anatomic_decisions(
    states: dict[str, _AnatomicCandidateState],
    decisions: list[AnatomicCandidateDecision],
    warnings: list[str],
) -> None:
    for decision in decisions:
        state = states.get(decision.candidate_id)
        if state is None:
            warnings.append(f"Classifier referenced unknown anatomic candidate: {decision.candidate_id}")
            continue
        state.selected = decision.selected
        state.rationale = getattr(decision, "rationale", None)


def _append_source_support_level_consistency_warnings(
    states: dict[str, _AnatomicCandidateState],
    warnings: list[str],
) -> None:
    by_concept_id = {state.result.concept_id: state for state in states.values()}
    strongly_supported = {"direct_source", "source_inferred_query", "ontology_context"}
    selected_levels_to_check = {"child_of_supported", "current_metadata", "search_only"}
    for state in states.values():
        if not state.selected or state.support_level not in selected_levels_to_check:
            continue
        for broader_id in state.broader_candidate_ids:
            broader = by_concept_id.get(broader_id)
            if broader is None or broader.support_level not in strongly_supported:
                continue
            selected_label = state.result.as_index_code().display or state.result.concept_text
            broader_label = broader.result.as_index_code().display or broader.result.concept_text
            warnings.append(
                "source support level consistency check: "
                f"selected '{selected_label}' ({state.support_level}) while broader candidate "
                f"'{broader_label}' had support_level={broader.support_level}"
            )
            break


def _dedupe_index_codes(codes: list[IndexCode]) -> list[IndexCode]:
    seen: set[tuple[str, str]] = set()
    deduped: list[IndexCode] = []
    for code in codes:
        key = (code.system, code.code)
        if key not in seen:
            seen.add(key)
            deduped.append(code)
    return deduped


def _ontology_report(states: dict[str, _OntologyCandidateState]) -> OntologyCandidateReport:
    canonical_codes: list[OntologyCandidate] = []
    review_candidates: list[OntologyCandidate] = []

    for state in states.values():
        candidate = OntologyCandidate(
            code=state.result.as_index_code(),
            relationship=state.relationship,
            rationale=state.rationale,
            rejection_reason=state.rejection_reason
            or _default_rejection_reason(state.relationship, state.selected_as_canonical),
        )
        if state.selected_as_canonical:
            canonical_codes.append(candidate)
        else:
            review_candidates.append(candidate)

    return OntologyCandidateReport(canonical_codes=canonical_codes, review_candidates=review_candidates)


def _anatomic_review(states: dict[str, _AnatomicCandidateState]) -> list[AnatomicCandidate]:
    return [
        AnatomicCandidate(
            location=state.result.as_index_code(),
            selected=state.selected,
            rationale=state.rationale,
            support_level=state.support_level,
            matched_terms=state.matched_terms,
            broader_candidate_ids=state.broader_candidate_ids,
        )
        for state in states.values()
    ]


def _selected_anatomic_locations(states: dict[str, _AnatomicCandidateState]) -> list[IndexCode]:
    return _dedupe_index_codes([state.result.as_index_code() for state in states.values() if state.selected])


def _record_ontology_cache(
    cache: OntologyLookupCache | None,
    states: dict[str, _OntologyCandidateState],
) -> None:
    if cache is None:
        return
    for state in states.values():
        usage: OntologyEvidenceUsage
        if state.selected_as_canonical:
            usage = "canonical_selected"
        elif state.rejection_reason is not None:
            usage = "rejected_candidate"
        else:
            usage = "related_candidate"
        cache.record_ontology_result(
            state.result,
            usage=usage,
            query=state.result.concept_text,
            relationship=state.relationship.value,
            rejection_reason=state.rejection_reason.value if state.rejection_reason is not None else None,
        )


STRUCTURED_METADATA_FIELDS = (
    "body_regions",
    "subspecialties",
    "etiologies",
    "entity_type",
    "applicable_modalities",
    "expected_time_course",
    "age_profile",
    "sex_specificity",
)

REQUIRED_METADATA_FIELDS = ("body_regions", "entity_type", "applicable_modalities")

CONFIDENCE_FIELDS: tuple[ConfidenceFieldKey, ...] = (
    "body_regions",
    "subspecialties",
    "etiologies",
    "entity_type",
    "applicable_modalities",
    "expected_time_course",
    "age_profile",
    "sex_specificity",
    "anatomic_locations",
    "index_codes",
)

CLEARABLE_FIELDS = {*CONFIDENCE_FIELDS}
CANDIDATE_METADATA_FIELDS = {"index_codes", "anatomic_locations"}
OPTIONAL_IDENTITY_FIELDS = {"etiologies", "expected_time_course"}
LOW_CONFIDENCE_OPTIONAL_FIELDS = {
    "subspecialties",
    "etiologies",
    "expected_time_course",
    "age_profile",
    "sex_specificity",
}
HIGH_CONFIDENCE_OPTIONAL_FIELDS = {"expected_time_course"}
ENTITY_TYPES_ALLOW_IDENTITY_CLEAR = {
    EntityType.MEASUREMENT,
    EntityType.ASSESSMENT,
    EntityType.RECOMMENDATION,
    EntityType.TECHNIQUE_ISSUE,
    EntityType.GROUPING,
}


def _default_rejection_reason(
    relationship: OntologyCandidateRelationship, selected_as_canonical: bool
) -> OntologyCandidateRejectionReason | None:
    if selected_as_canonical:
        return None
    return {
        OntologyCandidateRelationship.NARROWER: OntologyCandidateRejectionReason.TOO_SPECIFIC,
        OntologyCandidateRelationship.BROADER: OntologyCandidateRejectionReason.TOO_BROAD,
        OntologyCandidateRelationship.RELATED: OntologyCandidateRejectionReason.OVERLAPPING_SCOPE,
        OntologyCandidateRelationship.COMPLICATION: OntologyCandidateRejectionReason.TOO_SPECIFIC,
    }.get(relationship)


def _project_structured_field_value(
    finding_model: FindingModelFull,
    decision: MetadataAssignmentDecision,
    field_name: str,
    *,
    fill_blanks_only: bool,
) -> object:
    existing_value = getattr(finding_model, field_name)
    decision_value = getattr(decision, field_name)

    if fill_blanks_only:
        if existing_value is None and decision_value is not None:
            return decision_value
        return existing_value

    if decision_value is not None:
        return decision_value
    if field_name in decision.clear_fields:
        if field_name in REQUIRED_METADATA_FIELDS:
            return existing_value
        return None
    return existing_value


async def _gather_ontology_candidates(
    finding_model: FindingModelFull,
    *,
    warnings: list[str],
) -> tuple[CategorizedOntologyConcepts, float]:
    ontology_result = CategorizedOntologyConcepts(
        exact_matches=[],
        should_include=[],
        marginal_concepts=[],
        search_summary="",
        excluded_anatomical=[],
    )
    start = perf_counter()
    with logfire.span("assign_metadata.ontology_candidates", finding_name=finding_model.name):
        try:
            ontology_result = await match_ontology_concepts(
                finding_name=finding_model.name,
                finding_description=finding_model.description,
                exclude_anatomical=True,
            )
            logfire.info(
                "Ontology candidate gathering complete",
                exact_matches=len(ontology_result.exact_matches),
                should_include=len(ontology_result.should_include),
                marginal=len(ontology_result.marginal_concepts),
            )
        except Exception as exc:
            warning = f"Ontology candidate gathering failed: {exc}"
            warnings.append(warning)
            logger.exception(warning)
            logfire.warning("Ontology candidate gathering failed", error=str(exc))
    return ontology_result, perf_counter() - start


async def _gather_anatomic_candidates(
    finding_model: FindingModelFull,
    *,
    warnings: list[str],
    anatomic_index: AnatomicLocationIndex | None = None,
    anatomic_index_lock: asyncio.Lock | None = None,
    locality_labels: list[str] | None = None,
) -> tuple[AnatomicCandidateSearchResponse | LocationSearchResponse | None, float]:
    anatomic_result: AnatomicCandidateSearchResponse | LocationSearchResponse | None = None
    start = perf_counter()
    with logfire.span("assign_metadata.anatomic_candidates", finding_name=finding_model.name):
        try:
            if anatomic_index_lock is not None:
                async with anatomic_index_lock:
                    anatomic_result = await find_anatomic_locations(
                        finding_name=finding_model.name,
                        description=finding_model.description,
                        synonyms=list(finding_model.synonyms or []),
                        attribute_labels=_attribute_labels(finding_model),
                        locality_labels=locality_labels,
                        source_labels=_source_ontology_labels(finding_model),
                        index=anatomic_index,
                    )
            else:
                anatomic_result = await find_anatomic_locations(
                    finding_name=finding_model.name,
                    description=finding_model.description,
                    synonyms=list(finding_model.synonyms or []),
                    attribute_labels=_attribute_labels(finding_model),
                    locality_labels=locality_labels,
                    source_labels=_source_ontology_labels(finding_model),
                    index=anatomic_index,
                )
            candidate_count = (
                len(anatomic_result.candidates)
                if isinstance(anatomic_result, AnatomicCandidateSearchResponse)
                else len(anatomic_result.alternate_locations) + 1
            )
            logfire.info(
                "Anatomic candidate gathering complete",
                candidates=candidate_count,
            )
        except Exception as exc:
            warning = f"Anatomic candidate gathering failed: {exc}"
            warnings.append(warning)
            logger.exception(warning)
            logfire.warning("Anatomic candidate gathering failed", error=str(exc))
    return anatomic_result, perf_counter() - start


def _ontology_locality_labels(result: CategorizedOntologyConcepts | None) -> list[str]:
    if result is None:
        return []
    labels: list[str] = []
    for candidate in [*result.exact_matches, *result.should_include[:5]]:
        for label in (candidate.concept_text, candidate.as_index_code().display):
            if label and label not in labels:
                labels.append(label)
    return labels


def _source_ontology_labels(finding_model: FindingModelFull) -> list[str]:
    labels: list[str] = []
    for code in finding_model.index_codes or []:
        if code.display and code.display not in labels:
            labels.append(code.display)
    return labels


def _merge_existing_ontology_states(
    finding_model: FindingModelFull, states: dict[str, _OntologyCandidateState]
) -> None:
    for code in finding_model.index_codes or []:
        state_key = f"{code.system}:{code.code}"
        states[state_key] = _OntologyCandidateState(
            result=OntologySearchResult(
                concept_id=code.code,
                concept_text=code.display or code.code,
                score=0.0,
                table_name=code.system.lower(),
            ),
            relationship=OntologyCandidateRelationship.EXACT_MATCH,
            selected_as_canonical=True,
            source_bucket="existing_index_codes",
        )


def _merge_existing_anatomic_states(
    finding_model: FindingModelFull, states: dict[str, _AnatomicCandidateState]
) -> None:
    for code in finding_model.anatomic_locations or []:
        state_key = f"{code.system}:{code.code}"
        existing = states.get(state_key)
        if existing is not None:
            existing.selected = True
            existing.source_bucket = "existing_anatomic_locations"
            if existing.support_level in {None, "search_only"}:
                existing.support_level = "current_metadata"
            continue
        states[state_key] = _AnatomicCandidateState(
            result=OntologySearchResult(
                concept_id=code.code,
                concept_text=code.display or code.code,
                score=0.0,
                table_name="anatomic_locations",
            ),
            selected=True,
            source_bucket="existing_anatomic_locations",
            support_level="current_metadata",
        )


def _validate_fill_blanks_required_fields(
    finding_model: FindingModelFull, output: MetadataAssignmentDecision
) -> None:
    from pydantic_ai import ModelRetry

    blank_required = [field for field in REQUIRED_METADATA_FIELDS if getattr(finding_model, field) is None]
    missing_blank = [
        field_name
        for field_name in blank_required
        if _project_structured_field_value(
            finding_model,
            output,
            field_name,
            fill_blanks_only=True,
        )
        is None
    ]
    if missing_blank:
        raise ModelRetry(
            "Fill-blanks mode must populate every blank required field that remains empty: "
            + ", ".join(missing_blank)
        )


def _validate_reassess_required_fields(
    finding_model: FindingModelFull, output: MetadataAssignmentDecision
) -> None:
    from pydantic_ai import ModelRetry

    missing_after_reassess = [
        field_name
        for field_name in REQUIRED_METADATA_FIELDS
        if _project_structured_field_value(
            finding_model,
            output,
            field_name,
            fill_blanks_only=False,
        )
        is None
    ]
    if missing_after_reassess:
        raise ModelRetry(
            "Reassess mode cannot leave required fields empty after applying the decision: "
            + ", ".join(missing_after_reassess)
        )


def _validate_candidate_confidence_consistency(output: MetadataAssignmentDecision) -> None:
    from pydantic_ai import ModelRetry

    if output.entity_type in {EntityType.ASSESSMENT, EntityType.MEASUREMENT}:
        non_exact_canonical = [
            decision.candidate_id
            for decision in output.ontology_decisions
            if decision.selected_as_canonical
            and decision.relationship != OntologyCandidateRelationship.EXACT_MATCH
        ]
        if non_exact_canonical:
            raise ModelRetry(
                "Assessment and measurement models can only use exact canonical ontology concepts: "
                + ", ".join(non_exact_canonical)
            )


def _validate_classifier_output(
    finding_model: FindingModelFull,
    output: MetadataAssignmentDecision,
    *,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    fill_blanks_only: bool,
) -> MetadataAssignmentDecision:
    offered_ontology_ids = set(ontology_states.keys())
    output.ontology_decisions = [
        decision for decision in output.ontology_decisions if decision.candidate_id in offered_ontology_ids
    ]
    offered_anatomic_ids = set(anatomic_states.keys())
    output.anatomic_decisions = [
        decision for decision in output.anatomic_decisions if decision.candidate_id in offered_anatomic_ids
    ]

    if fill_blanks_only:
        _validate_fill_blanks_required_fields(finding_model, output)
    else:
        _validate_reassess_required_fields(finding_model, output)
    _validate_candidate_confidence_consistency(output)
    return output


def _validate_ontology_decision(
    output: OntologyDecision,
    *,
    ontology_states: dict[str, _OntologyCandidateState],
) -> OntologyDecision:
    offered_ontology_ids = set(ontology_states.keys())
    output.ontology_decisions = [
        decision for decision in output.ontology_decisions if decision.candidate_id in offered_ontology_ids
    ]
    return output


def _validate_anatomy_decision(
    output: AnatomyDecision,
    *,
    anatomic_states: dict[str, _AnatomicCandidateState],
) -> AnatomyDecision:
    offered_anatomic_ids = set(anatomic_states.keys())
    output.anatomic_decisions = [
        decision for decision in output.anatomic_decisions if decision.candidate_id in offered_anatomic_ids
    ]
    return output


def _validate_etiology_decision(
    output: EtiologyDecision,
    *,
    identity: IdentityDecision,
) -> EtiologyDecision:
    from pydantic_ai import ModelRetry

    if identity.entity_type in ENTITY_TYPES_ALLOW_IDENTITY_CLEAR and output.etiologies:
        raise ModelRetry(
            "Etiologies should be null for measurement, assessment, recommendation, "
            "technique_issue, and grouping outputs."
        )
    return output


def _merge_confidence(
    *confidence_maps: dict[ConfidenceFieldKey, FieldConfidenceScore],
) -> dict[ConfidenceFieldKey, FieldConfidenceScore]:
    merged: dict[ConfidenceFieldKey, FieldConfidenceScore] = {}
    for confidence_map in confidence_maps:
        merged.update(confidence_map)
    return merged


def _combine_focused_decisions(
    *,
    ontology: OntologyDecision,
    anatomy: AnatomyDecision,
    identity: IdentityDecision,
    etiology: EtiologyDecision,
    patient: PatientApplicabilityDecision,
    imaging_workflow: ImagingWorkflowDecision,
) -> MetadataAssignmentDecision:
    clear_fields = [
        *identity.clear_fields,
        *etiology.clear_fields,
        *patient.clear_fields,
        *imaging_workflow.clear_fields,
    ]
    return MetadataAssignmentDecision(
        body_regions=anatomy.body_regions,
        subspecialties=imaging_workflow.subspecialties,
        etiologies=etiology.etiologies,
        entity_type=identity.entity_type,
        applicable_modalities=imaging_workflow.applicable_modalities,
        expected_time_course=identity.expected_time_course,
        age_profile=patient.age_profile,
        sex_specificity=patient.sex_specificity,
        ontology_decisions=ontology.ontology_decisions,
        anatomic_decisions=anatomy.anatomic_decisions,
        clear_fields=clear_fields,
        classification_rationale="",
        field_confidence=_merge_confidence(
            identity.field_confidence,
            etiology.field_confidence,
            patient.field_confidence,
            imaging_workflow.field_confidence,
        ),
    )


async def _run_legacy_classifier(
    finding_model: FindingModelFull,
    *,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    fill_blanks_only: bool = False,
) -> tuple[MetadataAssignmentDecision, str, float]:
    """Run the pre-refactor aggregate classifier for compatibility with injected test agents."""
    model_used = settings.get_effective_model_string("metadata_assign")
    start = perf_counter()
    with logfire.span(
        "assign_metadata.legacy_classifier",
        finding_name=finding_model.name,
        ontology_candidates=len(ontology_states),
        anatomic_candidates=len(anatomic_states),
    ):
        agent = create_metadata_assignment_agent()

        @agent.output_validator
        def validate_decision(ctx: RunContext[None], output: MetadataAssignmentDecision) -> MetadataAssignmentDecision:
            _ = ctx
            return _validate_classifier_output(
                finding_model,
                output,
                ontology_states=ontology_states,
                anatomic_states=anatomic_states,
                fill_blanks_only=fill_blanks_only,
            )

        decision_result = await agent.run(
            _decision_prompt(
                finding_model,
                ontology_states,
                anatomic_states,
                fill_blanks_only=fill_blanks_only,
            )
        )
        decision = decision_result.output
    return decision, model_used, perf_counter() - start


async def _run_classifier(
    finding_model: FindingModelFull,
    *,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    fill_blanks_only: bool = False,
) -> tuple[MetadataAssignmentDecision, str, float]:
    if create_metadata_assignment_agent is not _DEFAULT_CREATE_METADATA_ASSIGNMENT_AGENT:
        return await _run_legacy_classifier(
            finding_model,
            ontology_states=ontology_states,
            anatomic_states=anatomic_states,
            fill_blanks_only=fill_blanks_only,
        )

    model_used = settings.get_effective_model_string("metadata_assign")
    start = perf_counter()
    with logfire.span(
        "assign_metadata.focused_decisions",
        finding_name=finding_model.name,
        ontology_candidates=len(ontology_states),
        anatomic_candidates=len(anatomic_states),
        ontology_prompt_candidates=min(len(ontology_states), settings.metadata_candidate_decision_limit),
        anatomic_prompt_candidates=min(len(anatomic_states), settings.metadata_candidate_decision_limit),
    ):
        ontology_agent = create_ontology_decision_agent()
        anatomy_agent = create_anatomy_decision_agent()

        @ontology_agent.output_validator
        def validate_ontology(ctx: RunContext[None], output: OntologyDecision) -> OntologyDecision:
            _ = ctx
            return _validate_ontology_decision(output, ontology_states=ontology_states)

        @anatomy_agent.output_validator
        def validate_anatomy(ctx: RunContext[None], output: AnatomyDecision) -> AnatomyDecision:
            _ = ctx
            return _validate_anatomy_decision(output, anatomic_states=anatomic_states)

        async def run_ontology_decision() -> AgentRunResult[OntologyDecision]:
            prompt_states = _ontology_candidate_prompt_states(ontology_states)
            with logfire.span(
                "assign_metadata.agent.ontology",
                finding_name=finding_model.name,
                candidates=len(prompt_states),
                gathered_candidates=len(ontology_states),
            ):
                return await ontology_agent.run(
                    _ontology_decision_prompt(
                        finding_model,
                        prompt_states,
                        fill_blanks_only=fill_blanks_only,
                    )
                )

        async def run_anatomy_decision() -> AgentRunResult[AnatomyDecision]:
            prompt_states = _anatomic_candidate_prompt_states(anatomic_states)
            with logfire.span(
                "assign_metadata.agent.anatomy",
                finding_name=finding_model.name,
                candidates=len(prompt_states),
                gathered_candidates=len(anatomic_states),
            ):
                return await anatomy_agent.run(
                    _anatomy_decision_prompt(
                        finding_model,
                        prompt_states,
                        fill_blanks_only=fill_blanks_only,
                    )
                )

        ontology_result, anatomy_result = await asyncio.gather(
            run_ontology_decision(),
            run_anatomy_decision(),
        )

        projected_ontology_states = {key: state.model_copy(deep=True) for key, state in ontology_states.items()}
        projected_anatomic_states = {key: state.model_copy(deep=True) for key, state in anatomic_states.items()}
        _apply_ontology_decisions(projected_ontology_states, ontology_result.output.ontology_decisions, [])
        _apply_anatomic_decisions(projected_anatomic_states, anatomy_result.output.anatomic_decisions, [])

        identity_agent = create_identity_assignment_agent()
        patient_agent = create_patient_applicability_agent()
        imaging_workflow_agent = create_imaging_workflow_agent()

        async def run_identity_decision() -> AgentRunResult[IdentityDecision]:
            with logfire.span("assign_metadata.agent.identity", finding_name=finding_model.name):
                return await identity_agent.run(
                    _identity_decision_prompt(
                        finding_model,
                        projected_ontology_states,
                        projected_anatomic_states,
                        fill_blanks_only=fill_blanks_only,
                    )
                )

        async def run_patient_decision() -> AgentRunResult[PatientApplicabilityDecision]:
            with logfire.span("assign_metadata.agent.patient", finding_name=finding_model.name):
                return await patient_agent.run(
                    _patient_applicability_prompt(
                        finding_model,
                        projected_ontology_states,
                        projected_anatomic_states,
                        fill_blanks_only=fill_blanks_only,
                    )
                )

        async def run_imaging_workflow_decision() -> AgentRunResult[ImagingWorkflowDecision]:
            with logfire.span("assign_metadata.agent.imaging_workflow", finding_name=finding_model.name):
                return await imaging_workflow_agent.run(
                    _imaging_workflow_prompt(
                        finding_model,
                        projected_ontology_states,
                        projected_anatomic_states,
                        anatomy=anatomy_result.output,
                        fill_blanks_only=fill_blanks_only,
                    )
                )

        identity_result, patient_result, imaging_workflow_result = await asyncio.gather(
            run_identity_decision(),
            run_patient_decision(),
            run_imaging_workflow_decision(),
        )
        etiology_agent = create_etiology_assignment_agent()

        @etiology_agent.output_validator
        def validate_etiology(ctx: RunContext[None], output: EtiologyDecision) -> EtiologyDecision:
            _ = ctx
            return _validate_etiology_decision(output, identity=identity_result.output)

        with logfire.span("assign_metadata.agent.etiology", finding_name=finding_model.name):
            etiology_result = await etiology_agent.run(
                _etiology_decision_prompt(
                    finding_model,
                    projected_ontology_states,
                    projected_anatomic_states,
                    identity=identity_result.output,
                    fill_blanks_only=fill_blanks_only,
                )
            )
        decision = _combine_focused_decisions(
            ontology=ontology_result.output,
            anatomy=anatomy_result.output,
            identity=identity_result.output,
            etiology=etiology_result.output,
            patient=patient_result.output,
            imaging_workflow=imaging_workflow_result.output,
        )
        decision = _validate_classifier_output(
            finding_model,
            decision,
            ontology_states=ontology_states,
            anatomic_states=anatomic_states,
            fill_blanks_only=fill_blanks_only,
        )
        logfire.info(
            "Focused decisions complete",
            fields_set=len(decision.field_confidence),
            ontology_decisions=len(decision.ontology_decisions),
            anatomic_decisions=len(decision.anatomic_decisions),
        )
    return decision, model_used, perf_counter() - start


def _assemble_fill_blanks(
    finding_model: FindingModelFull,
    decision: MetadataAssignmentDecision,
    selected_index_codes: list[IndexCode],
    selected_anatomic_locations: list[IndexCode],
    warnings: list[str],
) -> dict[str, Any]:
    """Build update dict for fill_blanks_only mode: only populate empty fields."""
    updates: dict[str, Any] = {}
    if decision.clear_fields:
        warnings.append("clear_fields ignored in fill_blanks_only mode")
    for field_name in STRUCTURED_METADATA_FIELDS:
        if getattr(finding_model, field_name) is None:
            value = getattr(decision, field_name)
            if value is not None:
                updates[field_name] = value
    if not finding_model.index_codes and selected_index_codes:
        updates["index_codes"] = selected_index_codes
    if not finding_model.anatomic_locations and selected_anatomic_locations:
        updates["anatomic_locations"] = selected_anatomic_locations
    _drop_low_confidence_optional_updates(updates, decision=decision, warnings=warnings)
    return updates


def _apply_clear_field(
    updates: dict[str, Any],
    *,
    finding_model: FindingModelFull,
    decision: MetadataAssignmentDecision,
    field_name: str,
    warnings: list[str],
) -> None:
    if field_name in REQUIRED_METADATA_FIELDS:
        if field_name in updates:
            return
        warnings.append(f"clear_fields: required field '{field_name}' ignored")
        return
    if field_name in CANDIDATE_METADATA_FIELDS:
        warnings.append(f"clear_fields: candidate field '{field_name}' ignored")
        return
    if field_name not in CLEARABLE_FIELDS:
        warnings.append(f"clear_fields: unknown field '{field_name}' ignored")
        return
    if (
        field_name in OPTIONAL_IDENTITY_FIELDS
        and getattr(finding_model, field_name, None) is not None
        and field_name not in updates
        and _is_low_or_medium_confidence(decision.field_confidence, field_name)  # type: ignore[arg-type]
    ):
        warnings.append(f"clear_fields: existing identity field '{field_name}' ignored")
        return
    updates[field_name] = None


def _preserve_existing_identity_fields_for_findings(
    updates: dict[str, Any],
    *,
    finding_model: FindingModelFull,
    decision: MetadataAssignmentDecision,
    warnings: list[str],
) -> None:
    if updates.get("entity_type", finding_model.entity_type) in ENTITY_TYPES_ALLOW_IDENTITY_CLEAR:
        return
    for field_name in OPTIONAL_IDENTITY_FIELDS:
        if (
            field_name in updates
            and updates[field_name] is None
            and getattr(finding_model, field_name, None) is not None
            and _is_low_or_medium_confidence(decision.field_confidence, field_name)  # type: ignore[arg-type]
        ):
            updates.pop(field_name, None)
            warnings.append(f"clear_fields: existing identity field '{field_name}' ignored")


def _ignore_low_confidence_optional_clears(
    updates: dict[str, Any],
    *,
    finding_model: FindingModelFull,
    decision: MetadataAssignmentDecision,
    warnings: list[str],
) -> None:
    for field_name in LOW_CONFIDENCE_OPTIONAL_FIELDS:
        if field_name in OPTIONAL_IDENTITY_FIELDS:
            continue
        if (
            field_name in updates
            and updates[field_name] is None
            and getattr(finding_model, field_name, None) is not None
            and _is_low_confidence(decision.field_confidence, field_name)  # type: ignore[arg-type]
        ):
            updates.pop(field_name, None)
            warnings.append(f"clear_fields: low-confidence optional field '{field_name}' ignored")


def _drop_low_confidence_optional_updates(
    updates: dict[str, Any],
    *,
    decision: MetadataAssignmentDecision,
    warnings: list[str],
) -> None:
    for field_name in sorted(HIGH_CONFIDENCE_OPTIONAL_FIELDS):
        if (
            field_name in updates
            and updates[field_name] is not None
            and not _is_high_confidence(decision.field_confidence, field_name)  # type: ignore[arg-type]
        ):
            updates.pop(field_name, None)
            warnings.append(f"Optional field '{field_name}' ignored without high confidence")
    for field_name in sorted(LOW_CONFIDENCE_OPTIONAL_FIELDS):
        if field_name in updates and _is_low_confidence(decision.field_confidence, field_name):  # type: ignore[arg-type]
            updates.pop(field_name, None)
            warnings.append(f"Low-confidence optional field '{field_name}' ignored")


def _assemble_reassess(
    finding_model: FindingModelFull,
    decision: MetadataAssignmentDecision,
    selected_index_codes: list[IndexCode],
    selected_anatomic_locations: list[IndexCode],
    warnings: list[str],
    *,
    ontology_reviewed: bool,
) -> dict[str, Any]:
    """Build update dict for reassess mode: apply all decisions including clear_fields."""
    updates: dict[str, Any] = {}
    for field_name in STRUCTURED_METADATA_FIELDS:
        value = getattr(decision, field_name)
        if value is not None:
            updates[field_name] = value
    for field_name in decision.clear_fields:
        _apply_clear_field(
            updates,
            finding_model=finding_model,
            decision=decision,
            field_name=field_name,
            warnings=warnings,
        )
    if (
        updates.get("entity_type", finding_model.entity_type) in ENTITY_TYPES_ALLOW_IDENTITY_CLEAR
        and ("etiologies" in updates or finding_model.etiologies is not None)
    ):
        if updates.get("etiologies"):
            warnings.append("etiologies ignored for non-disease entity_type")
        updates["etiologies"] = None
    if selected_index_codes:
        updates["index_codes"] = selected_index_codes
    elif ontology_reviewed and finding_model.index_codes:
        updates["index_codes"] = None
    if selected_anatomic_locations:
        updates["anatomic_locations"] = selected_anatomic_locations
    _drop_low_confidence_optional_updates(updates, decision=decision, warnings=warnings)
    _preserve_existing_identity_fields_for_findings(
        updates, finding_model=finding_model, decision=decision, warnings=warnings
    )
    _ignore_low_confidence_optional_clears(updates, finding_model=finding_model, decision=decision, warnings=warnings)
    return updates


async def assign_metadata(
    finding_model: FindingModelFull,
    *,
    fill_blanks_only: bool = False,
    ontology_cache: OntologyLookupCache | Path | str | None = None,
    anatomic_index: AnatomicLocationIndex | None = None,
    anatomic_index_lock: asyncio.Lock | None = None,
) -> MetadataAssignmentResult:
    """Assign canonical structured metadata to a finding model.

    Args:
        finding_model: The finding model to assign metadata to.
        fill_blanks_only: When True, only populate currently-empty fields.
            When False (default, "reassess" mode), always re-evaluate all fields.
        ontology_cache: Optional durable cache or cache path for ontology candidate evidence.
        anatomic_index: Optional shared anatomic index for candidate gathering.
        anatomic_index_lock: Optional lock protecting shared anatomic index access.
    """
    warnings: list[str] = []
    timings: dict[str, float] = {}
    assignment_mode = "fill_blanks_only" if fill_blanks_only else "reassess"
    resolved_ontology_cache = (
        ontology_cache
        if isinstance(ontology_cache, OntologyLookupCache) or ontology_cache is None
        else OntologyLookupCache(ontology_cache)
    )

    with logfire.span(
        "assign_metadata",
        oifm_id=finding_model.oifm_id,
        finding_name=finding_model.name,
        assignment_mode=assignment_mode,
    ):
        trace_id = _get_trace_id()
        logfire.info("Metadata assignment starting", assignment_mode=assignment_mode)

        ontology_result, timings["ontology_candidates"] = await _gather_ontology_candidates(
            finding_model,
            warnings=warnings,
        )
        anatomic_result, timings["anatomic_candidates"] = await _gather_anatomic_candidates(
            finding_model,
            warnings=warnings,
            anatomic_index=anatomic_index,
            anatomic_index_lock=anatomic_index_lock,
            locality_labels=_ontology_locality_labels(ontology_result),
        )

        ontology_states = _ontology_candidate_states(ontology_result)
        anatomic_states = _anatomic_candidate_states(anatomic_result)
        _filter_anatomic_ontology_states(ontology_states, anatomic_states)
        # Always merge existing data as context for the classifier.
        _merge_existing_ontology_states(finding_model, ontology_states)
        _merge_existing_anatomic_states(finding_model, anatomic_states)
        # Always run the classifier.
        decision, model_used, timings["classifier"] = await _run_classifier(
            finding_model,
            ontology_states=ontology_states,
            anatomic_states=anatomic_states,
            fill_blanks_only=fill_blanks_only,
        )

        _apply_ontology_decisions(ontology_states, decision.ontology_decisions, warnings)
        _apply_anatomic_decisions(anatomic_states, decision.anatomic_decisions, warnings)
        _append_source_support_level_consistency_warnings(anatomic_states, warnings)
        try:
            _record_ontology_cache(resolved_ontology_cache, ontology_states)
        finally:
            if resolved_ontology_cache is not None and not isinstance(ontology_cache, OntologyLookupCache):
                resolved_ontology_cache.close()

        start = perf_counter()
        with logfire.span("assign_metadata.assemble", finding_name=finding_model.name, mode=assignment_mode):
            ontology_report = _ontology_report(ontology_states)
            selected_index_codes = _dedupe_index_codes([
                candidate.code for candidate in ontology_report.canonical_codes
            ])
            selected_anatomic_locations = _selected_anatomic_locations(anatomic_states)
            ontology_reviewed = bool(decision.ontology_decisions)
            if _is_low_confidence(decision.field_confidence, "index_codes"):
                if selected_index_codes:
                    warnings.append("Low-confidence index_codes selection ignored")
                selected_index_codes = []
            if _is_low_confidence(decision.field_confidence, "anatomic_locations"):
                if selected_anatomic_locations:
                    warnings.append("Low-confidence anatomic_locations selection ignored")
                selected_anatomic_locations = []
            if fill_blanks_only:
                updates = _assemble_fill_blanks(
                    finding_model, decision, selected_index_codes, selected_anatomic_locations, warnings
                )
            else:
                updates = _assemble_reassess(
                    finding_model,
                    decision,
                    selected_index_codes,
                    selected_anatomic_locations,
                    warnings,
                    ontology_reviewed=ontology_reviewed,
                )
            updated_model = finding_model.model_copy(update=updates)
            logfire.info(
                "Final model assembled",
                mode=assignment_mode,
                canonical_index_codes=len(updated_model.index_codes or []),
                anatomic_locations=len(updated_model.anatomic_locations or []),
            )
        timings["assembly"] = perf_counter() - start

        review = MetadataAssignmentReview(
            oifm_id=finding_model.oifm_id,
            finding_name=finding_model.name,
            assignment_timestamp=datetime.now(tz=UTC),
            model_used=model_used,
            assignment_mode=assignment_mode,
            logfire_trace_id=trace_id,
            ontology_candidates=ontology_report,
            anatomic_candidates=_anatomic_review(anatomic_states),
            classification_rationale=decision.classification_rationale,
            field_confidence=decision.field_confidence,
            timings=timings,
            warnings=warnings,
        )

        return MetadataAssignmentResult(model=updated_model, review=review)


__all__ = [
    "MetadataAssignmentDecision",
    "assign_metadata",
    "create_etiology_assignment_agent",
    "create_metadata_assignment_agent",
]
