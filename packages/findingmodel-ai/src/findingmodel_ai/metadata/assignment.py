"""Canonical metadata-assignment pipeline for structured finding model metadata."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import logfire
from findingmodel import (
    AgeProfile,
    BodyRegion,
    EntityType,
    EtiologyCode,
    ExpectedTimeCourse,
    FindingModelFull,
    Modality,
    SexSpecificity,
    Subspecialty,
)
from findingmodel.protocols import OntologySearchResult
from oidm_common.models import IndexCode
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from findingmodel_ai import logger
from findingmodel_ai.config import settings
from findingmodel_ai.metadata.types import (
    AnatomicCandidate,
    FieldConfidence,
    MetadataAssignmentResult,
    MetadataAssignmentReview,
    OntologyCandidate,
    OntologyCandidateRejectionReason,
    OntologyCandidateRelationship,
    OntologyCandidateReport,
)
from findingmodel_ai.search.anatomic import LocationSearchResponse, find_anatomic_locations
from findingmodel_ai.search.ontology import CategorizedOntologyConcepts, match_ontology_concepts


class OntologyCandidateDecision(BaseModel):
    """Classifier decision for one ontology candidate."""

    candidate_id: str
    relationship: OntologyCandidateRelationship
    selected_as_canonical: bool = False
    rationale: str | None = None
    rejection_reason: OntologyCandidateRejectionReason | None = None


class AnatomicCandidateDecision(BaseModel):
    """Classifier decision for one anatomic candidate."""

    candidate_id: str
    selected: bool
    rationale: str | None = None


class MetadataAssignmentDecision(BaseModel):
    """Narrow structured output for ambiguous metadata-assignment decisions."""

    body_regions: list[BodyRegion] | None = None
    subspecialties: list[Subspecialty] | None = None
    etiologies: list[EtiologyCode] | None = None
    entity_type: EntityType | None = None
    applicable_modalities: list[Modality] | None = None
    expected_time_course: ExpectedTimeCourse | None = None
    age_profile: AgeProfile | None = None
    sex_specificity: SexSpecificity | None = None
    ontology_decisions: list[OntologyCandidateDecision] = Field(default_factory=list)
    anatomic_decisions: list[AnatomicCandidateDecision] = Field(default_factory=list)
    clear_fields: list[str] = Field(default_factory=list)
    classification_rationale: str = ""
    field_confidence: dict[str, FieldConfidence] = Field(default_factory=dict)


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


def _get_trace_id() -> str | None:
    """Return the current OpenTelemetry trace ID as a hex string if available."""
    span_context = otel_trace.get_current_span().get_span_context()
    if not span_context or not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"


def create_metadata_assignment_agent(
    model: Model | None = None,
) -> Agent[None, MetadataAssignmentDecision]:
    """Create the narrow classifier agent used by the metadata-assignment pipeline.

    The model parameter exists for test injection (TestModel/FunctionModel).
    Production code should not pass it — model selection goes through agent tags.
    """
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, MetadataAssignmentDecision](
        model=resolved_model,
        output_type=MetadataAssignmentDecision,
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
- `expected_time_course` should reflect the typical imaging course of the named finding itself, not
  every possible clinical variant.

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
- `pulmonary embolism` -> `body_regions=["chest"]`, `subspecialties=["CH","ER","VA"]`,
  `entity_type=diagnosis`, `applicable_modalities=["CT","XR"]`.
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
- `field_confidence` values must be the enum labels `high`, `medium`, or `low`, never numbers.
- `field_confidence` keys must be actual metadata field names only.

Return only the structured output.""",
    )


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


def _anatomic_candidate_states(result: LocationSearchResponse | None) -> dict[str, _AnatomicCandidateState]:
    states: dict[str, _AnatomicCandidateState] = {}
    if result is None:
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
            )
    return states


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


def _compact_anatomic_candidates(states: dict[str, _AnatomicCandidateState]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": key,
            "text": state.result.concept_text,
            "display": state.result.as_index_code().display,
            "source_bucket": state.source_bucket,
            "default_selected": state.selected,
        }
        for key, state in states.items()
    ]


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
        state.rationale = decision.rationale
        state.rejection_reason = decision.rejection_reason
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
        state.rationale = decision.rationale


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
        )
        for state in states.values()
    ]


def _selected_anatomic_locations(states: dict[str, _AnatomicCandidateState]) -> list[IndexCode]:
    return _dedupe_index_codes([state.result.as_index_code() for state in states.values() if state.selected])


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

CLEARABLE_FIELDS = {*STRUCTURED_METADATA_FIELDS, "index_codes", "anatomic_locations"}


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
) -> Any:
    existing_value = getattr(finding_model, field_name)
    decision_value = getattr(decision, field_name)

    if fill_blanks_only:
        if existing_value is None and decision_value is not None:
            return decision_value
        return existing_value

    if field_name in decision.clear_fields:
        return None
    if decision_value is not None:
        return decision_value
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
) -> tuple[LocationSearchResponse | None, float]:
    anatomic_result: LocationSearchResponse | None = None
    start = perf_counter()
    with logfire.span("assign_metadata.anatomic_candidates", finding_name=finding_model.name):
        try:
            anatomic_result = await find_anatomic_locations(
                finding_name=finding_model.name,
                description=finding_model.description,
            )
            logfire.info(
                "Anatomic candidate gathering complete",
                primary_location=anatomic_result.primary_location.concept_id,
                alternates=len(anatomic_result.alternate_locations),
            )
        except Exception as exc:
            warning = f"Anatomic candidate gathering failed: {exc}"
            warnings.append(warning)
            logger.exception(warning)
            logfire.warning("Anatomic candidate gathering failed", error=str(exc))
    return anatomic_result, perf_counter() - start


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
        states[state_key] = _AnatomicCandidateState(
            result=OntologySearchResult(
                concept_id=code.code,
                concept_text=code.display or code.code,
                score=0.0,
                table_name="anatomic_locations",
            ),
            selected=True,
            source_bucket="existing_anatomic_locations",
        )


async def _run_classifier(
    finding_model: FindingModelFull,
    *,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    fill_blanks_only: bool = False,
) -> tuple[MetadataAssignmentDecision, str, float]:
    model_used = settings.get_effective_model_string("metadata_assign")
    start = perf_counter()
    with logfire.span(
        "assign_metadata.classifier",
        finding_name=finding_model.name,
        ontology_candidates=len(ontology_states),
        anatomic_candidates=len(anatomic_states),
    ):
        agent = create_metadata_assignment_agent()

        @agent.output_validator
        def validate_decision(ctx: RunContext[None], output: MetadataAssignmentDecision) -> MetadataAssignmentDecision:
            from pydantic_ai import ModelRetry

            offered_ontology_ids = set(ontology_states.keys())
            for d in output.ontology_decisions:
                if d.candidate_id not in offered_ontology_ids:
                    raise ModelRetry(f"Unknown ontology candidate ID: {d.candidate_id}. Use only offered IDs.")
            offered_anatomic_ids = set(anatomic_states.keys())
            for ad in output.anatomic_decisions:
                if ad.candidate_id not in offered_anatomic_ids:
                    raise ModelRetry(f"Unknown anatomic candidate ID: {ad.candidate_id}. Use only offered IDs.")

            if fill_blanks_only:
                blank_required = [f for f in REQUIRED_METADATA_FIELDS if getattr(finding_model, f) is None]
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
            else:
                cleared_required = sorted(set(output.clear_fields) & set(REQUIRED_METADATA_FIELDS))
                if cleared_required:
                    raise ModelRetry(
                        "Do not clear required fields. Output corrected replacement values instead: "
                        + ", ".join(cleared_required)
                    )
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

            return output

        decision_result = await agent.run(
            _decision_prompt(
                finding_model,
                ontology_states,
                anatomic_states,
                fill_blanks_only=fill_blanks_only,
            )
        )
        decision = decision_result.output
        logfire.info(
            "Classifier complete",
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
    return updates


def _assemble_reassess(
    decision: MetadataAssignmentDecision,
    selected_index_codes: list[IndexCode],
    selected_anatomic_locations: list[IndexCode],
    warnings: list[str],
) -> dict[str, Any]:
    """Build update dict for reassess mode: apply all decisions including clear_fields."""
    updates: dict[str, Any] = {}
    for field_name in STRUCTURED_METADATA_FIELDS:
        value = getattr(decision, field_name)
        if value is not None:
            updates[field_name] = value
    if selected_index_codes:
        updates["index_codes"] = selected_index_codes
    if selected_anatomic_locations:
        updates["anatomic_locations"] = selected_anatomic_locations
    for field_name in decision.clear_fields:
        if field_name in CLEARABLE_FIELDS:
            updates[field_name] = None
        else:
            warnings.append(f"clear_fields: unknown field '{field_name}' ignored")
    return updates


async def assign_metadata(
    finding_model: FindingModelFull,
    *,
    fill_blanks_only: bool = False,
) -> MetadataAssignmentResult:
    """Assign canonical structured metadata to a finding model.

    Args:
        finding_model: The finding model to assign metadata to.
        fill_blanks_only: When True, only populate currently-empty fields.
            When False (default, "reassess" mode), always re-evaluate all fields.
    """
    warnings: list[str] = []
    timings: dict[str, float] = {}
    assignment_mode = "fill_blanks_only" if fill_blanks_only else "reassess"

    with logfire.span(
        "assign_metadata",
        oifm_id=finding_model.oifm_id,
        finding_name=finding_model.name,
        assignment_mode=assignment_mode,
    ):
        trace_id = _get_trace_id()
        logfire.info("Metadata assignment starting", assignment_mode=assignment_mode)

        # Always gather candidates (no fast-path skipping).
        (
            (ontology_result, timings["ontology_candidates"]),
            (anatomic_result, timings["anatomic_candidates"]),
        ) = await asyncio.gather(
            _gather_ontology_candidates(
                finding_model,
                warnings=warnings,
            ),
            _gather_anatomic_candidates(
                finding_model,
                warnings=warnings,
            ),
        )

        ontology_states = _ontology_candidate_states(ontology_result)
        anatomic_states = _anatomic_candidate_states(anatomic_result)
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

        start = perf_counter()
        with logfire.span("assign_metadata.assemble", finding_name=finding_model.name, mode=assignment_mode):
            ontology_report = _ontology_report(ontology_states)
            selected_index_codes = _dedupe_index_codes([
                candidate.code for candidate in ontology_report.canonical_codes
            ])
            selected_anatomic_locations = _selected_anatomic_locations(anatomic_states)
            if fill_blanks_only:
                updates = _assemble_fill_blanks(
                    finding_model, decision, selected_index_codes, selected_anatomic_locations, warnings
                )
            else:
                updates = _assemble_reassess(decision, selected_index_codes, selected_anatomic_locations, warnings)

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


__all__ = ["MetadataAssignmentDecision", "assign_metadata", "create_metadata_assignment_agent"]
