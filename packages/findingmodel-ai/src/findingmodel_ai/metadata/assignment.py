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
from findingmodel_ai.metadata.decisions import (
    AnatomicCandidateDecision,
    AnatomyDecision,
    EntityTypeDecision,
    EtiologyTempoDecision,
    MetadataAssignmentDecision,
    ModalityApplicabilityDecision,
    OntologyCandidateDecision,
    OntologyDecision,
    PatientApplicabilityDecision,
    SubspecialtyDomainDecision,
)
from findingmodel_ai.metadata.ontology_cache import OntologyEvidenceUsage, OntologyLookupCache
from findingmodel_ai.metadata.prompt_loader import load_metadata_prompt
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


def create_entity_type_agent(model: Model | None = None) -> Agent[None, EntityTypeDecision]:
    """Create the focused entity-type assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, EntityTypeDecision](
        model=resolved_model,
        output_type=EntityTypeDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions=load_metadata_prompt("entity_type"),
    )


def create_patient_applicability_agent(model: Model | None = None) -> Agent[None, PatientApplicabilityDecision]:
    """Create the focused age/sex applicability assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, PatientApplicabilityDecision](
        model=resolved_model,
        output_type=PatientApplicabilityDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions=load_metadata_prompt("patient_applicability"),
    )


def create_subspecialty_domain_agent(model: Model | None = None) -> Agent[None, SubspecialtyDomainDecision]:
    """Create the pilot focused subspecialty-domain assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, SubspecialtyDomainDecision](
        model=resolved_model,
        output_type=SubspecialtyDomainDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions=load_metadata_prompt("subspecialty_domain"),
    )


def create_modality_applicability_agent(model: Model | None = None) -> Agent[None, ModalityApplicabilityDecision]:
    """Create the pilot focused modality-applicability assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, ModalityApplicabilityDecision](
        model=resolved_model,
        output_type=ModalityApplicabilityDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions=load_metadata_prompt("modality_applicability"),
    )


def create_etiology_tempo_agent(model: Model | None = None) -> Agent[None, EtiologyTempoDecision]:
    """Create the pilot focused etiology-tempo assignment agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, EtiologyTempoDecision](
        model=resolved_model,
        output_type=EtiologyTempoDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions=load_metadata_prompt("etiology_tempo"),
    )


def create_ontology_decision_agent(model: Model | None = None) -> Agent[None, OntologyDecision]:
    """Create the focused ontology-candidate decision agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, OntologyDecision](
        model=resolved_model,
        output_type=OntologyDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions=load_metadata_prompt("ontology_decision"),
    )


def create_anatomy_decision_agent(model: Model | None = None) -> Agent[None, AnatomyDecision]:
    """Create the focused anatomic-candidate decision agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, AnatomyDecision](
        model=resolved_model,
        output_type=AnatomyDecision,
        model_settings=_metadata_classification_settings(),
        retries=2,
        instructions=load_metadata_prompt("anatomy_decision"),
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
        for search_candidate in result.candidates:
            idx_code = search_candidate.location.as_index_code()
            state_key = f"{idx_code.system}:{search_candidate.location.concept_id}"
            if state_key not in states:
                states[state_key] = _AnatomicCandidateState(
                    result=search_candidate.location,
                    selected=False,
                    source_bucket="candidate",
                    support_level=search_candidate.support_level,
                    matched_terms=search_candidate.matched_terms,
                    broader_candidate_ids=search_candidate.broader_candidate_ids,
                )
        return states

    location_candidates = [
        ("primary", result.primary_location, result.primary_location.concept_id != "NO_RESULTS"),
        *(("alternate", location_candidate, False) for location_candidate in result.alternate_locations),
    ]
    for source_bucket, location_candidate, selected in location_candidates:
        if location_candidate.concept_id == "NO_RESULTS":
            continue
        idx_code = location_candidate.as_index_code()
        state_key = f"{idx_code.system}:{location_candidate.concept_id}"
        if state_key not in states:
            states[state_key] = _AnatomicCandidateState(
                result=location_candidate,
                selected=selected,
                source_bucket=source_bucket,
                support_level=source_bucket,
            )
    return states


def _filter_anatomic_ontology_states(
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
) -> None:
    anatomic_displays = set()
    for state in anatomic_states.values():
        display = state.result.as_index_code().display
        if display:
            anatomic_displays.add(display.strip().lower())
    for key, ontology_state in list(ontology_states.items()):
        display = (ontology_state.result.as_index_code().display or "").strip().lower()
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
        field_name for field_name in REQUIRED_METADATA_FIELDS if getattr(model, field_name) is None
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
    payload = _agent_payload(
        model, {}, _anatomic_candidate_prompt_states(anatomic_states), fill_blanks_only=fill_blanks_only
    )
    payload["task"] = "Decide anatomic candidate selection and body_regions."
    return json.dumps(payload, indent=2)


def _entity_type_decision_prompt(
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
    payload["task"] = "Assign only entity_type."
    return json.dumps(payload, indent=2)


def _etiology_tempo_decision_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    entity_type: EntityTypeDecision,
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
    payload["task"] = "Assign only etiologies and expected_time_course."
    payload["entity_type_context"] = {
        "entity_type": entity_type.entity_type.value if entity_type.entity_type is not None else None,
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


def _imaging_metadata_prompt_payload(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    anatomy: AnatomyDecision,
    fill_blanks_only: bool,
) -> dict[str, Any]:
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
    payload["field_values_under_review"] = {
        "subspecialties": existing_metadata["subspecialties"],
        "applicable_modalities": existing_metadata["applicable_modalities"],
    }
    existing_metadata["subspecialties"] = None
    existing_metadata["applicable_modalities"] = None
    payload["anatomy_context"] = {
        "body_regions": [value.value for value in anatomy.body_regions] if anatomy.body_regions else None,
    }
    return payload


def _subspecialty_domain_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    anatomy: AnatomyDecision,
    fill_blanks_only: bool,
) -> str:
    payload = _imaging_metadata_prompt_payload(
        model,
        ontology_states,
        anatomic_states,
        anatomy=anatomy,
        fill_blanks_only=fill_blanks_only,
    )
    payload["task"] = "Assign only subspecialties."
    return json.dumps(payload, indent=2)


def _modality_applicability_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    *,
    anatomy: AnatomyDecision,
    fill_blanks_only: bool,
) -> str:
    payload = _imaging_metadata_prompt_payload(
        model,
        ontology_states,
        anatomic_states,
        anatomy=anatomy,
        fill_blanks_only=fill_blanks_only,
    )
    payload["task"] = "Assign only applicable_modalities."
    return json.dumps(payload, indent=2)


def _apply_ontology_decisions(
    states: dict[str, _OntologyCandidateState],
    decisions: list[OntologyCandidateDecision],
    warnings: list[str],
) -> None:
    for decision in decisions:
        state = states.get(decision.candidate_id)
        if state is None:
            warnings.append(f"Ontology decision referenced unknown candidate: {decision.candidate_id}")
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
            warnings.append(f"Anatomic decision referenced unknown candidate: {decision.candidate_id}")
            continue
        if (
            state.source_bucket == "existing_anatomic_locations"
            and not decision.selected
            and not (decision.rejection_reason or decision.rationale)
        ):
            warnings.append(
                f"Existing anatomic location kept because deselection had no reason: {decision.candidate_id}"
            )
            continue
        state.selected = decision.selected
        state.rationale = decision.rationale or decision.rejection_reason


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


_CARRY_FORWARD_INDEX_CODE_SYSTEMS = {"GAMUTS", "GMTS", "RADELEMENT", "CDES"}
_CARRY_FORWARD_INDEX_CODE_PREFIXES = ("RDES",)


def _is_carry_forward_index_code(code: IndexCode) -> bool:
    system = code.system.strip().upper()
    code_id = code.code.strip().upper()
    return system in _CARRY_FORWARD_INDEX_CODE_SYSTEMS or code_id.startswith(_CARRY_FORWARD_INDEX_CODE_PREFIXES)


def carry_forward_index_codes(finding_model: FindingModelFull) -> list[IndexCode]:
    return _dedupe_index_codes([
        code for code in finding_model.index_codes or [] if _is_carry_forward_index_code(code)
    ])


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

REQUIRED_METADATA_FIELDS = ("entity_type",)

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

    return decision_value


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


def _validate_fill_blanks_required_fields(finding_model: FindingModelFull, output: MetadataAssignmentDecision) -> None:
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
            "Fill-blanks mode must populate every blank required field that remains empty: " + ", ".join(missing_blank)
        )


def _validate_reassess_required_fields(finding_model: FindingModelFull, output: MetadataAssignmentDecision) -> None:
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


def _downgrade_noncanonical_assessment_measurement_ontology(
    output: MetadataAssignmentDecision,
    warnings: list[str],
) -> None:
    if output.entity_type in {EntityType.ASSESSMENT, EntityType.MEASUREMENT}:
        for decision in output.ontology_decisions:
            if not decision.selected_as_canonical:
                continue
            if decision.relationship in {
                OntologyCandidateRelationship.EXACT_MATCH,
                OntologyCandidateRelationship.CLINICALLY_SUBSTITUTABLE,
            }:
                continue
            decision.selected_as_canonical = False
            warnings.append(
                "Ontology candidate moved to review for assessment/measurement model because "
                f"relationship is {decision.relationship.value}: {decision.candidate_id}"
            )


def _validate_assignment_decision(
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


def _validate_etiology_tempo_decision(
    output: EtiologyTempoDecision,
    *,
    entity_type: EntityTypeDecision,
) -> EtiologyTempoDecision:
    from pydantic_ai import ModelRetry

    if entity_type.entity_type in ENTITY_TYPES_ALLOW_IDENTITY_CLEAR and output.etiologies:
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
    entity_type: EntityTypeDecision,
    etiology_tempo: EtiologyTempoDecision,
    patient: PatientApplicabilityDecision,
    subspecialty_domain: SubspecialtyDomainDecision,
    modality_applicability: ModalityApplicabilityDecision,
) -> MetadataAssignmentDecision:
    return MetadataAssignmentDecision(
        body_regions=anatomy.body_regions,
        subspecialties=subspecialty_domain.subspecialties,
        etiologies=etiology_tempo.etiologies,
        entity_type=entity_type.entity_type,
        applicable_modalities=modality_applicability.applicable_modalities,
        expected_time_course=etiology_tempo.expected_time_course,
        age_profile=patient.age_profile,
        sex_specificity=patient.sex_specificity,
        ontology_decisions=ontology.ontology_decisions,
        anatomic_decisions=anatomy.anatomic_decisions,
        classification_rationale="",
        field_confidence=_merge_confidence(
            entity_type.field_confidence,
            etiology_tempo.field_confidence,
            patient.field_confidence,
            subspecialty_domain.field_confidence,
            modality_applicability.field_confidence,
        ),
    )


async def _run_focused_decisions(
    finding_model: FindingModelFull,
    *,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    fill_blanks_only: bool = False,
) -> tuple[MetadataAssignmentDecision, str, float]:
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

        entity_type_agent = create_entity_type_agent()
        patient_agent = create_patient_applicability_agent()
        subspecialty_domain_agent = create_subspecialty_domain_agent()
        modality_applicability_agent = create_modality_applicability_agent()

        async def run_entity_type_decision() -> AgentRunResult[EntityTypeDecision]:
            with logfire.span("assign_metadata.agent.entity_type", finding_name=finding_model.name):
                return await entity_type_agent.run(
                    _entity_type_decision_prompt(
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

        async def run_subspecialty_domain_decision() -> AgentRunResult[SubspecialtyDomainDecision]:
            with logfire.span("assign_metadata.agent.subspecialty_domain", finding_name=finding_model.name):
                return await subspecialty_domain_agent.run(
                    _subspecialty_domain_prompt(
                        finding_model,
                        projected_ontology_states,
                        projected_anatomic_states,
                        anatomy=anatomy_result.output,
                        fill_blanks_only=fill_blanks_only,
                    )
                )

        async def run_modality_applicability_decision() -> AgentRunResult[ModalityApplicabilityDecision]:
            with logfire.span("assign_metadata.agent.modality_applicability", finding_name=finding_model.name):
                return await modality_applicability_agent.run(
                    _modality_applicability_prompt(
                        finding_model,
                        projected_ontology_states,
                        projected_anatomic_states,
                        anatomy=anatomy_result.output,
                        fill_blanks_only=fill_blanks_only,
                    )
                )

        (
            entity_type_result,
            patient_result,
            subspecialty_domain_result,
            modality_applicability_result,
        ) = await asyncio.gather(
            run_entity_type_decision(),
            run_patient_decision(),
            run_subspecialty_domain_decision(),
            run_modality_applicability_decision(),
        )
        etiology_tempo_agent = create_etiology_tempo_agent()

        @etiology_tempo_agent.output_validator
        def validate_etiology_tempo(ctx: RunContext[None], output: EtiologyTempoDecision) -> EtiologyTempoDecision:
            _ = ctx
            return _validate_etiology_tempo_decision(output, entity_type=entity_type_result.output)

        with logfire.span("assign_metadata.agent.etiology_tempo", finding_name=finding_model.name):
            etiology_tempo_result = await etiology_tempo_agent.run(
                _etiology_tempo_decision_prompt(
                    finding_model,
                    projected_ontology_states,
                    projected_anatomic_states,
                    entity_type=entity_type_result.output,
                    fill_blanks_only=fill_blanks_only,
                )
            )
        decision = _combine_focused_decisions(
            ontology=ontology_result.output,
            anatomy=anatomy_result.output,
            entity_type=entity_type_result.output,
            etiology_tempo=etiology_tempo_result.output,
            patient=patient_result.output,
            subspecialty_domain=subspecialty_domain_result.output,
            modality_applicability=modality_applicability_result.output,
        )
        decision = _validate_assignment_decision(
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
        if (
            field_name in updates
            and updates[field_name] is not None
            and _is_low_confidence(decision.field_confidence, field_name)  # type: ignore[arg-type]
        ):
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
    anatomy_reviewed: bool,
) -> dict[str, Any]:
    """Build update dict for reassess mode: apply split-agent decisions."""
    updates: dict[str, Any] = {}
    for field_name in STRUCTURED_METADATA_FIELDS:
        value = getattr(decision, field_name)
        updates[field_name] = value
        if value is None and getattr(finding_model, field_name) is not None:
            warnings.append(f"metadata cleared: {field_name}")
    if updates.get("entity_type", finding_model.entity_type) in ENTITY_TYPES_ALLOW_IDENTITY_CLEAR and (
        "etiologies" in updates or finding_model.etiologies is not None
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
    elif anatomy_reviewed and finding_model.anatomic_locations:
        updates["anatomic_locations"] = None
    _drop_low_confidence_optional_updates(updates, decision=decision, warnings=warnings)
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
        # Always merge existing data as context for split-agent decisions.
        _merge_existing_ontology_states(finding_model, ontology_states)
        _merge_existing_anatomic_states(finding_model, anatomic_states)
        decision, model_used, timings["focused_decisions"] = await _run_focused_decisions(
            finding_model,
            ontology_states=ontology_states,
            anatomic_states=anatomic_states,
            fill_blanks_only=fill_blanks_only,
        )

        _downgrade_noncanonical_assessment_measurement_ontology(decision, warnings)
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
            carried_forward_index_codes = carry_forward_index_codes(finding_model)
            selected_anatomic_locations = _selected_anatomic_locations(anatomic_states)
            ontology_reviewed = bool(decision.ontology_decisions)
            anatomy_reviewed = bool(decision.anatomic_decisions)
            if _is_low_confidence(decision.field_confidence, "index_codes"):
                if selected_index_codes:
                    warnings.append("Low-confidence index_codes selection ignored")
                selected_index_codes = []
            selected_index_codes = _dedupe_index_codes([
                *selected_index_codes,
                *carried_forward_index_codes,
            ])
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
                    anatomy_reviewed=anatomy_reviewed,
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
    "create_entity_type_agent",
    "create_etiology_tempo_agent",
    "create_modality_applicability_agent",
    "create_subspecialty_domain_agent",
]
