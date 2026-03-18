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
from pydantic_ai import Agent
from pydantic_ai.models import Model

from findingmodel_ai import logger
from findingmodel_ai.config import ModelTier, settings
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
    model_tier: ModelTier = "small",
    model: Model | None = None,
) -> Agent[None, MetadataAssignmentDecision]:
    """Create the narrow classifier agent used by the metadata-assignment pipeline.

    The model parameter exists for test injection (TestModel/FunctionModel).
    Production code should not pass it — model selection goes through agent tags.
    """
    resolved_model = model or settings.get_agent_model("metadata_assign", default_tier=model_tier)
    return Agent[None, MetadataAssignmentDecision](
        model=resolved_model,
        output_type=MetadataAssignmentDecision,
        instructions="""You assign canonical structured metadata for a radiology finding model.

You are not rewriting the authored finding definition. Preserve the finding's identity, name,
description, synonyms, attributes, and tags. Your job is only to decide:
- canonical structured metadata fields
- which ontology candidates are exact / clinically substitutable / related
- which anatomic candidates should be selected

Rules:
- Be conservative. If the evidence is weak, leave a field null/omitted rather than guessing.
- A slightly broader ontology concept can be acceptable when it preserves grouping of equivalent findings.
- A narrower ontology concept is not acceptable.
- Only mark ontology candidates as canonical when they are true equivalents for the finding.
- For non-canonical ontology candidates, provide a rejection reason whenever the evidence supports one.
- Use the provided candidate IDs exactly as given.
- Do not invent candidate IDs that are not in the prompt.
- Keep `classification_rationale` concise and specific.
- `field_confidence` should only include fields you actually set or override.

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
            # Keep the first-seen bucket/relationship when duplicates appear across categories.
            if candidate.concept_id not in states:
                states[candidate.concept_id] = _OntologyCandidateState(
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
        if candidate.concept_id not in states and candidate.concept_id != "NO_RESULTS":
            states[candidate.concept_id] = _AnatomicCandidateState(
                result=candidate,
                selected=selected,
                source_bucket=source_bucket,
            )
    return states


def _compact_ontology_candidates(states: dict[str, _OntologyCandidateState]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": state.result.concept_id,
            "text": state.result.concept_text,
            "table_name": state.result.table_name,
            "system": state.result.as_index_code().system,
            "source_bucket": state.source_bucket,
            "default_relationship": state.relationship.value,
            "default_selected_as_canonical": state.selected_as_canonical,
        }
        for state in states.values()
    ]


def _compact_anatomic_candidates(states: dict[str, _AnatomicCandidateState]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": state.result.concept_id,
            "text": state.result.concept_text,
            "source_bucket": state.source_bucket,
            "default_selected": state.selected,
        }
        for state in states.values()
    ]


def _decision_prompt(
    model: FindingModelFull,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
) -> str:
    payload = {
        "finding_model": _compact_model_context(model),
        "ontology_candidates": _compact_ontology_candidates(ontology_states),
        "anatomic_candidates": _compact_anatomic_candidates(anatomic_states),
    }
    return (
        "Review this finding model and candidate evidence. Decide only the canonical structured metadata "
        "and candidate selections that are justified.\n\n"
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


def _needs_structured_metadata(finding_model: FindingModelFull) -> bool:
    return any(getattr(finding_model, field_name) is None for field_name in STRUCTURED_METADATA_FIELDS)


def _needs_ontology(finding_model: FindingModelFull) -> bool:
    return not bool(finding_model.index_codes)


def _needs_anatomic(finding_model: FindingModelFull) -> bool:
    return not bool(finding_model.anatomic_locations)


def _needs_classifier(finding_model: FindingModelFull) -> bool:
    return _needs_structured_metadata(finding_model) or _needs_ontology(finding_model) or _needs_anatomic(finding_model)


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


async def _gather_ontology_candidates(
    finding_model: FindingModelFull,
    *,
    needs_ontology: bool,
    model_tier: ModelTier,
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
        if needs_ontology:
            try:
                ontology_result = await match_ontology_concepts(
                    finding_name=finding_model.name,
                    finding_description=finding_model.description,
                    exclude_anatomical=True,
                    model_tier=model_tier,
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
        else:
            logfire.info("Skipping ontology candidate gathering; canonical index_codes already present")
    return ontology_result, perf_counter() - start


async def _gather_anatomic_candidates(
    finding_model: FindingModelFull,
    *,
    needs_anatomic: bool,
    warnings: list[str],
) -> tuple[LocationSearchResponse | None, float]:
    anatomic_result: LocationSearchResponse | None = None
    start = perf_counter()
    with logfire.span("assign_metadata.anatomic_candidates", finding_name=finding_model.name):
        if needs_anatomic:
            try:
                anatomic_result = await find_anatomic_locations(
                    finding_name=finding_model.name,
                    description=finding_model.description,
                    model_tier="small",
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
        else:
            logfire.info("Skipping anatomic candidate gathering; canonical anatomic_locations already present")
    return anatomic_result, perf_counter() - start


def _merge_existing_ontology_states(
    finding_model: FindingModelFull, states: dict[str, _OntologyCandidateState], *, needs_ontology: bool
) -> None:
    if needs_ontology:
        return
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
    finding_model: FindingModelFull, states: dict[str, _AnatomicCandidateState], *, needs_anatomic: bool
) -> None:
    if needs_anatomic:
        return
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
    needs_classifier: bool,
    ontology_states: dict[str, _OntologyCandidateState],
    anatomic_states: dict[str, _AnatomicCandidateState],
    model_tier: ModelTier,
) -> tuple[MetadataAssignmentDecision, str, float]:
    model_used = settings.get_effective_model_string("metadata_assign", default_tier=model_tier)
    decision = MetadataAssignmentDecision(classification_rationale="Fast-path: existing canonical data was sufficient.")
    start = perf_counter()
    with logfire.span(
        "assign_metadata.classifier",
        finding_name=finding_model.name,
        ontology_candidates=len(ontology_states),
        anatomic_candidates=len(anatomic_states),
    ):
        if needs_classifier:
            agent = create_metadata_assignment_agent(model_tier=model_tier)
            decision_result = await agent.run(_decision_prompt(finding_model, ontology_states, anatomic_states))
            decision = decision_result.output
            logfire.info(
                "Classifier complete",
                fields_set=len(decision.field_confidence),
                ontology_decisions=len(decision.ontology_decisions),
                anatomic_decisions=len(decision.anatomic_decisions),
            )
        else:
            logfire.info("Skipping classifier; canonical model already has metadata, codes, and anatomy")
    return decision, model_used, perf_counter() - start


async def assign_metadata(
    finding_model: FindingModelFull,
    *,
    model_tier: ModelTier = "small",
) -> MetadataAssignmentResult:
    """Assign canonical structured metadata to a finding model."""
    warnings: list[str] = []
    timings: dict[str, float] = {}

    with logfire.span(
        "assign_metadata", oifm_id=finding_model.oifm_id, finding_name=finding_model.name, model_tier=model_tier
    ):
        trace_id = _get_trace_id()
        needs_ontology = _needs_ontology(finding_model)
        needs_anatomic = _needs_anatomic(finding_model)
        needs_classifier = _needs_classifier(finding_model)
        logfire.info(
            "Metadata-assignment fast-path status",
            needs_ontology=needs_ontology,
            needs_anatomic=needs_anatomic,
            needs_classifier=needs_classifier,
        )
        (
            (ontology_result, timings["ontology_candidates"]),
            (anatomic_result, timings["anatomic_candidates"]),
        ) = await asyncio.gather(
            _gather_ontology_candidates(
                finding_model,
                needs_ontology=needs_ontology,
                model_tier=model_tier,
                warnings=warnings,
            ),
            _gather_anatomic_candidates(
                finding_model,
                needs_anatomic=needs_anatomic,
                warnings=warnings,
            ),
        )

        ontology_states = _ontology_candidate_states(ontology_result)
        anatomic_states = _anatomic_candidate_states(anatomic_result)
        _merge_existing_ontology_states(finding_model, ontology_states, needs_ontology=needs_ontology)
        _merge_existing_anatomic_states(finding_model, anatomic_states, needs_anatomic=needs_anatomic)
        decision, model_used, timings["classifier"] = await _run_classifier(
            finding_model,
            needs_classifier=needs_classifier,
            ontology_states=ontology_states,
            anatomic_states=anatomic_states,
            model_tier=model_tier,
        )

        _apply_ontology_decisions(ontology_states, decision.ontology_decisions, warnings)
        _apply_anatomic_decisions(anatomic_states, decision.anatomic_decisions, warnings)

        start = perf_counter()
        with logfire.span("assign_metadata.assemble", finding_name=finding_model.name):
            ontology_report = _ontology_report(ontology_states)
            selected_index_codes = _dedupe_index_codes([
                candidate.code for candidate in ontology_report.canonical_codes
            ])
            selected_anatomic_locations = _selected_anatomic_locations(anatomic_states)
            updates: dict[str, Any] = {}
            for field_name in STRUCTURED_METADATA_FIELDS:
                value = getattr(decision, field_name)
                if value is not None:
                    updates[field_name] = value
            if selected_index_codes:
                updates["index_codes"] = selected_index_codes
            if selected_anatomic_locations:
                updates["anatomic_locations"] = selected_anatomic_locations

            updated_model = finding_model.model_copy(update=updates)
            logfire.info(
                "Final model assembled",
                canonical_index_codes=len(updated_model.index_codes or []),
                anatomic_locations=len(updated_model.anatomic_locations or []),
            )
        timings["assembly"] = perf_counter() - start

        review = MetadataAssignmentReview(
            oifm_id=finding_model.oifm_id,
            finding_name=finding_model.name,
            assignment_timestamp=datetime.now(tz=UTC),
            model_used=model_used,
            model_tier=model_tier,
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
