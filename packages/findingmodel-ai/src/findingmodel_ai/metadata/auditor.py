"""Lightweight enrichment sanity-check agent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from anatomic_locations import AnatomicLocationIndex
from findingmodel import FindingModelFull, SexSpecificity
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from findingmodel_ai.config import settings
from findingmodel_ai.metadata.ontology_cache import OntologyLookupCache

AuditSeverity = Literal["low", "medium", "high"]


class EnrichmentAuditFlag(BaseModel):
    """One potential enrichment issue for human review."""

    severity: AuditSeverity
    field: str
    message: str
    evidence: str | None = None


class EnrichmentAuditResult(BaseModel):
    """Structured output from enrichment sanity checking."""

    flags: list[EnrichmentAuditFlag] = Field(default_factory=list)


def create_enrichment_auditor_agent(model: Model | None = None) -> Agent[None, EnrichmentAuditResult]:
    """Create the lightweight enrichment auditor agent."""
    resolved_model = model or settings.get_agent_model("metadata_assign")
    return Agent[None, EnrichmentAuditResult](
        model=resolved_model,
        output_type=EnrichmentAuditResult,
        instructions="""You are a radiology finding-model enrichment auditor.

Your task is to sanity-check an already-enriched finding model. You are not re-enriching the model,
rewriting metadata, or proposing broad alternatives. Return flags only for likely problems that a
human reviewer should inspect.

Focus on high-impact problems:
- ontology `index_codes` that appear merely related, broader, narrower, or wrong for the modeled
  finding rather than exact/clinically substitutable
- ontology code/display inconsistencies not already reported by deterministic cache checks
- anatomy, body-region, modality, or subspecialty contradictions
- impossible or clearly unsupported age profile, sex specificity, or expected time course
- etiologies that are too broad, unsupported, or inappropriate for measurements/technique issues

Use the provided ontology cache evidence as factual evidence. If evidence is missing, do not invent
ontology facts. Missing-code evidence may already be flagged deterministically; do not duplicate that
flag unless the model content adds a distinct concern.

Be conservative. Do not flag preference-only changes, minor style issues, or reasonable judgment
calls. Each flag must include the field, severity, concise message, and concrete evidence.""",
    )


async def audit_enrichment(
    finding_model: FindingModelFull,
    *,
    ontology_cache: OntologyLookupCache | Path | str | None = None,
    anatomic_index: AnatomicLocationIndex | None = None,
    anatomic_index_lock: asyncio.Lock | None = None,
    include_llm: bool = True,
) -> EnrichmentAuditResult:
    """Audit an enriched finding model using deterministic ontology evidence plus a light LLM pass."""
    cache = (
        ontology_cache
        if isinstance(ontology_cache, OntologyLookupCache) or ontology_cache is None
        else OntologyLookupCache(ontology_cache)
    )
    try:
        if anatomic_index_lock is not None:
            async with anatomic_index_lock:
                deterministic_flags = _deterministic_flags(finding_model, cache, anatomic_index)
        else:
            deterministic_flags = _deterministic_flags(finding_model, cache, anatomic_index)
        if not include_llm:
            return EnrichmentAuditResult(flags=deterministic_flags)
        prompt = _audit_prompt(finding_model, cache, deterministic_flags)
        agent = create_enrichment_auditor_agent()
        result = await agent.run(prompt)
        return EnrichmentAuditResult(flags=[*deterministic_flags, *result.output.flags])
    finally:
        if cache is not None and not isinstance(ontology_cache, OntologyLookupCache):
            cache.close()


def _deterministic_flags(
    finding_model: FindingModelFull,
    cache: OntologyLookupCache | None,
    anatomic_index: AnatomicLocationIndex | None,
) -> list[EnrichmentAuditFlag]:
    return [
        *_ontology_evidence_flags(finding_model, cache),
        *_anatomy_sex_specificity_flags(finding_model, anatomic_index),
    ]


def _ontology_evidence_flags(
    finding_model: FindingModelFull, cache: OntologyLookupCache | None
) -> list[EnrichmentAuditFlag]:
    flags: list[EnrichmentAuditFlag] = []
    for code in finding_model.index_codes or []:
        evidence = cache.get(code.system, code.code) if cache else None
        code_label = f"{code.system}:{code.code}"
        if evidence is None:
            flags.append(
                EnrichmentAuditFlag(
                    severity="high",
                    field="index_codes",
                    message=f"Missing ontology lookup evidence for canonical code {code_label}.",
                    evidence=code.display,
                )
            )
            continue
        if code.display and evidence.preferred_display != code.display:
            flags.append(
                EnrichmentAuditFlag(
                    severity="medium",
                    field="index_codes",
                    message=f"Canonical code {code_label} display does not match cached preferred display.",
                    evidence=f"model={code.display!r}; cache={evidence.preferred_display!r}",
                )
            )
    return flags


def _anatomy_sex_specificity_flags(
    finding_model: FindingModelFull, anatomic_index: AnatomicLocationIndex | None
) -> list[EnrichmentAuditFlag]:
    if anatomic_index is None:
        return []
    flags: list[EnrichmentAuditFlag] = []
    model_sex = finding_model.sex_specificity
    for code in finding_model.anatomic_locations or []:
        try:
            location = anatomic_index.get(code.code)
        except KeyError:
            continue
        sex_text = (location.sex_specific or "").lower()
        expected: SexSpecificity | None = None
        if "male" in sex_text:
            expected = SexSpecificity.MALE_SPECIFIC
        elif "female" in sex_text:
            expected = SexSpecificity.FEMALE_SPECIFIC
        if expected is not None and model_sex != expected:
            flags.append(
                EnrichmentAuditFlag(
                    severity="high",
                    field="sex_specificity",
                    message="Selected anatomic location implies sex specificity inconsistent with the model.",
                    evidence=f"{code.display or code.code} sex_specific={location.sex_specific}; expected={expected.value}",
                )
            )
    return flags


def _audit_prompt(
    finding_model: FindingModelFull,
    cache: OntologyLookupCache | None,
    deterministic_flags: list[EnrichmentAuditFlag],
) -> str:
    ontology_evidence = []
    for code in finding_model.index_codes or []:
        evidence = cache.get(code.system, code.code) if cache else None
        ontology_evidence.append({
            "code": code.model_dump(mode="json"),
            "cache_evidence": evidence.model_dump(mode="json") if evidence else None,
        })

    payload = {
        "finding_model": finding_model.model_dump(mode="json", exclude_none=True),
        "ontology_evidence": ontology_evidence,
        "deterministic_flags_already_emitted": [
            flag.model_dump(mode="json", exclude_none=True) for flag in deterministic_flags
        ],
    }
    return (
        "Review this enriched finding model for likely metadata or ontology-code problems. "
        "Return only additional sanity-check flags that are not already covered by deterministic flags.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


__all__ = [
    "EnrichmentAuditFlag",
    "EnrichmentAuditResult",
    "audit_enrichment",
    "create_enrichment_auditor_agent",
]
