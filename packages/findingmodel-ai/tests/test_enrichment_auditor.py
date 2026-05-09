"""Tests for lightweight enrichment auditor support."""

from pathlib import Path

import pytest
from findingmodel import (
    BodyRegion,
    FindingModelFull,
    IndexCode,
)
from findingmodel_ai.metadata.auditor import EnrichmentAuditResult, audit_enrichment, create_enrichment_auditor_agent
from findingmodel_ai.metadata.ontology_cache import OntologyLookupCache
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False


def _auditor_with_no_extra_flags() -> object:
    return create_enrichment_auditor_agent(
        model=TestModel(custom_output_args=EnrichmentAuditResult(flags=[]).model_dump(mode="json"))
    )


@pytest.mark.asyncio
async def test_auditor_flags_missing_ontology_evidence(
    full_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "findingmodel_ai.metadata.auditor.create_enrichment_auditor_agent",
        lambda **_: _auditor_with_no_extra_flags(),
    )
    model = full_model.model_copy(
        update={"index_codes": [IndexCode(system="SNOMEDCT", code="233604007", display="Pneumonia")]}
    )

    result = await audit_enrichment(model)

    assert len(result.flags) == 1
    assert result.flags[0].severity == "high"
    assert result.flags[0].field == "index_codes"
    assert "Missing ontology lookup evidence" in result.flags[0].message


@pytest.mark.asyncio
async def test_auditor_can_run_deterministic_checks_without_llm(
    full_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_called(**_: object) -> object:
        raise AssertionError("LLM auditor should not run in deterministic-only mode")

    monkeypatch.setattr("findingmodel_ai.metadata.auditor.create_enrichment_auditor_agent", fail_if_called)
    model = full_model.model_copy(
        update={"index_codes": [IndexCode(system="SNOMEDCT", code="233604007", display="Pneumonia")]}
    )

    result = await audit_enrichment(model, include_llm=False)

    assert len(result.flags) == 1
    assert result.flags[0].field == "index_codes"


@pytest.mark.asyncio
async def test_auditor_flags_display_mismatch_from_cache(
    full_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "findingmodel_ai.metadata.auditor.create_enrichment_auditor_agent",
        lambda **_: _auditor_with_no_extra_flags(),
    )
    cache = OntologyLookupCache(tmp_path / "ontology-cache.duckdb")
    cache.record_index_code(
        IndexCode(system="SNOMEDCT", code="233604007", display="Pneumonia"),
        usage="fact_check_evidence",
    )
    model = full_model.model_copy(
        update={"index_codes": [IndexCode(system="SNOMEDCT", code="233604007", display="Wrong display")]}
    )

    result = await audit_enrichment(model, ontology_cache=cache)

    assert len(result.flags) == 1
    assert result.flags[0].severity == "medium"
    assert "display does not match cached preferred display" in result.flags[0].message
    assert "Wrong display" in (result.flags[0].evidence or "")


@pytest.mark.asyncio
async def test_auditor_does_not_use_anatomic_index_region_as_body_region_truth(
    full_model: FindingModelFull, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "findingmodel_ai.metadata.auditor.create_enrichment_auditor_agent",
        lambda **_: _auditor_with_no_extra_flags(),
    )
    model = full_model.model_copy(
        update={
            "body_regions": [BodyRegion.ABDOMEN],
            "anatomic_locations": [IndexCode(system="ANATOMICLOCATIONS", code="RID1301", display="lung")],
        }
    )

    result = await audit_enrichment(model)

    assert not any(flag.field == "body_regions" for flag in result.flags)
