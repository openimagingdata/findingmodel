"""Tests for ontology lookup evidence cache support."""

from pathlib import Path

from findingmodel.protocols import OntologySearchResult
from findingmodel_ai.metadata.ontology_cache import OntologyLookupCache
from oidm_common.models import IndexCode


def test_ontology_lookup_cache_roundtrips_duckdb_evidence(tmp_path: Path) -> None:
    cache_path = tmp_path / "ontology-cache.duckdb"

    with OntologyLookupCache(cache_path) as cache:
        cache.record_ontology_result(
            OntologySearchResult(
                concept_id="233604007",
                concept_text="Pneumonia",
                score=1.0,
                table_name="snomedct",
            ),
            usage="canonical_selected",
            query="pneumonia",
            relationship="exact",
        )

    evidence = cache.get("SNOMEDCT", "233604007")
    assert evidence is not None
    assert evidence.system == "SNOMEDCT"
    assert evidence.code == "233604007"
    assert evidence.preferred_display == "Pneumonia"
    assert evidence.labels == ["Pneumonia"]
    assert evidence.usage == "canonical_selected"
    assert evidence.query == "pneumonia"
    assert evidence.relationship == "exact"
    assert evidence.rejection_reason is None
    assert evidence.concept_uri == "http://purl.bioontology.org/ontology/SNOMEDCT/233604007"


def test_ontology_lookup_cache_get_many_returns_existing_codes(tmp_path: Path) -> None:
    cache = OntologyLookupCache(tmp_path / "ontology-cache.duckdb")
    cache.record_index_code(
        IndexCode(system="RADLEX", code="RID5350", display="pneumonia"),
        usage="fact_check_evidence",
        relationship="related",
        rejection_reason="overlapping_scope",
    )

    evidence = cache.get_many([
        IndexCode(system="RADLEX", code="RID5350", display="pneumonia"),
        IndexCode(system="RADLEX", code="RID9999", display="missing"),
    ])

    assert set(evidence) == {("RADLEX", "RID5350")}
    assert evidence["RADLEX", "RID5350"].usage == "fact_check_evidence"
    assert evidence["RADLEX", "RID5350"].relationship == "related"
    assert evidence["RADLEX", "RID5350"].rejection_reason == "overlapping_scope"
