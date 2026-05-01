"""Tests for shared protocol models."""

from findingmodel.protocols import OntologySearchResult


def test_ontology_search_result_omits_too_short_index_code_display() -> None:
    result = OntologySearchResult(concept_id="RID1", concept_text="T2", score=0.9, table_name="radlex")

    code = result.as_index_code()

    assert code.system == "RADLEX"
    assert code.code == "RID1"
    assert code.display is None
