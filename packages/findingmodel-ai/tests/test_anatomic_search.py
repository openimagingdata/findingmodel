"""Tests for generic anatomic search query preparation."""

from __future__ import annotations

import pytest
from findingmodel.protocols import OntologySearchResult
from findingmodel_ai.search.anatomic import (
    AnatomicQueryTerms,
    LocationSearchResponse,
    _explicit_context_terms,
    _retain_exact_term_locations,
    execute_anatomic_candidate_search,
    execute_anatomic_search,
)


def test_explicit_context_terms_humanize_names_without_medical_variant_tables() -> None:
    terms = _explicit_context_terms(
        "air_in_esophagus",
        None,
        synonyms=["aortic_measurements"],
        attribute_labels=["vertebral coronal cleft"],
        locality_labels=["sacroiliac joint disease"],
    )

    assert "air in esophagus" in terms
    assert "esophagus" in terms
    assert "aortic measurements" in terms
    assert "sacroiliac joint" in terms
    assert "aorta" not in terms
    assert "spine" not in terms


def test_explicit_context_terms_include_description_phrases_without_adjective_mapping() -> None:
    terms = _explicit_context_terms(
        "axillary_mass",
        "Composite hippocampal occupancy score; thyroid bed clips.",
        synonyms=["supraglottic mass"],
        attribute_labels=["renal cortical echogenicity"],
        locality_labels=None,
    )

    assert "axillary mass" in terms
    assert "supraglottic mass" in terms
    assert "renal cortical echogenicity" in terms
    assert "Composite hippocampal occupancy score" in terms
    assert "thyroid bed clips" in terms
    assert "axilla" not in terms
    assert "kidney" not in terms
    assert "larynx" not in terms


def test_explicit_context_terms_normalize_vertebral_scope() -> None:
    terms = _explicit_context_terms(
        "acquired fused vertebrae",
        "Union of two or more vertebrae due to disease, trauma, or surgery.",
        synonyms=None,
        attribute_labels=None,
        locality_labels=None,
    )

    assert "vertebral column" in terms
    assert "spine" in terms


@pytest.mark.asyncio
async def test_execute_anatomic_search_expands_exact_hits_to_ancestors() -> None:
    class Location:
        def __init__(self, loc_id: str, description: str, ancestors: list[Location] | None = None) -> None:
            self.id = loc_id
            self.description = description
            self.laterality = "generic"
            self._ancestors = ancestors or []

        def get_containment_ancestors(self, index: object) -> list[Location]:
            _ = index
            return self._ancestors

    urinary_tract = Location("RID204", "urinary tract")
    kidney = Location("RID205", "kidney", [urinary_tract])

    class Index:
        def get(self, term: str) -> Location:
            if term == "kidney":
                return kidney
            raise KeyError(term)

        async def search_batch(
            self,
            terms: list[str],
            *,
            limit: int,
            region: str | None,
            sided_filter: list[str],
        ) -> dict[str, list[Location]]:
            _ = terms, limit, region, sided_filter
            return {}

        def get_children_of(self, parent_id: str) -> list[Location]:
            _ = parent_id
            return []

    results = await execute_anatomic_search(AnatomicQueryTerms(region=None, terms=["kidney"]), Index())  # type: ignore[arg-type]

    assert [(result.concept_id, result.concept_text) for result in results] == [
        ("RID205", "kidney"),
        ("RID204", "urinary tract"),
    ]


@pytest.mark.asyncio
async def test_execute_anatomic_candidate_search_labels_supported_children() -> None:
    class Location:
        def __init__(
            self,
            loc_id: str,
            description: str,
            ancestors: list[Location] | None = None,
            children: list[Location] | None = None,
        ) -> None:
            self.id = loc_id
            self.description = description
            self.laterality = "generic"
            self._ancestors = ancestors or []
            self._children = children or []

        def get_containment_ancestors(self, index: object) -> list[Location]:
            _ = index
            return self._ancestors

    thoracic_aorta = Location("RID879", "thoracic aorta")
    aorta = Location("RID480", "aorta", children=[thoracic_aorta])
    thoracic_aorta._ancestors = [aorta]

    class Index:
        def get(self, term: str) -> Location:
            if term == "aorta":
                return aorta
            raise KeyError(term)

        async def search_batch(
            self,
            terms: list[str],
            *,
            limit: int,
            region: str | None,
            sided_filter: list[str],
        ) -> dict[str, list[Location]]:
            _ = terms, limit, region, sided_filter
            return {"aorta": [thoracic_aorta]}

        def get_children_of(self, parent_id: str) -> list[Location]:
            if parent_id == "RID480":
                return [thoracic_aorta]
            return []

    candidates = await execute_anatomic_candidate_search(
        AnatomicQueryTerms(region=None, terms=["aorta"]),
        Index(),  # type: ignore[arg-type]
        direct_terms=[],
        ontology_context_terms=[],
    )

    by_id = {candidate.location.concept_id: candidate for candidate in candidates}
    assert by_id["RID480"].support_level == "source_inferred_query"
    assert by_id["RID879"].support_level == "child_of_supported"
    assert by_id["RID879"].broader_candidate_ids == ["RID480"]


def test_retain_exact_term_locations_prioritizes_explicit_context_hits() -> None:
    response = LocationSearchResponse(
        primary_location=OntologySearchResult(
            concept_id="RID205",
            concept_text="kidney",
            score=0.0,
            table_name="anatomic_locations",
        ),
        alternate_locations=[
            OntologySearchResult(
                concept_id=f"RID{i}",
                concept_text=f"alternate {i}",
                score=0.0,
                table_name="anatomic_locations",
            )
            for i in range(8)
        ],
        reasoning="Selector filled all alternate slots.",
    )
    urinary_tract = OntologySearchResult(
        concept_id="RID204",
        concept_text="urinary tract",
        score=0.0,
        table_name="anatomic_locations",
    )

    retained = _retain_exact_term_locations(
        response,
        [urinary_tract, *response.alternate_locations],
        AnatomicQueryTerms(region=None, terms=["radiolucent calculus of urinary tract", "urinary tract"]),
    )

    assert retained.alternate_locations[0].concept_id == "RID204"
    assert len(retained.alternate_locations) == 8
