"""Tests for browse(), metadata-aware search(), and search_batch() (Slice 4).

Tests the metadata WHERE clause builder, browse with metadata filters,
and search/search_batch with metadata filtering against a populated fixture DB.

Test data reference (6 models in test_index.duckdb):
    - abdominal aortic aneurysm: abdomen, diagnosis, [VA,ER],     [CT,US,MR], aorta+abdominal_aorta, SNOMEDCT:233985008
    - aortic dissection:          chest,   diagnosis, [CA,CH,VA,ER], [CT,MR,XR], aorta
    - breast density:             breast,  measurement,[BR],       [MG],       female_breast
    - mammographic malignancy:    breast,  assessment, [BR],       [MG],       female_breast
    - pulmonary embolism:         chest,   diagnosis, [CH,ER,VA], [CT,XR],    lung, SNOMEDCT:59282003
    - ventricular diameters:      chest,   measurement,[CA],       [US,CT,MR], heart
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from findingmodel import BodyRegion, EntityType, EtiologyCode, Modality, SexSpecificity, Subspecialty
from findingmodel.index import FindingModelIndex

# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def index(prebuilt_db_path: Path) -> AsyncGenerator[FindingModelIndex, None]:
    """Load the pre-built test database (read-only)."""
    async with FindingModelIndex(prebuilt_db_path) as idx:
        yield idx


# ============================================================================
# _build_metadata_where_clause unit tests
# ============================================================================


class TestBuildFacetWhereClause:
    """Test the SQL WHERE clause builder for metadata filtering."""

    def test_no_metadatas_returns_empty(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause()
        assert clauses == []
        assert params == []

    def test_single_body_region(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(body_regions=[BodyRegion.CHEST])
        assert len(clauses) == 1
        assert "list_has_any" in clauses[0]
        assert "body_regions" in clauses[0]
        assert params == [["chest"]]

    def test_multiple_body_regions(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(
            body_regions=[BodyRegion.CHEST, BodyRegion.ABDOMEN]
        )
        assert len(clauses) == 1
        assert params == [["chest", "abdomen"]]

    def test_entity_type_scalar(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(entity_type=EntityType.FINDING)
        assert len(clauses) == 1
        assert "entity_type = ?" in clauses[0]
        assert params == ["finding"]

    def test_entity_type_list(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(
            entity_type=[EntityType.FINDING, EntityType.DIAGNOSIS]
        )
        assert len(clauses) == 1
        assert "IN" in clauses[0]
        assert params == ["finding", "diagnosis"]

    def test_sex_specificity_scalar(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(sex_specificity=SexSpecificity.FEMALE_SPECIFIC)
        assert len(clauses) == 1
        assert "sex_specificity = ?" in clauses[0]
        assert params == ["female-specific"]

    def test_tags_all_of_semantics(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(tags=["radiology", "chest"])
        assert len(clauses) == 2  # One subquery per tag
        assert all("tags" in c for c in clauses)
        assert params == ["radiology", "chest"]

    def test_multiple_metadatas_and_across(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.FINDING,
            subspecialties=[Subspecialty.CH],
        )
        assert len(clauses) == 3
        assert len(params) == 3  # [["chest"]], "finding", [["CH"]]

    def test_modalities(self) -> None:
        clauses, params = FindingModelIndex._build_metadata_where_clause(
            applicable_modalities=[Modality.CT, Modality.MR]
        )
        assert len(clauses) == 1
        assert "applicable_modalities" in clauses[0]
        assert params == [["CT", "MR"]]

    def test_etiologies(self) -> None:
        # Just verify it doesn't crash for available enum values
        clauses, _params = FindingModelIndex._build_metadata_where_clause(etiologies=list(EtiologyCode)[:1])
        assert len(clauses) == 1
        assert "etiologies" in clauses[0]


# ============================================================================
# browse() tests — positive matches against populated fixture
# ============================================================================


class TestBrowse:
    """Test browse() method with metadata filters against populated fixture DB."""

    @pytest.mark.asyncio
    async def test_browse_no_filters_returns_all(self, index: FindingModelIndex) -> None:
        entries, total = await index.browse()
        assert total == 6
        assert len(entries) == 6

    @pytest.mark.asyncio
    async def test_browse_with_pagination(self, index: FindingModelIndex) -> None:
        entries_all, total = await index.browse()
        entries_page, page_total = await index.browse(limit=2, offset=0)
        assert page_total == total
        assert len(entries_page) == 2
        assert entries_page[0].oifm_id == entries_all[0].oifm_id

    @pytest.mark.asyncio
    async def test_browse_body_region_chest(self, index: FindingModelIndex) -> None:
        """Chest models: aortic dissection, PE, ventricular diameters."""
        entries, total = await index.browse(body_regions=[BodyRegion.CHEST])
        assert total == 3
        names = {e.name for e in entries}
        assert "aortic dissection" in names
        assert "pulmonary embolism" in names
        assert "Ventricular diameters" in names

    @pytest.mark.asyncio
    async def test_browse_body_region_breast(self, index: FindingModelIndex) -> None:
        """Breast models: breast density, mammographic malignancy."""
        entries, total = await index.browse(body_regions=[BodyRegion.BREAST])
        assert total == 2
        names = {e.name for e in entries}
        assert "Breast density" in names
        assert "Mammographic malignancy assessment" in names

    @pytest.mark.asyncio
    async def test_browse_or_within_body_regions(self, index: FindingModelIndex) -> None:
        """OR-within: chest OR abdomen = 4 models."""
        _entries, total = await index.browse(body_regions=[BodyRegion.CHEST, BodyRegion.ABDOMEN])
        assert total == 4  # 3 chest + 1 abdomen

    @pytest.mark.asyncio
    async def test_browse_entity_type_diagnosis(self, index: FindingModelIndex) -> None:
        """Diagnoses: AAA, aortic dissection, PE."""
        _entries, total = await index.browse(entity_type=EntityType.DIAGNOSIS)
        assert total == 3

    @pytest.mark.asyncio
    async def test_browse_entity_type_measurement(self, index: FindingModelIndex) -> None:
        """Measurements: breast density, ventricular diameters."""
        _entries, total = await index.browse(entity_type=EntityType.MEASUREMENT)
        assert total == 2

    @pytest.mark.asyncio
    async def test_browse_and_across_metadatas(self, index: FindingModelIndex) -> None:
        """AND-across: chest AND diagnosis = aortic dissection + PE (not ventricular diameters)."""
        entries, total = await index.browse(
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.DIAGNOSIS,
        )
        assert total == 2
        names = {e.name for e in entries}
        assert "aortic dissection" in names
        assert "pulmonary embolism" in names

    @pytest.mark.asyncio
    async def test_browse_subspecialty(self, index: FindingModelIndex) -> None:
        """ER subspecialty: AAA, aortic dissection, PE."""
        _entries, total = await index.browse(subspecialties=[Subspecialty.ER])
        assert total == 3

    @pytest.mark.asyncio
    async def test_browse_modality_mg(self, index: FindingModelIndex) -> None:
        """MG modality: breast density, mammographic malignancy."""
        _entries, total = await index.browse(applicable_modalities=[Modality.MG])
        assert total == 2

    @pytest.mark.asyncio
    async def test_browse_female_specific(self, index: FindingModelIndex) -> None:
        """Female-specific models: breast density, mammographic malignancy."""
        entries, total = await index.browse(sex_specificity=SexSpecificity.FEMALE_SPECIFIC)
        assert total == 2
        names = {e.name for e in entries}
        assert names == {"Breast density", "Mammographic malignancy assessment"}

    @pytest.mark.asyncio
    async def test_browse_nonexistent_body_region_returns_empty(self, index: FindingModelIndex) -> None:
        _entries, total = await index.browse(body_regions=[BodyRegion.LOWER_EXTREMITY])
        assert total == 0

    @pytest.mark.asyncio
    async def test_browse_nonexistent_entity_type_returns_empty(self, index: FindingModelIndex) -> None:
        _entries, total = await index.browse(entity_type=EntityType.TECHNIQUE_ISSUE)
        assert total == 0


# ============================================================================
# search() with metadatas tests
# ============================================================================


class TestSearchWithFacets:
    """Test search() with metadata filters against populated fixture DB."""

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_without_metadatas_returns_results(self, index: FindingModelIndex) -> None:
        results = await index.search("aortic", limit=5)
        assert len(results) > 0

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_metadata_narrows_to_matching_region(self, index: FindingModelIndex) -> None:
        """'aortic' with abdomen filter should find AAA but not aortic dissection (chest)."""
        results = await index.search("aortic", limit=5, body_regions=[BodyRegion.ABDOMEN])
        names = {r.name for r in results}
        assert "abdominal aortic aneurysm" in names
        assert "aortic dissection" not in names

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_with_entity_type_assessment_matches_assessment(self, index: FindingModelIndex) -> None:
        results = await index.search("mammographic", limit=5, entity_type=EntityType.ASSESSMENT)
        names = {r.name for r in results}
        assert names == {"Mammographic malignancy assessment"}

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_nonexistent_metadata_returns_empty(self, index: FindingModelIndex) -> None:
        results = await index.search("aortic", limit=5, body_regions=[BodyRegion.LOWER_EXTREMITY])
        assert len(results) == 0


# ============================================================================
# search_batch() with metadatas tests
# ============================================================================


class TestSearchBatchWithFacets:
    """Test search_batch() with metadata filters against populated fixture DB."""

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_batch_without_metadatas(self, index: FindingModelIndex) -> None:
        results = await index.search_batch(["aortic", "breast"], limit=3)
        assert "aortic" in results
        assert "breast" in results

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_batch_with_entity_type_filter(self, index: FindingModelIndex) -> None:
        """Filter to measurements only — 'breast' should find breast density but not mammographic assessment."""
        results = await index.search_batch(
            ["breast"],
            limit=5,
            entity_type=EntityType.MEASUREMENT,
        )
        if "breast" in results:
            names = {r.name for r in results["breast"]}
            assert "Breast density" in names
            # mammographic malignancy is assessment, not measurement
            assert "Mammographic malignancy assessment" not in names

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_batch_with_restrictive_metadata(self, index: FindingModelIndex) -> None:
        results = await index.search_batch(
            ["aortic", "breast"],
            limit=3,
            body_regions=[BodyRegion.LOWER_EXTREMITY],
        )
        for entries in results.values():
            assert len(entries) == 0

    @pytest.mark.asyncio
    @pytest.mark.callout
    async def test_search_batch_with_breast_mg_filters_positive(self, index: FindingModelIndex) -> None:
        results = await index.search_batch(
            ["breast", "mammographic"],
            limit=5,
            body_regions=[BodyRegion.BREAST],
            applicable_modalities=[Modality.MG],
        )
        assert set(results) == {"breast", "mammographic"}
        assert {entry.name for entry in results["breast"]} == {
            "Breast density",
            "Mammographic malignancy assessment",
        }
        assert {entry.name for entry in results["mammographic"]} == {
            "Breast density",
            "Mammographic malignancy assessment",
        }
