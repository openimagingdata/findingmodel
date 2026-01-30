"""Integration tests for AnatomicLocationIndex (requires database).

These tests exercise real DuckDB queries using a pre-built test database.
No OpenAI API calls are made - embeddings are pre-generated fixtures.
"""

from pathlib import Path

import pytest
from anatomic_locations import (
    AnatomicLocationIndex,
    AnatomicRegion,
    BodySystem,
    Laterality,
    LocationType,
    StructureType,
)
from pydantic_ai import models

# Block all AI model requests - embeddings are pre-generated fixtures
models.ALLOW_MODEL_REQUESTS = False


# =============================================================================
# Context Manager and Lifecycle Tests
# =============================================================================


class TestAnatomicLocationIndexLifecycle:
    """Tests for index lifecycle (open/close/context manager)."""

    @pytest.mark.asyncio
    async def test_context_manager(self, prebuilt_db_path: Path) -> None:
        """Test using index as context manager."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Should be able to query
            location = index.get("RID39569")
            assert location.id == "RID39569"
            assert location.description == "whole body"

        # After context exit, connection should be closed
        assert index.conn is None

    @pytest.mark.asyncio
    async def test_explicit_open_close(self, prebuilt_db_path: Path) -> None:
        """Test explicit open and close."""
        index = AnatomicLocationIndex(prebuilt_db_path)

        # Should not be open initially
        assert index.conn is None

        # Open connection
        index.open()
        assert index.conn is not None

        # Can query
        location = index.get("RID39569")
        assert location.id == "RID39569"

        # Close connection
        index.close()
        assert index.conn is None

    @pytest.mark.asyncio
    async def test_open_returns_self(self, prebuilt_db_path: Path) -> None:
        """Test that open() returns self for chaining."""
        index = AnatomicLocationIndex(prebuilt_db_path)

        result = index.open()

        assert result is index
        index.close()

    @pytest.mark.asyncio
    async def test_error_when_not_open(self, prebuilt_db_path: Path) -> None:
        """Test that operations fail when index not open."""
        index = AnatomicLocationIndex(prebuilt_db_path)

        with pytest.raises(RuntimeError, match="connection not open"):
            index.get("RID39569")


# =============================================================================
# Core Lookup Tests
# =============================================================================


class TestAnatomicLocationIndexLookups:
    """Tests for core lookup methods."""

    @pytest.mark.asyncio
    async def test_get_existing_location(self, prebuilt_db_path: Path) -> None:
        """Test retrieving an existing location by ID."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            location = index.get("RID1301")

            assert location.id == "RID1301"
            assert location.description == "lung"
            assert location.region == AnatomicRegion.THORAX
            # Should be auto-bound to index
            assert location._index is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_location(self, prebuilt_db_path: Path) -> None:
        """Test error when location ID doesn't exist."""
        with (
            AnatomicLocationIndex(prebuilt_db_path) as index,
            pytest.raises(KeyError, match="Anatomic location not found: RID99999"),
        ):
            index.get("RID99999")

    @pytest.mark.asyncio
    async def test_get_loads_codes(self, prebuilt_db_path: Path) -> None:
        """Test that get() eagerly loads codes."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            location = index.get("RID1301")

            assert len(location.codes) >= 1
            # Should have SNOMED code
            snomed_code = location.get_code("SNOMED")
            assert snomed_code is not None
            assert snomed_code.code == "39607008"

    @pytest.mark.asyncio
    async def test_find_by_code_snomed(self, prebuilt_db_path: Path) -> None:
        """Test finding locations by SNOMED code."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            locations = index.find_by_code("SNOMED", "39607008")

            assert len(locations) >= 1
            assert any(loc.description == "lung" for loc in locations)

    @pytest.mark.asyncio
    async def test_find_by_code_fma(self, prebuilt_db_path: Path) -> None:
        """Test finding locations by FMA code."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            locations = index.find_by_code("FMA", "7197")

            assert len(locations) >= 1
            assert any(loc.description == "liver" for loc in locations)

    @pytest.mark.asyncio
    async def test_find_by_code_not_found(self, prebuilt_db_path: Path) -> None:
        """Test find_by_code returns empty list when code doesn't exist."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            locations = index.find_by_code("SNOMED", "99999999")

            assert locations == []

    @pytest.mark.asyncio
    async def test_search_basic(self, prebuilt_db_path: Path) -> None:
        """Test basic text search."""
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            results = await index.search("lung", limit=5)

            assert len(results) > 0
            # Should find lung-related locations
            descriptions = {loc.description for loc in results}
            assert "lung" in descriptions or "left lung" in descriptions or "right lung" in descriptions


# =============================================================================
# Hierarchy Navigation Tests
# =============================================================================


class TestAnatomicLocationIndexHierarchy:
    """Tests for hierarchy navigation methods."""

    @pytest.mark.asyncio
    async def test_get_containment_ancestors(self, prebuilt_db_path: Path) -> None:
        """Test getting containment ancestors."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Heart is in mediastinum, which is in thorax, which is in body
            ancestors = index.get_containment_ancestors("RID1385")

            assert len(ancestors) >= 1
            # Should include mediastinum (immediate parent)
            assert any(a.description == "mediastinum" for a in ancestors)

    @pytest.mark.asyncio
    async def test_get_containment_descendants(self, prebuilt_db_path: Path) -> None:
        """Test getting containment descendants."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Thorax contains mediastinum, which contains heart
            descendants = index.get_containment_descendants("RID1243")

            assert len(descendants) >= 1
            # Should find descendants
            descriptions = {d.description for d in descendants}
            # Mediastinum is directly in thorax
            assert "mediastinum" in descriptions or "heart" in descriptions

    @pytest.mark.asyncio
    async def test_get_children_of(self, prebuilt_db_path: Path) -> None:
        """Test getting direct children."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Whole body has direct children
            children = index.get_children_of("RID39569")

            assert len(children) >= 1
            # Should include thorax and abdomen
            descriptions = {c.description for c in children}
            assert "thorax" in descriptions or "abdomen" in descriptions

    @pytest.mark.asyncio
    async def test_get_partof_ancestors(self, prebuilt_db_path: Path) -> None:
        """Test getting part-of ancestors."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Superior nasal turbinate is part of ethmoid bone
            ancestors = index.get_partof_ancestors("RID10049")

            # Should have at least the ethmoid bone ancestor
            assert isinstance(ancestors, list)


# =============================================================================
# Filter and Iteration Tests
# =============================================================================


class TestAnatomicLocationIndexFilters:
    """Tests for filtering and iteration methods."""

    @pytest.mark.asyncio
    async def test_by_region(self, prebuilt_db_path: Path) -> None:
        """Test filtering by region."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            thorax_locations = index.by_region("Thorax")

            assert len(thorax_locations) >= 3
            descriptions = {loc.description for loc in thorax_locations}
            assert "lung" in descriptions or "heart" in descriptions

    @pytest.mark.asyncio
    async def test_by_location_type(self, prebuilt_db_path: Path) -> None:
        """Test filtering by location type."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            structures = index.by_location_type(LocationType.STRUCTURE)

            # All our test locations are structures
            assert len(structures) >= 5

    @pytest.mark.asyncio
    async def test_by_system(self, prebuilt_db_path: Path) -> None:
        """Test filtering by body system."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Our test data doesn't have body_system populated
            # Just verify method works without error
            results = index.by_system(BodySystem.RESPIRATORY)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_by_structure_type(self, prebuilt_db_path: Path) -> None:
        """Test filtering by structure type."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Our test data doesn't have structure_type populated
            # Just verify method works without error
            results = index.by_structure_type(StructureType.SOLID_ORGAN)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_iteration(self, prebuilt_db_path: Path) -> None:
        """Test iterating over all locations."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            locations = list(index)

            # Pre-built fixture has 42 records
            assert len(locations) >= 40
            # All should be bound to index
            assert all(loc._index is not None for loc in locations)


# =============================================================================
# Bound Location Navigation Tests
# =============================================================================


class TestBoundLocationNavigation:
    """Tests for navigation via bound AnatomicLocation objects."""

    @pytest.mark.asyncio
    async def test_bound_location_get_containment_ancestors(self, prebuilt_db_path: Path) -> None:
        """Test calling containment navigation without explicit index."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            heart = index.get("RID1385")

            # Should work without passing index (uses bound weakref)
            ancestors = heart.get_containment_ancestors()

            assert len(ancestors) >= 1
            assert any(a.description == "mediastinum" for a in ancestors)

    @pytest.mark.asyncio
    async def test_bound_location_get_containment_siblings(self, prebuilt_db_path: Path) -> None:
        """Test getting siblings via bound location."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            lung = index.get("RID1301")

            siblings = lung.get_containment_siblings()

            # Just verify it returns a list
            assert isinstance(siblings, list)

    @pytest.mark.asyncio
    async def test_bound_location_resolve_laterality(self, prebuilt_db_path: Path) -> None:
        """Test resolving laterality variants via bound location."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            lung = index.get("RID1301")

            # Generic lung should have left and right variants
            left_lung = lung.get_left()
            right_lung = lung.get_right()

            assert left_lung is not None
            assert left_lung.description == "left lung"
            assert right_lung is not None
            assert right_lung.description == "right lung"

    @pytest.mark.asyncio
    async def test_bound_location_get_laterality_variants(self, prebuilt_db_path: Path) -> None:
        """Test getting all laterality variants."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            lung = index.get("RID1301")

            variants = lung.get_laterality_variants()

            # Should have left and right
            assert Laterality.LEFT in variants
            assert Laterality.RIGHT in variants
            assert variants[Laterality.LEFT].description == "left lung"
            assert variants[Laterality.RIGHT].description == "right lung"

    @pytest.mark.asyncio
    async def test_bound_location_error_after_close(self, prebuilt_db_path: Path) -> None:
        """Test that bound location fails after index is closed."""
        index = AnatomicLocationIndex(prebuilt_db_path)
        index.open()

        lung = index.get("RID1301")

        # Close index
        index.close()

        # Trying to navigate should fail with RuntimeError (connection not open)
        with pytest.raises(RuntimeError, match="connection not open"):
            lung.get_containment_ancestors()

    @pytest.mark.asyncio
    async def test_get_parts(self, prebuilt_db_path: Path) -> None:
        """Test getting parts via bound location."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Use nasal cavity which has no hasPartsRefs in sample data
            nasal_cavity = index.get("RID9532")
            parts = nasal_cavity.get_parts()

            assert isinstance(parts, list)
            # Nasal cavity has no parts in sample data
            assert len(parts) == 0

    @pytest.mark.asyncio
    async def test_get_parts_returns_list(self, prebuilt_db_path: Path) -> None:
        """Test that get_parts returns list even when no parts exist."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Use mediastinum which has no hasPartsRefs in sample data
            mediastinum = index.get("RID1384")
            parts = mediastinum.get_parts()

            assert isinstance(parts, list)
            assert len(parts) == 0

    @pytest.mark.asyncio
    async def test_get_generic(self, prebuilt_db_path: Path) -> None:
        """Test getting generic variant via bound location."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Left lung has generic variant (unsided lung)
            left_lung = index.get("RID1326")
            generic = left_lung.get_generic()

            assert generic is not None
            assert generic.description == "lung"
            assert generic.id == "RID1301"

    @pytest.mark.asyncio
    async def test_get_generic_when_no_generic(self, prebuilt_db_path: Path) -> None:
        """Test getting generic variant when location has none."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Unsided lung has no generic variant (it IS the generic)
            lung = index.get("RID1301")
            generic = lung.get_generic()

            assert generic is None

    @pytest.mark.asyncio
    async def test_get_generic_nonlateral_location(self, prebuilt_db_path: Path) -> None:
        """Test getting generic variant for nonlateral location."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Heart is nonlateral, should have no generic variant
            heart = index.get("RID1385")
            generic = heart.get_generic()

            assert generic is None
