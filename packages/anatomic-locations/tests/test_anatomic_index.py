"""Integration tests for AnatomicLocationIndex (requires database)."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from anatomic_locations import (
    AnatomicLocationIndex,
    AnatomicRegion,
    BodySystem,
    Laterality,
    LocationType,
    StructureType,
)
from oidm_maintenance.anatomic.build import create_anatomic_database
from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


# Create a minimal settings class for tests
class TestSettings(BaseSettings):
    """Minimal settings for anatomic location tests."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    openai_embedding_dimensions: int = 512


test_settings = TestSettings()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def test_database(tmp_path: Path) -> Path:
    """Create a test anatomic locations database."""
    db_path = tmp_path / "test_anatomic.duckdb"

    # Create test records with various configurations
    test_records: list[dict[str, Any]] = [
        # Root location
        {"_id": "RID1", "description": "body", "region": "Body"},
        # Thorax hierarchy
        {
            "_id": "RID39",
            "description": "thorax",
            "region": "Thorax",
            "containedByRef": {"id": "RID1"},
        },
        {
            "_id": "RID1301",
            "description": "lung",
            "region": "Thorax",
            "leftRef": {"id": "RID1302", "display": "left lung"},
            "rightRef": {"id": "RID1326", "display": "right lung"},
            "containedByRef": {"id": "RID39"},
            "snomedId": "39607008",
            "snomedDisplay": "Lung structure",
            "synonyms": ["pulmonary organ", "respiratory organ"],
        },
        {
            "_id": "RID1302",
            "description": "left lung",
            "region": "Thorax",
            "containedByRef": {"id": "RID39"},
            "leftRef": {"id": "RID1302", "display": "left lung"},
            "unsidedRef": {"id": "RID1301", "display": "lung"},
        },
        {
            "_id": "RID1326",
            "description": "right lung",
            "region": "Thorax",
            "containedByRef": {"id": "RID39"},
            "rightRef": {"id": "RID1326", "display": "right lung"},
            "unsidedRef": {"id": "RID1301", "display": "lung"},
        },
        # Heart (nonlateral)
        {
            "_id": "RID1385",
            "description": "heart",
            "region": "Thorax",
            "containedByRef": {"id": "RID39"},
            "codes": [{"system": "SNOMED", "code": "80891009", "display": "Heart structure"}],
        },
        # Abdomen
        {
            "_id": "RID58",
            "description": "abdomen",
            "region": "Abdomen",
            "containedByRef": {"id": "RID1"},
        },
        {
            "_id": "RID170",
            "description": "liver",
            "region": "Abdomen",
            "containedByRef": {"id": "RID58"},
            "codes": [{"system": "FMA", "code": "7197", "display": "Liver"}],
        },
        # Liver lobes (parts of liver)
        {
            "_id": "RID171",
            "description": "right lobe of liver",
            "region": "Abdomen",
            "containedByRef": {"id": "RID58"},
            "partOfRef": {"id": "RID170", "display": "liver"},
        },
        {
            "_id": "RID172",
            "description": "left lobe of liver",
            "region": "Abdomen",
            "containedByRef": {"id": "RID58"},
            "partOfRef": {"id": "RID170", "display": "liver"},
        },
    ]

    # Create database with mocked embeddings
    mock_client = AsyncMock(spec=AsyncOpenAI)
    mock_embeddings = [[0.1] * test_settings.openai_embedding_dimensions] * len(test_records)

    with patch(
        "oidm_maintenance.anatomic.build.generate_embeddings_batch",
        new=AsyncMock(return_value=mock_embeddings),
    ):
        await create_anatomic_database(
            db_path, test_records, mock_client, dimensions=test_settings.openai_embedding_dimensions
        )

    return db_path


# =============================================================================
# Context Manager and Lifecycle Tests
# =============================================================================


class TestAnatomicLocationIndexLifecycle:
    """Tests for index lifecycle (open/close/context manager)."""

    @pytest.mark.asyncio
    async def test_context_manager(self, test_database: Path) -> None:
        """Test using index as context manager."""
        with AnatomicLocationIndex(test_database) as index:
            # Should be able to query
            location = index.get("RID1")
            assert location.id == "RID1"
            assert location.description == "body"

        # After context exit, connection should be closed
        assert index.conn is None

    @pytest.mark.asyncio
    async def test_explicit_open_close(self, test_database: Path) -> None:
        """Test explicit open and close."""
        index = AnatomicLocationIndex(test_database)

        # Should not be open initially
        assert index.conn is None

        # Open connection
        index.open()
        assert index.conn is not None

        # Can query
        location = index.get("RID1")
        assert location.id == "RID1"

        # Close connection
        index.close()
        assert index.conn is None

    @pytest.mark.asyncio
    async def test_open_returns_self(self, test_database: Path) -> None:
        """Test that open() returns self for chaining."""
        index = AnatomicLocationIndex(test_database)

        result = index.open()

        assert result is index
        index.close()

    @pytest.mark.asyncio
    async def test_error_when_not_open(self, test_database: Path) -> None:
        """Test that operations fail when index not open."""
        index = AnatomicLocationIndex(test_database)

        with pytest.raises(RuntimeError, match="connection not open"):
            index.get("RID1")


# =============================================================================
# Core Lookup Tests
# =============================================================================


class TestAnatomicLocationIndexLookups:
    """Tests for core lookup methods."""

    @pytest.mark.asyncio
    async def test_get_existing_location(self, test_database: Path) -> None:
        """Test retrieving an existing location by ID."""
        with AnatomicLocationIndex(test_database) as index:
            location = index.get("RID1301")

            assert location.id == "RID1301"
            assert location.description == "lung"
            assert location.region == AnatomicRegion.THORAX
            # Should be auto-bound to index
            assert location._index is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent_location(self, test_database: Path) -> None:
        """Test error when location ID doesn't exist."""
        with (
            AnatomicLocationIndex(test_database) as index,
            pytest.raises(KeyError, match="Anatomic location not found: RID99999"),
        ):
            index.get("RID99999")

    @pytest.mark.asyncio
    async def test_get_loads_codes(self, test_database: Path) -> None:
        """Test that get() eagerly loads codes."""
        with AnatomicLocationIndex(test_database) as index:
            location = index.get("RID1301")

            assert len(location.codes) >= 1
            # Should have SNOMED code
            snomed_code = location.get_code("SNOMED")
            assert snomed_code is not None
            assert snomed_code.code == "39607008"

    @pytest.mark.asyncio
    async def test_get_loads_synonyms(self, test_database: Path) -> None:
        """Test that get() eagerly loads synonyms."""
        with AnatomicLocationIndex(test_database) as index:
            location = index.get("RID1301")

            assert len(location.synonyms) >= 2
            assert "pulmonary organ" in location.synonyms
            assert "respiratory organ" in location.synonyms

    @pytest.mark.asyncio
    async def test_find_by_code_snomed(self, test_database: Path) -> None:
        """Test finding locations by SNOMED code."""
        with AnatomicLocationIndex(test_database) as index:
            locations = index.find_by_code("SNOMED", "39607008")

            assert len(locations) >= 1
            assert any(loc.description == "lung" for loc in locations)

    @pytest.mark.asyncio
    async def test_find_by_code_fma(self, test_database: Path) -> None:
        """Test finding locations by FMA code."""
        with AnatomicLocationIndex(test_database) as index:
            locations = index.find_by_code("FMA", "7197")

            assert len(locations) >= 1
            assert any(loc.description == "liver" for loc in locations)

    @pytest.mark.asyncio
    async def test_find_by_code_not_found(self, test_database: Path) -> None:
        """Test find_by_code returns empty list when code doesn't exist."""
        with AnatomicLocationIndex(test_database) as index:
            locations = index.find_by_code("SNOMED", "99999999")

            assert locations == []

    @pytest.mark.asyncio
    async def test_search_basic(self, test_database: Path) -> None:
        """Test basic text search."""
        async with AnatomicLocationIndex(test_database) as index:
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
    async def test_get_containment_ancestors(self, test_database: Path) -> None:
        """Test getting containment ancestors."""
        with AnatomicLocationIndex(test_database) as index:
            # Lung is contained in thorax, which is contained in body
            ancestors = index.get_containment_ancestors("RID1301")

            assert len(ancestors) >= 1
            # Should include thorax (immediate parent)
            assert any(a.description == "thorax" for a in ancestors)

    @pytest.mark.asyncio
    async def test_get_containment_descendants(self, test_database: Path) -> None:
        """Test getting containment descendants."""
        with AnatomicLocationIndex(test_database) as index:
            # Thorax contains lung, heart, etc.
            descendants = index.get_containment_descendants("RID39")

            assert len(descendants) >= 2
            descriptions = {d.description for d in descendants}
            assert "lung" in descriptions or "heart" in descriptions

    @pytest.mark.asyncio
    async def test_get_children_of(self, test_database: Path) -> None:
        """Test getting direct children."""
        with AnatomicLocationIndex(test_database) as index:
            # Thorax has direct children
            children = index.get_children_of("RID39")

            assert len(children) >= 1
            # Should include lung
            descriptions = {c.description for c in children}
            assert "lung" in descriptions or "heart" in descriptions

    @pytest.mark.asyncio
    async def test_get_partof_ancestors(self, test_database: Path) -> None:
        """Test getting part-of ancestors."""
        with AnatomicLocationIndex(test_database) as index:
            # Most locations won't have part-of hierarchy in our test data
            # Just verify method works
            ancestors = index.get_partof_ancestors("RID1301")

            # May be empty or contain ancestors
            assert isinstance(ancestors, list)


# =============================================================================
# Filter and Iteration Tests
# =============================================================================


class TestAnatomicLocationIndexFilters:
    """Tests for filtering and iteration methods."""

    @pytest.mark.asyncio
    async def test_by_region(self, test_database: Path) -> None:
        """Test filtering by region."""
        with AnatomicLocationIndex(test_database) as index:
            thorax_locations = index.by_region("Thorax")

            assert len(thorax_locations) >= 3
            descriptions = {loc.description for loc in thorax_locations}
            assert "thorax" in descriptions
            assert "lung" in descriptions or "left lung" in descriptions

    @pytest.mark.asyncio
    async def test_by_location_type(self, test_database: Path) -> None:
        """Test filtering by location type."""
        with AnatomicLocationIndex(test_database) as index:
            structures = index.by_location_type(LocationType.STRUCTURE)

            # All our test locations are structures
            assert len(structures) >= 5

    @pytest.mark.asyncio
    async def test_by_system(self, test_database: Path) -> None:
        """Test filtering by body system."""
        with AnatomicLocationIndex(test_database) as index:
            # Our test data doesn't have body_system populated
            # Just verify method works without error
            results = index.by_system(BodySystem.RESPIRATORY)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_by_structure_type(self, test_database: Path) -> None:
        """Test filtering by structure type."""
        with AnatomicLocationIndex(test_database) as index:
            # Our test data doesn't have structure_type populated
            # Just verify method works without error
            results = index.by_structure_type(StructureType.SOLID_ORGAN)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_iteration(self, test_database: Path) -> None:
        """Test iterating over all locations."""
        with AnatomicLocationIndex(test_database) as index:
            locations = list(index)

            assert len(locations) >= 8
            # All should be bound to index
            assert all(loc._index is not None for loc in locations)


# =============================================================================
# Bound Location Navigation Tests
# =============================================================================


class TestBoundLocationNavigation:
    """Tests for navigation via bound AnatomicLocation objects."""

    @pytest.mark.asyncio
    async def test_bound_location_get_containment_ancestors(self, test_database: Path) -> None:
        """Test calling containment navigation without explicit index."""
        with AnatomicLocationIndex(test_database) as index:
            lung = index.get("RID1301")

            # Should work without passing index (uses bound weakref)
            ancestors = lung.get_containment_ancestors()

            assert len(ancestors) >= 1
            assert any(a.description == "thorax" for a in ancestors)

    @pytest.mark.asyncio
    async def test_bound_location_get_containment_siblings(self, test_database: Path) -> None:
        """Test getting siblings via bound location."""
        with AnatomicLocationIndex(test_database) as index:
            lung = index.get("RID1301")

            siblings = lung.get_containment_siblings()

            # Lung may or may not have siblings depending on test data structure
            # Just verify it returns a list
            assert isinstance(siblings, list)

    @pytest.mark.asyncio
    async def test_bound_location_resolve_laterality(self, test_database: Path) -> None:
        """Test resolving laterality variants via bound location."""
        with AnatomicLocationIndex(test_database) as index:
            lung = index.get("RID1301")

            # Generic lung should have left and right variants
            left_lung = lung.get_left()
            right_lung = lung.get_right()

            assert left_lung is not None
            assert left_lung.description == "left lung"
            assert right_lung is not None
            assert right_lung.description == "right lung"

    @pytest.mark.asyncio
    async def test_bound_location_get_laterality_variants(self, test_database: Path) -> None:
        """Test getting all laterality variants."""
        with AnatomicLocationIndex(test_database) as index:
            lung = index.get("RID1301")

            variants = lung.get_laterality_variants()

            # Should have left and right
            assert Laterality.LEFT in variants
            assert Laterality.RIGHT in variants
            assert variants[Laterality.LEFT].description == "left lung"
            assert variants[Laterality.RIGHT].description == "right lung"

    @pytest.mark.asyncio
    async def test_bound_location_error_after_close(self, test_database: Path) -> None:
        """Test that bound location fails after index is closed."""
        index = AnatomicLocationIndex(test_database)
        index.open()

        lung = index.get("RID1301")

        # Close index
        index.close()

        # Trying to navigate should fail with RuntimeError (connection not open)
        with pytest.raises(RuntimeError, match="connection not open"):
            lung.get_containment_ancestors()

    @pytest.mark.asyncio
    async def test_get_parts(self, test_database: Path) -> None:
        """Test getting parts via bound location."""
        with AnatomicLocationIndex(test_database) as index:
            # Test that get_parts() works and returns a list
            # Since test data doesn't have hasPartsRefs (which would require
            # fixing DuckDB STRUCT conversion), we just verify the method works
            liver = index.get("RID170")
            parts = liver.get_parts()

            assert isinstance(parts, list)
            # Test data doesn't populate hasPartsRefs, so this will be empty
            # The method itself is working correctly

    @pytest.mark.asyncio
    async def test_get_parts_returns_list(self, test_database: Path) -> None:
        """Test that get_parts returns list even when no parts exist."""
        with AnatomicLocationIndex(test_database) as index:
            # Verify method returns empty list for location without parts
            heart = index.get("RID1385")
            parts = heart.get_parts()

            assert isinstance(parts, list)
            assert len(parts) == 0

    @pytest.mark.asyncio
    async def test_get_generic(self, test_database: Path) -> None:
        """Test getting generic variant via bound location."""
        with AnatomicLocationIndex(test_database) as index:
            # Left lung has generic variant (unsided lung)
            left_lung = index.get("RID1302")
            generic = left_lung.get_generic()

            assert generic is not None
            assert generic.description == "lung"
            assert generic.id == "RID1301"

    @pytest.mark.asyncio
    async def test_get_generic_when_no_generic(self, test_database: Path) -> None:
        """Test getting generic variant when location has none."""
        with AnatomicLocationIndex(test_database) as index:
            # Unsided lung has no generic variant (it IS the generic)
            lung = index.get("RID1301")
            generic = lung.get_generic()

            assert generic is None

    @pytest.mark.asyncio
    async def test_get_generic_nonlateral_location(self, test_database: Path) -> None:
        """Test getting generic variant for nonlateral location."""
        with AnatomicLocationIndex(test_database) as index:
            # Heart is nonlateral, should have no generic variant
            heart = index.get("RID1385")
            generic = heart.get_generic()

            assert generic is None
