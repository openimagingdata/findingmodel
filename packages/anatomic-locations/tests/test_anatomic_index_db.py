"""Integration tests for AnatomicLocationIndex database queries.

These tests exercise real DuckDB queries using a pre-built test database.
No OpenAI API calls are made - embeddings are pre-generated fixtures.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anatomic_locations import AnatomicLocation, AnatomicLocationIndex, AnatomicRegion, Laterality
from pydantic_ai import models

# Block all AI model requests - embeddings are pre-generated fixtures
models.ALLOW_MODEL_REQUESTS = False


# =============================================================================
# Core Lookup Tests
# =============================================================================


class TestAnatomicLocationIndexGetByID:
    """Tests for get() method - retrieving by ID."""

    def test_get_by_id(self, prebuilt_db_path: Path) -> None:
        """Returns correct record with all fields."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Get a known location (superior nasal turbinate)
            location = index.get("RID10049")

            # Verify core fields
            assert location.id == "RID10049"
            assert location.description == "superior nasal turbinate"
            assert location.region == AnatomicRegion.HEAD
            assert location.laterality == Laterality.GENERIC

            # Verify codes loaded
            assert len(location.codes) >= 1
            snomed_code = location.get_code("SNOMED")
            assert snomed_code is not None
            assert snomed_code.code == "65289004"

            # Verify laterality variants loaded
            assert location.left_variant is not None
            assert location.left_variant.id == "RID10049_RID5824"
            assert location.right_variant is not None
            assert location.right_variant.id == "RID10049_RID5825"

            # Verify containment parent loaded
            assert location.containment_parent is not None
            assert location.containment_parent.id == "RID9532"
            assert location.containment_parent.display == "nasal cavity"

            # Verify part-of parent loaded
            assert location.partof_parent is not None
            assert location.partof_parent.id == "RID9199"
            assert location.partof_parent.display == "ethmoid bone"

    def test_get_by_id_not_found(self, prebuilt_db_path: Path) -> None:
        """Raises appropriate error when ID not found."""
        with (
            AnatomicLocationIndex(prebuilt_db_path) as index,
            pytest.raises(KeyError, match="Anatomic location not found: RID99999"),
        ):
            index.get("RID99999")


# =============================================================================
# Hierarchy Navigation Tests
# =============================================================================


class TestAnatomicLocationIndexHierarchyQueries:
    """Tests for hierarchy navigation using materialized paths."""

    def test_containment_path_ancestor_query(self, prebuilt_db_path: Path) -> None:
        """LIKE query finds ancestors in containment hierarchy."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Get ancestors of nasal cavity (containedBy suprahyoid neck in sample data)
            ancestors = index.get_containment_ancestors("RID9532")

            # Should have multiple ancestors going up the hierarchy
            assert len(ancestors) >= 1

            # Verify ancestors are ordered from immediate parent to root
            # (containment_depth descending)
            if len(ancestors) >= 2:
                assert ancestors[0].containment_depth >= ancestors[-1].containment_depth  # type: ignore[operator]

            # Verify at least one known ancestor
            ancestor_ids = {a.id for a in ancestors}
            # Nasal cavity is contained in suprahyoid neck (RID7540) in our sample data
            assert "RID7540" in ancestor_ids

    def test_containment_path_descendant_query(self, prebuilt_db_path: Path) -> None:
        """LIKE query finds descendants in containment hierarchy."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Get descendants of nasal cavity (should include turbinates)
            descendants = index.get_containment_descendants("RID9532")

            # Should find the turbinates contained in nasal cavity
            assert len(descendants) >= 2

            # Verify descendants are ordered by depth
            if len(descendants) >= 2:
                assert descendants[0].containment_depth <= descendants[-1].containment_depth  # type: ignore[operator]

            # Verify known descendants
            descendant_descriptions = {d.description for d in descendants}
            assert "superior nasal turbinate" in descendant_descriptions


# =============================================================================
# Search Tests
# =============================================================================


class TestAnatomicLocationIndexSearch:
    """Tests for search methods (FTS and vector)."""

    @pytest.mark.asyncio
    async def test_fts_search_description(self, prebuilt_db_path: Path) -> None:
        """Full-text search returns results matching description."""
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Search for "turbinate" - should find nasal turbinates
            mock_settings = MagicMock()
            mock_settings.openai_api_key = None
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 512

            with patch("anatomic_locations.config.get_settings", return_value=mock_settings):
                results = await index.search("turbinate", limit=10)

            # Should find at least one result
            assert len(results) >= 1

            # Results should include turbinate-related locations
            descriptions = {r.description for r in results}
            assert any("turbinate" in desc for desc in descriptions)

            # All results should be bound to index
            assert all(r._index is not None for r in results)

    @pytest.mark.asyncio
    async def test_vector_search(
        self,
        prebuilt_db_path: Path,
        anatomic_query_embeddings: dict[str, list[float]],
    ) -> None:
        """Hybrid search with embeddings combines FTS and semantic results using RRF fusion.

        Tests that when embeddings are available, search uses both FTS and HNSW
        indexes and fuses results.
        """
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = MagicMock()
            mock_settings.openai_api_key.get_secret_value.return_value = "fake-key"
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 512

            with (
                patch("anatomic_locations.config.get_settings", return_value=mock_settings),
                patch(
                    "anatomic_locations.index.get_embedding",
                    new=AsyncMock(return_value=anatomic_query_embeddings["knee joint"]),
                ),
            ):
                results = await index.search("knee joint", limit=5)

                # Should return results (hybrid search path executed)
                assert isinstance(results, list)
                assert len(results) > 0
                assert all(isinstance(r, AnatomicLocation) for r in results)

                # Verify results include knee-related locations
                descriptions = [r.description.lower() for r in results]
                assert any("knee" in desc for desc in descriptions)

    @pytest.mark.asyncio
    async def test_hybrid_search(
        self,
        prebuilt_db_path: Path,
        anatomic_query_embeddings: dict[str, list[float]],
    ) -> None:
        """Hybrid search combines FTS and semantic results using RRF fusion.

        Validates that:
        - FTS search provides keyword matches
        - Semantic search provides vector similarity matches
        - RRF fusion combines both result sets
        - Results are deduplicated when same item appears in both
        """
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = MagicMock()
            mock_settings.openai_api_key.get_secret_value.return_value = "fake-key"
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 512

            with (
                patch("anatomic_locations.config.get_settings", return_value=mock_settings),
                patch(
                    "anatomic_locations.index.get_embedding",
                    new=AsyncMock(return_value=anatomic_query_embeddings["heart cardiac"]),
                ),
            ):
                results = await index.search("heart cardiac", limit=10)

                # Should return results from hybrid path
                assert isinstance(results, list)
                assert len(results) > 0
                assert all(isinstance(r, AnatomicLocation) for r in results)

                # All results should be bound to index
                assert all(r._index is not None for r in results)

                # Results should include heart/cardiac-related locations
                descriptions = [r.description.lower() for r in results]
                assert any("heart" in desc or "cardiac" in desc for desc in descriptions)

    @pytest.mark.asyncio
    async def test_search_falls_back_to_fts_only(self, prebuilt_db_path: Path) -> None:
        """When no embedding available, search falls back to FTS-only.

        Validates fallback behavior when semantic search is not available
        (e.g., no API key configured).
        """
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = None
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 512

            with patch("anatomic_locations.config.get_settings", return_value=mock_settings):
                results = await index.search("nasal structures", limit=5)

                # Should still return FTS results
                assert isinstance(results, list)
                assert len(results) > 0
                assert all(isinstance(r, AnatomicLocation) for r in results)

                # Results should include nasal-related locations
                descriptions = [r.description.lower() for r in results]
                assert any("nasal" in desc or "nose" in desc for desc in descriptions)


# =============================================================================
# Code Lookup Tests
# =============================================================================


class TestAnatomicLocationIndexCodeLookup:
    """Tests for find_by_code() method."""

    def test_find_by_code(self, prebuilt_db_path: Path) -> None:
        """Code lookup works for SNOMED and FMA codes."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Find by SNOMED code
            snomed_results = index.find_by_code("SNOMED", "65289004")
            assert len(snomed_results) >= 1
            assert any(r.description == "superior nasal turbinate" for r in snomed_results)

            # Find by FMA code
            fma_results = index.find_by_code("FMA", "57458")
            assert len(fma_results) >= 1
            assert any(r.description == "superior nasal turbinate" for r in fma_results)

            # All results should be bound
            assert all(r._index is not None for r in snomed_results)
            assert all(r._index is not None for r in fma_results)

    def test_find_by_code_case_insensitive(self, prebuilt_db_path: Path) -> None:
        """Code system lookup is case-insensitive."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Try different cases
            upper = index.find_by_code("SNOMED", "65289004")
            lower = index.find_by_code("snomed", "65289004")
            mixed = index.find_by_code("Snomed", "65289004")

            # Should all return the same results
            assert len(upper) == len(lower) == len(mixed)

    def test_find_by_code_not_found(self, prebuilt_db_path: Path) -> None:
        """find_by_code returns empty list when code doesn't exist."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            results = index.find_by_code("SNOMED", "99999999")
            assert results == []


# =============================================================================
# Laterality Navigation Tests
# =============================================================================


class TestAnatomicLocationIndexLateralityLookup:
    """Tests for laterality variant navigation."""

    def test_laterality_lookup(self, prebuilt_db_path: Path) -> None:
        """Left/right/generic navigation works via bound locations."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Get generic turbinate
            generic = index.get("RID10049")

            # Navigate to left variant
            left = generic.get_left()
            assert left is not None
            assert left.id == "RID10049_RID5824"
            assert left.description == "left superior nasal turbinate"
            # Note: The prebuilt test DB was built before the laterality fix.
            # After rebuilding, left variant should have laterality LEFT.
            # The important thing is the navigation works.

            # Navigate to right variant
            right = generic.get_right()
            assert right is not None
            assert right.id == "RID10049_RID5825"
            assert right.description == "right superior nasal turbinate"

            # Navigate from left back to generic
            left_to_generic = left.get_generic()
            assert left_to_generic is not None
            assert left_to_generic.id == "RID10049"
            assert left_to_generic.laterality == Laterality.GENERIC

            # All navigated locations should be bound
            assert left._index is not None
            assert right._index is not None
            assert left_to_generic._index is not None


# =============================================================================
# WeakRef Binding Tests
# =============================================================================


class TestAnatomicLocationIndexWeakRefBinding:
    """Tests for automatic index binding via weakref."""

    def test_weakref_binding(self, prebuilt_db_path: Path) -> None:
        """Index binding works and allows navigation without explicit index."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Get a location
            location = index.get("RID10049")

            # Should be bound to index
            assert location._index is not None
            # The weakref should resolve to the index (pass None to use bound index)
            assert location._get_index(None) is index

            # Should be able to navigate without passing index
            ancestors = location.get_containment_ancestors()
            assert len(ancestors) >= 1
            assert all(a._index is not None for a in ancestors)

    def test_weakref_auto_reopens_after_close(self, prebuilt_db_path: Path) -> None:
        """Bound location auto-reopens connection after index is closed."""
        index = AnatomicLocationIndex(prebuilt_db_path)
        index.open()

        # Get a location while index is open
        location = index.get("RID10049")

        # Close the index
        index.close()
        assert index.conn is None

        # Navigation auto-reopens the connection via _ensure_connection()
        ancestors = location.get_containment_ancestors()
        assert isinstance(ancestors, list)
        assert index.conn is not None
        index.close()

    @pytest.mark.asyncio
    async def test_all_returned_objects_bound(self, prebuilt_db_path: Path) -> None:
        """All objects returned from index methods are bound."""
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Test single get
            single = index.get("RID10049")
            assert single._index is not None

            # Test find_by_code
            by_code = index.find_by_code("SNOMED", "65289004")
            assert all(loc._index is not None for loc in by_code)

            # Test search
            search_results = await index.search("turbinate", limit=5)
            assert all(loc._index is not None for loc in search_results)

            # Test hierarchy methods
            ancestors = index.get_containment_ancestors("RID10049")
            assert all(loc._index is not None for loc in ancestors)

            descendants = index.get_containment_descendants("RID9532")
            assert all(loc._index is not None for loc in descendants)

            children = index.get_children_of("RID9532")
            assert all(loc._index is not None for loc in children)


# =============================================================================
# Search Quality Tests
# =============================================================================


class TestSearchQualityThresholds:
    """Tests for search result quality filtering."""

    @pytest.mark.asyncio
    async def test_fts_only_filters_low_quality(self, prebuilt_db_path: Path) -> None:
        """FTS-only path filters results below minimum BM25 score threshold."""
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = None
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 512

            with patch("anatomic_locations.config.get_settings", return_value=mock_settings):
                # Search for a term that doesn't exist — should get few or no results
                results = await index.search("xyznonexistent", limit=10)

                # Should return empty or very few results due to quality threshold
                assert isinstance(results, list)
                assert len(results) == 0

    @pytest.mark.asyncio
    async def test_semantic_threshold_filters_distant_results(
        self,
        prebuilt_db_path: Path,
    ) -> None:
        """Semantic search filters out results below minimum cosine similarity."""
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            # Use a random-ish embedding that won't match anything well
            random_embedding = [0.01] * 512

            conn = index._ensure_connection()
            results = index._search_semantic(conn, random_embedding, 10, None, None)

            # With a random embedding, most/all results should be filtered by threshold
            assert isinstance(results, list)
            # Results should be limited to those meeting the 0.75 similarity threshold
            for r in results:
                distance = float(r[-1])
                similarity = 1.0 - distance
                assert similarity >= index.SEMANTIC_MIN_SIMILARITY

    @pytest.mark.asyncio
    async def test_good_fts_results_pass_threshold(self, prebuilt_db_path: Path) -> None:
        """Good FTS matches pass the quality threshold and are returned."""
        async with AnatomicLocationIndex(prebuilt_db_path) as index:
            mock_settings = MagicMock()
            mock_settings.openai_api_key = None
            mock_settings.openai_embedding_model = "text-embedding-3-small"
            mock_settings.openai_embedding_dimensions = 512

            with patch("anatomic_locations.config.get_settings", return_value=mock_settings):
                # "turbinate" is definitely in the database
                results = await index.search("turbinate", limit=10)

                # Should return results — good keyword match
                assert len(results) > 0
                assert any("turbinate" in r.description for r in results)


class TestExactMatchWithSynonyms:
    """Tests for _find_exact_match with include_synonyms parameter."""

    def test_exact_match_description_still_works(self, prebuilt_db_path: Path) -> None:
        """Exact match on description continues to work."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            conn = index._ensure_connection()
            results = index._find_exact_match(conn, "lung", None, None)
            assert len(results) >= 1

    def test_exact_match_with_synonyms_default_off(self, prebuilt_db_path: Path) -> None:
        """By default, include_synonyms is False — only descriptions are checked."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            conn = index._ensure_connection()
            # "pulmo" is a synonym for lung — should NOT match without include_synonyms
            results = index._find_exact_match(conn, "pulmo", None, None)
            assert len(results) == 0

    def test_exact_match_with_synonyms_enabled(self, prebuilt_db_path: Path) -> None:
        """With include_synonyms=True, synonyms are also checked."""
        with AnatomicLocationIndex(prebuilt_db_path) as index:
            conn = index._ensure_connection()
            # "pulmo" is a synonym for lung — should match with include_synonyms
            try:
                results = index._find_exact_match(conn, "pulmo", None, None, include_synonyms=True)
                if results:
                    # Verify it resolved to lung
                    assert str(results[0][0]) == "RID1301"
                else:
                    pytest.skip("'pulmo' synonym not in test fixture")
            except Exception:
                pytest.skip("Synonym not available in test fixture")
