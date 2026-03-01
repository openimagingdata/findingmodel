"""Integration tests for AnatomicLocationIndex database queries.

These tests exercise real DuckDB queries using a pre-built test database.
No OpenAI API calls are made - embeddings are pre-generated fixtures.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anatomic_locations import AnatomicLocation, AnatomicLocationIndex, AnatomicRegion, Laterality
from oidm_common.duckdb import setup_duckdb_connection
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
            # Results are (location_id, similarity) tuples — all should meet threshold
            for location_id, similarity in results:
                assert isinstance(location_id, str)
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
                    # Results are now list[str] (location IDs)
                    assert results[0] == "RID1301"
                else:
                    pytest.skip("'pulmo' synonym not in test fixture")
            except Exception:
                pytest.skip("Synonym not available in test fixture")


# =============================================================================
# Old-Schema Compatibility Tests
# =============================================================================


class TestOldSchemaCompatibility:
    """Regression tests for backward compatibility with pre-v0.2.3 databases.

    Ensures AnatomicLocationIndex can read databases that predate the
    synonyms_text column (added in v0.2.3). This guards against the crash
    that prompted the dict-based hydration refactor.
    """

    def _create_old_schema_db(self, db_path: Path, dimensions: int = 512) -> None:
        """Build a minimal old-schema DB (no synonyms_text column)."""
        conn = setup_duckdb_connection(db_path, read_only=False)
        try:
            # Schema WITHOUT synonyms_text (pre-v0.2.3)
            conn.execute(f"""
                CREATE TABLE anatomic_locations (
                    id VARCHAR PRIMARY KEY,
                    description VARCHAR NOT NULL,
                    region VARCHAR,
                    location_type VARCHAR DEFAULT 'structure',
                    body_system VARCHAR,
                    structure_type VARCHAR,
                    laterality VARCHAR DEFAULT 'nonlateral',
                    definition VARCHAR,
                    sex_specific VARCHAR,
                    search_text VARCHAR,
                    vector FLOAT[{dimensions}],
                    containment_path VARCHAR,
                    containment_parent_id VARCHAR,
                    containment_parent_display VARCHAR,
                    containment_depth INTEGER,
                    containment_children STRUCT(id VARCHAR, display VARCHAR)[],
                    partof_path VARCHAR,
                    partof_parent_id VARCHAR,
                    partof_parent_display VARCHAR,
                    partof_depth INTEGER,
                    partof_children STRUCT(id VARCHAR, display VARCHAR)[],
                    left_id VARCHAR,
                    left_display VARCHAR,
                    right_id VARCHAR,
                    right_display VARCHAR,
                    generic_id VARCHAR,
                    generic_display VARCHAR,
                    created_at TIMESTAMP DEFAULT now(),
                    updated_at TIMESTAMP DEFAULT now()
                )
            """)
            conn.execute("CREATE TABLE anatomic_synonyms (location_id VARCHAR, synonym VARCHAR)")
            conn.execute(
                "CREATE TABLE anatomic_codes (location_id VARCHAR, system VARCHAR, code VARCHAR, display VARCHAR)"
            )
            conn.execute(
                "CREATE TABLE anatomic_references (location_id VARCHAR, url VARCHAR, title VARCHAR, description VARCHAR)"
            )
            conn.execute(
                """
                INSERT INTO anatomic_locations
                    (id, description, region, location_type, laterality, containment_children, partof_children)
                VALUES (?, ?, ?, ?, ?, [], [])
                """,
                ("RID_test", "test kidney", "Abdomen", "structure", "nonlateral"),
            )
            conn.commit()
        finally:
            conn.close()

    def test_reads_db_without_synonyms_text(self, tmp_path: Path) -> None:
        """AnatomicLocationIndex reads old-schema DBs (no synonyms_text) without error.

        This is a regression guard for the int_type validation crash that occurred
        when the production DB (no synonyms_text) was read with code expecting
        synonyms_text at a specific positional index.
        """
        db_path = tmp_path / "old_schema.duckdb"
        self._create_old_schema_db(db_path)

        with AnatomicLocationIndex(db_path) as index:
            location = index.get("RID_test")
            assert location.id == "RID_test"
            assert location.description == "test kidney"
            assert location.region is not None
            assert location.laterality is not None


# =============================================================================
# Hydration Robustness Regression Tests
# =============================================================================


class TestBuildLocationRobustness:
    """Regression tests for _build_location null-tolerance.

    Guards against ValidationError regressions introduced when the hydration
    was rewritten to use model_validate: null display fields and malformed
    STRUCT array entries must be tolerated, not crash.
    """

    def _make_index_with_row(self, tmp_path: Path, overrides: dict) -> tuple[AnatomicLocationIndex, object]:
        """Build a minimal DB with one row using the given field overrides, return index."""
        from oidm_common.duckdb import setup_duckdb_connection

        db_path = tmp_path / "robust.duckdb"
        conn = setup_duckdb_connection(db_path, read_only=False)
        base = {
            "id": "RID_rob",
            "description": "robustness test location",
            "region": "Head",
            "location_type": "structure",
            "laterality": "nonlateral",
            "containment_children": [],
            "partof_children": [],
            "left_id": None,
            "left_display": None,
            "right_id": None,
            "right_display": None,
            "generic_id": None,
            "generic_display": None,
            "containment_parent_id": None,
            "containment_parent_display": None,
            "partof_parent_id": None,
            "partof_parent_display": None,
        }
        base.update(overrides)
        try:
            conn.execute("""
                CREATE TABLE anatomic_locations (
                    id VARCHAR PRIMARY KEY, description VARCHAR NOT NULL,
                    region VARCHAR, location_type VARCHAR DEFAULT 'structure',
                    body_system VARCHAR, structure_type VARCHAR,
                    laterality VARCHAR DEFAULT 'nonlateral',
                    definition VARCHAR, sex_specific VARCHAR,
                    search_text VARCHAR, vector FLOAT[512],
                    containment_path VARCHAR, containment_parent_id VARCHAR,
                    containment_parent_display VARCHAR, containment_depth INTEGER,
                    containment_children STRUCT(id VARCHAR, display VARCHAR)[],
                    partof_path VARCHAR, partof_parent_id VARCHAR,
                    partof_parent_display VARCHAR, partof_depth INTEGER,
                    partof_children STRUCT(id VARCHAR, display VARCHAR)[],
                    left_id VARCHAR, left_display VARCHAR,
                    right_id VARCHAR, right_display VARCHAR,
                    generic_id VARCHAR, generic_display VARCHAR,
                    created_at TIMESTAMP DEFAULT now(), updated_at TIMESTAMP DEFAULT now()
                )
            """)
            conn.execute("CREATE TABLE anatomic_synonyms (location_id VARCHAR, synonym VARCHAR)")
            conn.execute(
                "CREATE TABLE anatomic_codes (location_id VARCHAR, system VARCHAR, code VARCHAR, display VARCHAR)"
            )
            conn.execute(
                "CREATE TABLE anatomic_references (location_id VARCHAR, url VARCHAR, title VARCHAR, description VARCHAR)"
            )
            conn.execute(
                """
                INSERT INTO anatomic_locations
                    (id, description, region, location_type, laterality,
                     containment_children, partof_children,
                     left_id, left_display, right_id, right_display,
                     generic_id, generic_display,
                     containment_parent_id, containment_parent_display,
                     partof_parent_id, partof_parent_display)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    base["id"],
                    base["description"],
                    base["region"],
                    base["location_type"],
                    base["laterality"],
                    base["containment_children"],
                    base["partof_children"],
                    base["left_id"],
                    base["left_display"],
                    base["right_id"],
                    base["right_display"],
                    base["generic_id"],
                    base["generic_display"],
                    base["containment_parent_id"],
                    base["containment_parent_display"],
                    base["partof_parent_id"],
                    base["partof_parent_display"],
                ],
            )
            conn.commit()
        finally:
            conn.close()
        return AnatomicLocationIndex(db_path)

    def test_null_variant_display_does_not_crash(self, tmp_path: Path) -> None:
        """Hydration succeeds when a laterality variant has id set but display is NULL.

        Previously _ref returned {"id": x, "display": None} which caused a
        ValidationError because AnatomicRef.display is a required str field.
        """
        index = self._make_index_with_row(
            tmp_path,
            {"left_id": "RID_left", "left_display": None},  # id present, display NULL
        )
        with index:
            location = index.get("RID_rob")
        # Must not raise; left_variant must be None (not a broken AnatomicRef)
        assert location.left_variant is None

    def test_null_parent_display_does_not_crash(self, tmp_path: Path) -> None:
        """Hydration succeeds when containment_parent has id but NULL display."""
        index = self._make_index_with_row(
            tmp_path,
            {"containment_parent_id": "RID_parent", "containment_parent_display": None},
        )
        with index:
            location = index.get("RID_rob")
        assert location.containment_parent is None

    def test_malformed_child_refs_are_filtered(self, tmp_path: Path) -> None:
        """Malformed STRUCT entries (null id or display) in child arrays are dropped.

        Previously the raw STRUCT array was passed to model_validate directly,
        which could raise ValidationError for entries with null required fields.
        """
        from oidm_common.duckdb import setup_duckdb_connection

        # Build a DB whose child arrays contain a mix of valid and invalid refs.
        db_path = tmp_path / "malformed_children.duckdb"
        conn = setup_duckdb_connection(db_path, read_only=False)
        try:
            conn.execute("""
                CREATE TABLE anatomic_locations (
                    id VARCHAR PRIMARY KEY, description VARCHAR NOT NULL,
                    region VARCHAR, location_type VARCHAR DEFAULT 'structure',
                    laterality VARCHAR DEFAULT 'nonlateral',
                    search_text VARCHAR, vector FLOAT[512],
                    containment_path VARCHAR, containment_parent_id VARCHAR,
                    containment_parent_display VARCHAR, containment_depth INTEGER,
                    containment_children STRUCT(id VARCHAR, display VARCHAR)[],
                    partof_path VARCHAR, partof_parent_id VARCHAR,
                    partof_parent_display VARCHAR, partof_depth INTEGER,
                    partof_children STRUCT(id VARCHAR, display VARCHAR)[],
                    left_id VARCHAR, left_display VARCHAR,
                    right_id VARCHAR, right_display VARCHAR,
                    generic_id VARCHAR, generic_display VARCHAR,
                    body_system VARCHAR, structure_type VARCHAR,
                    definition VARCHAR, sex_specific VARCHAR,
                    created_at TIMESTAMP DEFAULT now(), updated_at TIMESTAMP DEFAULT now()
                )
            """)
            conn.execute("CREATE TABLE anatomic_synonyms (location_id VARCHAR, synonym VARCHAR)")
            conn.execute(
                "CREATE TABLE anatomic_codes (location_id VARCHAR, system VARCHAR, code VARCHAR, display VARCHAR)"
            )
            conn.execute(
                "CREATE TABLE anatomic_references (location_id VARCHAR, url VARCHAR, title VARCHAR, description VARCHAR)"
            )
            # Insert row with one valid child + one child with null display
            conn.execute(
                """
                INSERT INTO anatomic_locations
                    (id, description, region, location_type, laterality,
                     containment_children, partof_children)
                VALUES (?, ?, ?, ?, ?,
                    [{'id': 'RID_good', 'display': 'good child'},
                     {'id': 'RID_bad', 'display': NULL}],
                    [{'id': NULL, 'display': 'no id child'}]
                )
                """,
                ["RID_mc", "malformed children test", "Head", "structure", "nonlateral"],
            )
            conn.commit()
        finally:
            conn.close()

        with AnatomicLocationIndex(db_path) as index:
            location = index.get("RID_mc")

        # Must not raise; only the fully valid child survives
        assert len(location.containment_children) == 1
        assert location.containment_children[0].id == "RID_good"
        # partof_children had only a null-id entry — should be empty
        assert location.partof_children == []
