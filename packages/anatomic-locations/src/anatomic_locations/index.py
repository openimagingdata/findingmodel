"""Index for looking up and navigating anatomic locations.

Provides high-level query interface over DuckDB anatomic locations database.
All returned AnatomicLocation objects are automatically bound to the index.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import duckdb
from asyncer import asyncify
from oidm_common.duckdb import ReadOnlyDuckDBIndex, rrf_fusion, setup_duckdb_connection
from oidm_common.embeddings import get_embedding

from anatomic_locations.models import (
    AnatomicLocation,
    BodySystem,
    LocationType,
    StructureType,
)


class AnatomicLocationIndex(ReadOnlyDuckDBIndex):
    """Index for looking up and navigating anatomic locations.

    Wraps DuckDB connection and provides high-level navigation API.
    Uses pre-computed materialized paths for instant hierarchy queries.

    All returned AnatomicLocation objects are automatically bound to this index
    via weakref, allowing navigation methods to be called without passing the index.

    Usage:
        # Context manager (CLI/scripts)
        with AnatomicLocationIndex() as index:
            location = index.get("RID2772")
            ancestors = location.get_containment_ancestors()  # Uses bound index

        # Auto-open (queries auto-open the connection)
        index = AnatomicLocationIndex()
        location = index.get("lung")  # opens connection automatically

        # Explicit open/close (FastAPI lifespan)
        index = AnatomicLocationIndex()
        index.open()
        # ... use index ...
        index.close()
    """

    # Quality thresholds to filter irrelevant search results
    SEMANTIC_MIN_SIMILARITY: float = 0.75  # Minimum cosine similarity (1 - distance)
    FTS_MIN_SCORE: float = 0.5  # Minimum BM25 score for FTS-only results

    # Base SELECT for all location queries. Correlated subqueries bring codes, synonyms,
    # and references inline — DuckDB optimizes these into a single join plan.
    # Note: aliased as 'refs' (not 'references') to avoid SQL reserved word.
    # Only exclude columns present in ALL schema versions; 'synonyms_text' (added v0.2.3)
    # is intentionally NOT excluded — old schemas lack it and EXCLUDE raises a Binder Error.
    _LOCATION_SELECT = """
        SELECT al.* EXCLUDE (search_text, vector),
            (SELECT list({system: system, code: code, display: display} ORDER BY system)
             FROM anatomic_codes WHERE location_id = al.id) AS codes,
            (SELECT list(synonym ORDER BY synonym)
             FROM anatomic_synonyms WHERE location_id = al.id) AS synonyms,
            (SELECT list({url: url, title: title, description: description} ORDER BY url)
             FROM anatomic_references WHERE location_id = al.id AND title IS NOT NULL) AS refs
        FROM anatomic_locations al
    """

    def __init__(self, db_path: Path | None = None) -> None:
        from anatomic_locations.config import ensure_anatomic_db

        super().__init__(db_path, ensure_db=ensure_anatomic_db)

    # =========================================================================
    # Core Lookups (all methods auto-bind returned objects)
    # =========================================================================

    def _resolve_location_id(self, conn: duckdb.DuckDBPyConnection, identifier: str) -> str | None:
        """Resolve identifier to location ID.

        Resolution order: direct ID -> case-insensitive description -> case-insensitive synonym.

        Args:
            conn: Active DuckDB connection
            identifier: RID, description, or synonym text

        Returns:
            Location ID if found, None otherwise
        """
        row = conn.execute("SELECT id FROM anatomic_locations WHERE id = ?", [identifier]).fetchone()
        if row:
            return str(row[0])
        row = conn.execute(
            "SELECT id FROM anatomic_locations WHERE LOWER(description) = LOWER(?) LIMIT 1", [identifier]
        ).fetchone()
        if row:
            return str(row[0])
        row = conn.execute(
            "SELECT location_id FROM anatomic_synonyms WHERE LOWER(synonym) = LOWER(?) LIMIT 1", [identifier]
        ).fetchone()
        if row:
            return str(row[0])
        return None

    def get(self, identifier: str) -> AnatomicLocation:
        """Get a single anatomic location by ID, description, or synonym.

        Resolution order: direct ID -> case-insensitive description -> case-insensitive synonym.
        The returned object is automatically bound to this index.

        Args:
            identifier: RID identifier (e.g., "RID2772"), description (e.g., "lung"),
                       or synonym (e.g., "pulmonary")

        Returns:
            AnatomicLocation bound to this index

        Raises:
            KeyError: If identifier not found by any strategy
        """
        conn = self._ensure_connection()
        location_id = self._resolve_location_id(conn, identifier)
        if location_id is None:
            raise KeyError(f"Anatomic location not found: {identifier}")
        return self._fetch_locations(conn, "WHERE al.id = ?", [location_id])[0]

    def find_by_code(self, system: str, code: str) -> list[AnatomicLocation]:
        """Find locations by external code (SNOMED, FMA, etc.).

        All returned objects are automatically bound to this index.

        Args:
            system: Code system (e.g., "SNOMED", "FMA")
            code: Code value in that system

        Returns:
            List of matching AnatomicLocation objects (may be empty)
        """
        conn = self._ensure_connection()
        return self._fetch_locations(
            conn,
            "WHERE al.id IN (SELECT location_id FROM anatomic_codes WHERE UPPER(system) = UPPER(?) AND code = ?)",
            [system, code],
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        region: str | None = None,
        sided_filter: list[str] | None = None,
    ) -> list[AnatomicLocation]:
        """Hybrid search combining FTS and semantic search with RRF fusion.

        All returned objects are automatically bound to this index.

        Args:
            query: Search query string
            limit: Maximum number of results to return
            region: Optional region filter (e.g., "Head", "Thorax")
            sided_filter: Optional list of allowed laterality values (e.g., ["generic", "nonlateral"])

        Returns:
            List of matching AnatomicLocation objects sorted by relevance
        """
        conn = self._ensure_connection()

        # Check for exact matches first (including synonyms) - wrap typed sync helper
        exact_ids = await asyncify(self._find_exact_match)(conn, query, region, sided_filter, include_synonyms=True)
        if exact_ids:
            return self._get_locations_by_ids(conn, exact_ids[:limit])

        # FTS search - wrap typed sync helper
        fts_results = await asyncify(self._search_fts)(conn, query, limit * 2, region, sided_filter)

        # Semantic search - embedding API already async, then wrap sync helper
        from anatomic_locations.config import get_settings as get_anatomic_settings

        anatomic_settings = get_anatomic_settings()
        api_key = anatomic_settings.openai_api_key.get_secret_value() if anatomic_settings.openai_api_key else ""
        embedding = (
            await get_embedding(
                query,
                api_key=api_key,
                model=anatomic_settings.openai_embedding_model,
                dimensions=anatomic_settings.openai_embedding_dimensions,
            )
            if api_key
            else None
        )
        semantic_results: list[tuple[str, float]] = []
        if embedding is not None:
            semantic_results = await asyncify(self._search_semantic)(conn, embedding, limit * 2, region, sided_filter)

        # If no semantic results, return FTS only (with quality threshold)
        if not semantic_results:
            filtered_ids = [lid for lid, score in fts_results if score >= self.FTS_MIN_SCORE]
            return self._get_locations_by_ids(conn, filtered_ids[:limit])

        # Apply RRF fusion - CPU-bound, fast, stays sync (no I/O)
        fused_ids = self._apply_rrf_fusion(fts_results, semantic_results, limit)
        return self._get_locations_by_ids(conn, fused_ids)

    async def search_batch(
        self,
        queries: list[str],
        *,
        limit: int = 10,
        region: str | None = None,
        sided_filter: list[str] | None = None,
    ) -> dict[str, list[AnatomicLocation]]:
        """Search multiple queries with a single embedding API call.

        Embeds ALL queries in one batch call, then runs the standard
        exact -> FTS -> semantic -> RRF fusion pipeline per query.

        Args:
            queries: List of search query strings
            limit: Maximum number of results per query
            region: Optional region filter (e.g., "Head", "Thorax")
            sided_filter: Optional list of allowed laterality values

        Returns:
            Dictionary mapping each non-blank query string to its results

        Raises:
            ValueError: If all queries are empty or whitespace-only
        """
        if not queries:
            return {}

        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            raise ValueError("All queries are empty or whitespace-only")

        conn = self._ensure_connection()

        # Batch embed all queries in a single API call
        from anatomic_locations.config import get_settings as get_anatomic_settings

        anatomic_settings = get_anatomic_settings()
        api_key = anatomic_settings.openai_api_key.get_secret_value() if anatomic_settings.openai_api_key else ""

        embeddings: list[list[float] | None] = []
        if api_key:
            from oidm_common.embeddings import get_embeddings_batch

            raw_embeddings = await get_embeddings_batch(
                valid_queries,
                api_key=api_key,
                model=anatomic_settings.openai_embedding_model,
                dimensions=anatomic_settings.openai_embedding_dimensions,
            )
            embeddings = list(raw_embeddings)
        else:
            embeddings = [None] * len(valid_queries)

        results: dict[str, list[AnatomicLocation]] = {}
        for query, embedding in zip(valid_queries, embeddings, strict=True):
            # Exact match (including synonyms)
            exact_ids = await asyncify(self._find_exact_match)(conn, query, region, sided_filter, include_synonyms=True)
            if exact_ids:
                results[query] = self._get_locations_by_ids(conn, exact_ids[:limit])
                continue

            # FTS
            fts_results = await asyncify(self._search_fts)(conn, query, limit * 2, region, sided_filter)

            # Semantic (using pre-computed embedding)
            semantic_results: list[tuple[str, float]] = []
            if embedding is not None:
                semantic_results = await asyncify(self._search_semantic)(
                    conn, embedding, limit * 2, region, sided_filter
                )

            if not semantic_results:
                filtered_ids = [lid for lid, score in fts_results if score >= self.FTS_MIN_SCORE]
                results[query] = self._get_locations_by_ids(conn, filtered_ids[:limit])
                continue

            fused_ids = self._apply_rrf_fusion(fts_results, semantic_results, limit)
            results[query] = self._get_locations_by_ids(conn, fused_ids)

        return results

    # =========================================================================
    # Hierarchy Navigation (using pre-computed paths, auto-binds results)
    # =========================================================================

    def get_containment_ancestors(self, location_id: str) -> list[AnatomicLocation]:
        """Get containedBy ancestors using materialized path.

        Returns list ordered from immediate parent to root (body).

        Args:
            location_id: RID identifier

        Returns:
            List of ancestor locations (may be empty if location has no parent)
        """
        conn = self._ensure_connection()
        path_row = self._execute_one(
            conn, "SELECT containment_path FROM anatomic_locations WHERE id = ?", [location_id]
        )
        if not path_row or not path_row["containment_path"]:
            return []
        path = str(path_row["containment_path"])
        return self._fetch_locations(
            conn,
            "WHERE ? LIKE al.containment_path || '%' AND al.id != ? ORDER BY al.containment_depth DESC",
            [path, location_id],
        )

    def get_containment_descendants(self, location_id: str) -> list[AnatomicLocation]:
        """Get containment descendants using materialized path.

        Args:
            location_id: RID identifier

        Returns:
            List of descendant locations (may be empty)
        """
        conn = self._ensure_connection()
        path_row = self._execute_one(
            conn, "SELECT containment_path FROM anatomic_locations WHERE id = ?", [location_id]
        )
        if not path_row or not path_row["containment_path"]:
            return []
        path = str(path_row["containment_path"])
        return self._fetch_locations(
            conn,
            "WHERE al.containment_path LIKE ? || '%' AND al.id != ? ORDER BY al.containment_depth",
            [path, location_id],
        )

    def get_partof_ancestors(self, location_id: str) -> list[AnatomicLocation]:
        """Get partOf ancestors using materialized path.

        Args:
            location_id: RID identifier

        Returns:
            List of part-of ancestors (may be empty)
        """
        conn = self._ensure_connection()
        path_row = self._execute_one(conn, "SELECT partof_path FROM anatomic_locations WHERE id = ?", [location_id])
        if not path_row or not path_row["partof_path"]:
            return []
        path = str(path_row["partof_path"])
        return self._fetch_locations(
            conn,
            "WHERE ? LIKE al.partof_path || '%' AND al.id != ? ORDER BY al.partof_depth DESC",
            [path, location_id],
        )

    def get_children_of(self, parent_id: str) -> list[AnatomicLocation]:
        """Get direct children (containment_parent_id = parent_id).

        Args:
            parent_id: RID identifier of parent

        Returns:
            List of child locations (may be empty)
        """
        conn = self._ensure_connection()
        return self._fetch_locations(conn, "WHERE al.containment_parent_id = ? ORDER BY al.description", [parent_id])

    # =========================================================================
    # Filtering and Iteration (auto-binds results)
    # =========================================================================

    def by_region(self, region: str) -> list[AnatomicLocation]:
        """Get all locations in a region.

        Args:
            region: Region name (e.g., "Head", "Thorax")

        Returns:
            List of locations in that region
        """
        conn = self._ensure_connection()
        return self._fetch_locations(conn, "WHERE al.region = ? ORDER BY al.description", [region])

    def by_location_type(self, ltype: LocationType) -> list[AnatomicLocation]:
        """Get all locations of a specific location type.

        Args:
            ltype: LocationType enum value (STRUCTURE, SPACE, REGION, etc.)

        Returns:
            List of locations with that type
        """
        conn = self._ensure_connection()
        return self._fetch_locations(conn, "WHERE al.location_type = ? ORDER BY al.description", [ltype.value])

    def by_system(self, system: BodySystem) -> list[AnatomicLocation]:
        """Get all locations in a body system.

        Args:
            system: BodySystem enum value

        Returns:
            List of locations in that system
        """
        conn = self._ensure_connection()
        return self._fetch_locations(conn, "WHERE al.body_system = ? ORDER BY al.description", [system.value])

    def by_structure_type(self, stype: StructureType) -> list[AnatomicLocation]:
        """Get all locations of a structure type.

        Only returns locations where location_type=STRUCTURE.

        Args:
            stype: StructureType enum value

        Returns:
            List of locations with that structure type
        """
        conn = self._ensure_connection()
        return self._fetch_locations(conn, "WHERE al.structure_type = ? ORDER BY al.description", [stype.value])

    def __iter__(self) -> Iterator[AnatomicLocation]:
        """Iterate over all anatomic locations.

        Yields:
            AnatomicLocation objects bound to this index
        """
        conn = self._ensure_connection()
        yield from self._fetch_locations(conn, "ORDER BY al.description")

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _find_exact_match(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        region: str | None,
        sided_filter: list[str] | None,
        *,
        include_synonyms: bool = False,
    ) -> list[str]:
        """Find exact matches on description, optionally also on synonyms (sync, typed).

        Args:
            conn: DuckDB connection
            query: Text to match exactly (case-insensitive)
            region: Optional region filter
            sided_filter: Optional list of allowed laterality values
            include_synonyms: If True, also check the anatomic_synonyms table

        Returns:
            List of matching location IDs
        """
        query_lower = query.lower()
        where_conditions = ["LOWER(description) = ?"]
        params: list[Any] = [query_lower]

        # Add region filter if specified
        if region is not None:
            where_conditions.append("region = ?")
            params.append(region)

        # Add laterality filter if specified
        if sided_filter is not None:
            placeholders = ",".join(["?" for _ in sided_filter])
            where_conditions.append(f"laterality IN ({placeholders})")
            params.extend(sided_filter)

        where_clause = " AND ".join(where_conditions)

        sql = f"SELECT id FROM anatomic_locations WHERE {where_clause}"
        results = conn.execute(sql, params).fetchall()

        if results:
            return [str(r[0]) for r in results]

        # If no description match and synonyms requested, check synonyms table
        if include_synonyms:
            synonym_conditions = ["LOWER(s.synonym) = ?"]
            synonym_params: list[Any] = [query_lower]

            if region is not None:
                synonym_conditions.append("a.region = ?")
                synonym_params.append(region)

            if sided_filter is not None:
                placeholders = ",".join(["?" for _ in sided_filter])
                synonym_conditions.append(f"a.laterality IN ({placeholders})")
                synonym_params.extend(sided_filter)

            synonym_where = " AND ".join(synonym_conditions)
            synonym_sql = f"""
                SELECT a.id FROM anatomic_locations a
                JOIN anatomic_synonyms s ON a.id = s.location_id
                WHERE {synonym_where}
            """
            results = conn.execute(synonym_sql, synonym_params).fetchall()
            return [str(r[0]) for r in results]

        return []

    def _search_fts(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        limit: int,
        region: str | None,
        sided_filter: list[str] | None,
    ) -> list[tuple[str, float]]:
        """FTS search with BM25 scoring (sync, typed).

        Args:
            conn: DuckDB connection
            query: Search query text
            limit: Maximum number of results
            region: Optional region filter
            sided_filter: Optional list of allowed laterality values

        Returns:
            List of (location_id, bm25_score) tuples sorted by score descending
        """
        where_conditions = ["score IS NOT NULL"]
        params: list[Any] = [query]

        # Add region filter if specified
        if region is not None:
            where_conditions.append("region = ?")
            params.append(region)

        # Add laterality filter if specified
        if sided_filter is not None:
            placeholders = ",".join(["?" for _ in sided_filter])
            where_conditions.append(f"laterality IN ({placeholders})")
            params.extend(sided_filter)

        where_clause = " AND ".join(where_conditions)
        params.append(limit)

        sql = f"""
            SELECT id, fts_main_anatomic_locations.match_bm25(id, ?) as score
            FROM anatomic_locations
            WHERE {where_clause}
            ORDER BY score DESC LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    def _search_semantic(
        self,
        conn: duckdb.DuckDBPyConnection,
        embedding: list[float],
        limit: int,
        region: str | None,
        sided_filter: list[str] | None,
    ) -> list[tuple[str, float]]:
        """Semantic search with cosine distance (sync, typed).

        Results are filtered by SEMANTIC_MIN_SIMILARITY to exclude irrelevant matches.
        Returns similarity scores (1 - distance), not raw distances.

        Args:
            conn: DuckDB connection
            embedding: Query embedding vector
            limit: Maximum number of results
            region: Optional region filter
            sided_filter: Optional list of allowed laterality values

        Returns:
            List of (location_id, similarity) tuples sorted by similarity descending
        """
        max_distance = 1.0 - self.SEMANTIC_MIN_SIMILARITY

        where_conditions = ["vector IS NOT NULL"]
        params: list[Any] = [embedding]

        # Add region filter if specified
        if region is not None:
            where_conditions.append("region = ?")
            params.append(region)

        # Add laterality filter if specified
        if sided_filter is not None:
            placeholders = ",".join(["?" for _ in sided_filter])
            where_conditions.append(f"laterality IN ({placeholders})")
            params.extend(sided_filter)

        where_clause = " AND ".join(where_conditions)
        params.append(limit)

        # Get dimensions from config
        from anatomic_locations.config import get_settings

        settings = get_settings()
        dimensions = settings.openai_embedding_dimensions

        sql = f"""
            SELECT id, array_cosine_distance(vector, ?::FLOAT[{dimensions}]) as distance
            FROM anatomic_locations
            WHERE {where_clause}
            ORDER BY distance ASC LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()

        # Filter by min similarity and convert distance to similarity
        return [(str(r[0]), 1.0 - float(r[1])) for r in rows if float(r[1]) <= max_distance]

    def _apply_rrf_fusion(
        self,
        fts_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
        limit: int,
    ) -> list[str]:
        """Apply RRF fusion to combine result sets (sync - CPU-bound, fast).

        Args:
            fts_results: FTS search results as (location_id, bm25_score) tuples
            semantic_results: Semantic search results as (location_id, similarity) tuples
            limit: Maximum number of results to return

        Returns:
            Ordered list of location IDs sorted by RRF score
        """
        # If no semantic results, just return FTS IDs
        if not semantic_results:
            return [location_id for location_id, _score in fts_results[:limit]]

        # Apply RRF fusion using utility function
        # Both inputs are already (id, score) where higher is better
        fused_scores = rrf_fusion(fts_results, semantic_results)

        return [location_id for location_id, _score in fused_scores[:limit]]

    def _fetch_locations(
        self,
        conn: duckdb.DuckDBPyConnection,
        suffix_sql: str = "",
        params: Sequence[object] | None = None,
    ) -> list[AnatomicLocation]:
        """Execute _LOCATION_SELECT + suffix_sql and hydrate results.

        Single entry point for all location queries. Codes, synonyms, and
        references arrive via correlated subqueries — one query total.
        """
        rows = self._execute_all(conn, self._LOCATION_SELECT + " " + suffix_sql, params)
        return [self._build_location(row) for row in rows]

    def _get_locations_by_ids(self, conn: duckdb.DuckDBPyConnection, location_ids: list[str]) -> list[AnatomicLocation]:
        """Fetch multiple locations by ID, preserving input order.

        IDs not found in the database are silently omitted.
        """
        if not location_ids:
            return []
        placeholders = ", ".join(["?" for _ in location_ids])
        locs = self._fetch_locations(conn, f"WHERE al.id IN ({placeholders})", location_ids)
        locs_by_id = {loc.id: loc for loc in locs}
        return [locs_by_id[lid] for lid in location_ids if lid in locs_by_id]

    def _build_location(self, row: dict[str, object]) -> AnatomicLocation:
        """Pure row→AnatomicLocation transform. No database access."""

        def _ref(id_key: str, display_key: str) -> dict[str, object] | None:
            # Both id AND display must be non-null; AnatomicRef.display is a required str.
            return (
                {"id": row[id_key], "display": row[display_key]} if row.get(id_key) and row.get(display_key) else None
            )

        def _child_refs(key: str) -> list[dict[str, object]]:
            # Filter STRUCT arrays: keep only entries where both id and display are non-null.
            raw = row.get(key)
            if not isinstance(raw, list):
                return []
            return [c for c in raw if isinstance(c, dict) and c.get("id") and c.get("display")]

        data = {
            **row,
            "containment_parent": _ref("containment_parent_id", "containment_parent_display"),
            "partof_parent": _ref("partof_parent_id", "partof_parent_display"),
            "left_variant": _ref("left_id", "left_display"),
            "right_variant": _ref("right_id", "right_display"),
            "generic_variant": _ref("generic_id", "generic_display"),
            "containment_children": _child_refs("containment_children"),
            "partof_children": _child_refs("partof_children"),
            "references": row.get("refs") or [],
            "codes": row.get("codes") or [],
            "synonyms": row.get("synonyms") or [],
        }
        return AnatomicLocation.model_validate(data).bind(self)


def get_database_stats(db_path: Path) -> dict[str, Any]:
    """Get statistics about an anatomic location database.

    Args:
        db_path: Path to the database file

    Returns:
        Dictionary with database statistics
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Get counts
        result = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
        total_records = result[0] if result else 0

        result = conn.execute("SELECT COUNT(*) FROM anatomic_locations WHERE vector IS NOT NULL").fetchone()
        vector_count = result[0] if result else 0

        # Get region count
        result = conn.execute(
            "SELECT COUNT(DISTINCT region) FROM anatomic_locations WHERE region IS NOT NULL"
        ).fetchone()
        region_count = result[0] if result else 0

        # Get laterality distribution
        laterality_dist = conn.execute("""
            SELECT laterality, COUNT(*) as count
            FROM anatomic_locations
            GROUP BY laterality
            ORDER BY count DESC
        """).fetchall()

        # Get synonym and code counts
        result = conn.execute("SELECT COUNT(*) FROM anatomic_synonyms").fetchone()
        synonym_count = result[0] if result else 0

        result = conn.execute("SELECT COUNT(*) FROM anatomic_codes").fetchone()
        code_count = result[0] if result else 0

        # Get hierarchy coverage
        result = conn.execute(
            "SELECT COUNT(*) FROM anatomic_locations WHERE containment_path IS NOT NULL AND containment_path != ''"
        ).fetchone()
        records_with_hierarchy = result[0] if result else 0

        # Get code system breakdown
        code_systems = conn.execute("""
            SELECT system, COUNT(*) as count
            FROM anatomic_codes
            GROUP BY system
            ORDER BY count DESC
        """).fetchall()

        # Get records with at least one code
        result = conn.execute("""
            SELECT COUNT(DISTINCT location_id) FROM anatomic_codes
        """).fetchone()
        records_with_codes = result[0] if result else 0

        return {
            "total_records": total_records,
            "records_with_vectors": vector_count,
            "unique_regions": region_count,
            "laterality_distribution": dict(laterality_dist),
            "total_synonyms": synonym_count,
            "total_codes": code_count,
            "records_with_hierarchy": records_with_hierarchy,
            "records_with_codes": records_with_codes,
            "code_systems": dict(code_systems),
            "file_size_mb": db_path.stat().st_size / (1024 * 1024),
        }

    finally:
        conn.close()


def _check_coverage(
    name: str, actual: int, total: int, threshold: float, expected_str: str
) -> tuple[dict[str, Any], str | None]:
    """Helper to create a coverage check result."""
    pct = (actual / total * 100) if total > 0 else 0
    passed = pct >= threshold
    check = {
        "name": name,
        "passed": passed,
        "value": f"{actual}/{total} ({pct:.1f}%)",
        "expected": expected_str,
    }
    error = None if passed else f"{name}: {pct:.1f}% (expected {expected_str})"
    return check, error


def _validate_sample_records(db_path: Path, sample_count: int) -> tuple[dict[str, Any], list[str]]:
    """Validate sample records can be retrieved and parsed."""
    sample_errors: list[str] = []
    sample_successes = 0

    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        sample_ids = conn.execute(
            "SELECT id FROM anatomic_locations ORDER BY RANDOM() LIMIT ?", [sample_count]
        ).fetchall()

        index = AnatomicLocationIndex(db_path)
        index.open()
        try:
            for (location_id,) in sample_ids:
                try:
                    location = index.get(location_id)
                    if not location.description:
                        sample_errors.append(f"{location_id}: missing description")
                    elif not location.region:
                        sample_errors.append(f"{location_id}: missing region")
                    else:
                        sample_successes += 1
                except KeyError:
                    sample_errors.append(f"{location_id}: not found")
                except Exception as e:
                    sample_errors.append(f"{location_id}: {e}")
        finally:
            index.close()
    finally:
        conn.close()

    sample_ok = sample_successes == len(sample_ids) and len(sample_errors) == 0
    check = {
        "name": "Sample Record Validation",
        "passed": sample_ok,
        "value": f"{sample_successes}/{len(sample_ids)} records parsed successfully",
        "expected": "100%",
    }
    return check, sample_errors


def run_sanity_check(db_path: Path, sample_count: int = 5) -> dict[str, Any]:
    """Run sanity checks on an anatomic location database.

    Validates that:
    1. Records can be retrieved and parsed into Pydantic models
    2. Laterality counts are consistent (left == right)
    3. Vector coverage is complete
    4. Hierarchy paths are populated

    Args:
        db_path: Path to the database file
        sample_count: Number of sample records to validate (default: 5)

    Returns:
        Dictionary with sanity check results including:
        - success: bool - overall pass/fail
        - checks: list of individual check results
        - errors: list of error messages
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    stats = get_database_stats(db_path)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    # Check 1: Vector coverage (should be 100%)
    check, error = _check_coverage(
        "Vector Coverage", stats["records_with_vectors"], stats["total_records"], 100, "100%"
    )
    checks.append(check)
    if error:
        errors.append(error)

    # Check 2: Laterality consistency (left count should equal right count)
    lat_dist = stats["laterality_distribution"]
    left_count = lat_dist.get("left", 0)
    right_count = lat_dist.get("right", 0)
    laterality_ok = left_count == right_count
    checks.append({
        "name": "Laterality Consistency",
        "passed": laterality_ok,
        "value": f"left={left_count}, right={right_count}",
        "expected": "left == right",
    })
    if not laterality_ok:
        errors.append(f"Laterality mismatch: left={left_count}, right={right_count}")

    # Check 3: Hierarchy coverage (≥90%)
    check, error = _check_coverage(
        "Hierarchy Coverage", stats["records_with_hierarchy"], stats["total_records"], 90, "≥90%"
    )
    checks.append(check)
    if error:
        errors.append(error)

    # Check 4: Reference coverage by ontology
    total = stats["total_records"]
    system_coverage = stats.get("code_systems", {})

    # Report coverage for key ontologies (informational, no threshold)
    for system in ["SNOMED", "FMA", "ACR"]:
        count = system_coverage.get(system, 0)
        pct = (count / total * 100) if total > 0 else 0
        checks.append({
            "name": f"  └ {system}",
            "passed": True,  # Informational only
            "value": f"{count}/{total} ({pct:.1f}%)",
            "expected": "—",
        })

    # Overall reference coverage check (at least 50% should have some code)
    check, error = _check_coverage("Reference Coverage", stats["records_with_codes"], total, 50, "≥50%")
    checks.append(check)
    if error:
        errors.append(error)

    # Check 5: Sample record validation
    check, sample_errors = _validate_sample_records(db_path, sample_count)
    checks.append(check)
    errors.extend(sample_errors)

    return {
        "success": all(c["passed"] for c in checks),
        "checks": checks,
        "stats": stats,
        "errors": errors,
    }


__all__ = ["AnatomicLocationIndex", "get_database_stats", "run_sanity_check"]
