"""Index for looking up and navigating anatomic locations.

Provides high-level query interface over DuckDB anatomic locations database.
All returned AnatomicLocation objects are automatically bound to the index.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import duckdb
from asyncer import asyncify
from oidm_common.duckdb import ReadOnlyDuckDBIndex, rrf_fusion, setup_duckdb_connection
from oidm_common.embeddings import get_embedding
from oidm_common.models import IndexCode, WebReference

from anatomic_locations.models import (
    AnatomicLocation,
    AnatomicRef,
    AnatomicRegion,
    BodySystem,
    Laterality,
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
        row = conn.execute("SELECT * FROM anatomic_locations WHERE id = ?", [location_id]).fetchone()
        assert row is not None, f"Resolved location_id {location_id} but row not found"
        return self._row_to_location(row)

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
        rows = conn.execute(
            """
            SELECT al.*
            FROM anatomic_locations al
            JOIN anatomic_codes alc ON al.id = alc.location_id
            WHERE UPPER(alc.system) = UPPER(?) AND alc.code = ?
            """,
            [system, code],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

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

        # Check for exact matches first - wrap typed sync helper
        exact_rows = await asyncify(self._find_exact_match)(conn, query, region, sided_filter)
        if exact_rows:
            return [self._row_to_location(r) for r in exact_rows[:limit]]

        # FTS search - wrap typed sync helper
        fts_rows = await asyncify(self._search_fts)(conn, query, limit * 2, region, sided_filter)

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
        semantic_rows = []
        if embedding is not None:
            semantic_rows = await asyncify(self._search_semantic)(conn, embedding, limit * 2, region, sided_filter)

        # If no semantic results, return FTS only
        if not semantic_rows:
            return [self._row_to_location(r) for r in fts_rows[:limit]]

        # Apply RRF fusion - CPU-bound, fast, stays sync (no I/O)
        fused = self._apply_rrf_fusion(fts_rows, semantic_rows, limit)
        return [self._row_to_location(r) for r in fused]

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
            # Exact match
            exact_rows = await asyncify(self._find_exact_match)(conn, query, region, sided_filter)
            if exact_rows:
                results[query] = [self._row_to_location(r) for r in exact_rows[:limit]]
                continue

            # FTS
            fts_rows = await asyncify(self._search_fts)(conn, query, limit * 2, region, sided_filter)

            # Semantic (using pre-computed embedding)
            semantic_rows: list[tuple[Any, ...]] = []
            if embedding is not None:
                semantic_rows = await asyncify(self._search_semantic)(conn, embedding, limit * 2, region, sided_filter)

            if not semantic_rows:
                results[query] = [self._row_to_location(r) for r in fts_rows[:limit]]
                continue

            fused = self._apply_rrf_fusion(fts_rows, semantic_rows, limit)
            results[query] = [self._row_to_location(r) for r in fused]

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

        # Get the location's path
        path_row = conn.execute(
            "SELECT containment_path FROM anatomic_locations WHERE id = ?",
            [location_id],
        ).fetchone()

        if not path_row or not path_row[0]:
            return []

        path = path_row[0]

        # Find all ancestors by matching their paths as prefixes
        # Order by depth descending (immediate parent first, root last)
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE ? LIKE containment_path || '%' AND id != ?
            ORDER BY containment_depth DESC
            """,
            [path, location_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def get_containment_descendants(self, location_id: str) -> list[AnatomicLocation]:
        """Get containment descendants using materialized path.

        Args:
            location_id: RID identifier

        Returns:
            List of descendant locations (may be empty)
        """
        conn = self._ensure_connection()

        # Get the location's path
        path_row = conn.execute(
            "SELECT containment_path FROM anatomic_locations WHERE id = ?",
            [location_id],
        ).fetchone()

        if not path_row or not path_row[0]:
            return []

        path = path_row[0]

        # Find all descendants by matching path prefix
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE containment_path LIKE ? || '%' AND id != ?
            ORDER BY containment_depth
            """,
            [path, location_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def get_partof_ancestors(self, location_id: str) -> list[AnatomicLocation]:
        """Get partOf ancestors using materialized path.

        Args:
            location_id: RID identifier

        Returns:
            List of part-of ancestors (may be empty)
        """
        conn = self._ensure_connection()

        # Get the location's part-of path
        path_row = conn.execute(
            "SELECT partof_path FROM anatomic_locations WHERE id = ?",
            [location_id],
        ).fetchone()

        if not path_row or not path_row[0]:
            return []

        path = path_row[0]

        # Find all part-of ancestors by matching their paths as prefixes
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE ? LIKE partof_path || '%' AND id != ?
            ORDER BY partof_depth DESC
            """,
            [path, location_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def get_children_of(self, parent_id: str) -> list[AnatomicLocation]:
        """Get direct children (containment_parent_id = parent_id).

        Args:
            parent_id: RID identifier of parent

        Returns:
            List of child locations (may be empty)
        """
        conn = self._ensure_connection()

        # Query for direct children via parent reference
        rows = conn.execute(
            """
            SELECT *
            FROM anatomic_locations
            WHERE containment_parent_id = ?
            ORDER BY description
            """,
            [parent_id],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

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
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE region = ? ORDER BY description",
            [region],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def by_location_type(self, ltype: LocationType) -> list[AnatomicLocation]:
        """Get all locations of a specific location type.

        Args:
            ltype: LocationType enum value (STRUCTURE, SPACE, REGION, etc.)

        Returns:
            List of locations with that type
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE location_type = ? ORDER BY description",
            [ltype.value],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def by_system(self, system: BodySystem) -> list[AnatomicLocation]:
        """Get all locations in a body system.

        Args:
            system: BodySystem enum value

        Returns:
            List of locations in that system
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE body_system = ? ORDER BY description",
            [system.value],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def by_structure_type(self, stype: StructureType) -> list[AnatomicLocation]:
        """Get all locations of a structure type.

        Only returns locations where location_type=STRUCTURE.

        Args:
            stype: StructureType enum value

        Returns:
            List of locations with that structure type
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT * FROM anatomic_locations WHERE structure_type = ? ORDER BY description",
            [stype.value],
        ).fetchall()

        return [self._row_to_location(row) for row in rows]

    def __iter__(self) -> Iterator[AnatomicLocation]:
        """Iterate over all anatomic locations.

        Yields:
            AnatomicLocation objects bound to this index
        """
        conn = self._ensure_connection()
        rows = conn.execute("SELECT * FROM anatomic_locations ORDER BY description").fetchall()

        for row in rows:
            yield self._row_to_location(row)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _find_exact_match(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        region: str | None,
        sided_filter: list[str] | None,
    ) -> list[tuple[Any, ...]]:
        """Find exact matches on description (sync, typed).

        Args:
            conn: DuckDB connection
            query: Text to match exactly (case-insensitive)
            region: Optional region filter
            sided_filter: Optional list of allowed laterality values

        Returns:
            List of matching rows from anatomic_locations table
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

        sql = f"SELECT * FROM anatomic_locations WHERE {where_clause}"
        return conn.execute(sql, params).fetchall()

    def _search_fts(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        limit: int,
        region: str | None,
        sided_filter: list[str] | None,
    ) -> list[tuple[Any, ...]]:
        """FTS search with BM25 scoring (sync, typed).

        Args:
            conn: DuckDB connection
            query: Search query text
            limit: Maximum number of results
            region: Optional region filter
            sided_filter: Optional list of allowed laterality values

        Returns:
            List of (id, description, ..., score) tuples sorted by BM25 score descending
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
            SELECT *, fts_main_anatomic_locations.match_bm25(id, ?, fields := 'description') as score
            FROM anatomic_locations
            WHERE {where_clause}
            ORDER BY score DESC LIMIT ?
        """
        return conn.execute(sql, params).fetchall()

    def _search_semantic(
        self,
        conn: duckdb.DuckDBPyConnection,
        embedding: list[float],
        limit: int,
        region: str | None,
        sided_filter: list[str] | None,
    ) -> list[tuple[Any, ...]]:
        """Semantic search with cosine distance (sync, typed).

        Args:
            conn: DuckDB connection
            embedding: Query embedding vector
            limit: Maximum number of results
            region: Optional region filter
            sided_filter: Optional list of allowed laterality values

        Returns:
            List of (id, description, ..., distance) tuples sorted by distance ascending
        """
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
            SELECT *, array_cosine_distance(vector, ?::FLOAT[{dimensions}]) as distance
            FROM anatomic_locations
            WHERE {where_clause}
            ORDER BY distance ASC LIMIT ?
        """
        return conn.execute(sql, params).fetchall()

    def _apply_rrf_fusion(
        self,
        fts_results: list[tuple[Any, ...]],
        semantic_results: list[tuple[Any, ...]],
        limit: int,
    ) -> list[tuple[Any, ...]]:
        """Apply RRF fusion to combine result sets (sync - CPU-bound, fast).

        Args:
            fts_results: FTS search results (rows with score in last column)
            semantic_results: Semantic search results (rows with distance in last column)
            limit: Maximum number of results to return

        Returns:
            Combined results sorted by RRF score, limited to max results
        """
        # If no semantic results, just return FTS results
        if not semantic_results:
            return fts_results[:limit]

        # Extract (id, score) tuples from results
        # FTS results have BM25 score in last column (higher is better)
        fts_scores = [(str(r[0]), float(r[-1])) for r in fts_results]

        # Semantic results have cosine distance in last column (lower is better)
        # Convert distance to similarity score (1 - distance for cosine)
        semantic_scores = [(str(r[0]), 1.0 - float(r[-1])) for r in semantic_results]

        # Apply RRF fusion using utility function
        fused_scores = rrf_fusion(fts_scores, semantic_scores)

        # Build result lookup by ID (use FTS results as base, they have all columns)
        result_map = {}
        for r in fts_results + semantic_results:
            if r[0] not in result_map:
                result_map[r[0]] = r

        # Reconstruct results with RRF scores
        combined_results = []
        for location_id, _score in fused_scores[:limit]:
            if location_id in result_map:
                combined_results.append(result_map[location_id])

        return combined_results

    def _row_to_location(self, row: tuple[object, ...]) -> AnatomicLocation:
        """Convert a database row to an AnatomicLocation object.

        Eagerly loads codes and synonyms via JOIN queries.
        Automatically binds the location to this index.

        Args:
            row: Database row from anatomic_locations table

        Returns:
            AnatomicLocation object bound to this index
        """
        conn = self._ensure_connection()

        # Map row columns to fields (based on table schema)
        # Schema: id, description, region, location_type, body_system, structure_type,
        #         laterality, definition, sex_specific, search_text, vector,
        #         containment_path, containment_parent_id, containment_parent_display,
        #         containment_depth, containment_children,
        #         partof_path, partof_parent_id, partof_parent_display,
        #         partof_depth, partof_children,
        #         left_id, left_display, right_id, right_display, generic_id, generic_display,
        #         created_at, updated_at
        location_id = str(row[0])
        description = str(row[1])
        region_str = row[2]
        location_type_str = row[3]
        body_system_str = row[4]
        structure_type_str = row[5]
        laterality_str = row[6]
        definition = str(row[7]) if row[7] else None
        sex_specific = str(row[8]) if row[8] else None
        # Skip search_text (row[9]) and vector (row[10])
        containment_path = str(row[11]) if row[11] else None
        containment_parent_id = str(row[12]) if row[12] else None
        containment_parent_display = str(row[13]) if row[13] else None
        containment_depth = cast(int, row[14]) if row[14] is not None else None
        containment_children_raw = cast(list[dict[str, str]], row[15]) if row[15] else []
        partof_path = str(row[16]) if row[16] else None
        partof_parent_id = str(row[17]) if row[17] else None
        partof_parent_display = str(row[18]) if row[18] else None
        partof_depth = cast(int, row[19]) if row[19] is not None else None
        partof_children_raw = cast(list[dict[str, str]], row[20]) if row[20] else []
        left_id = str(row[21]) if row[21] else None
        left_display = str(row[22]) if row[22] else None
        right_id = str(row[23]) if row[23] else None
        right_display = str(row[24]) if row[24] else None
        generic_id = str(row[25]) if row[25] else None
        generic_display = str(row[26]) if row[26] else None
        # Skip created_at (row[27]) and updated_at (row[28])

        # Load codes
        code_rows = conn.execute(
            "SELECT system, code, display FROM anatomic_codes WHERE location_id = ?",
            [location_id],
        ).fetchall()
        codes = [IndexCode(system=str(r[0]), code=str(r[1]), display=str(r[2]) if r[2] else None) for r in code_rows]

        # Load synonyms
        synonym_rows = conn.execute(
            "SELECT synonym FROM anatomic_synonyms WHERE location_id = ? ORDER BY synonym",
            [location_id],
        ).fetchall()
        synonyms = [str(r[0]) for r in synonym_rows]

        # Load references
        ref_rows = conn.execute(
            "SELECT url, title, description FROM anatomic_references WHERE location_id = ?",
            [location_id],
        ).fetchall()
        references = [
            WebReference(
                url=str(r[0]),
                title=str(r[1]),  # title is required in WebReference
                description=str(r[2]) if r[2] else None,
            )
            for r in ref_rows
            if r[1]  # Skip rows without title
        ]

        # Parse containment parent from denormalized columns
        containment_parent = None
        if containment_parent_id and containment_parent_display:
            containment_parent = AnatomicRef(id=containment_parent_id, display=containment_parent_display)

        # Parse containment children from STRUCT array (DuckDB returns list of dicts)
        containment_children = [
            AnatomicRef(id=str(child["id"]), display=str(child["display"]))
            for child in containment_children_raw
            if isinstance(child, dict) and "id" in child and "display" in child
        ]

        # Parse part-of parent from denormalized columns
        partof_parent = None
        if partof_parent_id and partof_parent_display:
            partof_parent = AnatomicRef(id=partof_parent_id, display=partof_parent_display)

        # Parse part-of children from STRUCT array (DuckDB returns list of dicts)
        partof_children = [
            AnatomicRef(id=str(child["id"]), display=str(child["display"]))
            for child in partof_children_raw
            if isinstance(child, dict) and "id" in child and "display" in child
        ]

        # Parse laterality variants from denormalized columns
        left_variant = None
        right_variant = None
        generic_variant = None

        if left_id and left_display:
            left_variant = AnatomicRef(id=left_id, display=left_display)
        if right_id and right_display:
            right_variant = AnatomicRef(id=right_id, display=right_display)
        if generic_id and generic_display:
            generic_variant = AnatomicRef(id=generic_id, display=generic_display)

        # Convert enum strings to enums
        region = AnatomicRegion(region_str) if region_str else None
        location_type = LocationType(location_type_str) if location_type_str else LocationType.STRUCTURE
        body_system = BodySystem(body_system_str) if body_system_str else None
        structure_type = StructureType(structure_type_str) if structure_type_str else None
        laterality = Laterality(laterality_str) if laterality_str else Laterality.NONLATERAL

        # Create and bind location
        location = AnatomicLocation(
            id=location_id,
            description=description,
            region=region,
            location_type=location_type,
            body_system=body_system,
            structure_type=structure_type,
            laterality=laterality,
            definition=definition,
            sex_specific=sex_specific,
            synonyms=synonyms,
            codes=codes,
            references=references,
            containment_path=containment_path,
            containment_parent=containment_parent,
            containment_depth=containment_depth,
            containment_children=containment_children,
            partof_path=partof_path,
            partof_parent=partof_parent,
            partof_depth=partof_depth,
            partof_children=partof_children,
            left_variant=left_variant,
            right_variant=right_variant,
            generic_variant=generic_variant,
        )

        return location.bind(self)


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
