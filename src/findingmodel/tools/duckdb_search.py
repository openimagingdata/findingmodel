"""DuckDB-based ontology search client for anatomic locations."""

from pathlib import Path
from typing import Any

import duckdb
from openai import AsyncOpenAI

from findingmodel.config import settings
from findingmodel.tools.duckdb_utils import get_embedding_for_duckdb, rrf_fusion, setup_duckdb_connection
from findingmodel.tools.ontology_search import OntologySearchProtocol, OntologySearchResult


class DuckDBOntologySearchClient(OntologySearchProtocol):
    """Simple DuckDB search client for anatomic locations."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the client.

        Args:
            db_path: Path to DuckDB database (defaults to config setting)
        """
        if db_path:
            self.db_path = Path(db_path)  # Honor explicit path
        else:
            # Use package data directory with optional download
            from findingmodel.config import ensure_anatomic_db

            self.db_path = ensure_anatomic_db()
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._openai_client: AsyncOpenAI | None = None

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

    async def __aenter__(self) -> "DuckDBOntologySearchClient":
        """Enter async context."""
        if self.conn is None:
            self.conn = setup_duckdb_connection(self.db_path, read_only=True)

        if self._openai_client is None and settings.openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,  # noqa: ANN401
    ) -> None:
        """Exit async context."""
        if self.conn:
            self.conn.close()
            self.conn = None

    async def search(
        self,
        queries: list[str],
        max_results: int = 30,
        filter_anatomical: bool = True,
    ) -> list[OntologySearchResult]:
        """Search for anatomic locations.

        Args:
            queries: List of search queries
            max_results: Maximum total results to return
            filter_anatomical: Ignored (always returns anatomical)

        Returns:
            List of OntologySearchResult objects
        """
        if not self.conn:
            raise RuntimeError("Not connected. Use async with block.")

        all_results = []
        results_per_query = max(1, max_results // len(queries)) if queries else max_results

        for query in queries:
            results = await self._search_single(query, limit=results_per_query)
            all_results.extend(results)

        # Deduplicate by concept_id
        seen = set()
        unique_results = []
        for result in all_results:
            if result.concept_id not in seen:
                seen.add(result.concept_id)
                unique_results.append(result)

        # Sort by score and limit
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:max_results]

    async def search_parallel(
        self,
        queries: list[str],
        tables: list[str] | None = None,
        limit_per_query: int = 30,
        filter_anatomical: bool = False,
    ) -> list[OntologySearchResult]:
        """Search multiple queries (sequentially for DuckDB safety).

        Args:
            queries: List of search queries
            tables: Ignored (always searches anatomic_locations)
            limit_per_query: Maximum results per query
            filter_anatomical: Ignored (always returns anatomical)

        Returns:
            Combined list of OntologySearchResult objects
        """
        if not self.conn:
            raise RuntimeError("Not connected. Use async with block.")

        # Run queries sequentially (DuckDB isn't thread-safe)
        results_lists = []
        for query in queries:
            results = await self._search_single(query, limit=limit_per_query)
            results_lists.append(results)

        # Combine and deduplicate
        seen = set()
        all_results = []
        for results in results_lists:
            for result in results:
                if result.concept_id not in seen:
                    seen.add(result.concept_id)
                    all_results.append(result)

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results

    async def search_with_filters(
        self,
        queries: list[str],
        region: str | None = None,
        sided_filter: list[str] | None = None,
        limit_per_query: int = 30,
    ) -> list[OntologySearchResult]:
        """Search with region and sided filtering.

        Args:
            queries: List of search queries
            region: Optional region to filter by
            sided_filter: List of allowed sided values (e.g., ['generic', 'nonlateral'])
            limit_per_query: Maximum results per query

        Returns:
            Combined filtered list of OntologySearchResult objects
        """
        if not self.conn:
            raise RuntimeError("Not connected. Use async with block.")

        # Run queries with filters
        results_lists = []
        for query in queries:
            results = await self._search_single_with_filters(
                query, region=region, sided_filter=sided_filter, limit=limit_per_query
            )
            results_lists.append(results)

        # Combine and deduplicate
        seen = set()
        all_results = []
        for results in results_lists:
            for result in results:
                if result.concept_id not in seen:
                    seen.add(result.concept_id)
                    all_results.append(result)

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results

    async def _search_single(self, query: str, limit: int = 30) -> list[OntologySearchResult]:
        """Search for a single query using hybrid search with exact match priority.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of OntologySearchResult objects with exact matches first
        """
        if not self.conn:
            raise RuntimeError("Not connected")

        # First, check for exact matches on description or synonyms
        exact_matches = self._find_exact_matches(query)

        # Get embedding for vector search
        embedding = await self._get_embedding(query)

        # FTS search
        fts_results = self.conn.execute(
            """
                WITH fts AS (
                    SELECT 
                        id,
                        description,
                        synonyms,
                        fts_main_anatomic_locations.match_bm25(id, ?) as score
                    FROM anatomic_locations
                    WHERE score IS NOT NULL
                    ORDER BY score DESC
                    LIMIT ?
            )
            SELECT * FROM fts
        """,
            [query, limit * 2],
        ).fetchall()

        # Vector search (if we have embeddings)
        vector_results = []
        if embedding:
            # Convert embedding list to FLOAT[dimensions] for DuckDB
            dimensions = settings.openai_embedding_dimensions
            vector_results = self.conn.execute(
                f"""
                SELECT
                    id,
                    description,
                    synonyms,
                    array_cosine_distance(vector, ?::FLOAT[{dimensions}]) as score
                FROM anatomic_locations
                WHERE vector IS NOT NULL
                ORDER BY score ASC
                LIMIT ?
            """,
                [embedding, limit * 2],
            ).fetchall()

        # Apply RRF fusion or use FTS only
        results = self._apply_rrf_fusion(fts_results, vector_results, limit)

        # Convert to OntologySearchResult objects
        ontology_results = [self._row_to_result(row, row[3]) for row in results]

        # Combine exact matches with other results
        return self._combine_with_exact_matches(exact_matches, ontology_results, limit)

    def _find_exact_matches(
        self, query: str, where_clause: str | None = None, params: list[Any] | None = None
    ) -> list[OntologySearchResult]:
        """Find exact matches on description or in synonyms array.

        Args:
            query: Text to match exactly (case-insensitive)
            where_clause: Additional WHERE conditions
            params: Parameters for WHERE clause

        Returns:
            List of exact matches
        """
        if not self.conn:
            return []

        query_lower = query.lower()
        params = params or []

        # Build base WHERE clause
        base_where = "LOWER(description) = ?"
        synonym_where = "list_contains(list_transform(synonyms, x -> LOWER(x)), ?)"

        if where_clause:
            base_where = f"{base_where} AND {where_clause}"
            synonym_where = f"{synonym_where} AND {where_clause}"

        # Check for exact match on description (case-insensitive)
        exact_results = self.conn.execute(
            f"""
            SELECT id, description, synonyms
            FROM anatomic_locations
            WHERE {base_where}
        """,
            [query_lower, *params],
        ).fetchall()

        # Also check if query matches any synonym
        synonym_results = self.conn.execute(
            f"""
            SELECT id, description, synonyms
            FROM anatomic_locations
            WHERE {synonym_where}
        """,
            [query_lower, *params],
        ).fetchall()

        # Combine and deduplicate
        seen_ids = set()
        matches = []

        for results in [exact_results, synonym_results]:
            for row in results:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    matches.append(self._row_to_result(row, 1.0))

        return matches

    def _row_to_result(self, row: tuple[Any, ...], score: float) -> OntologySearchResult:
        """Convert a database row to an OntologySearchResult.

        Args:
            row: Tuple with (id, description, synonyms, ...) from database
            score: Score for this result

        Returns:
            OntologySearchResult object
        """
        return OntologySearchResult(
            concept_id=row[0],
            concept_text=row[1],  # description only
            score=float(score),
            table_name="anatomic_locations",
        )

    def _apply_rrf_fusion(
        self,
        fts_results: list[tuple[Any, ...]],
        vector_results: list[tuple[Any, ...]],
        limit: int,
    ) -> list[tuple[Any, ...]]:
        """Apply Reciprocal Rank Fusion to combine FTS and vector search results.

        Args:
            fts_results: List of tuples (id, description, synonyms, score) from FTS
            vector_results: List of tuples (id, description, synonyms, score) from vector search
            limit: Maximum number of results to return

        Returns:
            Combined results sorted by RRF score
        """
        # If no vector results, just return FTS results
        if not vector_results:
            return fts_results[:limit]

        # Convert to (id, score) format for rrf_fusion utility
        fts_scores = [(r[0], r[3]) for r in fts_results]
        vec_scores = [(r[0], r[3]) for r in vector_results]

        # Apply RRF fusion using utility function
        fused_scores = rrf_fusion(fts_scores, vec_scores)

        # Build result lookup by ID
        result_map = {}
        for r in fts_results + vector_results:
            if r[0] not in result_map:
                result_map[r[0]] = r

        # Reconstruct results with RRF scores
        combined_results = []
        for id, score in fused_scores[:limit]:
            if id in result_map:
                row = result_map[id]
                combined_results.append((*row[:3], score))

        return combined_results

    def _combine_with_exact_matches(
        self, exact_matches: list[OntologySearchResult], other_results: list[OntologySearchResult], limit: int
    ) -> list[OntologySearchResult]:
        """Combine exact matches with other results, prioritizing exact matches.

        Args:
            exact_matches: List of exact match results (will get score=1.0)
            other_results: List of other search results
            limit: Maximum total results

        Returns:
            Combined list with exact matches first, limited to max results
        """
        if not exact_matches:
            return other_results[:limit]

        # Remove exact match IDs from other results to avoid duplicates
        exact_ids = {r.concept_id for r in exact_matches}
        filtered_results = [r for r in other_results if r.concept_id not in exact_ids]

        # Put exact matches first with perfect scores
        for match in exact_matches:
            match.score = 1.0  # Perfect score for exact matches

        # Combine: exact matches first, then other results up to limit
        combined_results = exact_matches + filtered_results
        return combined_results[:limit]

    async def _search_single_with_filters(
        self, query: str, region: str | None = None, sided_filter: list[str] | None = None, limit: int = 30
    ) -> list[OntologySearchResult]:
        """Search with region and sided filters applied.

        Args:
            query: Search query
            region: Optional region to filter by
            sided_filter: List of allowed sided values (e.g., ['generic', 'nonlateral'])
            limit: Maximum results

        Returns:
            List of filtered OntologySearchResult objects
        """
        if not self.conn:
            raise RuntimeError("Not connected")

        # Build WHERE clause for filters
        where_conditions = ["1=1"]  # Always true base condition
        params = []

        # Add region filter if specified
        if region:
            where_conditions.append("region = ?")
            params.append(region)

        # Add sided filter if specified
        if sided_filter is not None:
            placeholders = ",".join(["?" for _ in sided_filter])
            where_conditions.append(f"sided IN ({placeholders})")
            params.extend(sided_filter)

        where_clause = " AND ".join(where_conditions)

        # First, check for exact matches with filters
        exact_matches = self._find_exact_matches(query, where_clause, params)

        # Get embedding for vector search
        embedding = await self._get_embedding(query)

        # FTS search with filters
        fts_query = f"""
            WITH fts AS (
                SELECT 
                    id,
                    description,
                    synonyms,
                    fts_main_anatomic_locations.match_bm25(id, ?) as score
                FROM anatomic_locations
                WHERE score IS NOT NULL
                    AND {where_clause}
                ORDER BY score DESC
                LIMIT ?
            )
            SELECT * FROM fts
        """
        fts_results = self.conn.execute(fts_query, [query, *params, limit * 2]).fetchall()

        # Vector search with filters (if we have embeddings)
        vector_results = []
        if embedding:
            dimensions = settings.openai_embedding_dimensions
            vector_query = f"""
                SELECT
                    id,
                    description,
                    synonyms,
                    array_cosine_distance(vector, ?::FLOAT[{dimensions}]) as score
                FROM anatomic_locations
                WHERE vector IS NOT NULL
                    AND {where_clause}
                ORDER BY score ASC
                LIMIT ?
            """
            vector_results = self.conn.execute(vector_query, [embedding, *params, limit * 2]).fetchall()

        # Apply RRF fusion or use FTS only
        results = self._apply_rrf_fusion(fts_results, vector_results, limit)

        # Convert to OntologySearchResult objects
        ontology_results = [self._row_to_result(row, row[3]) for row in results]

        # Combine exact matches with other results
        return self._combine_with_exact_matches(exact_matches, ontology_results, limit)

    async def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if not available
        """
        return await get_embedding_for_duckdb(text, client=self._openai_client)
