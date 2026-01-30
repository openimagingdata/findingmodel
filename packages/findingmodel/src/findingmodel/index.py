"""DuckDB-backed implementation of the finding model index."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import duckdb
from asyncer import asyncify
from oidm_common.duckdb import (
    l2_to_cosine_similarity,
    normalize_scores,
    rrf_fusion,
    setup_duckdb_connection,
)
from pydantic import BaseModel, Field

from findingmodel import logger
from findingmodel.common import normalize_name
from findingmodel.config import settings
from findingmodel.contributor import Organization, Person
from findingmodel.finding_model import FindingModelBase, FindingModelFull
from findingmodel.tools.duckdb_utils import batch_embeddings_for_duckdb, get_embedding_for_duckdb

PLACEHOLDER_ATTRIBUTE_ID: str = "OIFMA_XXXX_000000"


class AttributeInfo(BaseModel):
    """Represents basic information about an attribute in a finding model."""

    attribute_id: str
    name: str
    type: str


class IndexEntry(BaseModel):
    """Represents an entry in the index with key metadata about a finding model."""

    oifm_id: str
    name: str
    slug_name: str
    filename: str
    file_hash_sha256: str
    description: str | None = None
    synonyms: list[str] | None = None
    tags: list[str] | None = None
    contributors: list[str] | None = None
    attributes: list[AttributeInfo] | None = Field(default=None, min_length=1)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def match(self, identifier: str) -> bool:
        """Check if the identifier matches the ID, name, or synonyms."""

        if self.oifm_id == identifier:
            return True
        if self.name.casefold() == identifier.casefold():
            return True
        return bool(self.synonyms and any(s.casefold() == identifier.casefold() for s in self.synonyms))


class DuckDBIndex:
    """DuckDB-based index with read-only connections."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path:
            self.db_path = Path(db_path).expanduser()  # Honor explicit path
        else:
            # Use package data directory with optional download
            from findingmodel.config import ensure_index_db

            self.db_path = ensure_index_db()
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._oifm_id_cache: dict[str, set[str]] = {}  # {source: {id, ...}}
        self._oifma_id_cache: dict[str, set[str]] = {}  # {source: {id, ...}}

    async def __aenter__(self) -> DuckDBIndex:
        """Enter async context manager, ensuring a connection is available."""

        self._ensure_connection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the database connection when leaving the context."""

        if self.conn is not None:
            self.conn.close()
            self.conn = None

    async def contains(self, identifier: str) -> bool:
        """Return True if an ID, name, or synonym exists in the index."""

        conn = self._ensure_connection()
        result = await asyncify(self._resolve_oifm_id)(conn, identifier)
        return result is not None

    async def get(self, identifier: str) -> IndexEntry | None:
        """Retrieve an index entry by ID, name, or synonym."""

        conn = self._ensure_connection()
        oifm_id = await asyncify(self._resolve_oifm_id)(conn, identifier)
        if oifm_id is None:
            return None
        return await asyncify(self._fetch_index_entry)(conn, oifm_id)

    async def get_full(self, oifm_id: str) -> FindingModelFull:
        """Get full FindingModelFull object by ID.

        Args:
            oifm_id: The OIFM ID to retrieve

        Returns:
            Full FindingModelFull object parsed from stored JSON

        Raises:
            KeyError: If model not found

        Example:
            >>> async with DuckDBIndex() as index:
            ...     model = await index.get_full("OIFM_RADLEX_000001")
            ...     # Returns complete FindingModelFull with all attributes
        """
        conn = self._ensure_connection()
        json_text = await asyncify(self._query_full_model)(conn, oifm_id)

        if not json_text:
            raise KeyError(f"Model not found: {oifm_id}")

        return FindingModelFull.model_validate_json(json_text)

    async def get_full_batch(self, oifm_ids: list[str]) -> dict[str, FindingModelFull]:
        """Get multiple full models efficiently.

        Args:
            oifm_ids: List of OIFM IDs to retrieve

        Returns:
            Dict mapping OIFM ID to FindingModelFull object. Only includes models that were found.

        Example:
            >>> async with DuckDBIndex() as index:
            ...     models = await index.get_full_batch(["OIFM_RADLEX_000001", "OIFM_CUSTOM_000042"])
            >>> # Returns {oifm_id: FindingModelFull, ...}
        """
        if not oifm_ids:
            return {}

        conn = self._ensure_connection()
        results = await asyncify(self._query_full_models_batch)(conn, oifm_ids)

        return {oifm_id: FindingModelFull.model_validate_json(json_text) for oifm_id, json_text in results}

    async def count(self) -> int:
        """Return the number of finding models in the index."""

        conn = self._ensure_connection()
        return await asyncify(self._query_count)(conn, "finding_models")

    async def count_people(self) -> int:
        """Return the number of people in the normalized table."""

        conn = self._ensure_connection()
        return await asyncify(self._query_count)(conn, "people")

    async def count_organizations(self) -> int:
        """Return the number of organizations in the normalized table."""

        conn = self._ensure_connection()
        return await asyncify(self._query_count)(conn, "organizations")

    async def get_person(self, github_username: str) -> Person | None:
        """Retrieve a person by GitHub username."""

        conn = self._ensure_connection()
        row = await asyncify(self._query_person)(conn, github_username)
        if row is None:
            return None
        return Person.model_validate({
            "github_username": row[0],
            "name": row[1],
            "email": row[2],
            "organization_code": row[3],
            "url": row[4],
        })

    async def get_organization(self, code: str) -> Organization | None:
        """Retrieve an organization by code."""

        conn = self._ensure_connection()
        row = await asyncify(self._query_organization)(conn, code)
        if row is None:
            return None
        return Organization.model_validate({"code": row[0], "name": row[1], "url": row[2]})

    async def get_people(self) -> list[Person]:
        """Retrieve all people from the index."""
        conn = self._ensure_connection()
        rows = await asyncify(self._query_all_people)(conn)
        return [
            Person.model_validate({
                "github_username": row[0],
                "name": row[1],
                "email": row[2],
                "organization_code": row[3],
                "url": row[4],
            })
            for row in rows
        ]

    async def get_organizations(self) -> list[Organization]:
        """Retrieve all organizations from the index."""
        conn = self._ensure_connection()
        rows = await asyncify(self._query_all_organizations)(conn)
        return [Organization.model_validate({"code": row[0], "name": row[1], "url": row[2]}) for row in rows]

    async def all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "name",
        order_dir: Literal["asc", "desc"] = "asc",
    ) -> tuple[list[IndexEntry], int]:
        """Get all finding models with pagination.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            order_by: Field to sort by ("name", "oifm_id", "created_at", "updated_at", "slug_name")
            order_dir: Sort direction ("asc" or "desc")

        Returns:
            Tuple of (list of IndexEntry objects, total count)

        Raises:
            ValueError: If order_by field is invalid

        Example:
            # Get page 3 (items 41-60) sorted by name
            models, total = index.all(limit=20, offset=40, order_by="name")
            print(f"Showing {len(models)} of {total} total models")
        """
        # Validate order_by field
        valid_fields = {"name", "oifm_id", "created_at", "updated_at", "slug_name"}
        if order_by not in valid_fields:
            raise ValueError(f"Invalid order_by field: {order_by}")

        # Validate order_dir
        if order_dir not in {"asc", "desc"}:
            raise ValueError(f"Invalid order_dir: {order_dir}")

        # Build order clause (use LOWER() for case-insensitive sorting on text fields)
        order_clause = f"LOWER({order_by})" if order_by in {"name", "slug_name"} else order_by
        order_clause = f"{order_clause} {order_dir.upper()}"

        conn = self._ensure_connection()
        # Use helper to execute query (no WHERE clause for list all)
        return await asyncify(self._execute_paginated_query)(
            conn, order_clause=order_clause, limit=limit, offset=offset
        )

    async def search_by_slug(
        self,
        pattern: str,
        limit: int = 100,
        offset: int = 0,
        match_type: Literal["exact", "prefix", "contains"] = "contains",
    ) -> tuple[list[IndexEntry], int]:
        """Search finding models by slug name pattern.

        Args:
            pattern: Search pattern (will be normalized via normalize_name)
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            match_type: How to match the pattern:
                - "exact": Exact match on slug_name
                - "prefix": slug_name starts with pattern
                - "contains": slug_name contains pattern (default)

        Returns:
            Tuple of (list of matching IndexEntry objects, total count)

        Example:
            # User searches for "abscess" - find all models with "abscess" in slug
            models, total = index.search_by_slug("abscess", limit=20, offset=0)
            # Internally: WHERE slug_name LIKE '%abscess%' LIMIT 20 OFFSET 0
        """
        # Build WHERE clause using helper
        where_clause, sql_pattern, normalized = self._build_slug_search_clause(pattern, match_type)

        # Build ORDER BY clause for relevance ranking
        order_clause = """
            CASE
                WHEN slug_name = ? THEN 0
                WHEN slug_name LIKE ? THEN 1
                ELSE 2
            END,
            LOWER(name)
        """

        conn = self._ensure_connection()
        # Use helper to execute query
        return await asyncify(self._execute_paginated_query)(
            conn,
            where_clause=where_clause,
            where_params=[sql_pattern],
            order_clause=order_clause,
            order_params=[normalized, f"{normalized}%"],
            limit=limit,
            offset=offset,
        )

    async def count_search(self, pattern: str, match_type: Literal["exact", "prefix", "contains"] = "contains") -> int:
        """Get count of finding models matching search pattern.

        Args:
            pattern: Search pattern (will be normalized)
            match_type: How to match the pattern

        Returns:
            Number of matching finding models

        Example:
            count = index.count_search("abscess", match_type="contains")
            print(f"Found {count} models matching 'abscess'")
        """
        # Build WHERE clause using helper
        where_clause, sql_pattern, _ = self._build_slug_search_clause(pattern, match_type)

        conn = self._ensure_connection()
        return await asyncify(self._query_search_count)(conn, where_clause, sql_pattern)

    def _build_slug_search_clause(
        self, pattern: str, match_type: Literal["exact", "prefix", "contains"]
    ) -> tuple[str, str, str]:
        """Build WHERE clause and patterns for slug matching.

        Args:
            pattern: Search pattern (will be normalized)
            match_type: How to match the pattern

        Returns:
            (where_clause, sql_pattern, normalized_pattern) tuple

        Example:
            where, sql_pat, norm = self._build_slug_search_clause("abscess", "contains")
            # ("slug_name LIKE ?", "%abscess%", "abscess")
        """
        normalized = normalize_name(pattern)

        if match_type == "exact":
            return ("slug_name = ?", normalized, normalized)
        elif match_type == "prefix":
            return ("slug_name LIKE ?", f"{normalized}%", normalized)
        else:  # contains
            return ("slug_name LIKE ?", f"%{normalized}%", normalized)

    def _execute_paginated_query(
        self,
        conn: duckdb.DuckDBPyConnection,
        where_clause: str = "",
        where_params: list[object] | None = None,
        order_clause: str = "LOWER(name)",
        order_params: list[object] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[IndexEntry], int]:
        """Execute paginated query with count and result fetching (sync - types preserved).

        Shared by list() and search_by_slug() to eliminate duplication.

        Args:
            conn: Active database connection
            where_clause: SQL WHERE clause (without WHERE keyword)
            where_params: Parameters for WHERE clause
            order_clause: SQL ORDER BY clause (without ORDER BY keyword)
            order_params: Parameters for ORDER BY clause (e.g., for CASE expressions)
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            (list of IndexEntry objects, total count) tuple
        """
        where_params = where_params or []
        order_params = order_params or []
        where_sql = f"WHERE {where_clause}" if where_clause else ""

        # Get total count (only uses WHERE params)
        count_result = conn.execute(f"SELECT COUNT(*) FROM finding_models {where_sql}", where_params).fetchone()
        total = int(count_result[0]) if count_result else 0

        # Get paginated results (uses WHERE + ORDER + pagination params)
        results = conn.execute(
            f"""
            SELECT oifm_id, name, slug_name, filename, file_hash_sha256, description, created_at, updated_at
            FROM finding_models
            {where_sql}
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
        """,
            where_params + order_params + [limit, offset],
        ).fetchall()

        # Build IndexEntry objects (note: no synonyms, tags, contributors, or attributes for performance)
        entries = [
            IndexEntry(
                oifm_id=row[0],
                name=row[1],
                slug_name=row[2],
                filename=row[3],
                file_hash_sha256=row[4],
                description=row[5],
                created_at=row[6],
                updated_at=row[7],
                attributes=None,  # Not fetched for list operations
            )
            for row in results
        ]

        return entries, total

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: Sequence[str] | None = None,
    ) -> list[IndexEntry]:
        """Search for finding models using hybrid search with RRF fusion.

        Uses Reciprocal Rank Fusion to combine FTS and semantic search results,
        returning exact matches immediately if found.

        Args:
            query: Search query string (must not be empty or whitespace-only)
            limit: Maximum number of results to return
            tags: Optional list of tags - models must have ALL specified tags

        Raises:
            ValueError: If query is empty or contains only whitespace
        """
        if not query or not query.strip():
            raise ValueError("Search query must not be empty or whitespace-only")

        conn = self._ensure_connection()

        # Exact matches take priority - return immediately if found
        exact_matches = await asyncify(self._search_exact)(conn, query, tags=tags)
        if exact_matches:
            return exact_matches[:limit]

        # Get both FTS and semantic results
        fts_matches = await asyncify(self._search_fts)(conn, query, limit=limit, tags=tags)
        semantic_matches = await self._search_semantic(conn, query, limit=limit, tags=tags)

        # If no vector results, just return FTS results
        if not semantic_matches:
            return [entry for entry, _ in fts_matches[:limit]]

        # Apply RRF fusion
        fts_scores = [(entry.oifm_id, score) for entry, score in fts_matches]
        semantic_scores = [(entry.oifm_id, score) for entry, score in semantic_matches]
        fused_scores = rrf_fusion(fts_scores, semantic_scores)

        # Build result lookup by oifm_id
        entry_map: dict[str, IndexEntry] = {}
        for entry, _ in fts_matches + semantic_matches:
            if entry.oifm_id not in entry_map:
                entry_map[entry.oifm_id] = entry

        # Return entries in RRF-ranked order
        results: list[IndexEntry] = []
        for oifm_id, _ in fused_scores[:limit]:
            if oifm_id in entry_map:
                results.append(entry_map[oifm_id])

        return results

    async def search_batch(  # noqa: C901
        self, queries: list[str], *, limit: int = 10
    ) -> dict[str, list[IndexEntry]]:
        """Search multiple queries efficiently with single embedding call and RRF fusion.

        Embeds ALL queries in a single OpenAI API call for efficiency,
        then performs hybrid search with RRF fusion for each query.
        Empty or whitespace-only queries are skipped.

        Args:
            queries: List of search query strings
            limit: Maximum number of results per query

        Returns:
            Dictionary mapping each non-blank query string to its list of results

        Raises:
            ValueError: If all queries are empty or whitespace-only
        """
        if not queries:
            return {}

        # Filter out blank queries
        valid_queries = [q for q in queries if q and q.strip()]
        if not valid_queries:
            raise ValueError("All queries are empty or whitespace-only")

        conn = self._ensure_connection()

        # Generate embeddings for all valid queries in a single batch API call
        embeddings = await batch_embeddings_for_duckdb(valid_queries)

        results: dict[str, list[IndexEntry]] = {}
        query: str
        for query, embedding in zip(valid_queries, embeddings, strict=True):
            # Check for exact match first
            exact_matches = self._search_exact(conn, query, tags=None)
            if exact_matches:
                results[query] = exact_matches[:limit]
                continue

            # Perform FTS search
            fts_matches = self._search_fts(conn, query, limit=limit, tags=None)

            # Perform semantic search using pre-generated embedding
            semantic_matches: list[tuple[IndexEntry, float]] = []
            if embedding is not None:
                semantic_matches = self._search_semantic_with_embedding(conn, embedding, limit=limit, tags=None)

            # If no vector results, just return FTS results
            if not semantic_matches:
                results[query] = [entry for entry, _ in fts_matches[:limit]]
                continue

            # Apply RRF fusion
            fts_scores = [(entry.oifm_id, score) for entry, score in fts_matches]
            semantic_scores = [(entry.oifm_id, score) for entry, score in semantic_matches]
            fused_scores = rrf_fusion(fts_scores, semantic_scores)

            # Build result lookup by oifm_id
            entry_map: dict[str, IndexEntry] = {}
            for entry, _ in fts_matches + semantic_matches:
                if entry.oifm_id not in entry_map:
                    entry_map[entry.oifm_id] = entry

            # Return entries in RRF-ranked order
            query_results: list[IndexEntry] = []
            for oifm_id, _ in fused_scores[:limit]:
                if oifm_id in entry_map:
                    query_results.append(entry_map[oifm_id])

            results[query] = query_results

        return results

    def _load_oifm_ids_for_source(self, source: str) -> set[str]:
        """Load all existing OIFM IDs for a source from database (cached).

        Results are cached per-instance to avoid repeated database queries.
        Cache is updated when new IDs are generated to prevent self-collision.

        Args:
            source: The source code (already validated)

        Returns:
            Set of existing OIFM IDs for this source
        """
        if source in self._oifm_id_cache:
            return self._oifm_id_cache[source]

        conn = self._ensure_connection()
        pattern = f"OIFM_{source}_%"
        rows = conn.execute("SELECT oifm_id FROM finding_models WHERE oifm_id LIKE ?", [pattern]).fetchall()
        ids = {row[0] for row in rows}
        self._oifm_id_cache[source] = ids
        logger.debug(f"Loaded {len(ids)} existing OIFM IDs for source {source}")
        return ids

    def _load_oifma_ids_for_source(self, source: str) -> set[str]:
        """Load all existing OIFMA IDs for a source from database (cached).

        Results are cached per-instance to avoid repeated database queries.
        Cache is updated when new IDs are generated to prevent self-collision.

        Args:
            source: The source code (already validated)

        Returns:
            Set of existing OIFMA IDs for this source
        """
        if source in self._oifma_id_cache:
            return self._oifma_id_cache[source]

        conn = self._ensure_connection()
        pattern = f"OIFMA_{source}_%"
        rows = conn.execute("SELECT attribute_id FROM attributes WHERE attribute_id LIKE ?", [pattern]).fetchall()
        ids = {row[0] for row in rows}
        self._oifma_id_cache[source] = ids
        logger.debug(f"Loaded {len(ids)} existing OIFMA IDs for source {source}")
        return ids

    def generate_model_id(self, source: str = "OIDM", max_attempts: int = 100) -> str:
        """Generate unique OIFM ID by querying Index database.

        Replaces GitHub-based ID registry. The Index database already contains
        all existing models, so we query it to get used IDs and check collisions
        in memory. The ID set is cached per source and updated as we generate
        new IDs to avoid stepping on our own feet.

        Args:
            source: 3-4 uppercase letter code for originating organization
                    (default: "OIDM" for Open Imaging Data Model)
            max_attempts: Maximum collision retry attempts

        Returns:
            Unique OIFM ID in format: OIFM_{SOURCE}_{6_DIGITS}

        Raises:
            ValueError: If source is invalid
            RuntimeError: If unable to generate unique ID after max_attempts

        Example:
            >>> async with DuckDBIndex() as index:
            ...     oifm_id = index.generate_model_id("GMTS")
            ...     # Returns "OIFM_GMTS_123456"
        """
        # Validate and normalize source
        source_upper = source.strip().upper()
        if not (3 <= len(source_upper) <= 4 and source_upper.isalpha()):
            raise ValueError(f"Source must be 3-4 uppercase letters, got: {source_upper}")

        # Load existing IDs for source (cached)
        from findingmodel.finding_model import _random_digits

        existing_ids = self._load_oifm_ids_for_source(source_upper)

        # Generate random ID with collision checking
        for attempt in range(max_attempts):
            candidate_id = f"OIFM_{source_upper}_{_random_digits(6)}"
            if candidate_id not in existing_ids:
                # Add to cache to prevent self-collision
                existing_ids.add(candidate_id)
                logger.debug(f"Generated new OIFM ID: {candidate_id} (attempt {attempt + 1})")
                return candidate_id
            logger.debug(f"Collision detected for {candidate_id}, retrying...")

        raise RuntimeError(f"Unable to generate unique OIFM ID for source {source_upper} after {max_attempts} attempts")

    def generate_attribute_id(
        self,
        model_oifm_id: str | None = None,
        source: str | None = None,
        max_attempts: int = 100,
    ) -> str:
        """Generate unique OIFMA ID by querying Index database.

        Replaces GitHub-based ID registry. Attribute IDs (OIFMA) identify
        individual attributes within finding models. Source can be inferred
        from the parent model's OIFM ID or provided explicitly.

        The ID set is cached per source and updated as we generate new IDs
        to avoid stepping on our own feet when generating multiple IDs.

        Args:
            model_oifm_id: Parent model's OIFM ID (source will be inferred)
            source: Explicit 3-4 uppercase letter source code (overrides inference)
            max_attempts: Maximum collision retry attempts

        Returns:
            Unique OIFMA ID in format: OIFMA_{SOURCE}_{6_DIGITS}

        Raises:
            ValueError: If source is invalid or cannot be inferred
            RuntimeError: If unable to generate unique ID after max_attempts

        Note:
            Value codes (OIFMA_XXX_NNNNNN.0, OIFMA_XXX_NNNNNN.1, etc.) are
            automatically generated from attribute IDs by the model editor.
            This method only generates the base attribute ID.

        Example:
            >>> async with DuckDBIndex() as index:
            ...     # Infer source from model
            ...     oifma_id = index.generate_attribute_id(model_oifm_id="OIFM_GMTS_123456")
            ...     # Returns "OIFMA_GMTS_234567"
            ...     # Or use explicit source
            ...     oifma_id = index.generate_attribute_id(source="GMTS")
        """
        # Determine source (explicit > infer from model_oifm_id > default "OIDM")
        if source is not None:
            resolved_source = source.strip().upper()
        elif model_oifm_id is not None:
            # Infer source from model_oifm_id: "OIFM_GMTS_123456" → "GMTS"
            parts = model_oifm_id.split("_")
            if len(parts) != 3 or parts[0] != "OIFM":
                raise ValueError(f"Cannot infer source from invalid model ID: {model_oifm_id}")
            resolved_source = parts[1]
        else:
            resolved_source = "OIDM"

        # Validate resolved_source (3-4 uppercase letters)
        if not (3 <= len(resolved_source) <= 4 and resolved_source.isalpha()):
            raise ValueError(f"Source must be 3-4 uppercase letters, got: {resolved_source}")

        # Load existing attribute IDs for source (cached)
        from findingmodel.finding_model import _random_digits

        existing_ids = self._load_oifma_ids_for_source(resolved_source)

        # Generate random ID with collision checking
        for attempt in range(max_attempts):
            candidate_id = f"OIFMA_{resolved_source}_{_random_digits(6)}"
            if candidate_id not in existing_ids:
                # Add to cache to prevent self-collision
                existing_ids.add(candidate_id)
                logger.debug(f"Generated new OIFMA ID: {candidate_id} (attempt {attempt + 1})")
                return candidate_id
            logger.debug(f"Collision detected for {candidate_id}, retrying...")

        raise RuntimeError(
            f"Unable to generate unique OIFMA ID for source {resolved_source} after {max_attempts} attempts"
        )

    def add_ids_to_model(
        self,
        finding_model: FindingModelBase | FindingModelFull,
        source: str,
    ) -> FindingModelFull:
        """Generate and add OIFM and OIFMA IDs to a finding model.

        Takes a FindingModelBase (which may lack IDs) and generates:
        - OIFM ID for the model if missing
        - OIFMA ID for each attribute that lacks one

        Args:
            finding_model: The finding model to add IDs to (FindingModelBase or FindingModelFull)
            source: 3-4 uppercase letter code for the originating organization

        Returns:
            FindingModelFull with all IDs populated

        Example:
            # Create model without IDs
            base_model = FindingModelBase(
                name="Pneumothorax",
                description="Air in pleural space",
                attributes=[...]
            )

            # Generate and add IDs
            async with Index() as index:
                full_model = index.add_ids_to_model(base_model, "GMTS")
            print(full_model.oifm_id)  # "OIFM_GMTS_472951"
        """
        finding_model_dict = finding_model.model_dump()

        # Generate OIFM ID if missing
        if "oifm_id" not in finding_model_dict:
            finding_model_dict["oifm_id"] = self.generate_model_id(source)
            logger.debug(f"Generated OIFM ID: {finding_model_dict['oifm_id']}")

        # Generate OIFMA IDs for attributes that lack them
        for attribute in finding_model_dict.get("attributes", []):
            if "oifma_id" not in attribute:
                attribute["oifma_id"] = self.generate_attribute_id(
                    model_oifm_id=finding_model_dict["oifm_id"], source=source
                )
                logger.debug(f"Generated OIFMA ID: {attribute['oifma_id']} for attribute {attribute.get('name')}")

        logger.info(f"Added IDs to finding model {finding_model.name} from source {source}")
        return FindingModelFull.model_validate(finding_model_dict)

    def finalize_placeholder_attribute_ids(
        self,
        finding_model: FindingModelFull,
        source: str | None = None,
    ) -> FindingModelFull:
        """Replace placeholder attribute IDs with generated IDs and renumber value codes.

        Looks for attributes with ID "OIFMA_XXXX_000000" and replaces them with
        unique generated IDs. Also renumbers value codes for choice attributes.

        Args:
            finding_model: Model containing attributes to update
            source: Optional 3-4 uppercase code identifying the source organization.
                    When omitted, the code is inferred from the model's OIFM ID.

        Returns:
            FindingModelFull with unique attribute IDs for all placeholders.
            If no placeholders were present, the original model is returned unchanged.

        Example:
            # Model with placeholder IDs
            model = FindingModelFull(
                oifm_id="OIFM_GMTS_123456",
                attributes=[
                    {"name": "Size", "oifma_id": "OIFMA_XXXX_000000", ...},
                    {"name": "Shape", "oifma_id": "OIFMA_GMTS_789012", ...}  # Keep this
                ]
            )

            async with Index() as index:
                updated = index.finalize_placeholder_attribute_ids(model)
            # First attribute gets real ID, second unchanged
        """
        # Resolve source (explicit or infer from model ID)
        if source:
            resolved_source = source.strip().upper()
            if not (3 <= len(resolved_source) <= 4 and resolved_source.isalpha()):
                raise ValueError(f"Source must be 3-4 uppercase letters, got: {source}")
        else:
            # Infer from model OIFM ID
            parts = finding_model.oifm_id.split("_")
            if len(parts) != 3 or parts[0] != "OIFM":
                raise ValueError(f"Cannot infer source from model ID: {finding_model.oifm_id}")
            resolved_source = parts[1]

        model_dict = finding_model.model_dump()

        # Track existing IDs to prevent collisions when generating multiple new IDs
        existing_ids: set[str] = set()
        existing_ids.update(
            attr.get("oifma_id")
            for attr in model_dict.get("attributes", [])
            if attr.get("oifma_id") and attr.get("oifma_id") != PLACEHOLDER_ATTRIBUTE_ID
        )

        updated = False

        for attr in model_dict.get("attributes", []):
            if attr.get("oifma_id") != PLACEHOLDER_ATTRIBUTE_ID:
                continue

            # Generate new unique ID
            new_id = self.generate_attribute_id(model_oifm_id=finding_model.oifm_id, source=resolved_source)
            attr["oifma_id"] = new_id
            existing_ids.add(new_id)
            updated = True
            logger.debug(f"Replaced placeholder with {new_id} for attribute {attr.get('name')}")

            # Renumber value codes for choice attributes
            if attr.get("type") == "choice":
                for idx, value in enumerate(attr.get("values", []) or []):
                    value["value_code"] = f"{new_id}.{idx}"

        if not updated:
            return finding_model

        return FindingModelFull.model_validate(model_dict)

    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        if self.conn is None:
            self.conn = setup_duckdb_connection(self.db_path, read_only=True)
        return self.conn

    # Typed sync helpers for asyncify - preserve type information
    # These wrap DuckDB operations for proper async execution

    def _query_full_model(self, conn: duckdb.DuckDBPyConnection, oifm_id: str) -> str | None:
        """Get full model JSON by ID (sync - types preserved)."""
        result = conn.execute("SELECT model_json FROM finding_model_json WHERE oifm_id = ?", [oifm_id]).fetchone()
        return result[0] if result else None

    def _query_full_models_batch(self, conn: duckdb.DuckDBPyConnection, oifm_ids: list[str]) -> list[tuple[str, str]]:
        """Get multiple full models efficiently (sync - types preserved)."""
        placeholders = ", ".join(["?"] * len(oifm_ids))
        return conn.execute(
            f"SELECT oifm_id, model_json FROM finding_model_json WHERE oifm_id IN ({placeholders})", oifm_ids
        ).fetchall()

    def _query_count(self, conn: duckdb.DuckDBPyConnection, table: str) -> int:
        """Count rows in table (sync - types preserved)."""
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def _query_person(self, conn: duckdb.DuckDBPyConnection, github_username: str) -> tuple[Any, ...] | None:
        """Get person by GitHub username (sync - types preserved)."""
        return conn.execute(
            """
            SELECT github_username, name, email, organization_code, url
            FROM people
            WHERE github_username = ?
            """,
            (github_username,),
        ).fetchone()

    def _query_organization(self, conn: duckdb.DuckDBPyConnection, code: str) -> tuple[Any, ...] | None:
        """Get organization by code (sync - types preserved)."""
        return conn.execute(
            """
            SELECT code, name, url
            FROM organizations
            WHERE code = ?
            """,
            (code,),
        ).fetchone()

    def _query_all_people(self, conn: duckdb.DuckDBPyConnection) -> list[tuple[Any, ...]]:
        """Get all people ordered by name (sync - types preserved)."""
        return conn.execute(
            """
            SELECT github_username, name, email, organization_code, url
            FROM people
            ORDER BY name
            """
        ).fetchall()

    def _query_all_organizations(self, conn: duckdb.DuckDBPyConnection) -> list[tuple[Any, ...]]:
        """Get all organizations ordered by name (sync - types preserved)."""
        return conn.execute(
            """
            SELECT code, name, url
            FROM organizations
            ORDER BY name
            """
        ).fetchall()

    def _query_search_count(self, conn: duckdb.DuckDBPyConnection, where_clause: str, sql_pattern: str) -> int:
        """Get count of finding models matching search pattern (sync - types preserved)."""
        result = conn.execute(f"SELECT COUNT(*) FROM finding_models WHERE {where_clause}", [sql_pattern]).fetchone()
        return int(result[0]) if result else 0

    def _fetch_index_entry(self, conn: duckdb.DuckDBPyConnection, oifm_id: str) -> IndexEntry | None:
        row = conn.execute(
            """
            SELECT oifm_id, name, slug_name, filename, file_hash_sha256, description, created_at, updated_at
            FROM finding_models
            WHERE oifm_id = ?
            """,
            (oifm_id,),
        ).fetchone()
        if row is None:
            return None

        synonyms = [
            r[0]
            for r in conn.execute(
                "SELECT synonym FROM synonyms WHERE oifm_id = ? ORDER BY synonym", (oifm_id,)
            ).fetchall()
        ]
        tags = [
            r[0] for r in conn.execute("SELECT tag FROM tags WHERE oifm_id = ? ORDER BY tag", (oifm_id,)).fetchall()
        ]
        attribute_rows = conn.execute(
            """
            SELECT attribute_id, attribute_name, attribute_type
            FROM attributes
            WHERE oifm_id = ?
            ORDER BY attribute_name
            """,
            (oifm_id,),
        ).fetchall()
        attributes = [AttributeInfo(attribute_id=r[0], name=r[1], type=r[2]) for r in attribute_rows]

        contributors = self._collect_contributors(conn, oifm_id)

        return IndexEntry(
            oifm_id=row[0],
            name=row[1],
            slug_name=row[2],
            filename=row[3],
            file_hash_sha256=row[4],
            description=row[5],
            created_at=row[6],
            updated_at=row[7],
            synonyms=synonyms or None,
            tags=tags or None,
            contributors=contributors or None,
            attributes=attributes or None,
        )

    def _collect_contributors(self, conn: duckdb.DuckDBPyConnection, oifm_id: str) -> list[str]:
        person_rows = conn.execute(
            "SELECT person_id, display_order FROM model_people WHERE oifm_id = ? ORDER BY display_order, person_id",
            (oifm_id,),
        ).fetchall()
        org_rows = conn.execute(
            "SELECT organization_id, display_order FROM model_organizations WHERE oifm_id = ? ORDER BY display_order, organization_id",
            (oifm_id,),
        ).fetchall()

        combined: list[tuple[int, str]] = []
        combined.extend((row[1] if row[1] is not None else idx, row[0]) for idx, row in enumerate(person_rows))
        base = len(combined)
        combined.extend((row[1] if row[1] is not None else base + idx, row[0]) for idx, row in enumerate(org_rows))
        combined.sort(key=lambda item: item[0])
        return [identifier for _, identifier in combined]

    def _resolve_oifm_id(self, conn: duckdb.DuckDBPyConnection, identifier: str) -> str | None:
        row = conn.execute("SELECT oifm_id FROM finding_models WHERE oifm_id = ?", (identifier,)).fetchone()
        if row is not None:
            return str(row[0])

        row = conn.execute(
            "SELECT oifm_id FROM finding_models WHERE LOWER(name) = LOWER(?)",
            (identifier,),
        ).fetchone()
        if row is not None:
            return str(row[0])

        slug = None
        if len(identifier) >= 3:
            try:
                slug = normalize_name(identifier)
            except (TypeError, ValueError):
                slug = None
        if slug:
            row = conn.execute(
                "SELECT oifm_id FROM finding_models WHERE slug_name = ?",
                (slug,),
            ).fetchone()
            if row is not None:
                return str(row[0])

        row = conn.execute(
            "SELECT oifm_id FROM synonyms WHERE LOWER(synonym) = LOWER(?) LIMIT 1",
            (identifier,),
        ).fetchone()
        if row is not None:
            return str(row[0])

        return None

    def _build_search_text(self, model: FindingModelFull) -> str:
        parts: list[str] = [model.name]
        if model.description:
            parts.append(model.description)
        if model.synonyms:
            parts.extend(model.synonyms)
        if model.tags:
            parts.extend(model.tags)
        parts.extend(attribute.name for attribute in model.attributes)
        return "\n".join(part for part in parts if part)

    def _build_embedding_text(self, model: FindingModelFull) -> str:
        parts: list[str] = [model.name]
        if model.description:
            parts.append(model.description)
        if model.synonyms:
            parts.append("Synonyms: " + ", ".join(model.synonyms))
        if model.tags:
            parts.append("Tags: " + ", ".join(model.tags))
        attribute_lines = [
            f"Attribute {attribute.name}: {attribute.description or attribute.type}" for attribute in model.attributes
        ]
        parts.extend(attribute_lines)
        return "\n".join(part for part in parts if part)

    def _search_exact(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        *,
        tags: Sequence[str] | None = None,
    ) -> list[IndexEntry]:
        oifm_id = self._resolve_oifm_id(conn, query)
        if oifm_id is None:
            return []

        entry = self._fetch_index_entry(conn, oifm_id)
        if entry is None:
            return []

        if tags and not self._entry_has_tags(entry, tags):
            return []

        return [entry]

    def _entry_has_tags(self, entry: IndexEntry, tags: Sequence[str]) -> bool:
        entry_tags = set(entry.tags or [])
        return all(tag in entry_tags for tag in tags)

    def _search_fts(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        *,
        limit: int,
        tags: Sequence[str] | None = None,
    ) -> list[tuple[IndexEntry, float]]:
        rows = conn.execute(
            """
            WITH candidates AS (
                SELECT
                    f.oifm_id,
                    fts_main_finding_models.match_bm25(f.oifm_id, ?) AS bm25_score
                FROM finding_models AS f
            )
            SELECT oifm_id, bm25_score
            FROM candidates
            WHERE bm25_score IS NOT NULL
            ORDER BY bm25_score DESC
            LIMIT ?
            """,
            (query, limit * 3),
        ).fetchall()

        if not rows:
            return []

        entries: list[IndexEntry] = []
        scores: list[float] = []
        for oifm_id, score in rows:
            entry = self._fetch_index_entry(conn, str(oifm_id))
            if entry is None:
                continue
            if tags and not self._entry_has_tags(entry, tags):
                continue
            entries.append(entry)
            scores.append(float(score))
            if len(entries) >= limit:
                break

        if not entries:
            return []

        normalized_scores = normalize_scores(scores)
        paired = list(zip(entries, normalized_scores, strict=True))
        paired.sort(key=lambda item: item[1], reverse=True)
        return [(entry, score) for entry, score in paired]

    async def _search_semantic(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        *,
        limit: int,
        tags: Sequence[str] | None = None,
    ) -> list[tuple[IndexEntry, float]]:
        """Perform semantic search by generating embedding for query text."""

        if limit <= 0:
            return []

        trimmed_query = query.strip()
        if not trimmed_query:
            return []

        embedding = await get_embedding_for_duckdb(trimmed_query)
        if embedding is None:
            return []

        return await asyncify(self._search_semantic_with_embedding)(conn, embedding, limit=limit, tags=tags)

    def _search_semantic_with_embedding(
        self,
        conn: duckdb.DuckDBPyConnection,
        embedding: list[float],
        *,
        limit: int,
        tags: Sequence[str] | None = None,
    ) -> list[tuple[IndexEntry, float]]:
        """Perform semantic search using a pre-computed embedding.

        This is used by search_batch() to avoid redundant embedding generation.

        Args:
            conn: Active database connection
            embedding: Pre-computed embedding vector
            limit: Maximum number of results to return
            tags: Optional list of tags - models must have ALL specified tags

        Returns:
            List of (IndexEntry, score) tuples sorted by descending similarity
        """
        if limit <= 0:
            return []

        dimensions = settings.openai_embedding_dimensions
        rows = conn.execute(
            f"""
            SELECT oifm_id, array_distance(embedding, CAST(? AS FLOAT[{dimensions}])) AS l2_distance
            FROM finding_models
            ORDER BY array_distance(embedding, CAST(? AS FLOAT[{dimensions}]))
            LIMIT ?
            """,
            (embedding, embedding, limit * 3),
        ).fetchall()

        if not rows:
            return []

        entries: list[IndexEntry] = []
        scores: list[float] = []
        for oifm_id, l2_distance in rows:
            entry = self._fetch_index_entry(conn, str(oifm_id))
            if entry is None:
                continue
            if tags and not self._entry_has_tags(entry, tags):
                continue
            scores.append(l2_to_cosine_similarity(float(l2_distance)))
            entries.append(entry)
            if len(entries) >= limit:
                break

        paired = list(zip(entries, scores, strict=True))
        paired.sort(key=lambda item: item[1], reverse=True)
        return [(entry, score) for entry, score in paired]


# Alias for backward compatibility
Index = DuckDBIndex

__all__ = [
    "AttributeInfo",
    "DuckDBIndex",
    "Index",
    "IndexEntry",
]
