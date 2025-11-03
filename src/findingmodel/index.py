"""DuckDB-backed implementation of the finding model index."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Literal

import duckdb
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from findingmodel import logger
from findingmodel.common import normalize_name
from findingmodel.config import settings
from findingmodel.contributor import Organization, Person
from findingmodel.finding_model import FindingModelBase, FindingModelFull
from findingmodel.tools.duckdb_utils import (
    batch_embeddings_for_duckdb,
    create_fts_index,
    create_hnsw_index,
    drop_search_indexes,
    get_embedding_for_duckdb,
    l2_to_cosine_similarity,
    normalize_scores,
    rrf_fusion,
    setup_duckdb_connection,
)

DEFAULT_CONTRIBUTOR_ROLE = "contributor"
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


class IndexReturnType(StrEnum):
    """Indicates whether an entry was added, updated, or unchanged."""

    ADDED = "added"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(slots=True)
class _BatchPayload:
    model_rows: list[tuple[object, ...]]
    synonym_rows: list[tuple[str, str]]
    tag_rows: list[tuple[str, str]]
    attribute_rows: list[tuple[str, str, str, str, str]]
    people_rows: list[tuple[str, str, str, str, str | None]]
    model_people_rows: list[tuple[str, str, str, int]]
    organization_rows: list[tuple[str, str, str | None]]
    model_organization_rows: list[tuple[str, str, str, int]]
    json_rows: list[tuple[str, str]]
    ids_to_delete: list[str]


@dataclass(slots=True)
class _RowData:
    model_rows: list[tuple[object, ...]]
    synonym_rows: list[tuple[str, str]]
    tag_rows: list[tuple[str, str]]
    attribute_rows: list[tuple[str, str, str, str, str]]
    people_rows: list[tuple[str, str, str, str, str | None]]
    model_people_rows: list[tuple[str, str, str, int]]
    organization_rows: list[tuple[str, str, str | None]]
    model_organization_rows: list[tuple[str, str, str, int]]
    json_rows: list[tuple[str, str]]


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS finding_models (
        oifm_id VARCHAR PRIMARY KEY,
        slug_name VARCHAR NOT NULL UNIQUE,
        name VARCHAR NOT NULL UNIQUE,
        filename VARCHAR NOT NULL UNIQUE,
        file_hash_sha256 VARCHAR NOT NULL,
        description TEXT,
        search_text TEXT NOT NULL,
        embedding FLOAT[512] NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS people (
        github_username VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        email VARCHAR NOT NULL,
        organization_code VARCHAR,
        url VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS organizations (
        code VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        url VARCHAR,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_people (
        oifm_id VARCHAR NOT NULL,
        person_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL DEFAULT 'contributor',
        display_order INTEGER,
        PRIMARY KEY (oifm_id, person_id, role)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_organizations (
        oifm_id VARCHAR NOT NULL,
        organization_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL DEFAULT 'contributor',
        display_order INTEGER,
        PRIMARY KEY (oifm_id, organization_id, role)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS synonyms (
        oifm_id VARCHAR NOT NULL,
        synonym VARCHAR NOT NULL,
        PRIMARY KEY (oifm_id, synonym)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attributes (
        attribute_id VARCHAR PRIMARY KEY,
        oifm_id VARCHAR NOT NULL,
        model_name VARCHAR NOT NULL,
        attribute_name VARCHAR NOT NULL,
        attribute_type VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tags (
        oifm_id VARCHAR NOT NULL,
        tag VARCHAR NOT NULL,
        PRIMARY KEY (oifm_id, tag)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS finding_model_json (
        oifm_id VARCHAR PRIMARY KEY,
        model_json TEXT NOT NULL
    )
    """,
)


_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_finding_models_name ON finding_models(name)",
    "CREATE INDEX IF NOT EXISTS idx_finding_models_slug_name ON finding_models(slug_name)",
    "CREATE INDEX IF NOT EXISTS idx_finding_models_filename ON finding_models(filename)",
    "CREATE INDEX IF NOT EXISTS idx_synonyms_synonym ON synonyms(synonym)",
    "CREATE INDEX IF NOT EXISTS idx_synonyms_model ON synonyms(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)",
    "CREATE INDEX IF NOT EXISTS idx_tags_model ON tags(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_people_model ON model_people(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_people_person ON model_people(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_orgs_model ON model_organizations(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_model_orgs_org ON model_organizations(organization_id)",
    "CREATE INDEX IF NOT EXISTS idx_attributes_model ON attributes(oifm_id)",
    "CREATE INDEX IF NOT EXISTS idx_attributes_name ON attributes(attribute_name)",
)


class DuckDBIndex:
    """DuckDB-based index with read-only connections by default."""

    def __init__(self, db_path: str | Path | None = None, *, read_only: bool = True) -> None:
        if db_path:
            self.db_path = Path(db_path).expanduser()  # Honor explicit path
        else:
            # Use package data directory with optional download
            from findingmodel.config import ensure_index_db

            self.db_path = ensure_index_db()
        self.read_only = read_only
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._openai_client: AsyncOpenAI | None = None
        self._oifm_id_cache: dict[str, set[str]] = {}  # {source: {id, ...}}
        self._oifma_id_cache: dict[str, set[str]] = {}  # {source: {id, ...}}

    async def setup(self) -> None:
        """Ensure the database exists, connection opened, and schema ready."""

        conn = self._ensure_connection()

        if self.read_only:
            return

        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        for statement in _INDEX_STATEMENTS:
            conn.execute(statement)
        self._create_search_indexes(conn)

        # Load base contributors if tables are empty
        self._load_base_contributors(conn)

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
        return self._resolve_oifm_id(conn, identifier) is not None

    async def get(self, identifier: str) -> IndexEntry | None:
        """Retrieve an index entry by ID, name, or synonym."""

        conn = self._ensure_connection()
        oifm_id = self._resolve_oifm_id(conn, identifier)
        if oifm_id is None:
            return None
        return self._fetch_index_entry(conn, oifm_id)

    async def get_full(self, oifm_id: str) -> FindingModelFull:
        """Get full FindingModelFull object by ID.

        Args:
            oifm_id: The OIFM ID to retrieve

        Returns:
            Full FindingModelFull object parsed from stored JSON

        Raises:
            KeyError: If model not found

        Example:
            >>> index = DuckDBIndex()
            >>> await index.setup()
            >>> model = await index.get_full("OIFM_RADLEX_000001")
            >>> # Returns complete FindingModelFull with all attributes
        """
        conn = self._ensure_connection()
        result = conn.execute("SELECT model_json FROM finding_model_json WHERE oifm_id = ?", [oifm_id]).fetchone()

        if not result:
            raise KeyError(f"Model not found: {oifm_id}")

        json_text = result[0]
        return FindingModelFull.model_validate_json(json_text)

    async def get_full_batch(self, oifm_ids: list[str]) -> dict[str, FindingModelFull]:
        """Get multiple full models efficiently.

        Args:
            oifm_ids: List of OIFM IDs to retrieve

        Returns:
            Dict mapping OIFM ID to FindingModelFull object. Only includes models that were found.

        Example:
            >>> index = DuckDBIndex()
            >>> await index.setup()
            >>> models = await index.get_full_batch(["OIFM_RADLEX_000001", "OIFM_CUSTOM_000042"])
            >>> # Returns {oifm_id: FindingModelFull, ...}
        """
        if not oifm_ids:
            return {}

        conn = self._ensure_connection()
        placeholders = ", ".join(["?"] * len(oifm_ids))
        results = conn.execute(
            f"SELECT oifm_id, model_json FROM finding_model_json WHERE oifm_id IN ({placeholders})", oifm_ids
        ).fetchall()

        return {oifm_id: FindingModelFull.model_validate_json(json_text) for oifm_id, json_text in results}

    async def count(self) -> int:
        """Return the number of finding models in the index."""

        conn = self._ensure_connection()
        row = conn.execute("SELECT COUNT(*) FROM finding_models").fetchone()
        return int(row[0]) if row else 0

    async def count_people(self) -> int:
        """Return the number of people in the normalized table."""

        conn = self._ensure_connection()
        row = conn.execute("SELECT COUNT(*) FROM people").fetchone()
        return int(row[0]) if row else 0

    async def count_organizations(self) -> int:
        """Return the number of organizations in the normalized table."""

        conn = self._ensure_connection()
        row = conn.execute("SELECT COUNT(*) FROM organizations").fetchone()
        return int(row[0]) if row else 0

    async def get_person(self, github_username: str) -> Person | None:
        """Retrieve a person by GitHub username."""

        conn = self._ensure_connection()
        row = conn.execute(
            """
            SELECT github_username, name, email, organization_code, url
            FROM people
            WHERE github_username = ?
            """,
            (github_username,),
        ).fetchone()
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
        row = conn.execute(
            """
            SELECT code, name, url
            FROM organizations
            WHERE code = ?
            """,
            (code,),
        ).fetchone()
        if row is None:
            return None
        return Organization.model_validate({"code": row[0], "name": row[1], "url": row[2]})

    async def get_people(self) -> list[Person]:
        """Retrieve all people from the index."""
        conn = self._ensure_connection()
        rows = conn.execute(
            """
            SELECT github_username, name, email, organization_code, url
            FROM people
            ORDER BY name
            """
        ).fetchall()
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
        rows = conn.execute(
            """
            SELECT code, name, url
            FROM organizations
            ORDER BY name
            """
        ).fetchall()
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

        # Use helper to execute query (no WHERE clause for list all)
        return self._execute_paginated_query(order_clause=order_clause, limit=limit, offset=offset)

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

        # Use helper to execute query
        return self._execute_paginated_query(
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
        result = conn.execute(f"SELECT COUNT(*) FROM finding_models WHERE {where_clause}", [sql_pattern]).fetchone()

        return int(result[0]) if result else 0

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
        where_clause: str = "",
        where_params: list[object] | None = None,
        order_clause: str = "LOWER(name)",
        order_params: list[object] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[IndexEntry], int]:
        """Execute paginated query with count and result fetching.

        Shared by list() and search_by_slug() to eliminate duplication.

        Args:
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

        conn = self._ensure_connection()

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

    async def add_or_update_entry_from_file(
        self,
        filename: str | Path,
        model: FindingModelFull | None = None,
        *,
        allow_duplicate_synonyms: bool = False,
    ) -> IndexReturnType:
        """Insert or update a finding model from a `.fm.json` file."""

        conn = self._ensure_writable_connection()
        await self.setup()

        file_path = filename if isinstance(filename, Path) else Path(filename)
        if not file_path.name.endswith(".fm.json"):
            raise ValueError("Expect filename to end with '.fm.json'")

        file_hash = self._calculate_file_hash(file_path)

        # Capture JSON text before parsing for storage in finding_model_json table
        if model is None:
            json_text = file_path.read_text(encoding="utf-8")
            model = FindingModelFull.model_validate_json(json_text)
        else:
            # Model was provided, serialize it to JSON text
            json_text = model.model_dump_json(indent=2, exclude_none=True)

        existing_rows = conn.execute(
            """
            SELECT oifm_id, file_hash_sha256
            FROM finding_models
            WHERE oifm_id = ? OR filename = ?
            """,
            (model.oifm_id, file_path.name),
        ).fetchall()
        existing = existing_rows[0] if existing_rows else None

        status = IndexReturnType.ADDED
        if existing is not None:
            status = IndexReturnType.UPDATED
            if existing[1] == file_hash and existing[0] == model.oifm_id:
                return IndexReturnType.UNCHANGED

        # Only validate for new models or when OIFM ID changes
        # (updating same model with same ID shouldn't fail validation)
        if existing is None or existing[0] != model.oifm_id:
            validation_errors = [] if allow_duplicate_synonyms else self._validate_model(model)
            if validation_errors:
                raise ValueError(f"Model validation failed: {'; '.join(validation_errors)}")
        else:
            validation_errors = []

        embedding_payload = self._build_embedding_text(model)
        embedding = await get_embedding_for_duckdb(
            embedding_payload,
            client=await self._ensure_openai_client(),
        )
        if embedding is None:
            raise RuntimeError("Failed to generate embedding for finding model")

        search_text = self._build_search_text(model)
        slug_name = normalize_name(model.name)

        conn.execute("BEGIN TRANSACTION")
        try:
            self._drop_search_indexes(conn)
            self._delete_denormalized_records(conn, [row[0] for row in existing_rows])
            conn.execute(
                "DELETE FROM finding_models WHERE oifm_id = ? OR filename = ?",
                (model.oifm_id, file_path.name),
            )

            conn.execute(
                """
                INSERT INTO finding_models (
                    oifm_id,
                    slug_name,
                    name,
                    filename,
                    file_hash_sha256,
                    description,
                    search_text,
                    embedding
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model.oifm_id,
                    slug_name,
                    model.name,
                    file_path.name,
                    file_hash,
                    model.description,
                    search_text,
                    embedding,
                ),
            )

            # Store full JSON in separate table
            conn.execute(
                """
                INSERT INTO finding_model_json (oifm_id, model_json)
                VALUES (?, ?)
                ON CONFLICT (oifm_id) DO UPDATE SET
                    model_json = EXCLUDED.model_json
                """,
                (model.oifm_id, json_text),
            )

            self._upsert_contributors(conn, model)
            self._replace_synonyms(conn, model.oifm_id, model.synonyms)
            self._replace_tags(conn, model.oifm_id, model.tags)
            self._replace_attributes(conn, model)

            conn.execute("COMMIT")
        except Exception:  # pragma: no cover - rollback path
            conn.execute("ROLLBACK")
            self._create_search_indexes(conn)
            raise

        self._create_search_indexes(conn)

        return status

    def _collect_directory_files(self, directory: Path) -> list[tuple[str, str, Path]]:
        files: list[tuple[str, str, Path]] = []
        for file_path in sorted(directory.glob("*.fm.json")):
            file_hash = self._calculate_file_hash(file_path)
            files.append((file_path.name, file_hash, file_path))
        return files

    def _stage_directory_files(self, conn: duckdb.DuckDBPyConnection, files: Sequence[tuple[str, str, Path]]) -> None:
        if not files:
            return

        values_clause = ", ".join(["(?, ?)"] * len(files))
        params: list[str] = []
        for filename, file_hash, _ in files:
            params.extend([filename, file_hash])
        conn.execute(f"INSERT INTO tmp_directory_files VALUES {values_clause}", params)

    def _classify_directory_changes(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> tuple[set[str], dict[str, str], set[str]]:
        rows = conn.execute(
            """
            SELECT
                dir.filename AS directory_filename,
                dir.file_hash_sha256 AS directory_hash,
                fm.oifm_id AS index_oifm_id,
                fm.filename AS index_filename,
                fm.file_hash_sha256 AS index_hash
            FROM tmp_directory_files AS dir
            FULL OUTER JOIN finding_models AS fm
              ON dir.filename = fm.filename
            """
        ).fetchall()

        added_filenames: set[str] = set()
        updated_entries: dict[str, str] = {}
        removed_ids: set[str] = set()

        for dir_filename, dir_hash, index_oifm_id, index_filename, index_hash in rows:
            if dir_filename is not None and index_filename is None:
                added_filenames.add(str(dir_filename))
            elif dir_filename is not None and index_filename is not None:
                if dir_hash != index_hash and index_oifm_id is not None:
                    updated_entries[str(dir_filename)] = str(index_oifm_id)
            elif index_filename is not None and index_oifm_id is not None:
                removed_ids.add(str(index_oifm_id))

        return added_filenames, updated_entries, removed_ids

    async def _prepare_batch_payload(
        self,
        filenames_to_process: Sequence[str],
        files_by_name: Mapping[str, tuple[str, Path]],
        updated_entries: Mapping[str, str],
        removed_ids: Iterable[str],
        *,
        allow_duplicate_synonyms: bool = False,
    ) -> _BatchPayload:
        metadata, embedding_payloads = self._load_models_metadata(
            filenames_to_process,
            files_by_name,
            updated_entries,
            allow_duplicate_synonyms=allow_duplicate_synonyms,
        )
        embeddings = await self._generate_embeddings(embedding_payloads)
        row_data = self._build_row_data(metadata, embeddings)

        ids_to_delete_set = set(removed_ids)
        ids_to_delete_set.update(
            updated_entries[filename]
            for filename in filenames_to_process
            if filename in updated_entries and updated_entries[filename] is not None
        )

        return _BatchPayload(
            model_rows=row_data.model_rows,
            synonym_rows=row_data.synonym_rows,
            tag_rows=row_data.tag_rows,
            attribute_rows=row_data.attribute_rows,
            people_rows=row_data.people_rows,
            model_people_rows=row_data.model_people_rows,
            organization_rows=row_data.organization_rows,
            model_organization_rows=row_data.model_organization_rows,
            json_rows=row_data.json_rows,
            ids_to_delete=sorted(ids_to_delete_set),
        )

    def _execute_batch_directory_update(
        self,
        conn: duckdb.DuckDBPyConnection,
        payload: _BatchPayload,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if not payload.ids_to_delete and not payload.model_rows:
            return

        if progress_callback:
            progress_callback("Dropping search indexes...")

        indexes_dropped = False
        conn.execute("BEGIN TRANSACTION")
        try:
            self._drop_search_indexes(conn)
            indexes_dropped = True

            # Delete old entries first if needed
            if payload.ids_to_delete:
                self._delete_old_entries(conn, payload.ids_to_delete, progress_callback)

            # Insert new/updated entries
            self._insert_models_with_progress(conn, payload, progress_callback)

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            if indexes_dropped:
                self._create_search_indexes(conn)
            raise

        if progress_callback:
            progress_callback("Rebuilding search indexes...")
        self._create_search_indexes(conn)

    def _delete_old_entries(
        self,
        conn: duckdb.DuckDBPyConnection,
        ids_to_delete: list[str],
        progress_callback: Callable[[str], None] | None,
    ) -> None:
        """Delete old entries from all tables."""
        if progress_callback:
            progress_callback(f"Removing {len(ids_to_delete)} old entries...")
        self._delete_denormalized_records(conn, ids_to_delete)
        placeholders = ", ".join(["?"] * len(ids_to_delete))
        conn.execute(
            f"DELETE FROM finding_models WHERE oifm_id IN ({placeholders})",
            ids_to_delete,
        )

    def _insert_models_with_progress(
        self,
        conn: duckdb.DuckDBPyConnection,
        payload: _BatchPayload,
        progress_callback: Callable[[str], None] | None,
    ) -> None:
        """Insert models with progress updates for large batches."""
        total_models = len(payload.model_rows)
        chunk_size = 500

        if total_models > chunk_size:
            if progress_callback:
                progress_callback(f"Processing {total_models} models in chunks of {chunk_size}...")

            # Process in chunks
            for chunk_start in range(0, total_models, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_models)
                chunk_num = (chunk_start // chunk_size) + 1
                total_chunks = (total_models + chunk_size - 1) // chunk_size

                if progress_callback:
                    progress_callback(
                        f"Writing chunk {chunk_num}/{total_chunks} ({chunk_end}/{total_models} models)..."
                    )

                chunk_payload = self._create_chunk_payload(payload, chunk_start, chunk_end)
                self._apply_batch_mutations(conn, chunk_payload)
        else:
            if progress_callback:
                progress_callback(f"Writing {total_models} models to database...")
            # Create payload without deletions (already done in _delete_old_entries)
            no_delete_payload = _BatchPayload(
                model_rows=payload.model_rows,
                synonym_rows=payload.synonym_rows,
                tag_rows=payload.tag_rows,
                attribute_rows=payload.attribute_rows,
                people_rows=payload.people_rows,
                model_people_rows=payload.model_people_rows,
                organization_rows=payload.organization_rows,
                model_organization_rows=payload.model_organization_rows,
                json_rows=payload.json_rows,
                ids_to_delete=[],
            )
            self._apply_batch_mutations(conn, no_delete_payload)

    def _create_chunk_payload(
        self,
        payload: _BatchPayload,
        start_idx: int,
        end_idx: int,
    ) -> _BatchPayload:
        """Create a chunk of the payload for batch processing."""
        # Get the oifm_ids in this chunk
        chunk_model_rows = payload.model_rows[start_idx:end_idx]
        chunk_oifm_ids = {row[0] for row in chunk_model_rows}  # oifm_id is first element

        # Filter all related rows to only those in this chunk
        chunk_synonym_rows = [row for row in payload.synonym_rows if row[0] in chunk_oifm_ids]
        chunk_tag_rows = [row for row in payload.tag_rows if row[0] in chunk_oifm_ids]
        chunk_attribute_rows = [
            row for row in payload.attribute_rows if row[1] in chunk_oifm_ids
        ]  # oifm_id is second element
        chunk_model_people_rows = [row for row in payload.model_people_rows if row[0] in chunk_oifm_ids]
        chunk_model_organization_rows = [row for row in payload.model_organization_rows if row[0] in chunk_oifm_ids]
        chunk_json_rows = [row for row in payload.json_rows if row[0] in chunk_oifm_ids]

        # For people and organizations, we need to include all that are referenced
        # (even if they're not in this chunk, to avoid conflicts)

        return _BatchPayload(
            model_rows=chunk_model_rows,
            synonym_rows=chunk_synonym_rows,
            tag_rows=chunk_tag_rows,
            attribute_rows=chunk_attribute_rows,
            people_rows=payload.people_rows,  # Include all people (upsert handles duplicates)
            model_people_rows=chunk_model_people_rows,
            organization_rows=payload.organization_rows,  # Include all organizations
            model_organization_rows=chunk_model_organization_rows,
            json_rows=chunk_json_rows,
            ids_to_delete=[],  # Only delete in first chunk
        )

    def _load_models_metadata(
        self,
        filenames_to_process: Sequence[str],
        files_by_name: Mapping[str, tuple[str, Path]],
        updated_entries: Mapping[str, str],
        *,
        allow_duplicate_synonyms: bool = False,
    ) -> tuple[list[tuple[FindingModelFull, str, str, str, str]], list[str]]:
        metadata: list[tuple[FindingModelFull, str, str, str, str]] = []
        embedding_payloads: list[str] = []

        for filename in filenames_to_process:
            if filename not in files_by_name:
                raise FileNotFoundError(f"File {filename} not found during directory ingestion")
            file_hash, file_path = files_by_name[filename]
            json_text = file_path.read_text(encoding="utf-8")
            model = FindingModelFull.model_validate_json(json_text)
            # Only validate new models (not updates of existing models)
            if filename not in updated_entries and not allow_duplicate_synonyms:
                validation_errors = self._validate_model(model)
                if validation_errors:
                    joined = "; ".join(validation_errors)
                    raise ValueError(f"Model validation failed for {filename}: {joined}")
            search_text = self._build_search_text(model)
            metadata.append((model, filename, file_hash, search_text, json_text))
            embedding_payloads.append(self._build_embedding_text(model))

        return metadata, embedding_payloads

    async def _generate_embeddings(self, embedding_payloads: Sequence[str]) -> list[list[float]]:
        if not embedding_payloads:
            return []

        client = await self._ensure_openai_client()
        raw_embeddings = await batch_embeddings_for_duckdb(embedding_payloads, client=client)

        embeddings: list[list[float]] = []
        for embedding in raw_embeddings:
            if embedding is None:
                raise RuntimeError("Failed to generate embeddings for one or more models")
            embeddings.append(embedding)
        return embeddings

    def _build_row_data(
        self,
        metadata: Sequence[tuple[FindingModelFull, str, str, str, str]],
        embeddings: Sequence[list[float]],
    ) -> _RowData:
        model_rows: list[tuple[object, ...]] = []
        synonym_rows: list[tuple[str, str]] = []
        tag_rows: list[tuple[str, str]] = []
        attribute_rows: list[tuple[str, str, str, str, str]] = []
        people_rows_dict: dict[str, tuple[str, str, str, str, str | None]] = {}
        model_people_rows: list[tuple[str, str, str, int]] = []
        organization_rows_dict: dict[str, tuple[str, str, str | None]] = {}
        model_organization_rows: list[tuple[str, str, str, int]] = []
        json_rows: list[tuple[str, str]] = []

        for (model, filename, file_hash, search_text, json_text), embedding in zip(metadata, embeddings, strict=True):
            model_rows.append((
                model.oifm_id,
                normalize_name(model.name),
                model.name,
                filename,
                file_hash,
                model.description,
                search_text,
                embedding,
            ))
            json_rows.append((model.oifm_id, json_text))

            # Deduplicate to avoid PRIMARY KEY violations
            unique_synonyms = list(dict.fromkeys(model.synonyms or []))
            unique_tags = list(dict.fromkeys(model.tags or []))
            synonym_rows.extend((model.oifm_id, synonym) for synonym in unique_synonyms)
            tag_rows.extend((model.oifm_id, tag) for tag in unique_tags)
            attribute_rows.extend(
                (
                    attribute.oifma_id,
                    model.oifm_id,
                    model.name,
                    attribute.name,
                    str(attribute.type),
                )
                for attribute in model.attributes
            )

            for order, contributor in enumerate(model.contributors or []):
                if isinstance(contributor, Person):
                    people_rows_dict[contributor.github_username] = (
                        contributor.github_username,
                        contributor.name,
                        str(contributor.email),
                        contributor.organization_code,
                        str(contributor.url) if contributor.url else None,
                    )
                    model_people_rows.append((
                        model.oifm_id,
                        contributor.github_username,
                        DEFAULT_CONTRIBUTOR_ROLE,
                        order,
                    ))
                elif isinstance(contributor, Organization):
                    organization_rows_dict[contributor.code] = (
                        contributor.code,
                        contributor.name,
                        str(contributor.url) if contributor.url else None,
                    )
                    model_organization_rows.append((
                        model.oifm_id,
                        contributor.code,
                        DEFAULT_CONTRIBUTOR_ROLE,
                        order,
                    ))

        return _RowData(
            model_rows=model_rows,
            synonym_rows=synonym_rows,
            tag_rows=tag_rows,
            attribute_rows=attribute_rows,
            people_rows=list(people_rows_dict.values()),
            model_people_rows=model_people_rows,
            organization_rows=list(organization_rows_dict.values()),
            model_organization_rows=model_organization_rows,
            json_rows=json_rows,
        )

    def _apply_batch_mutations(self, conn: duckdb.DuckDBPyConnection, payload: _BatchPayload) -> None:
        if payload.ids_to_delete:
            self._delete_denormalized_records(conn, payload.ids_to_delete)
            placeholders = ", ".join(["?"] * len(payload.ids_to_delete))
            conn.execute(
                f"DELETE FROM finding_models WHERE oifm_id IN ({placeholders})",
                payload.ids_to_delete,
            )
            logger.debug(
                "Deleted {} existing models during batch apply: {}",
                len(payload.ids_to_delete),
                sorted(payload.ids_to_delete),
            )

        statements: list[tuple[str, str, str, Sequence[tuple[object, ...]]]] = [
            (
                "finding_models",
                "inserted",
                """
                INSERT INTO finding_models (
                    oifm_id,
                    slug_name,
                    name,
                    filename,
                    file_hash_sha256,
                    description,
                    search_text,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload.model_rows,
            ),
            (
                "synonyms",
                "inserted",
                "INSERT INTO synonyms (oifm_id, synonym) VALUES (?, ?)",
                payload.synonym_rows,
            ),
            (
                "tags",
                "inserted",
                "INSERT INTO tags (oifm_id, tag) VALUES (?, ?)",
                payload.tag_rows,
            ),
            (
                "attributes",
                "inserted",
                """
                INSERT INTO attributes (
                    attribute_id,
                    oifm_id,
                    model_name,
                    attribute_name,
                    attribute_type
                ) VALUES (?, ?, ?, ?, ?)
                """,
                payload.attribute_rows,
            ),
            (
                "people",
                "upserted",
                """
                INSERT INTO people (
                    github_username,
                    name,
                    email,
                    organization_code,
                    url
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (github_username) DO UPDATE SET
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    organization_code = EXCLUDED.organization_code,
                    url = EXCLUDED.url,
                    updated_at = now()
                """,
                payload.people_rows,
            ),
            (
                "model_people",
                "upserted",
                """
                INSERT INTO model_people (oifm_id, person_id, role, display_order)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (oifm_id, person_id, role) DO UPDATE SET display_order = EXCLUDED.display_order
                """,
                payload.model_people_rows,
            ),
            (
                "organizations",
                "upserted",
                """
                INSERT INTO organizations (
                    code,
                    name,
                    url
                ) VALUES (?, ?, ?)
                ON CONFLICT (code) DO UPDATE SET
                    name = EXCLUDED.name,
                    url = EXCLUDED.url,
                    updated_at = now()
                """,
                payload.organization_rows,
            ),
            (
                "model_organizations",
                "upserted",
                """
                INSERT INTO model_organizations (oifm_id, organization_id, role, display_order)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (oifm_id, organization_id, role) DO UPDATE SET display_order = EXCLUDED.display_order
                """,
                payload.model_organization_rows,
            ),
            (
                "finding_model_json",
                "upserted",
                """
                INSERT INTO finding_model_json (oifm_id, model_json)
                VALUES (?, ?)
                ON CONFLICT (oifm_id) DO UPDATE SET
                    model_json = EXCLUDED.model_json
                """,
                payload.json_rows,
            ),
        ]

        for table_name, action, statement, rows in statements:
            if rows:
                conn.executemany(statement, rows)
                logger.debug(
                    "Batch {} {} rows in {}",
                    action,
                    len(rows),
                    table_name,
                )

    async def update_from_directory(
        self,
        directory: str | Path,
        *,
        allow_duplicate_synonyms: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, int]:
        """Batch-update the index to match the contents of a directory.

        Args:
            directory: Path to directory containing .fm.json files
            allow_duplicate_synonyms: Allow models with duplicate synonyms
            progress_callback: Optional callback for progress updates (receives status messages)
        """

        directory_path = Path(directory).expanduser()
        if not directory_path.is_dir():
            raise ValueError(f"{directory_path} is not a valid directory.")

        if progress_callback:
            progress_callback("Scanning directory for .fm.json files...")
        files = self._collect_directory_files(directory_path)
        logger.info(
            "Refreshing DuckDB index from {} ({} files)",
            directory_path,
            len(files),
        )

        conn = self._ensure_writable_connection()
        await self.setup()

        conn.execute("DROP TABLE IF EXISTS tmp_directory_files")
        conn.execute("CREATE TEMP TABLE tmp_directory_files(filename TEXT, file_hash_sha256 TEXT)")

        try:
            if progress_callback:
                progress_callback(f"Analyzing {len(files)} files...")
            self._stage_directory_files(conn, files)
            added_filenames, updated_entries, removed_ids = self._classify_directory_changes(conn)
            logger.debug(
                "Directory diff computed for {}: added={} updated={} removed={}",
                directory_path,
                len(added_filenames),
                len(updated_entries),
                len(removed_ids),
            )
            if not added_filenames and not updated_entries and not removed_ids:
                logger.info("DuckDB index already in sync with {}", directory_path)
                if progress_callback:
                    progress_callback("Index already up to date")
                return {"added": 0, "updated": 0, "removed": 0}

            files_by_name = {filename: (file_hash, path) for filename, file_hash, path in files}
            filenames_to_process = sorted(added_filenames | set(updated_entries.keys()))

            if progress_callback:
                total = len(filenames_to_process)
                progress_callback(f"Processing {total} models (loading and generating embeddings)...")

            payload = await self._prepare_batch_payload(
                filenames_to_process,
                files_by_name,
                updated_entries,
                removed_ids,
                allow_duplicate_synonyms=allow_duplicate_synonyms,
            )

            self._execute_batch_directory_update(conn, payload, progress_callback)

            if progress_callback:
                progress_callback(
                    f"Complete: {len(added_filenames)} added, {len(updated_entries)} updated, {len(removed_ids)} removed"
                )

            logger.info(
                "DuckDB index refreshed: added={} updated={} removed={}",
                len(added_filenames),
                len(updated_entries),
                len(removed_ids),
            )
            return {
                "added": len(added_filenames),
                "updated": len(updated_entries),
                "removed": len(removed_ids),
            }
        except Exception:
            logger.exception("Failed to refresh DuckDB index from {}", directory_path)
            raise
        finally:
            conn.execute("DROP TABLE IF EXISTS tmp_directory_files")

    async def remove_entry(self, oifm_id: str) -> bool:
        """Remove a finding model by ID."""

        conn = self._ensure_writable_connection()
        await self.setup()

        conn.execute("BEGIN TRANSACTION")
        try:
            self._drop_search_indexes(conn)
            self._delete_denormalized_records(conn, [oifm_id])
            deleted = conn.execute(
                "DELETE FROM finding_models WHERE oifm_id = ? RETURNING oifm_id",
                (oifm_id,),
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:  # pragma: no cover - rollback path
            conn.execute("ROLLBACK")
            self._create_search_indexes(conn)
            raise
        self._create_search_indexes(conn)
        return deleted is not None

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
            query: Search query string
            limit: Maximum number of results to return
            tags: Optional list of tags - models must have ALL specified tags
        """
        conn = self._ensure_connection()

        # Exact matches take priority - return immediately if found
        exact_matches = self._search_exact(conn, query, tags=tags)
        if exact_matches:
            return exact_matches[:limit]

        # Get both FTS and semantic results
        fts_matches = self._search_fts(conn, query, limit=limit, tags=tags)
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

    async def search_batch(self, queries: list[str], *, limit: int = 10) -> dict[str, list[IndexEntry]]:
        """Search multiple queries efficiently with single embedding call and RRF fusion.

        Embeds ALL queries in a single OpenAI API call for efficiency,
        then performs hybrid search with RRF fusion for each query.

        Args:
            queries: List of search query strings
            limit: Maximum number of results per query

        Returns:
            Dictionary mapping each query string to its list of results
        """
        if not queries:
            return {}

        conn = self._ensure_connection()
        client = await self._ensure_openai_client()

        # Generate embeddings for all queries in a single batch API call
        embeddings = await batch_embeddings_for_duckdb(queries, client=client)

        results: dict[str, list[IndexEntry]] = {}
        query: str
        for query, embedding in zip(queries, embeddings, strict=True):
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
            >>> index = DuckDBIndex(read_only=False)
            >>> await index.setup()
            >>> oifm_id = index.generate_model_id("GMTS")
            >>> # Returns "OIFM_GMTS_123456"
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
            >>> index = DuckDBIndex(read_only=False)
            >>> await index.setup()
            >>> # Infer source from model
            >>> oifma_id = index.generate_attribute_id(model_oifm_id="OIFM_GMTS_123456")
            >>> # Returns "OIFMA_GMTS_234567"
            >>> # Or use explicit source
            >>> oifma_id = index.generate_attribute_id(source="GMTS")
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
            index = Index()
            await index.setup()
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

            index = Index()
            await index.setup()
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
            if not self.read_only:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = setup_duckdb_connection(self.db_path, read_only=self.read_only)
        return self.conn

    def _ensure_writable_connection(self) -> duckdb.DuckDBPyConnection:
        if self.read_only:
            raise RuntimeError("DuckDBIndex is in read-only mode; write operation not permitted")
        return self._ensure_connection()

    async def _ensure_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            settings.check_ready_for_openai()
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return self._openai_client

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

    def _upsert_contributors(self, conn: duckdb.DuckDBPyConnection, model: FindingModelFull) -> None:
        contributors = list(model.contributors or [])
        for order, contributor in enumerate(contributors):
            if isinstance(contributor, Person):
                conn.execute(
                    """
                    INSERT INTO people (github_username, name, email, organization_code, url)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (github_username) DO UPDATE SET
                        name = EXCLUDED.name,
                        email = EXCLUDED.email,
                        organization_code = EXCLUDED.organization_code,
                        url = EXCLUDED.url,
                        updated_at = now()
                    """,
                    (
                        contributor.github_username,
                        contributor.name,
                        str(contributor.email),
                        contributor.organization_code,
                        str(contributor.url) if contributor.url else None,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO model_people (oifm_id, person_id, role, display_order)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (oifm_id, person_id, role) DO UPDATE SET display_order = EXCLUDED.display_order
                    """,
                    (model.oifm_id, contributor.github_username, DEFAULT_CONTRIBUTOR_ROLE, order),
                )
            elif isinstance(contributor, Organization):
                conn.execute(
                    """
                    INSERT INTO organizations (code, name, url)
                    VALUES (?, ?, ?)
                    ON CONFLICT (code) DO UPDATE SET
                        name = EXCLUDED.name,
                        url = EXCLUDED.url,
                        updated_at = now()
                    """,
                    (
                        contributor.code,
                        contributor.name,
                        str(contributor.url) if contributor.url else None,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO model_organizations (oifm_id, organization_id, role, display_order)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (oifm_id, organization_id, role) DO UPDATE SET display_order = EXCLUDED.display_order
                    """,
                    (model.oifm_id, contributor.code, DEFAULT_CONTRIBUTOR_ROLE, order),
                )

    def _replace_synonyms(
        self,
        conn: duckdb.DuckDBPyConnection,
        oifm_id: str,
        synonyms: Sequence[str] | None,
    ) -> None:
        conn.execute("DELETE FROM synonyms WHERE oifm_id = ?", (oifm_id,))
        if synonyms:
            # Deduplicate synonyms to avoid PRIMARY KEY violations
            unique_synonyms = list(dict.fromkeys(synonyms))
            conn.executemany(
                "INSERT INTO synonyms (oifm_id, synonym) VALUES (?, ?)",
                [(oifm_id, synonym) for synonym in unique_synonyms],
            )

    def _replace_tags(
        self,
        conn: duckdb.DuckDBPyConnection,
        oifm_id: str,
        tags: Sequence[str] | None,
    ) -> None:
        conn.execute("DELETE FROM tags WHERE oifm_id = ?", (oifm_id,))
        if tags:
            # Deduplicate tags to avoid PRIMARY KEY violations
            unique_tags = list(dict.fromkeys(tags))
            conn.executemany(
                "INSERT INTO tags (oifm_id, tag) VALUES (?, ?)",
                [(oifm_id, tag) for tag in unique_tags],
            )

    def _replace_attributes(self, conn: duckdb.DuckDBPyConnection, model: FindingModelFull) -> None:
        conn.execute("DELETE FROM attributes WHERE oifm_id = ?", (model.oifm_id,))
        attribute_rows = [
            (
                attribute.oifma_id,
                model.oifm_id,
                model.name,
                attribute.name,
                str(attribute.type),
            )
            for attribute in model.attributes
        ]
        conn.executemany(
            """
            INSERT INTO attributes (
                attribute_id,
                oifm_id,
                model_name,
                attribute_name,
                attribute_type
            ) VALUES (?, ?, ?, ?, ?)
            """,
            attribute_rows,
        )

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

    def _validate_model(self, model: FindingModelFull) -> list[str]:
        """Validate that a model can be added without conflicts.

        Checks for uniqueness of OIFM ID, name, slug_name, and attribute IDs.
        Returns a list of error messages (empty if valid).

        Args:
            model: The finding model to validate

        Returns:
            List of validation error messages (empty list means valid)
        """
        errors: list[str] = []
        conn = self._ensure_connection()

        # Check OIFM ID uniqueness
        row = conn.execute(
            "SELECT oifm_id FROM finding_models WHERE oifm_id = ?",
            (model.oifm_id,),
        ).fetchone()
        if row is not None:
            errors.append(f"OIFM ID '{model.oifm_id}' already exists")

        # Check name uniqueness (case-insensitive)
        row = conn.execute(
            "SELECT name FROM finding_models WHERE LOWER(name) = LOWER(?) AND oifm_id != ?",
            (model.name, model.oifm_id),
        ).fetchone()
        if row is not None:
            errors.append(f"Name '{model.name}' already in use")

        # Check slug_name uniqueness
        slug_name = normalize_name(model.name)
        row = conn.execute(
            "SELECT slug_name FROM finding_models WHERE slug_name = ? AND oifm_id != ?",
            (slug_name, model.oifm_id),
        ).fetchone()
        if row is not None:
            errors.append(f"Slug name '{slug_name}' already in use")

        # Check attribute ID conflicts (any attribute IDs already used by OTHER models)
        if model.attributes:
            attribute_ids = [attr.oifma_id for attr in model.attributes]
            if attribute_ids:
                placeholders = ", ".join("?" for _ in attribute_ids)
                conflicting_rows = conn.execute(
                    f"""
                    SELECT attribute_id, model_name
                    FROM attributes
                    WHERE attribute_id IN ({placeholders})
                      AND oifm_id != ?
                    """,
                    [*attribute_ids, model.oifm_id],
                ).fetchall()
                for attr_id, model_name in conflicting_rows:
                    errors.append(f"Attribute ID '{attr_id}' already used by model '{model_name}'")

        return errors

    def _calculate_file_hash(self, filename: Path) -> str:
        if not filename.exists() or not filename.is_file():
            raise FileNotFoundError(f"File {filename} not found")
        return hashlib.sha256(filename.read_bytes()).hexdigest()

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

    def _delete_denormalized_records(
        self,
        conn: duckdb.DuckDBPyConnection,
        oifm_ids: Sequence[str],
    ) -> None:
        unique_ids = list(dict.fromkeys(oifm_ids))
        if not unique_ids:
            return

        tables = ("model_people", "model_organizations", "synonyms", "attributes", "tags", "finding_model_json")
        placeholders = ", ".join("?" for _ in unique_ids)
        for table in tables:
            conn.execute(
                f"DELETE FROM {table} WHERE oifm_id IN ({placeholders})",
                unique_ids,
            )

    def _create_search_indexes(self, conn: duckdb.DuckDBPyConnection) -> None:
        create_hnsw_index(
            conn,
            table="finding_models",
            column="embedding",
            index_name="finding_models_embedding_hnsw",
            metric="l2sq",
        )
        create_fts_index(
            conn,
            "finding_models",
            "oifm_id",
            "search_text",
            stemmer="porter",
            stopwords="english",
            lower=1,
            overwrite=True,
        )

    def _load_base_contributors(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Load base organizations and people if the tables are empty."""
        import json
        from importlib.resources import files

        # Check if organizations table is empty
        org_result = conn.execute("SELECT COUNT(*) FROM organizations").fetchone()
        org_count = org_result[0] if org_result else 0
        if org_count == 0:
            # Load base organizations from package data
            base_orgs_file = files("findingmodel") / "data" / "base_organizations.jsonl"
            with base_orgs_file.open("r") as f:
                for line in f:
                    if line.strip():
                        org_data = json.loads(line)
                        conn.execute(
                            """
                            INSERT INTO organizations (code, name, url)
                            VALUES (?, ?, ?)
                            """,
                            (org_data["code"], org_data["name"], org_data.get("url")),
                        )

        # Check if people table is empty
        people_result = conn.execute("SELECT COUNT(*) FROM people").fetchone()
        people_count = people_result[0] if people_result else 0
        if people_count == 0:
            # Load base people from package data
            base_people_file = files("findingmodel") / "data" / "base_people.jsonl"
            with base_people_file.open("r") as f:
                for line in f:
                    if line.strip():
                        person_data = json.loads(line)
                        conn.execute(
                            """
                            INSERT INTO people (github_username, name, email, organization_code, url)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                person_data["github_username"],
                                person_data["name"],
                                person_data["email"],
                                person_data.get("organization_code"),
                                person_data.get("url"),
                            ),
                        )

    def _drop_search_indexes(self, conn: duckdb.DuckDBPyConnection) -> None:
        drop_search_indexes(conn, table="finding_models", hnsw_index_name="finding_models_embedding_hnsw")

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

        embedding = await get_embedding_for_duckdb(
            trimmed_query,
            client=await self._ensure_openai_client(),
        )
        if embedding is None:
            return []

        return self._search_semantic_with_embedding(conn, embedding, limit=limit, tags=tags)

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
    "IndexReturnType",
]
