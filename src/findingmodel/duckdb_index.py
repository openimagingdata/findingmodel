"""DuckDB-backed implementation of the finding model index."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from types import TracebackType

import duckdb
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from findingmodel.common import normalize_name
from findingmodel.config import settings
from findingmodel.contributor import Organization, Person
from findingmodel.finding_model import FindingModelFull
from findingmodel.tools.duckdb_utils import get_embedding_for_duckdb, setup_duckdb_connection

DEFAULT_CONTRIBUTOR_ROLE = "contributor"


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
    attributes: list[AttributeInfo] = Field(default_factory=list, min_length=1)

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
        PRIMARY KEY (oifm_id, person_id, role),
        FOREIGN KEY (oifm_id) REFERENCES finding_models(oifm_id) ON DELETE CASCADE,
        FOREIGN KEY (person_id) REFERENCES people(github_username) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_organizations (
        oifm_id VARCHAR NOT NULL,
        organization_id VARCHAR NOT NULL,
        role VARCHAR NOT NULL DEFAULT 'contributor',
        display_order INTEGER,
        PRIMARY KEY (oifm_id, organization_id, role),
        FOREIGN KEY (oifm_id) REFERENCES finding_models(oifm_id) ON DELETE CASCADE,
        FOREIGN KEY (organization_id) REFERENCES organizations(code) ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS synonyms (
        oifm_id VARCHAR NOT NULL,
        synonym VARCHAR NOT NULL,
        PRIMARY KEY (oifm_id, synonym),
        FOREIGN KEY (oifm_id) REFERENCES finding_models(oifm_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS attributes (
        attribute_id VARCHAR PRIMARY KEY,
        oifm_id VARCHAR NOT NULL,
        model_name VARCHAR NOT NULL,
        attribute_name VARCHAR NOT NULL,
        attribute_type VARCHAR NOT NULL,
        FOREIGN KEY (oifm_id) REFERENCES finding_models(oifm_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tags (
        oifm_id VARCHAR NOT NULL,
        tag VARCHAR NOT NULL,
        PRIMARY KEY (oifm_id, tag),
        FOREIGN KEY (oifm_id) REFERENCES finding_models(oifm_id) ON DELETE CASCADE
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
        self.db_path = Path(db_path or settings.duckdb_index_path).expanduser()
        self.read_only = read_only
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._openai_client: AsyncOpenAI | None = None

    async def setup(self) -> None:
        """Ensure the database exists, connection opened, and schema ready."""

        conn = self._ensure_connection()

        if self.read_only:
            return

        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        for statement in _INDEX_STATEMENTS:
            conn.execute(statement)

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
        if model is None:
            model = FindingModelFull.model_validate_json(file_path.read_text())

        existing = conn.execute(
            """
            SELECT oifm_id, file_hash_sha256
            FROM finding_models
            WHERE oifm_id = ? OR filename = ?
            """,
            (model.oifm_id, file_path.name),
        ).fetchone()

        status = IndexReturnType.ADDED
        if existing is not None:
            status = IndexReturnType.UPDATED
            if existing[1] == file_hash and existing[0] == model.oifm_id:
                return IndexReturnType.UNCHANGED

        validation_errors = [] if allow_duplicate_synonyms else self._validate_model(model)
        if validation_errors:
            raise ValueError(f"Model validation failed: {'; '.join(validation_errors)}")

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
                    embedding,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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

            self._upsert_contributors(conn, model)
            self._replace_synonyms(conn, model.oifm_id, model.synonyms)
            self._replace_tags(conn, model.oifm_id, model.tags)
            self._replace_attributes(conn, model)

            conn.execute("COMMIT")
        except Exception:  # pragma: no cover - rollback path
            conn.execute("ROLLBACK")
            raise

        return status

    async def remove_entry(self, oifm_id: str) -> bool:
        """Remove a finding model by ID."""

        conn = self._ensure_writable_connection()
        await self.setup()

        deleted = conn.execute(
            "DELETE FROM finding_models WHERE oifm_id = ? RETURNING oifm_id",
            (oifm_id,),
        ).fetchone()
        return deleted is not None

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        tags: Sequence[str] | None = None,
    ) -> list[IndexEntry]:
        """Search for finding models, returning exact matches if present."""

        conn = self._ensure_connection()
        exact_matches = self._search_exact(conn, query, tags=tags)
        if exact_matches:
            return exact_matches[:limit]
        return []

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
            SELECT oifm_id, name, slug_name, filename, file_hash_sha256, description
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
            synonyms=synonyms or None,
            tags=tags or None,
            contributors=contributors or None,
            attributes=attributes,
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
                    INSERT INTO people (github_username, name, email, organization_code, url, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (github_username) DO UPDATE SET
                        name = EXCLUDED.name,
                        email = EXCLUDED.email,
                        organization_code = EXCLUDED.organization_code,
                        url = EXCLUDED.url,
                        updated_at = CURRENT_TIMESTAMP
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
                    INSERT INTO organizations (code, name, url, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (code) DO UPDATE SET
                        name = EXCLUDED.name,
                        url = EXCLUDED.url,
                        updated_at = CURRENT_TIMESTAMP
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
            conn.executemany(
                "INSERT INTO synonyms (oifm_id, synonym) VALUES (?, ?)",
                [(oifm_id, synonym) for synonym in synonyms],
            )

    def _replace_tags(
        self,
        conn: duckdb.DuckDBPyConnection,
        oifm_id: str,
        tags: Sequence[str] | None,
    ) -> None:
        conn.execute("DELETE FROM tags WHERE oifm_id = ?", (oifm_id,))
        if tags:
            conn.executemany(
                "INSERT INTO tags (oifm_id, tag) VALUES (?, ?)",
                [(oifm_id, tag) for tag in tags],
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
        """Placeholder until validation is ported in later steps."""

        _ = model
        return []

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


__all__ = [
    "AttributeInfo",
    "DuckDBIndex",
    "IndexEntry",
    "IndexReturnType",
]
