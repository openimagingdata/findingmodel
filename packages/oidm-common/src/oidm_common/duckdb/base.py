"""Base class for read-only DuckDB-backed indexes."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from loguru import logger

from oidm_common.duckdb.connection import setup_duckdb_connection
from oidm_common.embeddings.config import ACTIVE_EMBEDDING_CONFIG, EmbeddingProfileSpec

if TYPE_CHECKING:
    from typing import Self


class ReadOnlyDuckDBIndex:
    """Base for read-only DuckDB-backed indexes.

    Provides: db_path resolution, connection lifecycle,
    auto-open, open/close, sync + async context managers,
    and embedding profile detection from DB metadata.

    Subclasses get a uniform lifecycle:
        - ``__init__`` resolves ``db_path`` from an explicit path or a callable
        - ``open()`` / ``close()`` for explicit lifecycle management
        - sync and async context managers
        - ``_ensure_connection()`` auto-opens on first use

    Subclasses that use embedding vectors should set ``_vector_table``
    and ``_vector_column`` class attributes for dimension detection
    fallback when the DB lacks an ``embedding_profile`` metadata table.
    """

    _vector_table: str = ""
    _vector_column: str = ""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        ensure_db: Callable[[], Path] | None = None,
    ) -> None:
        if db_path is not None:
            self.db_path = Path(db_path).expanduser()
        elif ensure_db is not None:
            self.db_path = ensure_db()
        else:
            raise ValueError("Either db_path or ensure_db must be provided")
        self.conn: duckdb.DuckDBPyConnection | None = None
        self._db_embedding_profile: EmbeddingProfileSpec | None = None
        self._embedding_mismatch_warned = False

    def _ensure_connection(self) -> duckdb.DuckDBPyConnection:
        """Return an open connection, auto-opening if necessary."""
        if self.conn is None:
            self.conn = setup_duckdb_connection(self.db_path, read_only=True)
        return self.conn

    def open(self) -> Self:
        """Open the database connection explicitly. Returns self for chaining."""
        self._ensure_connection()
        return self

    def close(self) -> None:
        """Close the database connection explicitly."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    # -- sync context manager --------------------------------------------------

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *_args: object) -> None:
        self.close()

    # -- async context manager -------------------------------------------------

    async def __aenter__(self) -> Self:
        return self.open()

    async def __aexit__(self, *_args: object) -> None:
        self.close()

    # -- named-dict query helpers ----------------------------------------------

    def _execute_one(
        self,
        conn: duckdb.DuckDBPyConnection,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> dict[str, object] | None:
        """Execute a query and return the single result row as a named dict, or None.

        Uses cursor.description to map column names, eliminating positional
        indexing brittleness. Safe against column additions and reorderings.

        Note: If the query produces duplicate column names, later columns
        silently overwrite earlier ones in the dict. Avoid ambiguous aliases.
        """
        result = conn.execute(sql, list(params) if params is not None else [])
        columns = [d[0] for d in result.description]
        row = result.fetchone()
        return dict(zip(columns, row, strict=True)) if row is not None else None

    def _execute_all(
        self,
        conn: duckdb.DuckDBPyConnection,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> list[dict[str, object]]:
        """Execute a query and return all result rows as named dicts.

        Uses cursor.description to map column names, eliminating positional
        indexing brittleness. Safe against column additions and reorderings.

        Note: If the query produces duplicate column names, later columns
        silently overwrite earlier ones in the dict. Avoid ambiguous aliases.
        """
        result = conn.execute(sql, list(params) if params is not None else [])
        columns = [d[0] for d in result.description]
        return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]

    # -- embedding profile helpers ---------------------------------------------

    def _get_db_embedding_profile(self, conn: duckdb.DuckDBPyConnection) -> EmbeddingProfileSpec:
        """Read embedding profile from DB metadata, falling back to defaults.

        Checks the ``embedding_profile`` metadata table first.  When absent,
        falls back to ``ACTIVE_EMBEDDING_CONFIG`` values, optionally parsing
        dimensions from the vector column type if ``_vector_table`` and
        ``_vector_column`` are set on the subclass.
        """
        if self._db_embedding_profile is not None:
            return self._db_embedding_profile

        provider = ACTIVE_EMBEDDING_CONFIG.provider
        model = ACTIVE_EMBEDDING_CONFIG.model
        dimensions = ACTIVE_EMBEDDING_CONFIG.dimensions

        try:
            has_profile = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'embedding_profile'"
            ).fetchone()
            if has_profile and int(has_profile[0]) > 0:
                row = conn.execute("SELECT provider, model, dimensions FROM embedding_profile LIMIT 1").fetchone()
                if row and isinstance(row[0], str) and isinstance(row[1], str) and isinstance(row[2], int):
                    self._db_embedding_profile = EmbeddingProfileSpec(
                        provider=row[0], model=row[1], dimensions=int(row[2])
                    )
                    return self._db_embedding_profile
        except Exception:
            pass

        # Fallback: parse vector column dimensions from schema if subclass specifies the table/column.
        if self._vector_table and self._vector_column:
            col_row = self._execute_one(
                conn,
                (
                    "SELECT data_type FROM information_schema.columns "
                    f"WHERE table_name = '{self._vector_table}' AND column_name = '{self._vector_column}' LIMIT 1"
                ),
            )
            data_type = col_row.get("data_type") if col_row else None
            if isinstance(data_type, str):
                match = re.search(r"FLOAT\[(\d+)\]", data_type.upper())
                if match is not None:
                    dimensions = int(match.group(1))

        self._db_embedding_profile = EmbeddingProfileSpec(provider=provider, model=model, dimensions=dimensions)
        return self._db_embedding_profile

    def _warn_embedding_mismatch_once(self, actual: int, expected: int) -> None:
        if self._embedding_mismatch_warned:
            return
        self._embedding_mismatch_warned = True
        logger.warning(
            f"Query embedding dimension mismatch (got {actual}, expected {expected}). Falling back to FTS-only results."
        )
