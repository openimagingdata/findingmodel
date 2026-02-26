"""Base class for read-only DuckDB-backed indexes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from oidm_common.duckdb.connection import setup_duckdb_connection

if TYPE_CHECKING:
    from typing_extensions import Self


class ReadOnlyDuckDBIndex:
    """Base for read-only DuckDB-backed indexes.

    Provides: db_path resolution, connection lifecycle,
    auto-open, open/close, sync + async context managers.

    Subclasses get a uniform lifecycle:
        - ``__init__`` resolves ``db_path`` from an explicit path or a callable
        - ``open()`` / ``close()`` for explicit lifecycle management
        - sync and async context managers
        - ``_ensure_connection()`` auto-opens on first use
    """

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
