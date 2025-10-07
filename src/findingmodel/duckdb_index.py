"""DuckDB-backed implementation of the finding model index."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import cast

import duckdb

from findingmodel.config import settings
from findingmodel.tools.duckdb_utils import setup_duckdb_connection


class DuckDBIndex:
    """DuckDB-based index with read-only by default connections."""

    def __init__(self, db_path: str | Path | None = None, *, read_only: bool = True) -> None:
        default_path = cast(str, getattr(settings, "duckdb_index_path", "data/finding_models.duckdb"))
        self.db_path = Path(db_path or default_path).expanduser()
        self.read_only = read_only
        self.conn: duckdb.DuckDBPyConnection | None = None

    async def setup(self) -> None:
        """Ensure the database exists and the connection is ready."""
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.conn is None:
            self.conn = setup_duckdb_connection(self.db_path, read_only=self.read_only)

    async def __aenter__(self) -> DuckDBIndex:
        """Return self when entering an async context."""
        if self.conn is None:
            self.conn = setup_duckdb_connection(self.db_path, read_only=self.read_only)
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
