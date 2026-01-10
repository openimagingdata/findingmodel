"""DuckDB connection management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable

import duckdb

_DEFAULT_EXTENSIONS: Final[tuple[str, ...]] = ("fts", "vss")


def setup_duckdb_connection(
    db_path: Path | str,
    *,
    read_only: bool = True,
    extensions: Iterable[str] = _DEFAULT_EXTENSIONS,
) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the standard extensions loaded.

    Args:
        db_path: Path to the DuckDB database file
        read_only: Whether to open the connection in read-only mode
        extensions: Extensions to install and load (default: fts and vss)

    Returns:
        Configured DuckDB connection with extensions loaded

    Note:
        INSTALL and LOAD are idempotent operations. Extensions are cached locally
        after the first install, and subsequent calls use the cached version.
    """
    connection = duckdb.connect(str(db_path), read_only=read_only)

    for extension in extensions:
        connection.execute(f"INSTALL {extension}")
        connection.execute(f"LOAD {extension}")

    if not read_only:
        connection.execute("SET hnsw_enable_experimental_persistence = true")

    return connection


__all__ = [
    "setup_duckdb_connection",
]
