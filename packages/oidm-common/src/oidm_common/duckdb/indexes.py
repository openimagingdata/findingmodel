"""DuckDB index management utilities."""

from __future__ import annotations

import logging
from contextlib import suppress

import duckdb

logger = logging.getLogger(__name__)


def create_fts_index(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    id_column: str,
    *text_columns: str,
    stemmer: str = "porter",
    stopwords: str = "english",
    lower: int = 0,
    overwrite: bool = True,
) -> None:
    """Create a full-text search index on the specified table and columns.

    Args:
        conn: Active DuckDB connection
        table: Table name to index
        id_column: ID column name (used for match_bm25 queries)
        text_columns: One or more text column names to include in the FTS index
        stemmer: Stemmer to use (default: "porter")
        stopwords: Stopword list to use (default: "english")
        lower: Whether to lowercase text during indexing (0=no, 1=yes; default: 0)
        overwrite: Whether to overwrite existing index (default: True)
    """
    if not text_columns:
        raise ValueError("At least one text column must be specified")

    columns_str = ", ".join([f"'{id_column}'"] + [f"'{col}'" for col in text_columns])
    overwrite_flag = 1 if overwrite else 0
    conn.execute(f"""
        PRAGMA create_fts_index(
            '{table}',
            {columns_str},
            stemmer = '{stemmer}',
            stopwords = '{stopwords}',
            lower = {lower},
            overwrite = {overwrite_flag}
        )
    """)
    logger.info(f"Created FTS index on table '{table}' with columns: {', '.join(text_columns)}")


def create_hnsw_index(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    index_name: str | None = None,
    *,
    metric: str = "cosine",
    ef_construction: int = 128,
    ef_search: int = 64,
    m: int = 16,
) -> None:
    """Create an HNSW vector similarity index.

    Args:
        conn: Active DuckDB connection
        table: Table name to index
        column: Vector column name to index
        index_name: Optional custom index name (default: idx_{table}_{column}_hnsw)
        metric: Distance metric (default: "cosine")
        ef_construction: HNSW construction parameter (default: 128)
        ef_search: HNSW search parameter (default: 64)
        m: HNSW M parameter (default: 16)
    """
    if index_name is None:
        index_name = f"idx_{table}_{column}_hnsw"

    try:
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table}
            USING HNSW ({column})
            WITH (metric = '{metric}', ef_construction = {ef_construction}, ef_search = {ef_search}, M = {m})
        """)
        logger.info(f"Created HNSW index '{index_name}' on {table}.{column}")
    except Exception as e:
        logger.warning(f"Could not create HNSW index '{index_name}': {e}")
        logger.warning("Vector search will use brute force instead of index")
        raise


def drop_search_indexes(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    hnsw_index_name: str | None = None,
) -> None:
    """Drop HNSW and FTS indexes for a table.

    Args:
        conn: Active DuckDB connection
        table: Table name whose indexes should be dropped
        hnsw_index_name: Optional HNSW index name (if not provided, no HNSW index is dropped)
    """
    # Drop HNSW index if specified
    if hnsw_index_name is not None:
        with suppress(duckdb.Error):  # Index may not exist or extension unavailable
            conn.execute(f"DROP INDEX IF EXISTS {hnsw_index_name}")
            logger.debug(f"Dropped HNSW index '{hnsw_index_name}'")

    # Drop FTS index
    with suppress(duckdb.Error):  # FTS index may not exist or extension unavailable
        conn.execute(f"PRAGMA drop_fts_index('{table}')")
        logger.debug(f"Dropped FTS index for table '{table}'")


__all__ = [
    "create_fts_index",
    "create_hnsw_index",
    "drop_search_indexes",
]
