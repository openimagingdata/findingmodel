"""Shared utilities for DuckDB-based search components."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Final, Iterable

import duckdb
from openai import AsyncOpenAI

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.tools.common import get_embedding, get_embeddings_batch

ScoreTuple = tuple[str, float]
_DEFAULT_EXTENSIONS: Final[tuple[str, ...]] = ("fts", "vss")


def setup_duckdb_connection(
    db_path: Path | str,
    *,
    read_only: bool = True,
    extensions: Iterable[str] = _DEFAULT_EXTENSIONS,
) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with the standard extensions loaded."""
    connection = duckdb.connect(str(db_path), read_only=read_only)

    for extension in extensions:
        if not read_only:
            connection.execute(f"INSTALL {extension}")
        connection.execute(f"LOAD {extension}")

    if not read_only:
        connection.execute("SET hnsw_enable_experimental_persistence = true")

    return connection


async def get_embedding_for_duckdb(
    text: str,
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float] | None:
    """Generate a float32 embedding suitable for DuckDB storage."""
    resolved_model = model or settings.openai_embedding_model
    resolved_dimensions = dimensions or settings.openai_embedding_dimensions
    embedding = await get_embedding(
        text,
        client=client,
        model=resolved_model,
        dimensions=resolved_dimensions,
    )

    if embedding is None:
        return None

    return _to_float32(embedding)


async def batch_embeddings_for_duckdb(
    texts: Sequence[str],
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[list[float] | None]:
    """Generate float32 embeddings for several texts in a single API call."""
    if not texts:
        return []

    resolved_model = model or settings.openai_embedding_model
    resolved_dimensions = dimensions or settings.openai_embedding_dimensions
    embeddings = await get_embeddings_batch(
        list(texts),
        client=client,
        model=resolved_model,
        dimensions=resolved_dimensions,
    )

    results: list[list[float] | None] = []
    for embedding in embeddings:
        results.append(None if embedding is None else _to_float32(embedding))

    return results


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalise scores to the [0, 1] range."""
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if minimum == maximum:
        return [1.0 for _ in scores]

    span = maximum - minimum
    return [(score - minimum) / span for score in scores]


def weighted_fusion(
    results_a: Sequence[ScoreTuple],
    results_b: Sequence[ScoreTuple],
    *,
    weight_a: float = 0.3,
    weight_b: float = 0.7,
    normalise: bool = True,
) -> list[ScoreTuple]:
    """Combine two result sets using weighted score fusion."""
    scores_a = dict(results_a)
    scores_b = dict(results_b)

    if normalise and scores_a:
        keys_a = tuple(scores_a.keys())
        normalised_a = normalize_scores(list(scores_a.values()))
        scores_a = dict(zip(keys_a, normalised_a, strict=True))

    if normalise and scores_b:
        keys_b = tuple(scores_b.keys())
        normalised_b = normalize_scores(list(scores_b.values()))
        scores_b = dict(zip(keys_b, normalised_b, strict=True))

    identifiers = set(scores_a) | set(scores_b)
    combined: list[ScoreTuple] = []

    for identifier in identifiers:
        combined_score = weight_a * scores_a.get(identifier, 0.0) + weight_b * scores_b.get(identifier, 0.0)
        combined.append((identifier, combined_score))

    combined.sort(key=lambda item: item[1], reverse=True)
    return combined


def rrf_fusion(
    results_a: Sequence[ScoreTuple],
    results_b: Sequence[ScoreTuple],
    *,
    k: int = 60,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
) -> list[ScoreTuple]:
    """Combine two result sets using Reciprocal Rank Fusion (RRF)."""
    ranks_a = {identifier: index + 1 for index, (identifier, _) in enumerate(results_a)}
    ranks_b = {identifier: index + 1 for index, (identifier, _) in enumerate(results_b)}

    identifiers = set(ranks_a) | set(ranks_b)
    combined: list[ScoreTuple] = []

    for identifier in identifiers:
        rank_a = ranks_a.get(identifier, len(results_a) + 1)
        rank_b = ranks_b.get(identifier, len(results_b) + 1)
        score = weight_a / (k + rank_a) + weight_b / (k + rank_b)
        combined.append((identifier, score))

    combined.sort(key=lambda item: item[1], reverse=True)
    return combined


def l2_to_cosine_similarity(l2_distance: float) -> float:
    """Convert an L2 distance to an approximate cosine similarity."""
    return 1.0 - (l2_distance / 2.0)


def _to_float32(values: Sequence[float]) -> list[float]:
    """Convert an iterable of floats to 32-bit precision."""
    return list(array("f", values))


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
    "ScoreTuple",
    "batch_embeddings_for_duckdb",
    "create_fts_index",
    "create_hnsw_index",
    "drop_search_indexes",
    "get_embedding_for_duckdb",
    "l2_to_cosine_similarity",
    "normalize_scores",
    "rrf_fusion",
    "setup_duckdb_connection",
    "weighted_fusion",
]
