"""Shared utilities for DuckDB-based search components."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Iterable

import duckdb
from openai import AsyncOpenAI

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


__all__ = [
    "ScoreTuple",
    "batch_embeddings_for_duckdb",
    "get_embedding_for_duckdb",
    "l2_to_cosine_similarity",
    "normalize_scores",
    "rrf_fusion",
    "setup_duckdb_connection",
    "weighted_fusion",
]
