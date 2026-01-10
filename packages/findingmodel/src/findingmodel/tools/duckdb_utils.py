"""Embedding utilities for DuckDB-based search components.

This module contains OpenAI-specific embedding functions that were not extracted
to oidm-common. For general DuckDB utilities, see oidm_common.duckdb.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence

from openai import AsyncOpenAI

from findingmodel.config import settings
from findingmodel.tools.common import get_embedding, get_embeddings_batch


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


def _to_float32(values: Sequence[float]) -> list[float]:
    """Convert an iterable of floats to 32-bit precision."""
    return list(array("f", values))


__all__ = [
    "batch_embeddings_for_duckdb",
    "get_embedding_for_duckdb",
]
