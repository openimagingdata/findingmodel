"""Embedding utilities for DuckDB-based search components.

This module provides convenience wrappers around oidm-common embedding generation
with findingmodel config defaults. For general DuckDB utilities, see oidm_common.duckdb.
"""

from __future__ import annotations

from collections.abc import Sequence

from oidm_common.embeddings import get_embedding, get_embeddings_batch

from findingmodel.config import settings


async def get_embedding_for_duckdb(
    text: str,
    *,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float] | None:
    """Generate a float32 embedding suitable for DuckDB storage.

    Args:
        text: Text to embed
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embedding (default: from config settings)

    Returns:
        Float32 embedding vector or None if unavailable
    """
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    resolved_model = model or settings.openai_embedding_model
    resolved_dimensions = dimensions or settings.openai_embedding_dimensions

    return await get_embedding(
        text,
        api_key=api_key,
        model=resolved_model,
        dimensions=resolved_dimensions,
    )


async def batch_embeddings_for_duckdb(
    texts: Sequence[str],
    *,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[list[float] | None]:
    """Generate float32 embeddings for several texts in a single API call.

    Args:
        texts: Sequence of texts to embed
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embeddings (default: from config settings)

    Returns:
        List of float32 embedding vectors (or None for each if unavailable)
    """
    if not texts:
        return []

    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    resolved_model = model or settings.openai_embedding_model
    resolved_dimensions = dimensions or settings.openai_embedding_dimensions

    return await get_embeddings_batch(
        list(texts),
        api_key=api_key,
        model=resolved_model,
        dimensions=resolved_dimensions,
    )


__all__ = [
    "batch_embeddings_for_duckdb",
    "get_embedding_for_duckdb",
]
