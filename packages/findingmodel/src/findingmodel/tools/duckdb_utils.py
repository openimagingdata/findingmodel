"""Embedding utilities for DuckDB-based search components.

This module provides convenience wrappers around oidm-common embedding generation
with config-based defaults. For general DuckDB utilities, see oidm_common.duckdb.
"""

from __future__ import annotations

from collections.abc import Sequence

from oidm_common.embeddings import generate_embedding, generate_embeddings_batch
from openai import AsyncOpenAI

from findingmodel.config import settings


async def get_embedding_for_duckdb(
    text: str,
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float] | None:
    """Generate a float32 embedding suitable for DuckDB storage.

    Args:
        text: Text to embed
        client: Optional OpenAI client (creates one if not provided)
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embedding (default: from config settings)

    Returns:
        Float32 embedding vector or None if failed
    """
    # Create client if not provided
    if client is None:
        if not settings.openai_api_key:
            return None
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    # Resolve config defaults
    resolved_model = model or settings.openai_embedding_model
    resolved_dimensions = dimensions or settings.openai_embedding_dimensions

    # Use oidm-common function (already returns float32)
    result: list[float] | None = await generate_embedding(
        text=text,
        client=client,
        model=resolved_model,
        dimensions=resolved_dimensions,
    )
    return result


async def batch_embeddings_for_duckdb(
    texts: Sequence[str],
    *,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[list[float] | None]:
    """Generate float32 embeddings for several texts in a single API call.

    Args:
        texts: Sequence of texts to embed
        client: Optional OpenAI client (creates one if not provided)
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embeddings (default: from config settings)

    Returns:
        List of float32 embedding vectors (or None for failed items)
    """
    if not texts:
        return []

    # Create client if not provided
    if client is None:
        if not settings.openai_api_key:
            return [None] * len(texts)
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    # Resolve config defaults
    resolved_model = model or settings.openai_embedding_model
    resolved_dimensions = dimensions or settings.openai_embedding_dimensions

    # Use oidm-common function (already returns float32)
    result: list[list[float] | None] = await generate_embeddings_batch(
        texts=list(texts),
        client=client,
        model=resolved_model,
        dimensions=resolved_dimensions,
    )
    return result


__all__ = [
    "batch_embeddings_for_duckdb",
    "get_embedding_for_duckdb",
]
