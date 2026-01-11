"""Embedding generation utilities for OpenAI models."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def _to_float32(embedding: list[float]) -> list[float]:
    """Convert embedding to 32-bit floats for DuckDB compatibility.

    Args:
        embedding: Embedding vector with 64-bit floats

    Returns:
        Embedding vector with 32-bit floats
    """
    return list(array("f", embedding))


async def generate_embedding(
    text: str,
    client: AsyncOpenAI,
    model: str,
    dimensions: int,
) -> list[float] | None:
    """Generate a single embedding vector for text.

    Args:
        text: Text to embed
        client: AsyncOpenAI client
        model: Embedding model name (e.g., "text-embedding-3-small")
        dimensions: Vector dimensions

    Returns:
        Float32 embedding vector, or None on error
    """
    try:
        response = await client.embeddings.create(
            input=[text],
            model=model,
            dimensions=dimensions,
        )

        if not response.data:
            logger.error("Empty response from OpenAI embeddings API")
            return None

        # Convert to float32 precision for DuckDB
        return _to_float32(response.data[0].embedding)

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


async def generate_embeddings_batch(
    texts: list[str],
    client: AsyncOpenAI,
    model: str,
    dimensions: int,
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts in a single API call.

    Args:
        texts: List of texts to embed
        client: AsyncOpenAI client
        model: Embedding model name (e.g., "text-embedding-3-small")
        dimensions: Vector dimensions

    Returns:
        List of float32 embedding vectors (or None for failed items)
    """
    if not texts:
        return []

    try:
        response = await client.embeddings.create(
            input=texts,
            model=model,
            dimensions=dimensions,
        )

        # Convert to float32 precision and return in order
        results: list[list[float] | None] = []
        for embedding_obj in response.data:
            results.append(_to_float32(embedding_obj.embedding))

        return results

    except Exception as e:
        logger.error(f"Error generating embeddings batch: {e}")
        return [None] * len(texts)


__all__ = ["generate_embedding", "generate_embeddings_batch"]
