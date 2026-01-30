"""Embedding generation utilities for OpenAI models.

This module provides high-level functions for generating embeddings that handle
client management internally. Downstream packages should use `get_embedding` and
`get_embeddings_batch` which only require an API key, not a client instance.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Module-level client cache for connection reuse
_client_cache: dict[str, "AsyncOpenAI"] = {}


def _to_float32(embedding: list[float]) -> list[float]:
    """Convert embedding to 32-bit floats for DuckDB compatibility.

    Args:
        embedding: Embedding vector with 64-bit floats

    Returns:
        Embedding vector with 32-bit floats
    """
    return list(array("f", embedding))


def _get_or_create_client(api_key: str) -> "AsyncOpenAI | None":
    """Get or create an AsyncOpenAI client, caching by API key.

    Returns None if openai is not installed (graceful degradation).
    """
    if not api_key:
        return None

    if api_key in _client_cache:
        return _client_cache[api_key]

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.debug("openai not installed - semantic search disabled")
        return None

    client = AsyncOpenAI(api_key=api_key)
    _client_cache[api_key] = client
    return client


async def get_embedding(
    text: str,
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
) -> list[float] | None:
    """Generate a single embedding vector for text.

    This is the primary high-level API for embedding generation.
    Handles client creation internally and gracefully returns None
    if openai is not available or API key is missing.

    Args:
        text: Text to embed
        api_key: OpenAI API key
        model: Embedding model name (default: "text-embedding-3-small")
        dimensions: Vector dimensions (default: 512)

    Returns:
        Float32 embedding vector, or None if unavailable
    """
    client = _get_or_create_client(api_key)
    if client is None:
        return None

    return await generate_embedding(text, client, model, dimensions)


async def get_embeddings_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts in a single API call.

    This is the primary high-level API for batch embedding generation.
    Handles client creation internally and gracefully returns None values
    if openai is not available or API key is missing.

    Args:
        texts: List of texts to embed
        api_key: OpenAI API key
        model: Embedding model name (default: "text-embedding-3-small")
        dimensions: Vector dimensions (default: 512)

    Returns:
        List of float32 embedding vectors (None for each if unavailable)
    """
    if not texts:
        return []

    client = _get_or_create_client(api_key)
    if client is None:
        return [None] * len(texts)

    return await generate_embeddings_batch(texts, client, model, dimensions)


# Low-level functions that require a client (for advanced use cases)


async def generate_embedding(
    text: str,
    client: "AsyncOpenAI",
    model: str,
    dimensions: int,
) -> list[float] | None:
    """Generate a single embedding vector for text (low-level API).

    For most use cases, prefer `get_embedding` which handles client management.

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
    client: "AsyncOpenAI",
    model: str,
    dimensions: int,
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts in a single API call (low-level API).

    For most use cases, prefer `get_embeddings_batch` which handles client management.

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


def create_openai_client(api_key: str) -> "AsyncOpenAI":
    """Create an AsyncOpenAI client for embedding generation.

    This factory is provided for advanced use cases that need direct client access.
    For most use cases, prefer `get_embedding` or `get_embeddings_batch`.

    Args:
        api_key: OpenAI API key

    Returns:
        AsyncOpenAI client instance

    Raises:
        ImportError: If openai package is not installed
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            "openai package required for embeddings. Install with: pip install oidm-common[openai]"
        ) from e

    return AsyncOpenAI(api_key=api_key)


__all__ = [
    "create_openai_client",
    "generate_embedding",
    "generate_embeddings_batch",
    "get_embedding",
    "get_embeddings_batch",
]
