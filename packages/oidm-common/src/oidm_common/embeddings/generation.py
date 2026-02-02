"""Embedding generation utilities for OpenAI models.

This module provides high-level functions for generating embeddings that handle
client management internally. Downstream packages should use `get_embedding` and
`get_embeddings_batch` which only require an API key, not a client instance.
"""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, cast

from loguru import logger

from oidm_common.embeddings.cache import EmbeddingCache

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_UNSET = object()
_default_cache: EmbeddingCache | None = None

# Module-level client cache for connection reuse
_client_cache: dict[str, "AsyncOpenAI"] = {}


def _get_default_cache() -> EmbeddingCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = EmbeddingCache()
    return _default_cache


def _resolve_cache(cache: EmbeddingCache | object | None) -> EmbeddingCache | None:
    if cache is _UNSET:
        return _get_default_cache()
    return cast(EmbeddingCache | None, cache)


async def _get_cached_embeddings(
    texts: list[str],
    cache: EmbeddingCache | None,
    model: str,
    dimensions: int,
) -> tuple[list[list[float] | None] | None, list[int], list[str]]:
    if cache is None:
        return None, list(range(len(texts))), list(texts)

    await cache.setup()
    cached_embeddings = await cache.get_embeddings_batch(texts, model, dimensions)
    missing_indices = [index for index, embedding in enumerate(cached_embeddings) if embedding is None]
    missing_texts = [texts[index] for index in missing_indices]
    return cached_embeddings, missing_indices, missing_texts


def _merge_embeddings(
    cached_embeddings: list[list[float] | None] | None,
    missing_indices: list[int],
    generated_embeddings: list[list[float] | None],
    total_count: int,
) -> list[list[float] | None]:
    if cached_embeddings is None:
        results: list[list[float] | None] = [None] * total_count
    else:
        results = cached_embeddings
    for index, embedding in zip(missing_indices, generated_embeddings, strict=True):
        results[index] = embedding
    return results


async def _store_embeddings_if_available(
    cache: EmbeddingCache | None,
    texts: list[str],
    model: str,
    dimensions: int,
    generated_embeddings: list[list[float] | None],
) -> None:
    if cache is None:
        return

    texts_to_store: list[str] = []
    embeddings_to_store: list[list[float]] = []
    for text, embedding in zip(texts, generated_embeddings, strict=True):
        if embedding is not None:
            texts_to_store.append(text)
            embeddings_to_store.append(embedding)

    if embeddings_to_store:
        await cache.store_embeddings_batch(texts_to_store, model, dimensions, embeddings_to_store)


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
    cache: EmbeddingCache | object | None = _UNSET,
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
        cache: EmbeddingCache instance, None to disable, or default singleton when omitted

    Returns:
        Float32 embedding vector, or None if unavailable
    """
    resolved_cache = _resolve_cache(cache)
    if resolved_cache is not None:
        await resolved_cache.setup()
        cached_embedding = await resolved_cache.get_embedding(text, model, dimensions)
        if cached_embedding is not None:
            return cached_embedding

    client = _get_or_create_client(api_key)
    if client is None:
        return None

    embedding = await generate_embedding(text, client, model, dimensions)
    if resolved_cache is not None and embedding is not None:
        await resolved_cache.store_embedding(text, model, dimensions, embedding)

    return embedding


async def get_embeddings_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
    cache: EmbeddingCache | object | None = _UNSET,
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
        cache: EmbeddingCache instance, None to disable, or default singleton when omitted

    Returns:
        List of float32 embedding vectors (None for each if unavailable)
    """
    if not texts:
        return []

    resolved_cache = _resolve_cache(cache)
    cached_embeddings, missing_indices, missing_texts = await _get_cached_embeddings(
        texts, resolved_cache, model, dimensions
    )

    if not missing_texts:
        return cached_embeddings or []

    client = _get_or_create_client(api_key)
    if client is None:
        if cached_embeddings is not None:
            return cached_embeddings
        return [None] * len(texts)

    generated_embeddings = await generate_embeddings_batch(missing_texts, client, model, dimensions)
    results = _merge_embeddings(cached_embeddings, missing_indices, generated_embeddings, len(texts))
    await _store_embeddings_if_available(resolved_cache, missing_texts, model, dimensions, generated_embeddings)
    return results


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
