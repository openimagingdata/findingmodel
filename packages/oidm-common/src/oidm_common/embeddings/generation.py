"""Embedding generation utilities for OpenAI and local models.

This module provides high-level functions for generating embeddings that handle
provider client/runtime management internally. Downstream packages should use
`get_embedding` and `get_embeddings_batch`.
"""

from __future__ import annotations

import asyncio
import tempfile
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger
from platformdirs import user_cache_dir

from oidm_common.embeddings.cache import EmbeddingCache

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_UNSET = object()
_default_cache: EmbeddingCache | None = None

# Module-level client/runtime caches
_client_cache: dict[str, "AsyncOpenAI"] = {}
_fastembed_model_cache: dict[tuple[str, int | None], object] = {}
_local_semaphore_cache: dict[int, asyncio.Semaphore] = {}

_LOCAL_EMBEDDING_DEFAULT_MAX_CONCURRENCY = 2
_DEFAULT_FASTEMBED_CACHE_DIR = (
    Path(user_cache_dir(appname="oidm-common", appauthor="openimagingdata", ensure_exists=False)) / "fastembed"
)
_TEMP_FASTEMBED_CACHE_DIR = Path(tempfile.gettempdir()) / "fastembed_cache"

EmbeddingProvider = Literal["openai", "fastembed"]
EmbeddingMode = Literal["query", "document"]


def _get_default_cache() -> EmbeddingCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = EmbeddingCache()
    return _default_cache


def _resolve_cache(cache: EmbeddingCache | object | None) -> EmbeddingCache | None:
    if cache is _UNSET:
        return _get_default_cache()
    return cast(EmbeddingCache | None, cache)


def _normalise_provider(provider: str) -> EmbeddingProvider:
    value = provider.strip().lower()
    if value in {"openai", "fastembed"}:
        return cast(EmbeddingProvider, value)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _cache_model_key(provider: EmbeddingProvider, model: str) -> str:
    """Use legacy key shape for OpenAI and namespaced keys for other providers."""
    if provider == "openai":
        return model
    return f"{provider}:{model}"


async def _get_cached_embeddings(
    texts: list[str],
    cache: EmbeddingCache | None,
    cache_model: str,
    dimensions: int,
) -> tuple[list[list[float] | None] | None, list[int], list[str]]:
    if cache is None:
        return None, list(range(len(texts))), list(texts)

    await cache.setup()
    cached_embeddings = await cache.get_embeddings_batch(texts, cache_model, dimensions)
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
    cache_model: str,
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
        await cache.store_embeddings_batch(texts_to_store, cache_model, dimensions, embeddings_to_store)


def _to_float32(embedding: list[float]) -> list[float]:
    """Convert embedding to 32-bit floats for DuckDB compatibility."""
    return list(array("f", embedding))


def _to_float32_from_iterable(values: object) -> list[float]:
    """Convert iterable values (numpy, list, etc.) to float32 list."""
    try:
        raw_values = values.tolist()  # numpy arrays
    except AttributeError:
        raw_values = list(values)
    return _to_float32([float(value) for value in raw_values])


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


def _resolve_fastembed_cache_dir() -> Path:
    primary = _DEFAULT_FASTEMBED_CACHE_DIR
    fallback = _TEMP_FASTEMBED_CACHE_DIR

    try:
        primary.mkdir(parents=True, exist_ok=True)
        return primary
    except Exception as e:
        logger.warning(f"FastEmbed cache dir unavailable at {primary}: {e}; using {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _get_or_create_fastembed_model(model: str, threads: int | None) -> object | None:
    """Get or create a FastEmbed TextEmbedding runtime for local embeddings."""
    key = (model, threads)
    if key in _fastembed_model_cache:
        return _fastembed_model_cache[key]

    try:
        from fastembed import TextEmbedding
    except ImportError:
        logger.debug("fastembed not installed - local semantic search disabled")
        return None

    try:
        cache_dir = str(_resolve_fastembed_cache_dir())
        runtime = TextEmbedding(model_name=model, threads=threads, cache_dir=cache_dir)
    except Exception as e:
        logger.warning(f"Failed to initialize fastembed model '{model}': {e}")
        return None

    _fastembed_model_cache[key] = runtime
    return runtime


def _local_semaphore(limit: int) -> asyncio.Semaphore:
    resolved_limit = max(1, limit)
    existing = _local_semaphore_cache.get(resolved_limit)
    if existing is not None:
        return existing
    semaphore = asyncio.Semaphore(resolved_limit)
    _local_semaphore_cache[resolved_limit] = semaphore
    return semaphore


def _fastembed_generate_batch_sync(
    texts: list[str],
    *,
    model: str,
    dimensions: int,
    mode: EmbeddingMode,
    threads: int | None,
    parallel: int | None,
) -> list[list[float] | None]:
    runtime = _get_or_create_fastembed_model(model, threads)
    if runtime is None:
        return [None] * len(texts)

    try:
        if mode == "query":
            vectors = list(runtime.query_embed(texts, parallel=parallel))
        elif hasattr(runtime, "passage_embed"):
            vectors = list(runtime.passage_embed(texts, parallel=parallel))
        else:
            vectors = list(runtime.embed(texts, parallel=parallel))
    except Exception as e:
        logger.warning(f"fastembed embedding generation failed: {e}")
        return [None] * len(texts)

    output: list[list[float] | None] = [_to_float32_from_iterable(vector) for vector in vectors]
    if len(output) != len(texts):
        logger.warning(f"fastembed returned {len(output)} vectors for {len(texts)} texts; filling missing values")
        if len(output) < len(texts):
            output.extend([None] * (len(texts) - len(output)))
        else:
            output = output[: len(texts)]

    for i, vector in enumerate(output):
        if vector is None:
            continue
        if len(vector) != dimensions:
            logger.warning(
                f"fastembed dimension mismatch for model '{model}': expected {dimensions}, got {len(vector)}"
            )
            output[i] = None

    return output


async def _generate_local_embeddings_batch(
    texts: list[str],
    *,
    model: str,
    dimensions: int,
    mode: EmbeddingMode,
    threads: int | None,
    parallel: int | None,
    max_concurrency: int,
) -> list[list[float] | None]:
    semaphore = _local_semaphore(max_concurrency)
    async with semaphore:
        return await asyncio.to_thread(
            _fastembed_generate_batch_sync,
            texts,
            model=model,
            dimensions=dimensions,
            mode=mode,
            threads=threads,
            parallel=parallel,
        )


async def get_embedding(
    text: str,
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
    provider: str = "openai",
    embed_mode: EmbeddingMode = "query",
    local_threads: int | None = None,
    local_parallel: int | None = None,
    local_max_concurrency: int = _LOCAL_EMBEDDING_DEFAULT_MAX_CONCURRENCY,
    cache: EmbeddingCache | object | None = _UNSET,
) -> list[float] | None:
    """Generate a single embedding vector for text.

    Args:
        text: Text to embed
        api_key: OpenAI API key (required only when provider='openai')
        model: Embedding model name
        dimensions: Target vector dimensions
        provider: Embedding provider ('openai' or 'fastembed')
        embed_mode: Embedding mode ('query' or 'document')
        local_threads: Optional thread count for local provider runtime
        local_parallel: Optional local provider parallel parameter
        local_max_concurrency: Maximum concurrent in-process local embedding jobs
        cache: EmbeddingCache instance, None to disable, or default singleton when omitted

    Returns:
        Float32 embedding vector, or None if unavailable
    """
    provider_name = _normalise_provider(provider)
    cache_model = _cache_model_key(provider_name, model)

    resolved_cache = _resolve_cache(cache)
    if resolved_cache is not None:
        await resolved_cache.setup()
        cached_embedding = await resolved_cache.get_embedding(text, cache_model, dimensions)
        if cached_embedding is not None:
            return cached_embedding

    if provider_name == "openai":
        client = _get_or_create_client(api_key)
        if client is None:
            return None
        embedding = await generate_embedding(text, client, model, dimensions)
    else:
        generated = await _generate_local_embeddings_batch(
            [text],
            model=model,
            dimensions=dimensions,
            mode=embed_mode,
            threads=local_threads,
            parallel=local_parallel,
            max_concurrency=local_max_concurrency,
        )
        embedding = generated[0] if generated else None

    if resolved_cache is not None and embedding is not None:
        await resolved_cache.store_embedding(text, cache_model, dimensions, embedding)

    return embedding


async def get_embeddings_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
    provider: str = "openai",
    embed_mode: EmbeddingMode = "query",
    local_threads: int | None = None,
    local_parallel: int | None = None,
    local_max_concurrency: int = _LOCAL_EMBEDDING_DEFAULT_MAX_CONCURRENCY,
    cache: EmbeddingCache | object | None = _UNSET,
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        api_key: OpenAI API key (required only when provider='openai')
        model: Embedding model name
        dimensions: Target vector dimensions
        provider: Embedding provider ('openai' or 'fastembed')
        embed_mode: Embedding mode ('query' or 'document')
        local_threads: Optional thread count for local provider runtime
        local_parallel: Optional local provider parallel parameter
        local_max_concurrency: Maximum concurrent in-process local embedding jobs
        cache: EmbeddingCache instance, None to disable, or default singleton when omitted

    Returns:
        List of float32 embedding vectors (None for failed/unavailable items)
    """
    if not texts:
        return []

    provider_name = _normalise_provider(provider)
    cache_model = _cache_model_key(provider_name, model)

    resolved_cache = _resolve_cache(cache)
    cached_embeddings, missing_indices, missing_texts = await _get_cached_embeddings(
        texts,
        resolved_cache,
        cache_model,
        dimensions,
    )

    if not missing_texts:
        return cached_embeddings or []

    generated_embeddings: list[list[float] | None]
    if provider_name == "openai":
        client = _get_or_create_client(api_key)
        if client is None:
            if cached_embeddings is not None:
                return cached_embeddings
            return [None] * len(texts)
        generated_embeddings = await generate_embeddings_batch(missing_texts, client, model, dimensions)
    else:
        generated_embeddings = await _generate_local_embeddings_batch(
            missing_texts,
            model=model,
            dimensions=dimensions,
            mode=embed_mode,
            threads=local_threads,
            parallel=local_parallel,
            max_concurrency=local_max_concurrency,
        )

    results = _merge_embeddings(cached_embeddings, missing_indices, generated_embeddings, len(texts))
    await _store_embeddings_if_available(resolved_cache, missing_texts, cache_model, dimensions, generated_embeddings)
    return results


# Low-level functions that require an OpenAI client (for advanced use cases)


async def generate_embedding(
    text: str,
    client: "AsyncOpenAI",
    model: str,
    dimensions: int,
) -> list[float] | None:
    """Generate a single embedding vector for text (low-level OpenAI API)."""
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
    *,
    chunk_size: int = 2048,
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts via the OpenAI API (low-level API)."""
    if not texts:
        return []

    results: list[list[float] | None] = []

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        try:
            response = await client.embeddings.create(
                input=chunk,
                model=model,
                dimensions=dimensions,
            )

            # Convert to float32 precision and return in order
            for embedding_obj in response.data:
                results.append(_to_float32(embedding_obj.embedding))

        except Exception as e:
            logger.error(f"Error generating embeddings batch (chunk {start}-{start + len(chunk)}): {e}")
            results.extend([None] * len(chunk))

    return results


def create_openai_client(api_key: str) -> "AsyncOpenAI":
    """Create an AsyncOpenAI client for embedding generation."""
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
