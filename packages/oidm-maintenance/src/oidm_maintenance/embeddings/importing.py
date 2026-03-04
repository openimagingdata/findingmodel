"""Import helpers for embedding caches."""

from __future__ import annotations

from pathlib import Path

from oidm_common.embeddings import EmbeddingCache


async def import_duckdb_into_current_cache(
    source_path: Path,
    *,
    assume_defaults: bool = True,
    default_model: str = "text-embedding-3-small",
    default_dimensions: int = 512,
) -> tuple[Path, dict[str, int]]:
    """Import a DuckDB embedding_cache table into current runtime cache."""
    cache = EmbeddingCache()
    stats = await cache.import_duckdb_file(
        source_path=source_path,
        assume_defaults=assume_defaults,
        default_model=default_model,
        default_dimensions=default_dimensions,
        strict=True,
    )
    return cache.cache_dir, stats


async def import_cache_into_current_cache(
    source_cache_dir: Path,
    *,
    upsert: bool = True,
) -> tuple[Path, dict[str, int]]:
    """Import another diskcache embedding cache into current runtime cache."""
    cache = EmbeddingCache()
    stats = await cache.import_cache_dir(source_cache_dir, upsert=upsert, strict=True)
    return cache.cache_dir, stats


__all__ = [
    "import_cache_into_current_cache",
    "import_duckdb_into_current_cache",
]
