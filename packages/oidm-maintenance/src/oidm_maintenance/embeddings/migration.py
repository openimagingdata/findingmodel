"""Migration helpers for embedding caches."""

from __future__ import annotations

from pathlib import Path

from oidm_common.embeddings import EmbeddingCache


async def migrate_default_cache() -> Path:
    """Run default embedding cache migrations.

    Returns:
        Runtime cache directory path.
    """
    cache = EmbeddingCache()
    await cache.require_cache_ready()
    return cache.cache_dir


async def get_default_cache_stats() -> tuple[Path, dict[str, object]]:
    """Read stats for the default runtime embedding cache."""
    cache = EmbeddingCache()
    stats = await cache.get_stats(strict=True)
    return cache.cache_dir, stats


__all__ = ["get_default_cache_stats", "migrate_default_cache"]
