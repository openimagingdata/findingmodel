"""Tests for oidm_maintenance.embeddings helper modules."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from oidm_maintenance.embeddings import importing, migration


def test_migrate_default_cache_requires_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Migration wrapper should fail if cache cannot be prepared."""
    fake_cache = MagicMock()
    fake_cache.cache_dir = Path("/tmp/oidm-cache")
    fake_cache.require_cache_ready = AsyncMock()
    monkeypatch.setattr(migration, "EmbeddingCache", MagicMock(return_value=fake_cache))

    result = asyncio.run(migration.migrate_default_cache())

    assert result == Path("/tmp/oidm-cache")
    fake_cache.require_cache_ready.assert_awaited_once_with()


def test_get_default_cache_stats_enforces_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stats wrapper should call strict cache stats API."""
    fake_cache = MagicMock()
    fake_cache.cache_dir = Path("/tmp/oidm-cache")
    fake_cache.get_stats = AsyncMock(return_value={"total_keys": 0})
    monkeypatch.setattr(migration, "EmbeddingCache", MagicMock(return_value=fake_cache))

    cache_dir, stats = asyncio.run(migration.get_default_cache_stats())

    assert cache_dir == Path("/tmp/oidm-cache")
    assert stats["total_keys"] == 0
    fake_cache.get_stats.assert_awaited_once_with(strict=True)


def test_import_duckdb_into_current_cache_enforces_strict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DuckDB import wrapper should call strict cache import."""
    source = tmp_path / "source.duckdb"
    fake_cache = MagicMock()
    fake_cache.cache_dir = Path("/tmp/oidm-cache")
    fake_cache.import_duckdb_file = AsyncMock(
        return_value={"written": 1, "new": 1, "updated": 0, "skipped": 0, "total": 1}
    )
    monkeypatch.setattr(importing, "EmbeddingCache", MagicMock(return_value=fake_cache))

    cache_dir, stats = asyncio.run(importing.import_duckdb_into_current_cache(source))

    assert cache_dir == Path("/tmp/oidm-cache")
    assert stats["written"] == 1
    fake_cache.import_duckdb_file.assert_awaited_once_with(
        source_path=source,
        assume_defaults=True,
        default_model="text-embedding-3-small",
        default_dimensions=512,
        strict=True,
    )


def test_import_cache_into_current_cache_enforces_strict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Diskcache import wrapper should call strict cache import."""
    source = tmp_path / "source.cache"
    fake_cache = MagicMock()
    fake_cache.cache_dir = Path("/tmp/oidm-cache")
    fake_cache.import_cache_dir = AsyncMock(
        return_value={"written": 2, "new": 0, "updated": 2, "skipped": 1, "total": 3}
    )
    monkeypatch.setattr(importing, "EmbeddingCache", MagicMock(return_value=fake_cache))

    cache_dir, stats = asyncio.run(importing.import_cache_into_current_cache(source, upsert=False))

    assert cache_dir == Path("/tmp/oidm-cache")
    assert stats["updated"] == 2
    fake_cache.import_cache_dir.assert_awaited_once_with(source, upsert=False, strict=True)
