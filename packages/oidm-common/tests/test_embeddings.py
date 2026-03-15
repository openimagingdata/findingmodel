"""Tests for oidm-common EmbeddingCache."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import duckdb
import pytest
from oidm_common.embeddings.cache import EmbeddingCache

if TYPE_CHECKING:
    from .conftest import TestSettings


class TestEmbeddingCacheContextManager:
    """Tests for EmbeddingCache async context manager."""

    async def test_embedding_cache_context_manager_setup(self, tmp_path: Path) -> None:
        """Test that context manager sets up the cache directory."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            assert cache.db_path == cache_path
            assert cache.cache_dir.exists()

    async def test_embedding_cache_context_manager_cleanup(self, tmp_path: Path) -> None:
        """Test that context manager closes cache on exit."""
        cache_path = tmp_path / "test_cache.duckdb"

        cache = EmbeddingCache(db_path=cache_path)
        async with cache:
            # Connection should be available during context
            pass

        # After context exit, cache should be closed
        assert cache._cache is None

    async def test_embedding_cache_context_manager_cleanup_on_exception(self, tmp_path: Path) -> None:
        """Test that context manager cleans up even on exception."""
        cache_path = tmp_path / "exception_test.duckdb"

        try:
            async with EmbeddingCache(db_path=cache_path) as cache:
                await cache.store_embedding("test", "model", 512, [0.1] * 512)
                raise ValueError("Test exception")
        except ValueError:
            pass

        cache = EmbeddingCache(db_path=cache_path)
        await cache.setup()
        result = await cache.get_embedding(text="test", model="model", dimensions=512)
        assert result is not None

    async def test_embedding_cache_default_path(self) -> None:
        """Test that EmbeddingCache uses default path when not specified."""
        cache = EmbeddingCache()

        # Should use findingmodel legacy migration source and oidm-common runtime cache dir
        assert cache.db_path.name == "embeddings.duckdb"
        assert cache.cache_dir.name == "embeddings.cache"
        assert "findingmodel" in str(cache.db_path)
        assert "oidm-common" in str(cache.cache_dir)

    async def test_embedding_cache_uses_platform_primary_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default cache location should use the platform primary cache directory."""
        from oidm_common.embeddings import cache as cache_module

        custom_cache_dir = tmp_path / "platform-default.cache"
        monkeypatch.setattr(cache_module, "_DEFAULT_DISKCACHE_DIR", custom_cache_dir)
        monkeypatch.setattr(cache_module, "_TEMP_DISKCACHE_DIR", tmp_path / "temp-fallback.cache")

        cache = EmbeddingCache()
        await cache.setup()

        assert cache.cache_dir == custom_cache_dir
        assert cache._cache is not None

    async def test_embedding_cache_falls_back_when_primary_is_invalid(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If primary cache path is unusable, setup should fall back to next candidate."""
        from oidm_common.embeddings import cache as cache_module

        invalid_primary = tmp_path / "not-a-directory"
        invalid_primary.write_text("x")
        fallback_dir = tmp_path / "fallback.cache"

        monkeypatch.setattr(cache_module, "_DEFAULT_DISKCACHE_DIR", invalid_primary)
        monkeypatch.setattr(cache_module, "_TEMP_DISKCACHE_DIR", fallback_dir)

        cache = EmbeddingCache()
        await cache.setup()

        assert cache.cache_dir == fallback_dir
        assert cache._cache is not None


class TestEmbeddingCacheSchema:
    """Tests for EmbeddingCache setup."""

    async def test_setup_creates_diskcache_directory(self, tmp_path: Path) -> None:
        """Test that setup() creates diskcache directory correctly."""
        cache_path = tmp_path / "setup_test.duckdb"
        cache = EmbeddingCache(cache_path)
        await cache.setup()

        assert cache.cache_dir.exists()
        assert cache._cache is not None


class TestEmbeddingCacheMigration:
    """Tests for one-time migration from legacy DuckDB cache."""

    async def test_setup_migrates_legacy_duckdb_with_default_model_dimensions(self, tmp_path: Path) -> None:
        """Test migration from legacy duckdb cache into diskcache."""
        cache_path = tmp_path / "legacy_cache.duckdb"
        text = "legacy text"
        text_hash = EmbeddingCache(db_path=cache_path)._hash_text(text)
        embedding = [0.1] * 512

        conn = duckdb.connect(str(cache_path), read_only=False)
        conn.execute("""
            CREATE TABLE embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding FLOAT[] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            """
            INSERT INTO embedding_cache (text_hash, model, dimensions, embedding, created_at)
            VALUES (?, 'old-model', 1536, ?, CURRENT_TIMESTAMP)
            """,
            (text_hash, embedding),
        )
        conn.close()

        cache = EmbeddingCache(db_path=cache_path)
        await cache.setup()
        result = await cache.get_embedding(text, "text-embedding-3-small", 512)
        assert result is not None
        assert len(result) == 512
        assert result[0] == pytest.approx(0.1, abs=1e-6)
        assert await cache.get_embedding(text, "old-model", 1536) is None

    async def test_setup_migration_is_one_time(self, tmp_path: Path) -> None:
        """Test migration marker prevents duplicate migrations."""
        cache_path = tmp_path / "legacy_once.duckdb"

        conn = duckdb.connect(str(cache_path), read_only=False)
        conn.execute("""
            CREATE TABLE embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding FLOAT[] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        first_text = "first legacy text"
        first_hash = EmbeddingCache(db_path=cache_path)._hash_text(first_text)
        conn.execute(
            """
            INSERT INTO embedding_cache (text_hash, model, dimensions, embedding, created_at)
            VALUES (?, 'ignored', 42, ?, CURRENT_TIMESTAMP)
            """,
            (first_hash, [0.2] * 512),
        )
        conn.close()

        first_cache = EmbeddingCache(db_path=cache_path)
        await first_cache.setup()

        conn = duckdb.connect(str(cache_path), read_only=False)
        second_text = "second legacy text"
        second_hash = EmbeddingCache(db_path=cache_path)._hash_text(second_text)
        conn.execute(
            """
            INSERT INTO embedding_cache (text_hash, model, dimensions, embedding, created_at)
            VALUES (?, 'ignored', 42, ?, CURRENT_TIMESTAMP)
            """,
            (second_hash, [0.3] * 512),
        )
        conn.close()

        second_cache = EmbeddingCache(db_path=cache_path)
        await second_cache.setup()

        assert await second_cache.get_embedding(first_text, "text-embedding-3-small", 512) is not None
        assert await second_cache.get_embedding(second_text, "text-embedding-3-small", 512) is None

    async def test_setup_migrates_legacy_findingmodel_diskcache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test migration from legacy findingmodel diskcache directory."""
        from oidm_common.embeddings import cache as cache_module

        legacy_root = tmp_path / "legacy_findingmodel"
        legacy_dir = legacy_root / "embeddings.cache"
        runtime_root = tmp_path / "runtime_oidm_common"
        runtime_dir = runtime_root / "embeddings.cache"
        legacy_duckdb = legacy_root / "embeddings.duckdb"

        monkeypatch.setattr(cache_module, "_LEGACY_FINDINGMODEL_CACHE_ROOT", legacy_root)
        monkeypatch.setattr(cache_module, "_LEGACY_DISKCACHE_DIR", legacy_dir)
        monkeypatch.setattr(cache_module, "_DEFAULT_RUNTIME_CACHE_ROOT", runtime_root)
        monkeypatch.setattr(cache_module, "_DEFAULT_DISKCACHE_DIR", runtime_dir)
        monkeypatch.setattr(cache_module, "_DEFAULT_CACHE_PATH", legacy_duckdb)

        from diskcache import Cache

        text = "legacy diskcache text"
        text_hash = EmbeddingCache()._hash_text(text)
        key = EmbeddingCache()._build_key(text_hash, "text-embedding-3-small", 512)
        source_cache = Cache(directory=str(legacy_dir), sqlite_journal_mode="wal")
        source_cache.set(key, {"embedding": [0.25] * 512, "created_at": time.time() - 123})
        source_cache.close()

        cache = EmbeddingCache()
        await cache.setup()
        result = await cache.get_embedding(text, "text-embedding-3-small", 512)
        assert result is not None
        assert result[0] == pytest.approx(0.25, abs=1e-6)

    async def test_setup_does_not_migrate_legacy_diskcache_into_non_default_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that custom cache dirs do not auto-import global legacy diskcache."""
        from oidm_common.embeddings import cache as cache_module

        legacy_root = tmp_path / "legacy_findingmodel"
        legacy_dir = legacy_root / "embeddings.cache"
        runtime_root = tmp_path / "runtime_oidm_common"
        runtime_dir = runtime_root / "embeddings.cache"
        legacy_duckdb = legacy_root / "embeddings.duckdb"

        monkeypatch.setattr(cache_module, "_LEGACY_FINDINGMODEL_CACHE_ROOT", legacy_root)
        monkeypatch.setattr(cache_module, "_LEGACY_DISKCACHE_DIR", legacy_dir)
        monkeypatch.setattr(cache_module, "_DEFAULT_RUNTIME_CACHE_ROOT", runtime_root)
        monkeypatch.setattr(cache_module, "_DEFAULT_DISKCACHE_DIR", runtime_dir)
        monkeypatch.setattr(cache_module, "_DEFAULT_CACHE_PATH", legacy_duckdb)

        from diskcache import Cache

        text = "legacy diskcache text"
        text_hash = EmbeddingCache()._hash_text(text)
        key = EmbeddingCache()._build_key(text_hash, "text-embedding-3-small", 512)
        source_cache = Cache(directory=str(legacy_dir), sqlite_journal_mode="wal")
        source_cache.set(key, {"embedding": [0.25] * 512, "created_at": time.time() - 123})
        source_cache.close()

        cache = EmbeddingCache(db_path=tmp_path / "custom.duckdb")
        await cache.setup()

        result = await cache.get_embedding(text, "text-embedding-3-small", 512)
        assert result is None

    async def test_import_duckdb_file_assume_defaults_upserts(self, tmp_path: Path) -> None:
        """Test explicit duckdb import with defaults and upsert behavior."""
        source_path = tmp_path / "source.duckdb"
        text_hash = EmbeddingCache()._hash_text("import me")

        conn = duckdb.connect(str(source_path), read_only=False)
        conn.execute("""
            CREATE TABLE embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding FLOAT[] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            """
            INSERT INTO embedding_cache (text_hash, model, dimensions, embedding, created_at)
            VALUES (?, 'legacy-model', 1536, ?, CURRENT_TIMESTAMP)
            """,
            (text_hash, [0.1] * 512),
        )
        conn.close()

        cache = EmbeddingCache(db_path=tmp_path / "cache.duckdb")
        await cache.setup()

        await cache.store_embedding("import me", "text-embedding-3-small", 512, [0.9] * 512)
        before = await cache.get_embedding("import me", "text-embedding-3-small", 512)
        assert before is not None
        assert before[0] == pytest.approx(0.9, abs=1e-6)

        stats = await cache.import_duckdb_file(source_path, assume_defaults=True)

        assert stats["total"] == 1
        assert stats["written"] == 1
        assert stats["new"] == 0
        assert stats["updated"] == 1
        assert stats["skipped"] == 0
        after = await cache.get_embedding("import me", "text-embedding-3-small", 512)
        assert after is not None
        assert after[0] == pytest.approx(0.1, abs=1e-6)

    async def test_import_duckdb_file_preserve_metadata(self, tmp_path: Path) -> None:
        """Test explicit duckdb import preserving model/dimensions columns."""
        source_path = tmp_path / "source_metadata.duckdb"
        text_hash = EmbeddingCache()._hash_text("metadata row")

        conn = duckdb.connect(str(source_path), read_only=False)
        conn.execute("""
            CREATE TABLE embedding_cache (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding FLOAT[] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute(
            """
            INSERT INTO embedding_cache (text_hash, model, dimensions, embedding, created_at)
            VALUES (?, 'custom-model', 3, ?, CURRENT_TIMESTAMP)
            """,
            (text_hash, [0.2, 0.3, 0.4]),
        )
        conn.close()

        cache = EmbeddingCache(db_path=tmp_path / "cache.duckdb")
        stats = await cache.import_duckdb_file(source_path, assume_defaults=False)
        assert stats["written"] == 1
        assert stats["new"] == 1
        assert stats["updated"] == 0
        result = await cache.get_embedding("metadata row", "custom-model", 3)
        assert result is not None
        assert result[0] == pytest.approx(0.2, abs=1e-6)

    async def test_import_cache_dir_upserts_existing_keys(self, tmp_path: Path) -> None:
        """Test importing from another diskcache directory with upsert behavior."""
        from diskcache import Cache

        text = "shared text"
        model = "text-embedding-3-small"
        dimensions = 3
        cache = EmbeddingCache(db_path=tmp_path / "target.duckdb")

        await cache.store_embedding(text, model, dimensions, [0.9, 0.9, 0.9])

        source_dir = tmp_path / "source.cache"
        source_cache = Cache(directory=str(source_dir), sqlite_journal_mode="wal")
        text_hash = cache._hash_text(text)
        key = cache._build_key(text_hash, model, dimensions)
        source_cache.set(key, {"embedding": [0.1, 0.2, 0.3], "created_at": time.time() - 10})
        source_cache.close()

        stats = await cache.import_cache_dir(source_dir, upsert=True)

        assert stats == {"imported": 1, "written": 1, "new": 0, "updated": 1, "skipped": 0, "total": 1}
        result = await cache.get_embedding(text, model, dimensions)
        assert result is not None
        assert result[0] == pytest.approx(0.1, abs=1e-6)

    async def test_import_duckdb_file_strict_raises_when_cache_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict import should fail if cache setup cannot provide a writable cache."""

        async def _broken_setup(self: EmbeddingCache) -> None:
            await asyncio.sleep(0)
            self._cache = None
            self._setup_complete = False

        monkeypatch.setattr(EmbeddingCache, "setup", _broken_setup)
        cache = EmbeddingCache(db_path=tmp_path / "cache.duckdb")

        with pytest.raises(RuntimeError, match="Embedding cache unavailable"):
            await cache.import_duckdb_file(tmp_path / "source.duckdb", strict=True)

    async def test_import_cache_dir_strict_raises_when_cache_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict cache-to-cache import should fail if cache setup is unavailable."""

        async def _broken_setup(self: EmbeddingCache) -> None:
            await asyncio.sleep(0)
            self._cache = None
            self._setup_complete = False

        monkeypatch.setattr(EmbeddingCache, "setup", _broken_setup)
        cache = EmbeddingCache(db_path=tmp_path / "cache.duckdb")

        with pytest.raises(RuntimeError, match="Embedding cache unavailable"):
            await cache.import_cache_dir(tmp_path / "source.cache", strict=True)

    async def test_get_stats_reports_model_distribution(self, tmp_path: Path) -> None:
        """Stats should report total keys, embedding keys, migration keys, and per-model counts."""
        cache = EmbeddingCache(db_path=tmp_path / "stats.duckdb")
        await cache.store_embedding("a", "model-a", 3, [0.1, 0.2, 0.3])
        await cache.store_embedding("b", "model-a", 3, [0.4, 0.5, 0.6])
        await cache.store_embedding("c", "model-b", 3, [0.7, 0.8, 0.9])

        stats = await cache.get_stats(strict=True)

        assert stats["cache_dir"] == str(cache.cache_dir)
        assert stats["embedding_keys"] == 3
        assert stats["migration_keys"] == 1
        assert stats["total_keys"] == 4
        assert stats["models"] == {"model-a": 2, "model-b": 1}


class TestEmbeddingCacheStoreAndGet:
    """Tests for store_embedding and get_embedding."""

    async def test_store_and_get_embedding_round_trip(self, tmp_path: Path) -> None:
        """Test storing and retrieving an embedding."""
        cache_path = tmp_path / "test_cache.duckdb"
        embedding = [0.1, 0.2, 0.3]

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Store embedding
            await cache.store_embedding(
                text="test text", model="text-embedding-3-small", dimensions=3, embedding=embedding
            )

            # Retrieve embedding
            result = await cache.get_embedding(text="test text", model="text-embedding-3-small", dimensions=3)

            assert result is not None
            assert len(result) == 3
            # Compare with approximate equality due to float32 conversion
            for i in range(3):
                assert result[i] == pytest.approx(embedding[i], abs=1e-6)

    async def test_get_embedding_cache_miss(self, tmp_path: Path) -> None:
        """Test that get_embedding returns None for cache miss."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            result = await cache.get_embedding(text="nonexistent", model="text-embedding-3-small", dimensions=3)

            assert result is None

    async def test_get_embedding_different_model_miss(self, tmp_path: Path) -> None:
        """Test that different model results in cache miss."""
        cache_path = tmp_path / "test_cache.duckdb"
        embedding = [0.1, 0.2, 0.3]

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding(text="test text", model="model-a", dimensions=3, embedding=embedding)

            result = await cache.get_embedding(text="test text", model="model-b", dimensions=3)

            assert result is None

    async def test_get_embedding_different_dimensions_miss(self, tmp_path: Path) -> None:
        """Test that different dimensions result in cache miss."""
        cache_path = tmp_path / "test_cache.duckdb"
        embedding = [0.1, 0.2, 0.3]

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding(
                text="test text", model="text-embedding-3-small", dimensions=3, embedding=embedding
            )

            result = await cache.get_embedding(text="test text", model="text-embedding-3-small", dimensions=512)

            assert result is None

    async def test_store_embedding_dimension_mismatch(self, tmp_path: Path) -> None:
        """Test that storing embedding with mismatched dimensions is gracefully handled."""
        cache_path = tmp_path / "test_cache.duckdb"
        embedding = [0.1, 0.2, 0.3]

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Store with wrong dimensions (should log warning but not raise)
            await cache.store_embedding(
                text="test text", model="text-embedding-3-small", dimensions=5, embedding=embedding
            )

            # Should not be retrievable
            result = await cache.get_embedding(text="test text", model="text-embedding-3-small", dimensions=3)
            assert result is None

    async def test_store_embedding_replace_existing(self, tmp_path: Path) -> None:
        """Test that storing same text/model/dimensions replaces existing embedding."""
        cache_path = tmp_path / "test_cache.duckdb"
        embedding_v1 = [0.1, 0.2, 0.3]
        embedding_v2 = [0.4, 0.5, 0.6]

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Store first version
            await cache.store_embedding(
                text="test text", model="text-embedding-3-small", dimensions=3, embedding=embedding_v1
            )

            # Store second version (should replace)
            await cache.store_embedding(
                text="test text", model="text-embedding-3-small", dimensions=3, embedding=embedding_v2
            )

            # Should retrieve the second version
            result = await cache.get_embedding(text="test text", model="text-embedding-3-small", dimensions=3)

            assert result is not None
            assert result[0] == pytest.approx(0.4, abs=1e-6)
            assert result[1] == pytest.approx(0.5, abs=1e-6)
            assert result[2] == pytest.approx(0.6, abs=1e-6)

    async def test_hash_collision_different_text(self, tmp_path: Path) -> None:
        """Test that different texts are stored separately."""
        cache_path = tmp_path / "test_cache.duckdb"
        embedding_a = [0.1, 0.2, 0.3]
        embedding_b = [0.4, 0.5, 0.6]

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding(text="text A", model="model", dimensions=3, embedding=embedding_a)
            await cache.store_embedding(text="text B", model="model", dimensions=3, embedding=embedding_b)

            result_a = await cache.get_embedding(text="text A", model="model", dimensions=3)
            result_b = await cache.get_embedding(text="text B", model="model", dimensions=3)

            assert result_a is not None
            assert result_b is not None
            assert result_a[0] == pytest.approx(0.1, abs=1e-6)
            assert result_b[0] == pytest.approx(0.4, abs=1e-6)

    async def test_embeddings_stored_with_float32_precision(self, tmp_path: Path) -> None:
        """Test that embeddings are stored with float32 precision."""
        cache_path = tmp_path / "test_cache.duckdb"
        text = "precision test"
        model = "text-embedding-3-small"
        dimensions = 512

        embedding = [1.123456789012345] * dimensions

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding(text, model, dimensions, embedding)
            retrieved = await cache.get_embedding(text, model, dimensions)

            assert retrieved is not None
            for value in retrieved:
                assert abs(value - 1.123456789012345) < 1e-6
                assert abs(value - 1.1234568) < 1e-7

    async def test_same_text_same_model_returns_cached(self, tmp_path: Path) -> None:
        """Test that same text with same model/dimensions returns cached result."""
        cache_path = tmp_path / "test_cache.duckdb"
        text = "consistent text"
        model = "text-embedding-3-small"
        dimensions = 512
        embedding = [0.5] * dimensions

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding(text, model, dimensions, embedding)

            result1 = await cache.get_embedding(text, model, dimensions)
            result2 = await cache.get_embedding(text, model, dimensions)

            assert result1 is not None
            assert result2 is not None
            assert result1 == result2

    async def test_cache_key_includes_model(self, tmp_path: Path) -> None:
        """Test same text with different models are stored independently."""
        cache_path = tmp_path / "test_cache.duckdb"
        text = "model variation test"
        dimensions = 3
        embedding_a = [0.1, 0.2, 0.3]
        embedding_b = [0.4, 0.5, 0.6]

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding(text, "model-a", dimensions, embedding_a)
            await cache.store_embedding(text, "model-b", dimensions, embedding_b)
            result_a = await cache.get_embedding(text, "model-a", dimensions)
            result_b = await cache.get_embedding(text, "model-b", dimensions)

            assert result_a is not None
            assert result_b is not None
            assert result_a[0] == pytest.approx(0.1, abs=1e-6)
            assert result_b[0] == pytest.approx(0.4, abs=1e-6)

    async def test_cache_key_includes_dimensions(self, tmp_path: Path) -> None:
        """Test same text/model with different dimensions are stored independently."""
        cache_path = tmp_path / "test_cache.duckdb"
        text = "dimension variation test"
        model = "text-embedding-3-small"
        embedding_3 = [0.1, 0.2, 0.3]
        embedding_2 = [0.9, 0.8]

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding(text, model, 3, embedding_3)
            await cache.store_embedding(text, model, 2, embedding_2)
            result_3 = await cache.get_embedding(text, model, 3)
            result_2 = await cache.get_embedding(text, model, 2)

            assert result_3 is not None
            assert result_2 is not None
            assert len(result_3) == 3
            assert len(result_2) == 2
            assert result_3[0] == pytest.approx(0.1, abs=1e-6)
            assert result_2[0] == pytest.approx(0.9, abs=1e-6)


class TestEmbeddingCacheBatchOperations:
    """Tests for batch embedding operations."""

    async def test_get_embeddings_batch_all_hits(self, tmp_path: Path) -> None:
        """Test batch get with all cache hits."""
        cache_path = tmp_path / "test_cache.duckdb"
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Store all embeddings
            for text, embedding in zip(texts, embeddings, strict=True):
                await cache.store_embedding(text=text, model="model", dimensions=2, embedding=embedding)

            # Batch retrieve
            results = await cache.get_embeddings_batch(texts=texts, model="model", dimensions=2)

            assert len(results) == 3
            for i, result in enumerate(results):
                assert result is not None
                assert result[0] == pytest.approx(embeddings[i][0], abs=1e-6)
                assert result[1] == pytest.approx(embeddings[i][1], abs=1e-6)

    async def test_get_embeddings_batch_partial_hits(self, tmp_path: Path) -> None:
        """Test batch get with partial cache hits."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Store only first and third
            await cache.store_embedding(text="text1", model="model", dimensions=2, embedding=[0.1, 0.2])
            await cache.store_embedding(text="text3", model="model", dimensions=2, embedding=[0.5, 0.6])

            # Batch retrieve all three
            results = await cache.get_embeddings_batch(texts=["text1", "text2", "text3"], model="model", dimensions=2)

            assert len(results) == 3
            assert results[0] is not None  # Hit
            assert results[1] is None  # Miss
            assert results[2] is not None  # Hit

    async def test_get_embeddings_batch_all_misses(self, tmp_path: Path) -> None:
        """Test batch get with all cache misses."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            results = await cache.get_embeddings_batch(
                texts=["nonexistent1", "nonexistent2"], model="model", dimensions=2
            )

            assert len(results) == 2
            assert results[0] is None
            assert results[1] is None

    async def test_get_embeddings_batch_empty_list(self, tmp_path: Path) -> None:
        """Test batch get with empty list returns empty list."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            results = await cache.get_embeddings_batch(texts=[], model="model", dimensions=2)

            assert results == []

    async def test_get_embeddings_batch_order_preservation(self, tmp_path: Path) -> None:
        """Test that batch operations preserve order of results."""
        cache_path = tmp_path / "test_cache.duckdb"
        texts = ["text a", "text b", "text c", "text d"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embeddings_batch(texts=texts, model="model", dimensions=2, embeddings=embeddings)

            shuffled_texts = ["text c", "text a", "text d", "text b"]
            retrieved = await cache.get_embeddings_batch(shuffled_texts, model="model", dimensions=2)

            assert len(retrieved) == 4
            assert all(emb is not None for emb in retrieved)
            assert retrieved[0] is not None
            assert retrieved[1] is not None
            assert retrieved[2] is not None
            assert retrieved[3] is not None
            assert retrieved[0][0] == pytest.approx(0.5, abs=1e-6)
            assert retrieved[1][0] == pytest.approx(0.1, abs=1e-6)
            assert retrieved[2][0] == pytest.approx(0.7, abs=1e-6)
            assert retrieved[3][0] == pytest.approx(0.3, abs=1e-6)

    async def test_store_embeddings_batch(self, tmp_path: Path) -> None:
        """Test batch store operation."""
        cache_path = tmp_path / "test_cache.duckdb"
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Batch store
            await cache.store_embeddings_batch(texts=texts, model="model", dimensions=2, embeddings=embeddings)

            # Verify all were stored
            for i, text in enumerate(texts):
                result = await cache.get_embedding(text=text, model="model", dimensions=2)
                assert result is not None
                assert result[0] == pytest.approx(embeddings[i][0], abs=1e-6)
                assert result[1] == pytest.approx(embeddings[i][1], abs=1e-6)

    async def test_store_embeddings_batch_empty_lists(self, tmp_path: Path) -> None:
        """Test batch store with empty lists does nothing."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Should not raise
            await cache.store_embeddings_batch(texts=[], model="model", dimensions=2, embeddings=[])

    async def test_store_embeddings_batch_length_mismatch(self, tmp_path: Path) -> None:
        """Test batch store with mismatched lengths is gracefully handled."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Should log warning but not raise
            await cache.store_embeddings_batch(
                texts=["text1", "text2"], model="model", dimensions=2, embeddings=[[0.1, 0.2]]
            )

            # Nothing should be stored
            result = await cache.get_embedding(text="text1", model="model", dimensions=2)
            assert result is None

    async def test_store_embeddings_batch_skip_dimension_mismatch(self, tmp_path: Path) -> None:
        """Test batch store skips embeddings with wrong dimensions."""
        cache_path = tmp_path / "test_cache.duckdb"
        texts = ["text1", "text2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4, 0.5]]  # Second has wrong dimensions

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Should store first, skip second
            await cache.store_embeddings_batch(texts=texts, model="model", dimensions=2, embeddings=embeddings)

            result1 = await cache.get_embedding(text="text1", model="model", dimensions=2)
            result2 = await cache.get_embedding(text="text2", model="model", dimensions=2)

            assert result1 is not None
            assert result2 is None  # Skipped due to dimension mismatch


class TestEmbeddingCacheClear:
    """Tests for clear_cache operation."""

    async def test_clear_cache_all(self, tmp_path: Path) -> None:
        """Test clearing entire cache."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Store some embeddings
            await cache.store_embedding(text="text1", model="model-a", dimensions=2, embedding=[0.1, 0.2])
            await cache.store_embedding(text="text2", model="model-b", dimensions=2, embedding=[0.3, 0.4])

            # Clear all
            deleted_count = await cache.clear_cache()

            assert deleted_count == 2

            # Verify all are gone
            result1 = await cache.get_embedding(text="text1", model="model-a", dimensions=2)
            result2 = await cache.get_embedding(text="text2", model="model-b", dimensions=2)

            assert result1 is None
            assert result2 is None

    async def test_clear_cache_by_model(self, tmp_path: Path) -> None:
        """Test clearing cache for specific model."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            # Store embeddings for different models
            await cache.store_embedding(text="text1", model="model-a", dimensions=2, embedding=[0.1, 0.2])
            await cache.store_embedding(text="text2", model="model-b", dimensions=2, embedding=[0.3, 0.4])

            # Clear only model-a
            deleted_count = await cache.clear_cache(model="model-a")

            assert deleted_count == 1

            # Verify model-a is gone but model-b remains
            result_a = await cache.get_embedding(text="text1", model="model-a", dimensions=2)
            result_b = await cache.get_embedding(text="text2", model="model-b", dimensions=2)

            assert result_a is None
            assert result_b is not None

    async def test_clear_cache_empty_database(self, tmp_path: Path) -> None:
        """Test clearing empty cache returns 0."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            deleted_count = await cache.clear_cache()

            assert deleted_count == 0

    async def test_clear_cache_older_than_days(self, tmp_path: Path) -> None:
        """Test clearing entries older than N days."""
        cache_path = tmp_path / "test_cache.duckdb"
        model = "text-embedding-3-small"
        dimensions = 512

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding("text 1", model, dimensions, [0.1] * dimensions)
            await cache.store_embedding("text 2", model, dimensions, [0.2] * dimensions)

            assert cache._cache is not None
            text_hash = cache._hash_text("text 1")
            key = cache._build_key(text_hash, model, dimensions)
            payload = cache._cache.get(key)
            assert isinstance(payload, dict)
            payload["created_at"] = time.time() - (10 * 86400)
            cache._cache.set(key, payload)

            deleted = await cache.clear_cache(older_than_days=5)

            assert deleted == 1
            assert await cache.get_embedding("text 1", model, dimensions) is None
            assert await cache.get_embedding("text 2", model, dimensions) is not None

    async def test_clear_cache_combined_filters(self, tmp_path: Path) -> None:
        """Test clear_cache with both model and older_than_days filters."""
        cache_path = tmp_path / "test_cache.duckdb"
        model1 = "text-embedding-3-small"
        model2 = "text-embedding-3-large"
        dimensions = 512

        async with EmbeddingCache(db_path=cache_path) as cache:
            await cache.store_embedding("text 1", model1, dimensions, [0.1] * dimensions)
            await cache.store_embedding("text 2", model1, dimensions, [0.2] * dimensions)
            await cache.store_embedding("text 3", model2, dimensions, [0.3] * dimensions)

            assert cache._cache is not None
            text_hash1 = cache._hash_text("text 1")
            text_hash3 = cache._hash_text("text 3")
            key1 = cache._build_key(text_hash1, model1, dimensions)
            key3 = cache._build_key(text_hash3, model2, dimensions)
            payload1 = cache._cache.get(key1)
            payload3 = cache._cache.get(key3)
            assert isinstance(payload1, dict)
            assert isinstance(payload3, dict)
            old_timestamp = time.time() - (10 * 86400)
            payload1["created_at"] = old_timestamp
            payload3["created_at"] = old_timestamp
            cache._cache.set(key1, payload1)
            cache._cache.set(key3, payload3)

            deleted = await cache.clear_cache(model=model1, older_than_days=5)

            assert deleted == 1
            assert await cache.get_embedding("text 1", model1, dimensions) is None
            assert await cache.get_embedding("text 2", model1, dimensions) is not None
            assert await cache.get_embedding("text 3", model2, dimensions) is not None


class TestEmbeddingGenerationCaching:
    """Tests for caching behavior in high-level embedding helpers."""

    async def test_get_embedding_cache_none_passthrough(self) -> None:
        """Test that cache=None bypasses caching and calls generation."""
        from oidm_common.embeddings import generation

        dummy_client = object()

        with (
            patch.object(generation, "_get_or_create_client", return_value=dummy_client) as mock_client,
            patch.object(generation, "generate_embedding", new=AsyncMock(return_value=[0.1, 0.2])) as mock_generate,
        ):
            result = await generation.get_embedding(
                "test text",
                api_key="test-key",
                model="test-model",
                dimensions=2,
                cache=None,
            )

        assert result == [0.1, 0.2]
        mock_client.assert_called_once_with("test-key")
        mock_generate.assert_called_once_with("test text", dummy_client, "test-model", 2)

    async def test_get_embedding_cache_hit_no_api_call(self, tmp_path: Path) -> None:
        """Test that cache hit skips OpenAI calls."""
        cache_path = tmp_path / "test_cache.duckdb"
        cache = EmbeddingCache(db_path=cache_path)
        await cache.setup()
        await cache.store_embedding("cached", "test-model", 2, [0.3, 0.4])

        from oidm_common.embeddings import generation

        with (
            patch.object(generation, "_get_or_create_client", return_value=object()) as mock_client,
            patch.object(generation, "generate_embedding", new=AsyncMock(return_value=[0.9, 1.0])) as mock_generate,
        ):
            result = await generation.get_embedding(
                "cached",
                api_key="test-key",
                model="test-model",
                dimensions=2,
                cache=cache,
            )

        assert result is not None
        assert result[0] == pytest.approx(0.3, abs=1e-6)
        mock_client.assert_not_called()
        mock_generate.assert_not_called()

    async def test_get_embedding_cache_miss_generates_and_stores(self, tmp_path: Path) -> None:
        """Test that cache misses generate and store embeddings."""
        cache_path = tmp_path / "test_cache.duckdb"
        cache = EmbeddingCache(db_path=cache_path)
        await cache.setup()

        from oidm_common.embeddings import generation

        dummy_client = object()
        with (
            patch.object(generation, "_get_or_create_client", return_value=dummy_client),
            patch.object(generation, "generate_embedding", new=AsyncMock(return_value=[0.7, 0.8])),
        ):
            result = await generation.get_embedding(
                "miss",
                api_key="test-key",
                model="test-model",
                dimensions=2,
                cache=cache,
            )

        assert result == [0.7, 0.8]
        cached = await cache.get_embedding("miss", "test-model", 2)
        assert cached is not None
        assert cached[0] == pytest.approx(0.7, abs=1e-6)

    async def test_get_embeddings_batch_partial_cache_hits(self, tmp_path: Path) -> None:
        """Test batch caching with partial hits."""
        cache_path = tmp_path / "test_cache.duckdb"
        cache = EmbeddingCache(db_path=cache_path)
        await cache.setup()
        await cache.store_embedding("cached", "test-model", 2, [0.2, 0.2])

        from oidm_common.embeddings import generation

        dummy_client = object()
        with (
            patch.object(generation, "_get_or_create_client", return_value=dummy_client),
            patch.object(
                generation,
                "generate_embeddings_batch",
                new=AsyncMock(return_value=[[0.3, 0.3], [0.4, 0.4]]),
            ) as mock_generate,
        ):
            result = await generation.get_embeddings_batch(
                ["cached", "miss1", "miss2"],
                api_key="test-key",
                model="test-model",
                dimensions=2,
                cache=cache,
            )

        assert result[0] == pytest.approx([0.2, 0.2], abs=1e-6)
        assert result[1] == [0.3, 0.3]
        assert result[2] == [0.4, 0.4]
        mock_generate.assert_called_once_with(["miss1", "miss2"], dummy_client, "test-model", 2)

    async def test_get_embeddings_batch_filters_none_before_storing(self, tmp_path: Path) -> None:
        """Test that None embeddings are filtered before caching."""

        class _RecordingCache(EmbeddingCache):
            def __init__(self, db_path: Path) -> None:
                super().__init__(db_path)
                self.calls: list[tuple[list[str], list[list[float]]]] = []

            async def store_embeddings_batch(
                self, texts: list[str], model: str, dimensions: int, embeddings: list[list[float]]
            ) -> None:
                _ = (model, dimensions)
                self.calls.append((list(texts), list(embeddings)))

        cache = _RecordingCache(tmp_path / "test_cache.duckdb")
        await cache.setup()

        from oidm_common.embeddings import generation

        dummy_client = object()
        with (
            patch.object(generation, "_get_or_create_client", return_value=dummy_client),
            patch.object(
                generation,
                "generate_embeddings_batch",
                new=AsyncMock(return_value=[[0.1, 0.1], None, [0.3, 0.3]]),
            ),
        ):
            result = await generation.get_embeddings_batch(
                ["a", "b", "c"],
                api_key="test-key",
                model="test-model",
                dimensions=2,
                cache=cache,
            )

        assert result == [[0.1, 0.1], None, [0.3, 0.3]]
        assert cache.calls == [(["a", "c"], [[0.1, 0.1], [0.3, 0.3]])]


class TestFastEmbedCachePathResolution:
    """Tests for FastEmbed model cache directory selection."""

    def test_fastembed_uses_oidm_common_cache_dir_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Runtime should use oidm-common platform cache dir by default."""
        from oidm_common.embeddings import generation

        calls: list[dict[str, object]] = []

        class _FakeTextEmbedding:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        monkeypatch.setattr(generation, "_DEFAULT_FASTEMBED_CACHE_DIR", tmp_path / "oidm-fastembed")
        monkeypatch.setattr(generation, "_TEMP_FASTEMBED_CACHE_DIR", tmp_path / "temp-fastembed")
        monkeypatch.setattr(generation, "_fastembed_model_cache", {})
        monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_FakeTextEmbedding))

        runtime = generation._get_or_create_fastembed_model("BAAI/bge-small-en-v1.5", threads=2)

        assert runtime is not None
        assert len(calls) == 1
        assert calls[0]["model_name"] == "BAAI/bge-small-en-v1.5"
        assert calls[0]["threads"] == 2
        assert calls[0]["cache_dir"] == str(tmp_path / "oidm-fastembed")
        assert (tmp_path / "oidm-fastembed").exists()

    def test_fastembed_falls_back_when_primary_cache_dir_unwritable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If primary cache path is invalid/unwritable, runtime should use temp fallback."""
        from oidm_common.embeddings import generation

        calls: list[dict[str, object]] = []

        class _FakeTextEmbedding:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        invalid_primary = tmp_path / "not-a-directory"
        invalid_primary.write_text("x")
        fallback = tmp_path / "fallback-fastembed-cache"

        monkeypatch.setattr(generation, "_DEFAULT_FASTEMBED_CACHE_DIR", invalid_primary)
        monkeypatch.setattr(generation, "_TEMP_FASTEMBED_CACHE_DIR", fallback)
        monkeypatch.setattr(generation, "_fastembed_model_cache", {})
        monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_FakeTextEmbedding))

        runtime = generation._get_or_create_fastembed_model("BAAI/bge-small-en-v1.5", threads=None)

        assert runtime is not None
        assert len(calls) == 1
        assert calls[0]["cache_dir"] == str(fallback)
        assert fallback.exists()


@pytest.mark.callout
class TestEmbeddingGenerationIntegration:
    """Integration tests for embedding generation with real OpenAI API calls."""

    async def test_get_embedding_returns_vector(self, test_settings: TestSettings) -> None:
        """Test that get_embedding returns a valid embedding vector."""
        from oidm_common.embeddings import get_embedding

        api_key = test_settings.openai_api_key.get_secret_value()
        if not api_key:
            pytest.skip("OPENAI_API_KEY not configured")

        result = await get_embedding(
            "test medical term: pneumonia",
            api_key=api_key,
            model="text-embedding-3-small",
            dimensions=512,
        )

        assert result is not None
        assert len(result) == 512

    async def test_get_embeddings_batch_returns_vectors(self, test_settings: TestSettings) -> None:
        """Test that get_embeddings_batch returns valid embedding vectors."""
        from oidm_common.embeddings import get_embeddings_batch

        api_key = test_settings.openai_api_key.get_secret_value()
        if not api_key:
            pytest.skip("OPENAI_API_KEY not configured")

        results = await get_embeddings_batch(
            ["pneumonia", "fracture", "tumor"],
            api_key=api_key,
            model="text-embedding-3-small",
            dimensions=512,
        )

        assert len(results) == 3
        assert all(r is not None and len(r) == 512 for r in results)
