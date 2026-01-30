"""Tests for oidm-common EmbeddingCache."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from oidm_common.embeddings.cache import EmbeddingCache

if TYPE_CHECKING:
    from .conftest import TestSettings


class TestEmbeddingCacheContextManager:
    """Tests for EmbeddingCache async context manager."""

    async def test_embedding_cache_context_manager_setup(self, tmp_path: Path) -> None:
        """Test that context manager sets up database and connection."""
        cache_path = tmp_path / "test_cache.duckdb"

        async with EmbeddingCache(db_path=cache_path) as cache:
            assert cache.db_path == cache_path
            assert cache_path.exists()

    async def test_embedding_cache_context_manager_cleanup(self, tmp_path: Path) -> None:
        """Test that context manager closes connection on exit."""
        cache_path = tmp_path / "test_cache.duckdb"

        cache = EmbeddingCache(db_path=cache_path)
        async with cache:
            # Connection should be available during context
            pass

        # After context exit, connection should be closed
        assert cache._conn is None

    async def test_embedding_cache_default_path(self) -> None:
        """Test that EmbeddingCache uses default path when not specified."""
        cache = EmbeddingCache()

        # Should use default cache directory path
        assert cache.db_path.name == "embeddings.duckdb"
        assert "findingmodel" in str(cache.db_path)


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


@pytest.mark.callout
class TestEmbeddingGenerationIntegration:
    """Integration tests for embedding generation with real OpenAI API calls."""

    async def test_get_embedding_returns_vector(self, test_settings: "TestSettings") -> None:
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

    async def test_get_embeddings_batch_returns_vectors(self, test_settings: "TestSettings") -> None:
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
