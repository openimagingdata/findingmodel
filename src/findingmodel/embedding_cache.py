"""DuckDB-based cache for OpenAI embeddings."""

from __future__ import annotations

import hashlib
from array import array
from pathlib import Path
from typing import Final

import duckdb
from platformdirs import user_cache_dir

from findingmodel import logger

_DEFAULT_CACHE_PATH: Final[Path] = (
    Path(user_cache_dir(appname="findingmodel", appauthor="openimagingdata", ensure_exists=True)) / "embeddings.duckdb"
)


class EmbeddingCache:
    """DuckDB-based cache for OpenAI embeddings.

    This cache stores embeddings with SHA256 text hashing to avoid redundant API calls.
    It operates in a fail-safe manner - cache errors never block embedding operations.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the embedding cache.

        Args:
            db_path: Path to cache database file. Defaults to data/embeddings_cache.duckdb
        """
        self.db_path = db_path or _DEFAULT_CACHE_PATH
        self._conn: duckdb.DuckDBPyConnection | None = None

    async def __aenter__(self) -> EmbeddingCache:
        """Enter context manager."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def setup(self) -> None:
        """Create schema if not exists."""
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create connection (read-write for setup)
            conn = duckdb.connect(str(self.db_path), read_only=False)

            # Create table with dynamic dimensions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding FLOAT[] NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_model
                ON embedding_cache(model, dimensions)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created
                ON embedding_cache(created_at)
            """)

            # Close setup connection to ensure schema is committed
            conn.close()

            logger.debug(f"Embedding cache initialized at {self.db_path}")

        except Exception as e:
            logger.warning(f"Failed to setup embedding cache: {e}")
            # Don't raise - graceful degradation

    def _get_connection(self, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        """Get or create connection with specified mode.

        Args:
            read_only: Whether to open in read-only mode

        Returns:
            DuckDB connection
        """
        # Always create a new connection for the requested mode
        # This avoids issues with switching between read/write modes
        return duckdb.connect(str(self.db_path), read_only=read_only)

    def _hash_text(self, text: str) -> str:
        """Generate SHA256 hash of text.

        Args:
            text: Input text to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _to_float32(self, embedding: list[float]) -> list[float]:
        """Convert embedding to 32-bit floats for storage.

        Args:
            embedding: Embedding vector with 64-bit floats

        Returns:
            Embedding vector with 32-bit floats
        """
        return list(array("f", embedding))

    async def get_embedding(self, text: str, model: str, dimensions: int) -> list[float] | None:
        """Get cached embedding or None if not found.

        Args:
            text: Text that was embedded
            model: OpenAI model name (e.g., "text-embedding-3-small")
            dimensions: Embedding dimension count

        Returns:
            Cached embedding vector or None if cache miss
        """
        try:
            text_hash = self._hash_text(text)
            conn = self._get_connection(read_only=True)

            result = conn.execute(
                """
                SELECT embedding
                FROM embedding_cache
                WHERE text_hash = ?
                  AND model = ?
                  AND dimensions = ?
            """,
                (text_hash, model, dimensions),
            ).fetchone()

            conn.close()

            if result is not None:
                logger.debug(f"Cache hit for text hash {text_hash[:8]}...")
                return list(result[0])

            logger.debug(f"Cache miss for text hash {text_hash[:8]}...")
            return None

        except Exception as e:
            logger.debug(f"Cache lookup error (non-fatal): {e}")
            return None

    async def store_embedding(self, text: str, model: str, dimensions: int, embedding: list[float]) -> None:
        """Store embedding in cache.

        Args:
            text: Text that was embedded
            model: OpenAI model name
            dimensions: Embedding dimension count
            embedding: Embedding vector to cache
        """
        try:
            # Validate dimensions match
            if len(embedding) != dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: expected {dimensions}, got {len(embedding)}. Not caching."
                )
                return

            text_hash = self._hash_text(text)
            embedding_f32 = self._to_float32(embedding)

            conn = self._get_connection(read_only=False)

            # Use INSERT OR REPLACE to handle duplicates
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                (text_hash, model, dimensions, embedding, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (text_hash, model, dimensions, embedding_f32),
            )

            conn.close()
            logger.debug(f"Cached embedding for text hash {text_hash[:8]}...")

        except Exception as e:
            logger.debug(f"Cache store error (non-fatal): {e}")
            # Don't raise - cache failures shouldn't break embedding operations

    async def get_embeddings_batch(self, texts: list[str], model: str, dimensions: int) -> list[list[float] | None]:
        """Get batch of embeddings, returning None for cache misses.

        Args:
            texts: List of texts to look up
            model: OpenAI model name
            dimensions: Embedding dimension count

        Returns:
            List of embeddings (or None for each cache miss)
        """
        if not texts:
            return []

        try:
            # Generate hashes for all texts
            text_hashes = [self._hash_text(text) for text in texts]
            hash_to_text_idx = {h: i for i, h in enumerate(text_hashes)}

            conn = self._get_connection(read_only=True)

            # Bulk query using IN clause
            placeholders = ",".join("?" * len(text_hashes))
            query = f"""
                SELECT text_hash, embedding
                FROM embedding_cache
                WHERE text_hash IN ({placeholders})
                  AND model = ?
                  AND dimensions = ?
            """

            results = conn.execute(query, (*text_hashes, model, dimensions)).fetchall()
            conn.close()

            # Build result list, preserving order
            embeddings: list[list[float] | None] = [None] * len(texts)
            hits = 0

            for text_hash, embedding in results:
                idx = hash_to_text_idx[text_hash]
                embeddings[idx] = list(embedding)
                hits += 1

            logger.debug(f"Batch cache: {hits}/{len(texts)} hits")
            return embeddings

        except Exception as e:
            logger.debug(f"Batch cache lookup error (non-fatal): {e}")
            return [None] * len(texts)

    async def store_embeddings_batch(
        self, texts: list[str], model: str, dimensions: int, embeddings: list[list[float]]
    ) -> None:
        """Store batch of embeddings.

        Args:
            texts: List of texts that were embedded
            model: OpenAI model name
            dimensions: Embedding dimension count
            embeddings: List of embedding vectors to cache
        """
        if not texts or not embeddings:
            return

        if len(texts) != len(embeddings):
            logger.warning(
                f"Text/embedding count mismatch: {len(texts)} texts, {len(embeddings)} embeddings. Not caching."
            )
            return

        try:
            conn = self._get_connection(read_only=False)

            # Prepare batch insert data
            records = []
            for text, embedding in zip(texts, embeddings, strict=True):
                if len(embedding) != dimensions:
                    logger.warning(f"Skipping embedding with wrong dimensions: {len(embedding)} != {dimensions}")
                    continue

                text_hash = self._hash_text(text)
                embedding_f32 = self._to_float32(embedding)
                records.append((text_hash, model, dimensions, embedding_f32))

            if not records:
                conn.close()
                return

            # Bulk insert with executemany
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache
                (text_hash, model, dimensions, embedding, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                records,
            )

            conn.close()
            logger.debug(f"Cached {len(records)} embeddings in batch")

        except Exception as e:
            logger.debug(f"Batch cache store error (non-fatal): {e}")
            # Don't raise - cache failures shouldn't break embedding operations

    async def clear_cache(self, model: str | None = None, older_than_days: int | None = None) -> int:
        """Clear cached embeddings with optional filters.

        Args:
            model: If provided, only clear embeddings for this model
            older_than_days: If provided, only clear embeddings older than this many days

        Returns:
            Number of entries deleted
        """
        try:
            conn = self._get_connection(read_only=False)

            conditions: list[str] = []
            params: list[str | int] = []

            if model is not None:
                conditions.append("model = ?")
                params.append(model)

            if older_than_days is not None:
                conditions.append("created_at < CURRENT_TIMESTAMP - INTERVAL ? DAY")
                params.append(older_than_days)

            if conditions:
                where_clause = " AND ".join(conditions)
                query = f"DELETE FROM embedding_cache WHERE {where_clause}"
            else:
                query = "DELETE FROM embedding_cache"

            result = conn.execute(query, params if params else None)
            row = result.fetchone()
            deleted_count = row[0] if row else 0

            conn.close()
            logger.info(f"Cleared {deleted_count} cached embeddings")
            return deleted_count

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return 0


__all__ = ["EmbeddingCache"]
