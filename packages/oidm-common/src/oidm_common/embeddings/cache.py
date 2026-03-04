"""DiskCache-backed cache for OpenAI embeddings."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from array import array
from datetime import datetime
from pathlib import Path
from typing import Final

import duckdb
from diskcache import Cache
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

_DEFAULT_RUNTIME_CACHE_ROOT: Final[Path] = Path(
    user_cache_dir(appname="oidm-common", appauthor="openimagingdata", ensure_exists=True)
)
_DEFAULT_DISKCACHE_DIR: Final[Path] = (
    _DEFAULT_RUNTIME_CACHE_ROOT / "embeddings.cache"
)
_LEGACY_FINDINGMODEL_CACHE_ROOT: Final[Path] = Path(
    user_cache_dir(appname="findingmodel", appauthor="openimagingdata", ensure_exists=True)
)
_DEFAULT_CACHE_PATH: Final[Path] = _LEGACY_FINDINGMODEL_CACHE_ROOT / "embeddings.duckdb"
_LEGACY_DISKCACHE_DIR: Final[Path] = _LEGACY_FINDINGMODEL_CACHE_ROOT / "embeddings.cache"
_MIGRATION_DUCKDB_STATE_KEY: Final[str] = "__oidm_embedding_cache_migration_duckdb_v1__"
_MIGRATION_DISKCACHE_STATE_KEY: Final[str] = "__oidm_embedding_cache_migration_diskcache_v1__"
_MIGRATION_MODEL: Final[str] = "text-embedding-3-small"
_MIGRATION_DIMENSIONS: Final[int] = 512


class _LegacyTableMissingError(RuntimeError):
    """Raised when legacy duckdb cache does not have embedding_cache table."""


class _EmbeddingCacheTableMissingError(RuntimeError):
    """Raised when duckdb file does not contain embedding_cache table."""


class EmbeddingCache:
    """DiskCache-backed cache for OpenAI embeddings.

    This cache stores embeddings with SHA256 text hashing to avoid redundant API calls.
    It operates in a fail-safe manner - cache errors never block embedding operations.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the embedding cache.

        Args:
            db_path: Legacy DuckDB cache file path used as migration source.
                Defaults to the legacy findingmodel cache file in the platform cache directory.
        """
        self.db_path = db_path or _DEFAULT_CACHE_PATH
        if db_path is None:
            self.cache_dir = _DEFAULT_DISKCACHE_DIR
        else:
            self.cache_dir = Path(f"{self.db_path}.cache")
        self._cache: Cache | None = None
        self._setup_complete = False
        self._setup_lock = asyncio.Lock()

    async def __aenter__(self) -> EmbeddingCache:
        """Enter context manager."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager."""
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        self._setup_complete = False

    async def setup(self) -> None:
        """Initialize diskcache and perform one-time legacy migration."""
        if self._setup_complete and self._cache is not None:
            return

        async with self._setup_lock:
            if self._setup_complete and self._cache is not None:
                return

            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache = Cache(
                    directory=str(self.cache_dir),
                    sqlite_journal_mode="wal",
                )
                self._migrate_legacy_diskcache_if_needed()
                self._migrate_legacy_duckdb_if_needed()
                self._setup_complete = True
                logger.debug(f"Embedding cache initialized at {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to setup embedding cache: {e}")
                if self._cache is not None:
                    self._cache.close()
                    self._cache = None
                self._setup_complete = False

    async def _ensure_cache_available(self, *, strict: bool = False) -> bool:
        await self.setup()
        if self._cache is not None:
            return True
        if strict:
            raise RuntimeError(f"Embedding cache unavailable at {self.cache_dir}")
        return False

    async def require_cache_ready(self) -> None:
        """Ensure cache storage is available, or raise RuntimeError."""
        await self._ensure_cache_available(strict=True)

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

    def _to_epoch(self, timestamp: object | None) -> float:
        if isinstance(timestamp, datetime):
            return timestamp.timestamp()
        if isinstance(timestamp, int | float):
            return float(timestamp)
        return time.time()

    def _build_key(self, text_hash: str, model: str, dimensions: int) -> str:
        return json.dumps([text_hash, model, dimensions], separators=(",", ":"), ensure_ascii=True)

    def _parse_key(self, key: object) -> tuple[str, str, int] | None:
        if not isinstance(key, str):
            return None
        try:
            values = json.loads(key)
        except Exception:
            return None
        if not isinstance(values, list) or len(values) != 3:
            return None
        text_hash, model, dimensions = values
        if not isinstance(text_hash, str) or not isinstance(model, str) or not isinstance(dimensions, int):
            return None
        return text_hash, model, dimensions

    def _migration_state(self, state_key: str) -> dict[str, object] | None:
        if self._cache is None:
            return None
        value = self._cache.get(state_key, default=None)
        if isinstance(value, dict):
            return value
        return None

    def _set_migration_state(self, state_key: str, **kwargs: object) -> None:
        if self._cache is None:
            return
        payload = {"at": time.time(), **kwargs}
        self._cache.set(state_key, payload)

    def _get_legacy_diskcache_source_dir(self) -> Path | None:
        source_dir = _LEGACY_DISKCACHE_DIR
        if source_dir == self.cache_dir:
            self._set_migration_state(
                _MIGRATION_DISKCACHE_STATE_KEY,
                status="same_dir",
                source=str(source_dir),
            )
            return None
        if not source_dir.exists():
            self._set_migration_state(
                _MIGRATION_DISKCACHE_STATE_KEY,
                status="no_source",
                source=str(source_dir),
            )
            return None
        return source_dir

    def _legacy_diskcache_payload_to_record(self, payload: object) -> dict[str, object] | None:
        if not isinstance(payload, dict):
            return None
        embedding = payload.get("embedding")
        created_at = payload.get("created_at")
        if not isinstance(embedding, list):
            return None
        if not isinstance(created_at, int | float):
            created_at = time.time()
        return {
            "embedding": self._to_float32(embedding),
            "created_at": float(created_at),
        }

    def _migrate_from_legacy_diskcache(self, source_cache: Cache) -> tuple[int, int]:
        if self._cache is None:
            return 0, 0

        migrated = 0
        skipped = 0
        with self._cache.transact():
            for key in source_cache.iterkeys():
                if self._parse_key(key) is None:
                    continue
                record = self._legacy_diskcache_payload_to_record(source_cache.get(key, default=None))
                if record is None:
                    skipped += 1
                    continue
                if self._cache.add(key, record):
                    migrated += 1
        return migrated, skipped

    def _import_from_diskcache(self, source_cache: Cache, *, upsert: bool) -> tuple[int, int, int, int]:
        """Import embedding entries from another diskcache instance."""
        if self._cache is None:
            return 0, 0, 0, 0

        written = 0
        new = 0
        updated = 0
        skipped = 0
        with self._cache.transact():
            for key in source_cache.iterkeys():
                if self._parse_key(key) is None:
                    continue
                record = self._legacy_diskcache_payload_to_record(source_cache.get(key, default=None))
                if record is None:
                    skipped += 1
                    continue

                if upsert:
                    existing = self._cache.get(key, default=None)
                    self._cache.set(key, record)
                    written += 1
                    if existing is None:
                        new += 1
                    else:
                        updated += 1
                    continue

                if self._cache.add(key, record):
                    written += 1
                    new += 1
                else:
                    skipped += 1
        return written, new, updated, skipped

    def _migrate_legacy_diskcache_if_needed(self) -> None:
        """Migrate entries from legacy findingmodel diskcache directory once."""
        if self._cache is None:
            return
        # Only default runtime cache should auto-migrate from global legacy findingmodel cache.
        if self.cache_dir != _DEFAULT_DISKCACHE_DIR:
            return
        if self._migration_state(_MIGRATION_DISKCACHE_STATE_KEY) is not None:
            return

        source_dir = self._get_legacy_diskcache_source_dir()
        if source_dir is None:
            return

        source_cache: Cache | None = None
        try:
            source_cache = Cache(directory=str(source_dir), sqlite_journal_mode="wal")
            migrated, skipped = self._migrate_from_legacy_diskcache(source_cache)

            self._set_migration_state(
                _MIGRATION_DISKCACHE_STATE_KEY,
                status="migrated",
                source=str(source_dir),
                migrated=migrated,
                skipped=skipped,
            )
            logger.info(
                "Embedding cache diskcache migration completed: "
                f"{migrated} migrated, {skipped} skipped from {source_dir}"
            )
        except Exception as e:
            logger.warning(f"Legacy diskcache migration failed: {e}")
            self._set_migration_state(
                _MIGRATION_DISKCACHE_STATE_KEY,
                status="failed",
                source=str(source_dir),
                error=str(e),
            )
        finally:
            if source_cache is not None:
                source_cache.close()

    def _read_legacy_rows(self) -> list[tuple[object, object, object]]:
        conn = duckdb.connect(str(self.db_path), read_only=True)
        try:
            table_exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'embedding_cache'"
            ).fetchone()
            if table_exists is None or table_exists[0] == 0:
                raise _LegacyTableMissingError("embedding_cache table not found")
            return conn.execute("SELECT text_hash, embedding, created_at FROM embedding_cache").fetchall()
        finally:
            conn.close()

    def _legacy_row_to_record(self, row: tuple[object, object, object]) -> tuple[str, dict[str, object]] | None:
        text_hash, embedding, created_at = row
        if not isinstance(text_hash, str) or not isinstance(embedding, list):
            return None
        if len(embedding) != _MIGRATION_DIMENSIONS:
            return None
        key = self._build_key(text_hash, _MIGRATION_MODEL, _MIGRATION_DIMENSIONS)
        payload = {
            "embedding": self._to_float32(embedding),
            "created_at": self._to_epoch(created_at),
        }
        return key, payload

    def _migrate_rows_into_cache(self, rows: list[tuple[object, object, object]]) -> tuple[int, int]:
        if self._cache is None:
            return 0, len(rows)

        migrated = 0
        skipped = 0
        with self._cache.transact():
            for row in rows:
                record = self._legacy_row_to_record(row)
                if record is None:
                    skipped += 1
                    continue
                key, payload = record
                if self._cache.add(key, payload):
                    migrated += 1
        return migrated, skipped

    def _migrate_legacy_duckdb_if_needed(self) -> None:
        """Migrate data from legacy DuckDB cache into diskcache once."""
        if self._cache is None:
            return
        if self._migration_state(_MIGRATION_DUCKDB_STATE_KEY) is not None:
            return

        if not self.db_path.exists():
            self._set_migration_state(_MIGRATION_DUCKDB_STATE_KEY, status="no_source", source=str(self.db_path))
            return

        try:
            rows = self._read_legacy_rows()
            migrated, skipped = self._migrate_rows_into_cache(rows)

            self._set_migration_state(
                _MIGRATION_DUCKDB_STATE_KEY,
                status="migrated",
                source=str(self.db_path),
                migrated=migrated,
                skipped=skipped,
                total=len(rows),
            )
            logger.info(
                "Embedding cache migration completed: "
                f"{migrated} migrated, {skipped} skipped from {self.db_path}"
            )
        except _LegacyTableMissingError:
            self._set_migration_state(_MIGRATION_DUCKDB_STATE_KEY, status="no_table", source=str(self.db_path))
        except Exception as e:
            logger.warning(f"Legacy embedding cache migration failed: {e}")
            self._set_migration_state(
                _MIGRATION_DUCKDB_STATE_KEY,
                status="failed",
                source=str(self.db_path),
                error=str(e),
            )

    def _read_import_rows(self, source_path: Path) -> list[tuple[object, object, object, object, object]]:
        conn = duckdb.connect(str(source_path), read_only=True)
        try:
            table_exists = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'embedding_cache'"
            ).fetchone()
            if table_exists is None or table_exists[0] == 0:
                raise _EmbeddingCacheTableMissingError("embedding_cache table not found")

            describe_rows = conn.execute("DESCRIBE embedding_cache").fetchall()
            columns = {row[0] for row in describe_rows if row and isinstance(row[0], str)}
            if "text_hash" not in columns or "embedding" not in columns:
                raise ValueError("embedding_cache must include text_hash and embedding columns")

            model_sql = "model" if "model" in columns else "NULL"
            dimensions_sql = "dimensions" if "dimensions" in columns else "NULL"
            created_at_sql = "created_at" if "created_at" in columns else "NULL"

            return conn.execute(
                f"""
                SELECT text_hash, {model_sql}, {dimensions_sql}, embedding, {created_at_sql}
                FROM embedding_cache
                """
            ).fetchall()
        finally:
            conn.close()

    def _import_row_to_record(
        self,
        row: tuple[object, object, object, object, object],
        *,
        assume_defaults: bool,
        default_model: str,
        default_dimensions: int,
    ) -> tuple[str, dict[str, object]] | None:
        text_hash, row_model, row_dimensions, embedding, created_at = row
        if not isinstance(text_hash, str) or not isinstance(embedding, list):
            return None

        if assume_defaults:
            model = default_model
            dimensions = default_dimensions
        else:
            if not isinstance(row_model, str) or not isinstance(row_dimensions, int):
                return None
            model = row_model
            dimensions = row_dimensions

        if len(embedding) != dimensions:
            return None

        key = self._build_key(text_hash, model, dimensions)
        payload = {
            "embedding": self._to_float32(embedding),
            "created_at": self._to_epoch(created_at),
        }
        return key, payload

    async def import_duckdb_file(
        self,
        source_path: Path,
        *,
        assume_defaults: bool = True,
        default_model: str = _MIGRATION_MODEL,
        default_dimensions: int = _MIGRATION_DIMENSIONS,
        strict: bool = False,
    ) -> dict[str, int]:
        """Import embeddings from a DuckDB embedding_cache table into current cache.

        Args:
            source_path: Path to source DuckDB file containing embedding_cache table.
            assume_defaults: If True, import all rows under default model/dimensions.
                If False, preserve model and dimensions from each row.
            default_model: Model used when assume_defaults=True.
            default_dimensions: Dimensions used when assume_defaults=True.
            strict: If True, raise when cache storage is unavailable.

        Returns:
            Stats dict with written/new/updated/skipped/total rows.
        """
        if not await self._ensure_cache_available(strict=strict):
            return {"imported": 0, "written": 0, "new": 0, "updated": 0, "skipped": 0, "total": 0}
        if default_dimensions <= 0:
            raise ValueError("default_dimensions must be positive")

        rows = self._read_import_rows(source_path)
        assert self._cache is not None
        written = 0
        new = 0
        updated = 0
        skipped = 0
        with self._cache.transact():
            for row in rows:
                record = self._import_row_to_record(
                    row,
                    assume_defaults=assume_defaults,
                    default_model=default_model,
                    default_dimensions=default_dimensions,
                )
                if record is None:
                    skipped += 1
                    continue
                key, payload = record
                # Upsert semantics: always overwrite existing value for the key.
                existing = self._cache.get(key, default=None)
                self._cache.set(key, payload)
                written += 1
                if existing is None:
                    new += 1
                else:
                    updated += 1

        return {
            "imported": written,
            "written": written,
            "new": new,
            "updated": updated,
            "skipped": skipped,
            "total": len(rows),
        }

    async def import_cache_dir(self, source_dir: Path, *, upsert: bool = True, strict: bool = False) -> dict[str, int]:
        """Import embedding entries from another diskcache directory.

        Args:
            source_dir: Source diskcache directory.
            upsert: If True, overwrite existing keys. If False, keep existing keys.
            strict: If True, raise when cache storage is unavailable.

        Returns:
            Stats dict with written/new/updated/skipped/total rows.
        """
        if not await self._ensure_cache_available(strict=strict):
            return {"imported": 0, "written": 0, "new": 0, "updated": 0, "skipped": 0, "total": 0}
        if source_dir == self.cache_dir:
            raise ValueError("source_dir must be different from current cache directory")
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)

        source_cache: Cache | None = None
        try:
            source_cache = Cache(directory=str(source_dir), sqlite_journal_mode="wal")
            written, new, updated, skipped = self._import_from_diskcache(source_cache, upsert=upsert)
            return {
                "imported": written,
                "written": written,
                "new": new,
                "updated": updated,
                "skipped": skipped,
                "total": written + skipped,
            }
        finally:
            if source_cache is not None:
                source_cache.close()

    def _embedding_keys(self) -> list[tuple[object, str]]:
        if self._cache is None:
            return []
        keys: list[tuple[object, str]] = []
        for key in self._cache.iterkeys():
            parsed = self._parse_key(key)
            if parsed is None:
                continue
            _text_hash, item_model, _dimensions = parsed
            keys.append((key, item_model))
        return keys

    def _is_older_than_cutoff(self, key: object, cutoff: float) -> bool:
        if self._cache is None:
            return False
        payload = self._cache.get(key, default=None)
        if not isinstance(payload, dict):
            return False
        created_at = payload.get("created_at")
        if not isinstance(created_at, int | float):
            return False
        return float(created_at) < cutoff

    def _collect_filtered_keys(self, model: str | None, older_than_days: int | None) -> list[object]:
        cutoff = time.time() - (older_than_days * 86400) if older_than_days is not None else None
        keys_to_delete: list[object] = []
        for key, item_model in self._embedding_keys():
            if model is not None and item_model != model:
                continue
            if cutoff is not None and not self._is_older_than_cutoff(key, cutoff):
                continue
            keys_to_delete.append(key)
        return keys_to_delete

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
            await self.setup()
            if self._cache is None:
                return None
            text_hash = self._hash_text(text)
            key = self._build_key(text_hash, model, dimensions)
            payload = self._cache.get(key, default=None)
            if isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
                logger.debug(f"Cache hit for text hash {text_hash[:8]}...")
                return list(payload["embedding"])

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
            await self.setup()
            if self._cache is None:
                return
            # Validate dimensions match
            if len(embedding) != dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: expected {dimensions}, got {len(embedding)}. Not caching."
                )
                return

            text_hash = self._hash_text(text)
            embedding_f32 = self._to_float32(embedding)
            key = self._build_key(text_hash, model, dimensions)
            self._cache.set(
                key,
                {
                    "embedding": embedding_f32,
                    "created_at": time.time(),
                },
            )
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
            await self.setup()
            if self._cache is None:
                return [None] * len(texts)
            # Generate hashes for all texts
            text_hashes = [self._hash_text(text) for text in texts]
            hash_to_text_idx = {h: i for i, h in enumerate(text_hashes)}

            # Build result list, preserving order
            embeddings: list[list[float] | None] = [None] * len(texts)
            hits = 0

            for text_hash in text_hashes:
                key = self._build_key(text_hash, model, dimensions)
                payload = self._cache.get(key, default=None)
                if isinstance(payload, dict) and isinstance(payload.get("embedding"), list):
                    idx = hash_to_text_idx[text_hash]
                    embeddings[idx] = list(payload["embedding"])
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
            await self.setup()
            if self._cache is None:
                return

            # Prepare batch insert data
            records = []
            for text, embedding in zip(texts, embeddings, strict=True):
                if len(embedding) != dimensions:
                    logger.warning(f"Skipping embedding with wrong dimensions: {len(embedding)} != {dimensions}")
                    continue

                text_hash = self._hash_text(text)
                embedding_f32 = self._to_float32(embedding)
                records.append(
                    (
                        self._build_key(text_hash, model, dimensions),
                        {
                            "embedding": embedding_f32,
                            "created_at": time.time(),
                        },
                    )
                )

            if not records:
                return

            with self._cache.transact():
                for key, payload in records:
                    self._cache.set(key, payload)

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
            await self.setup()
            if self._cache is None:
                return 0

            if model is None and older_than_days is None:
                keys_to_delete = [key for key, _ in self._embedding_keys()]
                with self._cache.transact():
                    for key in keys_to_delete:
                        self._cache.pop(key, default=None)
                deleted_count = len(keys_to_delete)
                logger.info(f"Cleared {deleted_count} cached embeddings")
                return deleted_count

            keys_to_delete = self._collect_filtered_keys(model, older_than_days)

            with self._cache.transact():
                for key in keys_to_delete:
                    self._cache.pop(key, default=None)

            deleted_count = len(keys_to_delete)
            logger.info(f"Cleared {deleted_count} cached embeddings")
            return deleted_count

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
            return 0

    async def get_stats(self, *, strict: bool = False) -> dict[str, object]:
        """Return cache key and model distribution statistics."""
        if not await self._ensure_cache_available(strict=strict):
            return {
                "cache_dir": str(self.cache_dir),
                "total_keys": 0,
                "embedding_keys": 0,
                "migration_keys": 0,
                "models": {},
            }
        assert self._cache is not None

        total_keys = 0
        migration_keys = 0
        model_counts: dict[str, int] = {}

        for key in self._cache.iterkeys():
            total_keys += 1
            parsed = self._parse_key(key)
            if parsed is not None:
                _text_hash, model, _dimensions = parsed
                model_counts[model] = model_counts.get(model, 0) + 1
                continue
            if isinstance(key, str) and key.startswith("__oidm_embedding_cache_migration_"):
                migration_keys += 1

        return {
            "cache_dir": str(self.cache_dir),
            "total_keys": total_keys,
            "embedding_keys": sum(model_counts.values()),
            "migration_keys": migration_keys,
            "models": model_counts,
        }


__all__ = ["EmbeddingCache"]
