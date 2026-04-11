"""Embedding configuration constants.

Defines the single supported embedding configuration and utilities for
reading embedding metadata from DuckDB artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingProfileSpec:
    """Embedding provider/model/dimensions specification."""

    provider: str
    model: str
    dimensions: int


ACTIVE_EMBEDDING_CONFIG = EmbeddingProfileSpec(
    provider="openai",
    model="text-embedding-3-small",
    dimensions=512,
)


def read_embedding_profile_from_db(db_path: str | Path) -> EmbeddingProfileSpec | None:
    """Read embedding provider/model/dimensions from a DuckDB artifact.

    Returns None when metadata is missing or unreadable.
    """
    try:
        import duckdb
    except Exception:
        return None

    try:
        path = Path(db_path)
    except Exception:
        return None
    if not path.exists():
        return None

    try:
        conn = duckdb.connect(str(path), read_only=True)
    except Exception:
        return None

    try:
        has_profile = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'embedding_profile'"
        ).fetchone()
        if not has_profile or int(has_profile[0]) <= 0:
            return None
        row = conn.execute("SELECT provider, model, dimensions FROM embedding_profile LIMIT 1").fetchone()
        if row is None:
            return None
        provider, model, dimensions = row
        if not isinstance(provider, str) or not isinstance(model, str):
            return None
        if not isinstance(dimensions, int):
            return None
        return EmbeddingProfileSpec(provider=provider, model=model, dimensions=int(dimensions))
    except Exception:
        return None
    finally:
        conn.close()


__all__ = [
    "ACTIVE_EMBEDDING_CONFIG",
    "EmbeddingProfileSpec",
    "read_embedding_profile_from_db",
]
