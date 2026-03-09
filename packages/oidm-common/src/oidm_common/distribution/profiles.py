"""Helpers for embedding profile naming and manifest key resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingProfileSpec:
    """Canonical embedding profile specification."""

    name: str
    provider: str
    model: str
    dimensions: int


OPENAI_PROFILE = EmbeddingProfileSpec(
    name="openai",
    provider="openai",
    model="text-embedding-3-small",
    dimensions=512,
)
LOCAL_PROFILE = EmbeddingProfileSpec(
    name="local",
    provider="fastembed",
    model="BAAI/bge-small-en-v1.5",
    dimensions=384,
)
SUPPORTED_EMBEDDING_PROFILES: tuple[EmbeddingProfileSpec, ...] = (OPENAI_PROFILE, LOCAL_PROFILE)


def slugify_profile_part(value: str) -> str:
    """Convert a profile value into a manifest-safe slug."""
    normalized = value.strip().lower()
    if not normalized:
        return "default"
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(".", "_")
    normalized = normalized.replace("/", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "default"


def build_profile_descriptor(provider: str, model: str, dimensions: int) -> str:
    """Build normalized profile descriptor string used in manifest aliases."""
    return f"{provider.strip().lower()}:{model.strip()}:{dimensions}"


def resolve_profile_name(provider: str, model: str, dimensions: int) -> str | None:
    """Return canonical profile name for known provider/model/dimensions."""
    descriptor = build_profile_descriptor(provider, model, dimensions)
    for spec in SUPPORTED_EMBEDDING_PROFILES:
        if descriptor == build_profile_descriptor(spec.provider, spec.model, spec.dimensions):
            return spec.name
    return None


def build_named_profile_manifest_key(base_key: str, profile_name: str) -> str:
    """Build explicit profile manifest key using profile label."""
    normalized = profile_name.strip().lower()
    if not normalized:
        raise ValueError("profile_name must be non-empty")
    return f"{base_key}__{normalized}"


def build_profile_manifest_key(base_key: str, provider: str, model: str, dimensions: int) -> str:
    """Build deterministic manifest key for a profile-specific database artifact."""
    resolved_name = resolve_profile_name(provider, model, dimensions)
    if resolved_name is not None:
        return build_named_profile_manifest_key(base_key, resolved_name)
    provider_slug = slugify_profile_part(provider)
    model_slug = slugify_profile_part(model)
    return f"{base_key}__{provider_slug}__{model_slug}__{dimensions}"


def read_embedding_profile_from_db(db_path: str | Path) -> tuple[str, str, int] | None:
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
        return provider, model, int(dimensions)
    except Exception:
        return None
    finally:
        conn.close()


__all__ = [
    "LOCAL_PROFILE",
    "OPENAI_PROFILE",
    "SUPPORTED_EMBEDDING_PROFILES",
    "EmbeddingProfileSpec",
    "build_named_profile_manifest_key",
    "build_profile_descriptor",
    "build_profile_manifest_key",
    "read_embedding_profile_from_db",
    "resolve_profile_name",
    "slugify_profile_part",
]
