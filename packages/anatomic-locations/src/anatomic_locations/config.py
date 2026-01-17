"""Configuration management for anatomic-locations package."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from oidm_common.distribution import ensure_db_file
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnatomicLocationSettings(BaseSettings):
    """Settings for anatomic locations database management.

    Configuration can be provided via environment variables with ANATOMIC_ prefix:
    - ANATOMIC_DB_PATH: Path to database file (absolute, relative to user data dir, or None for default)
    - ANATOMIC_REMOTE_DB_URL: URL to download database from
    - ANATOMIC_REMOTE_DB_HASH: Expected hash for database file (format: "sha256:...")
    - ANATOMIC_MANIFEST_URL: URL to JSON manifest for database versions
    - ANATOMIC_OPENAI_API_KEY: OpenAI API key for embeddings (optional, enables semantic search)
    - ANATOMIC_OPENAI_EMBEDDING_MODEL: OpenAI embedding model (default: text-embedding-3-small)
    - ANATOMIC_OPENAI_EMBEDDING_DIMENSIONS: Embedding dimensions (default: 512)
    """

    model_config = SettingsConfigDict(env_prefix="ANATOMIC_")

    db_path: str | None = None
    remote_db_url: str | None = None
    remote_db_hash: str | None = None
    manifest_url: str = "https://findingmodelsdata.t3.storage.dev/manifest.json"

    # Embedding configuration (for hybrid search)
    openai_api_key: SecretStr | None = Field(default=None)
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    openai_embedding_dimensions: int = Field(default=512)


# Singleton instance
_settings: AnatomicLocationSettings | None = None


def get_settings() -> AnatomicLocationSettings:
    """Get the singleton settings instance.

    Returns:
        Cached settings instance (loads from environment on first call)
    """
    global _settings
    if _settings is None:
        _settings = AnatomicLocationSettings()
    return _settings


def ensure_anatomic_db() -> Path:
    """Ensure anatomic locations database is available.

    Uses settings to determine path, URL, and hash configuration.
    Automatically downloads from manifest if no local file exists.

    Returns:
        Path to the anatomic locations database

    Raises:
        DistributionError: If explicit file doesn't exist or download fails
    """
    s = get_settings()
    logger.debug(
        f"Ensuring anatomic database (path={s.db_path}, "
        f"remote_url={'set' if s.remote_db_url else 'unset'}, "
        f"manifest_url={s.manifest_url})"
    )
    return ensure_db_file(
        file_path=s.db_path,
        remote_url=s.remote_db_url,
        remote_hash=s.remote_db_hash,
        manifest_key="anatomic_locations",
        manifest_url=s.manifest_url,
        app_name="anatomic-locations",
    )


__all__ = [
    "AnatomicLocationSettings",
    "ensure_anatomic_db",
    "get_settings",
]
