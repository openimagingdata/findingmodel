"""Configuration management for anatomic-locations package."""

from __future__ import annotations

from pathlib import Path

from oidm_common.distribution import ensure_db_file
from oidm_common.embeddings.config import read_embedding_profile_from_db
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when runtime configuration is incompatible with selected database."""


class AnatomicLocationSettings(BaseSettings):
    """Settings for anatomic locations database management.

    Configuration can be provided via environment variables with ANATOMIC_ prefix:
    - ANATOMIC_DB_PATH: Path to database file (absolute, relative to user data dir, or None for default)
    - ANATOMIC_REMOTE_DB_URL: URL to download database from
    - ANATOMIC_REMOTE_DB_HASH: Expected hash for database file (format: "sha256:...")
    - ANATOMIC_MANIFEST_URL: URL to JSON manifest for database versions
    - ANATOMIC_OPENAI_API_KEY or OPENAI_API_KEY: OpenAI API key
    """

    model_config = SettingsConfigDict(env_prefix="ANATOMIC_", env_file=".env", extra="ignore")

    db_path: str | None = None
    remote_db_url: str | None = None
    remote_db_hash: str | None = None
    manifest_url: str = "https://anatomiclocationsdata.t3.storage.dev/manifest.json"

    # OpenAI API key for semantic search
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANATOMIC_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )


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
    db_path = ensure_db_file(
        file_path=s.db_path,
        remote_url=s.remote_db_url,
        remote_hash=s.remote_db_hash,
        manifest_key="anatomic_locations",
        manifest_url=s.manifest_url,
        app_name="anatomic-locations",
    )
    detected = read_embedding_profile_from_db(db_path)
    if detected is not None:
        if detected.provider.strip().lower() != "openai":
            raise ConfigurationError(
                "The selected anatomic-locations database is not OpenAI-embedded "
                f"({detected.provider}/{detected.model}/{detected.dimensions}). "
                "anatomic-locations supports only OpenAI embeddings."
            )
        api_key = s.openai_api_key.get_secret_value().strip() if s.openai_api_key else ""
        if not api_key:
            raise ConfigurationError(
                "The selected anatomic-locations database uses OpenAI embeddings "
                f"({detected.model}/{detected.dimensions}), but OPENAI_API_KEY is not set."
            )
    return db_path


__all__ = [
    "AnatomicLocationSettings",
    "ConfigurationError",
    "ensure_anatomic_db",
    "get_settings",
]
