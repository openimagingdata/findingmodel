from __future__ import annotations

from pathlib import Path

from oidm_common.distribution import ensure_db_file as oidm_ensure_db_file
from oidm_common.embeddings.config import read_embedding_profile_from_db
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    pass


class FindingModelConfig(BaseSettings):
    """Settings for finding model database management.

    Configuration can be provided via environment variables with FINDINGMODEL_ prefix:
    - FINDINGMODEL_DB_PATH: Path to database file
    - FINDINGMODEL_REMOTE_DB_URL: URL to download database from
    - FINDINGMODEL_REMOTE_DB_HASH: Expected hash for database file
    - FINDINGMODEL_MANIFEST_URL: URL to JSON manifest for database versions
    - FINDINGMODEL_OPENAI_API_KEY or OPENAI_API_KEY: OpenAI API key
    """

    model_config = SettingsConfigDict(env_prefix="FINDINGMODEL_", env_file=".env", extra="ignore")

    # DuckDB configuration
    db_path: str | None = None
    remote_db_url: str | None = None
    remote_db_hash: str | None = None
    manifest_url: str = "https://findingmodelsdata.t3.storage.dev/manifest.json"

    # OpenAI API key for semantic search
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FINDINGMODEL_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )


# Lazy singleton instance
_settings: FindingModelConfig | None = None


def get_settings() -> FindingModelConfig:
    """Get the singleton settings instance.

    Returns:
        Cached settings instance (loads from environment on first call)
    """
    global _settings
    if _settings is None:
        _settings = FindingModelConfig()
    return _settings


def ensure_index_db() -> Path:
    """Ensure finding models index database is available.

    Uses settings to determine path, URL, and hash configuration.
    Automatically downloads from manifest if no local file exists.

    Returns:
        Path to the finding models index database
    """
    s = get_settings()
    db_path = oidm_ensure_db_file(
        file_path=s.db_path,
        remote_url=s.remote_db_url,
        remote_hash=s.remote_db_hash,
        manifest_key="finding_models",
        manifest_url=s.manifest_url,
        app_name="findingmodel",
    )
    detected = read_embedding_profile_from_db(db_path)
    if detected is not None:
        if detected.provider.strip().lower() != "openai":
            raise ConfigurationError(
                "The selected findingmodel database is not OpenAI-embedded "
                f"({detected.provider}/{detected.model}/{detected.dimensions}). "
                "findingmodel supports only OpenAI embeddings."
            )
        api_key = s.openai_api_key.get_secret_value().strip() if s.openai_api_key else ""
        if not api_key:
            raise ConfigurationError(
                "The selected findingmodel database uses OpenAI embeddings "
                f"({detected.model}/{detected.dimensions}), but OPENAI_API_KEY is not set."
            )
    return db_path


__all__ = [
    "ConfigurationError",
    "FindingModelConfig",
    "ensure_index_db",
    "get_settings",
]
