from __future__ import annotations

from pathlib import Path

from oidm_common.distribution import ensure_db_file as oidm_ensure_db_file
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    pass


class FindingModelConfig(BaseSettings):
    """Settings for finding model database management.

    Configuration can be provided via environment variables with FINDINGMODEL_ prefix:
    - FINDINGMODEL_DB_PATH: Path to database file (also accepts legacy DUCKDB_INDEX_PATH)
    - FINDINGMODEL_REMOTE_DB_URL: URL to download database from (also accepts legacy REMOTE_INDEX_DB_URL)
    - FINDINGMODEL_REMOTE_DB_HASH: Expected hash for database file (also accepts legacy REMOTE_INDEX_DB_HASH)
    - FINDINGMODEL_MANIFEST_URL: URL to JSON manifest for database versions

    Embedding configuration uses AliasChoices to fall back to standard env vars:
    - FINDINGMODEL_OPENAI_API_KEY or OPENAI_API_KEY: OpenAI API key for embeddings
    - FINDINGMODEL_OPENAI_EMBEDDING_MODEL or OPENAI_EMBEDDING_MODEL: model (default: text-embedding-3-small)
    - FINDINGMODEL_OPENAI_EMBEDDING_DIMENSIONS or OPENAI_EMBEDDING_DIMENSIONS: dimensions (default: 512)
    """

    model_config = SettingsConfigDict(env_prefix="FINDINGMODEL_", env_file=".env", extra="ignore")

    # DuckDB configuration
    # AliasChoices: new FINDINGMODEL_* names preferred, old DUCKDB_INDEX_* still accepted
    db_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FINDINGMODEL_DB_PATH", "DUCKDB_INDEX_PATH"),
    )
    remote_db_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FINDINGMODEL_REMOTE_DB_URL", "REMOTE_INDEX_DB_URL"),
    )
    remote_db_hash: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FINDINGMODEL_REMOTE_DB_HASH", "REMOTE_INDEX_DB_HASH"),
    )
    manifest_url: str = "https://findingmodelsdata.t3.storage.dev/manifest.json"

    # Embedding configuration
    # AliasChoices: try package-specific env var first, fall back to standard name
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FINDINGMODEL_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("FINDINGMODEL_OPENAI_EMBEDDING_MODEL", "OPENAI_EMBEDDING_MODEL"),
    )
    openai_embedding_dimensions: int = Field(
        default=512,
        validation_alias=AliasChoices("FINDINGMODEL_OPENAI_EMBEDDING_DIMENSIONS", "OPENAI_EMBEDDING_DIMENSIONS"),
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
    return oidm_ensure_db_file(
        file_path=s.db_path,
        remote_url=s.remote_db_url,
        remote_hash=s.remote_db_hash,
        manifest_key="finding_models",
        manifest_url=s.manifest_url,
        app_name="findingmodel",
    )


__all__ = [
    "ConfigurationError",
    "FindingModelConfig",
    "ensure_index_db",
    "get_settings",
]
