from pathlib import Path

from oidm_common.distribution import ensure_db_file as oidm_ensure_db_file
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class ConfigurationError(RuntimeError):
    pass


class FindingModelConfig(BaseSettings):
    # API Keys (kept for embeddings)
    openai_api_key: SecretStr = Field(default=SecretStr(""))

    # DuckDB configuration
    duckdb_index_path: str | None = Field(
        default=None,
        description="Path to finding models index database (absolute, relative to user data dir, or None for default)",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI model for generating embeddings"
    )
    openai_embedding_dimensions: int = Field(
        default=512, description="Embedding dimensions (512 for text-embedding-3-small reduced, 1536 for full)"
    )

    # Optional remote DuckDB download URLs
    remote_index_db_url: str | None = Field(
        default=None,
        description="URL to download finding models index database",
    )
    remote_index_db_hash: str | None = Field(
        default=None,
        description="SHA256 hash for index DB (e.g. 'sha256:def...')",
    )
    remote_manifest_url: str | None = Field(
        default="https://findingmodelsdata.t3.storage.dev/manifest.json",
        description="URL to JSON manifest for database versions",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    @model_validator(mode="after")
    def validate_remote_db_config(self) -> Self:
        """Validate that remote URL and hash are provided together (or neither)."""
        # Check index database config
        if (self.remote_index_db_url is None) != (self.remote_index_db_hash is None):
            raise ValueError(
                "Must provide both REMOTE_INDEX_DB_URL and REMOTE_INDEX_DB_HASH, or neither. "
                f"Got URL={'set' if self.remote_index_db_url else 'unset'}, "
                f"hash={'set' if self.remote_index_db_hash else 'unset'}"
            )

        return self


settings = FindingModelConfig()


def ensure_index_db() -> Path:
    """Ensure finding models index database is available.

    Uses settings to determine path, URL, and hash configuration.
    Automatically downloads from manifest if no local file exists.

    Returns:
        Path to the finding models index database
    """
    return oidm_ensure_db_file(
        file_path=settings.duckdb_index_path,
        remote_url=settings.remote_index_db_url,
        remote_hash=settings.remote_index_db_hash,
        manifest_key="finding_models",
        manifest_url=settings.remote_manifest_url,
        app_name="findingmodel",
    )


__all__ = [
    "ConfigurationError",
    "FindingModelConfig",
    "ensure_index_db",
    "settings",
]
