from __future__ import annotations

from pathlib import Path
from typing import Final

from oidm_common.distribution import ensure_db_file as oidm_ensure_db_file
from oidm_common.distribution.profiles import read_embedding_profile_from_db
from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    pass


_RUNTIME_EMBEDDING_PROFILES: Final[dict[str, tuple[str, str, int]]] = {
    "openai": ("openai", "text-embedding-3-small", 512),
    "local": ("fastembed", "BAAI/bge-small-en-v1.5", 384),
}


def _supported_profile_help(env_prefix: str) -> str:
    openai_provider, openai_model, openai_dims = _RUNTIME_EMBEDDING_PROFILES["openai"]
    local_provider, local_model, local_dims = _RUNTIME_EMBEDDING_PROFILES["local"]
    return (
        "Supported runtime embedding profiles are: "
        "auto (default, resolves to openai when OpenAI API key is set, else local), "
        f"openai ({openai_provider}/{openai_model}/{openai_dims}) and "
        f"local ({local_provider}/{local_model}/{local_dims}). "
        f"Set {env_prefix}_EMBEDDING_PROFILE to one of those values."
    )


class FindingModelConfig(BaseSettings):
    """Settings for finding model database management.

    Configuration can be provided via environment variables with FINDINGMODEL_ prefix:
    - FINDINGMODEL_DB_PATH: Path to database file
    - FINDINGMODEL_REMOTE_DB_URL: URL to download database from
    - FINDINGMODEL_REMOTE_DB_HASH: Expected hash for database file
    - FINDINGMODEL_MANIFEST_URL: URL to JSON manifest for database versions

    Embedding configuration:
    - FINDINGMODEL_EMBEDDING_PROFILE: runtime embedding profile (`auto`, `openai`, or `local`)
    - FINDINGMODEL_OPENAI_API_KEY or OPENAI_API_KEY: OpenAI API key (for provider=openai)
    """

    model_config = SettingsConfigDict(env_prefix="FINDINGMODEL_", env_file=".env", extra="ignore")

    # DuckDB configuration
    db_path: str | None = None
    remote_db_url: str | None = None
    remote_db_hash: str | None = None
    manifest_url: str = "https://findingmodelsdata.t3.storage.dev/manifest.json"

    # Embedding configuration
    # AliasChoices: try package-specific env var first, fall back to standard name
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FINDINGMODEL_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    embedding_profile: str = Field(
        default="auto",
        validation_alias=AliasChoices("FINDINGMODEL_EMBEDDING_PROFILE"),
    )

    @model_validator(mode="after")
    def _validate_embedding_profile(self) -> FindingModelConfig:
        requested_profile = self.embedding_profile.strip().lower()
        if requested_profile not in {"auto", *tuple(_RUNTIME_EMBEDDING_PROFILES.keys())}:
            raise ValueError(
                f"Invalid FINDINGMODEL_EMBEDDING_PROFILE: {self.embedding_profile!r}. "
                f"{_supported_profile_help('FINDINGMODEL')}"
            )
        if requested_profile == "auto":
            openai_key = self.openai_api_key.get_secret_value().strip() if self.openai_api_key else ""
            self.embedding_profile = "openai" if openai_key else "local"
        else:
            self.embedding_profile = requested_profile
        return self

    @property
    def embedding_provider(self) -> str:
        return _RUNTIME_EMBEDDING_PROFILES[self.embedding_profile][0]

    @property
    def embedding_model(self) -> str:
        return _RUNTIME_EMBEDDING_PROFILES[self.embedding_profile][1]

    @property
    def embedding_dimensions(self) -> int:
        return _RUNTIME_EMBEDDING_PROFILES[self.embedding_profile][2]

    @property
    def openai_embedding_model(self) -> str:
        """Alias for compatibility with existing call sites."""
        return self.embedding_model

    @property
    def openai_embedding_dimensions(self) -> int:
        """Alias for compatibility with existing call sites."""
        return self.embedding_dimensions


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
        embedding_provider=s.embedding_provider,
        embedding_model=s.embedding_model,
        embedding_dimensions=s.embedding_dimensions,
    )
    detected = read_embedding_profile_from_db(db_path)
    if detected is not None and detected[0].strip().lower() == "openai":
        api_key = s.openai_api_key.get_secret_value().strip() if s.openai_api_key else ""
        if not api_key:
            raise ConfigurationError(
                "The selected findingmodel database uses OpenAI embeddings "
                f"({detected[1]}/{detected[2]}), but OPENAI_API_KEY is not set."
            )
    return db_path


__all__ = [
    "ConfigurationError",
    "FindingModelConfig",
    "ensure_index_db",
    "get_settings",
]
