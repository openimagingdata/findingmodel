"""Configuration management for anatomic-locations package."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from oidm_common.distribution import ensure_db_file
from oidm_common.distribution.profiles import read_embedding_profile_from_db
from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when runtime configuration is incompatible with selected database."""


_RUNTIME_EMBEDDING_PROFILES: Final[dict[str, tuple[str, str, int]]] = {
    "openai": ("openai", "text-embedding-3-small", 512),
}


def _supported_profile_help(env_prefix: str) -> str:
    openai_provider, openai_model, openai_dims = _RUNTIME_EMBEDDING_PROFILES["openai"]
    return (
        "Supported runtime embedding profiles are: "
        "auto (default, resolves to openai) and "
        f"openai ({openai_provider}/{openai_model}/{openai_dims}). "
        f"Set {env_prefix}_EMBEDDING_PROFILE to one of those values."
    )


class AnatomicLocationSettings(BaseSettings):
    """Settings for anatomic locations database management.

    Configuration can be provided via environment variables with ANATOMIC_ prefix:
    - ANATOMIC_DB_PATH: Path to database file (absolute, relative to user data dir, or None for default)
    - ANATOMIC_REMOTE_DB_URL: URL to download database from
    - ANATOMIC_REMOTE_DB_HASH: Expected hash for database file (format: "sha256:...")
    - ANATOMIC_MANIFEST_URL: URL to JSON manifest for database versions

    Embedding configuration:
    - ANATOMIC_EMBEDDING_PROFILE: runtime embedding profile (`auto` or `openai`)
    - ANATOMIC_OPENAI_API_KEY or OPENAI_API_KEY: OpenAI API key (for provider=openai)
    """

    model_config = SettingsConfigDict(env_prefix="ANATOMIC_", env_file=".env", extra="ignore")

    db_path: str | None = None
    remote_db_url: str | None = None
    remote_db_hash: str | None = None
    manifest_url: str = "https://anatomiclocationsdata.t3.storage.dev/manifest.json"

    # Embedding configuration (for hybrid search)
    # AliasChoices: try package-specific env var first, fall back to standard name
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANATOMIC_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    embedding_profile: str = Field(
        default="auto",
        validation_alias=AliasChoices("ANATOMIC_EMBEDDING_PROFILE"),
    )

    @model_validator(mode="after")
    def _validate_embedding_profile(self) -> AnatomicLocationSettings:
        requested_profile = self.embedding_profile.strip().lower()
        if requested_profile not in {"auto", *tuple(_RUNTIME_EMBEDDING_PROFILES.keys())}:
            raise ValueError(
                f"Invalid ANATOMIC_EMBEDDING_PROFILE: {self.embedding_profile!r}. {_supported_profile_help('ANATOMIC')}"
            )
        if requested_profile == "auto":
            self.embedding_profile = "openai"
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
        embedding_provider=s.embedding_provider,
        embedding_model=s.embedding_model,
        embedding_dimensions=s.embedding_dimensions,
    )
    detected = read_embedding_profile_from_db(db_path)
    if detected is not None:
        provider, model, dimensions = detected
        if provider.strip().lower() != "openai":
            raise ConfigurationError(
                "The selected anatomic-locations database is not OpenAI-embedded "
                f"({provider}/{model}/{dimensions}). anatomic-locations currently supports only OpenAI embeddings."
            )
        api_key = s.openai_api_key.get_secret_value().strip() if s.openai_api_key else ""
        if not api_key:
            raise ConfigurationError(
                "The selected anatomic-locations database uses OpenAI embeddings "
                f"({model}/{dimensions}), but OPENAI_API_KEY is not set."
            )
    return db_path


__all__ = [
    "AnatomicLocationSettings",
    "ConfigurationError",
    "ensure_anatomic_db",
    "get_settings",
]
