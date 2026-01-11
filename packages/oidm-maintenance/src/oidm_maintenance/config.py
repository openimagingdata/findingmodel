"""Configuration for maintenance operations."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class MaintenanceSettings(BaseSettings):
    """Settings for database maintenance operations.

    Configuration for building and publishing database files to S3/Tigris storage.
    All settings can be overridden via environment variables with OIDM_MAINTAIN_ prefix.
    """

    model_config = SettingsConfigDict(env_prefix="OIDM_MAINTAIN_", extra="ignore")

    # S3/Tigris settings
    s3_endpoint_url: str = "https://fly.storage.tigris.dev"
    s3_bucket: str = "findingmodelsdata"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None

    # Manifest settings
    manifest_key: str = "manifest.json"
    manifest_backup_prefix: str = "manifests/archive/"

    # OpenAI for embeddings during build
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 512


_settings: MaintenanceSettings | None = None


def get_settings() -> MaintenanceSettings:
    """Get singleton settings instance.

    Returns:
        Singleton MaintenanceSettings loaded from environment

    Example:
        settings = get_settings()
        print(settings.s3_bucket)
    """
    global _settings
    if _settings is None:
        _settings = MaintenanceSettings()
    return _settings


__all__ = ["MaintenanceSettings", "get_settings"]
