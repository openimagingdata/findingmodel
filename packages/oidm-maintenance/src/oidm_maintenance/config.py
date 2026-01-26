"""Configuration for maintenance operations."""

import os

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MaintenanceSettings(BaseSettings):
    """Settings for database maintenance operations.

    Configuration for building and publishing database files to S3/Tigris storage.
    All settings can be overridden via environment variables with OIDM_MAINTAIN_ prefix.
    AWS credentials also accept standard AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY.
    """

    model_config = SettingsConfigDict(env_prefix="OIDM_MAINTAIN_", extra="ignore")

    # S3/Tigris settings
    s3_endpoint_url: str = "https://fly.storage.tigris.dev"
    s3_bucket: str = "findingmodelsdata"
    anatomic_s3_bucket: str = "anatomiclocationsdata"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None

    @model_validator(mode="after")
    def _fallback_to_standard_aws_vars(self) -> "MaintenanceSettings":
        """Fall back to standard AWS_* env vars if OIDM_MAINTAIN_AWS_* not set."""
        if self.aws_access_key_id is None:
            key = os.environ.get("AWS_ACCESS_KEY_ID")
            if key:
                self.aws_access_key_id = SecretStr(key)
        if self.aws_secret_access_key is None:
            secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if secret:
                self.aws_secret_access_key = SecretStr(secret)
        return self

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
