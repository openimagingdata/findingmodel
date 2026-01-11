"""Tests for config module."""

import pytest
from oidm_maintenance.config import MaintenanceSettings, get_settings


def test_default_settings() -> None:
    """Test default settings values."""
    settings = MaintenanceSettings()
    assert settings.s3_bucket == "findingmodelsdata"
    assert settings.s3_endpoint_url == "https://fly.storage.tigris.dev"
    assert settings.manifest_key == "manifest.json"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.openai_embedding_dimensions == 512


def test_get_settings_singleton() -> None:
    """Test singleton pattern."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_settings_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that env vars with OIDM_MAINTAIN_ prefix are loaded."""
    monkeypatch.setenv("OIDM_MAINTAIN_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("OIDM_MAINTAIN_OPENAI_EMBEDDING_DIMENSIONS", "1024")

    settings = MaintenanceSettings()
    assert settings.s3_bucket == "test-bucket"
    assert settings.openai_embedding_dimensions == 1024
