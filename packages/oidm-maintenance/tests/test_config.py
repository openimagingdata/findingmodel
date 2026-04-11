"""Tests for config module."""

from oidm_maintenance.config import MaintenanceSettings, get_settings


def test_default_settings() -> None:
    """Test default settings values."""
    settings = MaintenanceSettings()
    assert settings.s3_bucket == "findingmodelsdata"
    assert settings.s3_endpoint_url == "https://fly.storage.tigris.dev"
    assert settings.manifest_key == "manifest.json"


def test_get_settings_singleton() -> None:
    """Test singleton pattern."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
