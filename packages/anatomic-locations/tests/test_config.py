"""Tests for anatomic_locations configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from anatomic_locations.config import AnatomicLocationSettings, ensure_anatomic_db, get_settings
from oidm_common.embeddings.config import EmbeddingProfileSpec


class TestAnatomicLocationSettings:
    """Tests for AnatomicLocationSettings class."""

    def test_default_values(self) -> None:
        """Test that settings have correct default values (without .env file influence)."""
        settings = AnatomicLocationSettings(_env_file=None)

        assert settings.db_path is None
        assert settings.remote_db_url is None
        assert settings.remote_db_hash is None
        assert settings.manifest_url == "https://anatomiclocationsdata.t3.storage.dev/manifest.json"
        assert settings.openai_api_key is None

    def test_environment_variable_loading_with_anatomic_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables with ANATOMIC_ prefix are loaded correctly."""
        monkeypatch.setenv("ANATOMIC_DB_PATH", "/custom/path/anatomic.duckdb")
        monkeypatch.setenv("ANATOMIC_REMOTE_DB_URL", "https://example.com/db.duckdb")
        monkeypatch.setenv("ANATOMIC_REMOTE_DB_HASH", "sha256:abc123")
        monkeypatch.setenv("ANATOMIC_MANIFEST_URL", "https://example.com/manifest.json")

        settings = AnatomicLocationSettings()

        assert settings.db_path == "/custom/path/anatomic.duckdb"
        assert settings.remote_db_url == "https://example.com/db.duckdb"
        assert settings.remote_db_hash == "sha256:abc123"
        assert settings.manifest_url == "https://example.com/manifest.json"

    def test_partial_environment_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that only specified environment variables override defaults."""
        monkeypatch.setenv("ANATOMIC_MANIFEST_URL", "https://custom.example.com/manifest.json")

        settings = AnatomicLocationSettings()

        assert settings.manifest_url == "https://custom.example.com/manifest.json"
        assert settings.db_path is None
        assert settings.remote_db_url is None
        assert settings.remote_db_hash is None

    # --- OpenAI API key fallback tests (AliasChoices) ---

    def test_openai_api_key_from_standard_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENAI_API_KEY is picked up when ANATOMIC_OPENAI_API_KEY is not set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-standard-key")

        settings = AnatomicLocationSettings()

        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-standard-key"

    def test_anatomic_api_key_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that ANATOMIC_OPENAI_API_KEY takes priority over OPENAI_API_KEY."""
        monkeypatch.setenv("ANATOMIC_OPENAI_API_KEY", "sk-anatomic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-standard-key")

        settings = AnatomicLocationSettings()

        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-anatomic-key"

    def test_openai_api_key_none_when_unset(self) -> None:
        """Test that openai_api_key is None when neither env var is set."""
        settings = AnatomicLocationSettings(_env_file=None)
        assert settings.openai_api_key is None

    def test_reads_dotenv_file(self, tmp_path: Path) -> None:
        """Test that settings reads from a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n")

        settings = AnatomicLocationSettings(_env_file=str(env_file))

        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-from-dotenv"

    def test_env_var_overrides_dotenv_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables take priority over .env file values."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        settings = AnatomicLocationSettings(_env_file=str(env_file))

        assert settings.openai_api_key is not None
        assert settings.openai_api_key.get_secret_value() == "sk-from-env"


class TestGetSettings:
    """Tests for get_settings() singleton function."""

    def test_singleton_behavior(self) -> None:
        """Test that get_settings() returns the same instance on multiple calls."""
        import anatomic_locations.config as config_module

        config_module._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_loads_from_environment_on_first_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that settings are loaded from environment on first get_settings() call."""
        import anatomic_locations.config as config_module

        config_module._settings = None

        monkeypatch.setenv("ANATOMIC_DB_PATH", "/env/test.duckdb")

        settings = get_settings()

        assert settings.db_path == "/env/test.duckdb"


class TestEnsureAnatomicDb:
    """Tests for ensure_anatomic_db() function."""

    def test_calls_ensure_db_file_with_correct_parameters(self, tmp_path: Path) -> None:
        """Test that ensure_anatomic_db() calls ensure_db_file with correct parameters."""
        import anatomic_locations.config as config_module

        config_module._settings = None

        with (
            patch("anatomic_locations.config.ensure_db_file") as mock_ensure_db_file,
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True),
        ):
            mock_ensure_db_file.return_value = tmp_path / "anatomic_locations.duckdb"

            result = ensure_anatomic_db()

            mock_ensure_db_file.assert_called_once_with(
                file_path=None,
                remote_url=None,
                remote_hash=None,
                manifest_key="anatomic_locations",
                manifest_url="https://anatomiclocationsdata.t3.storage.dev/manifest.json",
                app_name="anatomic-locations",
            )

            assert result == tmp_path / "anatomic_locations.duckdb"

    def test_uses_settings_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that ensure_anatomic_db() uses values from settings."""
        import anatomic_locations.config as config_module

        config_module._settings = None

        monkeypatch.setenv("ANATOMIC_DB_PATH", "custom.duckdb")
        monkeypatch.setenv("ANATOMIC_REMOTE_DB_URL", "https://custom.example.com/db.duckdb")
        monkeypatch.setenv("ANATOMIC_REMOTE_DB_HASH", "sha256:custom123")
        monkeypatch.setenv("ANATOMIC_MANIFEST_URL", "https://custom.example.com/manifest.json")
        monkeypatch.setenv("OPENAI_API_KEY", "")

        with patch("anatomic_locations.config.ensure_db_file") as mock_ensure_db_file:
            mock_ensure_db_file.return_value = tmp_path / "custom.duckdb"

            result = ensure_anatomic_db()

            mock_ensure_db_file.assert_called_once_with(
                file_path="custom.duckdb",
                remote_url="https://custom.example.com/db.duckdb",
                remote_hash="sha256:custom123",
                manifest_key="anatomic_locations",
                manifest_url="https://custom.example.com/manifest.json",
                app_name="anatomic-locations",
            )

            assert result == tmp_path / "custom.duckdb"

    def test_manifest_key_is_always_anatomic_locations(self, tmp_path: Path) -> None:
        """Test that manifest_key is always 'anatomic_locations' regardless of settings."""
        import anatomic_locations.config as config_module

        config_module._settings = None

        with (
            patch("anatomic_locations.config.ensure_db_file") as mock_ensure_db_file,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_ensure_db_file.return_value = tmp_path / "test.duckdb"

            ensure_anatomic_db()

            call_kwargs = mock_ensure_db_file.call_args.kwargs
            assert call_kwargs["manifest_key"] == "anatomic_locations"

    def test_app_name_is_always_anatomic_locations(self, tmp_path: Path) -> None:
        """Test that app_name is always 'anatomic-locations' regardless of settings."""
        import anatomic_locations.config as config_module

        config_module._settings = None

        with (
            patch("anatomic_locations.config.ensure_db_file") as mock_ensure_db_file,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_ensure_db_file.return_value = tmp_path / "test.duckdb"

            ensure_anatomic_db()

            call_kwargs = mock_ensure_db_file.call_args.kwargs
            assert call_kwargs["app_name"] == "anatomic-locations"

    def test_returns_path_from_ensure_db_file(self, tmp_path: Path) -> None:
        """Test that ensure_anatomic_db() returns the path from ensure_db_file."""
        import anatomic_locations.config as config_module

        config_module._settings = None

        expected_path = tmp_path / "returned.duckdb"

        with (
            patch("anatomic_locations.config.ensure_db_file") as mock_ensure_db_file,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_ensure_db_file.return_value = expected_path

            result = ensure_anatomic_db()

            assert result == expected_path

    def test_errors_immediately_for_openai_db_without_key(self, tmp_path: Path) -> None:
        """OpenAI-profile DB should fail fast when no OpenAI key is configured."""
        import anatomic_locations.config as config_module

        config_module._settings = None
        db_path = tmp_path / "anatomic_openai.duckdb"

        with (
            patch("anatomic_locations.config.ensure_db_file", return_value=db_path),
            patch(
                "anatomic_locations.config.read_embedding_profile_from_db",
                return_value=EmbeddingProfileSpec(provider="openai", model="text-embedding-3-small", dimensions=512),
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True),
            pytest.raises(config_module.ConfigurationError, match="uses OpenAI embeddings"),
        ):
            ensure_anatomic_db()

    def test_errors_immediately_for_non_openai_profile_db(self, tmp_path: Path) -> None:
        """Non-OpenAI profile DBs should fail fast."""
        import anatomic_locations.config as config_module

        config_module._settings = None
        db_path = tmp_path / "anatomic_other.duckdb"

        with (
            patch("anatomic_locations.config.ensure_db_file", return_value=db_path),
            patch(
                "anatomic_locations.config.read_embedding_profile_from_db",
                return_value=EmbeddingProfileSpec(provider="unsupported-provider", model="some-model", dimensions=256),
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            pytest.raises(config_module.ConfigurationError, match="supports only OpenAI embeddings"),
        ):
            ensure_anatomic_db()
