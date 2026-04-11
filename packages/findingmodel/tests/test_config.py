"""Tests for configuration validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from findingmodel.config import FindingModelConfig, ensure_index_db, get_settings
from oidm_common.embeddings.config import EmbeddingProfileSpec


class TestFindingModelConfig:
    """Tests for FindingModelConfig class."""

    def test_default_values(self) -> None:
        """Test that settings have correct default values (without .env file influence)."""
        config = FindingModelConfig(_env_file=None)

        assert config.db_path is None
        assert config.remote_db_url is None
        assert config.remote_db_hash is None
        assert config.manifest_url == "https://findingmodelsdata.t3.storage.dev/manifest.json"
        assert config.openai_api_key is None

    def test_environment_variable_loading_with_findingmodel_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables with FINDINGMODEL_ prefix are loaded correctly."""
        monkeypatch.setenv("FINDINGMODEL_DB_PATH", "/custom/path/finding_models.duckdb")
        monkeypatch.setenv("FINDINGMODEL_REMOTE_DB_URL", "https://example.com/index.duckdb")
        monkeypatch.setenv("FINDINGMODEL_REMOTE_DB_HASH", "sha256:def456")
        monkeypatch.setenv("FINDINGMODEL_MANIFEST_URL", "https://example.com/manifest.json")

        config = FindingModelConfig()

        assert config.db_path == "/custom/path/finding_models.duckdb"
        assert config.remote_db_url == "https://example.com/index.duckdb"
        assert config.remote_db_hash == "sha256:def456"
        assert config.manifest_url == "https://example.com/manifest.json"

    def test_obsolete_path_aliases_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Obsolete path aliases should not be recognized."""
        monkeypatch.setenv("DUCKDB_INDEX_PATH", "/obsolete/path.duckdb")
        monkeypatch.setenv("REMOTE_INDEX_DB_URL", "https://obsolete.example.com/db.duckdb")
        monkeypatch.setenv("REMOTE_INDEX_DB_HASH", "sha256:obsolete")

        config = FindingModelConfig(_env_file=None)

        assert config.db_path is None
        assert config.remote_db_url is None
        assert config.remote_db_hash is None

    def test_remote_config_with_both_url_and_hash_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that providing both URL and hash succeeds."""
        monkeypatch.setenv("FINDINGMODEL_REMOTE_DB_URL", "https://example.com/index.duckdb")
        monkeypatch.setenv("FINDINGMODEL_REMOTE_DB_HASH", "sha256:def456")

        config = FindingModelConfig(_env_file=None)
        assert config.remote_db_url == "https://example.com/index.duckdb"
        assert config.remote_db_hash == "sha256:def456"

    def test_remote_config_with_neither_succeeds(self) -> None:
        """Test that providing neither URL nor hash succeeds (uses manifest)."""
        config = FindingModelConfig(_env_file=None)
        assert config.remote_db_url is None
        assert config.remote_db_hash is None

    # --- OpenAI API key fallback tests (AliasChoices) ---

    def test_openai_api_key_from_standard_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENAI_API_KEY is picked up when FINDINGMODEL_OPENAI_API_KEY is not set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-standard-key")

        config = FindingModelConfig()

        assert config.openai_api_key is not None
        assert config.openai_api_key.get_secret_value() == "sk-standard-key"

    def test_findingmodel_api_key_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that FINDINGMODEL_OPENAI_API_KEY takes priority over OPENAI_API_KEY."""
        monkeypatch.setenv("FINDINGMODEL_OPENAI_API_KEY", "sk-findingmodel-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-standard-key")

        config = FindingModelConfig()

        assert config.openai_api_key is not None
        assert config.openai_api_key.get_secret_value() == "sk-findingmodel-key"

    def test_openai_api_key_none_when_unset(self) -> None:
        """Test that openai_api_key is None when neither env var is set."""
        config = FindingModelConfig(_env_file=None)
        assert config.openai_api_key is None

    def test_reads_dotenv_file(self, tmp_path: Path) -> None:
        """Test that settings reads from a .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n")

        config = FindingModelConfig(_env_file=str(env_file))

        assert config.openai_api_key is not None
        assert config.openai_api_key.get_secret_value() == "sk-from-dotenv"

    def test_env_var_overrides_dotenv_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables take priority over .env file values."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        config = FindingModelConfig(_env_file=str(env_file))

        assert config.openai_api_key is not None
        assert config.openai_api_key.get_secret_value() == "sk-from-env"


class TestGetSettings:
    """Tests for get_settings() singleton function."""

    def test_singleton_behavior(self) -> None:
        """Test that get_settings() returns the same instance on multiple calls."""
        import findingmodel.config as config_module

        config_module._settings = None

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_loads_from_environment_on_first_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that settings are loaded from environment on first get_settings() call."""
        import findingmodel.config as config_module

        config_module._settings = None

        monkeypatch.setenv("FINDINGMODEL_DB_PATH", "/env/test.duckdb")

        settings = get_settings()

        assert settings.db_path == "/env/test.duckdb"


class TestEnsureIndexDb:
    """Tests for ensure_index_db() function."""

    def test_calls_ensure_db_file_with_correct_parameters(self, tmp_path: Path) -> None:
        """Test that ensure_index_db() calls oidm_ensure_db_file with correct parameters."""
        import findingmodel.config as config_module

        config_module._settings = None

        with (
            patch("findingmodel.config.oidm_ensure_db_file") as mock_ensure,
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True),
        ):
            mock_ensure.return_value = tmp_path / "finding_models.duckdb"

            result = ensure_index_db()

            mock_ensure.assert_called_once_with(
                file_path=None,
                remote_url=None,
                remote_hash=None,
                manifest_key="finding_models",
                manifest_url="https://findingmodelsdata.t3.storage.dev/manifest.json",
                app_name="findingmodel",
            )
            assert result == tmp_path / "finding_models.duckdb"

    def test_uses_settings_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that ensure_index_db() uses values from settings."""
        import findingmodel.config as config_module

        config_module._settings = None

        monkeypatch.setenv("FINDINGMODEL_DB_PATH", "custom.duckdb")
        monkeypatch.setenv("FINDINGMODEL_REMOTE_DB_URL", "https://custom.example.com/db.duckdb")
        monkeypatch.setenv("FINDINGMODEL_REMOTE_DB_HASH", "sha256:custom123")
        monkeypatch.setenv("FINDINGMODEL_MANIFEST_URL", "https://custom.example.com/manifest.json")
        monkeypatch.setenv("OPENAI_API_KEY", "")

        with patch("findingmodel.config.oidm_ensure_db_file") as mock_ensure:
            mock_ensure.return_value = tmp_path / "custom.duckdb"

            result = ensure_index_db()

            mock_ensure.assert_called_once_with(
                file_path="custom.duckdb",
                remote_url="https://custom.example.com/db.duckdb",
                remote_hash="sha256:custom123",
                manifest_key="finding_models",
                manifest_url="https://custom.example.com/manifest.json",
                app_name="findingmodel",
            )
            assert result == tmp_path / "custom.duckdb"

    def test_errors_immediately_for_openai_db_without_key(self, tmp_path: Path) -> None:
        """OpenAI-profile DB should fail fast when no OpenAI key is configured."""
        import findingmodel.config as config_module

        config_module._settings = None
        db_path = tmp_path / "fm_openai.duckdb"

        with (
            patch("findingmodel.config.oidm_ensure_db_file", return_value=db_path),
            patch(
                "findingmodel.config.read_embedding_profile_from_db",
                return_value=EmbeddingProfileSpec(provider="openai", model="text-embedding-3-small", dimensions=512),
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True),
            pytest.raises(config_module.ConfigurationError, match="uses OpenAI embeddings"),
        ):
            ensure_index_db()

    def test_errors_immediately_for_non_openai_profile_db(self, tmp_path: Path) -> None:
        """Non-OpenAI profile DBs should fail fast."""
        import findingmodel.config as config_module

        config_module._settings = None
        db_path = tmp_path / "fm_other.duckdb"

        with (
            patch("findingmodel.config.oidm_ensure_db_file", return_value=db_path),
            patch(
                "findingmodel.config.read_embedding_profile_from_db",
                return_value=EmbeddingProfileSpec(provider="unsupported-provider", model="some-model", dimensions=256),
            ),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True),
            pytest.raises(config_module.ConfigurationError, match="supports only OpenAI embeddings"),
        ):
            ensure_index_db()
