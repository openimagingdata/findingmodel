"""Tests for configuration validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from findingmodel.config import FindingModelConfig, ensure_index_db, get_settings


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
        assert config.openai_embedding_model == "text-embedding-3-small"
        assert config.openai_embedding_dimensions == 512

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

    # --- Legacy env var alias tests ---

    def test_legacy_duckdb_index_path_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that legacy DUCKDB_INDEX_PATH is still accepted."""
        monkeypatch.setenv("DUCKDB_INDEX_PATH", "/legacy/path.duckdb")

        config = FindingModelConfig(_env_file=None)

        assert config.db_path == "/legacy/path.duckdb"

    def test_findingmodel_db_path_takes_priority_over_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that FINDINGMODEL_DB_PATH takes priority over DUCKDB_INDEX_PATH."""
        monkeypatch.setenv("FINDINGMODEL_DB_PATH", "/new/path.duckdb")
        monkeypatch.setenv("DUCKDB_INDEX_PATH", "/legacy/path.duckdb")

        config = FindingModelConfig(_env_file=None)

        assert config.db_path == "/new/path.duckdb"

    def test_legacy_remote_index_db_url_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that legacy REMOTE_INDEX_DB_URL and REMOTE_INDEX_DB_HASH are still accepted."""
        monkeypatch.setenv("REMOTE_INDEX_DB_URL", "https://legacy.example.com/db.duckdb")
        monkeypatch.setenv("REMOTE_INDEX_DB_HASH", "sha256:legacy123")

        config = FindingModelConfig(_env_file=None)

        assert config.remote_db_url == "https://legacy.example.com/db.duckdb"
        assert config.remote_db_hash == "sha256:legacy123"

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

    def test_embedding_model_from_standard_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENAI_EMBEDDING_MODEL is picked up as fallback."""
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        config = FindingModelConfig()

        assert config.openai_embedding_model == "text-embedding-3-large"

    def test_findingmodel_embedding_model_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that FINDINGMODEL_OPENAI_EMBEDDING_MODEL takes priority over OPENAI_EMBEDDING_MODEL."""
        monkeypatch.setenv("FINDINGMODEL_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        config = FindingModelConfig()

        assert config.openai_embedding_model == "text-embedding-ada-002"

    def test_embedding_dimensions_from_standard_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OPENAI_EMBEDDING_DIMENSIONS is picked up as fallback."""
        monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")

        config = FindingModelConfig()

        assert config.openai_embedding_dimensions == 1536

    def test_findingmodel_embedding_dimensions_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that FINDINGMODEL_OPENAI_EMBEDDING_DIMENSIONS takes priority."""
        monkeypatch.setenv("FINDINGMODEL_OPENAI_EMBEDDING_DIMENSIONS", "256")
        monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")

        config = FindingModelConfig()

        assert config.openai_embedding_dimensions == 256

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
            patch.dict(os.environ, {}, clear=True),
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
