"""Tests for configuration validation."""

import pytest
from _pytest.monkeypatch import MonkeyPatch
from pydantic import ValidationError


def test_remote_anatomic_url_without_hash_raises_error(monkeypatch: MonkeyPatch) -> None:
    """Test that providing only URL without hash for anatomic DB raises validation error."""
    monkeypatch.setenv("REMOTE_ANATOMIC_DB_URL", "https://example.com/anatomic.duckdb")
    # No hash set

    from findingmodel.config import FindingModelConfig

    with pytest.raises(ValidationError, match="Must provide both REMOTE_ANATOMIC_DB_URL and REMOTE_ANATOMIC_DB_HASH"):
        FindingModelConfig()


def test_remote_anatomic_hash_without_url_raises_error(monkeypatch: MonkeyPatch) -> None:
    """Test that providing only hash without URL for anatomic DB raises validation error."""
    monkeypatch.setenv("REMOTE_ANATOMIC_DB_HASH", "sha256:abc123")
    # No URL set

    from findingmodel.config import FindingModelConfig

    with pytest.raises(ValidationError, match="Must provide both REMOTE_ANATOMIC_DB_URL and REMOTE_ANATOMIC_DB_HASH"):
        FindingModelConfig()


def test_remote_index_url_without_hash_raises_error(monkeypatch: MonkeyPatch) -> None:
    """Test that providing only URL without hash for index DB raises validation error."""
    monkeypatch.setenv("REMOTE_INDEX_DB_URL", "https://example.com/index.duckdb")
    # No hash set

    from findingmodel.config import FindingModelConfig

    with pytest.raises(ValidationError, match="Must provide both REMOTE_INDEX_DB_URL and REMOTE_INDEX_DB_HASH"):
        FindingModelConfig()


def test_remote_index_hash_without_url_raises_error(monkeypatch: MonkeyPatch) -> None:
    """Test that providing only hash without URL for index DB raises validation error."""
    monkeypatch.setenv("REMOTE_INDEX_DB_HASH", "sha256:def456")
    # No URL set

    from findingmodel.config import FindingModelConfig

    with pytest.raises(ValidationError, match="Must provide both REMOTE_INDEX_DB_URL and REMOTE_INDEX_DB_HASH"):
        FindingModelConfig()


def test_remote_config_with_both_url_and_hash_succeeds(monkeypatch: MonkeyPatch) -> None:
    """Test that providing both URL and hash succeeds."""
    monkeypatch.setenv("REMOTE_ANATOMIC_DB_URL", "https://example.com/anatomic.duckdb")
    monkeypatch.setenv("REMOTE_ANATOMIC_DB_HASH", "sha256:abc123")
    monkeypatch.setenv("REMOTE_INDEX_DB_URL", "https://example.com/index.duckdb")
    monkeypatch.setenv("REMOTE_INDEX_DB_HASH", "sha256:def456")

    from findingmodel.config import FindingModelConfig

    config = FindingModelConfig()
    assert config.remote_anatomic_db_url == "https://example.com/anatomic.duckdb"
    assert config.remote_anatomic_db_hash == "sha256:abc123"
    assert config.remote_index_db_url == "https://example.com/index.duckdb"
    assert config.remote_index_db_hash == "sha256:def456"


def test_remote_config_with_neither_succeeds(monkeypatch: MonkeyPatch) -> None:
    """Test that providing neither URL nor hash succeeds (uses manifest)."""
    # Ensure no remote config is set
    monkeypatch.delenv("REMOTE_ANATOMIC_DB_URL", raising=False)
    monkeypatch.delenv("REMOTE_ANATOMIC_DB_HASH", raising=False)
    monkeypatch.delenv("REMOTE_INDEX_DB_URL", raising=False)
    monkeypatch.delenv("REMOTE_INDEX_DB_HASH", raising=False)

    from findingmodel.config import FindingModelConfig

    config = FindingModelConfig()
    assert config.remote_anatomic_db_url is None
    assert config.remote_anatomic_db_hash is None
    assert config.remote_index_db_url is None
    assert config.remote_index_db_hash is None


def test_check_ready_for_tavily_with_key_succeeds() -> None:
    """Test that check_ready_for_tavily succeeds when API key is set."""
    from findingmodel.config import FindingModelConfig
    from pydantic import SecretStr

    config = FindingModelConfig(tavily_api_key=SecretStr("test-tavily-key"))
    assert config.check_ready_for_tavily() is True


def test_check_ready_for_tavily_without_key_raises_error() -> None:
    """Test that check_ready_for_tavily raises ConfigurationError when API key missing."""
    from findingmodel.config import ConfigurationError, FindingModelConfig
    from pydantic import SecretStr

    config = FindingModelConfig(tavily_api_key=SecretStr(""))
    with pytest.raises(ConfigurationError, match="Tavily API key is not set"):
        config.check_ready_for_tavily()
