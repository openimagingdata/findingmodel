"""Tests for Pooch integration and database file downloads."""

from pathlib import Path
from unittest.mock import patch

import pytest

from findingmodel.config import ensure_db_file


class TestEnsureDbFileMocked:
    """Mock-based tests for ensure_db_file function."""

    def test_calls_pooch_to_verify_hash(self, tmp_path: Path) -> None:
        """Test that Pooch is called even for existing files to verify hash."""
        # Create a fake existing file
        data_dir = tmp_path
        test_file = data_dir / "test.duckdb"
        test_file.write_text("existing data")

        with patch("findingmodel.config.user_data_dir") as mock_user_data_dir:
            # Mock user_data_dir to return our temp directory
            mock_user_data_dir.return_value = str(data_dir)
            with patch("pooch.retrieve") as mock_pooch_retrieve:
                # Mock pooch to return the existing file path
                mock_pooch_retrieve.return_value = str(test_file)

                result = ensure_db_file(
                    "test.duckdb",
                    "http://example.com/test.duckdb",
                    "sha256:abc123",
                    manifest_key="test",
                )

                # Should call pooch to verify hash (even for existing files)
                mock_pooch_retrieve.assert_called_once_with(
                    url="http://example.com/test.duckdb",
                    known_hash="sha256:abc123",
                    path=data_dir,
                    fname="test.duckdb",
                )

                # Should return the file
                assert result == test_file
                assert result.exists()

    def test_downloads_when_file_missing_and_url_configured(self, tmp_path: Path) -> None:
        """Test that missing files trigger Pooch download when URL is configured."""
        data_dir = tmp_path
        # Don't create file - ensure_db_file should do that
        downloaded_file = data_dir / "test.duckdb"

        with patch("findingmodel.config.user_data_dir") as mock_user_data_dir:
            mock_user_data_dir.return_value = str(data_dir)
            with patch("pooch.retrieve") as mock_pooch_retrieve:
                # Mock pooch to create the file when called
                def create_file_and_return(*args: object, **kwargs: object) -> str:
                    downloaded_file.write_text("downloaded data")
                    return str(downloaded_file)

                mock_pooch_retrieve.side_effect = create_file_and_return

                result = ensure_db_file(
                    "test.duckdb",
                    "http://example.com/test.duckdb",
                    "sha256:abc123def456",
                    manifest_key="test",
                )

                # Should call pooch with correct parameters
                mock_pooch_retrieve.assert_called_once_with(
                    url="http://example.com/test.duckdb",
                    known_hash="sha256:abc123def456",
                    path=data_dir,
                    fname="test.duckdb",
                )

                # Should return downloaded file path
                assert result == downloaded_file
                assert result.exists()

    def test_returns_path_when_no_url_configured(self, tmp_path: Path) -> None:
        """Test that function raises error when manifest fails and no URL configured."""
        data_dir = tmp_path

        with (
            patch("findingmodel.config.user_data_dir") as mock_user_data_dir,
            patch("pooch.retrieve") as mock_pooch_retrieve,
            patch("findingmodel.config.fetch_manifest", side_effect=Exception("Network error")),
        ):
            mock_user_data_dir.return_value = str(data_dir)
            # Should raise ConfigurationError when manifest fails and no fallback config
            from findingmodel.config import ConfigurationError

            with pytest.raises(ConfigurationError, match="manifest fetch failed"):
                ensure_db_file(
                    "missing.duckdb",
                    None,
                    None,
                    manifest_key="test",
                )

            # Should not call pooch
            mock_pooch_retrieve.assert_not_called()

    def test_creates_parent_directory_before_download(self, tmp_path: Path) -> None:
        """Test that parent directory is created before download."""
        # Use temp directory
        data_dir = tmp_path
        downloaded_file = data_dir / "test.duckdb"

        with patch("findingmodel.config.user_data_dir") as mock_user_data_dir:
            mock_user_data_dir.return_value = str(data_dir)
            with patch("pooch.retrieve") as mock_pooch_retrieve:
                # Create the file as part of the mock
                def create_file_and_return(*args: object, **kwargs: object) -> str:
                    downloaded_file.write_text("data")
                    return str(downloaded_file)

                mock_pooch_retrieve.side_effect = create_file_and_return

                result = ensure_db_file(
                    "test.duckdb",
                    "http://example.com/test.duckdb",
                    "sha256:hash",
                    manifest_key="test",
                )

                # Verify directory exists and file was created
                assert data_dir.exists()
                assert result == downloaded_file

    def test_handles_only_url_without_hash(self, tmp_path: Path) -> None:
        """Test that function raises error when manifest fails and only URL provided without hash."""
        data_dir = tmp_path

        with (
            patch("findingmodel.config.user_data_dir") as mock_user_data_dir,
            patch("pooch.retrieve") as mock_pooch_retrieve,
            patch("findingmodel.config.fetch_manifest", side_effect=Exception("Network error")),
        ):
            mock_user_data_dir.return_value = str(data_dir)
            from findingmodel.config import ConfigurationError

            # Should raise ConfigurationError (hash required for fallback)
            with pytest.raises(ConfigurationError, match="manifest fetch failed"):
                ensure_db_file(
                    "test.duckdb",
                    "http://example.com/test.duckdb",
                    None,
                    manifest_key="test",
                )

            # Should not download (both URL and hash required)
            mock_pooch_retrieve.assert_not_called()

    def test_handles_only_hash_without_url(self, tmp_path: Path) -> None:
        """Test that function raises error when manifest fails and only hash provided without URL."""
        data_dir = tmp_path

        with (
            patch("findingmodel.config.user_data_dir") as mock_user_data_dir,
            patch("pooch.retrieve") as mock_pooch_retrieve,
            patch("findingmodel.config.fetch_manifest", side_effect=Exception("Network error")),
        ):
            mock_user_data_dir.return_value = str(data_dir)
            from findingmodel.config import ConfigurationError

            # Should raise ConfigurationError (URL required for fallback)
            with pytest.raises(ConfigurationError, match="manifest fetch failed"):
                ensure_db_file(
                    "test.duckdb",
                    None,
                    "sha256:abc123",
                    manifest_key="test",
                )

            # Should not download (both URL and hash required)
            mock_pooch_retrieve.assert_not_called()


@pytest.mark.callout
class TestEnsureDbFileRealDownload:
    """Real download tests using actual remote file."""

    def test_downloads_from_real_url(self, tmp_path: Path) -> None:
        """Test actual download from remote URL."""
        url = "https://findingmodelsdata.t3.storage.dev/findingmodels-test.duckdb"
        file_hash = "sha256:0819dc80984253d5954787c137818cad43d54870e184a8af096b840893075fef"

        data_dir = tmp_path

        with patch("findingmodel.config.user_data_dir") as mock_user_data_dir:
            mock_user_data_dir.return_value = str(data_dir)

            # First call - should download
            result = ensure_db_file(
                "findingmodels-test.duckdb",
                url,
                file_hash,
                manifest_key="test",
            )

            # Should return valid path
            assert result.exists()
            assert result.name == "findingmodels-test.duckdb"
            assert result.parent == data_dir

            # Verify it's a valid DuckDB file by checking it has some content
            assert result.stat().st_size > 0

            # Second call - should reuse cached file without re-download
            original_mtime = result.stat().st_mtime
            result2 = ensure_db_file(
                "findingmodels-test.duckdb",
                url,
                file_hash,
                manifest_key="test",
            )

            # Should return same file
            assert result2 == result
            assert result2.exists()

            # File should not have been modified (no re-download)
            assert result2.stat().st_mtime == original_mtime
