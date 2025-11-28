"""Tests for Pooch integration and database file downloads."""

from pathlib import Path
from unittest.mock import patch

import pytest

from findingmodel.config import ensure_db_file


class TestEnsureDbFileMocked:
    """Mock-based tests for ensure_db_file function."""

    def test_existing_file_with_no_hash_uses_file_directly(self, tmp_path: Path) -> None:
        """Test that existing file with no URL/hash is used directly without any verification."""
        data_dir = tmp_path
        test_file = data_dir / "test.duckdb"
        test_file.write_text("existing data")

        with (
            patch("findingmodel.config.user_data_dir") as mock_user_data_dir,
            patch("pooch.retrieve") as mock_pooch_retrieve,
            patch("pooch.file_hash") as mock_file_hash,
        ):
            mock_user_data_dir.return_value = str(data_dir)

            result = ensure_db_file(
                "test.duckdb",
                None,  # No URL
                None,  # No hash
                manifest_key="test",
            )

            # Should NOT call pooch at all
            mock_pooch_retrieve.assert_not_called()
            mock_file_hash.assert_not_called()

            # Should return the existing file
            assert result == test_file

    def test_existing_file_with_matching_hash_skips_download(self, tmp_path: Path) -> None:
        """Test that file with matching hash is used without downloading."""
        import pooch

        data_dir = tmp_path
        test_file = data_dir / "test.duckdb"
        test_content = "test content for hashing"
        test_file.write_text(test_content)

        # Compute actual hash of the file
        actual_hash = pooch.file_hash(str(test_file), alg="sha256")

        with (
            patch("findingmodel.config.user_data_dir") as mock_user_data_dir,
            patch("pooch.retrieve") as mock_pooch_retrieve,
        ):
            mock_user_data_dir.return_value = str(data_dir)

            result = ensure_db_file(
                "test.duckdb",
                "http://example.com/test.duckdb",
                f"sha256:{actual_hash}",  # Use actual hash so it matches
                manifest_key="test",
            )

            # Should NOT download since hash matches
            mock_pooch_retrieve.assert_not_called()

            # Should return the existing file
            assert result == test_file

    def test_existing_file_with_mismatched_hash_redownloads(self, tmp_path: Path) -> None:
        """Test that file with mismatched hash triggers re-download."""
        data_dir = tmp_path
        test_file = data_dir / "test.duckdb"
        test_file.write_text("existing data")

        with (
            patch("findingmodel.config.user_data_dir") as mock_user_data_dir,
            patch("pooch.retrieve") as mock_pooch_retrieve,
        ):
            mock_user_data_dir.return_value = str(data_dir)
            mock_pooch_retrieve.return_value = str(test_file)

            result = ensure_db_file(
                None,  # Use managed download mode
                "http://example.com/test.duckdb",
                "sha256:wronghash123",  # Hash won't match
                manifest_key="test",
            )

            # Should download because hash doesn't match
            mock_pooch_retrieve.assert_called_once_with(
                url="http://example.com/test.duckdb",
                known_hash="sha256:wronghash123",
                path=data_dir,
                fname="test.duckdb",
            )

            assert result == test_file

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
                    None,  # Use managed download mode
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

            with pytest.raises(ConfigurationError, match="Cannot fetch manifest"):
                ensure_db_file(
                    None,  # Use managed download mode
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
                    None,  # Use managed download mode
                    "http://example.com/test.duckdb",
                    "sha256:hash",
                    manifest_key="test",
                )

                # Verify directory exists and file was created
                assert data_dir.exists()
                assert result == downloaded_file


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
                None,  # Use managed download mode
                url,
                file_hash,
                manifest_key="test",
            )

            # Should return valid path
            assert result.exists()
            # In managed mode (file_path=None), filename is derived from manifest_key
            assert result.name == "test.duckdb"
            assert result.parent == data_dir

            # Verify it's a valid DuckDB file by checking it has some content
            assert result.stat().st_size > 0

            # Second call - should reuse cached file without re-download
            original_mtime = result.stat().st_mtime
            result2 = ensure_db_file(
                None,  # Same as first call - managed download mode
                url,
                file_hash,
                manifest_key="test",
            )

            # Should return same file
            assert result2 == result
            assert result2.exists()

            # File should not have been modified (no re-download)
            assert result2.stat().st_mtime == original_mtime


class TestWrapperFunctions:
    """Tests for ensure_index_db() and ensure_anatomic_db() wrappers."""

    def test_ensure_index_db_calls_ensure_db_file_correctly(self, tmp_path: Path) -> None:
        """Test that ensure_index_db() calls ensure_db_file() with correct parameters."""
        from unittest.mock import patch

        from findingmodel.config import ensure_index_db

        with (
            patch("findingmodel.config.ensure_db_file") as mock_ensure,
            patch("findingmodel.config.settings") as mock_settings,
        ):
            mock_settings.duckdb_index_path = "test.duckdb"
            mock_settings.remote_index_db_url = "http://example.com/index.duckdb"
            mock_settings.remote_index_db_hash = "sha256:abc123"
            mock_ensure.return_value = tmp_path / "test.duckdb"

            result = ensure_index_db()

            # Verify ensure_db_file was called with correct parameters
            mock_ensure.assert_called_once_with(
                "test.duckdb",
                "http://example.com/index.duckdb",
                "sha256:abc123",
                manifest_key="finding_models",
            )
            assert result == tmp_path / "test.duckdb"

    def test_ensure_anatomic_db_calls_ensure_db_file_correctly(self, tmp_path: Path) -> None:
        """Test that ensure_anatomic_db() calls ensure_db_file() with correct parameters."""
        from unittest.mock import patch

        from findingmodel.config import ensure_anatomic_db

        with (
            patch("findingmodel.config.ensure_db_file") as mock_ensure,
            patch("findingmodel.config.settings") as mock_settings,
        ):
            mock_settings.duckdb_anatomic_path = "anatomic.duckdb"
            mock_settings.remote_anatomic_db_url = "http://example.com/anatomic.duckdb"
            mock_settings.remote_anatomic_db_hash = "sha256:def456"
            mock_ensure.return_value = tmp_path / "anatomic.duckdb"

            result = ensure_anatomic_db()

            # Verify ensure_db_file was called with correct parameters
            mock_ensure.assert_called_once_with(
                "anatomic.duckdb",
                "http://example.com/anatomic.duckdb",
                "sha256:def456",
                manifest_key="anatomic_locations",
            )
            assert result == tmp_path / "anatomic.duckdb"
