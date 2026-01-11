"""Tests for S3 module (mocked)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from oidm_maintenance.config import MaintenanceSettings
from oidm_maintenance.s3 import (
    create_s3_client,
    load_manifest_from_s3,
    update_manifest_entry,
    upload_file_to_s3,
)
from pydantic import SecretStr


def test_create_s3_client_with_credentials() -> None:
    """Test S3 client creation with credentials."""
    settings = MaintenanceSettings(
        aws_access_key_id=SecretStr("test-key"),
        aws_secret_access_key=SecretStr("test-secret"),
    )

    with patch("boto3.client") as mock_client:
        create_s3_client(settings)

        mock_client.assert_called_once()
        call_args = mock_client.call_args.args
        call_kwargs = mock_client.call_args.kwargs
        assert call_args[0] == "s3"
        assert call_kwargs["endpoint_url"] == settings.s3_endpoint_url
        assert call_kwargs["aws_access_key_id"] == "test-key"
        assert call_kwargs["aws_secret_access_key"] == "test-secret"


def test_create_s3_client_without_credentials() -> None:
    """Test that S3 client creation raises ValueError without credentials."""
    settings = MaintenanceSettings()

    with pytest.raises(ValueError, match="AWS credentials required"):
        create_s3_client(settings)


def test_upload_file_to_s3(tmp_path: Path) -> None:
    """Test file upload to S3."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    mock_client = MagicMock()
    mock_client.meta.endpoint_url = "https://fly.storage.tigris.dev"

    url = upload_file_to_s3(mock_client, "mybucket", "data/test.txt", test_file)

    mock_client.upload_file.assert_called_once_with(str(test_file), "mybucket", "data/test.txt")
    assert url == "https://mybucket.fly.storage.tigris.dev/data/test.txt"


def test_upload_file_to_s3_missing_file(tmp_path: Path) -> None:
    """Test that upload raises FileNotFoundError for missing file."""
    missing_file = tmp_path / "missing.txt"
    mock_client = MagicMock()

    with pytest.raises(FileNotFoundError, match="File not found"):
        upload_file_to_s3(mock_client, "mybucket", "data/test.txt", missing_file)


def test_load_manifest_from_s3_existing(tmp_path: Path) -> None:
    """Test loading existing manifest from S3."""
    manifest_data = {
        "manifest_version": "1.0",
        "generated_at": "2025-01-11T12:00:00Z",
        "databases": {"test_db": {"version": "1.0"}},
    }

    mock_client = MagicMock()

    # Mock download_file to write the manifest JSON
    def mock_download(bucket: str, key: str, local_path: str) -> None:
        Path(local_path).write_text(json.dumps(manifest_data))

    mock_client.download_file = mock_download

    result = load_manifest_from_s3(mock_client, "mybucket", "manifest.json")

    assert result == manifest_data


def test_load_manifest_from_s3_not_found() -> None:
    """Test loading manifest returns empty structure when not found."""
    from typing import Any

    from botocore.exceptions import ClientError

    mock_client = MagicMock()

    # Mock ClientError for NoSuchKey
    error_response: Any = {"Error": {"Code": "NoSuchKey"}}
    mock_client.download_file.side_effect = ClientError(error_response, "GetObject")

    result = load_manifest_from_s3(mock_client, "mybucket", "manifest.json")

    assert result["manifest_version"] == "1.0"
    assert "databases" in result
    assert result["databases"] == {}


def test_update_manifest_entry() -> None:
    """Test updating manifest entry."""
    manifest = {
        "manifest_version": "1.0",
        "generated_at": "2025-01-11T12:00:00Z",
        "databases": {
            "existing_db": {"version": "1.0"},
        },
    }

    entry = {
        "version": "2.0",
        "url": "https://example.com/test.duckdb",
        "hash": "sha256:abc123",
        "size_bytes": 1024,
        "record_count": 100,
        "description": "Test database",
    }

    updated = update_manifest_entry(manifest, "test_db", entry)

    # Original manifest unchanged
    assert "test_db" not in manifest["databases"]

    # New manifest has updated entry
    assert "test_db" in updated["databases"]
    assert updated["databases"]["test_db"] == entry
    assert updated["databases"]["existing_db"] == {"version": "1.0"}
    assert updated["manifest_version"] == "1.0"
    assert updated["generated_at"] != manifest["generated_at"]  # Timestamp refreshed
