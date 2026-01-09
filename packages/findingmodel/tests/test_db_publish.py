"""Tests for db_publish module sanity check functionality."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from findingmodel.db_publish import (
    DatabaseStats,
    ManifestDatabaseEntry,
    ManifestUpdateInfo,
    PublishManifest,
    SanityCheckResult,
    get_complete_model_json,
    get_database_stats,
    get_sample_oifm_ids,
    update_and_publish_manifest,
)
from findingmodel.finding_model import FindingModelFull


def test_get_database_stats_nonexistent_file(tmp_path: Path) -> None:
    """Test that get_database_stats raises FileNotFoundError for missing database."""
    nonexistent_db = tmp_path / "nonexistent.duckdb"
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        get_database_stats(nonexistent_db)


def test_get_sample_oifm_ids_nonexistent_file(tmp_path: Path) -> None:
    """Test that get_sample_oifm_ids raises FileNotFoundError for missing database."""
    nonexistent_db = tmp_path / "nonexistent.duckdb"
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        get_sample_oifm_ids(nonexistent_db)


def test_get_complete_model_json_nonexistent_file(tmp_path: Path) -> None:
    """Test that get_complete_model_json raises FileNotFoundError for missing database."""
    nonexistent_db = tmp_path / "nonexistent.duckdb"
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        get_complete_model_json(nonexistent_db, "OIFM_TEST_000001")


def test_database_stats_dataclass() -> None:
    """Test DatabaseStats dataclass construction."""
    stats = DatabaseStats(
        record_count=100,
        size_bytes=1024,
        database_path=Path("/test/db.duckdb"),
    )
    assert stats.record_count == 100
    assert stats.size_bytes == 1024
    assert stats.database_path == Path("/test/db.duckdb")


def test_sanity_check_result_dataclass() -> None:
    """Test SanityCheckResult dataclass construction."""
    stats = DatabaseStats(
        record_count=100,
        size_bytes=1024,
        database_path=Path("/test/db.duckdb"),
    )
    result = SanityCheckResult(
        stats=stats,
        sample_ids=["OIFM_TEST_000001", "OIFM_TEST_000002"],
        model_preview='{"oifm_id": "OIFM_TEST_000001"}',
    )
    assert result.stats == stats
    assert len(result.sample_ids) == 2
    assert "OIFM_TEST_000001" in result.model_preview


# Phase 3 tests: Manifest Update and Publishing


def test_manifest_update_info_dataclass() -> None:
    """Test ManifestUpdateInfo dataclass construction."""
    info = ManifestUpdateInfo(
        database_key="finding_models",
        version="2025-11-17",
        url="https://example.com/db.duckdb",
        hash_value="sha256:abc123",
        size_bytes=52428800,
        record_count=1234,
        description="Test database",
    )
    assert info.database_key == "finding_models"
    assert info.version == "2025-11-17"
    assert info.url == "https://example.com/db.duckdb"
    assert info.hash_value == "sha256:abc123"
    assert info.size_bytes == 52428800
    assert info.record_count == 1234
    assert info.description == "Test database"


@patch("findingmodel.db_publish.load_manifest_from_s3")
@patch("findingmodel.db_publish.update_manifest_entry")
def test_update_and_publish_manifest_success(
    mock_update_entry: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Test successful manifest update and publish workflow."""
    # Setup mocks
    mock_s3_client = MagicMock()
    mock_s3_client.meta.endpoint_url = "https://t3.storage.dev"
    mock_config = MagicMock()
    mock_config.s3_bucket = "test-bucket"
    mock_config.manifest_backup_prefix = "manifests/archive/"

    # Mock existing manifest
    existing_manifest = PublishManifest(
        manifest_version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        databases={},
    )
    mock_load.return_value = existing_manifest

    # Mock updated manifest
    mock_db_entry = ManifestDatabaseEntry(
        version="2025-11-17",
        url="https://example.com/db.duckdb",
        hash="sha256:abc123",
        size_bytes=52428800,
        record_count=1234,
        description="Test database",
    )
    updated_manifest = PublishManifest(
        manifest_version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        databases={"finding_models": mock_db_entry},
    )
    mock_update_entry.return_value = updated_manifest

    # Execute
    update_info = ManifestUpdateInfo(
        database_key="finding_models",
        version="2025-11-17",
        url="https://example.com/db.duckdb",
        hash_value="sha256:abc123",
        size_bytes=52428800,
        record_count=1234,
        description="Test database",
    )

    manifest_url, backup_url = update_and_publish_manifest(mock_s3_client, mock_config, update_info)

    # Verify URLs are constructed correctly
    assert manifest_url == "https://test-bucket.t3.storage.dev/manifest.json"
    assert backup_url.startswith("https://test-bucket.t3.storage.dev/manifests/archive/manifest_")
    assert backup_url.endswith(".json")

    # Verify function calls
    mock_load.assert_called_once_with(mock_s3_client, "test-bucket")
    assert mock_s3_client.upload_file.call_count == 2  # backup + manifest
    mock_update_entry.assert_called_once()


@patch("findingmodel.db_publish.load_manifest_from_s3")
def test_update_and_publish_manifest_cleanup_on_error(
    mock_load: MagicMock,
) -> None:
    """Test that temporary files are cleaned up on error."""
    # Setup mocks
    mock_s3_client = MagicMock()
    mock_s3_client.meta.endpoint_url = "https://t3.storage.dev"
    mock_config = MagicMock()
    mock_config.s3_bucket = "test-bucket"
    mock_config.manifest_backup_prefix = "manifests/archive/"

    # Setup successful initial steps
    existing_manifest = PublishManifest(
        manifest_version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        databases={},
    )
    mock_load.return_value = existing_manifest

    # Make upload fail on the second call (manifest upload)
    from botocore.exceptions import ClientError

    mock_s3_client.upload_file.side_effect = [
        None,  # backup succeeds
        ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}}, "upload_file"
        ),  # manifest upload fails
    ]

    update_info = ManifestUpdateInfo(
        database_key="finding_models",
        version="2025-11-17",
        url="https://example.com/db.duckdb",
        hash_value="sha256:abc123",
        size_bytes=52428800,
        record_count=1234,
        description="Test database",
    )

    # Verify error is raised
    with pytest.raises(RuntimeError, match="S3 operation failed"):
        update_and_publish_manifest(mock_s3_client, mock_config, update_info)

    # Verify temp file was cleaned up (if it existed)
    temp_file = Path("manifest_temp.json")
    assert not temp_file.exists()


# Phase 5: Additional Test Coverage


# 1. CLI Command Tests


def test_publish_command_requires_option() -> None:
    """Test that publish command fails when neither --defs-dir nor --database is provided."""
    from click.testing import CliRunner
    from findingmodel.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["index", "publish"])

    assert result.exit_code == 1
    assert "Error: Provide exactly one of --defs-dir or --database" in result.output


def test_publish_command_mutually_exclusive(tmp_path: Path) -> None:
    """Test that publish command fails when both --defs-dir and --database are provided."""
    from click.testing import CliRunner
    from findingmodel.cli import cli

    # Create dummy paths
    defs_dir = tmp_path / "defs"
    defs_dir.mkdir()
    db_file = tmp_path / "test.duckdb"
    db_file.write_text("dummy")

    runner = CliRunner()
    result = runner.invoke(cli, ["index", "publish", "--defs-dir", str(defs_dir), "--database", str(db_file)])

    assert result.exit_code == 1
    assert "Error: Provide exactly one of --defs-dir or --database" in result.output


@patch("findingmodel.db_publish.create_s3_client")
@patch("findingmodel.db_publish.compute_file_hash")
@patch("findingmodel.db_publish.get_database_stats")
@patch("findingmodel.db_publish.update_and_publish_manifest")
def test_publish_command_with_database_mock(
    mock_update_manifest: MagicMock,
    mock_get_stats: MagicMock,
    mock_hash: MagicMock,
    mock_s3_client: MagicMock,
    tmp_path: Path,
) -> None:
    """Test publish-only mode with mocked S3 operations."""
    from click.testing import CliRunner
    from findingmodel.cli import cli
    from findingmodel.db_publish import DatabaseStats

    # Create a dummy database file
    db_file = tmp_path / "test.duckdb"
    db_file.write_text("dummy database content")

    # Setup mocks
    mock_hash.return_value = "sha256:abc123"
    mock_get_stats.return_value = DatabaseStats(
        record_count=100,
        size_bytes=1024,
        database_path=db_file,
    )
    mock_client_instance = MagicMock()
    mock_client_instance.meta.endpoint_url = "https://t3.storage.dev"
    mock_s3_client.return_value = mock_client_instance
    mock_update_manifest.return_value = (
        "https://bucket.t3.storage.dev/manifest.json",
        "https://bucket.t3.storage.dev/manifests/archive/manifest_20251117_120000.json",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["index", "publish", "--database", str(db_file), "--skip-checks"])

    # Verify command succeeded
    assert result.exit_code == 0
    assert "Database uploaded" in result.output
    assert "Manifest updated" in result.output

    # Verify S3 client was created and used
    mock_s3_client.assert_called_once()
    mock_client_instance.upload_file.assert_called_once()
    mock_update_manifest.assert_called_once()


@patch("findingmodel.cli._run_sanity_check_with_confirmation")
@patch("findingmodel.db_publish.create_s3_client")
@patch("findingmodel.db_publish.compute_file_hash")
@patch("findingmodel.db_publish.get_database_stats")
@patch("findingmodel.db_publish.update_and_publish_manifest")
def test_publish_command_with_skip_checks(
    mock_update_manifest: MagicMock,
    mock_get_stats: MagicMock,
    mock_hash: MagicMock,
    mock_s3_client: MagicMock,
    mock_sanity_check: MagicMock,
    tmp_path: Path,
) -> None:
    """Test that --skip-checks flag bypasses sanity check."""
    from click.testing import CliRunner
    from findingmodel.cli import cli
    from findingmodel.db_publish import DatabaseStats

    # Create a dummy database file
    db_file = tmp_path / "test.duckdb"
    db_file.write_text("dummy database content")

    # Setup mocks
    mock_hash.return_value = "sha256:abc123"
    mock_get_stats.return_value = DatabaseStats(
        record_count=100,
        size_bytes=1024,
        database_path=db_file,
    )
    mock_client_instance = MagicMock()
    mock_client_instance.meta.endpoint_url = "https://t3.storage.dev"
    mock_s3_client.return_value = mock_client_instance
    mock_update_manifest.return_value = (
        "https://bucket.t3.storage.dev/manifest.json",
        "https://bucket.t3.storage.dev/manifests/archive/manifest_20251117_120000.json",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["index", "publish", "--database", str(db_file), "--skip-checks"])

    # Verify command succeeded
    assert result.exit_code == 0

    # Verify sanity check was NOT called
    mock_sanity_check.assert_not_called()

    # Verify output indicates checks were skipped
    assert "Skipping sanity checks" in result.output


# 2. Core Function Tests


def test_compute_file_hash(tmp_path: Path) -> None:
    """Test that compute_file_hash returns correct SHA256 hash format."""
    from findingmodel.db_publish import compute_file_hash

    # Create a test file with known content
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, world!")

    hash_value = compute_file_hash(test_file)

    # Verify format is "sha256:hexdigest"
    assert hash_value.startswith("sha256:")
    assert len(hash_value) == 71  # "sha256:" (7 chars) + 64 hex chars

    # Verify it's deterministic
    hash_value_2 = compute_file_hash(test_file)
    assert hash_value == hash_value_2


def test_create_s3_client_missing_credentials() -> None:
    """Test that create_s3_client raises ValueError when credentials are missing."""
    from findingmodel.db_publish import PublishConfig, create_s3_client

    # Create config without credentials
    config = PublishConfig(
        aws_access_key_id=None,
        aws_secret_access_key=None,
    )

    with pytest.raises(ValueError, match="AWS credentials required"):
        create_s3_client(config)


def test_load_manifest_creates_empty_on_missing() -> None:
    """Test that load_manifest_from_s3 creates empty manifest when file not found."""
    from botocore.exceptions import ClientError
    from findingmodel.db_publish import load_manifest_from_s3

    # Setup mock S3 client that raises NoSuchKey error
    mock_s3_client = MagicMock()
    mock_s3_client.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}}, "download_file"
    )

    manifest = load_manifest_from_s3(mock_s3_client, "test-bucket")

    # Verify empty manifest is returned
    assert manifest.manifest_version == "1.0"
    assert len(manifest.databases) == 0
    assert manifest.generated_at is not None


def test_update_manifest_entry_preserves_other_databases() -> None:
    """Test that update_manifest_entry preserves other database entries."""
    from findingmodel.db_publish import ManifestDatabaseEntry, PublishManifest, update_manifest_entry

    # Create manifest with existing entry
    existing_entry = ManifestDatabaseEntry(
        version="2025-11-01",
        url="https://example.com/anatomic_locations.duckdb",
        hash="sha256:old123",
        size_bytes=1024,
        record_count=50,
        description="Anatomic locations database",
    )
    manifest = PublishManifest(
        manifest_version="1.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        databases={"anatomic_locations": existing_entry},
    )

    # Update with new finding_models entry
    updated_manifest = update_manifest_entry(
        manifest,
        database_key="finding_models",
        version="2025-11-17",
        url="https://example.com/finding_models.duckdb",
        hash_value="sha256:new456",
        size_bytes=2048,
        record_count=100,
        description="Finding models database",
    )

    # Verify both entries exist
    assert len(updated_manifest.databases) == 2
    assert "anatomic_locations" in updated_manifest.databases
    assert "finding_models" in updated_manifest.databases

    # Verify existing entry is unchanged
    assert updated_manifest.databases["anatomic_locations"] == existing_entry

    # Verify new entry is correct
    new_entry = updated_manifest.databases["finding_models"]
    assert new_entry.version == "2025-11-17"
    assert new_entry.record_count == 100


# 3. Error Scenario Tests


def test_upload_handles_s3_error() -> None:
    """Test that S3 upload failures are handled correctly."""
    from botocore.exceptions import ClientError
    from findingmodel.db_publish import (
        ManifestUpdateInfo,
        PublishConfig,
        update_and_publish_manifest,
    )

    # Setup mock S3 client that fails on upload
    mock_s3_client = MagicMock()
    mock_s3_client.meta.endpoint_url = "https://t3.storage.dev"
    mock_s3_client.upload_file.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "upload_file"
    )

    # Mock load_manifest to return an empty manifest
    with patch("findingmodel.db_publish.load_manifest_from_s3") as mock_load:
        mock_load.return_value = PublishManifest(
            manifest_version="1.0",
            generated_at=datetime.now(timezone.utc).isoformat(),
            databases={},
        )

        config = PublishConfig(s3_bucket="test-bucket")
        update_info = ManifestUpdateInfo(
            database_key="finding_models",
            version="2025-11-17",
            url="https://example.com/db.duckdb",
            hash_value="sha256:abc123",
            size_bytes=1024,
            record_count=100,
            description="Test database",
        )

        # Verify error is raised
        with pytest.raises(RuntimeError, match="S3 operation failed"):
            update_and_publish_manifest(mock_s3_client, config, update_info)


def test_manifest_update_handles_invalid_json(tmp_path: Path) -> None:
    """Test that corrupted manifest JSON is handled correctly."""
    from findingmodel.db_publish import load_manifest_from_s3

    # Create a temp file with invalid JSON
    invalid_json_file = tmp_path / "manifest.json"
    invalid_json_file.write_text("{invalid json content")

    # Setup mock S3 client that returns the invalid file
    mock_s3_client = MagicMock()
    mock_s3_client.download_file.side_effect = lambda _bucket, _key, path: Path(path).write_text("{invalid json")

    # Verify that JSON parsing error is raised
    with pytest.raises(json.JSONDecodeError):
        load_manifest_from_s3(mock_s3_client, "test-bucket")


# 4. Integration Test


@pytest.mark.callout
def test_publish_integration_with_fm_test_bucket(tmp_path: Path, full_model: FindingModelFull) -> None:
    """Full workflow test using real fm-test bucket.

    This test:
    1. Creates a small test database
    2. Uploads to fm-test bucket
    3. Verifies manifest created/updated
    4. Cleans up test files from S3

    Requires AWS credentials in .env file.
    """
    from findingmodel.db_publish import (
        ManifestUpdateInfo,
        PublishConfig,
        compute_file_hash,
        create_s3_client,
        load_manifest_from_s3,
        update_and_publish_manifest,
    )

    # Skip if credentials not available (check via PublishConfig which loads .env)
    try:
        test_config = PublishConfig()
        if not test_config.aws_access_key_id or not test_config.aws_secret_access_key:
            pytest.skip("AWS credentials not available for integration test")
    except Exception:
        pytest.skip("AWS credentials not available for integration test")

    # Create a minimal test database
    test_db_path = tmp_path / "test_integration.duckdb"

    # Import DuckDB utilities
    from findingmodel.tools.duckdb_utils import setup_duckdb_connection

    conn = setup_duckdb_connection(test_db_path, read_only=False)
    try:
        # Create minimal schema
        conn.execute(
            """
            CREATE TABLE finding_models (
                oifm_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                description VARCHAR
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE finding_model_json (
                oifm_id VARCHAR PRIMARY KEY,
                model_json VARCHAR
            )
        """
        )

        # Insert test data using fixture
        conn.execute(
            "INSERT INTO finding_models (oifm_id, name, description) VALUES (?, ?, ?)",
            [full_model.oifm_id, full_model.name, full_model.description],
        )
        conn.execute(
            "INSERT INTO finding_model_json (oifm_id, model_json) VALUES (?, ?)",
            [full_model.oifm_id, full_model.model_dump_json()],
        )

    finally:
        conn.close()

    # Setup S3 client for fm-test bucket
    config = PublishConfig(
        s3_bucket="fm-test",
        s3_endpoint_url="https://fly.storage.tigris.dev",
    )

    s3_client = create_s3_client(config)

    # Generate unique test filename
    test_filename = f"test_integration_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.duckdb"

    try:
        # Upload database
        s3_client.upload_file(str(test_db_path), config.s3_bucket, test_filename)

        # Verify upload succeeded
        response = s3_client.head_object(Bucket=config.s3_bucket, Key=test_filename)
        assert response["ContentLength"] == test_db_path.stat().st_size

        # Compute hash
        hash_value = compute_file_hash(test_db_path)

        # Get database stats
        from findingmodel.db_publish import get_database_stats

        stats = get_database_stats(test_db_path)

        # Update manifest
        update_info = ManifestUpdateInfo(
            database_key="test_integration",
            version=f"{datetime.now(timezone.utc):%Y-%m-%d}",
            url=f"https://{config.s3_bucket}.fly.storage.tigris.dev/{test_filename}",
            hash_value=hash_value,
            size_bytes=stats.size_bytes,
            record_count=stats.record_count,
            description="Integration test database",
        )

        manifest_url, backup_url = update_and_publish_manifest(s3_client, config, update_info)

        # Verify manifest was updated
        assert "manifest.json" in manifest_url
        assert "manifests/archive" in backup_url

        # Load and verify manifest
        manifest = load_manifest_from_s3(s3_client, config.s3_bucket)
        assert "test_integration" in manifest.databases
        assert manifest.databases["test_integration"].record_count == 1

    finally:
        # Clean up: delete test database from S3
        try:
            s3_client.delete_object(Bucket=config.s3_bucket, Key=test_filename)
        except Exception as e:
            print(f"Warning: Failed to delete test file {test_filename}: {e}")

        # Clean up: remove test_integration entry from manifest
        try:
            # Download current manifest
            manifest_response = s3_client.get_object(Bucket=config.s3_bucket, Key="manifest.json")
            manifest_data = json.loads(manifest_response["Body"].read())

            # Remove test entry if present
            if "test_integration" in manifest_data.get("databases", {}):
                del manifest_data["databases"]["test_integration"]

                # Upload cleaned manifest
                s3_client.put_object(
                    Bucket=config.s3_bucket,
                    Key="manifest.json",
                    Body=json.dumps(manifest_data, indent=2),
                )
        except Exception as e:
            print(f"Warning: Failed to clean up manifest entry: {e}")

        # Note: We don't delete manifest backup as it preserves manifest history


# Integration tests above require real S3 access and are marked with @pytest.mark.callout
