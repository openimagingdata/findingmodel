"""S3/Tigris storage utilities for database publishing."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

from oidm_maintenance.config import MaintenanceSettings


def create_s3_client(settings: MaintenanceSettings) -> S3Client:
    """Create boto3 S3 client configured for Tigris or AWS.

    Args:
        settings: MaintenanceSettings with AWS credentials and endpoint

    Returns:
        Configured boto3 S3 client

    Raises:
        ValueError: If credentials are missing

    Example:
        settings = get_settings()
        client = create_s3_client(settings)
    """
    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        raise ValueError(
            "AWS credentials required for S3 operations. "
            "Set OIDM_MAINTAIN_AWS_ACCESS_KEY_ID and OIDM_MAINTAIN_AWS_SECRET_ACCESS_KEY environment variables."
        )

    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
        config=Config(s3={"addressing_style": "virtual"}),  # Required for Tigris buckets created after Feb 2025
    )


def verify_bucket_access(client: S3Client, bucket: str) -> None:
    """Verify that the client has access to the specified bucket.

    Uses head_bucket to validate credentials and bucket existence without
    listing or downloading any objects.

    Args:
        client: Configured boto3 S3 client
        bucket: S3 bucket name to verify

    Raises:
        ClientError: If bucket doesn't exist or credentials are invalid
            - 403: Invalid credentials or access denied
            - 404: Bucket does not exist

    Example:
        client = create_s3_client(settings)
        verify_bucket_access(client, "mybucket")  # Raises if invalid
    """
    client.head_bucket(Bucket=bucket)


def upload_file_to_s3(client: S3Client, bucket: str, key: str, local_path: Path) -> str:
    """Upload a file to S3 and return its public URL.

    Args:
        client: Configured boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key (path within bucket)
        local_path: Path to local file to upload

    Returns:
        Public HTTPS URL to the uploaded file

    Raises:
        FileNotFoundError: If local file doesn't exist
        ClientError: If S3 upload fails

    Example:
        url = upload_file_to_s3(client, "mybucket", "data/file.db", Path("local.db"))
        # Returns: "https://mybucket.fly.storage.tigris.dev/data/file.db"
    """
    if not local_path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    client.upload_file(str(local_path), bucket, key)

    # Construct public URL from endpoint
    endpoint_url = client.meta.endpoint_url
    base_url = endpoint_url.removeprefix("https://").removeprefix("http://")
    return f"https://{bucket}.{base_url}/{key}"


def load_manifest_from_s3(client: S3Client, bucket: str, key: str) -> dict[str, Any]:
    """Load manifest JSON from S3.

    Downloads the manifest file from S3 and parses it as JSON. Returns an empty
    manifest structure if the file doesn't exist.

    Args:
        client: Configured boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key for the manifest file

    Returns:
        Parsed manifest dict, or empty manifest if not found

    Example:
        manifest = load_manifest_from_s3(client, "mybucket", "manifest.json")
    """
    temp_path = Path("manifest.json.tmp")

    try:
        # Download file from S3
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(temp_path))

        # Parse JSON
        manifest_data: dict[str, Any] = json.loads(temp_path.read_text())
        return manifest_data

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        # Handle both "NoSuchKey" and "404" (from head_object in download_file)
        if error_code in ("NoSuchKey", "404"):
            # Return empty manifest structure
            return {
                "manifest_version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "databases": {},
            }
        raise

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def backup_manifest(client: S3Client, bucket: str, manifest: dict[str, Any], backup_prefix: str) -> str:
    """Backup manifest to S3 archive folder with timestamp.

    Args:
        client: Configured boto3 S3 client
        bucket: S3 bucket name
        manifest: Manifest dict to backup
        backup_prefix: S3 key prefix for backups (e.g., "manifests/archive/")

    Returns:
        Public HTTPS URL to the backup file

    Raises:
        ClientError: If S3 upload fails

    Example:
        backup_url = backup_manifest(client, "mybucket", manifest, "manifests/archive/")
        # Returns: "https://mybucket.../manifests/archive/manifest_20250111_143022.json"
    """
    # Generate timestamped backup key
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_key = f"{backup_prefix}manifest_{timestamp}.json"

    # Create temporary file for backup
    temp_fd, temp_str = tempfile.mkstemp(suffix=".json", prefix="manifest_backup_")
    temp_path = Path(temp_str)

    # Close file descriptor (we'll use path for operations)
    import os

    os.close(temp_fd)

    try:
        # Write manifest to temp file
        manifest_json = json.dumps(manifest, indent=2)
        temp_path.write_text(manifest_json)

        # Upload to S3
        return upload_file_to_s3(client, bucket, backup_key, temp_path)

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def save_manifest_to_s3(client: S3Client, bucket: str, key: str, manifest: dict[str, Any]) -> None:
    """Save manifest dict to S3 as JSON.

    Args:
        client: Configured boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key for the manifest file
        manifest: Manifest dict to save

    Raises:
        ClientError: If S3 upload fails

    Example:
        save_manifest_to_s3(client, "mybucket", "manifest.json", manifest)
    """
    # Create temporary file
    temp_fd, temp_str = tempfile.mkstemp(suffix=".json", prefix="manifest_")
    temp_path = Path(temp_str)

    # Close file descriptor
    import os

    os.close(temp_fd)

    try:
        # Write manifest to temp file
        manifest_json = json.dumps(manifest, indent=2)
        temp_path.write_text(manifest_json)

        # Upload to S3
        upload_file_to_s3(client, bucket, key, temp_path)

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def update_manifest_entry(manifest: dict[str, Any], db_key: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Update a database entry in the manifest.

    Creates a new manifest dict with the updated entry and refreshed timestamp.
    Does not modify the input manifest.

    Args:
        manifest: Current manifest dict to update
        db_key: Database key to update (e.g., "finding_models")
        entry: New database entry dict with keys: version, url, hash, size_bytes, record_count, description

    Returns:
        New manifest dict with updated entry and timestamp

    Example:
        updated = update_manifest_entry(
            manifest,
            "finding_models",
            {
                "version": "2025-01-11",
                "url": "https://example.com/db.duckdb",
                "hash": "sha256:abc123",
                "size_bytes": 52428800,
                "record_count": 1234,
                "description": "Finding model index"
            }
        )
    """
    # Copy databases dict and update entry
    updated_databases = manifest.get("databases", {}).copy()
    updated_databases[db_key] = entry

    # Return new manifest with updated timestamp
    return {
        "manifest_version": manifest.get("manifest_version", "1.0"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "databases": updated_databases,
    }


__all__ = [
    "backup_manifest",
    "create_s3_client",
    "load_manifest_from_s3",
    "save_manifest_to_s3",
    "update_manifest_entry",
    "upload_file_to_s3",
    "verify_bucket_access",
]
