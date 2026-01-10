"""Infrastructure primitives for publishing FindingModel databases to S3."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import boto3
import pooch
from botocore.client import Config
from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client
from pydantic import BaseModel, Field, SecretStr
from rich.console import Console

from findingmodel import logger
from findingmodel.config import FindingModelConfig


class PublishConfig(FindingModelConfig):
    """Settings for database publishing to S3/Tigris storage.

    Extends FindingModelConfig with publishing-specific AWS/Tigris credentials
    and S3 configuration.
    """

    # AWS/Tigris credentials (standard AWS env vars)
    aws_access_key_id: SecretStr | None = Field(
        default=None,
        description="AWS/Tigris access key ID for S3 operations",
    )
    aws_secret_access_key: SecretStr | None = Field(
        default=None,
        description="AWS/Tigris secret access key for S3 operations",
    )

    # Tigris S3 settings
    s3_endpoint_url: str = Field(
        default="https://t3.storage.dev",
        description="S3 endpoint URL (Tigris or AWS)",
    )
    s3_bucket: str = Field(
        default="findingmodelsdata",
        description="S3 bucket name for database storage",
    )
    manifest_backup_prefix: str = Field(
        default="manifests/archive/",
        description="S3 key prefix for manifest backups",
    )


class ManifestDatabaseEntry(BaseModel):
    """Entry for a single database in the manifest."""

    version: str = Field(description="Database version in YYYY-MM-DD format")
    url: str = Field(description="HTTPS URL to database file")
    hash: str = Field(description="SHA256 hash in format 'sha256:hexdigest'")
    size_bytes: int = Field(description="File size in bytes")
    record_count: int = Field(description="Number of records in database")
    description: str = Field(description="Human-readable description of database contents")


class PublishManifest(BaseModel):
    """Manifest tracking all published databases."""

    manifest_version: str = Field(default="1.0", description="Manifest schema version")
    generated_at: str = Field(description="ISO 8601 timestamp of generation")
    databases: dict[str, ManifestDatabaseEntry] = Field(
        default_factory=dict, description="Map of database keys to entries"
    )


def create_s3_client(config: PublishConfig) -> S3Client:
    """Create boto3 S3 client configured for Tigris or AWS.

    Args:
        config: PublishConfig with AWS credentials and endpoint

    Returns:
        Configured boto3 S3 client

    Raises:
        ValueError: If credentials are missing
    """
    if not config.aws_access_key_id or not config.aws_secret_access_key:
        raise ValueError(
            "AWS credentials required for publishing. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment "
            "variables or in .env file."
        )

    logger.debug(f"Creating S3 client for endpoint {config.s3_endpoint_url}")

    return boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint_url,
        aws_access_key_id=config.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=config.aws_secret_access_key.get_secret_value(),
        config=Config(s3={"addressing_style": "virtual"}),  # Required for Tigris buckets created after Feb 2025
    )


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file using Pooch.

    Args:
        file_path: Path to file to hash

    Returns:
        Hash in format "sha256:hexdigest"

    Example:
        hash_str = compute_file_hash(Path("database.duckdb"))
        # Returns: "sha256:abc123def456..."
    """
    logger.debug(f"Computing SHA256 hash for {file_path}")
    digest = pooch.file_hash(str(file_path), alg="sha256")
    return f"sha256:{digest}"


def load_manifest_from_s3(s3_client: S3Client, bucket: str) -> PublishManifest:
    """Load manifest.json from S3.

    Args:
        s3_client: Configured boto3 S3 client
        bucket: S3 bucket name

    Returns:
        PublishManifest parsed from S3, or empty manifest if not found

    Example:
        s3_client = create_s3_client(config)
        manifest = load_manifest_from_s3(s3_client, "mybucket")
    """
    manifest_key = "manifest.json"
    temp_path = Path("manifest.json.tmp")

    try:
        logger.info(f"Loading manifest from s3://{bucket}/{manifest_key}")

        # Download file from S3 (inlined from download_file_from_s3)
        logger.info(f"Downloading s3://{bucket}/{manifest_key} to {temp_path}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, manifest_key, str(temp_path))
        logger.info(f"Download successful: {temp_path}")

        # Parse JSON into Pydantic model
        manifest_data = json.loads(temp_path.read_text())
        manifest = PublishManifest(**manifest_data)
        logger.info(f"Loaded manifest with {len(manifest.databases)} database entries")
        return manifest

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            logger.warning(f"Manifest not found at s3://{bucket}/{manifest_key}, creating empty manifest")
            return PublishManifest(
                generated_at=datetime.now(timezone.utc).isoformat(),
                databases={},
            )
        raise

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def update_manifest_entry(
    manifest: PublishManifest,
    database_key: str,
    version: str,
    url: str,
    hash_value: str,
    size_bytes: int,
    record_count: int,
    description: str,
) -> PublishManifest:
    """Update a database entry in the manifest.

    Args:
        manifest: Current manifest to update
        database_key: Key identifying the database (e.g., "finding_models")
        version: Database version in YYYY-MM-DD format
        url: HTTPS URL to database file
        hash_value: SHA256 hash in format "sha256:hexdigest"
        size_bytes: File size in bytes
        record_count: Number of records in database
        description: Human-readable description of the database

    Returns:
        Updated manifest with new entry and refreshed timestamp

    Example:
        manifest = load_manifest_from_s3(s3_client, "mybucket")
        manifest = update_manifest_entry(
            manifest,
            "finding_models",
            "2025-11-17",
            "https://example.com/db.duckdb",
            "sha256:abc123",
            52428800,
            1234,
            "Finding model index with embeddings and full JSON"
        )
    """
    # Create new database entry
    new_entry = ManifestDatabaseEntry(
        version=version,
        url=url,
        hash=hash_value,
        size_bytes=size_bytes,
        record_count=record_count,
        description=description,
    )

    # Update the databases dict (preserves other entries)
    updated_databases = manifest.databases.copy()
    updated_databases[database_key] = new_entry

    # Return new manifest with updated timestamp
    return PublishManifest(
        manifest_version=manifest.manifest_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        databases=updated_databases,
    )


# Phase 2: Sanity Check Workflow


@dataclass
class DatabaseStats:
    """Statistics about a database file."""

    record_count: int
    size_bytes: int
    database_path: Path


def get_database_stats(db_path: Path) -> DatabaseStats:
    """Get statistics about a database file.

    Args:
        db_path: Path to DuckDB database file

    Returns:
        DatabaseStats with record count, file size, and path

    Raises:
        FileNotFoundError: If database file doesn't exist
        RuntimeError: If database cannot be queried

    Example:
        stats = get_database_stats(Path("findingmodels.duckdb"))
        print(f"Records: {stats.record_count}, Size: {stats.size_bytes} bytes")
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Import locally to avoid circular dependency
    from oidm_common.duckdb import setup_duckdb_connection

    logger.debug(f"Querying database stats for {db_path}")

    # Open read-only connection
    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Query record count
        result = conn.execute("SELECT COUNT(*) FROM finding_models").fetchone()
        if result is None:
            raise RuntimeError(f"Failed to query record count from {db_path}")
        record_count = int(result[0])

        # Get file size
        size_bytes = db_path.stat().st_size

        logger.debug(f"Database stats: {record_count} records, {size_bytes} bytes")

        return DatabaseStats(
            record_count=record_count,
            size_bytes=size_bytes,
            database_path=db_path,
        )
    finally:
        conn.close()


def get_sample_oifm_ids(db_path: Path, limit: int = 3) -> list[str]:
    """Get sample OIFM IDs from the database.

    Args:
        db_path: Path to DuckDB database file
        limit: Number of IDs to retrieve (default: 3)

    Returns:
        List of OIFM IDs (may be shorter than limit if database has fewer records)

    Raises:
        FileNotFoundError: If database file doesn't exist
        RuntimeError: If database cannot be queried

    Example:
        ids = get_sample_oifm_ids(Path("findingmodels.duckdb"), limit=5)
        print(ids)  # ['OIFM_FM_000001', 'OIFM_FM_000002', 'OIFM_FM_000003']
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Import locally to avoid circular dependency
    from oidm_common.duckdb import setup_duckdb_connection

    logger.debug(f"Querying {limit} sample OIFM IDs from {db_path}")

    # Open read-only connection
    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Query first N OIFM IDs
        result = conn.execute("SELECT oifm_id FROM finding_models LIMIT ?", [limit]).fetchall()
        ids = [row[0] for row in result]

        logger.debug(f"Retrieved {len(ids)} sample IDs")
        return ids
    finally:
        conn.close()


def get_complete_model_json(db_path: Path, oifm_id: str) -> str:
    """Get complete model JSON for a specific OIFM ID (first 20 lines).

    Args:
        db_path: Path to DuckDB database file
        oifm_id: OIFM ID to retrieve

    Returns:
        First 20 lines of pretty-printed JSON model

    Raises:
        FileNotFoundError: If database file doesn't exist
        ValueError: If OIFM ID not found in database
        RuntimeError: If database cannot be queried

    Example:
        json_preview = get_complete_model_json(Path("db.duckdb"), "OIFM_FM_000001")
        print(json_preview)
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Import locally to avoid circular dependency
    from oidm_common.duckdb import setup_duckdb_connection

    logger.debug(f"Querying model JSON for {oifm_id} from {db_path}")

    # Open read-only connection
    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Query model JSON from finding_model_json table
        result = conn.execute("SELECT model_json FROM finding_model_json WHERE oifm_id = ?", [oifm_id]).fetchone()

        if result is None:
            raise ValueError(f"OIFM ID not found: {oifm_id}")

        model_json_str = result[0]

        # Parse and pretty-print JSON
        model_data = json.loads(model_json_str)
        pretty_json = json.dumps(model_data, indent=2)

        # Return first 20 lines
        lines = pretty_json.split("\n")
        preview_lines = lines[:20]
        preview = "\n".join(preview_lines)

        logger.debug(f"Retrieved {len(lines)} total lines, returning first 20")
        return preview
    finally:
        conn.close()


@dataclass
class SanityCheckResult:
    """Results of database sanity check."""

    stats: DatabaseStats
    sample_ids: list[str]
    model_preview: str


def run_sanity_check(db_path: Path) -> SanityCheckResult:
    """Run complete sanity check on database.

    Args:
        db_path: Path to DuckDB database file

    Returns:
        SanityCheckResult with stats, sample IDs, and model preview

    Raises:
        FileNotFoundError: If database file doesn't exist
        ValueError: If database is empty (0 records)
        RuntimeError: If database cannot be queried

    Example:
        result = run_sanity_check(Path("findingmodels.duckdb"))
        print(f"Records: {result.stats.record_count}")
        print(f"Sample IDs: {result.sample_ids}")
        print(result.model_preview)
    """
    logger.info(f"Running sanity check on {db_path}")

    # Get database statistics
    stats = get_database_stats(db_path)

    # Handle empty database
    if stats.record_count == 0:
        raise ValueError(f"Database is empty (0 records): {db_path}")

    # Get sample IDs
    sample_ids = get_sample_oifm_ids(db_path, limit=3)

    # Get model preview (use first sample ID)
    model_preview = get_complete_model_json(db_path, sample_ids[0])

    logger.info(f"Sanity check complete: {stats.record_count} records, {len(sample_ids)} samples")

    return SanityCheckResult(
        stats=stats,
        sample_ids=sample_ids,
        model_preview=model_preview,
    )


def prompt_user_confirmation(check_result: SanityCheckResult) -> Literal["yes", "no", "cancel"]:
    """Prompt user to confirm sanity check results.

    Displays database statistics, sample IDs, and model preview using rich Console,
    then prompts for confirmation.

    Args:
        check_result: SanityCheckResult to display

    Returns:
        User response: "yes", "no", or "cancel"

    Example:
        result = run_sanity_check(Path("db.duckdb"))
        response = prompt_user_confirmation(result)
        if response == "yes":
            # Continue with upload
            pass
    """
    console = Console()

    # Display header
    console.print("\n[bold cyan]Database Sanity Check Results[/bold cyan]\n")

    # Display statistics
    console.print("[yellow]Statistics:[/yellow]")
    console.print(f"  Records: {check_result.stats.record_count:,}")
    console.print(
        f"  Size: {check_result.stats.size_bytes:,} bytes ({check_result.stats.size_bytes / 1024 / 1024:.2f} MB)"
    )
    console.print(f"  Path: {check_result.stats.database_path}")

    # Display sample IDs
    console.print(f"\n[yellow]Sample OIFM IDs ({len(check_result.sample_ids)}):[/yellow]")
    for oifm_id in check_result.sample_ids:
        console.print(f"  - {oifm_id}")

    # Display model preview
    console.print(f"\n[yellow]Model Preview (first 20 lines of {check_result.sample_ids[0]}):[/yellow]")
    console.print(check_result.model_preview)

    # Prompt for confirmation
    console.print()
    while True:
        response = console.input("[bold]Does this look correct? [yes/no/cancel]:[/bold] ").strip().lower()
        if response in ("yes", "no", "cancel"):
            return response  # type: ignore[return-value]
        console.print(f"[red]Invalid response '{response}'. Please enter 'yes', 'no', or 'cancel'.[/red]")


# Phase 3: Manifest Update and Publishing Workflow


@dataclass
class ManifestUpdateInfo:
    """Information needed to update a manifest entry."""

    database_key: str
    version: str
    url: str
    hash_value: str
    size_bytes: int
    record_count: int
    description: str


def update_and_publish_manifest(
    s3_client: S3Client,
    config: PublishConfig,
    update_info: ManifestUpdateInfo,
) -> tuple[str, str]:
    """Update manifest with new database entry and publish to S3.

    Workflow:
    1. Download current manifest from S3
    2. Generate backup filename with timestamp
    3. Upload backup manifest to S3 archive folder
    4. Update manifest with new database entry
    5. Upload updated manifest to S3
    6. Return both URLs for confirmation

    Args:
        s3_client: Configured boto3 S3 client
        config: PublishConfig with bucket and backup prefix settings
        update_info: ManifestUpdateInfo with database entry details

    Returns:
        Tuple of (manifest_url, backup_url)

    Raises:
        ClientError: If S3 operations fail
        RuntimeError: If manifest update fails

    Example:
        s3_client = create_s3_client(config)
        update_info = ManifestUpdateInfo(
            database_key="finding_models",
            version="2025-11-17",
            url="https://example.com/db.duckdb",
            hash_value="sha256:abc123",
            size_bytes=52428800,
            record_count=1234,
            description="Finding model index with embeddings and full JSON"
        )
        manifest_url, backup_url = update_and_publish_manifest(s3_client, config, update_info)
    """
    logger.info("Starting manifest update and publish workflow")

    # Step 1: Download current manifest from S3
    logger.info(f"Downloading current manifest from s3://{config.s3_bucket}/manifest.json")
    manifest = load_manifest_from_s3(s3_client, config.s3_bucket)

    # Step 2: Generate backup filename with timestamp (inlined from generate_manifest_backup_key)
    backup_s3_key = f"manifests/archive/manifest_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    logger.info(f"Generated backup S3 key: {backup_s3_key}")

    # Create temporary file for manifest operations
    temp_manifest_fd, temp_manifest_str = tempfile.mkstemp(suffix=".json", prefix="manifest_")
    temp_manifest_path = Path(temp_manifest_str)
    # Close the file descriptor; we'll use the path for file operations
    import os

    os.close(temp_manifest_fd)

    try:
        # Step 3: Upload backup manifest to S3 archive folder
        logger.info(f"Backing up current manifest to s3://{config.s3_bucket}/{backup_s3_key}")

        # Serialize current manifest to temp file
        manifest_json = manifest.model_dump_json(indent=2)
        temp_manifest_path.write_text(manifest_json)

        # Upload backup (inlined from upload_file_to_s3)
        if not temp_manifest_path.exists():
            raise FileNotFoundError(f"File not found: {temp_manifest_path}")
        logger.info(f"Uploading {temp_manifest_path} to s3://{config.s3_bucket}/{backup_s3_key}")
        s3_client.upload_file(str(temp_manifest_path), config.s3_bucket, backup_s3_key)
        endpoint_url = s3_client.meta.endpoint_url
        backup_url = f"https://{config.s3_bucket}.{endpoint_url.removeprefix('https://').removeprefix('http://')}/{backup_s3_key}"
        logger.info(f"Upload successful: {backup_url}")

        # Step 4: Update manifest with new database entry
        logger.info(f"Updating manifest entry for database key: {update_info.database_key}")
        updated_manifest = update_manifest_entry(
            manifest,
            update_info.database_key,
            update_info.version,
            update_info.url,
            update_info.hash_value,
            update_info.size_bytes,
            update_info.record_count,
            update_info.description,
        )

        # Step 5: Upload updated manifest to S3
        logger.info("Uploading updated manifest to S3")

        # Serialize updated manifest to temp file
        updated_json = updated_manifest.model_dump_json(indent=2)
        temp_manifest_path.write_text(updated_json)

        # Upload to root as manifest.json (inlined from upload_file_to_s3)
        manifest_key = "manifest.json"
        if not temp_manifest_path.exists():
            raise FileNotFoundError(f"File not found: {temp_manifest_path}")
        logger.info(f"Uploading {temp_manifest_path} to s3://{config.s3_bucket}/{manifest_key}")
        s3_client.upload_file(str(temp_manifest_path), config.s3_bucket, manifest_key)
        manifest_url = (
            f"https://{config.s3_bucket}.{endpoint_url.removeprefix('https://').removeprefix('http://')}/{manifest_key}"
        )
        logger.info(f"Upload successful: {manifest_url}")

        # Step 6: Return both URLs
        logger.info("Manifest update and publish workflow complete")
        return (manifest_url, backup_url)

    except ClientError as e:
        error_msg = f"S3 operation failed during manifest update: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    finally:
        # Clean up temporary file
        if temp_manifest_path.exists():
            logger.debug(f"Cleaning up temporary file: {temp_manifest_path}")
            temp_manifest_path.unlink()


__all__ = [
    "DatabaseStats",
    "ManifestDatabaseEntry",
    "ManifestUpdateInfo",
    "PublishConfig",
    "PublishManifest",
    "SanityCheckResult",
    "compute_file_hash",
    "create_s3_client",
    "get_complete_model_json",
    "get_database_stats",
    "get_sample_oifm_ids",
    "load_manifest_from_s3",
    "prompt_user_confirmation",
    "run_sanity_check",
    "update_and_publish_manifest",
    "update_manifest_entry",
]
