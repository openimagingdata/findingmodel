"""Publish anatomic-locations database to S3/Tigris."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from oidm_maintenance.config import get_settings
from oidm_maintenance.hashing import compute_file_hash
from oidm_maintenance.s3 import (
    backup_manifest,
    create_s3_client,
    load_manifest_from_s3,
    save_manifest_to_s3,
    update_manifest_entry,
    upload_file_to_s3,
)

console = Console()


def get_anatomic_stats(db_path: Path) -> dict[str, Any]:
    """Get statistics about an anatomic database.

    Opens the database in read-only mode and queries for statistics.

    Args:
        db_path: Path to the DuckDB file

    Returns:
        Dict with: record_count, sample_descriptions, regions

    Example:
        stats = get_anatomic_stats(Path("anatomic_locations.duckdb"))
        print(f"Records: {stats['record_count']}")
    """
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        count_result = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
        if count_result is None:
            raise ValueError("Failed to get count from database")
        count = count_result[0]

        samples = conn.execute("SELECT description FROM anatomic_locations LIMIT 5").fetchall()
        regions = conn.execute("SELECT DISTINCT region FROM anatomic_locations ORDER BY region").fetchall()
        return {
            "record_count": count,
            "sample_descriptions": [s[0] for s in samples],
            "regions": [r[0] for r in regions],
        }
    finally:
        conn.close()


def display_anatomic_stats(stats: dict[str, Any]) -> None:
    """Display database statistics using Rich.

    Args:
        stats: Statistics dict from get_anatomic_stats()

    Example:
        stats = get_anatomic_stats(db_path)
        display_anatomic_stats(stats)
    """
    table = Table(title="Anatomic Database Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Record Count", str(stats["record_count"]))
    table.add_row("Regions", ", ".join(stats["regions"]))
    table.add_row("Sample Descriptions", "\n".join(stats["sample_descriptions"][:3]))
    console.print(table)


def publish_anatomic_database(
    db_path: Path,
    version: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Publish anatomic database to S3.

    This function performs the following steps:
    1. Computes file hash and gathers database statistics
    2. Displays statistics and prompts for confirmation
    3. If confirmed and not dry_run, uploads file to S3
    4. Updates the manifest with the new database entry
    5. Backs up the old manifest before saving the new one

    Args:
        db_path: Path to the DuckDB file to publish.
        version: Version string (default: YYYY-MM-DD format from current date).
        dry_run: If True, show what would happen without uploading.

    Returns:
        True if publish succeeded, False if cancelled by user.

    Raises:
        FileNotFoundError: If db_path doesn't exist
        ValueError: If AWS credentials are missing (from create_s3_client)
        ClientError: If S3 operations fail

    Example:
        success = publish_anatomic_database(
            Path("anatomic_locations.duckdb"),
            version="2025-01-11",
            dry_run=False
        )
    """
    settings = get_settings()

    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Default version to current date
    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Step 1: Compute hash and gather stats
    console.print("[bold]Step 1:[/bold] Analyzing database...")
    file_hash = compute_file_hash(db_path)
    stats = get_anatomic_stats(db_path)
    display_anatomic_stats(stats)

    file_size = db_path.stat().st_size
    console.print(f"\nFile size: [cyan]{file_size:,}[/cyan] bytes")
    console.print(f"File hash: [cyan]{file_hash}[/cyan]")
    console.print(f"Version: [cyan]{version}[/cyan]")

    if not Confirm.ask("\nProceed with upload?"):
        console.print("[yellow]Upload cancelled by user[/yellow]")
        return False

    if dry_run:
        console.print("[yellow]Dry run - no changes made[/yellow]")
        return True

    # Step 2: Upload to S3
    console.print("\n[bold]Step 2:[/bold] Uploading to S3...")
    client = create_s3_client(settings)

    # Upload database file
    db_key = f"anatomic_locations/{version}/anatomic_locations.duckdb"
    db_url = upload_file_to_s3(client, settings.s3_bucket, db_key, db_path)
    console.print(f"Uploaded to: [cyan]{db_url}[/cyan]")

    # Step 3: Update manifest
    console.print("\n[bold]Step 3:[/bold] Updating manifest...")

    # Load current manifest
    manifest = load_manifest_from_s3(client, settings.s3_bucket, settings.manifest_key)

    # Backup old manifest
    if manifest.get("databases"):
        backup_url = backup_manifest(client, settings.s3_bucket, manifest, settings.manifest_backup_prefix)
        console.print(f"Backed up old manifest to: [cyan]{backup_url}[/cyan]")

    # Create new database entry
    db_entry = {
        "version": version,
        "url": db_url,
        "hash": file_hash,
        "size_bytes": file_size,
        "record_count": stats["record_count"],
        "description": "Anatomic locations database with semantic search",
    }

    # Update manifest
    updated_manifest = update_manifest_entry(manifest, "anatomic_locations", db_entry)

    # Save new manifest
    save_manifest_to_s3(client, settings.s3_bucket, settings.manifest_key, updated_manifest)
    console.print(f"Updated manifest at: [cyan]{settings.manifest_key}[/cyan]")

    console.print("\n[bold green]✓ Publish complete![/bold green]")
    return True


__all__ = [
    "display_anatomic_stats",
    "get_anatomic_stats",
    "publish_anatomic_database",
]
