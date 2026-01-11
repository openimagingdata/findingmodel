"""Publish findingmodel database to S3/Tigris."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
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


@dataclass
class DatabaseStats:
    """Statistics about a findingmodel database."""

    model_count: int
    synonyms_count: int
    tags_count: int
    size_bytes: int
    database_path: Path


@dataclass
class SanityCheckResult:
    """Results of database sanity check."""

    success: bool
    sample_count: int
    matched_count: int
    errors: list[str]
    stats: DatabaseStats


def get_findingmodel_stats(db_path: Path) -> DatabaseStats:
    """Get statistics about a findingmodel database.

    Args:
        db_path: Path to DuckDB database file

    Returns:
        DatabaseStats with model, synonym, tag counts, file size, and path

    Raises:
        FileNotFoundError: If database file doesn't exist
        RuntimeError: If database cannot be queried

    Example:
        stats = get_findingmodel_stats(Path("findingmodels.duckdb"))
        print(f"Models: {stats.model_count}, Size: {stats.size_bytes} bytes")
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Import locally to avoid circular dependency
    from oidm_common.duckdb import setup_duckdb_connection

    # Open read-only connection
    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Query model count
        result = conn.execute("SELECT COUNT(*) FROM finding_models").fetchone()
        if result is None:
            raise RuntimeError(f"Failed to query model count from {db_path}")
        model_count = int(result[0])

        # Query synonym count from separate table
        result = conn.execute("SELECT COUNT(*) FROM synonyms").fetchone()
        synonyms_count = int(result[0]) if result and result[0] else 0

        # Query tag count from separate table
        result = conn.execute("SELECT COUNT(*) FROM tags").fetchone()
        tags_count = int(result[0]) if result and result[0] else 0

        # Get file size
        size_bytes = db_path.stat().st_size

        return DatabaseStats(
            model_count=model_count,
            synonyms_count=synonyms_count,
            tags_count=tags_count,
            size_bytes=size_bytes,
            database_path=db_path,
        )
    finally:
        conn.close()


def get_sample_oifm_ids(db_path: Path, count: int = 3) -> list[str]:
    """Get sample OIFM IDs from the database.

    Args:
        db_path: Path to DuckDB database file
        count: Number of IDs to retrieve (default: 3)

    Returns:
        List of OIFM IDs (may be shorter than count if database has fewer records)

    Raises:
        FileNotFoundError: If database file doesn't exist
        RuntimeError: If database cannot be queried

    Example:
        ids = get_sample_oifm_ids(Path("findingmodels.duckdb"), count=5)
        print(ids)  # ['OIFM_FM_000001', 'OIFM_FM_000002', 'OIFM_FM_000003']
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Import locally to avoid circular dependency
    from oidm_common.duckdb import setup_duckdb_connection

    # Open read-only connection
    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Query first N OIFM IDs
        result = conn.execute("SELECT oifm_id FROM finding_models LIMIT ?", [count]).fetchall()
        ids = [row[0] for row in result]
        return ids
    finally:
        conn.close()


def get_complete_model_json(db_path: Path, oifm_id: str) -> dict[str, Any] | None:
    """Get complete model JSON for a specific OIFM ID.

    Args:
        db_path: Path to DuckDB database file
        oifm_id: OIFM ID to retrieve

    Returns:
        Model JSON as dict, or None if not found

    Raises:
        FileNotFoundError: If database file doesn't exist
        RuntimeError: If database cannot be queried

    Example:
        model = get_complete_model_json(Path("db.duckdb"), "OIFM_FM_000001")
        if model:
            print(model["name"])
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    # Import locally to avoid circular dependency
    from oidm_common.duckdb import setup_duckdb_connection

    # Open read-only connection
    conn = setup_duckdb_connection(db_path, read_only=True)
    try:
        # Query model JSON from finding_model_json table
        result = conn.execute("SELECT model_json FROM finding_model_json WHERE oifm_id = ?", [oifm_id]).fetchone()

        if result is None:
            return None

        model_json_str = result[0]
        model_data: dict[str, Any] = json.loads(model_json_str)
        return model_data
    finally:
        conn.close()


def run_sanity_check(db_path: Path) -> SanityCheckResult:
    """Run complete sanity check on findingmodel database.

    Validates that models can be retrieved and their JSON roundtrips correctly.

    Args:
        db_path: Path to DuckDB database file

    Returns:
        SanityCheckResult with success flag, sample counts, and errors

    Raises:
        FileNotFoundError: If database file doesn't exist
        RuntimeError: If database cannot be queried

    Example:
        result = run_sanity_check(Path("findingmodels.duckdb"))
        if result.success:
            print(f"Sanity check passed: {result.matched_count}/{result.sample_count} models OK")
        else:
            print(f"Errors: {result.errors}")
    """
    # Get database statistics
    stats = get_findingmodel_stats(db_path)

    # Handle empty database
    if stats.model_count == 0:
        return SanityCheckResult(
            success=False,
            sample_count=0,
            matched_count=0,
            errors=["Database is empty (0 records)"],
            stats=stats,
        )

    # Get sample IDs
    sample_count = min(3, stats.model_count)
    sample_ids = get_sample_oifm_ids(db_path, count=sample_count)

    # Try to retrieve JSON for each sample
    errors = []
    matched_count = 0
    for oifm_id in sample_ids:
        try:
            model = get_complete_model_json(db_path, oifm_id)
            if model is None:
                errors.append(f"Model {oifm_id} not found in finding_model_json table")
            else:
                # Validate JSON structure has required fields
                if "oifm_id" not in model or "name" not in model:
                    errors.append(f"Model {oifm_id} missing required fields (oifm_id, name)")
                else:
                    matched_count += 1
        except json.JSONDecodeError as e:
            errors.append(f"Model {oifm_id} has invalid JSON: {e}")
        except Exception as e:
            errors.append(f"Error retrieving model {oifm_id}: {e}")

    success = len(errors) == 0 and matched_count == sample_count

    return SanityCheckResult(
        success=success,
        sample_count=sample_count,
        matched_count=matched_count,
        errors=errors,
        stats=stats,
    )


def display_findingmodel_stats(stats: DatabaseStats) -> None:
    """Display database statistics using Rich table.

    Args:
        stats: DatabaseStats to display

    Example:
        stats = get_findingmodel_stats(Path("db.duckdb"))
        display_findingmodel_stats(stats)
    """
    table = Table(title="FindingModel Database Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="yellow", width=20)
    table.add_column("Value", style="white")

    table.add_row("Models", f"{stats.model_count:,}")
    table.add_row("Synonyms", f"{stats.synonyms_count:,}")
    table.add_row("Tags", f"{stats.tags_count:,}")
    table.add_row("File Size", f"{stats.size_bytes:,} bytes ({stats.size_bytes / 1024 / 1024:.2f} MB)")
    table.add_row("Path", str(stats.database_path))

    console.print()
    console.print(table)
    console.print()


def publish_findingmodel_database(  # noqa: C901
    db_path: Path,
    version: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Publish findingmodel database to S3.

    Performs sanity checks, displays statistics, prompts for confirmation,
    and uploads the database file and updated manifest to S3.

    Args:
        db_path: Path to the DuckDB file to publish.
        version: Version string (default: YYYY-MM-DD today's date).
        dry_run: If True, show what would happen without uploading.

    Returns:
        True if publish succeeded, False if cancelled or failed.

    Example:
        success = publish_findingmodel_database(
            Path("findingmodels.duckdb"),
            version="2025-01-11",
            dry_run=False
        )
    """
    settings = get_settings()

    # Use today's date as default version
    if version is None:
        version = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    console.print("\n[bold cyan]Publishing FindingModel Database[/bold cyan]")
    console.print(f"Version: {version}")
    console.print(f"Database: {db_path}")
    console.print(f"Dry run: {dry_run}\n")

    # Step 1: Run sanity check
    console.print("[yellow]Running sanity check...[/yellow]")
    try:
        sanity_result = run_sanity_check(db_path)
    except Exception as e:
        console.print(f"[red]Sanity check failed: {e}[/red]")
        return False

    # Display statistics
    display_findingmodel_stats(sanity_result.stats)

    # Display sanity check results
    console.print("[yellow]Sanity Check Results:[/yellow]")
    if sanity_result.success:
        console.print(
            f"  [green]✓[/green] Validated {sanity_result.matched_count}/{sanity_result.sample_count} sample models"
        )
    else:
        console.print(
            f"  [red]✗[/red] Only {sanity_result.matched_count}/{sanity_result.sample_count} sample models validated"
        )
        console.print("[red]Errors:[/red]")
        for error in sanity_result.errors:
            console.print(f"  - {error}")

        # Ask if user wants to continue despite errors
        continue_anyway = (
            console.input("\n[bold]Continue with publish despite errors? [yes/no]:[/bold] ").strip().lower()
        )
        if continue_anyway != "yes":
            console.print("[yellow]Publish cancelled.[/yellow]")
            return False

    # Step 2: Compute file hash
    console.print("\n[yellow]Computing file hash...[/yellow]")
    try:
        file_hash = compute_file_hash(db_path)
        console.print(f"  Hash: {file_hash}")
    except Exception as e:
        console.print(f"[red]Failed to compute hash: {e}[/red]")
        return False

    # Step 3: Prompt for confirmation
    console.print()
    if not dry_run:
        confirm = console.input("[bold]Proceed with upload to S3? [yes/no]:[/bold] ").strip().lower()
        if confirm != "yes":
            console.print("[yellow]Publish cancelled.[/yellow]")
            return False
    else:
        console.print("[bold]DRY RUN - Skipping upload[/bold]")

    # Step 4: Upload to S3
    if not dry_run:
        console.print("\n[yellow]Uploading to S3...[/yellow]")
        try:
            # Create S3 client
            s3_client = create_s3_client(settings)

            # Construct S3 key
            s3_key = f"findingmodel/{version}/findingmodels.duckdb"

            # Upload database file
            db_url = upload_file_to_s3(s3_client, settings.s3_bucket, s3_key, db_path)
            console.print(f"  [green]✓[/green] Uploaded database: {db_url}")

            # Step 5: Update manifest
            console.print("\n[yellow]Updating manifest...[/yellow]")

            # Load current manifest
            manifest = load_manifest_from_s3(s3_client, settings.s3_bucket, settings.manifest_key)

            # Backup current manifest
            backup_url = backup_manifest(s3_client, settings.s3_bucket, manifest, settings.manifest_backup_prefix)
            console.print(f"  [green]✓[/green] Backed up manifest: {backup_url}")

            # Update manifest entry
            entry = {
                "version": version,
                "url": db_url,
                "hash": file_hash,
                "size_bytes": sanity_result.stats.size_bytes,
                "record_count": sanity_result.stats.model_count,
                "description": f"FindingModel index with {sanity_result.stats.model_count} models, "
                f"{sanity_result.stats.synonyms_count} synonyms, {sanity_result.stats.tags_count} tags",
            }
            updated_manifest = update_manifest_entry(manifest, "finding_models", entry)

            # Save updated manifest
            save_manifest_to_s3(s3_client, settings.s3_bucket, settings.manifest_key, updated_manifest)
            manifest_url = f"https://{settings.s3_bucket}.{settings.s3_endpoint_url.removeprefix('https://').removeprefix('http://')}/{settings.manifest_key}"
            console.print(f"  [green]✓[/green] Updated manifest: {manifest_url}")

            # Success!
            console.print("\n[bold green]Publish successful![/bold green]")
            return True

        except Exception as e:
            console.print(f"\n[red]Publish failed: {e}[/red]")
            return False
    else:
        # Dry run - just show what would happen
        console.print("\n[bold]Would perform:[/bold]")
        console.print(f"  1. Upload database to s3://{settings.s3_bucket}/findingmodel/{version}/findingmodels.duckdb")
        console.print(f"  2. Backup current manifest to s3://{settings.s3_bucket}/{settings.manifest_backup_prefix}...")
        console.print(f"  3. Update manifest entry for 'finding_models' with version {version}")
        console.print(f"  4. Upload updated manifest to s3://{settings.s3_bucket}/{settings.manifest_key}")
        console.print("\n[bold green]Dry run complete![/bold green]")
        return True


__all__ = [
    "DatabaseStats",
    "SanityCheckResult",
    "display_findingmodel_stats",
    "get_complete_model_json",
    "get_findingmodel_stats",
    "get_sample_oifm_ids",
    "publish_findingmodel_database",
    "run_sanity_check",
]
