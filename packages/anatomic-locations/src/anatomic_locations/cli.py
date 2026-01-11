"""Anatomic location CLI commands."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from anatomic_locations.index import AnatomicLocationIndex
from anatomic_locations.migration import (
    create_anatomic_database,
    get_database_stats,
    load_anatomic_data,
    validate_anatomic_record,
)


@click.group()
def main() -> None:
    """Anatomic location management tools."""
    pass


@main.command("build")
@click.option(
    "--source",
    "-s",
    help="URL or file path for anatomic location data (default: config setting or standard URL)",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output database path (default: config setting)")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing database")
def build(source: str | None, output: Path | None, force: bool) -> None:
    """Build anatomic location database from source data."""

    console = Console()

    async def _do_build(source: str | None, output: Path | None, force: bool) -> None:
        # Determine source and output paths
        # Try to get ensure_anatomic_db from config if available
        db_path: Path
        if output:
            db_path = output
        else:
            try:
                from anatomic_locations.config import ensure_anatomic_db

                db_path = ensure_anatomic_db()
            except ImportError:
                console.print("[bold red]Error: --output is required (config module not available for default path)")
                raise click.Abort() from None

        data_source = source or "https://oidm-public.t3.storage.dev/anatomic_locations_20251220.json"

        # Check if database already exists
        if db_path.exists() and not force:
            console.print(f"[yellow]Database already exists at {db_path}")
            console.print("[yellow]Use --force to overwrite")
            raise click.Abort()

        if db_path.exists() and force:
            console.print(f"[yellow]Removing existing database at {db_path}")
            db_path.unlink()

        console.print("[bold green]Building anatomic location database")
        console.print(f"[gray]Source: [yellow]{data_source}")
        console.print(f"[gray]Output: [yellow]{db_path.absolute()}")

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load data
            with console.status("[bold green]Loading anatomic location data..."):
                records = await load_anatomic_data(data_source)

            # Create OpenAI client for embeddings
            # Get API key from environment variable
            import os

            api_key: str | None = os.getenv("OPENAI_API_KEY")

            if not api_key:
                console.print("[bold red]Error: OPENAI_API_KEY not configured")
                console.print("[yellow]Set OPENAI_API_KEY environment variable")
                raise click.Abort()

            # Import openai here since it's an optional build dependency
            try:
                from openai import AsyncOpenAI
            except ImportError:
                console.print("[bold red]Error: openai package not installed")
                console.print("[yellow]Install with: pip install anatomic-locations[build]")
                raise click.Abort() from None

            client = AsyncOpenAI(api_key=api_key)

            # Enable logging for progress visibility
            from loguru import logger

            logger.enable("anatomic_locations")

            # Create database (progress logged to console)
            console.print("[bold green]Creating database and generating embeddings...")
            console.print(f"[gray]Processing {len(records)} records in batches of 50...\n")
            successful, failed = await create_anatomic_database(db_path, records, client)

            # Display results
            console.print("\n[bold green]Database built successfully!")
            console.print(f"[green]✓ Records inserted: {successful}")
            if failed > 0:
                console.print(f"[yellow]⚠ Records failed: {failed}")
            console.print(f"[gray]Database location: [yellow]{db_path.absolute()}")

        except Exception as e:
            console.print(f"[bold red]Error building database: {e}")
            raise

    asyncio.run(_do_build(source, output, force))


@main.command("validate")
@click.option(
    "--source",
    "-s",
    help="URL or file path for anatomic location data (default: standard URL)",
)
def validate(source: str | None) -> None:
    """Validate anatomic location data without building database."""

    console = Console()

    async def _do_validate(source: str | None) -> None:
        data_source = source or "https://oidm-public.t3.storage.dev/anatomic_locations_20251220.json"

        console.print("[bold green]Validating anatomic location data")
        console.print(f"[gray]Source: [yellow]{data_source}\n")

        try:
            # Load data
            with console.status("[bold green]Loading data..."):
                records = await load_anatomic_data(data_source)

            # Validate each record
            validation_errors: dict[str, list[str]] = {}
            for i, record in enumerate(records, 1):
                record_id = record.get("_id", f"record_{i}")
                errors = validate_anatomic_record(record)
                if errors:
                    validation_errors[record_id] = errors

            # Display results
            if validation_errors:
                console.print(f"[bold red]Validation failed for {len(validation_errors)} record(s):\n")
                for record_id, errors in validation_errors.items():
                    console.print(f"[yellow]{record_id}:")
                    for error in errors:
                        console.print(f"  [red]✗ {error}")
                    console.print()
                sys.exit(1)
            else:
                console.print(f"[bold green]✓ All {len(records)} records validated successfully!")

        except Exception as e:
            console.print(f"[bold red]Error validating data: {e}")
            raise

    asyncio.run(_do_validate(source))


@main.command("stats")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def stats(db_path: Path | None) -> None:
    """Show anatomic location database statistics."""
    console = Console()

    # Determine database path
    database_path: Path
    if db_path:
        database_path = db_path
    else:
        try:
            from anatomic_locations.config import ensure_anatomic_db

            database_path = ensure_anatomic_db()
        except ImportError:
            console.print("[bold red]Error: --db-path is required (config module not available for default path)")
            raise click.Abort() from None

    console.print("[bold green]Anatomic Location Database Statistics\n")
    console.print(f"[gray]Database: [yellow]{database_path.absolute()}\n")

    try:
        stats_data = get_database_stats(database_path)

        # Create summary table
        summary_table = Table(title="Database Summary", show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row("Total Records", str(stats_data["total_records"]))
        summary_table.add_row("Records with Vectors", str(stats_data["records_with_vectors"]))
        summary_table.add_row("Unique Regions", str(stats_data["unique_regions"]))
        summary_table.add_row("File Size", f"{stats_data['file_size_mb']:.2f} MB")

        console.print(summary_table)

        # Display laterality distribution
        console.print("\n[bold cyan]Laterality Distribution:")
        laterality_dist = stats_data["laterality_distribution"]
        for laterality, count in sorted(laterality_dist.items(), key=lambda x: x[1], reverse=True):
            laterality_label = laterality if laterality else "NULL"
            console.print(f"  {laterality_label}: {count}")

    except Exception as e:
        console.print(f"[bold red]Error reading database: {e}")
        raise


@main.group("query")
def query() -> None:
    """Query anatomic locations."""
    pass


@query.command("ancestors")
@click.argument("location_id")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def query_ancestors(location_id: str, db_path: Path | None) -> None:
    """Show containment ancestors for a location."""
    console = Console()

    # Determine database path
    database_path: Path
    if db_path:
        database_path = db_path
    else:
        try:
            from anatomic_locations.config import ensure_anatomic_db

            database_path = ensure_anatomic_db()
        except ImportError:
            console.print("[bold red]Error: --db-path is required (config module not available for default path)")
            raise click.Abort() from None

    try:
        with AnatomicLocationIndex(database_path) as index:
            try:
                location = index.get(location_id)
            except KeyError:
                console.print(f"[bold red]Location not found: {location_id}")
                sys.exit(1)

            # Display the location itself
            console.print(f"[bold cyan]Location: [white]{location.description} [gray]({location.id})")
            region_str = location.region.value if location.region else "N/A"
            console.print(f"[gray]Region: {region_str}, Type: {location.location_type.value}\n")

            # Get and display ancestors
            ancestors = location.get_containment_ancestors()

            if not ancestors:
                console.print("[yellow]No ancestors found (this may be a root location)")
                return

            # Create table for ancestors
            table = Table(title="Containment Ancestors (nearest to root)", show_header=True, header_style="bold cyan")
            table.add_column("Level", style="gray", justify="right", width=6)
            table.add_column("ID", style="yellow", width=20)
            table.add_column("Name", style="white")
            table.add_column("Region", style="cyan", width=15)

            # Ancestors are returned from immediate parent to root
            for i, ancestor in enumerate(ancestors, 1):
                table.add_row(
                    str(i),
                    ancestor.id,
                    ancestor.description,
                    ancestor.region.value if ancestor.region else "N/A",
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error querying ancestors: {e}")
        raise


@query.command("descendants")
@click.argument("location_id")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def query_descendants(location_id: str, db_path: Path | None) -> None:
    """Show containment descendants for a location."""
    console = Console()

    # Determine database path
    database_path: Path
    if db_path:
        database_path = db_path
    else:
        try:
            from anatomic_locations.config import ensure_anatomic_db

            database_path = ensure_anatomic_db()
        except ImportError:
            console.print("[bold red]Error: --db-path is required (config module not available for default path)")
            raise click.Abort() from None

    try:
        with AnatomicLocationIndex(database_path) as index:
            try:
                location = index.get(location_id)
            except KeyError:
                console.print(f"[bold red]Location not found: {location_id}")
                sys.exit(1)

            # Display the location itself
            console.print(f"[bold cyan]Location: [white]{location.description} [gray]({location.id})")
            region_str = location.region.value if location.region else "N/A"
            console.print(f"[gray]Region: {region_str}, Type: {location.location_type.value}\n")

            # Get and display descendants
            descendants = location.get_containment_descendants()

            if not descendants:
                console.print("[yellow]No descendants found (this may be a leaf location)")
                return

            # Create table for descendants
            table = Table(title="Containment Descendants", show_header=True, header_style="bold cyan")
            table.add_column("Depth", style="gray", justify="right", width=6)
            table.add_column("ID", style="yellow", width=20)
            table.add_column("Name", style="white")
            table.add_column("Region", style="cyan", width=15)

            for descendant in descendants:
                table.add_row(
                    str(descendant.containment_depth),
                    descendant.id,
                    descendant.description,
                    descendant.region.value if descendant.region else "N/A",
                )

            console.print(table)
            console.print(f"\n[gray]Total descendants: {len(descendants)}")

    except Exception as e:
        console.print(f"[bold red]Error querying descendants: {e}")
        raise


@query.command("laterality")
@click.argument("location_id")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def query_laterality(location_id: str, db_path: Path | None) -> None:
    """Show laterality variants for a location."""
    console = Console()

    # Determine database path
    database_path: Path
    if db_path:
        database_path = db_path
    else:
        try:
            from anatomic_locations.config import ensure_anatomic_db

            database_path = ensure_anatomic_db()
        except ImportError:
            console.print("[bold red]Error: --db-path is required (config module not available for default path)")
            raise click.Abort() from None

    try:
        with AnatomicLocationIndex(database_path) as index:
            try:
                location = index.get(location_id)
            except KeyError:
                console.print(f"[bold red]Location not found: {location_id}")
                sys.exit(1)

            # Display the location itself
            console.print(f"[bold cyan]Location: [white]{location.description} [gray]({location.id})")
            region_str = location.region.value if location.region else "N/A"
            console.print(f"[gray]Laterality: {location.laterality.value}, Region: {region_str}\n")

            # Get and display laterality variants
            variants = location.get_laterality_variants()

            if not variants:
                console.print("[yellow]No laterality variants found for this location")
                return

            # Create table for variants
            table = Table(title="Laterality Variants", show_header=True, header_style="bold cyan")
            table.add_column("Laterality", style="cyan", width=12)
            table.add_column("ID", style="yellow", width=20)
            table.add_column("Name", style="white")

            # Sort by laterality for consistent display
            for laterality in sorted(variants.keys(), key=lambda x: x.value):
                variant = variants[laterality]
                table.add_row(
                    laterality.value,
                    variant.id,
                    variant.description,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error querying laterality: {e}")
        raise


@query.command("code")
@click.argument("system")
@click.argument("code")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def query_code(system: str, code: str, db_path: Path | None) -> None:
    """Find locations by external code (e.g., SNOMED, FMA)."""
    console = Console()

    # Determine database path
    database_path: Path
    if db_path:
        database_path = db_path
    else:
        try:
            from anatomic_locations.config import ensure_anatomic_db

            database_path = ensure_anatomic_db()
        except ImportError:
            console.print("[bold red]Error: --db-path is required (config module not available for default path)")
            raise click.Abort() from None

    try:
        with AnatomicLocationIndex(database_path) as index:
            locations = index.find_by_code(system, code)

            if not locations:
                console.print(f"[yellow]No locations found for {system} code: {code}")
                return

            # Create table for results
            table = Table(
                title=f"Locations with {system.upper()} Code: {code}",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("ID", style="yellow", width=20)
            table.add_column("Name", style="white")
            table.add_column("Region", style="cyan", width=15)
            table.add_column("Laterality", style="green", width=12)

            for location in locations:
                table.add_row(
                    location.id,
                    location.description,
                    location.region.value if location.region else "N/A",
                    location.laterality.value,
                )

            console.print(table)
            console.print(f"\n[gray]Total matches: {len(locations)}")

    except Exception as e:
        console.print(f"[bold red]Error querying by code: {e}")
        raise


if __name__ == "__main__":
    main()
