"""CLI for OIDM maintenance operations."""

from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """OIDM Maintenance Tools - Build and publish databases."""
    pass


@main.group()
def anatomic() -> None:
    """Anatomic-locations database operations."""
    pass


@anatomic.command(name="build")
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Source JSON file with anatomic location data",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output DuckDB file path",
)
@click.option("--no-embeddings", is_flag=True, help="Skip embedding generation")
def anatomic_build(source: Path, output: Path, no_embeddings: bool) -> None:
    """Build anatomic-locations database from source JSON."""
    import asyncio

    from oidm_maintenance.anatomic.build import build_anatomic_database

    console.print("[bold]Building anatomic database...[/bold]")
    console.print(f"  Source: {source}")
    console.print(f"  Output: {output}")

    result = asyncio.run(
        build_anatomic_database(
            source_json=source,
            output_path=output,
            generate_embeddings=not no_embeddings,
        )
    )
    console.print(f"\n[bold green]✓ Created:[/bold green] {result}")


@anatomic.command(name="publish")
@click.argument("db_path", type=click.Path(exists=True, path_type=Path))
@click.option("--version", "-v", help="Version string (default: YYYY-MM-DD)")
@click.option("--dry-run", is_flag=True, help="Show what would happen without uploading")
def anatomic_publish(db_path: Path, version: str | None, dry_run: bool) -> None:
    """Publish anatomic-locations database to S3."""
    from oidm_maintenance.anatomic.publish import publish_anatomic_database

    success = publish_anatomic_database(db_path, version=version, dry_run=dry_run)
    if not success:
        raise SystemExit(1)


@main.group()
def findingmodel() -> None:
    """FindingModel database operations."""
    pass


@findingmodel.command(name="build")
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Source directory containing .fm.json files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output DuckDB file path",
)
@click.option("--no-embeddings", is_flag=True, help="Skip embedding generation")
def findingmodel_build(source: Path, output: Path, no_embeddings: bool) -> None:
    """Build findingmodel database from source models."""
    import asyncio

    from oidm_maintenance.findingmodel.build import build_findingmodel_database

    console.print("[bold]Building findingmodel database...[/bold]")
    console.print(f"  Source: {source}")
    console.print(f"  Output: {output}")

    result = asyncio.run(
        build_findingmodel_database(
            source_dir=source,
            output_path=output,
            generate_embeddings=not no_embeddings,
        )
    )
    console.print(f"\n[bold green]✓ Created:[/bold green] {result}")


@findingmodel.command(name="publish")
@click.argument("db_path", type=click.Path(exists=True, path_type=Path))
@click.option("--version", "-v", help="Version string (default: YYYY-MM-DD)")
@click.option("--dry-run", is_flag=True, help="Show what would happen without uploading")
def findingmodel_publish(db_path: Path, version: str | None, dry_run: bool) -> None:
    """Publish findingmodel database to S3."""
    from oidm_maintenance.findingmodel.publish import publish_findingmodel_database

    success = publish_findingmodel_database(db_path, version=version, dry_run=dry_run)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
