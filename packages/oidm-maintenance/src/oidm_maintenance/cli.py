"""CLI for OIDM maintenance operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oidm_maintenance.anatomic.validate import ValidationResult

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


@anatomic.command(name="validate")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
def anatomic_validate(source: Path) -> None:
    """Validate anatomic-locations source JSON."""
    from oidm_maintenance.anatomic.validate import validate_anatomic_json

    console.print(f"[bold]Validating {source}...[/bold]")
    result = validate_anatomic_json(source)
    _print_validation_result(result)


def _print_validation_result(result: ValidationResult) -> None:
    """Print validation result details to console."""
    for w in result.warnings:
        console.print(f"  [yellow]⚠ {w}[/yellow]")

    if result.parse_errors:
        console.print("\n[bold red]Parse errors:[/bold red]")
        for rid, errs in result.parse_errors.items():
            for e in errs:
                console.print(f"  {rid}: {e}")

    if result.relationship_errors:
        console.print("\n[bold red]Relationship errors:[/bold red]")
        for e in result.relationship_errors:
            console.print(f"  {e}")

    console.print()
    if result.ok:
        console.print(f"[bold green]✓ {result.summary()}[/bold green]")
    else:
        console.print(f"[bold red]✗ {result.summary()}[/bold red]")
        raise SystemExit(1)


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


@main.group()
def embeddings() -> None:
    """Embedding cache operations."""
    pass


@embeddings.command(name="migrate")
def embeddings_migrate() -> None:
    """Run one-time migration into current embedding cache namespace."""
    import asyncio

    from oidm_maintenance.embeddings import migrate_default_cache

    console.print("[bold]Migrating embedding cache...[/bold]")
    try:
        cache_dir = asyncio.run(migrate_default_cache())
    except Exception as e:
        console.print(f"[bold red]✗ Migration failed:[/bold red] {e}")
        raise SystemExit(1) from e
    console.print(f"[bold green]✓ Cache ready:[/bold green] {cache_dir}")


@embeddings.command(name="stats")
def embeddings_stats() -> None:
    """Show current embedding cache statistics."""
    import asyncio

    from oidm_maintenance.embeddings import get_default_cache_stats

    console.print("[bold]Embedding cache stats...[/bold]")
    try:
        cache_dir, stats = asyncio.run(get_default_cache_stats())
    except Exception as e:
        console.print(f"[bold red]✗ Stats failed:[/bold red] {e}")
        raise SystemExit(1) from e

    models = stats["models"] if isinstance(stats.get("models"), dict) else {}
    console.print(f"[bold green]✓ Cache:[/bold green] {cache_dir}")
    console.print(f"  Total keys: {stats['total_keys']}")
    console.print(f"  Embedding keys: {stats['embedding_keys']}")
    console.print(f"  Migration keys: {stats['migration_keys']}")
    if models:
        console.print("  Models:")
        for model, count in sorted(models.items()):
            console.print(f"    - {model}: {count}")


@embeddings.command(name="import-duckdb")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--assume-defaults/--preserve-metadata",
    default=True,
    show_default=True,
    help=(
        "Assume all rows belong to --default-model/--default-dimensions "
        "(recommended for legacy text_hash-keyed cache files)."
    ),
)
@click.option("--default-model", default="text-embedding-3-small", show_default=True, help="Default model value.")
@click.option("--default-dimensions", default=512, type=int, show_default=True, help="Default dimensions value.")
def embeddings_import_duckdb(source: Path, assume_defaults: bool, default_model: str, default_dimensions: int) -> None:
    """Import a DuckDB embedding_cache table and upsert into current cache."""
    import asyncio

    from oidm_maintenance.embeddings import import_duckdb_into_current_cache

    console.print("[bold]Importing embeddings from DuckDB...[/bold]")
    console.print(f"  Source: {source}")
    try:
        cache_dir, stats = asyncio.run(
            import_duckdb_into_current_cache(
                source_path=source,
                assume_defaults=assume_defaults,
                default_model=default_model,
                default_dimensions=default_dimensions,
            )
        )
    except Exception as e:
        console.print(f"[bold red]✗ Import failed:[/bold red] {e}")
        raise SystemExit(1) from e

    console.print(f"[bold green]✓ Cache:[/bold green] {cache_dir}")
    console.print(
        f"[bold green]✓ Written:[/bold green] {stats['written']} "
        f"(new: {stats['new']}, updated: {stats['updated']}, "
        f"skipped: {stats['skipped']}, total: {stats['total']})"
    )


@embeddings.command(name="import-cache")
@click.argument("source_cache_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--upsert/--skip-existing",
    default=True,
    show_default=True,
    help="Overwrite existing entries when keys overlap.",
)
def embeddings_import_cache(source_cache_dir: Path, upsert: bool) -> None:
    """Import entries from another diskcache embedding cache directory."""
    import asyncio

    from oidm_maintenance.embeddings import import_cache_into_current_cache

    console.print("[bold]Importing embeddings from diskcache...[/bold]")
    console.print(f"  Source cache: {source_cache_dir}")
    try:
        cache_dir, stats = asyncio.run(
            import_cache_into_current_cache(
                source_cache_dir=source_cache_dir,
                upsert=upsert,
            )
        )
    except Exception as e:
        console.print(f"[bold red]✗ Import failed:[/bold red] {e}")
        raise SystemExit(1) from e

    console.print(f"[bold green]✓ Cache:[/bold green] {cache_dir}")
    console.print(
        f"[bold green]✓ Written:[/bold green] {stats['written']} "
        f"(new: {stats['new']}, updated: {stats['updated']}, "
        f"skipped: {stats['skipped']}, total: {stats['total']})"
    )


if __name__ == "__main__":
    main()
