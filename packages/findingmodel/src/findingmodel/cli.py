import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .finding_model import FindingModelBase, FindingModelFull
from .index import Index


@click.group()
def cli() -> None:
    pass


@cli.command()
def config() -> None:
    """Show the currently active configuration."""
    console = Console()
    console.print("[yellow bold]Finding Model Forge configuration:")
    console.print_json(get_settings().model_dump_json())


@cli.command()
@click.argument("finding_model_path", type=click.Path(exists=True, path_type=Path, dir_okay=False))
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding model."
)
def fm_to_markdown(finding_model_path: Path, output: Path | None) -> None:
    """Convert finding model JSON file to Markdown format."""

    console = Console()
    console.print("[bold green]Loading finding model...")
    with open(finding_model_path, "r") as f:
        json = f.read()
        if "oifm_id" in json:
            fm_full = FindingModelFull.model_validate_json(json)
            markdown = fm_full.as_markdown()
        else:
            fm_base = FindingModelBase.model_validate_json(json)
            markdown = fm_base.as_markdown()
    if output:
        with open(output, "w") as f:
            f.write(markdown.strip() + "\n")
        console.print(f"[green]Saved Markdown to [yellow]{output}")
    else:
        from rich.markdown import Markdown

        console.print(Markdown(markdown))


@cli.command()
@click.option("--index", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def stats(index: Path | None) -> None:
    """Show index statistics."""

    console = Console()

    async def _do_stats(index: Path | None) -> None:
        from findingmodel.config import ensure_index_db

        db_path = index or ensure_index_db()

        if not db_path.exists():
            console.print(f"[bold red]Error: Database not found: {db_path}[/bold red]")
            console.print("[yellow]Hint: Use 'oidm-maintain findingmodel build' to create a database.[/yellow]")
            raise SystemExit(1)

        console.print(f"[bold green]Index Statistics for [yellow]{db_path}\n")

        try:
            async with Index(db_path=db_path) as idx:
                # Get counts
                model_count = await idx.count()
                people_count = await idx.count_people()
                org_count = await idx.count_organizations()

                # Get file size
                file_size = db_path.stat().st_size
                size_mb = file_size / (1024 * 1024)

                # Create summary table
                summary_table = Table(title="Database Summary", show_header=True, header_style="bold cyan")
                summary_table.add_column("Metric", style="cyan")
                summary_table.add_column("Value", style="green", justify="right")

                summary_table.add_row("Database Path", str(db_path.absolute()))
                summary_table.add_row("File Size", f"{size_mb:.2f} MB")
                summary_table.add_row("Total Models", str(model_count))
                summary_table.add_row("Total People", str(people_count))
                summary_table.add_row("Total Organizations", str(org_count))

                console.print(summary_table)

                # Check for search indexes
                console.print("\n[bold cyan]Index Status:")
                conn = idx.conn
                if conn:
                    # Check for HNSW index
                    hnsw_result = conn.execute(
                        "SELECT count(*) FROM duckdb_indexes() WHERE index_name = 'finding_models_embedding_hnsw'"
                    ).fetchone()
                    hnsw_exists = hnsw_result[0] > 0 if hnsw_result else False

                    # Check for FTS index by attempting to use it
                    try:
                        conn.execute(
                            "SELECT COUNT(*) FROM finding_models WHERE fts_main_finding_models.match_bm25(oifm_id, 'test') IS NOT NULL"
                        ).fetchone()
                        fts_exists = True
                    except Exception:
                        fts_exists = False

                    console.print(f"  HNSW Vector Index: {'[green]✓ Present' if hnsw_exists else '[red]✗ Missing'}")
                    console.print(f"  FTS Text Index: {'[green]✓ Present' if fts_exists else '[red]✗ Missing'}")

        except Exception as e:
            console.print(f"[bold red]Error reading index: {e}")
            raise

    asyncio.run(_do_stats(index))


@cli.command()
@click.argument("query")
@click.option("--limit", type=int, default=10, show_default=True, help="Maximum number of results")
@click.option("--tag", "tags", multiple=True, help="Filter results by tag (repeatable)")
@click.option("--index", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def search(query: str, limit: int, tags: tuple[str, ...], index: Path | None) -> None:
    """Search for finding models using hybrid FTS + semantic search."""
    console = Console()

    async def _do_search() -> None:
        from findingmodel.config import ensure_index_db

        db_path = index or ensure_index_db()

        if not db_path.exists():
            console.print(f"[bold red]Error: Database not found: {db_path}[/bold red]")
            console.print("[yellow]Hint: Use 'oidm-maintain findingmodel build' to create a database.[/yellow]")
            raise SystemExit(1)

        async with Index(db_path=db_path) as idx:
            tag_list = list(tags) if tags else None
            results = await idx.search(query, limit=limit, tags=tag_list)

            if not results:
                console.print(f"[yellow]No results found for: {query}")
                return

            table = Table(title=f'Search Results for "{query}"', show_header=True, header_style="bold cyan")
            table.add_column("ID", style="yellow", width=18)
            table.add_column("Name", style="white")
            table.add_column("Description", style="white")
            table.add_column("Tags", style="cyan")

            for entry in results:
                description = entry.description or ""
                tags_text = ", ".join(entry.tags or [])
                table.add_row(entry.oifm_id, entry.name, description, tags_text)

            console.print(table)
            console.print(f"\n[gray]Total results: {len(results)}")

    try:
        asyncio.run(_do_search())
    except Exception as e:
        console.print(f"[bold red]Error searching: {e}")
        raise


if __name__ == "__main__":
    cli()
