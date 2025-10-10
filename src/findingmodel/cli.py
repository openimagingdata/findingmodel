import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import settings
from .duckdb_index import DuckDBIndex
from .finding_info import FindingInfo
from .finding_model import FindingModelBase, FindingModelFull
from .tools import (
    add_ids_to_finding_model,
    add_standard_codes_to_finding_model,
    create_finding_model_from_markdown,
    create_finding_model_stub_from_finding_info,
    describe_finding_name,
    get_detail_on_finding,
)


@click.group()
def cli() -> None:
    pass


@cli.command()
def config() -> None:
    """Show the currently active configuration."""
    console = Console()
    console.print("[yellow bold]Finding Model Forge configuration:")
    console.print_json(settings.model_dump_json())


def print_info_truncate_detail(console: Console, finding_info: FindingInfo) -> None:
    out = finding_info.model_dump()
    if out.get("detail") and len(out["detail"]) > 100:
        out["detail"] = out["detail"][0:100] + "..."
    console.print(out)


@cli.command()
@click.argument("finding_name", default="Pneumothorax")
@click.option("--detailed", "-d", is_flag=True, help="Get detailed information on the finding.")
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding info."
)
def make_info(finding_name: str, detailed: bool, output: Path | None) -> None:
    """Generate description/synonyms and more details/citations for a finding name."""

    console = Console()

    async def _do_make_info(finding_name: str, detailed: bool, output: Path | None) -> None:
        with console.status("[bold green]Getting description and synonyms..."):
            described_finding = await describe_finding_name(finding_name)
        if not isinstance(described_finding, FindingInfo):
            raise ValueError("Finding info not returned.")
        if detailed:
            with console.status("Getting detailed information... "):
                detailed_response = await get_detail_on_finding(described_finding)
            if not isinstance(detailed_response, FindingInfo):
                raise ValueError("Detailed finding info not returned.")
            described_finding = detailed_response
        if output:
            with open(output, "w") as f:
                f.write(described_finding.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding info to [yellow]{output}")
        else:
            print_info_truncate_detail(console, described_finding)

    asyncio.run(_do_make_info(finding_name, detailed, output))


@cli.command()
@click.argument("finding_name", default="Pneumothorax")
@click.option("--tags", "-t", multiple=True, help="Tags to add to the model.")
@click.option("--with-codes", "-c", is_flag=True, help="Include standard index codes in the model.")
@click.option("--with-ids", "-i", is_flag=True, help="Include OIFM IDs in the model.")
@click.option("--source", "-s", help="Three/four letter code of originating organization (required for IDs).")
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding model."
)
def make_stub_model(
    finding_name: str, tags: list[str], with_codes: bool, with_ids: bool, source: str | None, output: Path | None
) -> None:
    """Generate a simple finding model object (presence and change elements only) from a finding name."""

    console = Console()

    async def _do_make_stub_model(
        finding_name: str, tags: list[str], with_codes: bool, with_ids: bool, source: str | None, output: Path | None
    ) -> None:
        console.print(f"[gray] Getting stub model for [yellow bold]{finding_name}")
        # Get it from the database if it's already there
        with console.status("[bold green]Getting description and synonyms..."):
            described_finding = await describe_finding_name(finding_name)
        assert isinstance(described_finding, FindingInfo)
        stub = create_finding_model_stub_from_finding_info(described_finding, tags)
        if with_ids:
            if source and len(source) in [3, 4]:
                stub = add_ids_to_finding_model(stub, source.upper())  # type: ignore
            else:
                console.print("[red]Error: --source is required to generate IDs")
            if with_codes:
                add_standard_codes_to_finding_model(stub)  # type: ignore
        if with_codes and not with_ids:
            console.print("[red]Error: --with-codes requires --with-ids to be set")
        if output:
            with open(output, "w") as f:
                f.write(stub.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding model to [yellow]{output}")
        else:
            console.print_json(stub.model_dump_json(indent=2, exclude_none=True))

    asyncio.run(_do_make_stub_model(finding_name, tags, with_codes, with_ids, source, output))


@cli.command()
# Indicate that the argument should be a filename
@click.argument("finding_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(exists=False, dir_okay=True), help="Output file to save the finding info."
)
@click.option("--with-ids", "-i", is_flag=True, help="Include OIFM IDs in the model.")
@click.option("--source", "-s", help="Three/four letter code of originating organization (required for IDs).")
def markdown_to_fm(finding_path: Path, with_ids: bool, source: str | None, output: Path | None) -> None:
    """Convert markdown file to finding model format."""

    console = Console()

    async def _do_markdown_to_fm(finding_path: Path, with_ids: bool, source: str | None, output: Path | None) -> None:
        finding_name = finding_path.stem.replace("_", " ").replace("-", " ")
        with console.status("[bold green]Getting description..."):
            described_finding = await describe_finding_name(finding_name)
        print_info_truncate_detail(console, described_finding)
        assert isinstance(described_finding, FindingInfo), "Finding info not returned."

        with console.status("Creating model from Markdown description..."):
            model: FindingModelBase | FindingModelFull = await create_finding_model_from_markdown(
                described_finding, markdown_path=finding_path
            )
        if with_ids:
            if source and len(source) in [3, 4]:
                assert isinstance(model, FindingModelBase)
                model = add_ids_to_finding_model(model, source.upper())
            else:
                console.print("[red]Error: --source is required to generate IDs")
        if output:
            with open(output, "w") as f:
                f.write(model.model_dump_json(indent=2, exclude_none=True))
            console.print(f"[green]Saved finding model to [yellow]{output}")
        else:
            console.print_json(model.model_dump_json(indent=2, exclude_none=True))

    asyncio.run(_do_markdown_to_fm(finding_path, with_ids, source, output))


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


@cli.group()
def index() -> None:
    """Index management commands."""
    pass


@index.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output database path (default: config setting)")
def build(directory: Path, output: Path | None) -> None:
    """Build index from directory of *.fm.json files."""

    console = Console()

    async def _do_build(directory: Path, output: Path | None) -> None:
        db_path = output if output else Path(settings.duckdb_index_path)
        console.print(f"[bold green]Building index at [yellow]{db_path}")
        console.print(f"[gray]Source directory: [yellow]{directory.absolute()}")

        try:
            with console.status("[bold green]Creating database and scanning directory..."):
                async with DuckDBIndex(db_path=db_path, read_only=False) as idx:
                    await idx.setup()
                    result = await idx.update_from_directory(directory)

            # Display results with color coding
            console.print("\n[bold green]Index built successfully!")
            console.print(f"[green]✓ Added: {result['added']}")
            console.print(f"[yellow]✓ Updated: {result['updated']}")
            console.print(f"[red]✓ Removed: {result['removed']}")

        except Exception as e:
            console.print(f"[bold red]Error building index: {e}")
            raise

    asyncio.run(_do_build(directory, output))


@index.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--index", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def update(directory: Path, index: Path | None) -> None:
    """Update existing index from directory."""

    console = Console()

    async def _do_update(directory: Path, index: Path | None) -> None:
        db_path = index if index else Path(settings.duckdb_index_path)

        # Check if database exists
        if not db_path.exists():
            console.print(f"[bold red]Error: Database not found at {db_path}")
            console.print("[yellow]Use 'index build' to create a new index.")
            raise click.Abort()

        console.print(f"[bold green]Updating index at [yellow]{db_path}")
        console.print(f"[gray]Source directory: [yellow]{directory.absolute()}")

        try:
            with console.status("[bold green]Scanning directory and updating index..."):
                async with DuckDBIndex(db_path=db_path, read_only=False) as idx:
                    result = await idx.update_from_directory(directory)

            # Display results with color coding
            console.print("\n[bold green]Index updated successfully!")
            console.print(f"[green]✓ Added: {result['added']}")
            console.print(f"[yellow]✓ Updated: {result['updated']}")
            console.print(f"[red]✓ Removed: {result['removed']}")

        except Exception as e:
            console.print(f"[bold red]Error updating index: {e}")
            raise

    asyncio.run(_do_update(directory, index))


async def _validate_single_file(fm_file: Path, directory: Path, idx: DuckDBIndex) -> tuple[str, list[str]] | None:
    """Validate a single finding model file. Returns (filename, errors) if validation fails, None if successful."""
    try:
        # Read and parse model
        with open(fm_file, "r") as f:
            model_data = json.load(f)

        # Check if it's a full model with oifm_id
        if "oifm_id" not in model_data:
            return (str(fm_file.relative_to(directory)), ["Missing oifm_id (not a full model)"])

        model = FindingModelFull.model_validate(model_data)

        # Validate using the index
        errors = idx._validate_model(model)
        if errors:
            return (str(fm_file.relative_to(directory)), errors)

        # Add to index for subsequent validation (check for conflicts)
        await idx.add_or_update_entry_from_file(fm_file)
        return None

    except Exception as e:
        return (str(fm_file.relative_to(directory)), [f"Parse error: {e}"])


@index.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
def validate(directory: Path) -> None:
    """Validate finding models without writing to index."""

    console = Console()

    async def _do_validate(directory: Path) -> None:
        console.print(f"[bold green]Validating models in [yellow]{directory.absolute()}")

        # Collect all *.fm.json files
        fm_files = sorted(directory.glob("**/*.fm.json"))
        if not fm_files:
            console.print("[yellow]No *.fm.json files found in directory.")
            return

        console.print(f"[gray]Found {len(fm_files)} model files to validate\n")

        # Create temporary index for validation context
        import tempfile

        validation_errors: dict[str, list[str]] = {}

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_db = Path(temp_dir) / "validation.duckdb"

                with console.status("[bold green]Loading models and validating..."):
                    async with DuckDBIndex(db_path=temp_db, read_only=False) as idx:
                        await idx.setup()

                        # Load and validate each file
                        for fm_file in fm_files:
                            result = await _validate_single_file(fm_file, directory, idx)
                            if result:
                                filename, errors = result
                                validation_errors[filename] = errors

        except Exception as e:
            console.print(f"[bold red]Error during validation: {e}")
            raise

        # Display results
        if validation_errors:
            console.print(f"[bold red]Validation failed for {len(validation_errors)} file(s):\n")
            for filename, errors in validation_errors.items():
                console.print(f"[yellow]{filename}:")
                for error in errors:
                    console.print(f"  [red]✗ {error}")
                console.print()
            sys.exit(1)
        else:
            console.print(f"[bold green]✓ All {len(fm_files)} models validated successfully!")

    asyncio.run(_do_validate(directory))


@index.command()
@click.option("--index", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def stats(index: Path | None) -> None:
    """Show index statistics."""

    console = Console()

    async def _do_stats(index: Path | None) -> None:
        db_path = index if index else Path(settings.duckdb_index_path)

        if not db_path.exists():
            console.print(f"[bold red]Error: Database not found at {db_path}")
            console.print("[yellow]Use 'index build' to create a new index.")
            raise click.Abort()

        console.print(f"[bold green]Index Statistics for [yellow]{db_path}\n")

        try:
            async with DuckDBIndex(db_path=db_path, read_only=True) as idx:
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


if __name__ == "__main__":
    cli()
