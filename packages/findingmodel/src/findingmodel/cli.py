import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import settings
from .db_publish import DatabaseStats, ManifestUpdateInfo
from .finding_info import FindingInfo
from .finding_model import FindingModelBase, FindingModelFull
from .index import DuckDBIndex
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
        from findingmodel.config import ensure_index_db

        db_path = output or ensure_index_db()
        console.print(f"[bold green]Building index at [yellow]{db_path}")
        console.print(f"[gray]Source directory: [yellow]{directory.absolute()}")

        def progress_update(message: str) -> None:
            console.print(f"[cyan]→[/cyan] {message}")

        try:
            async with DuckDBIndex(db_path=db_path, read_only=False) as idx:
                await idx.setup()
                result = await idx.update_from_directory(directory, progress_callback=progress_update)

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
        from findingmodel.config import ensure_index_db

        db_path = index or ensure_index_db()

        console.print(f"[bold green]Updating index at [yellow]{db_path}")
        console.print(f"[gray]Source directory: [yellow]{directory.absolute()}")

        def progress_update(message: str) -> None:
            console.print(f"[cyan]→[/cyan] {message}")

        try:
            async with DuckDBIndex(db_path=db_path, read_only=False) as idx:
                result = await idx.update_from_directory(directory, progress_callback=progress_update)

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


async def _build_database_from_definitions(defs_dir: Path, console: Console) -> Path:
    """Build database from definitions directory to temporary file."""
    import os
    import tempfile

    console.print(f"[bold green]Building database from [yellow]{defs_dir.absolute()}")

    # Create temporary file for build
    temp_fd, temp_file_str = tempfile.mkstemp(suffix=".duckdb")
    os.close(temp_fd)  # Close file descriptor, DuckDB will open it
    temp_db_path = Path(temp_file_str)

    console.print(f"[gray]Building to temporary file: [yellow]{temp_db_path}")

    def progress_update(message: str) -> None:
        console.print(f"[cyan]→[/cyan] {message}")

    # Use existing index build logic
    async with DuckDBIndex(db_path=temp_db_path, read_only=False) as idx:
        await idx.setup()
        result = await idx.update_from_directory(defs_dir, progress_callback=progress_update)

    console.print(
        f"[green]✓ Built: {result['added']} added, {result['updated']} updated, {result['removed']} removed\n"
    )
    return temp_db_path


def _run_sanity_check_with_confirmation(db_path: Path, console: Console) -> None:
    """Run sanity check and prompt for user confirmation."""
    from findingmodel.db_publish import prompt_user_confirmation, run_sanity_check

    console.print("[bold cyan]Running sanity check...")
    check_result = run_sanity_check(db_path)

    response = prompt_user_confirmation(check_result)

    if response == "no":
        console.print("[yellow]Upload cancelled by user (answered 'no')")
        sys.exit(0)
    elif response == "cancel":
        console.print("[yellow]Upload cancelled by user")
        sys.exit(0)
    # response == "yes", continue
    console.print("[green]✓ Sanity check passed, continuing...\n")


def _display_publication_summary(
    db_url: str,
    hash_value: str,
    stats: DatabaseStats,
    update_info: ManifestUpdateInfo,
    manifest_url: str,
    backup_url: str,
    console: Console,
) -> None:
    """Display final publication summary."""
    console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[bold green]✓ Publication Complete!")
    console.print("[bold green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    console.print("[cyan]Published Database:")
    console.print(f"  URL: {db_url}")
    console.print(f"  Records: {stats.record_count:,}")
    console.print(f"  Size: {stats.size_bytes:,} bytes ({stats.size_bytes / 1024 / 1024:.2f} MB)")
    console.print(f"  Version: {update_info.version}")
    console.print(f"  Hash: {hash_value}\n")

    console.print("[cyan]Manifest:")
    console.print(f"  Updated: {manifest_url}")
    console.print(f"  Backup: {backup_url}\n")


@index.command("publish")
@click.option(
    "--defs-dir", type=click.Path(exists=True, path_type=Path), help="Directory with *.fm.json files (build mode)"
)
@click.option(
    "--database", type=click.Path(exists=True, path_type=Path), help="Existing database file (publish-only mode)"
)
@click.option("--skip-checks", is_flag=True, help="Skip sanity checks and confirmation")
def publish(defs_dir: Path | None, database: Path | None, skip_checks: bool) -> None:
    """Publish database to S3 with automatic manifest update.

    Two modes:
    1. Build and Publish: --defs-dir path/to/definitions
    2. Publish Existing: --database path/to/existing.duckdb
    """

    console = Console()

    async def _do_publish(defs_dir: Path | None, database: Path | None, skip_checks: bool) -> None:
        from datetime import datetime, timezone

        from findingmodel.db_publish import (
            ManifestUpdateInfo,
            PublishConfig,
            compute_file_hash,
            create_s3_client,
            get_database_stats,
            update_and_publish_manifest,
        )

        # Step 1: Validate args (exactly one of --defs-dir or --database required)
        if (defs_dir is None) == (database is None):
            console.print("[bold red]Error: Provide exactly one of --defs-dir or --database")
            sys.exit(1)

        temp_db_path: Path | None = None
        db_path: Path

        try:
            # Step 2: Build database to temp file if defs-dir provided
            if defs_dir:
                temp_db_path = await _build_database_from_definitions(defs_dir, console)
                db_path = temp_db_path
            else:
                # Publish existing database
                assert database is not None
                db_path = database
                console.print(f"[bold green]Publishing existing database: [yellow]{db_path.absolute()}\n")

            # Step 3: Run sanity check unless --skip-checks
            if not skip_checks:
                _run_sanity_check_with_confirmation(db_path, console)
            else:
                console.print("[yellow]Skipping sanity checks (--skip-checks enabled)\n")

            # Step 4: Compute hash
            console.print("[bold cyan]Computing database hash...")
            hash_value = compute_file_hash(db_path)
            console.print(f"[green]✓ Hash: {hash_value}\n")

            # Step 5: Create S3 client and PublishConfig
            console.print("[bold cyan]Initializing S3 connection...")
            try:
                publish_config = PublishConfig()
                s3_client = create_s3_client(publish_config)
                console.print(f"[green]✓ Connected to {publish_config.s3_endpoint_url}/{publish_config.s3_bucket}\n")
            except ValueError as e:
                console.print(f"[bold red]Error: {e}")
                console.print("\n[yellow]Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
                sys.exit(1)

            # Step 6: Generate database filename with current date
            today = datetime.now(timezone.utc)
            db_filename = f"findingmodels_{today:%Y%m%d}.duckdb"
            console.print(f"[bold cyan]Database filename: [yellow]{db_filename}")

            # Step 7: Upload database to S3
            console.print("[bold cyan]Uploading database to S3...")
            s3_client.upload_file(str(db_path), publish_config.s3_bucket, db_filename)

            # Construct database URL
            endpoint_url = s3_client.meta.endpoint_url
            db_url = f"https://{publish_config.s3_bucket}.{endpoint_url.removeprefix('https://').removeprefix('http://')}/{db_filename}"
            console.print(f"[green]✓ Database uploaded: {db_url}\n")

            # Step 8: Update and publish manifest
            console.print("[bold cyan]Updating manifest...")

            # Get database stats for manifest
            stats = get_database_stats(db_path)

            # Create manifest update info
            update_info = ManifestUpdateInfo(
                database_key="finding_models",
                version=f"{today:%Y-%m-%d}",  # ISO date format with hyphens
                url=db_url,
                hash_value=hash_value,
                size_bytes=stats.size_bytes,
                record_count=stats.record_count,
                description="Finding model index with embeddings and full JSON",
            )

            manifest_url, backup_url = update_and_publish_manifest(s3_client, publish_config, update_info)
            console.print(f"[green]✓ Manifest updated: {manifest_url}")
            console.print(f"[green]✓ Backup created: {backup_url}\n")

            # Display final summary
            _display_publication_summary(db_url, hash_value, stats, update_info, manifest_url, backup_url, console)

        except Exception as e:
            console.print(f"\n[bold red]Publication failed: {e}")
            raise

        finally:
            # Clean up temp database file if built
            if temp_db_path and temp_db_path.exists():
                console.print(f"[gray]Cleaning up temporary database: {temp_db_path}")
                temp_db_path.unlink()

    asyncio.run(_do_publish(defs_dir, database, skip_checks))


@index.command()
@click.option("--index", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def stats(index: Path | None) -> None:
    """Show index statistics."""

    console = Console()

    async def _do_stats(index: Path | None) -> None:
        from findingmodel.config import ensure_index_db

        if index:
            db_path = index
            # If custom path doesn't exist, create empty database first
            if not db_path.exists():
                # Create temporary non-read-only index to initialize database
                async with DuckDBIndex(db_path=db_path, read_only=False) as temp_idx:
                    await temp_idx.setup()  # This will create schema and load base contributors
        else:
            db_path = ensure_index_db()

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
