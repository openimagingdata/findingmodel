import asyncio
import json
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .index import Index
from .types.models import FindingModelBase, FindingModelFull


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
        json_text = f.read()
        payload = json.loads(json_text)
        if _is_full_model_payload(payload):
            fm_full = FindingModelFull.model_validate_json(json_text)
            markdown = fm_full.as_markdown()
        else:
            fm_base = FindingModelBase.model_validate_json(json_text)
            markdown = fm_base.as_markdown()
    if output:
        with open(output, "w") as f:
            f.write(markdown.strip() + "\n")
        console.print(f"[green]Saved Markdown to [yellow]{output}")
    else:
        from rich.markdown import Markdown

        console.print(Markdown(markdown))


def _is_full_model_payload(payload: object) -> bool:
    """Treat payload as full model only when an OIFM ID is present."""
    if not isinstance(payload, dict):
        return False
    return "oifm_id" in payload


def _format_error_location(loc: tuple[Any, ...]) -> str:
    if not loc:
        return "<root>"

    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            if path:
                path += "."
            path += str(part)
    return path


def _validate_model_json(
    json_text: str, is_full_model: bool
) -> tuple[FindingModelBase | FindingModelFull | None, str, list[str], list[str]]:
    """Validate JSON using strict extras first, then retry when only extras are present."""
    model_cls = FindingModelFull if is_full_model else FindingModelBase
    model_kind = "full" if is_full_model else "base"

    try:
        validated = model_cls.model_validate_json(json_text, extra="forbid")
        return validated, model_kind, [], []
    except ValidationError as exc:
        errors = exc.errors()
        extra_errors = [err for err in errors if err.get("type") == "extra_forbidden"]
        non_extra_errors = [err for err in errors if err.get("type") != "extra_forbidden"]
        if non_extra_errors:
            return (
                None,
                model_kind,
                [],
                [
                    f"{_format_error_location(tuple(err.get('loc', ()) or ()))}: {err.get('msg', 'Validation error')}"
                    for err in non_extra_errors
                ],
            )

        warnings = [
            f"{_format_error_location(tuple(err.get('loc', ()) or ()))}: unknown field ignored" for err in extra_errors
        ]
        validated = model_cls.model_validate_json(json_text)
        return validated, model_kind, warnings, []


def _expand_target_files(paths: tuple[Path, ...]) -> tuple[list[Path], list[str]]:
    """Expand input paths to files, recursively resolving directories to *.fm.json."""
    files: list[Path] = []
    errors: list[str] = []
    seen: set[str] = set()

    for input_path in paths:
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = sorted(path for path in input_path.rglob("*.fm.json") if path.is_file())
            if not candidates:
                errors.append(f"{input_path}: no '*.fm.json' files found")
        else:
            errors.append(f"{input_path}: not a file or directory")
            continue

        for candidate in candidates:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(candidate)

    return files, errors


def _optional_empty_list_fields(model: FindingModelBase | FindingModelFull) -> set[str]:
    """Collect optional top-level list fields that are empty and should be omitted."""
    exclude_fields: set[str] = set()
    for name, field in model.__class__.model_fields.items():
        value = getattr(model, name, None)
        if value == [] and not field.is_required() and field.default is None:
            exclude_fields.add(name)
    return exclude_fields


def _canonical_json(model: FindingModelBase | FindingModelFull) -> str:
    exclude = _optional_empty_list_fields(model)
    return model.model_dump_json(exclude_none=True, indent=2, exclude=exclude or None) + "\n"


def _process_validate_file(file_path: Path, reformat: bool, console: Console) -> tuple[bool, int, bool]:
    """Validate one file and optionally rewrite it.

    Returns:
        tuple: (is_valid, warning_count, was_reformatted)
    """
    try:
        json_text = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        console.print(f"[red]FAIL[/red] {file_path}: unable to read file ({exc})")
        return False, 0, False

    try:
        parsed_json = json.loads(json_text)
    except json.JSONDecodeError as exc:
        console.print(f"[red]FAIL[/red] {file_path}: invalid JSON ({exc.msg} at line {exc.lineno}, col {exc.colno})")
        return False, 0, False

    model, model_kind, warnings, errors = _validate_model_json(json_text, _is_full_model_payload(parsed_json))
    if errors:
        console.print(f"[red]FAIL[/red] {file_path} ({model_kind})")
        for error in errors:
            console.print(f"  [red]- {error}[/red]")
        return False, 0, False

    for warning in warnings:
        console.print(f"[yellow]WARN[/yellow] {file_path}: {warning}")

    if not reformat:
        console.print(f"[green]OK[/green] {file_path} ({model_kind})")
        return True, len(warnings), False

    if model is None:
        console.print(f"[red]FAIL[/red] {file_path}: internal validation state error")
        return False, len(warnings), False

    formatted = _canonical_json(model)
    if formatted != json_text:
        try:
            file_path.write_text(formatted, encoding="utf-8")
        except Exception as exc:
            console.print(f"[red]FAIL[/red] {file_path}: unable to write file ({exc})")
            return False, len(warnings), False
        console.print(f"[green]OK[/green] {file_path} ({model_kind}) [cyan]reformatted[/cyan]")
        return True, len(warnings), True

    console.print(f"[green]OK[/green] {file_path} ({model_kind}) [dim]unchanged[/dim]")
    return True, len(warnings), False


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--reformat", is_flag=True, help="Rewrite valid files in canonical JSON format.")
def validate(paths: tuple[Path, ...], reformat: bool) -> None:
    """Validate finding model JSON files or directories of *.fm.json files."""
    console = Console()
    files, path_errors = _expand_target_files(paths)

    total = len(files)
    valid = 0
    invalid = 0
    warnings_count = 0
    reformatted = 0

    for error in path_errors:
        invalid += 1
        console.print(f"[red]FAIL[/red] {error}")

    if not files:
        console.print("[bold red]No files to validate.")
        raise SystemExit(1)

    for file_path in files:
        is_valid, warning_count, was_reformatted = _process_validate_file(file_path, reformat=reformat, console=console)
        if is_valid:
            valid += 1
            warnings_count += warning_count
            if was_reformatted:
                reformatted += 1
        else:
            invalid += 1

    console.print(
        "\n"
        f"[bold]Summary:[/bold] scanned={total}, valid={valid}, invalid={invalid}, "
        f"warnings={warnings_count}, reformatted={reformatted}"
    )

    if invalid > 0:
        raise SystemExit(1)


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
