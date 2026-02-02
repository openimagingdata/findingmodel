"""Anatomic location CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from anatomic_locations.index import AnatomicLocationIndex, get_database_stats, run_sanity_check
from anatomic_locations.models import AnatomicLocation


@click.group()
def main() -> None:
    """Anatomic location query and statistics tools."""
    pass


def _resolve_location(
    index: AnatomicLocationIndex,
    query: str,
    console: Console,
) -> AnatomicLocation:
    """Resolve a location from ID or partial name/synonym match.

    Args:
        index: The anatomic location index
        query: Location ID (RID*) or search term
        console: Rich console for output

    Returns:
        The resolved AnatomicLocation

    Raises:
        SystemExit: If location not found or multiple matches
    """
    # If it looks like an ID, try direct lookup first
    if query.upper().startswith("RID"):
        try:
            return index.get(query.upper())
        except KeyError:
            console.print(f"[bold red]Location not found: {query}")
            sys.exit(1)

    # Otherwise, do exact match on description or synonym
    query_lower = query.lower()

    # Check exact description match
    try:
        conn = index._ensure_connection()
        row = conn.execute(
            "SELECT * FROM anatomic_locations WHERE LOWER(description) = ?",
            [query_lower],
        ).fetchone()
        if row:
            return index._row_to_location(row)
    except Exception:
        pass

    # Check synonym match
    try:
        conn = index._ensure_connection()
        row = conn.execute(
            """
            SELECT al.* FROM anatomic_locations al
            JOIN anatomic_synonyms als ON al.id = als.location_id
            WHERE LOWER(als.synonym) = ?
            """,
            [query_lower],
        ).fetchone()
        if row:
            return index._row_to_location(row)
    except Exception:
        pass

    # No exact match found - show suggestions
    console.print(f"[bold red]Location not found: {query}")

    # Try to find similar descriptions
    try:
        conn = index._ensure_connection()
        rows = conn.execute(
            "SELECT id, description FROM anatomic_locations "
            "WHERE LOWER(description) LIKE ? ORDER BY description LIMIT 5",
            [f"%{query_lower}%"],
        ).fetchall()
        if rows:
            console.print("\n[yellow]Did you mean one of these?")
            for rid, desc in rows:
                console.print(f"  [cyan]{rid}[/cyan]: [white]{desc}[/white]")
    except Exception:
        pass

    sys.exit(1)


def _resolve_db_path(db_path: Path | None, console: Console) -> Path:
    """Resolve database path from argument or config."""
    if db_path:
        return db_path
    try:
        from anatomic_locations.config import ensure_anatomic_db

        return ensure_anatomic_db()
    except ImportError:
        console.print("[bold red]Error: --db-path is required (config module not available for default path)")
        raise click.Abort() from None


def _display_stats_table(console: Console, stats_data: dict[str, Any]) -> None:
    """Display the main statistics table."""
    summary_table = Table(title="Database Summary", show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total Records", str(stats_data["total_records"]))
    summary_table.add_row("Records with Vectors", str(stats_data["records_with_vectors"]))
    summary_table.add_row("Records with Hierarchy", str(stats_data["records_with_hierarchy"]))
    summary_table.add_row("Records with Codes", str(stats_data["records_with_codes"]))
    summary_table.add_row("Unique Regions", str(stats_data["unique_regions"]))
    summary_table.add_row("Total Synonyms", str(stats_data["total_synonyms"]))
    summary_table.add_row("Total Codes", str(stats_data["total_codes"]))
    summary_table.add_row("File Size", f"{stats_data['file_size_mb']:.2f} MB")

    console.print(summary_table)


def _display_sanity_results(console: Console, check_result: dict[str, Any]) -> None:
    """Display detailed validation results."""
    check_table = Table(title="Validation Results", show_header=True, header_style="bold cyan")
    check_table.add_column("Check", style="cyan")
    check_table.add_column("Status", justify="center")
    check_table.add_column("Value", style="white")
    check_table.add_column("Expected", style="dim")

    for check in check_result["checks"]:
        status = "[green]✓ PASS[/green]" if check["passed"] else "[red]✗ FAIL[/red]"
        check_table.add_row(check["name"], status, check["value"], check["expected"])

    console.print(check_table)

    if check_result["errors"]:
        console.print("\n[bold red]Errors:")
        for error in check_result["errors"]:
            console.print(f"  [red]• {error}[/red]")

    if check_result["success"]:
        console.print("\n[bold green]✓ All validation checks passed![/bold green]")
    else:
        console.print("\n[bold red]✗ Some validation checks failed[/bold red]")


@main.command("stats")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
@click.option("--detailed", is_flag=True, help="Show detailed validation checks")
def stats(db_path: Path | None, detailed: bool) -> None:
    """Show anatomic location database statistics."""
    console = Console()
    database_path = _resolve_db_path(db_path, console)

    console.print("[bold green]Anatomic Location Database Statistics\n")
    console.print(f"[gray]Database: [yellow]{database_path.absolute()}\n")

    try:
        stats_data = get_database_stats(database_path)
        _display_stats_table(console, stats_data)

        # Display laterality distribution
        console.print("\n[bold cyan]Laterality Distribution:")
        for laterality, count in sorted(
            stats_data["laterality_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            console.print(f"  {laterality or 'NULL'}: {count}")

        # Display code system breakdown
        if stats_data.get("code_systems"):
            console.print("\n[bold cyan]Code Systems:")
            for system, count in stats_data["code_systems"].items():
                console.print(f"  {system}: {count}")

        # Run detailed validation if requested
        if detailed:
            console.print("\n[bold yellow]Running Detailed Validation...\n")
            check_result = run_sanity_check(database_path)
            _display_sanity_results(console, check_result)

    except Exception as e:
        console.print(f"[bold red]Error reading database: {e}")
        raise


@main.command("ancestors")
@click.argument("query")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def ancestors(query: str, db_path: Path | None) -> None:
    """Show containment ancestors for a location (by ID or name)."""
    console = Console()
    database_path = _resolve_db_path(db_path, console)

    try:
        with AnatomicLocationIndex(database_path) as index:
            location = _resolve_location(index, query, console)

            # Display the location itself
            console.print(f"[bold cyan]Location: [white]{location.description} [gray]({location.id})")
            region_str = location.region.value if location.region else "N/A"
            console.print(f"[gray]Region: {region_str}, Type: {location.location_type.value}\n")

            # Get and display ancestors
            ancestors_list = location.get_containment_ancestors()

            if not ancestors_list:
                console.print("[yellow]No ancestors found (this may be a root location)")
                return

            # Create table for ancestors
            table = Table(title="Containment Ancestors (nearest to root)", show_header=True, header_style="bold cyan")
            table.add_column("Level", style="dim", justify="right", width=6)
            table.add_column("ID", style="yellow", width=20)
            table.add_column("Name", style="white")
            table.add_column("Region", style="cyan", width=15)

            # Ancestors are returned from immediate parent to root
            for i, ancestor in enumerate(ancestors_list, 1):
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


@main.command("descendants")
@click.argument("query")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def descendants(query: str, db_path: Path | None) -> None:
    """Show containment descendants for a location (by ID or name)."""
    console = Console()
    database_path = _resolve_db_path(db_path, console)

    try:
        with AnatomicLocationIndex(database_path) as index:
            location = _resolve_location(index, query, console)

            # Display the location itself
            console.print(f"[bold cyan]Location: [white]{location.description} [gray]({location.id})")
            region_str = location.region.value if location.region else "N/A"
            console.print(f"[gray]Region: {region_str}, Type: {location.location_type.value}\n")

            # Get and display descendants
            descendants_list = location.get_containment_descendants()

            if not descendants_list:
                console.print("[yellow]No descendants found (this may be a leaf location)")
                return

            # Create table for descendants
            table = Table(title="Containment Descendants", show_header=True, header_style="bold cyan")
            table.add_column("Depth", style="dim", justify="right", width=6)
            table.add_column("ID", style="yellow", width=20)
            table.add_column("Name", style="white")
            table.add_column("Region", style="cyan", width=15)

            for descendant in descendants_list:
                table.add_row(
                    str(descendant.containment_depth),
                    descendant.id,
                    descendant.description,
                    descendant.region.value if descendant.region else "N/A",
                )

            console.print(table)
            console.print(f"\n[gray]Total descendants: {len(descendants_list)}")

    except Exception as e:
        console.print(f"[bold red]Error querying descendants: {e}")
        raise


@main.command("laterality")
@click.argument("query")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def laterality(query: str, db_path: Path | None) -> None:
    """Show laterality variants for a location (by ID or name)."""
    console = Console()
    database_path = _resolve_db_path(db_path, console)

    try:
        with AnatomicLocationIndex(database_path) as index:
            location = _resolve_location(index, query, console)

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
            for lat in sorted(variants.keys(), key=lambda x: x.value):
                variant = variants[lat]
                table.add_row(
                    lat.value,
                    variant.id,
                    variant.description,
                )

            console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error querying laterality: {e}")
        raise


@main.command("code")
@click.argument("system")
@click.argument("code")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def code(system: str, code: str, db_path: Path | None) -> None:
    """Find locations by external code (e.g., SNOMED, FMA)."""
    console = Console()
    database_path = _resolve_db_path(db_path, console)

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


@main.command("children")
@click.argument("query")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def children(query: str, db_path: Path | None) -> None:
    """Show direct children (containment) for a location (by ID or name)."""
    console = Console()
    database_path = _resolve_db_path(db_path, console)

    try:
        with AnatomicLocationIndex(database_path) as index:
            location = _resolve_location(index, query, console)

            # Display the location itself
            console.print(f"[bold cyan]Location: [white]{location.description} [gray]({location.id})")
            region_str = location.region.value if location.region else "N/A"
            console.print(f"[gray]Region: {region_str}, Type: {location.location_type.value}\n")

            # Get and display children
            children_list = index.get_children_of(location.id)

            if not children_list:
                console.print("[yellow]No children found (this may be a leaf location)")
                return

            # Create table for children
            table = Table(title="Direct Children", show_header=True, header_style="bold cyan")
            table.add_column("ID", style="yellow", width=20)
            table.add_column("Name", style="white")
            table.add_column("Region", style="cyan", width=15)
            table.add_column("Laterality", style="green", width=12)

            for child in children_list:
                table.add_row(
                    child.id,
                    child.description,
                    child.region.value if child.region else "N/A",
                    child.laterality.value,
                )

            console.print(table)
            console.print(f"\n[gray]Total children: {len(children_list)}")

    except Exception as e:
        console.print(f"[bold red]Error querying children: {e}")
        raise


def _get_sort_key_parts(description: str) -> tuple[str, int]:
    """Extract base name and laterality order from description.

    Detects laterality from description prefix for consistent user-visible ordering.
    Groups by base name, then orders: unsided/generic < left < right

    Args:
        description: Location description

    Returns:
        Tuple of (base_name, laterality_order) for sorting
    """
    lower = description.lower()

    # Determine laterality order from prefix
    if lower.startswith("left "):
        order = 1
        base = description[5:]
    elif lower.startswith("right "):
        order = 2
        base = description[6:]
    else:
        order = 0
        base = description

    # Normalize base name by removing common suffixes to group variants
    # e.g., "pectoralis major muscle" -> "pectoralis major"
    base_lower = base.lower()
    for suffix in [" muscle", " structure", " organ"]:
        if base_lower.endswith(suffix):
            base = base[: -len(suffix)]
            base_lower = base.lower()
            break

    return (base_lower.strip(), order)


def _laterality_sort_key(location: AnatomicLocation) -> tuple[str, int]:
    """Create sort key for grouping laterality triads.

    Groups by base name, then orders: generic/unsided < left < right < nonlateral

    Args:
        location: Anatomic location

    Returns:
        Tuple of (base_name, laterality_order) for sorting
    """
    return _get_sort_key_parts(location.description)


def _print_descendant_tree(
    console: Console,
    descendants_list: list[AnatomicLocation],
    location_id: str,
    base_indent: str,
) -> None:
    """Print descendants as a tree structure using proper tree drawing algorithm."""
    # Build parent -> children mapping
    parent_map: dict[str, list[AnatomicLocation]] = {}

    for desc in descendants_list:
        parent_id = desc.containment_parent.id if desc.containment_parent else None
        if parent_id:
            if parent_id not in parent_map:
                parent_map[parent_id] = []
            parent_map[parent_id].append(desc)

    # Sort children at each level to group laterality triads
    for parent_id in parent_map:
        parent_map[parent_id].sort(key=_laterality_sort_key)

    # Recursive function to print tree with proper indentation
    # prefix_parts: list of indentation strings for each level ("│   " or "    ")
    def _print_tree(node_id: str, prefix_parts: list[str]) -> None:
        children = parent_map.get(node_id, [])
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            # Connector: ├── for non-last, └── for last
            connector = "└── " if is_last else "├── "
            # Build full prefix from all ancestor levels
            full_prefix = "".join(prefix_parts)
            console.print(f"{full_prefix}{connector}{child.description} - {child.id}")
            # For this child's children:
            # - If this child is last, use spaces (no vertical line needed)
            # - If this child is not last, use │ to connect to remaining siblings
            child_prefix = "    " if is_last else "│   "
            _print_tree(child.id, [*prefix_parts, child_prefix])

    _print_tree(location_id, [base_indent])


@main.command("hierarchy")
@click.argument("query")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def hierarchy(query: str, db_path: Path | None) -> None:
    """Show full hierarchy: ancestors up to whole body, then descendants tree (by ID or name)."""
    console = Console()
    database_path = _resolve_db_path(db_path, console)

    try:
        with AnatomicLocationIndex(database_path) as index:
            location = _resolve_location(index, query, console)

            # Get ancestors and descendants
            ancestors_list = location.get_containment_ancestors()
            descendants_list = location.get_containment_descendants()

            # Build tree structure
            console.print("\n[bold cyan]Hierarchy Tree:[/bold cyan]\n")

            # Display ancestors from root to immediate parent
            prefix_parts: list[str] = []
            for i, ancestor in enumerate(reversed(ancestors_list)):
                is_last = i == len(ancestors_list) - 1
                is_first = i == 0
                indent = "".join(prefix_parts)
                # Root node (first) has no connector, others use tree connectors
                connector = "" if is_first else ("└── " if is_last else "├── ")
                console.print(f"{indent}{connector}{ancestor.description} - {ancestor.id}")
                # For next level: use │ to continue line if this wasn't the last ancestor
                prefix_parts.append("│   " if not is_last else "    ")

            # Display the index node (highlighted)
            indent = "".join(prefix_parts)
            connector = "└── " if ancestors_list else ""
            console.print(
                f"{indent}{connector}[bold bright_cyan]▶ {location.description} - {location.id} ◀[/bold bright_cyan]"
            )

            # Display descendants as a tree
            if descendants_list:
                _print_descendant_tree(console, descendants_list, location.id, indent + "    ")

            console.print()

    except Exception as e:
        console.print(f"[bold red]Error querying hierarchy: {e}")
        raise


@main.command("search")
@click.argument("query")
@click.option("--limit", type=int, default=10, help="Maximum number of results (default: 10)")
@click.option("--db-path", type=click.Path(path_type=Path), help="Database path (default: config setting)")
def search_cmd(query: str, limit: int, db_path: Path | None) -> None:
    """Search for anatomic locations using hybrid FTS + semantic search."""
    import asyncio

    console = Console()
    database_path = _resolve_db_path(db_path, console)

    async def _do_search() -> None:
        async with AnatomicLocationIndex(database_path) as index:
            results = await index.search(query, limit=limit)

            if not results:
                console.print(f"[yellow]No results found for: {query}")
                return

            # Create table for results
            table = Table(
                title=f'Search Results for "{query}"',
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("ID", style="yellow", width=12)
            table.add_column("Name", style="white")
            table.add_column("Region", style="cyan", width=15)
            table.add_column("Laterality", style="green", width=12)

            for location in results:
                table.add_row(
                    location.id,
                    location.description,
                    location.region.value if location.region else "N/A",
                    location.laterality.value,
                )

            console.print(table)
            console.print(f"\n[gray]Total results: {len(results)}")

    try:
        asyncio.run(_do_search())
    except Exception as e:
        console.print(f"[bold red]Error searching: {e}")
        raise


if __name__ == "__main__":
    main()
