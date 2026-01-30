#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["platformdirs", "rich"]
# ///
"""
Verify packages install and work correctly in isolation.

Builds packages, backs up existing databases, tests installation in isolation
using uv's --no-project --find-links, then compares database hashes.
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from platformdirs import user_data_dir
from rich.console import Console

console = Console()

# Package info: (package_name, version, import_name, cli_command, data_subdir, db_filename)
PACKAGES = [
    ("oidm-common", "0.2.0", "oidm_common", None, None, None),
    ("findingmodel", "1.0.0", "findingmodel", "findingmodel", "findingmodel", "finding_models.duckdb"),
    ("anatomic-locations", "0.2.0", "anatomic_locations", "anatomic-locations", "anatomic-locations", "anatomic_locations.duckdb"),
    ("findingmodel-ai", "0.2.0", "findingmodel_ai", "findingmodel-ai", None, None),
]


def file_hash(path: Path) -> str | None:
    """Compute SHA256 hash of a file, or None if it doesn't exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_db_path(data_subdir: str, db_filename: str) -> Path:
    """Get the platform-appropriate database path."""
    return Path(user_data_dir(data_subdir)) / db_filename


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, printing it first."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    return subprocess.run(cmd, check=check)


def run_capture(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and capture output."""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def build_packages() -> None:
    """Build all packages with --no-sources."""
    console.print("\n[bold]=== Building packages (--no-sources) ===[/bold]")
    shutil.rmtree("dist", ignore_errors=True)
    for pkg_name, _, _, _, _, _ in PACKAGES:
        run(["uv", "build", "--package", pkg_name, "--no-sources"])


def backup_databases() -> dict[Path, Path]:
    """Backup existing databases, return mapping of original -> backup paths."""
    console.print("\n[bold]=== Backing up existing databases ===[/bold]")
    backups = {}
    for _, _, _, _, data_subdir, db_filename in PACKAGES:
        if data_subdir and db_filename:
            db_path = get_db_path(data_subdir, db_filename)
            if db_path.exists():
                backup_path = db_path.with_suffix(".duckdb.backup")
                shutil.move(db_path, backup_path)
                console.print(f"  Backed up: {db_path}")
                backups[db_path] = backup_path
    return backups


def restore_databases(backups: dict[Path, Path]) -> None:
    """Restore databases from backups."""
    console.print("\n[bold]=== Restoring databases from backup ===[/bold]")
    for db_path, backup_path in backups.items():
        if backup_path.exists():
            shutil.move(backup_path, db_path)
            console.print(f"  Restored: {db_path}")


def cleanup_backups(backups: dict[Path, Path]) -> None:
    """Remove backup files."""
    for backup_path in backups.values():
        if backup_path.exists():
            backup_path.unlink()


def test_imports() -> None:
    """Test that all packages can be imported."""
    console.print("\n[bold]=== Testing package imports ===[/bold]")
    for pkg_name, version, import_name, _, _, _ in PACKAGES:
        result = run_capture([
            "uv", "run", "--no-project", "--find-links", "dist/",
            "--with", f"{pkg_name}=={version}",
            "--", "python", "-c",
            f"import {import_name}; print(f'{import_name} {{{import_name}.__version__}}')"
        ])
        if result.returncode != 0:
            console.print(f"[red]Failed to import {import_name}[/red]")
            console.print(result.stderr)
            raise RuntimeError(f"Import failed: {import_name}")
        console.print(f"  {result.stdout.strip()}")


def test_cli_entry_points() -> None:
    """Test that CLI entry points work."""
    console.print("\n[bold]=== Testing CLI entry points ===[/bold]")
    for pkg_name, version, _, cli_cmd, _, _ in PACKAGES:
        if cli_cmd:
            result = run_capture([
                "uv", "run", "--no-project", "--find-links", "dist/",
                "--with", f"{pkg_name}=={version}",
                "--", cli_cmd, "--help"
            ])
            if result.returncode != 0:
                console.print(f"[red]Failed: {cli_cmd} --help[/red]")
                console.print(result.stderr)
                raise RuntimeError(f"CLI failed: {cli_cmd}")
            # Print first line of help
            first_line = result.stdout.split("\n")[0]
            console.print(f"  {cli_cmd}: {first_line}")


def test_database_access() -> None:
    """Test database download and access."""
    console.print("\n[bold]=== Testing database download + access ===[/bold]")

    # Test findingmodel
    console.print("\n[dim]findingmodel:[/dim]")
    run([
        "uv", "run", "--no-project", "--find-links", "dist/",
        "--with", "findingmodel==1.0.0",
        "--", "findingmodel", "index", "stats"
    ])

    # Test anatomic-locations
    console.print("\n[dim]anatomic-locations:[/dim]")
    run([
        "uv", "run", "--no-project", "--find-links", "dist/",
        "--with", "anatomic-locations==0.2.0",
        "--", "anatomic-locations", "stats"
    ])


def handle_database_changes(backups: dict[Path, Path]) -> None:
    """Compare database hashes and handle changes."""
    console.print("\n[bold]=== Checking database changes ===[/bold]")

    changes_detected = False
    for db_path, backup_path in backups.items():
        old_hash = file_hash(backup_path)
        new_hash = file_hash(db_path)

        if old_hash and new_hash:
            if old_hash == new_hash:
                console.print(f"  {db_path.name}: [green]unchanged[/green]")
            else:
                console.print(f"  {db_path.name}: [yellow]CHANGED[/yellow]")
                changes_detected = True
        elif not old_hash and new_hash:
            console.print(f"  {db_path.name}: [blue]newly downloaded[/blue]")

    if changes_detected:
        console.print()
        response = console.input("[yellow]Databases changed. Keep new versions?[/yellow] [Y/n] ")
        if response.lower() in ("n", "no"):
            restore_databases(backups)
            console.print("Reverted to previous databases.")
        else:
            cleanup_backups(backups)
            console.print("Keeping new databases.")
    else:
        cleanup_backups(backups)


def main() -> int:
    backups: dict[Path, Path] = {}

    try:
        build_packages()
        backups = backup_databases()
        test_imports()
        test_cli_entry_points()
        test_database_access()

        console.print("\n[bold green]All verification tests passed.[/bold green]")
        handle_database_changes(backups)
        return 0

    except Exception as e:
        console.print(f"\n[bold red]Verification failed: {e}[/bold red]")
        if backups:
            restore_databases(backups)
        return 1


if __name__ == "__main__":
    sys.exit(main())
