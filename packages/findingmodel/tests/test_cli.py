"""Minimal tests for CLI commands.

Note: The build, update, and validate commands have been moved to oidm-maintenance.
This file now only tests the stats command which remains in findingmodel.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from findingmodel.cli import cli

# ============================================================================
# Index Stats Tests
# ============================================================================


def test_index_stats_basic() -> None:
    """Test happy path: show stats for existing database."""
    runner = CliRunner()

    # Use the pre-built test database
    db_path = Path(__file__).parent / "data" / "test_index.duckdb"

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Index Statistics" in result.output
    assert "Database Summary" in result.output


def test_index_stats_with_missing_database(tmp_path: Path) -> None:
    """Test stats errors gracefully when database doesn't exist."""
    runner = CliRunner()
    nonexistent_db = tmp_path / "nonexistent_stats.duckdb"

    # Database doesn't exist initially
    assert not nonexistent_db.exists()

    result = runner.invoke(cli, ["index", "stats", "--index", str(nonexistent_db)])

    # Should error and provide helpful message
    assert result.exit_code == 1
    assert "Database not found" in result.output
    assert "oidm-maintain findingmodel build" in result.output


def test_index_help() -> None:
    """Test that index subcommand shows only stats."""
    runner = CliRunner()
    result = runner.invoke(cli, ["index", "--help"])

    assert result.exit_code == 0
    assert "stats" in result.output
    # These commands have been removed
    assert "build" not in result.output
    assert "update" not in result.output
    assert "validate" not in result.output
