"""Minimal tests for CLI commands.

Note: The build, update, and validate commands have been moved to oidm-maintenance.
This file tests the top-level stats and search commands that remain in findingmodel.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from findingmodel.cli import cli

# ============================================================================
# Stats Tests
# ============================================================================


def test_stats_basic() -> None:
    """Test happy path: show stats for existing database."""
    runner = CliRunner()

    # Use the pre-built test database
    db_path = Path(__file__).parent / "data" / "test_index.duckdb"

    # Get stats
    result = runner.invoke(cli, ["stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Index Statistics" in result.output
    assert "Database Summary" in result.output


def test_stats_with_missing_database(tmp_path: Path) -> None:
    """Test stats errors gracefully when database doesn't exist."""
    runner = CliRunner()
    nonexistent_db = tmp_path / "nonexistent_stats.duckdb"

    # Database doesn't exist initially
    assert not nonexistent_db.exists()

    result = runner.invoke(cli, ["stats", "--index", str(nonexistent_db)])

    # Should error and provide helpful message
    assert result.exit_code == 1
    assert "Database not found" in result.output
    assert "oidm-maintain findingmodel build" in result.output


def test_cli_help() -> None:
    """Test that CLI shows stats and search as top-level commands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "stats" in result.output
    assert "search" in result.output
    # These commands have been removed
    assert "build" not in result.output
    assert "update" not in result.output
    assert "validate" not in result.output
