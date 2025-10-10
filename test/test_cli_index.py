"""Tests for CLI index commands (build, update, validate, stats)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Generator

from findingmodel import duckdb_index
from findingmodel.cli import cli
from findingmodel.config import settings
from findingmodel.duckdb_index import DuckDBIndex
from findingmodel.finding_model import FindingModelFull


def _fake_openai_client(*_: Any, **__: Any) -> object:  # pragma: no cover - test helper
    """Return a dummy OpenAI client for patched calls."""
    return object()


async def _fake_embedding_deterministic(
    text: str,
    *,
    client: object | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:  # pragma: no cover - test helper
    """Deterministic fake embedding based on text hash."""
    _ = (client, model)
    target_dims = dimensions or settings.openai_embedding_dimensions
    await asyncio.sleep(0)
    # Use simple hash-based embedding for determinism
    hash_val = sum(ord(c) for c in text)
    return [(hash_val % 100) / 100.0] * target_dims


async def _fake_client_for_testing() -> object:  # pragma: no cover - test helper
    """Return fake OpenAI client for testing."""
    await asyncio.sleep(0)
    return _fake_openai_client()


@pytest.fixture(scope="session")
def _session_cli_monkeypatch_setup() -> Generator[None, None, None]:
    """Session-scoped monkeypatch setup for mocking embeddings in CLI tests.

    Note: This patches module-level functions at session start and undoes at session end.
    """
    # Store original functions
    original_get_embedding = duckdb_index.get_embedding_for_duckdb  # type: ignore[attr-defined]
    original_batch_embeddings = duckdb_index.batch_embeddings_for_duckdb  # type: ignore[attr-defined]
    original_ensure_client = DuckDBIndex._ensure_openai_client

    # Patch with fakes
    duckdb_index.get_embedding_for_duckdb = _fake_embedding_deterministic  # type: ignore[attr-defined]
    duckdb_index.batch_embeddings_for_duckdb = lambda texts, client: asyncio.gather(  # type: ignore[attr-defined,assignment,misc]
        *[_fake_embedding_deterministic(t, client=client) for t in texts]
    )
    DuckDBIndex._ensure_openai_client = lambda _: _fake_client_for_testing()  # type: ignore[assignment,return-value,method-assign]

    yield

    # Restore originals
    duckdb_index.get_embedding_for_duckdb = original_get_embedding  # type: ignore[attr-defined]
    duckdb_index.batch_embeddings_for_duckdb = original_batch_embeddings  # type: ignore[attr-defined]
    DuckDBIndex._ensure_openai_client = original_ensure_client  # type: ignore[method-assign]


# ============================================================================
# CLI index build command tests
# ============================================================================


def test_index_build_creates_database(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index build' creates a new database from directory."""
    runner = CliRunner()
    db_path = tmp_path / "test_build.duckdb"

    result = runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    assert result.exit_code == 0
    assert db_path.exists(), "Database file should be created"
    assert "Index built successfully" in result.output
    assert "Added:" in result.output


def test_index_build_with_output_path(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test 'index build' with custom output path."""
    runner = CliRunner()
    db_path = tmp_path / "custom_location.duckdb"

    result = runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    assert result.exit_code == 0
    assert db_path.exists()
    assert "custom_location.duckdb" in result.output


def test_index_build_uses_default_path_from_settings(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index build' uses default path from settings when no --output provided."""
    runner = CliRunner()

    # Temporarily override settings
    with patch.object(settings, "duckdb_index_path", str(tmp_path / "default.duckdb")):
        result = runner.invoke(cli, ["index", "build", str(tmp_defs_path)])

        assert result.exit_code == 0
        assert "Building index at" in result.output
        assert "default.duckdb" in result.output


def test_index_build_empty_directory(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test 'index build' with empty directory (no files)."""
    runner = CliRunner()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    db_path = tmp_path / "empty_test.duckdb"

    result = runner.invoke(cli, ["index", "build", str(empty_dir), "--output", str(db_path)])

    assert result.exit_code == 0
    assert db_path.exists()
    assert "Added: 0" in result.output
    assert "Updated: 0" in result.output
    assert "Removed: 0" in result.output


def test_index_build_displays_correct_counts(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index build' displays correct counts."""
    runner = CliRunner()
    db_path = tmp_path / "counts_test.duckdb"

    # Count files in directory
    fm_files = list(tmp_defs_path.glob("*.fm.json"))
    expected_count = len(fm_files)

    result = runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    assert result.exit_code == 0
    assert f"Added: {expected_count}" in result.output
    assert "Updated: 0" in result.output
    assert "Removed: 0" in result.output


def test_index_build_shows_source_directory(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index build' shows source directory in output."""
    runner = CliRunner()
    db_path = tmp_path / "source_test.duckdb"

    result = runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    assert result.exit_code == 0
    assert "Source directory:" in result.output
    assert "/defs" in result.output  # Just check the directory name is present


def test_index_build_nonexistent_directory(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test 'index build' with nonexistent directory fails gracefully."""
    runner = CliRunner()
    nonexistent_dir = tmp_path / "does_not_exist"
    db_path = tmp_path / "fail_test.duckdb"

    result = runner.invoke(cli, ["index", "build", str(nonexistent_dir), "--output", str(db_path)])

    # Click will fail validation before our code runs
    assert result.exit_code != 0


# ============================================================================
# CLI index update command tests
# ============================================================================


def test_index_update_succeeds_with_existing_database(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index update' successfully updates an existing database."""
    runner = CliRunner()
    db_path = tmp_path / "update_test.duckdb"

    # First, build the index
    result = runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])
    assert result.exit_code == 0

    # Now update it (no changes)
    result = runner.invoke(cli, ["index", "update", str(tmp_defs_path), "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Index updated successfully" in result.output
    assert "Added: 0" in result.output
    assert "Updated: 0" in result.output
    assert "Removed: 0" in result.output


def test_index_update_error_when_database_does_not_exist(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index update' errors when database doesn't exist."""
    runner = CliRunner()
    nonexistent_db = tmp_path / "nonexistent.duckdb"

    result = runner.invoke(cli, ["index", "update", str(tmp_defs_path), "--index", str(nonexistent_db)])

    assert result.exit_code != 0
    assert "Error: Database not found" in result.output
    assert "Use 'index build' to create a new index" in result.output


def test_index_update_with_custom_index_path(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test 'index update' with --index option for custom path."""
    runner = CliRunner()
    db_path = tmp_path / "custom_update.duckdb"

    # Build first
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Update with custom path
    result = runner.invoke(cli, ["index", "update", str(tmp_defs_path), "--index", str(db_path)])

    assert result.exit_code == 0
    assert "custom_update.duckdb" in result.output


def test_index_update_uses_default_path_from_settings(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index update' uses default path from settings when no --index provided."""
    runner = CliRunner()
    default_path = tmp_path / "default_update.duckdb"

    # Build first with default path
    with patch.object(settings, "duckdb_index_path", str(default_path)):
        runner.invoke(cli, ["index", "build", str(tmp_defs_path)])

        # Update with default path
        result = runner.invoke(cli, ["index", "update", str(tmp_defs_path)])

        assert result.exit_code == 0
        assert "Updating index at" in result.output


def test_index_update_displays_correct_counts_after_changes(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index update' displays correct counts after modifications."""
    runner = CliRunner()
    db_path = tmp_path / "counts_update_test.duckdb"

    # Build initial index
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Modify a file
    fm_files = list(tmp_defs_path.glob("*.fm.json"))
    if fm_files:
        model = FindingModelFull.model_validate_json(fm_files[0].read_text())
        model.description = "Updated description for testing"
        fm_files[0].write_text(model.model_dump_json(indent=2, exclude_none=True))

        # Update index
        result = runner.invoke(cli, ["index", "update", str(tmp_defs_path), "--index", str(db_path)])

        assert result.exit_code == 0
        assert "Updated: 1" in result.output


def test_index_update_shows_source_directory(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index update' shows source directory in output."""
    runner = CliRunner()
    db_path = tmp_path / "source_update_test.duckdb"

    # Build first
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Update
    result = runner.invoke(cli, ["index", "update", str(tmp_defs_path), "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Source directory:" in result.output
    assert "/defs" in result.output  # Just check the directory name is present


# ============================================================================
# CLI index validate command tests
# ============================================================================


def test_index_validate_successful_with_no_errors(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index validate' succeeds with valid models (exit code 0)."""
    runner = CliRunner()

    result = runner.invoke(cli, ["index", "validate", str(tmp_defs_path)])

    assert result.exit_code == 0
    assert "models validated successfully" in result.output


def test_index_validate_detects_validation_errors(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index validate' detects validation errors (exit code 1)."""
    runner = CliRunner()

    # Create a directory with an invalid model (duplicate OIFM ID)
    test_dir = tmp_path / "invalid_models"
    test_dir.mkdir()

    # Create two models with the same OIFM ID
    model1 = {
        "oifm_id": "OIFM_TEST_123456",
        "name": "Test Model 1",
        "description": "First test model",
        "attributes": [],
    }
    model2 = {
        "oifm_id": "OIFM_TEST_123456",  # Duplicate ID
        "name": "Test Model 2",
        "description": "Second test model",
        "attributes": [],
    }

    (test_dir / "model1.fm.json").write_text(json.dumps(model1, indent=2))
    (test_dir / "model2.fm.json").write_text(json.dumps(model2, indent=2))

    result = runner.invoke(cli, ["index", "validate", str(test_dir)])

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_index_validate_groups_errors_by_file(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index validate' groups error messages by file."""
    runner = CliRunner()

    # Create a directory with an invalid model
    test_dir = tmp_path / "error_grouping"
    test_dir.mkdir()

    # Create a model missing required fields
    invalid_model = {
        "oifm_id": "OIFM_TEST_999999",
        "name": "Invalid Model",
        # Missing description and attributes
    }

    (test_dir / "invalid.fm.json").write_text(json.dumps(invalid_model, indent=2))

    result = runner.invoke(cli, ["index", "validate", str(test_dir)])

    assert result.exit_code == 1
    assert "invalid.fm.json:" in result.output
    # Should show error symbol
    assert "✗" in result.output


def test_index_validate_empty_directory(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test 'index validate' with empty directory."""
    runner = CliRunner()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(cli, ["index", "validate", str(empty_dir)])

    assert result.exit_code == 0
    assert "No *.fm.json files found" in result.output


def test_index_validate_shows_file_count(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index validate' shows count of files to validate."""
    runner = CliRunner()

    fm_files = list(tmp_defs_path.glob("*.fm.json"))
    expected_count = len(fm_files)

    result = runner.invoke(cli, ["index", "validate", str(tmp_defs_path)])

    assert result.exit_code == 0
    assert f"Found {expected_count} model files to validate" in result.output


def test_index_validate_detects_missing_oifm_id(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index validate' detects models without oifm_id."""
    runner = CliRunner()

    test_dir = tmp_path / "missing_id"
    test_dir.mkdir()

    # Create a base model (without oifm_id)
    base_model = {
        "name": "Base Model",
        "description": "Model without OIFM ID",
        "attributes": [],
    }

    (test_dir / "base_model.fm.json").write_text(json.dumps(base_model, indent=2))

    result = runner.invoke(cli, ["index", "validate", str(test_dir)])

    assert result.exit_code == 1
    assert "Missing oifm_id" in result.output


def test_index_validate_handles_parse_errors(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index validate' handles JSON parse errors gracefully."""
    runner = CliRunner()

    test_dir = tmp_path / "parse_error"
    test_dir.mkdir()

    # Create a file with invalid JSON
    (test_dir / "invalid.fm.json").write_text("{invalid json content")

    result = runner.invoke(cli, ["index", "validate", str(test_dir)])

    assert result.exit_code == 1
    assert "invalid.fm.json:" in result.output
    assert "Parse error" in result.output


# ============================================================================
# CLI index stats command tests
# ============================================================================


def test_index_stats_displays_statistics(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index stats' displays statistics for existing database."""
    runner = CliRunner()
    db_path = tmp_path / "stats_test.duckdb"

    # Build index first
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Index Statistics" in result.output
    assert "Database Summary" in result.output


def test_index_stats_error_when_database_does_not_exist(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index stats' errors when database doesn't exist."""
    runner = CliRunner()
    nonexistent_db = tmp_path / "nonexistent_stats.duckdb"

    result = runner.invoke(cli, ["index", "stats", "--index", str(nonexistent_db)])

    assert result.exit_code != 0
    assert "Error: Database not found" in result.output
    assert "Use 'index build' to create a new index" in result.output


def test_index_stats_shows_model_count(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index stats' shows model count."""
    runner = CliRunner()
    db_path = tmp_path / "model_count_test.duckdb"

    # Build index
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Total Models" in result.output


def test_index_stats_shows_people_count(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index stats' shows people count."""
    runner = CliRunner()
    db_path = tmp_path / "people_count_test.duckdb"
    test_dir = tmp_path / "people_test"
    test_dir.mkdir()

    # Create a model with contributors
    model = {
        "oifm_id": "OIFM_TEST_111111",
        "name": "Test Model With Contributors",
        "description": "Test",
        "attributes": [],
        "contributors": [
            {
                "github_username": "testuser",
                "name": "Test User",
                "email": "test@example.com",
                "organization_code": "TEST",
            }
        ],
    }
    (test_dir / "model.fm.json").write_text(json.dumps(model, indent=2))

    # Build index
    runner.invoke(cli, ["index", "build", str(test_dir), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Total People" in result.output


def test_index_stats_shows_organization_count(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index stats' shows organization count."""
    runner = CliRunner()
    db_path = tmp_path / "org_count_test.duckdb"
    test_dir = tmp_path / "org_test"
    test_dir.mkdir()

    # Create a model with organization contributors
    model = {
        "oifm_id": "OIFM_TEST_222222",
        "name": "Test Model With Org",
        "description": "Test",
        "attributes": [],
        "contributors": [{"code": "TORG", "name": "Test Organization"}],
    }
    (test_dir / "model.fm.json").write_text(json.dumps(model, indent=2))

    # Build index
    runner.invoke(cli, ["index", "build", str(test_dir), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Total Organizations" in result.output


def test_index_stats_shows_file_size(tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test that 'index stats' shows file size in MB."""
    runner = CliRunner()
    db_path = tmp_path / "file_size_test.duckdb"

    # Build index
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "File Size" in result.output
    assert "MB" in result.output


def test_index_stats_shows_hnsw_index_status(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index stats' shows HNSW vector index status."""
    runner = CliRunner()
    db_path = tmp_path / "hnsw_status_test.duckdb"

    # Build index
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Index Status:" in result.output
    assert "HNSW Vector Index:" in result.output


def test_index_stats_shows_fts_index_status(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index stats' shows FTS text index status."""
    runner = CliRunner()
    db_path = tmp_path / "fts_status_test.duckdb"

    # Build index
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "FTS Text Index:" in result.output


def test_index_stats_uses_default_path_from_settings(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that 'index stats' uses default path from settings when no --index provided."""
    runner = CliRunner()
    default_path = tmp_path / "default_stats.duckdb"

    # Build with default path
    with patch.object(settings, "duckdb_index_path", str(default_path)):
        runner.invoke(cli, ["index", "build", str(tmp_defs_path)])

        # Get stats with default path
        result = runner.invoke(cli, ["index", "stats"])

        assert result.exit_code == 0
        assert "Index Statistics" in result.output
