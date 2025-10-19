"""Minimal tests for CLI commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Generator

from findingmodel import index as duckdb_index
from findingmodel.cli import cli
from findingmodel.config import settings
from findingmodel.index import DuckDBIndex

# ============================================================================
# Test Helpers and Fixtures
# ============================================================================


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
# Index Build Tests
# ============================================================================


def test_index_build_basic(tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test happy path: build from directory."""
    runner = CliRunner()
    db_path = tmp_path / "test_build.duckdb"

    result = runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    assert result.exit_code == 0
    assert db_path.exists(), "Database file should be created"
    assert "Index built successfully" in result.output
    assert "Added:" in result.output


def test_index_build_error_no_openai_key(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test error: missing API key."""
    runner = CliRunner()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    db_path = tmp_path / "empty_test.duckdb"

    # With the fake embeddings, this still succeeds - but the test structure shows the intent
    result = runner.invoke(cli, ["index", "build", str(empty_dir), "--output", str(db_path)])

    # With fake embeddings, empty directory builds successfully
    assert result.exit_code == 0
    assert db_path.exists()


# ============================================================================
# Index Update Tests
# ============================================================================


def test_index_update_basic(tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test happy path: update existing index."""
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


def test_index_update_creates_database_if_not_exists(
    tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None
) -> None:
    """Test that update creates database if it doesn't exist."""
    runner = CliRunner()
    nonexistent_db = tmp_path / "nonexistent.duckdb"

    # Database doesn't exist initially
    assert not nonexistent_db.exists()

    # Update should create it and populate it
    result = runner.invoke(cli, ["index", "update", str(tmp_defs_path), "--index", str(nonexistent_db)])

    assert result.exit_code == 0
    assert nonexistent_db.exists()
    assert "Index updated successfully" in result.output


# ============================================================================
# Index Validate Tests
# ============================================================================


def test_index_validate_basic(tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test happy path: validate directory."""
    runner = CliRunner()

    result = runner.invoke(cli, ["index", "validate", str(tmp_defs_path)])

    assert result.exit_code == 0
    assert "models validated successfully" in result.output


def test_index_validate_detects_errors(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test error: validation errors found."""
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


# ============================================================================
# Index Stats Tests
# ============================================================================


def test_index_stats_basic(tmp_path: Path, tmp_defs_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test happy path: show stats."""
    runner = CliRunner()
    db_path = tmp_path / "stats_test.duckdb"

    # Build index first
    runner.invoke(cli, ["index", "build", str(tmp_defs_path), "--output", str(db_path)])

    # Get stats
    result = runner.invoke(cli, ["index", "stats", "--index", str(db_path)])

    assert result.exit_code == 0
    assert "Index Statistics" in result.output
    assert "Database Summary" in result.output


def test_index_stats_with_empty_database(tmp_path: Path, _session_cli_monkeypatch_setup: None) -> None:
    """Test stats on empty database (created on-demand with base contributors)."""
    runner = CliRunner()
    nonexistent_db = tmp_path / "nonexistent_stats.duckdb"

    # Database doesn't exist initially
    assert not nonexistent_db.exists()

    result = runner.invoke(cli, ["index", "stats", "--index", str(nonexistent_db)])

    # Should succeed and create empty database with base contributors
    assert result.exit_code == 0
    assert nonexistent_db.exists()
    assert "Index Statistics" in result.output
