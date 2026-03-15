"""Tests for findingmodel CLI commands.

Focus on wrapper mechanics (path expansion, routing, result aggregation, reformat flow)
rather than re-testing Pydantic's schema behavior.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import findingmodel.cli as cli_module
from click.testing import CliRunner
from findingmodel.cli import cli
from pytest import MonkeyPatch

TEST_DATA_DIR = Path(__file__).parent / "data"
FULL_MODEL_FIXTURE = TEST_DATA_DIR / "pulmonary_embolism.fm.json"


class _FakeValidatedModel:
    """Tiny stand-in for validated models in wrapper-only tests."""

    def model_dump_json(self, **_: Any) -> str:
        return '{\n  "name": "fake"\n}'


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
    """Test top-level command visibility."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "validate" in result.output
    assert "stats" in result.output
    assert "search" in result.output
    # These commands have been removed
    assert "build" not in result.output
    assert "update" not in result.output


# ============================================================================
# Validate Tests
# ============================================================================


def test_validate_recursive_directory_and_dedup(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Validate supports file+directory inputs, recursive scanning, and de-duplication."""
    defs_dir = tmp_path / "defs"
    nested_dir = defs_dir / "nested"
    nested_dir.mkdir(parents=True)
    top_file = defs_dir / "top.fm.json"
    nested_file = nested_dir / "nested.fm.json"
    shutil.copy(FULL_MODEL_FIXTURE, top_file)
    shutil.copy(FULL_MODEL_FIXTURE, nested_file)

    calls: list[tuple[Path, bool]] = []

    def fake_validate_model_json(
        json_text: str, is_full_model: bool
    ) -> tuple[_FakeValidatedModel, str, list[str], list[str]]:
        payload = json.loads(json_text)
        path_hint = payload.get("name", "unknown")
        calls.append((Path(path_hint), is_full_model))
        return _FakeValidatedModel(), "full", [], []

    monkeypatch.setattr(cli_module, "_validate_model_json", fake_validate_model_json)

    runner = CliRunner()
    # top_file appears explicitly and via directory expansion; should only be processed once.
    result = runner.invoke(cli, ["validate", str(top_file), str(defs_dir)])

    assert result.exit_code == 0, result.output
    assert "Summary:" in result.output
    assert "scanned=2" in result.output
    assert "invalid=0" in result.output
    assert len(calls) == 2
    assert all(is_full for _, is_full in calls)


def test_validate_routes_to_base_model_when_ids_absent(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Validation routes to base parser when oifm_id is absent, even with full-only extras."""
    data = json.loads(FULL_MODEL_FIXTURE.read_text())
    data.pop("oifm_id", None)
    data["contributors"] = []
    for attribute in data.get("attributes", []):
        if isinstance(attribute, dict):
            attribute["oifma_id"] = "OIFMA_TEST_123456"
            for idx, value in enumerate(
                attribute.get("values", []) if isinstance(attribute.get("values"), list) else []
            ):
                if isinstance(value, dict):
                    value["value_code"] = f"OIFMA_TEST_123456.{idx}"
    base_file = tmp_path / "base.fm.json"
    base_file.write_text(json.dumps(data, indent=2) + "\n")

    observed_is_full: list[bool] = []

    def fake_validate_model_json(_: str, is_full_model: bool) -> tuple[_FakeValidatedModel, str, list[str], list[str]]:
        observed_is_full.append(is_full_model)
        model_kind = "full" if is_full_model else "base"
        return _FakeValidatedModel(), model_kind, [], []

    monkeypatch.setattr(cli_module, "_validate_model_json", fake_validate_model_json)

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(base_file)])

    assert result.exit_code == 0, result.output
    assert observed_is_full == [False]
    assert "(base)" in result.output


def test_validate_warns_on_unknown_fields(tmp_path: Path) -> None:
    """Unknown fields are warning-only and do not fail validation."""
    data = json.loads(FULL_MODEL_FIXTURE.read_text())
    data["wrapper_test_extra"] = "extra"
    unknown_file = tmp_path / "unknown.fm.json"
    unknown_file.write_text(json.dumps(data, indent=2) + "\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(unknown_file)])

    assert result.exit_code == 0, result.output
    assert "WARN" in result.output
    assert "invalid=0" in result.output


def test_validate_reformat_batch_rewrites_valid_files_even_when_some_invalid(tmp_path: Path) -> None:
    """Batch reformat uses best-effort writes for valid files, exits non-zero if any invalid."""
    valid_data = json.loads(FULL_MODEL_FIXTURE.read_text())
    valid_data["contributors"] = []
    valid_data["unknown_for_reformat"] = "drop me"

    valid_file = tmp_path / "valid.fm.json"
    # Intentionally compact/non-canonical input to ensure rewrite.
    valid_file.write_text(json.dumps(valid_data, separators=(",", ":")))

    invalid_file = tmp_path / "invalid.fm.json"
    invalid_file.write_text("{ not valid json }")

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(valid_file), str(invalid_file), "--reformat"])

    assert result.exit_code == 1, result.output
    updated = valid_file.read_text()
    assert updated.endswith("\n")
    assert '"unknown_for_reformat"' not in updated
    assert '"contributors": []' not in updated
    assert '\n  "oifm_id":' in updated
    assert "reformatted" in result.output


def test_validate_empty_directory_fails(tmp_path: Path) -> None:
    """Directory input with no matching files is an error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(empty_dir)])

    assert result.exit_code == 1
    assert "no '*.fm.json' files found" in result.output
    assert "No files to validate" in result.output
