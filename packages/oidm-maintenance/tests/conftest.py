"""Test fixtures for oidm-maintenance."""

from pathlib import Path

import pytest


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Temporary path for test databases."""
    return tmp_path / "test.duckdb"
