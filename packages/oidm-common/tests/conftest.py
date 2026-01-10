"""Shared test fixtures for oidm-common tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pytest
from oidm_common.duckdb import setup_duckdb_connection
from oidm_common.models.index_code import IndexCode
from oidm_common.models.web_reference import WebReference


@pytest.fixture
def tmp_duckdb_path(tmp_path: Path) -> Path:
    """Provide a temporary path for DuckDB test databases."""
    return tmp_path / "test.duckdb"


@pytest.fixture
def sample_index_codes() -> list[IndexCode]:
    """Provide sample IndexCode instances for testing."""
    return [
        IndexCode(system="SNOMED", code="123456", display="Example Finding"),
        IndexCode(system="RadLex", code="RID001", display="Anatomical Structure"),
        IndexCode(system="LOINC", code="12345-6"),  # No display
    ]


@pytest.fixture
def sample_web_references() -> list[WebReference]:
    """Provide sample WebReference instances for testing."""
    return [
        WebReference(
            url="https://radiopaedia.org/articles/example",
            title="Example Article",
            description="A sample medical reference",
            accessed_date=date(2025, 1, 1),
        ),
        WebReference(
            url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/",
            title="Research Paper on Imaging",
            content="AI-extracted content from the page",
            published_date="2024-12-15",
            accessed_date=date(2025, 1, 5),
        ),
    ]


@pytest.fixture
def tavily_search_result() -> dict[str, Any]:
    """Provide a sample Tavily search result for testing."""
    return {
        "url": "https://example.com/article",
        "title": "Medical Imaging Reference",
        "content": "Extracted content about medical imaging findings",
        "published_date": "2024-06-15",
    }


@pytest.fixture
def temp_duckdb_with_test_table(tmp_duckdb_path: Path) -> duckdb.DuckDBPyConnection:
    """Create a temporary DuckDB connection with a test table for index tests."""
    conn = setup_duckdb_connection(tmp_duckdb_path, read_only=False)

    # Create test table with text and vector columns
    conn.execute("""
        CREATE TABLE test_table (
            id VARCHAR PRIMARY KEY,
            title VARCHAR,
            description VARCHAR,
            embedding FLOAT[3]
        )
    """)

    # Insert sample data
    conn.execute("""
        INSERT INTO test_table VALUES
        ('item1', 'First Item', 'Description of first item', [0.1, 0.2, 0.3]),
        ('item2', 'Second Item', 'Description of second item', [0.4, 0.5, 0.6]),
        ('item3', 'Third Item', 'Another description', [0.7, 0.8, 0.9])
    """)

    yield conn
    conn.close()
