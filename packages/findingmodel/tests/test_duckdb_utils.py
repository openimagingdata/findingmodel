"""Tests for DuckDB utility functions."""

from __future__ import annotations

import duckdb
import pytest
from oidm_common.duckdb import (
    create_fts_index,
    create_hnsw_index,
    drop_search_indexes,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _fts_index_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Check if FTS index exists for a table."""
    try:
        # Try to use the FTS index - if it exists, this won't raise
        conn.execute(f"SELECT fts_main_{table}.match_bm25(id, 'test') FROM {table} LIMIT 1").fetchall()
        return True
    except duckdb.Error:
        return False


def _hnsw_index_exists(conn: duckdb.DuckDBPyConnection, index_name: str) -> bool:
    """Check if HNSW index exists by name."""
    rows = conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
    return any(row[0] == index_name for row in rows)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def memory_conn() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with extensions loaded."""
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL fts")
    conn.execute("LOAD fts")
    conn.execute("INSTALL vss")
    conn.execute("LOAD vss")
    conn.execute("SET hnsw_enable_experimental_persistence = true")
    return conn


@pytest.fixture
def test_table(memory_conn: duckdb.DuckDBPyConnection) -> str:
    """Create a test table with text columns for FTS testing."""
    memory_conn.execute("""
        CREATE TABLE test_models (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            description VARCHAR,
            notes VARCHAR
        )
    """)
    memory_conn.execute("""
        INSERT INTO test_models VALUES
        ('id1', 'First Model', 'A description of the first model', 'Some notes'),
        ('id2', 'Second Model', 'Description for second', 'More notes'),
        ('id3', 'Third Model', 'Final description', 'Last notes')
    """)
    return "test_models"


@pytest.fixture
def vector_table(memory_conn: duckdb.DuckDBPyConnection) -> str:
    """Create a test table with vector column for HNSW testing."""
    memory_conn.execute("""
        CREATE TABLE vector_models (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            embedding FLOAT[512]
        )
    """)
    # Insert some test vectors
    test_vector = "[" + ",".join(["0.1"] * 512) + "]"
    memory_conn.execute(f"""
        INSERT INTO vector_models VALUES
        ('v1', 'Vector Model 1', {test_vector}::FLOAT[512]),
        ('v2', 'Vector Model 2', {test_vector}::FLOAT[512]),
        ('v3', 'Vector Model 3', {test_vector}::FLOAT[512])
    """)
    return "vector_models"


# ============================================================================
# Tests for create_fts_index()
# ============================================================================


def test_create_fts_index_basic(memory_conn: duckdb.DuckDBPyConnection, test_table: str) -> None:
    """Test creating FTS index with default parameters."""
    # Create index with default parameters
    create_fts_index(memory_conn, test_table, "id", "name", "description")

    # Verify index exists and works
    assert _fts_index_exists(memory_conn, test_table)

    # Test search works - search for "model" which appears in the data
    results = memory_conn.execute(
        f"SELECT id, fts_main_{test_table}.match_bm25(id, 'model') AS score "
        f"FROM {test_table} WHERE score IS NOT NULL ORDER BY score DESC"
    ).fetchall()
    assert len(results) >= 1
    assert results[0][1] > 0  # Score should be positive


def test_create_fts_index_custom_parameters(memory_conn: duckdb.DuckDBPyConnection, test_table: str) -> None:
    """Test creating FTS index with custom stemmer, stopwords, and lower settings."""
    # Create index with custom parameters
    create_fts_index(
        memory_conn,
        test_table,
        "id",
        "name",
        "description",
        stemmer="english",
        stopwords="none",
        lower=1,
    )

    # Verify index exists
    assert _fts_index_exists(memory_conn, test_table)

    # Test case-insensitive search (lower=1) - search for "MODEL" in uppercase
    results = memory_conn.execute(
        f"SELECT id, fts_main_{test_table}.match_bm25(id, 'MODEL') AS score "
        f"FROM {test_table} WHERE score IS NOT NULL ORDER BY score DESC"
    ).fetchall()
    assert len(results) >= 1
    assert results[0][1] > 0  # Score should be positive


def test_create_fts_index_multiple_columns(memory_conn: duckdb.DuckDBPyConnection, test_table: str) -> None:
    """Test creating FTS index on multiple text columns."""
    # Create index on three text columns
    create_fts_index(memory_conn, test_table, "id", "name", "description", "notes")

    # Verify index exists
    assert _fts_index_exists(memory_conn, test_table)

    # Test search across all columns - search for "notes" which appears in the notes column
    results = memory_conn.execute(
        f"SELECT id, fts_main_{test_table}.match_bm25(id, 'notes') AS score "
        f"FROM {test_table} WHERE score IS NOT NULL ORDER BY score DESC"
    ).fetchall()
    assert len(results) >= 1
    assert results[0][1] > 0  # Score should be positive


def test_create_fts_index_no_text_columns(memory_conn: duckdb.DuckDBPyConnection, test_table: str) -> None:
    """Test that creating FTS index without text columns raises ValueError."""
    with pytest.raises(ValueError, match="At least one text column must be specified"):
        create_fts_index(memory_conn, test_table, "id")


# ============================================================================
# Tests for create_hnsw_index()
# ============================================================================


def test_create_hnsw_index_basic(memory_conn: duckdb.DuckDBPyConnection, vector_table: str) -> None:
    """Test creating HNSW index with default parameters and auto-generated name."""
    # Create index without specifying name (should auto-generate)
    create_hnsw_index(memory_conn, vector_table, "embedding")

    # Verify index exists with default name pattern
    default_name = f"idx_{vector_table}_embedding_hnsw"
    assert _hnsw_index_exists(memory_conn, default_name)


def test_create_hnsw_index_custom_name(memory_conn: duckdb.DuckDBPyConnection, vector_table: str) -> None:
    """Test creating HNSW index with explicit custom name."""
    custom_name = "my_custom_hnsw_index"
    create_hnsw_index(memory_conn, vector_table, "embedding", index_name=custom_name)

    # Verify index exists with custom name
    assert _hnsw_index_exists(memory_conn, custom_name)


def test_create_hnsw_index_custom_metric(memory_conn: duckdb.DuckDBPyConnection, vector_table: str) -> None:
    """Test creating HNSW index with L2 squared metric instead of cosine."""
    create_hnsw_index(memory_conn, vector_table, "embedding", metric="l2sq")

    # Verify index exists
    default_name = f"idx_{vector_table}_embedding_hnsw"
    assert _hnsw_index_exists(memory_conn, default_name)

    # Index should work with L2 squared metric for vector search
    test_vector = "[" + ",".join(["0.1"] * 512) + "]"
    results = memory_conn.execute(
        f"SELECT id FROM {vector_table} ORDER BY array_distance(embedding, {test_vector}::FLOAT[512]) LIMIT 3"
    ).fetchall()
    assert len(results) == 3


def test_create_hnsw_index_custom_parameters(memory_conn: duckdb.DuckDBPyConnection, vector_table: str) -> None:
    """Test creating HNSW index with custom ef_construction, ef_search, and m parameters."""
    # Create index with custom HNSW parameters
    create_hnsw_index(
        memory_conn,
        vector_table,
        "embedding",
        ef_construction=256,
        ef_search=128,
        m=32,
    )

    # Verify index exists
    default_name = f"idx_{vector_table}_embedding_hnsw"
    assert _hnsw_index_exists(memory_conn, default_name)


# ============================================================================
# Tests for drop_search_indexes()
# ============================================================================


def test_drop_search_indexes_both(memory_conn: duckdb.DuckDBPyConnection, test_table: str, vector_table: str) -> None:
    """Test dropping both FTS and HNSW indexes."""
    # Setup: Create both a test table with text and vector columns
    memory_conn.execute(f"""
        CREATE TABLE combined_table AS
        SELECT id, name, description,
               [{",".join(["0.1"] * 512)}]::FLOAT[512] as embedding
        FROM {test_table}
    """)

    # Create both indexes
    create_fts_index(memory_conn, "combined_table", "id", "name", "description")
    create_hnsw_index(memory_conn, "combined_table", "embedding")

    hnsw_index_name = "idx_combined_table_embedding_hnsw"

    # Verify both exist
    assert _fts_index_exists(memory_conn, "combined_table")
    assert _hnsw_index_exists(memory_conn, hnsw_index_name)

    # Drop both indexes
    drop_search_indexes(memory_conn, "combined_table", hnsw_index_name=hnsw_index_name)

    # Verify both are gone
    assert not _fts_index_exists(memory_conn, "combined_table")
    assert not _hnsw_index_exists(memory_conn, hnsw_index_name)


def test_drop_search_indexes_fts_only(memory_conn: duckdb.DuckDBPyConnection, test_table: str) -> None:
    """Test dropping FTS index when no HNSW index exists."""
    # Create only FTS index
    create_fts_index(memory_conn, test_table, "id", "name", "description")

    # Verify FTS exists
    assert _fts_index_exists(memory_conn, test_table)

    # Drop indexes (no HNSW index name provided)
    drop_search_indexes(memory_conn, test_table, hnsw_index_name=None)

    # Verify FTS is gone
    assert not _fts_index_exists(memory_conn, test_table)


def test_drop_search_indexes_missing_gracefully(memory_conn: duckdb.DuckDBPyConnection, test_table: str) -> None:
    """Test that dropping non-existent indexes doesn't raise errors."""
    # Don't create any indexes

    # Verify no indexes exist
    assert not _fts_index_exists(memory_conn, test_table)

    # Drop indexes - should not raise error even though they don't exist
    drop_search_indexes(memory_conn, test_table, hnsw_index_name="nonexistent_index")

    # Should complete successfully without error
    # (The suppress(duckdb.Error) pattern in drop_search_indexes handles this)
