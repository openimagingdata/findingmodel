"""Test asyncify usage with DuckDB for concurrent sync operations.

This demonstrates using asyncer.asyncify to wrap synchronous DuckDB operations
for concurrent execution in async contexts. Type hints are preserved through asyncify.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb
import pytest
from asyncer import asyncify


# Define sync helpers with full type hints - types are preserved through asyncify
def query_all_rows(conn: duckdb.DuckDBPyConnection) -> list[tuple[int, str]]:
    """Sync helper with typed return - asyncify preserves this."""
    return conn.execute("SELECT * FROM test").fetchall()


def query_count(conn: duckdb.DuckDBPyConnection) -> int:
    """Sync helper returning count."""
    result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
    return result[0] if result else 0


def query_sum(conn: duckdb.DuckDBPyConnection, column: str) -> int:
    """Sync helper with parameter and typed return."""
    result = conn.execute(f"SELECT SUM({column}) FROM test").fetchone()
    return result[0] if result and result[0] is not None else 0


def execute_insert(conn: duckdb.DuckDBPyConnection, id_val: int, name: str) -> None:
    """Sync helper for mutation operations."""
    conn.execute("INSERT INTO test VALUES (?, ?)", [id_val, name])


@pytest.mark.asyncio
async def test_asyncify_typed_sync_function() -> None:
    """Test asyncify with typed sync function - types preserved."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test VALUES (1, 'alice'), (2, 'bob')")

    # Wrap typed sync helper - return type flows through
    rows = await asyncify(query_all_rows)(conn)

    assert len(rows) == 2
    assert rows[0] == (1, "alice")
    assert rows[1] == (2, "bob")
    conn.close()


@pytest.mark.asyncio
async def test_asyncify_concurrent_typed_calls() -> None:
    """Multiple typed async calls can run concurrently with separate connections.

    Note: DuckDB connections are not thread-safe, so each concurrent operation
    needs its own connection when running truly concurrent operations.
    """
    # Create a shared in-memory database
    conn1 = duckdb.connect(":memory:")
    conn1.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn1.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b'), (3, 'c')")

    # For truly concurrent operations, create separate connections
    # (though with :memory: they won't share data - this demonstrates the pattern)
    conn2 = duckdb.connect(":memory:")
    conn2.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn2.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b'), (3, 'c')")

    conn3 = duckdb.connect(":memory:")
    conn3.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn3.execute("INSERT INTO test VALUES (1, 'a'), (2, 'b'), (3, 'c')")

    # Each call uses a different connection - can run concurrently
    result1, result2, result3 = await asyncio.gather(
        asyncify(query_count)(conn1),
        asyncify(query_count)(conn2),
        asyncify(query_count)(conn3),
    )

    assert result1 == 3
    assert result2 == 3
    assert result3 == 3
    conn1.close()
    conn2.close()
    conn3.close()


@pytest.mark.asyncio
async def test_asyncify_with_parameters() -> None:
    """Test asyncify preserves parameters and types."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test VALUES (1, 'alice'), (2, 'bob'), (3, 'charlie')")

    # Call with parameter
    total = await asyncify(query_sum)(conn, "id")

    assert total == 6
    conn.close()


@pytest.mark.asyncio
async def test_asyncify_mixed_operations() -> None:
    """Test asyncify with both queries and mutations."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")

    # Insert initial data asynchronously
    await asyncio.gather(
        asyncify(execute_insert)(conn, 1, "alice"),
        asyncify(execute_insert)(conn, 2, "bob"),
        asyncify(execute_insert)(conn, 3, "charlie"),
    )

    # Query the count
    count = await asyncify(query_count)(conn)
    assert count == 3

    # Query all rows
    rows = await asyncify(query_all_rows)(conn)
    assert len(rows) == 3

    conn.close()


@pytest.mark.asyncio
async def test_asyncify_error_handling() -> None:
    """Test that errors in sync functions propagate correctly."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")

    # This should raise an error - invalid SQL
    with pytest.raises(duckdb.CatalogException):
        await asyncify(lambda c: c.execute("SELECT * FROM nonexistent_table").fetchall())(conn)

    conn.close()


@pytest.mark.asyncio
async def test_asyncify_sequential_reads() -> None:
    """Test sequential async reads using asyncify.

    DuckDB connections are not thread-safe, so we run operations sequentially
    but still benefit from async/await patterns for code organization.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test VALUES (1, 'alice'), (2, 'bob'), (3, 'charlie'), (4, 'dave')")

    # Sequential async operations using the same connection
    count_result = await asyncify(query_count)(conn)
    sum_result = await asyncify(query_sum)(conn, "id")
    all_rows = await asyncify(query_all_rows)(conn)

    assert count_result == 4
    assert sum_result == 10
    assert len(all_rows) == 4

    conn.close()


@pytest.mark.asyncio
async def test_asyncify_concurrent_with_file_db(tmp_path: Path) -> None:
    """Test concurrent reads from file-based database with separate connections.

    File-based DuckDB databases support multiple read-only connections,
    enabling true concurrent reads.
    """
    db_path = tmp_path / "test.duckdb"

    # Create and populate the database
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test VALUES (1, 'alice'), (2, 'bob'), (3, 'charlie'), (4, 'dave')")
    conn.close()

    # Open multiple read-only connections for concurrent reads
    conn1 = duckdb.connect(str(db_path), read_only=True)
    conn2 = duckdb.connect(str(db_path), read_only=True)
    conn3 = duckdb.connect(str(db_path), read_only=True)

    # Multiple concurrent read operations with separate connections
    count_result, sum_result, all_rows = await asyncio.gather(
        asyncify(query_count)(conn1),
        asyncify(query_sum)(conn2, "id"),
        asyncify(query_all_rows)(conn3),
    )

    assert count_result == 4
    assert sum_result == 10
    assert len(all_rows) == 4

    conn1.close()
    conn2.close()
    conn3.close()
