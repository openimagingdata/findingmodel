"""Tests for ReadOnlyDuckDBIndex base class."""

from __future__ import annotations

from pathlib import Path

import pytest
from oidm_common.duckdb import ReadOnlyDuckDBIndex, setup_duckdb_connection


@pytest.fixture
def seeded_db(tmp_duckdb_path: Path) -> Path:
    """Create a DuckDB file with a small table for lifecycle tests."""
    conn = setup_duckdb_connection(tmp_duckdb_path, read_only=False)
    conn.execute("CREATE TABLE items (id VARCHAR PRIMARY KEY, name VARCHAR)")
    conn.execute("INSERT INTO items VALUES ('a', 'alpha'), ('b', 'beta')")
    conn.close()
    return tmp_duckdb_path


class TestReadOnlyDuckDBIndexInit:
    """Tests for __init__ resolution logic."""

    def test_explicit_path(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(seeded_db)
        assert idx.db_path == seeded_db
        assert idx.conn is None

    def test_ensure_db_callable(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(ensure_db=lambda: seeded_db)
        assert idx.db_path == seeded_db

    def test_no_path_no_ensure_db_raises(self) -> None:
        with pytest.raises(ValueError, match="Either db_path or ensure_db"):
            ReadOnlyDuckDBIndex()

    def test_path_takes_precedence_over_ensure_db(self, seeded_db: Path, tmp_path: Path) -> None:
        other = tmp_path / "other.duckdb"
        idx = ReadOnlyDuckDBIndex(seeded_db, ensure_db=lambda: other)
        assert idx.db_path == seeded_db

    def test_string_path_converted(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(str(seeded_db))
        assert idx.db_path == seeded_db


class TestReadOnlyDuckDBIndexLifecycle:
    """Tests for open/close/auto-open."""

    def test_open_returns_self(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(seeded_db)
        result = idx.open()
        assert result is idx
        assert idx.conn is not None
        idx.close()

    def test_close_sets_conn_none(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(seeded_db)
        idx.open()
        idx.close()
        assert idx.conn is None

    def test_close_when_not_open_is_noop(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(seeded_db)
        idx.close()  # should not raise
        assert idx.conn is None

    def test_ensure_connection_auto_opens(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(seeded_db)
        conn = idx._ensure_connection()
        assert conn is not None
        row = conn.execute("SELECT count(*) FROM items").fetchone()
        assert row is not None and row[0] == 2
        idx.close()

    def test_ensure_connection_reuses_existing(self, seeded_db: Path) -> None:
        idx = ReadOnlyDuckDBIndex(seeded_db)
        conn1 = idx._ensure_connection()
        conn2 = idx._ensure_connection()
        assert conn1 is conn2
        idx.close()


class TestReadOnlyDuckDBIndexContextManagers:
    """Tests for sync and async context managers."""

    def test_sync_context_manager(self, seeded_db: Path) -> None:
        with ReadOnlyDuckDBIndex(seeded_db) as idx:
            assert idx.conn is not None
            row = idx.conn.execute("SELECT name FROM items WHERE id = 'a'").fetchone()
            assert row is not None and row[0] == "alpha"
        assert idx.conn is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self, seeded_db: Path) -> None:
        async with ReadOnlyDuckDBIndex(seeded_db) as idx:
            assert idx.conn is not None
            row = idx.conn.execute("SELECT name FROM items WHERE id = 'b'").fetchone()
            assert row is not None and row[0] == "beta"
        assert idx.conn is None


class TestReadOnlyDuckDBIndexHelpers:
    """Tests for _execute_one and _execute_all named-dict helpers."""

    @pytest.fixture
    def two_col_db(self, tmp_path: Path) -> Path:
        """In-memory-style DB with a two-column table for helper tests."""
        db = tmp_path / "helper_test.duckdb"
        conn = setup_duckdb_connection(db, read_only=False)
        conn.execute("CREATE TABLE things (id INTEGER, label VARCHAR)")
        conn.execute("INSERT INTO things VALUES (1, 'alpha'), (2, 'beta'), (3, 'gamma')")
        conn.close()
        return db

    def test_execute_one_returns_dict(self, two_col_db: Path) -> None:
        """_execute_one maps column names to dict keys."""
        idx = ReadOnlyDuckDBIndex(two_col_db)
        conn = idx._ensure_connection()
        row = idx._execute_one(conn, "SELECT id, label FROM things WHERE id = ?", [1])
        assert row is not None
        assert row["id"] == 1
        assert row["label"] == "alpha"
        idx.close()

    def test_execute_one_returns_none_when_not_found(self, two_col_db: Path) -> None:
        """_execute_one returns None when the query produces no rows."""
        idx = ReadOnlyDuckDBIndex(two_col_db)
        conn = idx._ensure_connection()
        row = idx._execute_one(conn, "SELECT id, label FROM things WHERE id = ?", [999])
        assert row is None
        idx.close()

    def test_execute_all_returns_list_of_dicts(self, two_col_db: Path) -> None:
        """_execute_all returns a list of named dicts, one per row."""
        idx = ReadOnlyDuckDBIndex(two_col_db)
        conn = idx._ensure_connection()
        rows = idx._execute_all(conn, "SELECT id, label FROM things ORDER BY id")
        assert len(rows) == 3
        assert rows[0]["id"] == 1
        assert rows[0]["label"] == "alpha"
        assert rows[2]["id"] == 3
        assert rows[2]["label"] == "gamma"
        idx.close()

    def test_execute_all_returns_empty_list(self, two_col_db: Path) -> None:
        """_execute_all returns an empty list when no rows match."""
        idx = ReadOnlyDuckDBIndex(two_col_db)
        conn = idx._ensure_connection()
        rows = idx._execute_all(conn, "SELECT id, label FROM things WHERE id > ?", [100])
        assert rows == []
        idx.close()


class TestReadOnlyDuckDBIndexSubclass:
    """Test that subclasses work correctly."""

    def test_subclass_with_ensure_db(self, seeded_db: Path) -> None:
        class MyIndex(ReadOnlyDuckDBIndex):
            def __init__(self) -> None:
                super().__init__(ensure_db=lambda: seeded_db)

        with MyIndex() as idx:
            assert idx.db_path == seeded_db
            row = idx.conn.execute("SELECT count(*) FROM items").fetchone()  # type: ignore[union-attr]
            assert row is not None and row[0] == 2

    def test_subclass_auto_open_on_query(self, seeded_db: Path) -> None:
        class MyIndex(ReadOnlyDuckDBIndex):
            def query_count(self) -> int:
                conn = self._ensure_connection()
                row = conn.execute("SELECT count(*) FROM items").fetchone()
                return int(row[0]) if row else 0

        idx = MyIndex(seeded_db)
        assert idx.conn is None
        assert idx.query_count() == 2  # auto-opens
        assert idx.conn is not None
        idx.close()
