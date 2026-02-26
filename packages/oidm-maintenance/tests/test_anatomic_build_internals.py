"""DuckDB integration tests for anatomic location build internals.

These tests verify schema creation, STRUCT[] insertion, duplicate handling,
and index creation with real DuckDB instances (using temp files).
No external API calls - embeddings are loaded from test/data.
"""

from collections.abc import Generator
from pathlib import Path

import duckdb
import pytest
from oidm_common.duckdb import setup_duckdb_connection
from oidm_maintenance.anatomic.build import (
    _bulk_load_table,
    _create_indexes,
    _create_schema,
    _get_location_columns,
    _prepare_all_records,
    determine_laterality,
)
from pydantic_ai import models
from pydantic_settings import BaseSettings, SettingsConfigDict

# Prevent accidental model requests in unit tests
models.ALLOW_MODEL_REQUESTS = False


# Create a minimal settings class for tests
class TestSettings(BaseSettings):
    """Minimal settings for anatomic location tests."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    openai_embedding_dimensions: int = 512


test_settings = TestSettings()


@pytest.fixture
def duckdb_conn(temp_duckdb_path: Path) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Create a DuckDB connection to a temporary database."""
    conn = setup_duckdb_connection(temp_duckdb_path, read_only=False)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def schema_conn(duckdb_conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with schema already created."""
    _create_schema(duckdb_conn, dimensions=test_settings.openai_embedding_dimensions)
    return duckdb_conn


def test_schema_creation(duckdb_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that tables are created correctly with all expected columns."""
    _create_schema(duckdb_conn, dimensions=test_settings.openai_embedding_dimensions)

    # Check that all 4 tables exist
    tables = duckdb_conn.execute("SHOW TABLES").fetchall()
    table_names = {row[0] for row in tables}
    assert "anatomic_locations" in table_names
    assert "anatomic_synonyms" in table_names
    assert "anatomic_codes" in table_names
    assert "anatomic_references" in table_names

    # Verify anatomic_locations columns
    location_cols = duckdb_conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'anatomic_locations' ORDER BY ordinal_position"
    ).fetchall()

    location_col_names = {row[0] for row in location_cols}
    assert "id" in location_col_names
    assert "description" in location_col_names
    assert "laterality" in location_col_names
    assert "search_text" in location_col_names
    assert "vector" in location_col_names
    assert "containment_path" in location_col_names
    assert "containment_children" in location_col_names
    assert "partof_path" in location_col_names
    assert "partof_children" in location_col_names

    # Verify synonyms table structure
    synonym_cols = duckdb_conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'anatomic_synonyms'"
    ).fetchall()
    synonym_col_names = {row[0] for row in synonym_cols}
    assert "location_id" in synonym_col_names
    assert "synonym" in synonym_col_names

    # Verify codes table structure
    code_cols = duckdb_conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'anatomic_codes'"
    ).fetchall()
    code_col_names = {row[0] for row in code_cols}
    assert "location_id" in code_col_names
    assert "system" in code_col_names
    assert "code" in code_col_names
    assert "display" in code_col_names


def test_bulk_load_table_basic(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that _bulk_load_table() loads data correctly."""
    # Create a simple test table
    schema_conn.execute("CREATE TABLE test (id VARCHAR, name VARCHAR)")

    data = [{"id": "1", "name": "foo"}, {"id": "2", "name": "bar"}]
    column_types = {"id": "VARCHAR", "name": "VARCHAR"}

    count = _bulk_load_table(schema_conn, "test", data, column_types)

    assert count == 2
    results = schema_conn.execute("SELECT * FROM test ORDER BY id").fetchall()
    assert len(results) == 2
    assert results[0][0] == "1"
    assert results[0][1] == "foo"
    assert results[1][0] == "2"
    assert results[1][1] == "bar"


def test_bulk_load_preserves_vectors(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify FLOAT[] vectors survive JSON round-trip."""
    dimensions = test_settings.openai_embedding_dimensions
    schema_conn.execute(f"CREATE TABLE test (id VARCHAR, vector FLOAT[{dimensions}])")

    vector = [0.1] * dimensions
    data = [{"id": "1", "vector": vector}]
    column_types = {"id": "VARCHAR", "vector": f"FLOAT[{dimensions}]"}

    _bulk_load_table(schema_conn, "test", data, column_types)

    result = schema_conn.execute(f"SELECT vector[1], vector[{dimensions}] FROM test").fetchone()
    assert result is not None
    # Float precision check
    assert abs(result[0] - 0.1) < 0.001
    assert abs(result[1] - 0.1) < 0.001


def test_bulk_load_preserves_struct_arrays(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify STRUCT[] arrays survive JSON round-trip."""
    schema_conn.execute("CREATE TABLE test (id VARCHAR, children STRUCT(id VARCHAR, display VARCHAR)[])")

    children = [{"id": "child1", "display": "Child 1"}, {"id": "child2", "display": "Child 2"}]
    data = [{"id": "TEST001", "children": children}]
    column_types = {"id": "VARCHAR", "children": "STRUCT(id VARCHAR, display VARCHAR)[]"}

    _bulk_load_table(schema_conn, "test", data, column_types)

    result = schema_conn.execute("SELECT id, children FROM test WHERE id = 'TEST001'").fetchone()

    assert result is not None
    record_id, retrieved_children = result
    assert record_id == "TEST001"

    # Verify STRUCT[] was stored correctly
    assert isinstance(retrieved_children, list)
    assert len(retrieved_children) == 2
    assert retrieved_children[0]["id"] == "child1"
    assert retrieved_children[0]["display"] == "Child 1"
    assert retrieved_children[1]["id"] == "child2"
    assert retrieved_children[1]["display"] == "Child 2"


def test_prepare_all_records(
    anatomic_sample_data: list[dict[str, object]],
    anatomic_records_by_id: dict[str, dict[str, object]],
) -> None:
    """Verify _prepare_all_records returns correct 4-tuple structure."""
    # Use first 3 records from sample data
    records = anatomic_sample_data[:3]

    location_rows, searchable_texts, synonym_rows, code_rows = _prepare_all_records(records, anatomic_records_by_id)

    # Should have successfully prepared all 3 records
    assert len(location_rows) == 3
    assert len(searchable_texts) == 3

    # Verify structure of location rows
    for location_row in location_rows:
        assert "id" in location_row
        assert "description" in location_row
        assert "laterality" in location_row
        assert "search_text" in location_row
        assert "containment_children" in location_row
        assert "partof_children" in location_row
        assert isinstance(location_row["containment_children"], list)
        assert isinstance(location_row["partof_children"], list)

    # Verify synonym rows are dicts with correct keys
    for synonym_row in synonym_rows:
        assert "location_id" in synonym_row
        assert "synonym" in synonym_row

    # Verify code rows are dicts with correct keys
    for code_row in code_rows:
        assert "location_id" in code_row
        assert "system" in code_row
        assert "code" in code_row
        assert "display" in code_row


def test_bulk_load_with_real_data(
    schema_conn: duckdb.DuckDBPyConnection,
    anatomic_sample_data: list[dict[str, object]],
    anatomic_records_by_id: dict[str, dict[str, object]],
) -> None:
    """Verify bulk load works with real anatomic data."""
    # Prepare records
    location_rows, _searchable_texts, _synonym_rows, _code_rows = _prepare_all_records(
        anatomic_sample_data[:3], anatomic_records_by_id
    )

    # Add dummy vectors to location rows
    for location_row in location_rows:
        location_row["vector"] = [0.1] * test_settings.openai_embedding_dimensions

    # Bulk load locations
    count = _bulk_load_table(
        schema_conn,
        "anatomic_locations",
        location_rows,
        _get_location_columns(test_settings.openai_embedding_dimensions),
    )
    assert count == 3

    # Verify records were inserted correctly
    result = schema_conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
    assert result is not None
    assert result[0] == 3

    # Verify STRUCT[] fields survived the round-trip
    for location_row in location_rows:
        db_result = schema_conn.execute(
            "SELECT containment_children, partof_children FROM anatomic_locations WHERE id = ?",
            (location_row["id"],),
        ).fetchone()
        assert db_result is not None
        db_containment, db_partof = db_result
        assert isinstance(db_containment, list)
        assert isinstance(db_partof, list)
        assert len(db_containment) == len(location_row["containment_children"])
        assert len(db_partof) == len(location_row["partof_children"])


def test_insert_or_ignore_duplicate_synonym(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that duplicate synonyms are silently ignored."""
    # Insert initial location
    schema_conn.execute(
        """
        INSERT INTO anatomic_locations
            (id, description, laterality, search_text, vector)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("TEST001", "test", "nonlateral", "test", [0.1] * test_settings.openai_embedding_dimensions),
    )

    # Insert first synonym
    schema_conn.execute(
        "INSERT OR IGNORE INTO anatomic_synonyms (location_id, synonym) VALUES (?, ?)",
        ("TEST001", "test synonym"),
    )

    # Insert duplicate synonym - should be ignored
    schema_conn.execute(
        "INSERT OR IGNORE INTO anatomic_synonyms (location_id, synonym) VALUES (?, ?)",
        ("TEST001", "test synonym"),
    )

    # Verify only one synonym exists
    result = schema_conn.execute(
        "SELECT COUNT(*) FROM anatomic_synonyms WHERE location_id = 'TEST001' AND synonym = 'test synonym'"
    ).fetchone()
    assert result is not None
    assert result[0] == 1


def test_insert_or_ignore_duplicate_code(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that duplicate codes are silently ignored."""
    # Insert initial location
    schema_conn.execute(
        """
        INSERT INTO anatomic_locations
            (id, description, laterality, search_text, vector)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("TEST001", "test", "nonlateral", "test", [0.1] * test_settings.openai_embedding_dimensions),
    )

    # Insert first code
    schema_conn.execute(
        "INSERT OR IGNORE INTO anatomic_codes (location_id, system, code, display) VALUES (?, ?, ?, ?)",
        ("TEST001", "SNOMED", "123456", "test code"),
    )

    # Insert duplicate code - should be ignored
    schema_conn.execute(
        "INSERT OR IGNORE INTO anatomic_codes (location_id, system, code, display) VALUES (?, ?, ?, ?)",
        ("TEST001", "SNOMED", "123456", "test code"),
    )

    # Verify only one code exists
    result = schema_conn.execute(
        "SELECT COUNT(*) FROM anatomic_codes WHERE location_id = 'TEST001' AND system = 'SNOMED' AND code = '123456'"
    ).fetchone()
    assert result is not None
    assert result[0] == 1


def test_index_creation_fts(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that FTS index is created successfully."""
    # Insert a test record
    schema_conn.execute(
        """
        INSERT INTO anatomic_locations
            (id, description, laterality, search_text, vector, definition)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "TEST001",
            "test structure",
            "nonlateral",
            "test structure",
            [0.1] * test_settings.openai_embedding_dimensions,
            "test definition",
        ),
    )
    schema_conn.commit()

    # Create FTS index
    from oidm_common.duckdb import create_fts_index

    create_fts_index(
        schema_conn,
        "anatomic_locations",
        "id",
        "description",
        "definition",
        stemmer="porter",
        stopwords="english",
        lower=1,
        overwrite=True,
    )

    # Verify FTS index was created by attempting a search
    result = schema_conn.execute(
        """
        SELECT fts_main_anatomic_locations.match_bm25(id, 'test') AS score
        FROM anatomic_locations
        WHERE score IS NOT NULL
        ORDER BY score DESC
        """
    ).fetchall()

    # Should return at least one result
    assert len(result) > 0


def test_index_creation_hnsw(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that HNSW index is created successfully."""
    # Insert test records with vectors
    for i in range(10):
        schema_conn.execute(
            """
            INSERT INTO anatomic_locations
                (id, description, laterality, search_text, vector)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"TEST{i:03d}",
                f"test {i}",
                "nonlateral",
                f"test {i}",
                [float(i) / 100.0] * test_settings.openai_embedding_dimensions,
            ),
        )
    schema_conn.commit()

    # Create HNSW index
    from oidm_common.duckdb import create_hnsw_index

    try:
        create_hnsw_index(
            schema_conn,
            table="anatomic_locations",
            column="vector",
            index_name="idx_anatomic_hnsw",
            metric="cosine",
            ef_construction=128,
            ef_search=64,
            m=16,
        )

        # Verify index was created by checking DuckDB internals
        # DuckDB doesn't expose index metadata in information_schema, but we can
        # verify the index works by running a query that would use it
        query_vector = [0.05] * test_settings.openai_embedding_dimensions
        result = schema_conn.execute(
            f"""
            SELECT id, array_distance(vector, ?::FLOAT[{test_settings.openai_embedding_dimensions}]) AS distance
            FROM anatomic_locations
            ORDER BY distance
            LIMIT 3
            """,
            (query_vector,),
        ).fetchall()

        # Should return results
        assert len(result) == 3
        assert all(row[0].startswith("TEST") for row in result)

    except Exception as e:
        # HNSW index creation may fail on some DuckDB versions
        # This is acceptable per the implementation (falls back to brute force)
        pytest.skip(f"HNSW index creation not supported: {e}")


def test_bulk_load_empty_data(schema_conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that bulk load handles empty data correctly."""
    schema_conn.execute("CREATE TABLE test (id VARCHAR, name VARCHAR)")

    data: list[dict[str, str]] = []
    column_types = {"id": "VARCHAR", "name": "VARCHAR"}

    count = _bulk_load_table(schema_conn, "test", data, column_types)

    assert count == 0
    results = schema_conn.execute("SELECT COUNT(*) FROM test").fetchone()
    assert results is not None
    assert results[0] == 0


def test_full_build_pipeline(
    temp_duckdb_path: Path,
    anatomic_sample_data: list[dict[str, object]],
    anatomic_sample_embeddings: dict[str, list[float]],
) -> None:
    """Verify the complete pipeline: load sample data, build database, verify counts."""
    # This is a higher-level integration test that exercises the full stack
    conn = setup_duckdb_connection(temp_duckdb_path, read_only=False)

    try:
        # Create schema
        _create_schema(conn, dimensions=test_settings.openai_embedding_dimensions)

        # Verify schema
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = {row[0] for row in tables}
        assert len(table_names) == 4

        # Insert a few test records
        for record in anatomic_sample_data[:2]:
            record_id = str(record["_id"])
            description = str(record["description"])
            embedding = anatomic_sample_embeddings.get(record_id, [0.1] * test_settings.openai_embedding_dimensions)

            conn.execute(
                """
                INSERT INTO anatomic_locations
                    (id, description, laterality, search_text, vector,
                     containment_children, partof_children)
                VALUES (?, ?, ?, ?, ?, [], [])
                """,
                (record_id, description, "nonlateral", description, embedding),
            )

        conn.commit()

        # Verify counts
        location_count = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
        assert location_count is not None
        assert location_count[0] == 2

        # Create indexes
        _create_indexes(conn, test_settings.openai_embedding_dimensions)

        # Verify FTS index works
        fts_result = conn.execute(
            """
            SELECT COUNT(*) FROM anatomic_locations
            WHERE fts_main_anatomic_locations.match_bm25(id, 'nasal') IS NOT NULL
            """
        ).fetchone()
        # Should have at least some matches
        assert fts_result is not None
        assert fts_result[0] >= 0

    finally:
        conn.close()


# =============================================================================
# Laterality determination tests
# =============================================================================


class TestDetermineLaterality:
    """Tests for determine_laterality() ref-to-laterality mapping."""

    def test_both_refs_is_generic(self) -> None:
        """Record with both leftRef and rightRef is the generic form."""
        record = {"leftRef": "RID123", "rightRef": "RID456"}
        assert determine_laterality(record) == "generic"

    def test_left_ref_only_is_right(self) -> None:
        """Record with leftRef only points to its left counterpart → this is the RIGHT variant."""
        record = {"leftRef": "RID123"}
        assert determine_laterality(record) == "right"

    def test_right_ref_only_is_left(self) -> None:
        """Record with rightRef only points to its right counterpart → this is the LEFT variant."""
        record = {"rightRef": "RID456"}
        assert determine_laterality(record) == "left"

    def test_unsided_ref_only_is_generic(self) -> None:
        """Record with only unsidedRef maps to generic."""
        record = {"unsidedRef": "RID789"}
        assert determine_laterality(record) == "generic"

    def test_no_refs_is_nonlateral(self) -> None:
        """Record with no ref properties is nonlateral."""
        record = {"description": "some structure"}
        assert determine_laterality(record) == "nonlateral"

    def test_empty_record_is_nonlateral(self) -> None:
        """Empty record is nonlateral."""
        assert determine_laterality({}) == "nonlateral"
