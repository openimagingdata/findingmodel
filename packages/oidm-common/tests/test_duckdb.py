"""Tests for oidm-common DuckDB utilities."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from oidm_common.duckdb import (
    ScoreTuple,
    create_fts_index,
    create_hnsw_index,
    drop_search_indexes,
    l2_to_cosine_similarity,
    normalize_scores,
    rrf_fusion,
    setup_duckdb_connection,
    weighted_fusion,
)


class TestSetupDuckDBConnection:
    """Tests for setup_duckdb_connection function."""

    def test_setup_duckdb_connection_default_extensions(self, tmp_duckdb_path: Path) -> None:
        """Test creating connection with default extensions (fts, vss)."""
        conn = setup_duckdb_connection(tmp_duckdb_path, read_only=False)

        # Verify connection is active
        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)

        # Verify extensions are loaded by trying to use them
        # FTS extension check
        conn.execute("""
            CREATE TABLE test_fts (id VARCHAR, text VARCHAR)
        """)
        conn.execute("PRAGMA create_fts_index('test_fts', 'id', 'text')")

        # VSS extension check - create a table with vector column
        conn.execute("""
            CREATE TABLE test_vss (id VARCHAR, vec FLOAT[3])
        """)

        conn.close()

    def test_setup_duckdb_connection_read_only(self, tmp_duckdb_path: Path) -> None:
        """Test creating read-only connection."""
        # First create a database file with some data
        write_conn = setup_duckdb_connection(tmp_duckdb_path, read_only=False)
        write_conn.execute("CREATE TABLE test (id INTEGER)")
        write_conn.execute("INSERT INTO test VALUES (1)")
        write_conn.close()

        # Now open in read-only mode
        read_conn = setup_duckdb_connection(tmp_duckdb_path, read_only=True)

        # Read should work
        result = read_conn.execute("SELECT * FROM test").fetchall()
        assert result == [(1,)]

        # Write should fail - DuckDB may raise different exceptions for read-only violations
        with pytest.raises(Exception) as exc_info:
            read_conn.execute("INSERT INTO test VALUES (2)")

        # Verify it's a DuckDB exception related to read-only
        assert isinstance(exc_info.value, (duckdb.IOException, duckdb.CatalogException, duckdb.Error))

        read_conn.close()

    def test_setup_duckdb_connection_custom_extensions(self, tmp_duckdb_path: Path) -> None:
        """Test creating connection with custom extension list."""
        # First create the database
        init_conn = duckdb.connect(str(tmp_duckdb_path))
        init_conn.close()

        # Now open with only FTS extension in read-only mode
        conn = setup_duckdb_connection(tmp_duckdb_path, read_only=True, extensions=["fts"])

        # Can use the connection
        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)

        conn.close()

    def test_setup_duckdb_connection_with_path_string(self, tmp_duckdb_path: Path) -> None:
        """Test that function accepts path as string."""
        conn = setup_duckdb_connection(str(tmp_duckdb_path), read_only=False)

        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)

        conn.close()

    def test_setup_duckdb_connection_enables_hnsw_persistence(self, tmp_duckdb_path: Path) -> None:
        """Test that write connections enable HNSW experimental persistence."""
        conn = setup_duckdb_connection(tmp_duckdb_path, read_only=False)

        # Verify the setting was applied (check boolean value, not string)
        result = conn.execute("SELECT current_setting('hnsw_enable_experimental_persistence')").fetchone()
        assert result is not None
        # The setting should be enabled (can be 'true' or True depending on DuckDB version)
        assert str(result[0]).lower() == "true"

        conn.close()


class TestNormalizeScores:
    """Tests for normalize_scores function."""

    def test_normalize_scores_empty_list(self) -> None:
        """Test normalizing empty score list returns empty list."""
        result = normalize_scores([])

        assert result == []

    def test_normalize_scores_single_value(self) -> None:
        """Test normalizing single score returns 1.0."""
        result = normalize_scores([5.0])

        assert result == [1.0]

    def test_normalize_scores_identical_values(self) -> None:
        """Test normalizing identical scores returns all 1.0."""
        result = normalize_scores([3.0, 3.0, 3.0])

        assert result == [1.0, 1.0, 1.0]

    def test_normalize_scores_range_0_to_1(self) -> None:
        """Test normalizing scores from 0 to 1."""
        result = normalize_scores([0.0, 0.5, 1.0])

        assert result == [0.0, 0.5, 1.0]

    def test_normalize_scores_arbitrary_range(self) -> None:
        """Test normalizing scores from arbitrary range."""
        result = normalize_scores([10.0, 20.0, 30.0])

        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_normalize_scores_negative_values(self) -> None:
        """Test normalizing scores with negative values."""
        result = normalize_scores([-10.0, 0.0, 10.0])

        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_normalize_scores_preserves_order(self) -> None:
        """Test that normalization preserves relative ordering."""
        scores = [5.0, 2.0, 8.0, 1.0]
        result = normalize_scores(scores)

        # Check that relative order is preserved
        assert result[0] > result[1]  # 5 > 2
        assert result[2] > result[0]  # 8 > 5
        assert result[1] > result[3]  # 2 > 1


class TestWeightedFusion:
    """Tests for weighted_fusion function."""

    def test_weighted_fusion_basic(self) -> None:
        """Test basic weighted fusion with default weights."""
        results_a: list[ScoreTuple] = [("item1", 0.5), ("item2", 0.8)]
        results_b: list[ScoreTuple] = [("item2", 0.6), ("item3", 0.9)]

        result = weighted_fusion(results_a, results_b, normalise=False)

        # With default weights (0.3, 0.7):
        # item1: 0.3*0.5 + 0.7*0.0 = 0.15
        # item2: 0.3*0.8 + 0.7*0.6 = 0.24 + 0.42 = 0.66
        # item3: 0.3*0.0 + 0.7*0.9 = 0.63
        result_dict = dict(result)
        assert result_dict["item1"] == pytest.approx(0.15)
        assert result_dict["item2"] == pytest.approx(0.66)
        assert result_dict["item3"] == pytest.approx(0.63)

    def test_weighted_fusion_with_normalization(self) -> None:
        """Test weighted fusion normalizes scores before combining."""
        results_a: list[ScoreTuple] = [("item1", 10.0), ("item2", 20.0)]
        results_b: list[ScoreTuple] = [("item2", 50.0), ("item3", 100.0)]

        result = weighted_fusion(results_a, results_b, normalise=True)

        # After normalization:
        # results_a: item1=0.0, item2=1.0
        # results_b: item2=0.0, item3=1.0
        # Combined (0.3, 0.7):
        # item1: 0.3*0.0 + 0.7*0.0 = 0.0
        # item2: 0.3*1.0 + 0.7*0.0 = 0.3
        # item3: 0.3*0.0 + 0.7*1.0 = 0.7
        result_dict = dict(result)
        assert result_dict["item1"] == pytest.approx(0.0)
        assert result_dict["item2"] == pytest.approx(0.3)
        assert result_dict["item3"] == pytest.approx(0.7)

    def test_weighted_fusion_custom_weights(self) -> None:
        """Test weighted fusion with custom weights."""
        results_a: list[ScoreTuple] = [("item1", 1.0)]
        results_b: list[ScoreTuple] = [("item1", 1.0)]

        result = weighted_fusion(results_a, results_b, weight_a=0.8, weight_b=0.2, normalise=False)

        result_dict = dict(result)
        assert result_dict["item1"] == pytest.approx(1.0)  # 0.8*1.0 + 0.2*1.0

    def test_weighted_fusion_sorted_by_score_descending(self) -> None:
        """Test that results are sorted by score in descending order."""
        results_a: list[ScoreTuple] = [("item1", 0.3), ("item2", 0.5), ("item3", 0.1)]
        results_b: list[ScoreTuple] = [("item1", 0.2), ("item2", 0.4), ("item3", 0.9)]

        result = weighted_fusion(results_a, results_b, normalise=False)

        # Check that scores are in descending order
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_weighted_fusion_empty_results(self) -> None:
        """Test weighted fusion with empty result sets."""
        results_a: list[ScoreTuple] = []
        results_b: list[ScoreTuple] = []

        result = weighted_fusion(results_a, results_b)

        assert result == []


class TestRRFFusion:
    """Tests for rrf_fusion function."""

    def test_rrf_fusion_basic(self) -> None:
        """Test basic RRF fusion with default parameters."""
        results_a: list[ScoreTuple] = [("item1", 10.0), ("item2", 5.0)]
        results_b: list[ScoreTuple] = [("item2", 20.0), ("item3", 15.0)]

        result = rrf_fusion(results_a, results_b)

        # With k=60, weight_a=0.5, weight_b=0.5:
        # item1: rank_a=1, rank_b=3 (missing): 0.5/(60+1) + 0.5/(60+3)
        # item2: rank_a=2, rank_b=1: 0.5/(60+2) + 0.5/(60+1)
        # item3: rank_a=3 (missing), rank_b=2: 0.5/(60+3) + 0.5/(60+2)
        result_dict = dict(result)
        assert "item1" in result_dict
        assert "item2" in result_dict
        assert "item3" in result_dict

        # item2 should have highest score (appears first in both lists)
        assert result[0][0] == "item2"

    def test_rrf_fusion_custom_k(self) -> None:
        """Test RRF fusion with custom k parameter."""
        results_a: list[ScoreTuple] = [("item1", 1.0)]
        results_b: list[ScoreTuple] = [("item1", 1.0)]

        result = rrf_fusion(results_a, results_b, k=10)

        # Both rank 1: 0.5/(10+1) + 0.5/(10+1) = 1/11
        result_dict = dict(result)
        assert result_dict["item1"] == pytest.approx(1.0 / 11.0)

    def test_rrf_fusion_custom_weights(self) -> None:
        """Test RRF fusion with custom weights."""
        results_a: list[ScoreTuple] = [("item1", 1.0)]
        results_b: list[ScoreTuple] = [("item2", 1.0)]

        result = rrf_fusion(results_a, results_b, weight_a=0.8, weight_b=0.2)

        # item1: 0.8/(60+1) + 0.2/(60+2)
        # item2: 0.8/(60+2) + 0.2/(60+1)
        result_dict = dict(result)
        expected_item1 = 0.8 / 61 + 0.2 / 62
        expected_item2 = 0.8 / 62 + 0.2 / 61
        assert result_dict["item1"] == pytest.approx(expected_item1)
        assert result_dict["item2"] == pytest.approx(expected_item2)

    def test_rrf_fusion_sorted_by_score_descending(self) -> None:
        """Test that RRF results are sorted by score in descending order."""
        results_a: list[ScoreTuple] = [("item1", 10.0), ("item2", 5.0), ("item3", 2.0)]
        results_b: list[ScoreTuple] = [("item3", 20.0), ("item2", 15.0), ("item1", 10.0)]

        result = rrf_fusion(results_a, results_b)

        # Check that scores are in descending order
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_fusion_empty_results(self) -> None:
        """Test RRF fusion with empty result sets."""
        results_a: list[ScoreTuple] = []
        results_b: list[ScoreTuple] = []

        result = rrf_fusion(results_a, results_b)

        assert result == []


class TestL2ToCosineSimiliarity:
    """Tests for l2_to_cosine_similarity function."""

    def test_l2_to_cosine_similarity_zero_distance(self) -> None:
        """Test converting zero L2 distance returns 1.0 similarity."""
        result = l2_to_cosine_similarity(0.0)

        assert result == pytest.approx(1.0)

    def test_l2_to_cosine_similarity_max_distance(self) -> None:
        """Test converting max L2 distance (2.0) returns 0.0 similarity."""
        result = l2_to_cosine_similarity(2.0)

        assert result == pytest.approx(0.0)

    def test_l2_to_cosine_similarity_mid_distance(self) -> None:
        """Test converting mid-range L2 distance."""
        result = l2_to_cosine_similarity(1.0)

        assert result == pytest.approx(0.5)

    def test_l2_to_cosine_similarity_arbitrary_values(self) -> None:
        """Test converting arbitrary L2 distances."""
        assert l2_to_cosine_similarity(0.5) == pytest.approx(0.75)
        assert l2_to_cosine_similarity(1.5) == pytest.approx(0.25)


class TestCreateFTSIndex:
    """Tests for create_fts_index function."""

    def test_create_fts_index_single_column(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test creating FTS index on single text column."""
        create_fts_index(temp_duckdb_with_test_table, "test_table", "id", "title")

        # Verify index was created by checking we can still query the table
        result = temp_duckdb_with_test_table.execute("""
            SELECT COUNT(*) FROM test_table
        """).fetchone()

        assert result == (3,)

    def test_create_fts_index_multiple_columns(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test creating FTS index on multiple text columns."""
        create_fts_index(temp_duckdb_with_test_table, "test_table", "id", "title", "description")

        # Verify index was created successfully
        result = temp_duckdb_with_test_table.execute("""
            SELECT COUNT(*) FROM test_table
        """).fetchone()

        assert result == (3,)

    def test_create_fts_index_custom_parameters(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test creating FTS index with custom stemmer and stopwords."""
        create_fts_index(
            temp_duckdb_with_test_table, "test_table", "id", "title", stemmer="none", stopwords="none", lower=1
        )

        # Index should be created successfully
        result = temp_duckdb_with_test_table.execute("""
            SELECT id FROM test_table LIMIT 1
        """).fetchone()

        assert result is not None

    def test_create_fts_index_no_text_columns_raises(
        self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection
    ) -> None:
        """Test that creating FTS index without text columns raises ValueError."""
        with pytest.raises(ValueError, match="At least one text column must be specified"):
            create_fts_index(temp_duckdb_with_test_table, "test_table", "id")


class TestCreateHNSWIndex:
    """Tests for create_hnsw_index function."""

    def test_create_hnsw_index_default_name(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test creating HNSW index with default index name."""
        create_hnsw_index(temp_duckdb_with_test_table, "test_table", "embedding")

        # Verify index was created by checking system tables
        result = temp_duckdb_with_test_table.execute("""
            SELECT index_name FROM duckdb_indexes()
            WHERE index_name = 'idx_test_table_embedding_hnsw'
        """).fetchone()

        assert result is not None

    def test_create_hnsw_index_custom_name(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test creating HNSW index with custom index name."""
        create_hnsw_index(temp_duckdb_with_test_table, "test_table", "embedding", index_name="custom_idx")

        # Verify custom index name was used
        result = temp_duckdb_with_test_table.execute("""
            SELECT index_name FROM duckdb_indexes()
            WHERE index_name = 'custom_idx'
        """).fetchone()

        assert result is not None

    def test_create_hnsw_index_custom_parameters(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test creating HNSW index with custom parameters."""
        create_hnsw_index(
            temp_duckdb_with_test_table,
            "test_table",
            "embedding",
            metric="l2sq",
            ef_construction=64,
            ef_search=32,
            m=8,
        )

        # Verify index was created
        result = temp_duckdb_with_test_table.execute("""
            SELECT index_name FROM duckdb_indexes()
            WHERE index_name = 'idx_test_table_embedding_hnsw'
        """).fetchone()

        assert result is not None


class TestDropSearchIndexes:
    """Tests for drop_search_indexes function."""

    def test_drop_search_indexes_fts_only(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test dropping FTS index only."""
        # Create FTS index
        create_fts_index(temp_duckdb_with_test_table, "test_table", "id", "title")

        # Drop indexes (should not raise any errors)
        drop_search_indexes(temp_duckdb_with_test_table, "test_table")

        # Table should still exist
        result = temp_duckdb_with_test_table.execute("SELECT COUNT(*) FROM test_table").fetchone()
        assert result == (3,)

    def test_drop_search_indexes_hnsw_and_fts(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test dropping both HNSW and FTS indexes."""
        # Create FTS index
        create_fts_index(temp_duckdb_with_test_table, "test_table", "id", "title")

        # Drop indexes (including non-existent HNSW - should handle gracefully)
        drop_search_indexes(temp_duckdb_with_test_table, "test_table", hnsw_index_name="test_hnsw_idx")

        # Table should still exist
        result = temp_duckdb_with_test_table.execute("SELECT COUNT(*) FROM test_table").fetchone()
        assert result == (3,)

    def test_drop_search_indexes_nonexistent(self, temp_duckdb_with_test_table: duckdb.DuckDBPyConnection) -> None:
        """Test dropping indexes that don't exist doesn't raise error."""
        # Should not raise any exception
        drop_search_indexes(temp_duckdb_with_test_table, "test_table", hnsw_index_name="nonexistent_idx")
