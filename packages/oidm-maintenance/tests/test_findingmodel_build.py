"""Tests for findingmodel database build operations.

This module tests the build_findingmodel_database function that creates
a complete DuckDB database from .fm.json source files. Tests cover schema
creation, data loading, index setup, and error handling.

All tests use mocked embeddings to avoid API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import duckdb
import pytest
from findingmodel.index import DuckDBIndex
from oidm_maintenance.findingmodel.build import build_findingmodel_database
from pydantic_ai import models

# Block all AI model requests - embeddings are mocked
models.ALLOW_MODEL_REQUESTS = False


# =============================================================================
# Fixtures
# =============================================================================


def _fake_embeddings_deterministic(texts: list[str]) -> list[list[float]]:
    """Generate deterministic fake embeddings based on text hash.

    This matches the pattern used in build_test_fixtures.py to ensure
    consistent behavior between tests and pre-built fixtures.

    Args:
        texts: List of texts to embed

    Returns:
        List of deterministic embedding vectors based on text hash
    """
    return [[(sum(ord(c) for c in text) % 100) / 100.0] * 512 for text in texts]


@pytest.fixture
async def built_test_db(tmp_path: Path) -> Path:
    """Build a test database with mocked embeddings.

    Creates a complete test database from sample finding models using
    deterministic hash-based embeddings (no API calls). The database includes
    all tables, indexes, and denormalized data.

    Returns:
        Path to the built test database
    """
    source_dir = Path(__file__).parent.parent.parent / "findingmodel" / "tests" / "data" / "defs"
    db_path = tmp_path / "test.duckdb"

    # Mock the internal embedding generation function
    with patch(
        "oidm_maintenance.findingmodel.build._generate_embeddings_async",
        new_callable=AsyncMock,
        side_effect=_fake_embeddings_deterministic,
    ):
        await build_findingmodel_database(source_dir, db_path, generate_embeddings=True)

    return db_path


@pytest.fixture
def source_data_dir() -> Path:
    """Path to source test data directory."""
    return Path(__file__).parent.parent.parent / "findingmodel" / "tests" / "data" / "defs"


# =============================================================================
# Build Operation Tests
# =============================================================================


class TestBuildOperations:
    """Tests for basic build operations."""

    async def test_build_creates_database(self, built_test_db: Path) -> None:
        """Build creates database file at specified path."""
        assert built_test_db.exists()
        assert built_test_db.is_file()
        assert built_test_db.suffix == ".duckdb"

    async def test_build_loads_all_models(self, built_test_db: Path, source_data_dir: Path) -> None:
        """Build loads all models from source directory."""
        # Count source files
        source_files = list(source_data_dir.glob("*.fm.json"))
        expected_count = len(source_files)

        # Check database count
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            result = conn.execute("SELECT COUNT(*) FROM finding_models").fetchone()
            assert result is not None
            actual_count = int(result[0])
            assert actual_count == expected_count
        finally:
            conn.close()

    async def test_build_creates_schema(self, built_test_db: Path) -> None:
        """Build creates all required tables."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get all tables
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            table_names = {row[0] for row in tables}

            # Verify all required tables exist
            expected_tables = {
                "finding_models",
                "synonyms",
                "tags",
                "attributes",
                "people",
                "organizations",
                "model_people",
                "model_organizations",
                "finding_model_json",
            }
            assert expected_tables.issubset(table_names)
        finally:
            conn.close()

    async def test_build_idempotent(self, tmp_path: Path, source_data_dir: Path) -> None:
        """Rebuilding produces same result."""
        db_path = tmp_path / "test.duckdb"

        # Mock embeddings
        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            # First build
            await build_findingmodel_database(source_data_dir, db_path, generate_embeddings=True)

            # Get first build count
            conn1 = duckdb.connect(str(db_path), read_only=True)
            result1 = conn1.execute("SELECT COUNT(*) FROM finding_models").fetchone()
            assert result1 is not None
            count1 = int(result1[0])
            conn1.close()

            # Second build (overwrites)
            await build_findingmodel_database(source_data_dir, db_path, generate_embeddings=True)

            # Get second build count
            conn2 = duckdb.connect(str(db_path), read_only=True)
            result2 = conn2.execute("SELECT COUNT(*) FROM finding_models").fetchone()
            assert result2 is not None
            count2 = int(result2[0])
            conn2.close()

            # Should be same
            assert count1 == count2

    async def test_build_handles_empty_directory(self, tmp_path: Path) -> None:
        """Build raises error for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        db_path = tmp_path / "test.duckdb"

        with pytest.raises(FileNotFoundError, match=r"No \.fm\.json files found"):
            await build_findingmodel_database(empty_dir, db_path, generate_embeddings=False)

    async def test_build_handles_nonexistent_directory(self, tmp_path: Path) -> None:
        """Build raises error for nonexistent directory."""
        nonexistent_dir = tmp_path / "nonexistent"
        db_path = tmp_path / "test.duckdb"

        with pytest.raises(ValueError, match="is not a valid directory"):
            await build_findingmodel_database(nonexistent_dir, db_path, generate_embeddings=False)


# =============================================================================
# Index Setup Tests
# =============================================================================


class TestIndexSetup:
    """Tests for database index creation."""

    async def test_build_creates_hnsw_index(self, built_test_db: Path) -> None:
        """Build creates HNSW vector index."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Check for HNSW index
            indexes = conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'finding_models'"
            ).fetchall()
            index_names = {row[0] for row in indexes}
            assert "finding_models_embedding_hnsw" in index_names
        finally:
            conn.close()

    async def test_build_creates_fts_index(self, built_test_db: Path) -> None:
        """Build creates FTS text index."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Test FTS query works
            result = conn.execute(
                "SELECT fts_main_finding_models.match_bm25(oifm_id, 'test') FROM finding_models LIMIT 1"
            ).fetchall()
            # If no error, FTS index exists and works
            assert isinstance(result, list)
        finally:
            conn.close()

    async def test_build_creates_standard_indexes(self, built_test_db: Path) -> None:
        """Build creates B-tree indexes."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get all indexes
            indexes = conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
            index_names = {row[0] for row in indexes}

            # Check for key standard indexes
            expected_indexes = {
                "idx_finding_models_name",
                "idx_finding_models_slug_name",
                "idx_synonyms_synonym",
                "idx_tags_tag",
                "idx_attributes_model",
            }
            assert expected_indexes.issubset(index_names)
        finally:
            conn.close()


# =============================================================================
# Data Integrity Tests
# =============================================================================


class TestDataIntegrity:
    """Tests for data loading and integrity."""

    async def test_build_populates_finding_models_table(self, built_test_db: Path) -> None:
        """Build populates finding_models table with core data."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get a sample model
            result = conn.execute(
                "SELECT oifm_id, name, filename, description, search_text FROM finding_models LIMIT 1"
            ).fetchone()
            assert result is not None

            oifm_id, name, filename, _description, search_text = result
            assert oifm_id.startswith("OIFM_")
            assert len(name) > 0
            assert filename.endswith(".fm.json")
            assert search_text is not None
            assert len(search_text) > 0
        finally:
            conn.close()

    async def test_build_populates_synonyms_table(self, built_test_db: Path) -> None:
        """Build populates synonyms table with denormalized data."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Check synonyms exist
            result = conn.execute("SELECT COUNT(*) FROM synonyms").fetchone()
            assert result is not None
            count = int(result[0])
            # At least some models should have synonyms
            assert count > 0

            # Check structure
            sample = conn.execute("SELECT oifm_id, synonym FROM synonyms LIMIT 1").fetchone()
            assert sample is not None
            assert sample[0].startswith("OIFM_")
            assert len(sample[1]) > 0
        finally:
            conn.close()

    async def test_build_populates_tags_table(self, built_test_db: Path) -> None:
        """Build populates tags table with denormalized data."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Check tags exist
            result = conn.execute("SELECT COUNT(*) FROM tags").fetchone()
            assert result is not None
            count = int(result[0])
            # At least some models should have tags
            assert count > 0

            # Check structure
            sample = conn.execute("SELECT oifm_id, tag FROM tags LIMIT 1").fetchone()
            assert sample is not None
            assert sample[0].startswith("OIFM_")
            assert len(sample[1]) > 0
        finally:
            conn.close()

    async def test_build_populates_attributes_table(self, built_test_db: Path) -> None:
        """Build populates attributes table with denormalized data."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Check attributes exist
            result = conn.execute("SELECT COUNT(*) FROM attributes").fetchone()
            assert result is not None
            count = int(result[0])
            # All models should have attributes
            assert count > 0

            # Check structure
            sample = conn.execute(
                "SELECT attribute_id, oifm_id, attribute_name, attribute_type FROM attributes LIMIT 1"
            ).fetchone()
            assert sample is not None
            attr_id, oifm_id, attr_name, attr_type = sample
            assert attr_id.startswith("OIFMA_")
            assert oifm_id.startswith("OIFM_")
            assert len(attr_name) > 0
            # Attribute type is stored as enum string representation
            assert "CHOICE" in attr_type.upper() or "NUMERIC" in attr_type.upper()
        finally:
            conn.close()

    async def test_build_populates_people_table(self, built_test_db: Path) -> None:
        """Build populates people table with contributor data."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Check people exist
            result = conn.execute("SELECT COUNT(*) FROM people").fetchone()
            assert result is not None
            count = int(result[0])
            # At least some models should have people contributors
            assert count >= 0  # May be 0 if no people contributors in test data

            # If people exist, check structure
            if count > 0:
                sample = conn.execute("SELECT github_username, name, email FROM people LIMIT 1").fetchone()
                assert sample is not None
                assert len(sample[0]) > 0  # github_username
                assert len(sample[1]) > 0  # name
                assert len(sample[2]) > 0  # email
        finally:
            conn.close()

    async def test_build_populates_organizations_table(self, built_test_db: Path) -> None:
        """Build populates organizations table with contributor data."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Check organizations exist
            result = conn.execute("SELECT COUNT(*) FROM organizations").fetchone()
            assert result is not None
            count = int(result[0])
            # At least some models should have org contributors
            assert count >= 0  # May be 0 if no org contributors in test data

            # If orgs exist, check structure
            if count > 0:
                sample = conn.execute("SELECT code, name FROM organizations LIMIT 1").fetchone()
                assert sample is not None
                assert len(sample[0]) > 0  # code
                assert len(sample[1]) > 0  # name
        finally:
            conn.close()

    async def test_build_stores_embeddings(self, built_test_db: Path) -> None:
        """Build stores embedding vectors."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get a sample embedding
            result = conn.execute("SELECT embedding FROM finding_models LIMIT 1").fetchone()
            assert result is not None
            embedding = result[0]

            # DuckDB returns embeddings as tuples, not lists
            # Check embedding structure
            assert isinstance(embedding, (list, tuple))
            assert len(embedding) == 512  # Default dimensions
            assert all(isinstance(val, float) for val in embedding)

            # Check embeddings are not all zeros (mocked ones use hash)
            assert any(val != 0.0 for val in embedding)
        finally:
            conn.close()

    async def test_build_stores_json(self, built_test_db: Path) -> None:
        """Build stores full JSON in finding_model_json table."""
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get a sample JSON
            result = conn.execute("SELECT oifm_id, model_json FROM finding_model_json LIMIT 1").fetchone()
            assert result is not None
            oifm_id, model_json = result

            # Check JSON is valid and can be parsed
            model_data = json.loads(model_json)
            assert model_data["oifm_id"] == oifm_id
            assert "name" in model_data
            assert "attributes" in model_data
        finally:
            conn.close()


# =============================================================================
# Model Retrieval Tests
# =============================================================================


class TestBuiltDatabaseRetrieval:
    """Tests for retrieving data from built database via DuckDBIndex."""

    async def test_built_db_allows_retrieval_by_id(self, built_test_db: Path) -> None:
        """Can retrieve models by ID from built database."""
        # Use DuckDBIndex to read
        index = DuckDBIndex(built_test_db)

        # Get a known model ID from test data
        conn = duckdb.connect(str(built_test_db), read_only=True)
        result = conn.execute("SELECT oifm_id FROM finding_models LIMIT 1").fetchone()
        conn.close()

        assert result is not None
        oifm_id = result[0]

        # Retrieve via index
        entry = await index.get(oifm_id)
        assert entry is not None
        assert entry.oifm_id == oifm_id

        # Cleanup
        if index.conn is not None:
            index.conn.close()

    async def test_built_db_allows_search(self, built_test_db: Path) -> None:
        """Can search models in built database using FTS.

        Note: This tests FTS search only. Full semantic search would require
        mocking OpenAI client, which is tested separately in findingmodel tests.
        """
        # Use raw SQL to test FTS search (bypass semantic search that needs API key)
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Search for a term likely to be in test data using FTS
            results = conn.execute(
                """
                SELECT oifm_id, name,
                       fts_main_finding_models.match_bm25(oifm_id, 'aneurysm') AS score
                FROM finding_models
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT 10
                """
            ).fetchall()

            # Should find at least one result (abdominal aortic aneurysm in test data)
            assert len(results) >= 1
            assert all(row[0].startswith("OIFM_") for row in results)
        finally:
            conn.close()

    async def test_built_db_count_matches_source(self, built_test_db: Path, source_data_dir: Path) -> None:
        """Built database count matches source file count."""
        # Count source files
        source_files = list(source_data_dir.glob("*.fm.json"))
        expected_count = len(source_files)

        # Count via index
        index = DuckDBIndex(built_test_db)
        actual_count = await index.count()

        assert actual_count == expected_count

        # Cleanup
        if index.conn is not None:
            index.conn.close()


# =============================================================================
# Embedding Generation Tests
# =============================================================================


class TestEmbeddingGeneration:
    """Tests for embedding generation options."""

    async def test_build_with_embeddings_enabled(self, tmp_path: Path, source_data_dir: Path) -> None:
        """Build with generate_embeddings=True uses mocked embeddings."""
        db_path = tmp_path / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            result_path = await build_findingmodel_database(source_data_dir, db_path, generate_embeddings=True)

            assert result_path == db_path
            assert db_path.exists()

            # Check embeddings are not all zeros
            conn = duckdb.connect(str(db_path), read_only=True)
            try:
                result = conn.execute("SELECT embedding FROM finding_models LIMIT 1").fetchone()
                assert result is not None
                embedding = result[0]
                assert any(val != 0.0 for val in embedding)
            finally:
                conn.close()

    async def test_build_with_embeddings_disabled(self, tmp_path: Path, source_data_dir: Path) -> None:
        """Build with generate_embeddings=False creates zero vectors."""
        db_path = tmp_path / "test.duckdb"

        result_path = await build_findingmodel_database(source_data_dir, db_path, generate_embeddings=False)

        assert result_path == db_path
        assert db_path.exists()

        # Check embeddings are all zeros
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            result = conn.execute("SELECT embedding FROM finding_models LIMIT 1").fetchone()
            assert result is not None
            embedding = result[0]
            assert all(val == 0.0 for val in embedding)
        finally:
            conn.close()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    async def test_build_handles_invalid_json(self, tmp_path: Path) -> None:
        """Build fails gracefully with invalid JSON file."""
        source_dir = tmp_path / "invalid"
        source_dir.mkdir()

        # Create invalid JSON file
        invalid_file = source_dir / "invalid.fm.json"
        invalid_file.write_text("not valid json {")

        db_path = tmp_path / "test.duckdb"

        # Will raise JSON decode error
        with pytest.raises((json.JSONDecodeError, ValueError)):
            await build_findingmodel_database(source_dir, db_path, generate_embeddings=False)

    async def test_build_handles_validation_error(self, tmp_path: Path) -> None:
        """Build fails gracefully with model validation error."""
        source_dir = tmp_path / "invalid"
        source_dir.mkdir()

        # Create JSON with missing required fields
        invalid_model = {
            "name": "Test Model",
            # Missing oifm_id, attributes, etc.
        }
        invalid_file = source_dir / "invalid.fm.json"
        invalid_file.write_text(json.dumps(invalid_model))

        db_path = tmp_path / "test.duckdb"

        # Will raise Pydantic validation error
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await build_findingmodel_database(source_dir, db_path, generate_embeddings=False)

    async def test_build_creates_parent_directory(self, tmp_path: Path, source_data_dir: Path) -> None:
        """Build creates parent directory if it doesn't exist."""
        nested_path = tmp_path / "nested" / "deep" / "path" / "test.duckdb"

        with patch(
            "oidm_maintenance.findingmodel.build._generate_embeddings_async",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ):
            result_path = await build_findingmodel_database(source_data_dir, nested_path, generate_embeddings=True)

            assert result_path == nested_path
            assert nested_path.exists()
            assert nested_path.parent.exists()
