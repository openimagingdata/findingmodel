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
from findingmodel.index import FindingModelIndex
from oidm_maintenance.config import MaintenanceSettings
from oidm_maintenance.findingmodel.build import build_findingmodel_database
from pydantic import SecretStr
from pydantic_ai import models

# Block all AI model requests - embeddings are mocked
models.ALLOW_MODEL_REQUESTS = False


# =============================================================================
# Fixtures
# =============================================================================


def _fake_embeddings_deterministic(
    texts: list[str],
    *,
    api_key: str | None = None,
    model: str | None = None,
    dimensions: int = 512,
    cache: object | None = None,
) -> list[list[float]]:
    """Generate deterministic fake embeddings based on text hash.

    This matches the pattern used in build_test_fixtures.py to ensure
    consistent behavior between tests and pre-built fixtures.

    Args:
        texts: List of texts to embed

    Returns:
        List of deterministic embedding vectors based on text hash
    """
    _ = (api_key, model, cache)
    return [[(sum(ord(c) for c in text) % 100) / 100.0] * dimensions for text in texts]


def _fake_settings_with_openai_key() -> MaintenanceSettings:
    return MaintenanceSettings(openai_api_key=SecretStr("fake-key"))


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
    with (
        patch("oidm_maintenance.findingmodel.build.get_settings", return_value=_fake_settings_with_openai_key()),
        patch(
            "oidm_maintenance.findingmodel.build.get_embeddings_batch",
            new_callable=AsyncMock,
            side_effect=_fake_embeddings_deterministic,
        ),
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
        with (
            patch("oidm_maintenance.findingmodel.build.get_settings", return_value=_fake_settings_with_openai_key()),
            patch(
                "oidm_maintenance.findingmodel.build.get_embeddings_batch",
                new_callable=AsyncMock,
                side_effect=_fake_embeddings_deterministic,
            ),
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
    """Tests for retrieving data from built database via FindingModelIndex."""

    async def test_built_db_allows_retrieval_by_id(self, built_test_db: Path) -> None:
        """Can retrieve models by ID from built database."""
        # Use FindingModelIndex to read
        index = FindingModelIndex(built_test_db)

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
        index = FindingModelIndex(built_test_db)
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

        with (
            patch("oidm_maintenance.findingmodel.build.get_settings", return_value=_fake_settings_with_openai_key()),
            patch(
                "oidm_maintenance.findingmodel.build.get_embeddings_batch",
                new_callable=AsyncMock,
                side_effect=_fake_embeddings_deterministic,
            ),
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

        with (
            patch("oidm_maintenance.findingmodel.build.get_settings", return_value=_fake_settings_with_openai_key()),
            patch(
                "oidm_maintenance.findingmodel.build.get_embeddings_batch",
                new_callable=AsyncMock,
                side_effect=_fake_embeddings_deterministic,
            ),
        ):
            result_path = await build_findingmodel_database(source_data_dir, nested_path, generate_embeddings=True)

            assert result_path == nested_path
            assert nested_path.exists()
            assert nested_path.parent.exists()


# =============================================================================
# Duplicate Validation Tests
# =============================================================================


class TestDuplicateValidation:
    """Tests for duplicate ID detection during build.

    These tests verify that the database properly rejects duplicate
    OIFM IDs and attribute IDs via PRIMARY KEY constraints.
    """

    async def test_build_rejects_duplicate_oifm_id(self, tmp_path: Path) -> None:
        """Build fails when source contains duplicate OIFM IDs."""
        source_dir = tmp_path / "duplicates"
        source_dir.mkdir()

        # Create two models with the same OIFM ID
        model1 = {
            "oifm_id": "OIFM_TEST_000001",
            "name": "Model One",
            "description": "First model",
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000001",
                    "name": "attr1",
                    "description": "Attribute 1",
                    "type": "numeric",
                }
            ],
        }
        model2 = {
            "oifm_id": "OIFM_TEST_000001",  # Duplicate OIFM ID!
            "name": "Model Two",
            "description": "Second model with same ID",
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000002",
                    "name": "attr2",
                    "description": "Attribute 2",
                    "type": "numeric",
                }
            ],
        }

        (source_dir / "model1.fm.json").write_text(json.dumps(model1))
        (source_dir / "model2.fm.json").write_text(json.dumps(model2))

        db_path = tmp_path / "test.duckdb"

        # Build should fail due to PRIMARY KEY constraint violation
        with pytest.raises(duckdb.ConstraintException, match="Duplicate key"):
            await build_findingmodel_database(source_dir, db_path, generate_embeddings=False)

    async def test_build_rejects_duplicate_attribute_id(self, tmp_path: Path) -> None:
        """Build fails when source contains duplicate attribute IDs across models."""
        source_dir = tmp_path / "duplicates"
        source_dir.mkdir()

        # Create two models with attributes having the same ID
        model1 = {
            "oifm_id": "OIFM_TEST_000001",
            "name": "Model One",
            "description": "First model",
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000001",  # This ID will be duplicated
                    "name": "shared_attr",
                    "description": "Attribute with shared ID",
                    "type": "numeric",
                }
            ],
        }
        model2 = {
            "oifm_id": "OIFM_TEST_000002",  # Different OIFM ID
            "name": "Model Two",
            "description": "Second model",
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000001",  # Duplicate attribute ID!
                    "name": "another_attr",
                    "description": "Different attribute, same ID",
                    "type": "numeric",
                }
            ],
        }

        (source_dir / "model1.fm.json").write_text(json.dumps(model1))
        (source_dir / "model2.fm.json").write_text(json.dumps(model2))

        db_path = tmp_path / "test.duckdb"

        # Build should fail due to PRIMARY KEY constraint violation on attributes table
        with pytest.raises(duckdb.ConstraintException, match="Duplicate key"):
            await build_findingmodel_database(source_dir, db_path, generate_embeddings=False)

    async def test_embeddings_error_missing_api_key(self, tmp_path: Path) -> None:
        """Build fails with clear error when OpenAI API key not configured."""
        source_dir = tmp_path / "models"
        source_dir.mkdir()

        # Create a valid model
        model = {
            "oifm_id": "OIFM_TEST_000001",
            "name": "Test Model",
            "description": "Test model for embeddings error",
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000001",
                    "name": "test_attr",
                    "description": "Test attribute",
                    "type": "numeric",
                }
            ],
        }
        (source_dir / "model.fm.json").write_text(json.dumps(model))

        db_path = tmp_path / "test.duckdb"

        # Mock settings to return None for openai_api_key
        from oidm_maintenance.config import MaintenanceSettings

        mock_settings = MaintenanceSettings(openai_api_key=None)

        # Patch get_settings to return our mock
        with (
            patch("oidm_maintenance.findingmodel.build.get_settings", return_value=mock_settings),
            pytest.raises(RuntimeError, match="OPENAI_API_KEY not configured"),
        ):
            await build_findingmodel_database(source_dir, db_path, generate_embeddings=True)

    async def test_embeddings_error_none_embeddings(self, tmp_path: Path, source_data_dir: Path) -> None:
        """Build fails when embedding generation returns None values.

        This tests the internal validation that checks for None embeddings
        from the underlying embedding provider.
        """
        db_path = tmp_path / "test.duckdb"

        # Mock settings to have valid API key (so we get past the API key check)
        from oidm_maintenance.config import MaintenanceSettings
        from pydantic import SecretStr

        mock_settings = MaintenanceSettings(openai_api_key=SecretStr("fake-key-for-test"))

        # Mock the underlying get_embeddings_batch to return None values
        # This simulates the OpenAI API returning None for a failed embedding
        def mock_batch_embeddings_with_none(
            texts: list[str],
            *,
            api_key: str | None = None,
            model: str | None = None,
            dimensions: int = 512,
            cache: object | None = None,
        ) -> list[list[float] | None]:
            """Return None embeddings to simulate API failure."""
            _ = (api_key, model, dimensions, cache)
            return [None] * len(texts)

        # Patch both settings and the embedding batch function
        with (
            patch("oidm_maintenance.findingmodel.build.get_settings", return_value=mock_settings),
            patch(
                "oidm_maintenance.findingmodel.build.get_embeddings_batch",
                new_callable=AsyncMock,
                side_effect=mock_batch_embeddings_with_none,
            ),
            pytest.raises(RuntimeError, match="Failed to generate embedding for model at index 0"),
        ):
            await build_findingmodel_database(source_data_dir, db_path, generate_embeddings=True)


# =============================================================================
# Data Correctness Tests (Phase 4)
# =============================================================================


class TestDataCorrectness:
    """Tests for data correctness and integrity in built database.

    These tests verify that the build process correctly transforms and stores
    data according to schema requirements and business logic.
    """

    async def test_duplicate_name_slug_rejection(self, tmp_path: Path) -> None:
        """Build fails when source contains duplicate names (same slug).

        Two models with different filenames but same name should be rejected
        due to UNIQUE constraint on name or slug_name in finding_models table.
        """
        source_dir = tmp_path / "duplicate_names"
        source_dir.mkdir()

        # Create two models with same name but different OIFM IDs
        # Use minimal valid model with numeric attribute to avoid validation errors
        model1 = {
            "oifm_id": "OIFM_TEST_000001",
            "name": "Test Finding",  # Same name
            "description": "First version",
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000001",
                    "name": "size",
                    "description": "Test attribute",
                    "type": "numeric",
                }
            ],
        }
        model2 = {
            "oifm_id": "OIFM_TEST_000002",
            "name": "Test Finding",  # Same name - will create same slug
            "description": "Second version",
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000002",
                    "name": "diameter",
                    "description": "Different attribute",
                    "type": "numeric",
                }
            ],
        }

        (source_dir / "test_finding_1.fm.json").write_text(json.dumps(model1))
        (source_dir / "test_finding_2.fm.json").write_text(json.dumps(model2))

        db_path = tmp_path / "test.duckdb"

        # Build should fail due to UNIQUE constraint on name or slug_name
        with pytest.raises(duckdb.ConstraintException, match=r"Duplicate key|UNIQUE constraint"):
            await build_findingmodel_database(source_dir, db_path, generate_embeddings=False)

    async def test_slug_name_equals_normalize_name(self, built_test_db: Path, source_data_dir: Path) -> None:
        """Verify slug_name equals normalize_name(name) for all models.

        The slug_name column should always be the normalized version of the name,
        computed using the normalize_name() function.
        """
        from findingmodel.common import normalize_name

        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get all models
            results = conn.execute("SELECT name, slug_name FROM finding_models").fetchall()

            # Verify slug matches normalized name
            for name, slug_name in results:
                expected_slug = normalize_name(name)
                assert slug_name == expected_slug, (
                    f"slug_name mismatch for '{name}': expected '{expected_slug}', got '{slug_name}'"
                )
        finally:
            conn.close()

    async def test_file_hash_sha256_correctness(self, built_test_db: Path, source_data_dir: Path) -> None:
        """Verify file_hash_sha256 matches actual file hash.

        The stored hash should match the SHA256 hash of the source .fm.json file.
        """
        import hashlib

        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get all models with their file hashes
            results = conn.execute("SELECT filename, file_hash_sha256 FROM finding_models").fetchall()

            # Verify each hash
            for filename, stored_hash in results:
                file_path = source_data_dir / filename
                assert file_path.exists(), f"Source file not found: {file_path}"

                # Compute actual hash
                actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

                assert stored_hash == actual_hash, (
                    f"Hash mismatch for {filename}: expected {actual_hash}, got {stored_hash}"
                )
        finally:
            conn.close()

    async def test_search_text_contains_expected_components(self, built_test_db: Path) -> None:
        """Verify search_text includes name, description, synonyms, tags, and attributes.

        The search_text field should be a concatenation of all searchable text
        from the model: name, description, synonyms, tags, and attribute names.
        """
        conn = duckdb.connect(str(built_test_db), read_only=True)
        try:
            # Get abdominal aortic aneurysm - known to have synonyms and attributes
            result = conn.execute(
                """
                SELECT fm.name, fm.description, fm.search_text, fm.oifm_id
                FROM finding_models fm
                WHERE fm.oifm_id = 'OIFM_MSFT_134126'
                """
            ).fetchone()

            assert result is not None, "Test model OIFM_MSFT_134126 not found"
            name, description, search_text, oifm_id = result

            # Verify search_text contains core components
            assert name in search_text, f"search_text missing name: {name}"
            if description:
                assert description in search_text, "search_text missing description"

            # Check for known synonym (AAA)
            synonyms = conn.execute("SELECT synonym FROM synonyms WHERE oifm_id = ?", (oifm_id,)).fetchall()
            for (synonym,) in synonyms:
                assert synonym in search_text, f"search_text missing synonym: {synonym}"

            # Check for tags if present
            tags = conn.execute("SELECT tag FROM tags WHERE oifm_id = ?", (oifm_id,)).fetchall()
            for (tag,) in tags:
                assert tag in search_text, f"search_text missing tag: {tag}"

            # Check for attribute names
            attributes = conn.execute("SELECT attribute_name FROM attributes WHERE oifm_id = ?", (oifm_id,)).fetchall()
            for (attr_name,) in attributes:
                assert attr_name in search_text, f"search_text missing attribute: {attr_name}"

        finally:
            conn.close()

    async def test_tags_synonyms_deduplication(self, tmp_path: Path) -> None:
        """Verify duplicate tags and synonyms are deduplicated.

        When a model has duplicate tags or synonyms in the source JSON,
        the build process should deduplicate them, storing only unique values.
        """
        source_dir = tmp_path / "duplicates"
        source_dir.mkdir()

        # Create model with duplicate tags and synonyms
        model = {
            "oifm_id": "OIFM_TEST_000003",
            "name": "Test Deduplication",
            "description": "Model with duplicates",
            "synonyms": ["SYN1", "SYN2", "SYN1", "SYN3", "SYN2"],  # Duplicates: SYN1, SYN2
            "tags": ["tag1", "tag2", "tag1", "tag3"],  # Duplicate: tag1
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000003",
                    "name": "presence",
                    "description": "Test attribute",
                    "type": "numeric",
                }
            ],
        }

        (source_dir / "test_dedup.fm.json").write_text(json.dumps(model))

        db_path = tmp_path / "test.duckdb"

        # Build should succeed (deduplication happens before insert)
        await build_findingmodel_database(source_dir, db_path, generate_embeddings=False)

        # Verify only unique values stored
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            # Count synonyms
            synonym_count = conn.execute("SELECT COUNT(*) FROM synonyms WHERE oifm_id = 'OIFM_TEST_000003'").fetchone()
            assert synonym_count is not None
            assert synonym_count[0] == 3, f"Expected 3 unique synonyms, got {synonym_count[0]}"

            # Verify unique synonyms
            synonyms = conn.execute(
                "SELECT synonym FROM synonyms WHERE oifm_id = 'OIFM_TEST_000003' ORDER BY synonym"
            ).fetchall()
            assert synonyms == [("SYN1",), ("SYN2",), ("SYN3",)]

            # Count tags
            tag_count = conn.execute("SELECT COUNT(*) FROM tags WHERE oifm_id = 'OIFM_TEST_000003'").fetchone()
            assert tag_count is not None
            assert tag_count[0] == 3, f"Expected 3 unique tags, got {tag_count[0]}"

            # Verify unique tags
            tags = conn.execute("SELECT tag FROM tags WHERE oifm_id = 'OIFM_TEST_000003' ORDER BY tag").fetchall()
            assert tags == [("tag1",), ("tag2",), ("tag3",)]
        finally:
            conn.close()

    async def test_contributor_ordering_and_role(self, tmp_path: Path) -> None:
        """Verify contributor display_order and role are correct.

        Contributors should be stored with:
        - display_order matching their position in the contributors list
        - role set to 'contributor' (default)
        - Separate handling for Person vs Organization contributors
        """
        source_dir = tmp_path / "contributors"
        source_dir.mkdir()

        # Create model with mixed Person and Organization contributors
        model = {
            "oifm_id": "OIFM_TEST_000004",
            "name": "Test Contributors",
            "description": "Model with contributors",
            "contributors": [
                {  # Order 0 - Person
                    "github_username": "person1",
                    "name": "First Person",
                    "email": "person1@example.com",
                    "organization_code": "MSFT",  # Must be 3-4 uppercase letters
                },
                {  # Order 1 - Organization
                    "name": "Test Organization",
                    "code": "MSFT",  # Must be 3-4 uppercase letters
                    "url": "https://example.com",
                },
                {  # Order 2 - Person (no organization)
                    "github_username": "person2",
                    "name": "Second Person",
                    "email": "person2@example.com",
                    "organization_code": "TEST",  # organization_code is required
                },
            ],
            "attributes": [
                {
                    "oifma_id": "OIFMA_TEST_000004",
                    "name": "presence",
                    "description": "Test attribute",
                    "type": "numeric",
                }
            ],
        }

        (source_dir / "test_contrib.fm.json").write_text(json.dumps(model))

        db_path = tmp_path / "test.duckdb"

        await build_findingmodel_database(source_dir, db_path, generate_embeddings=False)

        # Verify contributor data
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            # Check people contributors
            people_results = conn.execute(
                """
                SELECT person_id, role, display_order
                FROM model_people
                WHERE oifm_id = 'OIFM_TEST_000004'
                ORDER BY display_order
                """
            ).fetchall()

            assert len(people_results) == 2, f"Expected 2 person contributors, got {len(people_results)}"

            # Verify order and role
            assert people_results[0] == ("person1", "contributor", 0)
            assert people_results[1] == ("person2", "contributor", 2)

            # Check organization contributors
            org_results = conn.execute(
                """
                SELECT organization_id, role, display_order
                FROM model_organizations
                WHERE oifm_id = 'OIFM_TEST_000004'
                ORDER BY display_order
                """
            ).fetchall()

            assert len(org_results) == 1, f"Expected 1 organization contributor, got {len(org_results)}"
            assert org_results[0] == ("MSFT", "contributor", 1)

        finally:
            conn.close()
