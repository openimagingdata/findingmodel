"""Tests for the MCP server.

Note: These tests use mocking to avoid requiring a populated database.
Integration tests marked with @pytest.mark.callout use real databases.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from findingmodel.index import AttributeInfo, DuckDBIndex, IndexEntry
from findingmodel.mcp_server import (
    count_finding_models,
    get_finding_model,
    search_finding_models,
)


@pytest.mark.asyncio
async def test_search_finding_models_basic() -> None:
    """Test basic search functionality."""
    mock_entry = IndexEntry(
        oifm_id="OIFM_TEST_000001",
        name="Test Finding",
        slug_name="test_finding",
        filename="test_finding.fm.json",
        file_hash_sha256="abc123",
        description="A test finding",
        synonyms=["test"],
        tags=["test"],
        attributes=[AttributeInfo(attribute_id="OIFMA_TEST_000001", name="Severity", type="ChoiceAttribute")],
    )

    mock_index = AsyncMock()
    mock_index.search = AsyncMock(return_value=[mock_entry])
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await search_finding_models(query="pneumothorax", limit=5)

        assert result.query == "pneumothorax"
        assert result.limit == 5
        assert result.result_count == 1
        assert len(result.results) == 1
        assert result.results[0].oifm_id == "OIFM_TEST_000001"


@pytest.mark.asyncio
async def test_search_finding_models_limit_validation() -> None:
    """Test that limit is validated correctly."""
    mock_index = AsyncMock()
    mock_index.search = AsyncMock(return_value=[])
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        # Test minimum limit
        result_min = await search_finding_models(query="test", limit=0)
        assert result_min.limit == 1

        # Test maximum limit
        result_max = await search_finding_models(query="test", limit=200)
        assert result_max.limit == 100


@pytest.mark.asyncio
async def test_search_finding_models_with_tags() -> None:
    """Test search with tag filtering."""
    mock_index = AsyncMock()
    mock_index.search = AsyncMock(return_value=[])
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await search_finding_models(query="lesion", limit=10, tags=["chest"])

        assert result.query == "lesion"
        assert result.tags == ["chest"]
        mock_index.search.assert_called_once_with("lesion", limit=10, tags=["chest"])


@pytest.mark.asyncio
async def test_search_finding_models_empty_tags() -> None:
    """Test that empty tags list is normalized to None."""
    mock_index = AsyncMock()
    mock_index.search = AsyncMock(return_value=[])
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await search_finding_models(query="test", limit=5, tags=[])

        assert result.tags is None
        mock_index.search.assert_called_once_with("test", limit=5, tags=None)


@pytest.mark.asyncio
async def test_get_finding_model_by_id() -> None:
    """Test retrieving a finding model by OIFM ID."""
    mock_entry = IndexEntry(
        oifm_id="OIFM_TEST_000001",
        name="Test Finding",
        slug_name="test_finding",
        filename="test_finding.fm.json",
        file_hash_sha256="abc123",
        attributes=[AttributeInfo(attribute_id="OIFMA_TEST_000001", name="Severity", type="ChoiceAttribute")],
    )

    mock_index = AsyncMock()
    mock_index.get = AsyncMock(return_value=mock_entry)
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await get_finding_model(identifier="OIFM_TEST_000001")

        assert result is not None
        assert result.oifm_id == "OIFM_TEST_000001"
        mock_index.get.assert_called_once_with("OIFM_TEST_000001")


@pytest.mark.asyncio
async def test_get_finding_model_not_found() -> None:
    """Test retrieving a non-existent finding model."""
    mock_index = AsyncMock()
    mock_index.get = AsyncMock(return_value=None)
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await get_finding_model(identifier="NONEXISTENT_ID_12345")

        assert result is None


@pytest.mark.asyncio
async def test_count_finding_models() -> None:
    """Test getting statistics."""
    mock_index = AsyncMock()
    mock_index.count = AsyncMock(return_value=150)
    mock_index.count_people = AsyncMock(return_value=45)
    mock_index.count_organizations = AsyncMock(return_value=12)
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await count_finding_models()

        assert result == {
            "finding_models": 150,
            "people": 45,
            "organizations": 12,
        }


@pytest.mark.asyncio
async def test_search_result_structure() -> None:
    """Test that search results have the correct structure."""
    mock_entry = IndexEntry(
        oifm_id="OIFM_TEST_000001",
        name="Test Finding",
        slug_name="test_finding",
        filename="test_finding.fm.json",
        file_hash_sha256="abc123",
        description="A test finding",
        synonyms=["test1", "test2"],
        tags=["tag1", "tag2"],
        contributors=["contributor1"],
        attributes=[
            AttributeInfo(attribute_id="OIFMA_TEST_000001", name="Severity", type="ChoiceAttribute"),
            AttributeInfo(attribute_id="OIFMA_TEST_000002", name="Size", type="NumericAttribute"),
        ],
    )

    mock_index = AsyncMock()
    mock_index.search = AsyncMock(return_value=[mock_entry])
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await search_finding_models(query="test", limit=1)

        assert result.result_count == 1
        model = result.results[0]

        # Check required fields
        assert model.oifm_id == "OIFM_TEST_000001"
        assert model.name == "Test Finding"
        assert model.slug_name == "test_finding"
        assert model.filename == "test_finding.fm.json"

        # Check optional fields
        assert model.description == "A test finding"
        assert model.synonyms == ["test1", "test2"]
        assert model.tags == ["tag1", "tag2"]
        assert model.contributors == ["contributor1"]

        # Check attributes
        assert len(model.attributes) == 2
        assert model.attributes[0].attribute_id == "OIFMA_TEST_000001"
        assert model.attributes[0].name == "Severity"
        assert model.attributes[0].type == "ChoiceAttribute"


@pytest.mark.asyncio
@pytest.mark.callout
async def test_mcp_server_integration(tmp_path: Path) -> None:
    """Integration test: MCP server with real DuckDB database.

    This test verifies the MCP server works end-to-end with a real database.
    Run with: pytest test/test_mcp_server.py::test_mcp_server_integration -v
    """
    from findingmodel.finding_model import ChoiceAttributeIded, ChoiceValueIded, FindingModelFull

    # Create a test database with real data
    db_path = tmp_path / "test_mcp.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    try:
        # Create a simple test model
        model = FindingModelFull(
            oifm_id="OIFM_TEST_000001",
            name="Pneumothorax",
            description="Air in the pleural space",
            synonyms=["collapsed lung", "PTX"],
            tags=["chest", "emergency"],
            attributes=[
                ChoiceAttributeIded(
                    oifma_id="OIFMA_TEST_000001",
                    name="Severity",
                    values=[
                        ChoiceValueIded(value_code="OIFMA_TEST_000001.0", name="Small"),
                        ChoiceValueIded(value_code="OIFMA_TEST_000001.1", name="Large"),
                    ],
                )
            ],
        )

        # Write model to file and add to index
        model_file = tmp_path / "pneumothorax.fm.json"
        model_file.write_text(model.model_dump_json(indent=2, exclude_none=True))
        await index.add_or_update_entry_from_file(model_file, model)

        # Close the index so MCP server can open it read-only
        if index.conn is not None:
            index.conn.close()

        # Now test MCP server functions with the real database
        with patch("findingmodel.mcp_server.DuckDBIndex") as mock_duckdb_class:
            # Make the mock return our real index instance
            test_index = DuckDBIndex(db_path, read_only=True)
            await test_index.setup()

            mock_duckdb_class.return_value.__aenter__.return_value = test_index

            # Test search
            search_result = await search_finding_models(query="pneumothorax", limit=5)
            assert search_result.result_count == 1
            assert search_result.results[0].oifm_id == "OIFM_TEST_000001"
            assert search_result.results[0].name == "Pneumothorax"
            assert "chest" in (search_result.results[0].tags or [])

            # Test get by ID
            get_result = await get_finding_model(identifier="OIFM_TEST_000001")
            assert get_result is not None
            assert get_result.name == "Pneumothorax"

            # Test get by synonym
            synonym_result = await get_finding_model(identifier="collapsed lung")
            assert synonym_result is not None
            assert synonym_result.oifm_id == "OIFM_TEST_000001"

            # Test count
            count_result = await count_finding_models()
            assert count_result["finding_models"] == 1
            assert "people" in count_result
            assert "organizations" in count_result

            # Cleanup
            if test_index.conn is not None:
                test_index.conn.close()

    finally:
        # Cleanup
        if index.conn is not None:
            index.conn.close()
