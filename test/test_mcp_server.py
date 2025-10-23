"""Tests for the MCP server.

Note: These tests use mocking to avoid requiring a populated database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from findingmodel.index import AttributeInfo, IndexEntry
from findingmodel.mcp_server import (
    count_finding_models,
    get_finding_model,
    list_finding_model_tags,
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
async def test_list_finding_model_tags() -> None:
    """Test listing all tags."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        ("abdominal",),
        ("cardiac",),
        ("chest",),
    ]

    mock_index = AsyncMock()
    mock_index._ensure_connection = MagicMock(return_value=mock_conn)
    mock_index.__aenter__ = AsyncMock(return_value=mock_index)
    mock_index.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.mcp_server.DuckDBIndex", return_value=mock_index):
        result = await list_finding_model_tags()

        assert result == ["abdominal", "cardiac", "chest"]


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
