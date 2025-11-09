"""Tests for anatomic location functionality (search, migration, and CLI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import duckdb
import httpx
import pytest
from click.testing import CliRunner
from openai import AsyncOpenAI
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from collections.abc import Generator

from pydantic_ai import models

from findingmodel.anatomic_migration import (
    create_anatomic_database,
    create_searchable_text,
    determine_sided,
    get_database_stats,
    load_anatomic_data,
    validate_anatomic_record,
)
from findingmodel.cli import cli
from findingmodel.config import settings
from findingmodel.tools.anatomic_location_search import (
    AnatomicQueryTerms,
    LocationSearchResponse,
    create_location_selection_agent,
    execute_anatomic_search,
    find_anatomic_locations,
    generate_anatomic_query_terms,
)
from findingmodel.tools.duckdb_search import DuckDBOntologySearchClient
from findingmodel.tools.ontology_search import OntologySearchResult

# Prevent accidental model requests in unit tests
# Tests marked with @pytest.mark.callout can enable this as needed
models.ALLOW_MODEL_REQUESTS = False

# =============================================================================
# Test data constants and fixtures
# =============================================================================

TEST_DATA_FILE = Path(__file__).parent / "data" / "anatomic_locations_test.json"


def _fake_openai_client(*_: Any, **__: Any) -> object:  # pragma: no cover - test helper
    """Return a dummy OpenAI client for patched calls."""
    return object()


def _fake_embedding_anatomic(text: str, **kwargs: Any) -> list[float]:  # pragma: no cover - test helper
    """Deterministic fake embedding based on text hash."""
    target_dims = kwargs.get("dimensions") or settings.openai_embedding_dimensions
    # Use simple hash-based embedding for determinism
    hash_val = sum(ord(c) for c in text)
    return [(hash_val % 100) / 100.0] * target_dims


async def _fake_batch_embeddings(texts: list[str], **kwargs: Any) -> list[list[float] | None]:  # noqa: RUF029
    """Fake batch embeddings."""
    # Note: keeping as async because batch_embeddings_for_duckdb is async in source code
    return [_fake_embedding_anatomic(text, **kwargs) for text in texts]


@pytest.fixture(scope="module")
def _module_anatomic_monkeypatch() -> Generator[None, None, None]:
    """Module-scoped monkeypatch for anatomic migration tests."""
    from findingmodel import anatomic_migration

    # Store originals
    original_batch = anatomic_migration.batch_embeddings_for_duckdb  # type: ignore[attr-defined]

    # Patch with fakes
    anatomic_migration.batch_embeddings_for_duckdb = _fake_batch_embeddings  # type: ignore[attr-defined, assignment]

    yield

    # Restore originals
    anatomic_migration.batch_embeddings_for_duckdb = original_batch  # type: ignore[attr-defined]


# =============================================================================
# Search workflow tests (from test_anatomic_location_search.py)
# =============================================================================


class TestOntologySearchResult:
    """Tests for the OntologySearchResult class."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating an OntologySearchResult with all fields."""
        result = OntologySearchResult(concept_id="RID1301", concept_text="lung", score=0.95, table_name="radlex")

        assert result.concept_id == "RID1301"
        assert result.concept_text == "lung"
        assert result.score == 0.95
        assert result.table_name == "radlex"

    def test_as_index_code_radlex(self) -> None:
        """Test converting a RadLex result to IndexCode."""
        result = OntologySearchResult(concept_id="RID1301", concept_text="lung", score=0.95, table_name="radlex")

        index_code = result.as_index_code()

        assert index_code.system == "RADLEX"
        assert index_code.code == "RID1301"
        assert index_code.display == "lung"

    def test_as_index_code_snomedct(self) -> None:
        """Test converting a SNOMED CT result to IndexCode."""
        result = OntologySearchResult(
            concept_id="39607008", concept_text="Lung structure", score=0.88, table_name="snomedct"
        )

        index_code = result.as_index_code()

        assert index_code.system == "SNOMEDCT"
        assert index_code.code == "39607008"
        assert index_code.display == "Lung structure"

    def test_as_index_code_anatomic_locations(self) -> None:
        """Test converting an anatomic locations result to IndexCode."""
        result = OntologySearchResult(
            concept_id="A123", concept_text="Heart", score=0.99, table_name="anatomic_locations"
        )

        index_code = result.as_index_code()

        assert index_code.system == "ANATOMICLOCATIONS"
        assert index_code.code == "A123"
        assert index_code.display == "Heart"

    def test_as_index_code_unknown_table(self) -> None:
        """Test converting with unknown table falls back to table name."""
        result = OntologySearchResult(concept_id="X999", concept_text="Unknown", score=0.5, table_name="custom_table")

        index_code = result.as_index_code()

        # Should use table name as system when not in mapping
        assert index_code.system == "custom_table"
        assert index_code.code == "X999"
        assert index_code.display == "Unknown"

    def test_as_index_code_with_concept_normalization(self) -> None:
        """Test that concept text is normalized when converting to IndexCode."""
        result = OntologySearchResult(
            concept_id="A999",
            concept_text="Heart chamber\n(additional info)",
            score=0.9,
            table_name="anatomic_locations",
        )

        index_code = result.as_index_code()

        assert index_code.system == "ANATOMICLOCATIONS"
        assert index_code.code == "A999"
        assert index_code.display == "Heart chamber"  # Both newline and parenthetical stripped


class TestAnatomicQueryTerms:
    """Tests for AnatomicQueryTerms model."""

    def test_creation_with_region(self) -> None:
        """Test creating AnatomicQueryTerms with region."""
        query_terms = AnatomicQueryTerms(region="Thorax", terms=["lung", "pulmonary", "respiratory"])

        assert query_terms.region == "Thorax"
        assert query_terms.terms == ["lung", "pulmonary", "respiratory"]

    def test_creation_without_region(self) -> None:
        """Test creating AnatomicQueryTerms without region."""
        query_terms = AnatomicQueryTerms(region=None, terms=["heart"])

        assert query_terms.region is None
        assert query_terms.terms == ["heart"]

    def test_empty_terms(self) -> None:
        """Test creating AnatomicQueryTerms with empty terms."""
        query_terms = AnatomicQueryTerms(region="Abdomen", terms=[])

        assert query_terms.region == "Abdomen"
        assert query_terms.terms == []


class TestLocationSearchResponse:
    """Tests for LocationSearchResponse model."""

    def test_creation(self) -> None:
        """Test creating LocationSearchResponse."""
        primary = OntologySearchResult(concept_id="RID1301", concept_text="lung", score=0.95, table_name="radlex")

        alternates = [
            OntologySearchResult(concept_id="RID1302", concept_text="lung parenchyma", score=0.90, table_name="radlex")
        ]

        response = LocationSearchResponse(
            primary_location=primary,
            alternate_locations=alternates,
            reasoning="Selected lung as the most appropriate location",
        )

        assert response.primary_location == primary
        assert len(response.alternate_locations) == 1
        assert response.reasoning == "Selected lung as the most appropriate location"


class TestGenerateAnatomicQueryTerms:
    """Tests for generate_anatomic_query_terms function."""

    @pytest.mark.asyncio
    async def test_successful_generation(self) -> None:
        """Test successful query term generation."""
        with patch("findingmodel.tools.anatomic_location_search.get_model") as mock_model:
            # Mock the agent
            mock_model.return_value = TestModel()

            # The TestModel will return a default AnatomicQueryTerms
            result = await generate_anatomic_query_terms("pneumonia")

            # Should return AnatomicQueryTerms (even if empty from TestModel)
            assert isinstance(result, AnatomicQueryTerms)

    @pytest.mark.asyncio
    async def test_generation_with_description(self) -> None:
        """Test query term generation with description."""
        with patch("findingmodel.tools.anatomic_location_search.get_model") as mock_model:
            mock_model.return_value = TestModel()

            result = await generate_anatomic_query_terms("pneumonia", "Infection of the lung parenchyma")

            assert isinstance(result, AnatomicQueryTerms)

    @pytest.mark.asyncio
    async def test_generation_fallback_on_error(self) -> None:
        """Test fallback when generation fails."""
        with patch("findingmodel.tools.anatomic_location_search.Agent.run") as mock_run:
            mock_run.side_effect = Exception("API error")

            result = await generate_anatomic_query_terms("test finding")

            # Should fallback to just the finding name
            assert result.region is None
            assert result.terms == ["test finding"]


class TestExecuteAnatomicSearch:
    """Tests for execute_anatomic_search function."""

    @pytest.mark.asyncio
    async def test_search_with_duckdb(self) -> None:
        """Test executing search with DuckDB client."""
        query_info = AnatomicQueryTerms(region="Thorax", terms=["lung", "pulmonary"])

        mock_client = AsyncMock(spec=DuckDBOntologySearchClient)
        mock_client.search_with_filters = AsyncMock(
            return_value=[
                OntologySearchResult(
                    concept_id="RID1301", concept_text="lung", score=0.95, table_name="anatomic_locations"
                )
            ]
        )

        results = await execute_anatomic_search(query_info, mock_client)

        assert len(results) == 1
        assert results[0].concept_id == "RID1301"

        # Verify search_with_filters was called with correct params
        mock_client.search_with_filters.assert_called_once_with(
            queries=["lung", "pulmonary"], region="Thorax", sided_filter=["generic", "nonlateral"], limit_per_query=30
        )

    @pytest.mark.asyncio
    async def test_search_without_region(self) -> None:
        """Test executing search without region filter."""
        query_info = AnatomicQueryTerms(region=None, terms=["heart"])

        mock_client = AsyncMock(spec=DuckDBOntologySearchClient)
        mock_client.search_with_filters = AsyncMock(return_value=[])

        results = await execute_anatomic_search(query_info, mock_client, limit=50)

        # Verify empty results are returned and region=None is passed through
        assert results == []
        mock_client.search_with_filters.assert_called_once_with(
            queries=["heart"], region=None, sided_filter=["generic", "nonlateral"], limit_per_query=50
        )


class TestCreateLocationSelectionAgent:
    """Tests for create_location_selection_agent function."""

    def test_agent_creation(self) -> None:
        """Test creating location selection agent."""
        with patch("findingmodel.tools.anatomic_location_search.get_model") as mock_model:
            mock_model.return_value = TestModel()

            agent = create_location_selection_agent()

            # Should create an agent
            assert agent is not None
            # Output type should be LocationSearchResponse
            assert agent._output_type == LocationSearchResponse

    def test_agent_with_custom_model(self) -> None:
        """Test creating agent with custom model."""
        with patch("findingmodel.tools.anatomic_location_search.get_model") as mock_model:
            mock_model.return_value = TestModel()

            agent = create_location_selection_agent("full")

            # Should pass model tier to get_model and create an agent
            mock_model.assert_called_once_with("full")
            assert agent is not None


class TestFindAnatomicLocations:
    """Tests for the main find_anatomic_locations function."""

    @pytest.mark.asyncio
    async def test_successful_search_with_duckdb(self) -> None:
        """Test successful anatomic location search using DuckDB."""
        # Mock the DuckDB client
        mock_client = AsyncMock(spec=DuckDBOntologySearchClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.search_with_filters = AsyncMock(
            return_value=[
                OntologySearchResult(
                    concept_id="RID1301", concept_text="lung", score=0.95, table_name="anatomic_locations"
                ),
                OntologySearchResult(
                    concept_id="RID1302", concept_text="lung parenchyma", score=0.90, table_name="anatomic_locations"
                ),
            ]
        )

        with (
            patch("findingmodel.tools.anatomic_location_search.DuckDBOntologySearchClient", return_value=mock_client),
            patch("findingmodel.tools.anatomic_location_search.generate_anatomic_query_terms") as mock_generate,
            patch("findingmodel.tools.anatomic_location_search.create_location_selection_agent") as mock_create_agent,
        ):
            # Mock query generation
            mock_generate.return_value = AnatomicQueryTerms(region="Thorax", terms=["lung", "pulmonary"])

            # Mock selection agent
            mock_agent = AsyncMock()
            mock_result = MagicMock()
            mock_result.output = LocationSearchResponse(
                primary_location=OntologySearchResult(
                    concept_id="RID1301", concept_text="lung", score=0.95, table_name="anatomic_locations"
                ),
                alternate_locations=[],
                reasoning="Selected lung as primary location",
            )
            mock_agent.run = AsyncMock(return_value=mock_result)
            mock_create_agent.return_value = mock_agent

            # Execute search
            result = await find_anatomic_locations("pneumonia", use_duckdb=True)

            # Verify result
            assert isinstance(result, LocationSearchResponse)
            assert result.primary_location.concept_id == "RID1301"
            assert result.reasoning == "Selected lung as primary location"

    @pytest.mark.asyncio
    async def test_empty_search_results(self) -> None:
        """Test handling of empty search results."""
        mock_client = AsyncMock(spec=DuckDBOntologySearchClient)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.search_with_filters = AsyncMock(return_value=[])

        with (
            patch("findingmodel.tools.anatomic_location_search.DuckDBOntologySearchClient", return_value=mock_client),
            patch("findingmodel.tools.anatomic_location_search.generate_anatomic_query_terms") as mock_generate,
        ):
            mock_generate.return_value = AnatomicQueryTerms(region=None, terms=["unknown"])

            result = await find_anatomic_locations("unknown finding")

            # Should return default response
            assert result.primary_location.concept_id == "NO_RESULTS"
            assert result.primary_location.concept_text == "unspecified anatomic location"
            assert len(result.alternate_locations) == 0


# =============================================================================
# Migration function tests (from test_anatomic_migration.py)
# =============================================================================


class TestCreateSearchableText:
    """Tests for create_searchable_text function."""

    def test_with_all_fields(self) -> None:
        """Test creating searchable text with all fields present."""
        record = {
            "description": "lung parenchyma",
            "synonyms": ["pulmonary tissue", "lung tissue"],
            "definition": "The functional tissue of the lung consisting of alveoli.",
        }

        result = create_searchable_text(record)

        assert "lung parenchyma" in result
        assert "pulmonary tissue" in result
        assert "lung tissue" in result
        assert "functional tissue" in result
        assert " | " in result  # Check delimiter is present

    def test_with_missing_synonyms(self) -> None:
        """Test with missing synonyms field."""
        record = {"description": "heart", "definition": "The cardiovascular organ."}

        result = create_searchable_text(record)

        assert "heart" in result
        assert "cardiovascular organ" in result
        assert "also known as" not in result

    def test_with_missing_definition(self) -> None:
        """Test with missing definition field."""
        record = {"description": "kidney", "synonyms": ["renal organ"]}

        result = create_searchable_text(record)

        assert "kidney" in result
        assert "renal organ" in result

    def test_with_empty_synonyms_list(self) -> None:
        """Test with empty synonyms list."""
        record = {"description": "liver", "synonyms": [], "definition": "Hepatic organ."}

        result = create_searchable_text(record)

        assert "liver" in result
        assert "also known as" not in result
        assert "Hepatic organ" in result

    def test_with_null_synonyms(self) -> None:
        """Test with null synonyms value."""
        record = {"description": "spleen", "synonyms": None, "definition": "Lymphoid organ."}

        result = create_searchable_text(record)

        assert "spleen" in result
        assert "also known as" not in result

    def test_with_many_synonyms(self) -> None:
        """Test that only first 5 synonyms are included."""
        record = {
            "description": "brain",
            "synonyms": ["cerebrum", "encephalon", "brain tissue", "neural tissue", "cerebral matter", "extra synonym"],
        }

        result = create_searchable_text(record)

        assert "brain" in result
        assert "cerebrum" in result
        assert "cerebral matter" in result
        # Should not include 6th synonym
        assert "extra synonym" not in result

    def test_with_long_definition(self) -> None:
        """Test that long definitions are truncated."""
        long_definition = "A" * 300  # 300 characters
        record = {"description": "test organ", "definition": long_definition}

        result = create_searchable_text(record)

        assert len(result) < len(record["description"]) + 205  # 200 + "..." + delimiter overhead
        assert "..." in result

    def test_with_short_definition(self) -> None:
        """Test that short definitions are not truncated."""
        short_definition = "This is a short definition."
        record = {"description": "test organ", "definition": short_definition}

        result = create_searchable_text(record)

        assert "..." not in result
        assert short_definition in result

    def test_with_only_description(self) -> None:
        """Test with only description field."""
        record = {"description": "simple organ"}

        result = create_searchable_text(record)

        assert result == "simple organ"


class TestDetermineSided:
    """Tests for determine_sided function."""

    def test_generic_with_both_left_and_right(self) -> None:
        """Test record with both leftRef and rightRef returns 'generic'."""
        record = {"leftRef": {"id": "RID123_L"}, "rightRef": {"id": "RID123_R"}}

        result = determine_sided(record)

        assert result == "generic"

    def test_left_only(self) -> None:
        """Test record with only leftRef returns 'left'."""
        record = {"leftRef": {"id": "RID123_L"}}

        result = determine_sided(record)

        assert result == "left"

    def test_right_only(self) -> None:
        """Test record with only rightRef returns 'right'."""
        record = {"rightRef": {"id": "RID123_R"}}

        result = determine_sided(record)

        assert result == "right"

    def test_unsided(self) -> None:
        """Test record with only unsidedRef returns 'unsided'."""
        record = {"unsidedRef": {"id": "RID123"}}

        result = determine_sided(record)

        assert result == "unsided"

    def test_nonlateral_no_refs(self) -> None:
        """Test record with no laterality refs returns 'nonlateral'."""
        record = {"description": "midline structure"}

        result = determine_sided(record)

        assert result == "nonlateral"

    def test_nonlateral_with_unrelated_refs(self) -> None:
        """Test record with only partOfRef returns 'nonlateral'."""
        record = {"partOfRef": {"id": "RID456"}}

        result = determine_sided(record)

        assert result == "nonlateral"

    def test_generic_with_extra_refs(self) -> None:
        """Test generic determination ignores other refs."""
        record = {
            "leftRef": {"id": "RID123_L"},
            "rightRef": {"id": "RID123_R"},
            "unsidedRef": {"id": "RID123"},
            "partOfRef": {"id": "RID456"},
        }

        result = determine_sided(record)

        assert result == "generic"

    def test_left_with_unsided(self) -> None:
        """Test that leftRef takes precedence over unsidedRef."""
        record = {"leftRef": {"id": "RID123_L"}, "unsidedRef": {"id": "RID123"}}

        result = determine_sided(record)

        assert result == "left"


class TestLoadAnatomicData:
    """Tests for load_anatomic_data function."""

    @pytest.mark.asyncio
    async def test_load_from_local_file(self, tmp_path: Path) -> None:
        """Test loading data from local file path."""
        test_data = [
            {"_id": "RID123", "description": "test location 1"},
            {"_id": "RID124", "description": "test location 2"},
        ]

        test_file = tmp_path / "test_data.json"
        test_file.write_text(json.dumps(test_data))

        result = await load_anatomic_data(str(test_file))

        assert len(result) == 2
        assert result[0]["_id"] == "RID123"
        assert result[1]["description"] == "test location 2"

    @pytest.mark.asyncio
    async def test_load_from_url(self) -> None:
        """Test loading data from URL."""
        test_data = [{"_id": "RID999", "description": "remote location"}]

        # Create a proper mock response
        mock_response = AsyncMock()
        mock_response.json = lambda: test_data  # Use lambda to return data directly
        mock_response.raise_for_status = lambda: None

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            result = await load_anatomic_data("https://example.com/data.json")

            assert len(result) == 1
            assert result[0]["_id"] == "RID999"

    @pytest.mark.asyncio
    async def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file doesn't exist."""
        nonexistent_file = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError, match="File not found"):
            await load_anatomic_data(str(nonexistent_file))

    @pytest.mark.asyncio
    async def test_invalid_json(self, tmp_path: Path) -> None:
        """Test error when JSON is invalid."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            await load_anatomic_data(str(test_file))

    @pytest.mark.asyncio
    async def test_not_a_list(self, tmp_path: Path) -> None:
        """Test error when JSON is not a list."""
        test_file = tmp_path / "not_list.json"
        test_file.write_text(json.dumps({"key": "value"}))

        with pytest.raises(ValueError, match="Expected list of records"):
            await load_anatomic_data(str(test_file))

    @pytest.mark.asyncio
    async def test_network_error(self) -> None:
        """Test handling of network errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection failed")
            )

            with pytest.raises(httpx.ConnectError):
                await load_anatomic_data("https://example.com/data.json")

    @pytest.mark.asyncio
    async def test_http_error(self) -> None:
        """Test handling of HTTP errors."""

        def raise_http_error() -> None:
            # Create mock request and response for HTTPStatusError
            mock_request = httpx.Request("GET", "https://example.com/data.json")
            mock_response_obj = httpx.Response(404, request=mock_request)
            raise httpx.HTTPStatusError("404", request=mock_request, response=mock_response_obj)

        mock_response = AsyncMock()
        mock_response.raise_for_status = raise_http_error

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

            with pytest.raises(httpx.HTTPStatusError):
                await load_anatomic_data("https://example.com/data.json")


class TestValidateAnatomicRecord:
    """Tests for validate_anatomic_record function."""

    def test_valid_record_with_all_fields(self) -> None:
        """Test validation of a complete valid record."""
        record = {
            "_id": "RID123",
            "description": "test location",
            "region": "Thorax",
            "synonyms": ["alternative name"],
            "definition": "A test anatomic location.",
        }

        errors = validate_anatomic_record(record)

        assert len(errors) == 0

    def test_valid_record_minimal(self) -> None:
        """Test validation of minimal valid record."""
        record = {"_id": "RID123", "description": "test location"}

        errors = validate_anatomic_record(record)

        assert len(errors) == 0

    def test_missing_id(self) -> None:
        """Test error when _id is missing."""
        record = {"description": "test location"}

        errors = validate_anatomic_record(record)

        assert len(errors) == 1
        assert "Missing required field: _id" in errors

    def test_missing_description(self) -> None:
        """Test error when description is missing."""
        record = {"_id": "RID123"}

        errors = validate_anatomic_record(record)

        assert len(errors) == 1
        assert "Missing required field: description" in errors

    def test_multiple_missing_fields(self) -> None:
        """Test multiple validation errors."""
        record: dict[str, Any] = {}

        errors = validate_anatomic_record(record)

        assert len(errors) == 2
        assert any("_id" in err for err in errors)
        assert any("description" in err for err in errors)

    def test_invalid_synonyms_type(self) -> None:
        """Test error when synonyms is not a list."""
        record = {"_id": "RID123", "description": "test", "synonyms": "not a list"}

        errors = validate_anatomic_record(record)

        assert len(errors) == 1
        assert "synonyms" in errors[0]
        assert "must be a list" in errors[0]

    def test_empty_synonyms_list_valid(self) -> None:
        """Test that empty synonyms list is valid."""
        record = {"_id": "RID123", "description": "test", "synonyms": []}

        errors = validate_anatomic_record(record)

        assert len(errors) == 0

    def test_empty_id(self) -> None:
        """Test error when _id is empty string."""
        record = {"_id": "", "description": "test"}

        errors = validate_anatomic_record(record)

        assert len(errors) == 1
        assert "_id" in errors[0]

    def test_empty_description(self) -> None:
        """Test error when description is empty string."""
        record = {"_id": "RID123", "description": ""}

        errors = validate_anatomic_record(record)

        assert len(errors) == 1
        assert "description" in errors[0]


class TestCreateAnatomicDatabase:
    """Tests for create_anatomic_database function."""

    @pytest.mark.asyncio
    async def test_successful_database_creation(self, tmp_path: Path) -> None:
        """Test successful creation of anatomic database."""
        db_path = tmp_path / "test_anatomic.duckdb"

        test_records = [
            {
                "_id": "RID123",
                "description": "lung",
                "region": "Thorax",
                "synonyms": ["pulmonary organ"],
                "definition": "The respiratory organ.",
                "leftRef": {"id": "RID123_L"},
                "rightRef": {"id": "RID123_R"},
            }
        ]

        # Mock OpenAI client
        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_embedding = [0.1] * settings.openai_embedding_dimensions

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", return_value=[mock_embedding]):
            successful, failed = await create_anatomic_database(db_path, test_records, mock_client)

        assert successful == 1
        assert failed == 0
        assert db_path.exists()

        # Verify database structure
        conn = duckdb.connect(str(db_path), read_only=True)
        result_tuple = conn.execute("SELECT * FROM anatomic_locations WHERE id = 'RID123'").fetchone()
        assert result_tuple is not None
        assert result_tuple[1] == "lung"  # description
        assert result_tuple[2] == "Thorax"  # region
        assert result_tuple[3] == "generic"  # sided
        conn.close()

    @pytest.mark.asyncio
    async def test_database_with_multiple_records(self, tmp_path: Path) -> None:
        """Test creating database with multiple records."""
        db_path = tmp_path / "test_multi.duckdb"

        test_records = [{"_id": f"RID{i}", "description": f"location {i}", "region": "Test"} for i in range(5)]

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_embeddings = [[0.1] * settings.openai_embedding_dimensions] * 5

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", return_value=mock_embeddings):
            successful, failed = await create_anatomic_database(db_path, test_records, mock_client)

        assert successful == 5
        assert failed == 0

        # Verify count
        conn = duckdb.connect(str(db_path), read_only=True)
        count_result = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()
        assert count_result is not None
        assert count_result[0] == 5
        conn.close()

    @pytest.mark.asyncio
    async def test_database_with_validation_errors(self, tmp_path: Path) -> None:
        """Test handling of invalid records during database creation."""
        db_path = tmp_path / "test_invalid.duckdb"

        test_records = [
            {"_id": "RID123", "description": "valid record"},
            {"_id": "RID124"},  # Missing description
            {"description": "missing id"},  # Missing _id
        ]

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_embeddings = [[0.1] * settings.openai_embedding_dimensions]

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", return_value=mock_embeddings):
            successful, failed = await create_anatomic_database(db_path, test_records, mock_client)

        assert successful == 1
        assert failed == 2

    @pytest.mark.asyncio
    async def test_database_indexes_created(self, tmp_path: Path) -> None:
        """Test that indexes are created properly."""
        db_path = tmp_path / "test_indexes.duckdb"

        test_records = [{"_id": "RID123", "description": "test location", "region": "Test"}]

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_embeddings = [[0.1] * settings.openai_embedding_dimensions]

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", return_value=mock_embeddings):
            await create_anatomic_database(db_path, test_records, mock_client)

        # Verify indexes exist
        conn = duckdb.connect(str(db_path), read_only=True)

        # Check for standard indexes
        indexes = conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()
        index_names = [idx[0] for idx in indexes]

        assert any("region" in name for name in index_names)
        assert any("sided" in name for name in index_names)
        assert any("description" in name for name in index_names)

        conn.close()

    @pytest.mark.asyncio
    async def test_database_with_batch_processing(self, tmp_path: Path) -> None:
        """Test batch processing with batch_size parameter."""
        db_path = tmp_path / "test_batch.duckdb"

        # Create 75 records to test multiple batches (batch_size=50)
        test_records = [{"_id": f"RID{i:04d}", "description": f"location {i}"} for i in range(75)]

        mock_client = AsyncMock(spec=AsyncOpenAI)

        # Mock will be called twice: once for 50 records, once for 25
        mock_embeddings_batch1: list[list[float]] = [[0.1] * settings.openai_embedding_dimensions] * 50
        mock_embeddings_batch2: list[list[float]] = [[0.1] * settings.openai_embedding_dimensions] * 25

        call_count = 0

        async def mock_batch_embeddings(texts: list[str], **kwargs: Any) -> list[list[float]]:  # noqa: RUF029
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_embeddings_batch1
            else:
                return mock_embeddings_batch2

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", side_effect=mock_batch_embeddings):
            successful, failed = await create_anatomic_database(db_path, test_records, mock_client, batch_size=50)

        assert successful == 75
        assert failed == 0
        assert call_count == 2


class TestGetDatabaseStats:
    """Tests for get_database_stats function."""

    @pytest.mark.asyncio
    async def test_stats_for_valid_database(self, tmp_path: Path) -> None:
        """Test getting stats from a valid database."""
        db_path = tmp_path / "test_stats.duckdb"

        test_records: list[dict[str, Any]] = [
            {"_id": "RID001", "description": "location 1", "region": "Region A", "leftRef": {"id": "L"}},
            {"_id": "RID002", "description": "location 2", "region": "Region A", "rightRef": {"id": "R"}},
            {"_id": "RID003", "description": "location 3", "region": "Region B"},
        ]

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_embeddings = [[0.1] * settings.openai_embedding_dimensions] * 3

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", return_value=mock_embeddings):
            await create_anatomic_database(db_path, test_records, mock_client)

        # Get stats
        stats = get_database_stats(db_path)

        assert stats["total_records"] == 3
        assert stats["records_with_vectors"] == 3
        assert stats["unique_regions"] == 2
        assert "sided_distribution" in stats
        assert stats["file_size_mb"] > 0

    def test_stats_database_not_found(self, tmp_path: Path) -> None:
        """Test error when database file doesn't exist."""
        nonexistent_db = tmp_path / "nonexistent.duckdb"

        with pytest.raises(FileNotFoundError, match="Database not found"):
            get_database_stats(nonexistent_db)

    @pytest.mark.asyncio
    async def test_stats_empty_database(self, tmp_path: Path) -> None:
        """Test stats for database with no records."""
        db_path = tmp_path / "empty_db.duckdb"

        mock_client = AsyncMock(spec=AsyncOpenAI)

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", return_value=[]):
            await create_anatomic_database(db_path, [], mock_client)

        stats = get_database_stats(db_path)

        assert stats["total_records"] == 0
        assert stats["records_with_vectors"] == 0
        assert stats["unique_regions"] == 0

    @pytest.mark.asyncio
    async def test_stats_sided_distribution(self, tmp_path: Path) -> None:
        """Test sided distribution in stats."""
        db_path = tmp_path / "sided_test.duckdb"

        test_records: list[dict[str, Any]] = [
            {"_id": "RID001", "description": "generic", "leftRef": {"id": "L"}, "rightRef": {"id": "R"}},
            {"_id": "RID002", "description": "left only", "leftRef": {"id": "L"}},
            {"_id": "RID003", "description": "right only", "rightRef": {"id": "R"}},
            {"_id": "RID004", "description": "unsided", "unsidedRef": {"id": "U"}},
            {"_id": "RID005", "description": "nonlateral"},
        ]

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_embeddings = [[0.1] * settings.openai_embedding_dimensions] * 5

        with patch("findingmodel.anatomic_migration.batch_embeddings_for_duckdb", return_value=mock_embeddings):
            await create_anatomic_database(db_path, test_records, mock_client)

        stats = get_database_stats(db_path)

        assert stats["sided_distribution"]["generic"] == 1
        assert stats["sided_distribution"]["left"] == 1
        assert stats["sided_distribution"]["right"] == 1
        assert stats["sided_distribution"]["unsided"] == 1
        assert stats["sided_distribution"]["nonlateral"] == 1


# =============================================================================
# CLI tests - MINIMAL (reduced from test_cli_anatomic.py)
# =============================================================================


class TestAnatomicCLI:
    """Tests for anatomic CLI commands (build, validate, stats)."""

    def test_build_basic(self, tmp_path: Path, _module_anatomic_monkeypatch: None) -> None:
        """Happy path: build from local file."""
        runner = CliRunner()
        db_path = tmp_path / "test_anatomic.duckdb"

        with patch("findingmodel.cli.AsyncOpenAI", return_value=_fake_openai_client()):
            result = runner.invoke(
                cli, ["anatomic", "build", "--source", str(TEST_DATA_FILE), "--output", str(db_path)]
            )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert db_path.exists()
        assert "Database built successfully" in result.output
        assert "Records inserted:" in result.output

    def test_build_error_no_api_key(self, tmp_path: Path, _module_anatomic_monkeypatch: None) -> None:
        """Error case: missing API key."""
        runner = CliRunner()
        db_path = tmp_path / "no_api_key.duckdb"

        with patch.object(settings, "openai_api_key") as mock_key:
            mock_key.get_secret_value.return_value = ""

            result = runner.invoke(
                cli, ["anatomic", "build", "--source", str(TEST_DATA_FILE), "--output", str(db_path)]
            )

        assert result.exit_code != 0
        assert "OPENAI_API_KEY not configured" in result.output

    def test_validate_basic(self, _module_anatomic_monkeypatch: None) -> None:
        """Happy path: validate valid data."""
        runner = CliRunner()

        result = runner.invoke(cli, ["anatomic", "validate", "--source", str(TEST_DATA_FILE)])

        assert result.exit_code == 0
        assert "validated successfully" in result.output

    def test_validate_error_invalid_data(self, tmp_path: Path, _module_anatomic_monkeypatch: None) -> None:
        """Error case: validation errors."""
        test_file = tmp_path / "invalid.json"
        test_data = [
            {"_id": "RID001", "description": "valid"},
            {"description": "missing id"},  # Invalid
        ]
        test_file.write_text(json.dumps(test_data))

        runner = CliRunner()
        result = runner.invoke(cli, ["anatomic", "validate", "--source", str(test_file)])

        assert result.exit_code == 1
        assert "Validation failed" in result.output
        assert "_id" in result.output

    def test_stats_basic(self, tmp_path: Path, _module_anatomic_monkeypatch: None) -> None:
        """Happy path: show stats."""
        runner = CliRunner()
        db_path = tmp_path / "stats_test.duckdb"

        # Build database first
        with patch("findingmodel.cli.AsyncOpenAI", return_value=_fake_openai_client()):
            build_result = runner.invoke(
                cli, ["anatomic", "build", "--source", str(TEST_DATA_FILE), "--output", str(db_path)]
            )
            assert build_result.exit_code == 0

        # Get stats
        result = runner.invoke(cli, ["anatomic", "stats", "--db-path", str(db_path)])

        assert result.exit_code == 0
        assert "Anatomic Location Database Statistics" in result.output
        assert "Database Summary" in result.output

    def test_stats_error_no_database(self, tmp_path: Path, _module_anatomic_monkeypatch: None) -> None:
        """Error case: database not found (custom path with no remote URL)."""
        runner = CliRunner()
        nonexistent_db = tmp_path / "does_not_exist.duckdb"

        result = runner.invoke(cli, ["anatomic", "stats", "--db-path", str(nonexistent_db)])

        # get_database_stats raises FileNotFoundError for nonexistent database
        assert result.exit_code != 0
        assert "Database not found" in result.output or "FileNotFoundError" in result.output


# =============================================================================
# Integration test - sanity check only
# =============================================================================


@pytest.mark.callout
@pytest.mark.asyncio
async def test_find_anatomic_locations_basic_wiring() -> None:
    """Sanity check: Verify basic wiring with real API and DuckDB.

    All comprehensive behavioral testing is in evals/anatomic_search.py.
    This test only verifies the function can be called successfully with
    real OpenAI API and DuckDB backend.
    """
    # Skip if OpenAI API key not configured
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
        pytest.skip("OPENAI_API_KEY not configured")

    # Skip if DuckDB anatomic database not available
    # Try to ensure database exists - it will raise FileNotFoundError if unavailable
    try:
        from findingmodel.config import ensure_anatomic_db

        ensure_anatomic_db()
    except (FileNotFoundError, Exception):
        pytest.skip("DuckDB anatomic locations database not available")

    # Save original state
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Single API call with simple input - use fast model for integration test
        result = await find_anatomic_locations(
            finding_name="pneumonia",
            description="Infection of the lung parenchyma",
            use_duckdb=True,
            model_tier="small",
        )

        # Assert only on structure, not behavior
        assert isinstance(result, LocationSearchResponse)
        assert hasattr(result, "primary_location")
        assert hasattr(result, "alternate_locations")
        assert hasattr(result, "reasoning")
        assert isinstance(result.primary_location, OntologySearchResult)
        assert isinstance(result.alternate_locations, list)
        assert isinstance(result.reasoning, str)

        # NO behavioral assertions - those are in evals

    finally:
        # Restore original state
        models.ALLOW_MODEL_REQUESTS = original
