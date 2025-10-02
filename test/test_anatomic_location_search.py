"""Tests for anatomic location search functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from findingmodel.tools.anatomic_location_search import (
    AnatomicQueryTerms,
    LocationSearchResponse,
    create_location_selection_agent,
    execute_anatomic_search,
    find_anatomic_locations,
    generate_anatomic_query_terms,
)
from findingmodel.tools.duckdb_search import DuckDBOntologySearchClient
from findingmodel.tools.ontology_search import (
    OntologySearchResult,
)


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
        with patch("findingmodel.tools.anatomic_location_search.get_openai_model") as mock_model:
            # Mock the agent
            mock_model.return_value = TestModel()

            # The TestModel will return a default AnatomicQueryTerms
            result = await generate_anatomic_query_terms("pneumonia")

            # Should return AnatomicQueryTerms (even if empty from TestModel)
            assert isinstance(result, AnatomicQueryTerms)

    @pytest.mark.asyncio
    async def test_generation_with_description(self) -> None:
        """Test query term generation with description."""
        with patch("findingmodel.tools.anatomic_location_search.get_openai_model") as mock_model:
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
        with patch("findingmodel.tools.anatomic_location_search.get_openai_model") as mock_model:
            mock_model.return_value = TestModel()

            agent = create_location_selection_agent()

            # Should create an agent
            assert agent is not None
            # Output type should be LocationSearchResponse
            assert agent._output_type == LocationSearchResponse

    def test_agent_with_custom_model(self) -> None:
        """Test creating agent with custom model."""
        with patch("findingmodel.tools.anatomic_location_search.get_openai_model") as mock_model:
            mock_model.return_value = TestModel()

            agent = create_location_selection_agent("gpt-4")

            # Should pass model to get_openai_model and create an agent
            mock_model.assert_called_once_with("gpt-4")
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
