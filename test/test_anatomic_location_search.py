"""Unit tests for anatomic location search functionality and supporting ontology search components."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from findingmodel.index_code import IndexCode
from findingmodel.tools.anatomic_location_search import (
    LocationSearchResponse,
    RawSearchResults,
    SearchContext,
    create_matching_agent,
    create_search_agent,
    find_anatomic_locations,
    ontology_search_tool,
)
from findingmodel.tools.ontology_search import (
    ONTOLOGY_TABLES,
    TABLE_TO_INDEX_CODE_SYSTEM,
    OntologySearchClient,
    OntologySearchResult,
)

# Prevent accidental API calls during testing
models.ALLOW_MODEL_REQUESTS = False

# ===== ONTOLOGY SEARCH TESTS =====
# Tests for the OntologySearchResult and OntologySearchClient classes
# which are supporting components for the anatomic location search functionality.


class TestOntologySearchResult:
    """Tests for OntologySearchResult class."""

    def test_as_index_code_known_table_radlex(self) -> None:
        """Test as_index_code method with known table name: radlex."""
        result = OntologySearchResult(
            concept_id="RID12345", concept_text="Test Concept", score=0.95, table_name="radlex"
        )

        index_code = result.as_index_code()

        assert isinstance(index_code, IndexCode)
        assert index_code.system == "RADLEX"
        assert index_code.code == "RID12345"
        assert index_code.display == "Test Concept"

    def test_as_index_code_known_table_anatomic_locations(self) -> None:
        """Test as_index_code method with known table name: anatomic_locations."""
        result = OntologySearchResult(
            concept_id="A12345", concept_text="Heart", score=0.88, table_name="anatomic_locations"
        )

        index_code = result.as_index_code()

        assert isinstance(index_code, IndexCode)
        assert index_code.system == "ANATOMICLOCATIONS"
        assert index_code.code == "A12345"
        assert index_code.display == "Heart"

    def test_as_index_code_known_table_snomedct(self) -> None:
        """Test as_index_code method with known table name: snomedct."""
        result = OntologySearchResult(
            concept_id="123456789", concept_text="Myocardial Infarction", score=0.92, table_name="snomedct"
        )

        index_code = result.as_index_code()

        assert isinstance(index_code, IndexCode)
        assert index_code.system == "SNOMEDCT"
        assert index_code.code == "123456789"
        assert index_code.display == "Myocardial Infarction"

    def test_as_index_code_unknown_table_fallback(self) -> None:
        """Test as_index_code method with unknown table name uses fallback."""
        result = OntologySearchResult(
            concept_id="CUSTOM123", concept_text="Custom Concept", score=0.75, table_name="custom_table"
        )

        index_code = result.as_index_code()

        assert isinstance(index_code, IndexCode)
        assert index_code.system == "custom_table"  # Fallback to original table name
        assert index_code.code == "CUSTOM123"
        assert index_code.display == "Custom Concept"

    def test_as_index_code_strips_newline_content(self) -> None:
        """Test as_index_code strips content after newline."""
        result = OntologySearchResult(
            concept_id="RID789",
            concept_text="Posterior cruciate ligament\nKnee joint structure\nAdditional info",
            score=0.90,
            table_name="radlex"
        )

        index_code = result.as_index_code()

        assert index_code.system == "RADLEX"
        assert index_code.code == "RID789"
        assert index_code.display == "Posterior cruciate ligament"  # Only first line

    def test_as_index_code_strips_parenthetical_content(self) -> None:
        """Test as_index_code strips parenthetical content at end."""
        result = OntologySearchResult(
            concept_id="123456",
            concept_text="Knee joint (body structure)",
            score=0.85,
            table_name="snomedct"
        )

        index_code = result.as_index_code()

        assert index_code.system == "SNOMEDCT"
        assert index_code.code == "123456"
        assert index_code.display == "Knee joint"  # Parenthetical removed

    def test_as_index_code_strips_both_newline_and_parenthetical(self) -> None:
        """Test as_index_code handles both newline and parenthetical content."""
        result = OntologySearchResult(
            concept_id="A999",
            concept_text="Heart chamber (anatomical structure)\nCardiac anatomy\nMore details",
            score=0.88,
            table_name="anatomic_locations"
        )

        index_code = result.as_index_code()

        assert index_code.system == "ANATOMICLOCATIONS"
        assert index_code.code == "A999"
        assert index_code.display == "Heart chamber"  # Both newline and parenthetical stripped


class TestOntologySearchClient:
    """Tests for OntologySearchClient class."""

    def test_initialization_with_defaults(self) -> None:
        """Test basic initialization with default parameters."""
        client = OntologySearchClient()

        assert client._db_conn is None
        assert client._tables == {}
        assert not client.connected

    def test_initialization_with_custom_uri(self) -> None:
        """Test initialization with custom URI."""
        custom_uri = "lancedb://custom-uri"
        client = OntologySearchClient(lancedb_uri=custom_uri)

        assert client._uri == custom_uri

    def test_initialization_with_custom_api_key(self) -> None:
        """Test initialization with custom API key."""
        custom_key = "custom-api-key-123"
        client = OntologySearchClient(api_key=custom_key)

        assert client._api_key == custom_key

    def test_connected_property_initially_false(self) -> None:
        """Test that connected property returns False initially."""
        client = OntologySearchClient()

        assert client.connected is False

    def test_uri_fallback_behavior(self) -> None:
        """Test URI fallback behavior when None is provided."""
        client = OntologySearchClient(lancedb_uri=None)

        # Should use settings.lancedb_uri when None is provided
        # We can't test the exact value without importing settings
        # but we can verify the attribute exists
        assert hasattr(client, "_uri")

    def test_api_key_fallback_behavior(self) -> None:
        """Test API key fallback behavior when None is provided."""
        client = OntologySearchClient(api_key=None)

        # Should use settings.lancedb_api_key when None is provided
        # We can't test the exact value without importing settings
        # but we can verify the attribute exists
        assert hasattr(client, "_api_key")


class TestOntologyConstants:
    """Tests for ontology module constants."""

    def test_ontology_tables_contents(self) -> None:
        """Test ONTOLOGY_TABLES contains exactly the expected tables."""
        expected_tables = ["anatomic_locations", "radlex", "snomedct"]

        assert expected_tables == ONTOLOGY_TABLES
        assert len(ONTOLOGY_TABLES) == 3

    def test_table_to_index_code_system_mappings(self) -> None:
        """Test TABLE_TO_INDEX_CODE_SYSTEM has correct mappings."""
        expected_mappings = {"anatomic_locations": "ANATOMICLOCATIONS", "radlex": "RADLEX", "snomedct": "SNOMEDCT"}

        assert expected_mappings == TABLE_TO_INDEX_CODE_SYSTEM
        assert len(TABLE_TO_INDEX_CODE_SYSTEM) == 3

    def test_constants_consistency(self) -> None:
        """Test that both constants are consistent with each other."""
        # Every table in ONTOLOGY_TABLES should have a mapping in TABLE_TO_INDEX_CODE_SYSTEM
        for table in ONTOLOGY_TABLES:
            assert table in TABLE_TO_INDEX_CODE_SYSTEM

        # Every key in TABLE_TO_INDEX_CODE_SYSTEM should be in ONTOLOGY_TABLES
        for table in TABLE_TO_INDEX_CODE_SYSTEM:
            assert table in ONTOLOGY_TABLES


# ===== ANATOMIC LOCATION SEARCH TESTS =====
# Tests for the main anatomic location search functionality.


class TestModels:
    """Test business logic validation for our custom models."""

    def test_location_search_response_max_alternates(self) -> None:
        """Test that LocationSearchResponse enforces max 3 alternates - business rule validation."""
        primary = OntologySearchResult(
            concept_id="A123", concept_text="Heart", score=0.95, table_name="anatomic_locations"
        )
        # Try to create 4 alternates (should be limited to 3) - this tests our business logic
        too_many_alternates = [
            OntologySearchResult(
                concept_id=f"A{i}", concept_text=f"Location{i}", score=0.8, table_name="anatomic_locations"
            )
            for i in range(4)
        ]

        with pytest.raises(ValueError):
            LocationSearchResponse(
                primary_location=primary,
                alternate_locations=too_many_alternates,
                reasoning="Test reasoning",
            )


class TestOntologySearchTool:
    """Test the ontology search tool function."""

    @pytest.mark.asyncio
    async def test_ontology_search_tool_success(self) -> None:
        """Test ontology_search_tool with successful search."""
        # Create mock search results
        mock_results = [
            OntologySearchResult(concept_id="A123", concept_text="Heart", score=0.95, table_name="anatomic_locations"),
            OntologySearchResult(
                concept_id="A456", concept_text="Cardiac muscle", score=0.88, table_name="anatomic_locations"
            ),
        ]

        # Create mock search client
        mock_client = AsyncMock()
        mock_client.search_tables.return_value = {"anatomic_locations": mock_results}

        # Create mock context
        mock_context = MagicMock()
        mock_context.deps = SearchContext(search_client=mock_client)

        # Test the tool
        result = await ontology_search_tool(mock_context, "heart", limit=10)

        # Verify the call was made correctly
        mock_client.search_tables.assert_called_once_with("heart", tables=["anatomic_locations"], limit_per_table=10)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "anatomic_locations" in result
        assert len(result["anatomic_locations"]) == 2

        # Verify the results are serialized properly
        heart_result = result["anatomic_locations"][0]
        assert heart_result["concept_id"] == "A123"
        assert heart_result["concept_text"] == "Heart"
        assert heart_result["score"] == 0.95

    @pytest.mark.asyncio
    async def test_ontology_search_tool_error_handling(self) -> None:
        """Test ontology_search_tool error handling."""
        # Create mock search client that raises exception
        mock_client = AsyncMock()
        mock_client.search_tables.side_effect = Exception("Database connection failed")

        # Create mock context
        mock_context = MagicMock()
        mock_context.deps = SearchContext(search_client=mock_client)

        # Test the tool
        result = await ontology_search_tool(mock_context, "heart", limit=10)

        # Should return error dict instead of raising
        assert isinstance(result, dict)
        assert "error" in result
        assert "Search failed: Database connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_ontology_search_tool_empty_results(self) -> None:
        """Test ontology_search_tool with empty results."""
        # Create mock search client with empty results
        mock_client = AsyncMock()
        mock_client.search_tables.return_value = {"anatomic_locations": []}

        # Create mock context
        mock_context = MagicMock()
        mock_context.deps = SearchContext(search_client=mock_client)

        # Test the tool
        result = await ontology_search_tool(mock_context, "nonexistent", limit=10)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "anatomic_locations" in result
        assert len(result["anatomic_locations"]) == 0


class TestAgentConfiguration:
    """Test agent configuration and setup - focusing on our code's behavior."""

    def test_search_agent_configuration(self) -> None:
        """Test that search agent is configured correctly with tools and system prompt."""
        agent = create_search_agent("gpt-4")

        # Test agent configuration - check function toolset has tools
        tools = agent._function_toolset.tools if agent._function_toolset else []
        assert len(tools) > 0  # Should have the ontology search tool
        assert "ontology_search_tool" in tools

        # Verify system prompt contains key medical terminology guidance
        system_prompt = agent._system_prompts[0].lower() if agent._system_prompts else ""
        assert "medical terminology" in system_prompt or "medical" in system_prompt
        assert "search" in system_prompt
        assert "anatomic" in system_prompt or "anatomical" in system_prompt

        # Verify result type is correct
        assert agent.output_type == RawSearchResults

    def test_matching_agent_configuration(self) -> None:
        """Test that matching agent is configured correctly without tools."""
        agent = create_matching_agent("gpt-4")

        # Matching agent should have no tools (it works with provided data)
        tools = agent._function_toolset.tools if agent._function_toolset else []
        assert len(tools) == 0

        # Verify system prompt contains location selection guidance
        system_prompt = agent._system_prompts[0].lower() if agent._system_prompts else ""
        assert "primary" in system_prompt or "location" in system_prompt
        assert "select" in system_prompt or "pick" in system_prompt

        # Verify result type is correct
        assert agent.output_type == LocationSearchResponse


class TestFindAnatomicLocations:
    """Test main API function workflow using Pydantic AI test utilities."""

    @pytest.mark.asyncio
    async def test_two_agent_workflow_success(self) -> None:
        """Test the two-agent workflow with controlled responses - tests our workflow logic."""
        # Create controlled search results
        mock_results = [
            OntologySearchResult(
                concept_id="A123",
                concept_text="Posterior cruciate ligament",
                score=0.95,
                table_name="anatomic_locations",
            ),
            OntologySearchResult(
                concept_id="A456", concept_text="Knee joint", score=0.88, table_name="anatomic_locations"
            ),
        ]

        # Create controlled responses
        search_response = RawSearchResults(
            search_terms_used=["PCL", "posterior cruciate ligament", "knee"],
            search_results=mock_results,
        )

        matching_response = LocationSearchResponse(
            primary_location=mock_results[0],
            alternate_locations=[mock_results[1]],
            reasoning="PCL is most specific for this ligament tear finding.",
        )

        # Mock the search client to avoid database connections
        with patch("findingmodel.tools.anatomic_location_search.OntologySearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.connected = True
            mock_client.disconnect = MagicMock()
            mock_client_class.return_value = mock_client

            # Test the actual workflow with controlled agent responses using TestModel
            search_agent = create_search_agent("gpt-4")
            matching_agent = create_matching_agent("gpt-4")

            with (
                search_agent.override(model=TestModel(custom_output_args=search_response)),
                matching_agent.override(model=TestModel(custom_output_args=matching_response)),
                patch("findingmodel.tools.anatomic_location_search.create_search_agent", return_value=search_agent),
                patch("findingmodel.tools.anatomic_location_search.create_matching_agent", return_value=matching_agent),
            ):
                # Test the actual workflow
                result = await find_anatomic_locations("PCL tear", "Tear of the posterior cruciate ligament")

                # Verify our workflow logic works correctly
                assert isinstance(result, LocationSearchResponse)
                assert result.primary_location is not None
                assert result.primary_location.concept_text == "Posterior cruciate ligament"
                assert len(result.alternate_locations) >= 0
                assert result.reasoning is not None and len(result.reasoning) > 0

            # Verify client connection lifecycle
            mock_client.connect.assert_called_once()
            mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_handles_connection_errors(self) -> None:
        """Test that our workflow properly handles database connection failures."""
        with patch("findingmodel.tools.anatomic_location_search.OntologySearchClient") as mock_client_class:
            # Setup mock client that fails to connect - tests our error handling logic
            mock_client = MagicMock()
            mock_client.connect = AsyncMock(side_effect=Exception("Connection failed"))
            mock_client.connected = False
            mock_client.disconnect = MagicMock()
            mock_client_class.return_value = mock_client

            # Test that our code properly propagates connection errors
            with pytest.raises(Exception, match="Connection failed"):
                await find_anatomic_locations("test finding")

            # Verify our error handling doesn't call disconnect when connect fails
            mock_client.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_handles_empty_search_results(self) -> None:
        """Test our workflow logic when search returns no results."""
        # Create controlled responses for empty search scenario
        empty_search_response = RawSearchResults(
            search_terms_used=["unknown finding"],
            search_results=[],
        )

        primary = OntologySearchResult(
            concept_id="A000", concept_text="Unknown location", score=0.1, table_name="anatomic_locations"
        )
        fallback_matching_response = LocationSearchResponse(
            primary_location=primary,
            alternate_locations=[],
            reasoning="No specific locations found, using generic location.",
        )

        with patch("findingmodel.tools.anatomic_location_search.OntologySearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.connected = True
            mock_client.disconnect = MagicMock()
            mock_client_class.return_value = mock_client

            search_agent = create_search_agent("gpt-4")
            matching_agent = create_matching_agent("gpt-4")

            with (
                search_agent.override(model=TestModel(custom_output_args=empty_search_response)),
                matching_agent.override(model=TestModel(custom_output_args=fallback_matching_response)),
                patch("findingmodel.tools.anatomic_location_search.create_search_agent", return_value=search_agent),
                patch("findingmodel.tools.anatomic_location_search.create_matching_agent", return_value=matching_agent),
            ):
                # Test that our workflow handles empty search results gracefully
                result = await find_anatomic_locations("unknown finding")

                # Should still return a valid result despite empty search
                assert isinstance(result, LocationSearchResponse)
                assert result.primary_location.concept_text == "Unknown location"
                assert len(result.alternate_locations) == 0

    @pytest.mark.asyncio
    async def test_workflow_cleanup_on_exceptions(self) -> None:
        """Test that our workflow always performs cleanup even when exceptions occur."""
        with patch("findingmodel.tools.anatomic_location_search.OntologySearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.connected = True
            mock_client.disconnect = MagicMock()
            mock_client_class.return_value = mock_client

            # Create a simple mock that raises an exception for cleanup testing
            mock_search_agent = AsyncMock()
            mock_search_agent.run.side_effect = Exception("Search agent failed")

            with patch(
                "findingmodel.tools.anatomic_location_search.create_search_agent", return_value=mock_search_agent
            ):
                # Test that exceptions are propagated but cleanup still happens
                with pytest.raises(Exception, match="Search agent failed"):
                    await find_anatomic_locations("test finding")

                # Verify our cleanup logic works even with exceptions
                mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_with_custom_model_parameters(self) -> None:
        """Test that our workflow properly passes custom model parameters to agent creation."""
        # Create controlled responses
        search_results = [
            OntologySearchResult(concept_id="A123", concept_text="Heart", score=0.95, table_name="anatomic_locations")
        ]
        search_response = RawSearchResults(search_terms_used=["heart"], search_results=search_results)

        matching_response = LocationSearchResponse(
            primary_location=search_results[0],
            alternate_locations=[],
            reasoning="Heart is the primary location.",
        )

        with (
            patch("findingmodel.tools.anatomic_location_search.OntologySearchClient") as mock_client_class,
            patch("findingmodel.tools.anatomic_location_search.create_search_agent") as mock_create_search,
            patch("findingmodel.tools.anatomic_location_search.create_matching_agent") as mock_create_matching,
        ):
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.connected = True
            mock_client.disconnect = MagicMock()
            mock_client_class.return_value = mock_client

            # Create mock agents with controlled behavior
            search_agent = create_search_agent("gpt-4o-mini")
            matching_agent = create_matching_agent("gpt-4o")
            mock_create_search.return_value = search_agent
            mock_create_matching.return_value = matching_agent

            with (
                search_agent.override(model=TestModel(custom_output_args=search_response)),
                matching_agent.override(model=TestModel(custom_output_args=matching_response)),
            ):
                # Test our workflow passes custom model names correctly
                result = await find_anatomic_locations(
                    "heart murmur",
                    search_model="gpt-4o-mini",
                    matching_model="gpt-4o",
                )

                # Verify our workflow logic called agent creation with correct models
                mock_create_search.assert_called_once_with("gpt-4o-mini")
                mock_create_matching.assert_called_once_with("gpt-4o")

                assert isinstance(result, LocationSearchResponse)

    @pytest.mark.callout
    @pytest.mark.asyncio
    async def test_find_anatomic_locations_integration(self) -> None:
        """Integration test with real LanceDB and AI agents."""
        # This test requires real API keys and LanceDB connection
        # Temporarily enable model requests for this integration test
        original_setting = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True

        try:
            result = await find_anatomic_locations(
                finding_name="PCL tear",
                description="Tear of the posterior cruciate ligament",
                search_model="gpt-4o-mini",  # Use smaller model for faster testing
                matching_model="gpt-4o-mini",
            )

            # Verify structure
            assert isinstance(result, LocationSearchResponse)
            assert result.primary_location is not None
            assert result.primary_location.concept_text is not None
            assert isinstance(result.alternate_locations, list)
            assert len(result.reasoning) > 0

            # Verify the primary location is reasonable for PCL tear
            primary_text = result.primary_location.concept_text.lower()
            # Should be related to knee, ligament, or leg anatomy
            expected_terms = ["knee", "ligament", "cruciate", "leg", "joint", "lower", "limb"]
            assert any(term in primary_text for term in expected_terms), (
                f"Expected anatomic location related to knee/ligament, got: {primary_text}"
            )

            # Verify alternates are also reasonable
            for alternate in result.alternate_locations:
                assert alternate.concept_text is not None
                assert len(alternate.concept_text) > 0

        except Exception as e:
            # If LanceDB or API keys aren't available, skip the test
            if any(term in str(e).lower() for term in ["connection", "api", "key", "auth", "lance"]):
                pytest.skip(f"Integration test requires LanceDB connection and API keys: {e}")
            else:
                raise
        finally:
            # Always restore the original setting
            models.ALLOW_MODEL_REQUESTS = original_setting


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_workflow_handles_empty_finding_name(self) -> None:
        """Test our workflow can handle edge case of empty finding name."""
        # Create controlled responses for empty input scenario
        empty_search_response = RawSearchResults(search_terms_used=[], search_results=[])

        primary = OntologySearchResult(
            concept_id="A000", concept_text="Unknown", score=0.1, table_name="anatomic_locations"
        )
        empty_matching_response = LocationSearchResponse(
            primary_location=primary, alternate_locations=[], reasoning="No specific location found."
        )

        with patch("findingmodel.tools.anatomic_location_search.OntologySearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.connected = True
            mock_client.disconnect = MagicMock()
            mock_client_class.return_value = mock_client

            search_agent = create_search_agent("gpt-4")
            matching_agent = create_matching_agent("gpt-4")

            with (
                search_agent.override(model=TestModel(custom_output_args=empty_search_response)),
                matching_agent.override(model=TestModel(custom_output_args=empty_matching_response)),
                patch("findingmodel.tools.anatomic_location_search.create_search_agent", return_value=search_agent),
                patch("findingmodel.tools.anatomic_location_search.create_matching_agent", return_value=matching_agent),
            ):
                # Test that our workflow handles empty input gracefully
                result = await find_anatomic_locations("")
                assert isinstance(result, LocationSearchResponse)

    @pytest.mark.asyncio
    async def test_workflow_handles_very_long_input(self) -> None:
        """Test our workflow can handle edge case of very long inputs."""
        long_finding = "very long finding name " * 50  # ~1000 characters
        long_description = "very long description " * 100  # ~2000 characters

        # Create controlled responses for long input scenario
        long_search_response = RawSearchResults(search_terms_used=["long"], search_results=[])

        primary = OntologySearchResult(
            concept_id="A000", concept_text="Generic location", score=0.1, table_name="anatomic_locations"
        )
        long_matching_response = LocationSearchResponse(
            primary_location=primary, alternate_locations=[], reasoning="Handled long input."
        )

        with patch("findingmodel.tools.anatomic_location_search.OntologySearchClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock()
            mock_client.connected = True
            mock_client.disconnect = MagicMock()
            mock_client_class.return_value = mock_client

            search_agent = create_search_agent("gpt-4")
            matching_agent = create_matching_agent("gpt-4")

            with (
                search_agent.override(model=TestModel(custom_output_args=long_search_response)),
                matching_agent.override(model=TestModel(custom_output_args=long_matching_response)),
                patch("findingmodel.tools.anatomic_location_search.create_search_agent", return_value=search_agent),
                patch("findingmodel.tools.anatomic_location_search.create_matching_agent", return_value=matching_agent),
            ):
                # Test that our workflow handles long inputs without crashing
                result = await find_anatomic_locations(long_finding, long_description)
                assert isinstance(result, LocationSearchResponse)

    def test_raw_search_results_empty_lists(self) -> None:
        """Test RawSearchResults handles empty lists properly - edge case validation."""
        results = RawSearchResults(search_terms_used=[], search_results=[])

        assert results.search_terms_used == []
        assert results.search_results == []

    def test_location_search_response_empty_alternates(self) -> None:
        """Test LocationSearchResponse handles empty alternates properly - edge case validation."""
        primary = OntologySearchResult(
            concept_id="A123", concept_text="Heart", score=0.95, table_name="anatomic_locations"
        )

        response = LocationSearchResponse(
            primary_location=primary,
            alternate_locations=[],
            reasoning="Only one location found.",
        )

        assert len(response.alternate_locations) == 0
        assert response.primary_location.concept_text == "Heart"
