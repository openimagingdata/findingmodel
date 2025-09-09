"""Tests for the ontology concept search tool."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from findingmodel.tools.ontology_concept_match import (
    CategorizationContext,
    CategorizedConcepts,
    create_categorization_agent,
    create_query_generator_agent,
    ensure_exact_matches_post_process,
    execute_ontology_search,
    generate_finding_query_terms,
    match_ontology_concepts,
)
from findingmodel.tools.ontology_search import OntologySearchResult

# Prevent accidental API calls during testing
models.ALLOW_MODEL_REQUESTS = False


# Query Terms Generation Tests


def test_query_terms_deduplication() -> None:
    """Test that duplicate terms are removed from list."""
    # Test deduplication logic with a list of terms
    terms = ["Pulmonary Embolism", "pulmonary embolism", "PE", "lung embolism", "PE", "embolism", "thromboembolism"]

    # Remove duplicates (case-insensitive)
    unique_terms = []
    seen_lower = set()
    for term in terms:
        if term.lower() not in seen_lower:
            unique_terms.append(term)
            seen_lower.add(term.lower())

    # Should not have duplicate "pulmonary embolism" (case insensitive)
    assert unique_terms[0] == "Pulmonary Embolism"
    assert "pulmonary embolism" not in unique_terms[1:]  # Not in rest of list

    # Should only have one "PE"
    pe_count = sum(1 for term in unique_terms if term == "PE")
    assert pe_count == 1

    # Check expected unique terms
    assert "lung embolism" in unique_terms
    assert "embolism" in unique_terms
    assert "thromboembolism" in unique_terms


# Generate Query Terms Tests


@pytest.mark.asyncio
async def test_generate_finding_query_terms_single_word() -> None:
    """Test query generation for single word."""
    # Mock the query generator agent to avoid API calls
    with patch("findingmodel.tools.ontology_concept_match.create_query_generator_agent") as mock_create:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = ["pneumonia", "lung infection", "pneumonitis", "bronchitis"]
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        result = await generate_finding_query_terms("pneumonia")

        assert "pneumonia" in result
        assert "lung infection" in result


@pytest.mark.asyncio
async def test_generate_finding_query_terms_with_description() -> None:
    """Test query generation with description."""
    # Mock the query generator agent
    with patch("findingmodel.tools.ontology_concept_match.create_query_generator_agent") as mock_create:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = ["pneumonia", "lung infection", "lung pneumonia"]  # Should infer anatomy
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        result = await generate_finding_query_terms("pneumonia", "lung inflammation")

        assert "pneumonia" in result
        # Should have anatomy inference
        assert "lung pneumonia" in result


def test_create_query_generator_agent() -> None:
    """Test that the query generator agent is created correctly."""
    with patch("findingmodel.tools.ontology_concept_match.get_openai_model") as mock_model:
        agent = create_query_generator_agent()

        # Should get the default model
        mock_model.assert_called_once()

        # Check agent configuration - output should be list[str]
        # Note: _output_type is the actual type annotation list[str], not just list
        assert str(agent._output_type).startswith("<class 'list") or str(agent._output_type) == "list[str]"

        # Agent should be created successfully
        assert agent is not None


# Execute Ontology Search Tests


@pytest.mark.asyncio
async def test_execute_ontology_search() -> None:
    """Test executing search with filtering."""
    # Create mock client
    mock_client = MagicMock()
    mock_results = [
        OntologySearchResult(concept_id="1", concept_text="pneumonia", score=0.95, table_name="radlex"),
        OntologySearchResult(concept_id="2", concept_text="Pneumonia", score=0.93, table_name="snomedct"),
        OntologySearchResult(concept_id="3", concept_text="viral pneumonia", score=0.8, table_name="radlex"),
    ]
    mock_client.search = AsyncMock(return_value=mock_results)

    # Create query terms
    query_terms = ["pneumonia", "lung infection"]

    # Execute search
    results = await execute_ontology_search(query_terms=query_terms, client=mock_client, exclude_anatomical=True)

    # Verify client was called with correct parameters
    mock_client.search.assert_called_once()
    call_args = mock_client.search.call_args
    assert "pneumonia" in call_args.kwargs["queries"]
    assert "lung infection" in call_args.kwargs["queries"]
    assert call_args.kwargs["filter_anatomical"] is True

    # Check results
    assert len(results) == 3
    # Results should be normalized
    assert results[0].concept_text == "pneumonia"


# Ensure Exact Matches Post Process Tests


def test_ensure_exact_matches_adds_missing() -> None:
    """Test that missing exact matches are added."""
    # Create test data
    output = CategorizedConcepts(
        exact_matches=["RID5350"],
        should_include=["RID5351"],
        marginal=["SCTID-233604007"],  # This should be moved to exact
        rationale="Initial categorization",
    )

    search_results = [
        OntologySearchResult(concept_id="RID5350", concept_text="pneumonia", score=0.95, table_name="radlex"),
        OntologySearchResult(concept_id="SCTID-233604007", concept_text="Pneumonia", score=0.93, table_name="snomedct"),
        OntologySearchResult(concept_id="RID5351", concept_text="viral pneumonia", score=0.8, table_name="radlex"),
    ]

    query_terms = ["pneumonia"]

    # Process
    corrected = ensure_exact_matches_post_process(output, search_results, query_terms)

    # Verify corrections
    assert "RID5350" in corrected.exact_matches
    assert "SCTID-233604007" in corrected.exact_matches
    assert "SCTID-233604007" not in corrected.marginal
    assert "Auto-corrected" in corrected.rationale


def test_ensure_exact_matches_respects_limit() -> None:
    """Test that max_length of 5 is respected."""
    # Create output already at limit
    output = CategorizedConcepts(
        exact_matches=["1", "2", "3", "4", "5"],
        should_include=[],
        marginal=["6"],  # This is an exact match but can't be added
        rationale="At limit",
    )

    search_results = [
        OntologySearchResult(concept_id=str(i), concept_text="test", score=0.9, table_name="radlex")
        for i in range(1, 7)
    ]

    query_terms = ["test"]

    # Process - should not exceed limit
    corrected = ensure_exact_matches_post_process(output, search_results, query_terms)

    assert len(corrected.exact_matches) == 5


# Categorization Agent Tests


def test_categorization_agent_creation() -> None:
    """Test that agent is created properly."""
    with patch("findingmodel.tools.ontology_concept_match.get_openai_model") as mock_model:
        agent = create_categorization_agent()

        # Should get the default model
        mock_model.assert_called_once()

        # Agent should exist
        assert agent is not None


@pytest.mark.asyncio
async def test_categorization_with_test_model() -> None:
    """Test categorization using TestModel."""
    # Create test response
    test_output = CategorizedConcepts(
        exact_matches=["RID5350"],
        should_include=["RID5351"],
        marginal=["RID5352"],
        rationale="Test categorization",
    )

    # Create agent and override with TestModel
    agent = create_categorization_agent()
    test_model = TestModel(custom_output_args=test_output)

    # Create context
    context = CategorizationContext(
        finding_name="pneumonia",
        search_results=[
            OntologySearchResult(concept_id="RID5350", concept_text="pneumonia", score=0.95, table_name="radlex"),
        ],
        query_terms=["pneumonia"],
    )

    # Run with override
    with agent.override(model=test_model):
        result = await agent.run("Test prompt", deps=context)

        # Verify we got our test output
        assert result.output.exact_matches == ["RID5350"]
        assert result.output.rationale == "Test categorization"


# Main Orchestration Tests


@pytest.mark.asyncio
async def test_match_ontology_concepts_integration() -> None:
    """Test the complete workflow with explicit client."""
    # Create a mock client and provide it directly to avoid auto-detection
    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="RID5350", concept_text="pneumonia", score=0.95, table_name="radlex"),
        ]
    )
    # Mock the async context manager behavior
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock the categorization agent
    with patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_create:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = CategorizedConcepts(
            exact_matches=["RID5350"], should_include=[], marginal=[], rationale="Test"
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        # Run with explicit client (no auto-detection)
        result = await match_ontology_concepts(
            finding_name="pneumonia",
            finding_description="lung infection",
            search_clients=mock_client,  # Provide client explicitly
        )

        # Verify context manager was used
        mock_client.__aenter__.assert_called_once()
        mock_client.__aexit__.assert_called_once()

        # Check result
        assert result is not None
        assert len(result.exact_matches) == 1


@pytest.mark.asyncio
async def test_match_cleanup_on_error() -> None:
    """Test that client context manager handles errors gracefully."""
    # Setup mock client that fails on enter
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(side_effect=ConnectionError("Connection failed"))
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Run and expect exception
    with pytest.raises(ConnectionError, match="Connection failed"):
        await match_ontology_concepts("test", search_clients=mock_client)

    # Verify context manager entry was attempted
    mock_client.__aenter__.assert_called_once()


# Auto-detection Tests


@pytest.mark.asyncio
async def test_auto_detect_lancedb_only() -> None:
    """Test auto-detection when only LanceDB credentials are available."""
    # Mock settings to have only LanceDB credentials
    mock_settings = MagicMock()
    mock_settings.lancedb_uri = "test-uri"
    mock_settings.lancedb_api_key = MagicMock()
    mock_settings.lancedb_api_key.get_secret_value.return_value = "test-key"
    mock_settings.bioontology_api_key = None  # No BioOntology key

    with (
        patch("findingmodel.tools.ontology_concept_match.settings", mock_settings),
        patch("findingmodel.tools.ontology_concept_match.LanceDBOntologySearchClient") as MockLanceDB,
        patch("findingmodel.tools.ontology_concept_match.generate_finding_query_terms") as mock_query_gen,
        patch("findingmodel.tools.ontology_concept_match.categorize_with_validation") as mock_categorize,
        patch("findingmodel.tools.ontology_concept_match.build_final_output") as mock_build,
    ):
        # Setup mocks
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        MockLanceDB.return_value = mock_client

        mock_query_gen.return_value = ["test"]
        mock_categorize.return_value = MagicMock(exact_matches=[], should_include=[], marginal=[], rationale="test")
        mock_build.return_value = MagicMock()

        # Call without providing clients (triggers auto-detection)
        await match_ontology_concepts(
            finding_name="test",
            search_clients=None,  # Triggers auto-detection
        )

        # Verify LanceDB client was created
        MockLanceDB.assert_called_once_with(lancedb_uri="test-uri", api_key="test-key")

        # Verify context manager was used
        mock_client.__aenter__.assert_called_once()
        mock_client.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_auto_detect_bioontology_only() -> None:
    """Test auto-detection when only BioOntology API key is available."""
    # Mock settings to have only BioOntology credentials
    mock_settings = MagicMock()
    mock_settings.lancedb_uri = None  # No LanceDB
    mock_settings.lancedb_api_key = None
    mock_settings.bioontology_api_key = "test-bio-key"

    with (
        patch("findingmodel.tools.ontology_concept_match.settings", mock_settings),
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient") as MockBio,
        patch("findingmodel.tools.ontology_concept_match.generate_finding_query_terms") as mock_query_gen,
        patch("findingmodel.tools.ontology_concept_match.categorize_with_validation") as mock_categorize,
        patch("findingmodel.tools.ontology_concept_match.build_final_output") as mock_build,
    ):
        # Setup mocks
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=[])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        MockBio.return_value = mock_client

        mock_query_gen.return_value = ["test"]
        mock_categorize.return_value = MagicMock(exact_matches=[], should_include=[], marginal=[], rationale="test")
        mock_build.return_value = MagicMock()

        # Call without providing clients
        await match_ontology_concepts(finding_name="test", search_clients=None)

        # Verify BioOntology client was created without arguments
        # (it gets API key from settings internally)
        MockBio.assert_called_once_with()

        # Verify context manager was used
        mock_client.__aenter__.assert_called_once()


@pytest.mark.asyncio
async def test_auto_detect_both_backends() -> None:
    """Test auto-detection when both backends are configured."""
    import asyncio

    # Mock settings with both backends configured
    mock_settings = MagicMock()
    mock_settings.lancedb_uri = "test-uri"
    mock_settings.lancedb_api_key = MagicMock()
    mock_settings.lancedb_api_key.get_secret_value.return_value = "test-key"
    mock_settings.bioontology_api_key = "test-bio-key"

    with (
        patch("findingmodel.tools.ontology_concept_match.settings", mock_settings),
        patch("findingmodel.tools.ontology_concept_match.LanceDBOntologySearchClient") as MockLanceDB,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient") as MockBio,
        patch("findingmodel.tools.ontology_concept_match.generate_finding_query_terms") as mock_query_gen,
        patch("findingmodel.tools.ontology_concept_match.categorize_with_validation") as mock_categorize,
        patch("findingmodel.tools.ontology_concept_match.build_final_output") as mock_build,
    ):
        # Setup mock clients
        mock_lance = MagicMock()
        mock_lance.search = AsyncMock(
            return_value=[
                OntologySearchResult(concept_id="L1", concept_text="lance result", score=0.9, table_name="test")
            ]
        )
        mock_lance.__aenter__ = AsyncMock(return_value=mock_lance)
        mock_lance.__aexit__ = AsyncMock(return_value=None)
        MockLanceDB.return_value = mock_lance

        mock_bio = MagicMock()
        mock_bio.search = AsyncMock(
            return_value=[
                OntologySearchResult(concept_id="B1", concept_text="bio result", score=0.8, table_name="test")
            ]
        )
        mock_bio.__aenter__ = AsyncMock(return_value=mock_bio)
        mock_bio.__aexit__ = AsyncMock(return_value=None)
        MockBio.return_value = mock_bio

        mock_query_gen.return_value = ["test"]
        mock_categorize.return_value = MagicMock(
            exact_matches=["L1", "B1"], should_include=[], marginal=[], rationale="test"
        )
        mock_build.return_value = MagicMock()

        # Patch asyncio.gather to verify it's called
        original_gather = asyncio.gather
        gather_called = False

        async def mock_gather(*args: Any, **kwargs: Any) -> Any:
            nonlocal gather_called
            gather_called = True
            return await original_gather(*args, **kwargs)

        with patch("findingmodel.tools.ontology_concept_match.asyncio.gather", mock_gather):
            # Call without providing clients
            await match_ontology_concepts(finding_name="test", search_clients=None)

            # Verify both clients were created
            MockLanceDB.assert_called_once()
            MockBio.assert_called_once()

            # Verify asyncio.gather was used for parallel execution
            assert gather_called, "asyncio.gather should be called for multiple backends"

            # Verify both clients were used
            mock_lance.__aenter__.assert_called_once()
            mock_bio.__aenter__.assert_called_once()


@pytest.mark.asyncio
async def test_auto_detect_no_backends_raises() -> None:
    """Test that auto-detection raises ValueError when no backends are configured."""
    # Mock settings with no backends configured
    mock_settings = MagicMock()
    mock_settings.lancedb_uri = None
    mock_settings.lancedb_api_key = None
    mock_settings.bioontology_api_key = None

    with (
        patch("findingmodel.tools.ontology_concept_match.settings", mock_settings),
        pytest.raises(ValueError, match="No ontology search backends configured"),
    ):
        await match_ontology_concepts(
            finding_name="test",
            search_clients=None,  # Triggers auto-detection
        )


# Multiple Backend Tests


@pytest.mark.asyncio
async def test_multiple_backends_parallel_search() -> None:
    """Test that multiple backends execute searches in parallel using asyncio.gather."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import OntologySearchResult

    # Create two mock clients
    mock_client1 = MagicMock()
    mock_client1.search = AsyncMock(
        return_value=[OntologySearchResult(concept_id="C1", concept_text="result1", score=0.95, table_name="source1")]
    )
    mock_client1.__aenter__ = AsyncMock(return_value=mock_client1)
    mock_client1.__aexit__ = AsyncMock(return_value=None)

    mock_client2 = MagicMock()
    mock_client2.search = AsyncMock(
        return_value=[OntologySearchResult(concept_id="C2", concept_text="result2", score=0.90, table_name="source2")]
    )
    mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
    mock_client2.__aexit__ = AsyncMock(return_value=None)

    # Track gather calls
    gather_called = False
    gather_args = None
    original_gather = asyncio.gather

    async def mock_gather(*args: Any, **kwargs: Any) -> Any:
        nonlocal gather_called, gather_args
        gather_called = True
        gather_args = args
        return await original_gather(*args, **kwargs)

    with (
        patch("findingmodel.tools.ontology_concept_match.asyncio.gather", mock_gather),
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent,
    ):
        mock_categorizer = MagicMock()
        mock_categorizer.run = AsyncMock(
            return_value=MagicMock(
                output=MagicMock(exact_matches=["C1"], should_include=["C2"], marginal=[], rationale="test")
            )
        )
        mock_agent.return_value = mock_categorizer

        # Call with list of clients
        await match_ontology_concepts(finding_name="test", search_clients=[mock_client1, mock_client2])

        # Verify asyncio.gather was called
        assert gather_called, "asyncio.gather should be called for multiple clients"
        assert len(gather_args) == 2, "gather should be called with 2 tasks"

        # Verify both clients were used
        mock_client1.search.assert_called_once()
        mock_client2.search.assert_called_once()


@pytest.mark.asyncio
async def test_multiple_backends_result_merging() -> None:
    """Test that results from multiple backends are properly merged."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import OntologySearchResult

    # Create mock clients with different results
    mock_client1 = MagicMock()
    mock_client1.search = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="R1", concept_text="result1", score=0.95, table_name="source1"),
            OntologySearchResult(concept_id="R2", concept_text="result2", score=0.85, table_name="source1"),
        ]
    )
    mock_client1.__aenter__ = AsyncMock(return_value=mock_client1)
    mock_client1.__aexit__ = AsyncMock(return_value=None)

    mock_client2 = MagicMock()
    mock_client2.search = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="R3", concept_text="result3", score=0.90, table_name="source2"),
            OntologySearchResult(concept_id="R4", concept_text="result4", score=0.80, table_name="source2"),
        ]
    )
    mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
    mock_client2.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent:
        # Capture the search results passed to categorization
        captured_results = None

        def capture_categorizer_input(prompt: Any, deps: Any) -> Any:
            nonlocal captured_results
            captured_results = deps.search_results
            return MagicMock(
                output=MagicMock(
                    exact_matches=["R1", "R3"], should_include=["R2", "R4"], marginal=[], rationale="merged"
                )
            )

        mock_categorizer = MagicMock()
        mock_categorizer.run = AsyncMock(side_effect=capture_categorizer_input)
        mock_agent.return_value = mock_categorizer

        # Call with multiple clients
        await match_ontology_concepts(finding_name="test", search_clients=[mock_client1, mock_client2])

        # Verify all results were merged
        assert captured_results is not None
        result_ids = {r.concept_id for r in captured_results}
        assert result_ids == {"R1", "R2", "R3", "R4"}, "All results should be merged"

        # Verify results are sorted by score
        scores = [r.score for r in captured_results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"


@pytest.mark.asyncio
async def test_multiple_backends_deduplication() -> None:
    """Test that duplicate results from multiple backends are deduplicated by concept_id."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import OntologySearchResult

    # Create mock clients with overlapping results
    mock_client1 = MagicMock()
    mock_client1.search = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="DUP1", concept_text="duplicate", score=0.95, table_name="source1"),
            OntologySearchResult(concept_id="UNIQ1", concept_text="unique1", score=0.85, table_name="source1"),
        ]
    )
    mock_client1.__aenter__ = AsyncMock(return_value=mock_client1)
    mock_client1.__aexit__ = AsyncMock(return_value=None)

    mock_client2 = MagicMock()
    mock_client2.search = AsyncMock(
        return_value=[
            OntologySearchResult(
                concept_id="DUP1", concept_text="duplicate", score=0.90, table_name="source2"
            ),  # Duplicate!
            OntologySearchResult(concept_id="UNIQ2", concept_text="unique2", score=0.80, table_name="source2"),
        ]
    )
    mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
    mock_client2.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent:
        # Capture the search results passed to categorization
        captured_results = None

        def capture_categorizer_input(prompt: Any, deps: Any) -> Any:
            nonlocal captured_results
            captured_results = deps.search_results
            return MagicMock(
                output=MagicMock(
                    exact_matches=["DUP1"], should_include=["UNIQ1", "UNIQ2"], marginal=[], rationale="dedup"
                )
            )

        mock_categorizer = MagicMock()
        mock_categorizer.run = AsyncMock(side_effect=capture_categorizer_input)
        mock_agent.return_value = mock_categorizer

        # Call with multiple clients
        await match_ontology_concepts(finding_name="test", search_clients=[mock_client1, mock_client2])

        # Verify duplicates were removed
        assert captured_results is not None
        result_ids = [r.concept_id for r in captured_results]
        assert result_ids.count("DUP1") == 1, "Duplicate concept_id should appear only once"

        # Verify all unique results are present
        unique_ids = set(result_ids)
        assert unique_ids == {"DUP1", "UNIQ1", "UNIQ2"}, "All unique concepts should be present"


@pytest.mark.asyncio
async def test_single_backend_in_list() -> None:
    """Test edge case: list with single client should work without gather."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import OntologySearchResult

    # Create one mock client
    mock_client = MagicMock()
    mock_client.search = AsyncMock(
        return_value=[OntologySearchResult(concept_id="S1", concept_text="single", score=0.95, table_name="source")]
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent:
        mock_categorizer = MagicMock()
        mock_categorizer.run = AsyncMock(
            return_value=MagicMock(
                output=MagicMock(exact_matches=["S1"], should_include=[], marginal=[], rationale="single")
            )
        )
        mock_agent.return_value = mock_categorizer

        # Call with list containing single client
        result = await match_ontology_concepts(
            finding_name="test",
            search_clients=[mock_client],  # List with one client
        )

        # Verify client was used
        mock_client.search.assert_called_once()
        mock_client.__aenter__.assert_called_once()

        # Verify result
        assert result is not None
        assert len(result.exact_matches) == 1


# Error Handling Tests


@pytest.mark.asyncio
async def test_partial_backend_failure() -> None:
    """Test that when one backend fails, others still succeed and results are returned."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import OntologySearchResult

    # Create one successful client
    mock_success_client = MagicMock()
    mock_success_client.search = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="SUCCESS1", concept_text="success", score=0.95, table_name="good")
        ]
    )
    mock_success_client.__aenter__ = AsyncMock(return_value=mock_success_client)
    mock_success_client.__aexit__ = AsyncMock(return_value=None)

    # Create one failing client
    mock_fail_client = MagicMock()
    mock_fail_client.search = AsyncMock(side_effect=ConnectionError("Backend unavailable"))
    mock_fail_client.__aenter__ = AsyncMock(return_value=mock_fail_client)
    mock_fail_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent,
        patch("findingmodel.tools.ontology_concept_match.logger") as mock_logger,
    ):
        # Capture the search results to verify partial success
        captured_results = None

        def capture_categorizer_input(prompt: Any, deps: Any) -> Any:
            nonlocal captured_results
            captured_results = deps.search_results
            return MagicMock(
                output=MagicMock(exact_matches=["SUCCESS1"], should_include=[], marginal=[], rationale="partial")
            )

        mock_categorizer = MagicMock()
        mock_categorizer.run = AsyncMock(side_effect=capture_categorizer_input)
        mock_agent.return_value = mock_categorizer

        # Call with both clients
        result = await match_ontology_concepts(
            finding_name="test", search_clients=[mock_success_client, mock_fail_client]
        )

        # Verify partial failure was logged
        mock_logger.warning.assert_called()
        warning_call = str(mock_logger.warning.call_args)
        assert "failed" in warning_call.lower()

        # Verify successful results were still processed
        assert captured_results is not None
        assert len(captured_results) == 1
        assert captured_results[0].concept_id == "SUCCESS1"

        # Verify final result is returned
        assert result is not None
        assert len(result.exact_matches) == 1


@pytest.mark.asyncio
async def test_all_backends_fail() -> None:
    """Test graceful handling when all backends fail."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts

    # Create two failing clients
    mock_fail1 = MagicMock()
    mock_fail1.search = AsyncMock(side_effect=ConnectionError("Backend 1 unavailable"))
    mock_fail1.__aenter__ = AsyncMock(return_value=mock_fail1)
    mock_fail1.__aexit__ = AsyncMock(return_value=None)

    mock_fail2 = MagicMock()
    mock_fail2.search = AsyncMock(side_effect=TimeoutError("Backend 2 timeout"))
    mock_fail2.__aenter__ = AsyncMock(return_value=mock_fail2)
    mock_fail2.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent,
        patch("findingmodel.tools.ontology_concept_match.logger") as mock_logger,
    ):
        mock_categorizer = MagicMock()
        # Categorizer should receive empty results
        mock_categorizer.run = AsyncMock(
            return_value=MagicMock(
                output=MagicMock(exact_matches=[], should_include=[], marginal=[], rationale="no results")
            )
        )
        mock_agent.return_value = mock_categorizer

        # Call with failing clients
        result = await match_ontology_concepts(finding_name="test", search_clients=[mock_fail1, mock_fail2])

        # Verify failures were logged
        assert mock_logger.warning.call_count >= 2, "Both failures should be logged"

        # Verify empty result is returned gracefully
        assert result is not None
        assert len(result.exact_matches) == 0
        assert len(result.should_include) == 0
        assert len(result.marginal_concepts) == 0


@pytest.mark.asyncio
async def test_backend_exception_logging() -> None:
    """Test that backend exceptions are properly logged with details."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts

    # Create one working client and one failing client (need multiple clients for exception handling)
    mock_working_client = MagicMock()
    mock_working_client.search = AsyncMock(return_value=[])  # Empty results
    mock_working_client.__aenter__ = AsyncMock(return_value=mock_working_client)
    mock_working_client.__aexit__ = AsyncMock(return_value=None)

    mock_failing_client = MagicMock()
    test_exception = ValueError("Invalid API response format")
    mock_failing_client.search = AsyncMock(side_effect=test_exception)
    mock_failing_client.__aenter__ = AsyncMock(return_value=mock_failing_client)
    mock_failing_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent,
        patch("findingmodel.tools.ontology_concept_match.logger") as mock_logger,
    ):
        mock_categorizer = MagicMock()
        mock_categorizer.run = AsyncMock(
            return_value=MagicMock(
                output=MagicMock(exact_matches=[], should_include=[], marginal=[], rationale="error")
            )
        )
        mock_agent.return_value = mock_categorizer

        # Call with both clients (error handling only works with multiple clients)
        result = await match_ontology_concepts(
            finding_name="test", search_clients=[mock_working_client, mock_failing_client]
        )

        # Verify exception was logged with details
        mock_logger.warning.assert_called()
        warning_args = mock_logger.warning.call_args[0][0]
        assert "Invalid API response format" in warning_args

        # Verify function still returns a result
        assert result is not None


@pytest.mark.asyncio
async def test_mixed_exception_types() -> None:
    """Test handling of different exception types from multiple backends."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import OntologySearchResult

    # Create clients with different failure modes
    mock_timeout = MagicMock()
    mock_timeout.search = AsyncMock(side_effect=TimeoutError("Request timeout"))
    mock_timeout.__aenter__ = AsyncMock(return_value=mock_timeout)
    mock_timeout.__aexit__ = AsyncMock(return_value=None)

    mock_connection = MagicMock()
    mock_connection.search = AsyncMock(side_effect=ConnectionError("Connection refused"))
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock(return_value=None)

    mock_success = MagicMock()
    mock_success.search = AsyncMock(
        return_value=[OntologySearchResult(concept_id="OK1", concept_text="ok", score=0.9, table_name="good")]
    )
    mock_success.__aenter__ = AsyncMock(return_value=mock_success)
    mock_success.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_agent,
        patch("findingmodel.tools.ontology_concept_match.logger") as mock_logger,
    ):
        mock_categorizer = MagicMock()
        mock_categorizer.run = AsyncMock(
            return_value=MagicMock(
                output=MagicMock(exact_matches=["OK1"], should_include=[], marginal=[], rationale="mixed")
            )
        )
        mock_agent.return_value = mock_categorizer

        # Call with mixed success/failure clients
        result = await match_ontology_concepts(
            finding_name="test", search_clients=[mock_timeout, mock_connection, mock_success]
        )

        # Verify different exception types were logged
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("timeout" in call.lower() for call in warning_calls)
        assert any("connection" in call.lower() for call in warning_calls)

        # Verify successful results are still returned
        assert result is not None
        assert len(result.exact_matches) == 1


# Integration Tests (require real backends)


@pytest.mark.callout
@pytest.mark.asyncio
async def test_lancedb_through_protocol() -> None:
    """Integration test: Use real LanceDBOntologySearchClient through Protocol interface."""
    from findingmodel.config import settings
    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import LanceDBOntologySearchClient

    # Skip if no LanceDB credentials
    if not settings.lancedb_uri or not settings.lancedb_api_key:
        pytest.skip("LanceDB credentials not configured")

    # Create real LanceDB client
    client = LanceDBOntologySearchClient(
        lancedb_uri=settings.lancedb_uri,
        api_key=settings.lancedb_api_key.get_secret_value() if settings.lancedb_api_key else None,
    )

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

        # Test with real search
        result = await match_ontology_concepts(
            finding_name="pneumonia",
            finding_description="lung infection",
            search_clients=client,  # Use real LanceDB client
            max_exact_matches=3,
            max_should_include=5,
        )

        # Verify we got results
        assert result is not None
        # At minimum, we should have some categorized results
        total_results = len(result.exact_matches) + len(result.should_include) + len(result.marginal_concepts)
        assert total_results > 0, "Should find some ontology concepts for pneumonia"

        # Verify result structure
        assert hasattr(result, "exact_matches")
        assert hasattr(result, "should_include")
        assert hasattr(result, "marginal_concepts")
        assert hasattr(result, "search_summary")

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_through_protocol() -> None:
    """Integration test: Use real BioOntologySearchClient through Protocol interface."""
    from findingmodel.config import settings
    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import BioOntologySearchClient

    # Skip if no BioOntology API key
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    # Create real BioOntology client
    # Note: BioOntologySearchClient.__init__ already handles SecretStr correctly,
    # so we can just pass None and let it get the key from settings
    client = BioOntologySearchClient()

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

        # Test with real search
        result = await match_ontology_concepts(
            finding_name="fracture",
            finding_description="bone break",
            search_clients=client,  # Use real BioOntology client
            max_exact_matches=3,
            max_should_include=5,
        )

        # Verify we got results
        assert result is not None
        # At minimum, we should have some categorized results
        total_results = len(result.exact_matches) + len(result.should_include) + len(result.marginal_concepts)
        assert total_results > 0, "Should find some ontology concepts for fracture"

        # Verify result contains OntologySearchResult objects with expected fields
        for code in result.exact_matches:
            assert hasattr(code, "concept_id")
            assert hasattr(code, "concept_text")
            assert hasattr(code, "score")
            assert hasattr(code, "table_name")

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow


@pytest.mark.callout
@pytest.mark.asyncio
async def test_both_backends_integration() -> None:
    """Integration test: Use both real backends together via auto-detection."""
    from findingmodel.config import settings
    from findingmodel.tools.ontology_concept_match import match_ontology_concepts

    # Skip if neither backend is configured
    has_lancedb = settings.lancedb_uri and settings.lancedb_api_key
    has_bioontology = getattr(settings, "bioontology_api_key", None)

    if not has_lancedb and not has_bioontology:
        pytest.skip("No ontology backends configured")

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

        # Test with auto-detection (will use whatever backends are configured)
        result = await match_ontology_concepts(
            finding_name="hepatomegaly",
            finding_description="enlarged liver",
            search_clients=None,  # Auto-detect available backends
            max_exact_matches=5,
            max_should_include=10,
        )

        # Verify we got results
        assert result is not None

        # If both backends are configured, we should get more comprehensive results
        if has_lancedb and has_bioontology:
            # With both backends, we expect richer results
            total_results = len(result.exact_matches) + len(result.should_include) + len(result.marginal_concepts)
            assert total_results >= 1, "Should find concepts with at least one backend"

        # Verify search summary is populated
        assert result.search_summary is not None
        assert len(result.search_summary) > 0

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow


@pytest.mark.callout
@pytest.mark.asyncio
async def test_multiple_backends_deduplication_integration() -> None:
    """Integration test: Verify deduplication works with real backends."""
    from findingmodel.config import settings
    from findingmodel.tools.ontology_concept_match import match_ontology_concepts
    from findingmodel.tools.ontology_search import BioOntologySearchClient, LanceDBOntologySearchClient

    # Skip if both backends aren't configured
    if not (settings.lancedb_uri and settings.lancedb_api_key):
        pytest.skip("LanceDB not configured")
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology not configured")

    # Create both real clients
    lance_client = LanceDBOntologySearchClient(
        lancedb_uri=settings.lancedb_uri,
        api_key=settings.lancedb_api_key.get_secret_value() if settings.lancedb_api_key else None,
    )

    # Let BioOntologySearchClient get API key from settings
    bio_client = BioOntologySearchClient()

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

        # Search for a common term that both backends should find
        result = await match_ontology_concepts(
            finding_name="pneumonia",
            search_clients=[lance_client, bio_client],  # Use both real clients
            max_exact_matches=10,
        )

        # Verify we got results
        assert result is not None
        assert len(result.exact_matches) > 0, "Should find pneumonia concepts"

        # Check for no duplicate codes (deduplication should work)
        # result contains OntologySearchResult objects
        seen_codes = set()
        for code in result.exact_matches:
            code_id = code.concept_id
            assert code_id not in seen_codes, f"Duplicate code found: {code_id}"
            seen_codes.add(code_id)

        # Similar check for should_include
        for code in result.should_include:
            code_id = code.concept_id
            assert code_id not in seen_codes, f"Duplicate code found: {code_id}"
            seen_codes.add(code_id)

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow
