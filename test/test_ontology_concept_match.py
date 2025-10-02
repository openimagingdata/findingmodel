"""Tests for the ontology concept search tool (simplified BioOntology-only version)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
    from pydantic_ai.models.test import TestModel

    with patch("findingmodel.tools.ontology_concept_match.get_openai_model") as mock_model:
        # Mock the get_openai_model to return a TestModel instead of trying to create a real OpenAI client
        test_model = TestModel()
        mock_model.return_value = test_model

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
async def test_execute_ontology_search_with_cohere_enabled() -> None:
    """Test execute_ontology_search when Cohere reranking is enabled and configured."""
    # Mock the BioOntologySearchClient and settings
    mock_client = MagicMock()
    mock_search_results = [
        OntologySearchResult(concept_id="1", concept_text="pneumonia", score=0.95, table_name="radlex"),
        OntologySearchResult(concept_id="2", concept_text="Pneumonia", score=0.93, table_name="snomedct"),
        OntologySearchResult(concept_id="3", concept_text="viral pneumonia", score=0.8, table_name="radlex"),
    ]
    mock_client.search_as_ontology_results = AsyncMock(return_value=mock_search_results)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock rerank_with_cohere to return the same documents (for testing)
    mock_reranked_results = mock_search_results.copy()

    with (
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
        patch("findingmodel.tools.ontology_concept_match.rerank_with_cohere", new_callable=AsyncMock) as mock_rerank,
    ):
        # Configure settings for Cohere enabled
        mock_settings.bioontology_api_key = "test-key"
        mock_settings.use_cohere_with_ontology_concept_match = True
        mock_settings.cohere_api_key = "test-cohere-key"

        # Configure rerank mock to return the same documents
        mock_rerank.return_value = mock_reranked_results

        # Execute search with query terms
        query_terms = ["pneumonia", "lung infection"]
        results = await execute_ontology_search(query_terms=query_terms, exclude_anatomical=True)

        # Verify client was created and used
        mock_client.__aenter__.assert_called_once()
        mock_client.search_as_ontology_results.assert_called_once()

        # Verify Cohere reranking was called
        mock_rerank.assert_called_once()

        # Verify the Cohere query format is correct
        call_args = mock_rerank.call_args
        expected_query = (
            "What is the correct medical ontology term to represent 'pneumonia' (alternates: lung infection)?"
        )
        assert call_args.kwargs["query"] == expected_query

        # Verify other rerank parameters
        assert call_args.kwargs["documents"] == mock_search_results
        assert call_args.kwargs["retry_attempts"] == 1

        # Check results
        assert len(results) == 3
        assert results[0].concept_text == "pneumonia"  # Should be normalized


@pytest.mark.asyncio
async def test_execute_ontology_search_with_cohere_disabled() -> None:
    """Test execute_ontology_search when Cohere reranking is disabled via config."""
    # Mock the BioOntologySearchClient and settings
    mock_client = MagicMock()
    mock_search_results = [
        OntologySearchResult(concept_id="1", concept_text="pneumonia", score=0.95, table_name="radlex"),
        OntologySearchResult(concept_id="2", concept_text="Pneumonia", score=0.93, table_name="snomedct"),
    ]
    mock_client.search_as_ontology_results = AsyncMock(return_value=mock_search_results)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
        patch("findingmodel.tools.ontology_concept_match.rerank_with_cohere", new_callable=AsyncMock) as mock_rerank,
    ):
        # Configure settings with Cohere disabled
        mock_settings.bioontology_api_key = "test-key"
        mock_settings.use_cohere_with_ontology_concept_match = False  # Disabled
        mock_settings.cohere_api_key = "test-cohere-key"

        # Execute search
        query_terms = ["pneumonia", "lung infection"]
        results = await execute_ontology_search(query_terms=query_terms, exclude_anatomical=True)

        # Verify client was created and used
        mock_client.__aenter__.assert_called_once()
        mock_client.search_as_ontology_results.assert_called_once()

        # Verify Cohere reranking was NOT called
        mock_rerank.assert_not_called()

        # Check results
        assert len(results) == 2
        assert results[0].concept_text == "pneumonia"  # Should be normalized


@pytest.mark.asyncio
async def test_execute_ontology_search_with_cohere_no_api_key() -> None:
    """Test execute_ontology_search when Cohere config is enabled but no API key is provided."""
    # Mock the BioOntologySearchClient and settings
    mock_client = MagicMock()
    mock_search_results = [
        OntologySearchResult(concept_id="1", concept_text="pneumonia", score=0.95, table_name="radlex"),
        OntologySearchResult(concept_id="2", concept_text="lung pneumonia", score=0.88, table_name="snomedct"),
    ]
    mock_client.search_as_ontology_results = AsyncMock(return_value=mock_search_results)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
        patch("findingmodel.tools.ontology_concept_match.rerank_with_cohere", new_callable=AsyncMock) as mock_rerank,
    ):
        # Configure settings with Cohere enabled but no API key
        mock_settings.bioontology_api_key = "test-key"
        mock_settings.use_cohere_with_ontology_concept_match = True  # Enabled
        mock_settings.cohere_api_key = None  # No API key

        # Execute search
        query_terms = ["pneumonia", "lung infection"]
        results = await execute_ontology_search(query_terms=query_terms, exclude_anatomical=True)

        # Verify client was created and used
        mock_client.__aenter__.assert_called_once()
        mock_client.search_as_ontology_results.assert_called_once()

        # Verify Cohere reranking was NOT called (no API key)
        mock_rerank.assert_not_called()

        # Check results
        assert len(results) == 2
        assert results[0].concept_text == "pneumonia"  # Should be normalized


@pytest.mark.asyncio
async def test_execute_ontology_search_with_cohere_single_query_term() -> None:
    """Test execute_ontology_search with Cohere enabled and single query term (no alternates)."""
    # Mock the BioOntologySearchClient and settings
    mock_client = MagicMock()
    mock_search_results = [
        OntologySearchResult(concept_id="1", concept_text="pneumonia", score=0.95, table_name="radlex"),
    ]
    mock_client.search_as_ontology_results = AsyncMock(return_value=mock_search_results)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
        patch("findingmodel.tools.ontology_concept_match.rerank_with_cohere", new_callable=AsyncMock) as mock_rerank,
    ):
        # Configure settings for Cohere enabled
        mock_settings.bioontology_api_key = "test-key"
        mock_settings.use_cohere_with_ontology_concept_match = True
        mock_settings.cohere_api_key = "test-cohere-key"

        # Configure rerank mock
        mock_rerank.return_value = mock_search_results

        # Execute search with single query term
        query_terms = ["pneumonia"]
        results = await execute_ontology_search(query_terms=query_terms, exclude_anatomical=True)

        # Verify Cohere reranking was called
        mock_rerank.assert_called_once()

        # Verify the Cohere query format for single term (no alternates)
        call_args = mock_rerank.call_args
        expected_query = "What is the correct medical ontology term to represent 'pneumonia'?"
        assert call_args.kwargs["query"] == expected_query

        # Verify other parameters
        assert call_args.kwargs["documents"] == mock_search_results
        assert call_args.kwargs["retry_attempts"] == 1

        # Check results
        assert len(results) == 1
        assert results[0].concept_text == "pneumonia"


@pytest.mark.asyncio
async def test_execute_ontology_search() -> None:
    """Test executing search with filtering using BioOntology API."""
    # Mock the BioOntologySearchClient and settings
    mock_client = MagicMock()
    mock_results = [
        OntologySearchResult(concept_id="1", concept_text="pneumonia", score=0.95, table_name="radlex"),
        OntologySearchResult(concept_id="2", concept_text="Pneumonia", score=0.93, table_name="snomedct"),
        OntologySearchResult(concept_id="3", concept_text="viral pneumonia", score=0.8, table_name="radlex"),
    ]
    mock_client.search_as_ontology_results = AsyncMock(return_value=mock_results)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
    ):
        # Mock that API key is configured
        mock_settings.bioontology_api_key = "test-key"
        # Add config mocks to prevent Cohere issues
        mock_settings.use_cohere_with_ontology_concept_match = False
        mock_settings.cohere_api_key = None

        # Create query terms
        query_terms = ["pneumonia", "lung infection"]

        # Execute search
        results = await execute_ontology_search(query_terms=query_terms, exclude_anatomical=True)

        # Verify client was created and used
        mock_client.__aenter__.assert_called_once()
        mock_client.search_as_ontology_results.assert_called_once()

        # Verify search was called with correct parameters
        call_args = mock_client.search_as_ontology_results.call_args
        assert call_args.kwargs["query"] == "pneumonia OR lung infection"
        assert call_args.kwargs["ontologies"] is None  # Default ontologies used
        assert "max_results" in call_args.kwargs

        # Check results
        assert len(results) == 3
        # Results should be normalized
        assert results[0].concept_text == "pneumonia"


@pytest.mark.asyncio
async def test_execute_ontology_search_with_custom_ontologies() -> None:
    """Test execute_ontology_search with custom ontologies parameter."""
    # Mock the BioOntologySearchClient and settings
    mock_client = MagicMock()
    mock_results = [
        OntologySearchResult(concept_id="SCT-123", concept_text="pneumonia", score=0.95, table_name="snomedct"),
        OntologySearchResult(concept_id="GAMUTS-456", concept_text="lung infiltrate", score=0.88, table_name="gamuts"),
    ]
    mock_client.search_as_ontology_results = AsyncMock(return_value=mock_results)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
    ):
        # Mock that API key is configured
        mock_settings.bioontology_api_key = "test-key"
        # Add config mocks to prevent Cohere issues
        mock_settings.use_cohere_with_ontology_concept_match = False
        mock_settings.cohere_api_key = None

        # Test with custom ontologies
        query_terms = ["pneumonia", "lung infection"]
        custom_ontologies = ["SNOMEDCT", "GAMUTS"]

        results = await execute_ontology_search(
            query_terms=query_terms, exclude_anatomical=True, ontologies=custom_ontologies
        )

        # Verify client was created and used
        mock_client.__aenter__.assert_called_once()
        mock_client.search_as_ontology_results.assert_called_once()

        # Verify search was called with correct parameters including custom ontologies
        call_args = mock_client.search_as_ontology_results.call_args
        assert call_args.kwargs["query"] == "pneumonia OR lung infection"
        assert call_args.kwargs["ontologies"] == ["SNOMEDCT", "GAMUTS"]
        assert "max_results" in call_args.kwargs

        # Check results
        assert len(results) == 2
        assert results[0].concept_id == "SCT-123"
        assert results[1].concept_id == "GAMUTS-456"


@pytest.mark.asyncio
async def test_execute_ontology_search_with_none_ontologies() -> None:
    """Test execute_ontology_search with ontologies=None (uses defaults)."""
    # Mock the BioOntologySearchClient and settings
    mock_client = MagicMock()
    mock_results = [
        OntologySearchResult(concept_id="RID5350", concept_text="pneumonia", score=0.95, table_name="radlex"),
    ]
    mock_client.search_as_ontology_results = AsyncMock(return_value=mock_results)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
    ):
        # Mock that API key is configured
        mock_settings.bioontology_api_key = "test-key"
        # Add config mocks to prevent Cohere issues
        mock_settings.use_cohere_with_ontology_concept_match = False
        mock_settings.cohere_api_key = None

        # Test with ontologies=None (default behavior)
        query_terms = ["pneumonia"]

        results = await execute_ontology_search(
            query_terms=query_terms,
            exclude_anatomical=True,
            ontologies=None,  # Explicitly pass None
        )

        # Verify search was called with ontologies=None (uses defaults)
        call_args = mock_client.search_as_ontology_results.call_args
        assert call_args.kwargs["ontologies"] is None

        # Check results
        assert len(results) == 1
        assert results[0].concept_id == "RID5350"


@pytest.mark.asyncio
async def test_execute_ontology_search_missing_api_key() -> None:
    """Test that execute_ontology_search raises ValueError when BioOntology API key is not configured."""
    with patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings:
        # Mock that API key is not configured
        mock_settings.bioontology_api_key = None

        # Execute search should raise ValueError
        with pytest.raises(ValueError, match="BioOntology API key is required"):
            await execute_ontology_search(query_terms=["test"])


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
    from pydantic_ai.models.test import TestModel

    with patch("findingmodel.tools.ontology_concept_match.get_openai_model") as mock_model:
        # Mock the get_openai_model to return a TestModel instead of trying to create a real OpenAI client
        test_model = TestModel()
        mock_model.return_value = test_model

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
    """Test the complete workflow using BioOntology API."""
    # Create a mock client
    mock_client = MagicMock()
    mock_client.search_as_ontology_results = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="RID5350", concept_text="pneumonia", score=0.95, table_name="radlex"),
        ]
    )
    # Mock the async context manager behavior
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock the categorization agent
    with (
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_create,
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
    ):
        # Mock that API key is configured
        mock_settings.bioontology_api_key = "test-key"
        # Add config mocks to prevent Cohere issues
        mock_settings.use_cohere_with_ontology_concept_match = False
        mock_settings.cohere_api_key = None

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = CategorizedConcepts(
            exact_matches=["RID5350"], should_include=[], marginal=[], rationale="Test"
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        # Run the simplified API
        result = await match_ontology_concepts(
            finding_name="pneumonia",
            finding_description="lung infection",
        )

        # Verify context manager was used
        mock_client.__aenter__.assert_called_once()
        mock_client.__aexit__.assert_called_once()

        # Check result
        assert result is not None
        assert len(result.exact_matches) == 1


@pytest.mark.asyncio
async def test_match_ontology_concepts_with_custom_ontologies() -> None:
    """Test match_ontology_concepts with custom ontologies parameter."""
    # Create a mock client
    mock_client = MagicMock()
    mock_client.search_as_ontology_results = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="RLX-123", concept_text="pneumonia", score=0.95, table_name="radlex"),
            OntologySearchResult(concept_id="LOINC-456", concept_text="lung pneumonia", score=0.88, table_name="loinc"),
        ]
    )
    # Mock the async context manager behavior
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock the categorization agent
    with (
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_create,
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
    ):
        # Mock that API key is configured
        mock_settings.bioontology_api_key = "test-key"
        # Add config mocks to prevent Cohere issues
        mock_settings.use_cohere_with_ontology_concept_match = False
        mock_settings.cohere_api_key = None

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = CategorizedConcepts(
            exact_matches=["RLX-123"],
            should_include=["LOINC-456"],
            marginal=[],
            rationale="Test categorization with custom ontologies",
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        # Test with custom ontologies
        custom_ontologies = ["RADLEX", "LOINC"]
        result = await match_ontology_concepts(
            finding_name="pneumonia", finding_description="lung infection", ontologies=custom_ontologies
        )

        # Verify context manager was used
        mock_client.__aenter__.assert_called_once()
        mock_client.__aexit__.assert_called_once()

        # Verify search was called with custom ontologies
        mock_client.search_as_ontology_results.assert_called_once()
        call_args = mock_client.search_as_ontology_results.call_args
        assert call_args.kwargs["ontologies"] == ["RADLEX", "LOINC"]

        # Check result
        assert result is not None
        assert len(result.exact_matches) == 1
        assert len(result.should_include) == 1
        assert result.exact_matches[0].concept_id == "RLX-123"
        assert result.should_include[0].concept_id == "LOINC-456"


@pytest.mark.asyncio
async def test_match_ontology_concepts_with_none_ontologies() -> None:
    """Test match_ontology_concepts with ontologies=None (uses defaults)."""
    # Create a mock client
    mock_client = MagicMock()
    mock_client.search_as_ontology_results = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="RID5350", concept_text="pneumonia", score=0.95, table_name="radlex"),
        ]
    )
    # Mock the async context manager behavior
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock the categorization agent
    with (
        patch("findingmodel.tools.ontology_concept_match.create_categorization_agent") as mock_create,
        patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings,
        patch("findingmodel.tools.ontology_concept_match.BioOntologySearchClient", return_value=mock_client),
    ):
        # Mock that API key is configured
        mock_settings.bioontology_api_key = "test-key"
        # Add config mocks to prevent Cohere issues
        mock_settings.use_cohere_with_ontology_concept_match = False
        mock_settings.cohere_api_key = None

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = CategorizedConcepts(
            exact_matches=["RID5350"],
            should_include=[],
            marginal=[],
            rationale="Test categorization with default ontologies",
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        # Test with ontologies=None (default behavior)
        result = await match_ontology_concepts(
            finding_name="pneumonia",
            finding_description="lung infection",
            ontologies=None,  # Explicitly pass None
        )

        # Verify context manager was used
        mock_client.__aenter__.assert_called_once()
        mock_client.__aexit__.assert_called_once()

        # Verify search was called with ontologies=None (uses defaults)
        mock_client.search_as_ontology_results.assert_called_once()
        call_args = mock_client.search_as_ontology_results.call_args
        assert call_args.kwargs["ontologies"] is None

        # Check result
        assert result is not None
        assert len(result.exact_matches) == 1
        assert result.exact_matches[0].concept_id == "RID5350"


@pytest.mark.asyncio
async def test_match_ontology_concepts_missing_api_key() -> None:
    """Test that match_ontology_concepts raises ValueError when BioOntology API key is not configured."""
    with patch("findingmodel.tools.ontology_concept_match.settings") as mock_settings:
        # Mock that API key is not configured
        mock_settings.bioontology_api_key = None

        # Run and expect ValueError
        with pytest.raises(ValueError, match="BioOntology API key is required"):
            await match_ontology_concepts("test")


# Integration Tests (require real backends)


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_integration() -> None:
    """Integration test: Use real BioOntologySearchClient."""
    from findingmodel.config import settings

    # Skip if no BioOntology API key
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    # Test with real search - use simplified API (no search_clients parameter)
    result = await match_ontology_concepts(
        finding_name="fracture",
        finding_description="bone break",
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
