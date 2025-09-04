"""Tests for the ontology concept search tool."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from findingmodel.tools.ontology_concept_search import (
    CategorizationContext,
    CategorizedConcepts,
    QueryTerms,
    create_categorization_agent,
    create_query_generator_agent,
    ensure_exact_matches_post_process,
    execute_ontology_search,
    generate_query_terms,
    search_ontology_concepts,
)
from findingmodel.tools.ontology_search import OntologySearchResult

# Prevent accidental API calls during testing
models.ALLOW_MODEL_REQUESTS = False


# QueryTerms Model Tests


def test_query_terms_creation() -> None:
    """Test creating QueryTerms with basic functionality."""
    terms = QueryTerms(
        primary_term="pneumonia", synonyms=["lung infection", "pneumonitis"], related_terms=["bronchitis"]
    )

    assert terms.primary_term == "pneumonia"
    assert len(terms.synonyms) == 2
    assert len(terms.related_terms) == 1

    # Test all_terms property
    all_terms = terms.all_terms
    assert all_terms[0] == "pneumonia"  # Primary always first
    assert "lung infection" in all_terms
    assert "pneumonitis" in all_terms
    assert "bronchitis" in all_terms
    assert len(all_terms) == 4  # No duplicates


def test_query_terms_deduplication() -> None:
    """Test that QueryTerms properly deduplicates terms."""
    query_terms = QueryTerms(
        primary_term="Pulmonary Embolism",
        synonyms=["pulmonary embolism", "PE", "lung embolism"],  # Duplicate with different case
        related_terms=["PE", "embolism", "thromboembolism"],  # PE is duplicate
    )

    all_terms = query_terms.all_terms

    # Should not have duplicate "pulmonary embolism" (case insensitive)
    assert all_terms[0] == "Pulmonary Embolism"
    assert "pulmonary embolism" not in all_terms[1:]  # Not in rest of list

    # Should only have one "PE"
    pe_count = sum(1 for term in all_terms if term == "PE")
    assert pe_count == 1

    # Check expected unique terms
    assert "lung embolism" in all_terms
    assert "embolism" in all_terms
    assert "thromboembolism" in all_terms


def test_query_terms_max_limits() -> None:
    """Test that field limits are enforced."""

    # Test synonym limit (max 10)
    many_synonyms = [f"synonym_{i}" for i in range(15)]

    with pytest.raises(ValueError, match="List should have at most 10 items"):
        QueryTerms(primary_term="test", synonyms=many_synonyms, related_terms=[])

    # Test related terms limit (max 5)
    many_related = [f"related_{i}" for i in range(10)]

    with pytest.raises(ValueError, match="List should have at most 5 items"):
        QueryTerms(primary_term="test", synonyms=[], related_terms=many_related)

    # Test valid limits
    query_terms = QueryTerms(
        primary_term="test",
        synonyms=[f"syn_{i}" for i in range(10)],  # Exactly 10
        related_terms=[f"rel_{i}" for i in range(5)],  # Exactly 5
    )

    assert len(query_terms.synonyms) == 10
    assert len(query_terms.related_terms) == 5
    assert len(query_terms.all_terms) == 16  # 1 primary + 10 synonyms + 5 related


# Generate Query Terms Tests


@pytest.mark.asyncio
async def test_generate_query_terms_single_word() -> None:
    """Test query generation for single word."""
    # Mock the query generator agent to avoid API calls
    with patch("findingmodel.tools.ontology_concept_search.create_query_generator_agent") as mock_create:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = QueryTerms(
            primary_term="pneumonia", synonyms=["lung infection", "pneumonitis"], related_terms=["bronchitis"]
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        result = await generate_query_terms("pneumonia")

        assert result.primary_term == "pneumonia"
        assert "lung infection" in result.synonyms


@pytest.mark.asyncio
async def test_generate_query_terms_with_description() -> None:
    """Test query generation with description."""
    # Mock the query generator agent
    with patch("findingmodel.tools.ontology_concept_search.create_query_generator_agent") as mock_create:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = QueryTerms(
            primary_term="pneumonia",
            synonyms=["lung infection"],
            related_terms=["lung pneumonia"],  # Should infer anatomy
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        result = await generate_query_terms("pneumonia", "lung inflammation")

        assert result.primary_term == "pneumonia"
        # Should have anatomy inference
        assert "lung pneumonia" in result.all_terms


def test_create_query_generator_agent() -> None:
    """Test that the query generator agent is created correctly."""
    with patch("findingmodel.tools.ontology_concept_search.get_openai_model") as mock_model:
        agent = create_query_generator_agent()

        # Should get the default model
        mock_model.assert_called_once()

        # Check agent configuration
        assert agent._output_type == QueryTerms

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
    mock_client.search_parallel = AsyncMock(return_value=mock_results)

    # Create query terms
    query_terms = QueryTerms(primary_term="pneumonia", synonyms=["lung infection"], related_terms=[])

    # Execute search
    results = await execute_ontology_search(query_terms=query_terms, client=mock_client, exclude_anatomical=True)

    # Verify client was called with correct parameters
    mock_client.search_parallel.assert_called_once()
    call_args = mock_client.search_parallel.call_args
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

    query_terms = QueryTerms(primary_term="pneumonia", synonyms=[], related_terms=[])

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

    query_terms = QueryTerms(primary_term="test", synonyms=[], related_terms=[])

    # Process - should not exceed limit
    corrected = ensure_exact_matches_post_process(output, search_results, query_terms)

    assert len(corrected.exact_matches) == 5


# Categorization Agent Tests


def test_categorization_agent_creation() -> None:
    """Test that agent is created properly."""
    with patch("findingmodel.tools.ontology_concept_search.get_openai_model") as mock_model:
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
        query_terms=QueryTerms(primary_term="pneumonia", synonyms=[], related_terms=[]),
    )

    # Run with override
    with agent.override(model=test_model):
        result = await agent.run("Test prompt", deps=context)

        # Verify we got our test output
        assert result.output.exact_matches == ["RID5350"]
        assert result.output.rationale == "Test categorization"


# Main Orchestration Tests


@patch("findingmodel.tools.ontology_concept_search.OntologySearchClient")
@pytest.mark.asyncio
async def test_search_ontology_concepts_integration(mock_client_class: Mock) -> None:
    """Test the complete workflow."""
    # Setup mock client
    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = MagicMock()
    mock_client.search_parallel = AsyncMock(
        return_value=[
            OntologySearchResult(concept_id="RID5350", concept_text="pneumonia", score=0.95, table_name="radlex"),
        ]
    )
    mock_client_class.return_value = mock_client

    # Mock the categorization agent
    with patch("findingmodel.tools.ontology_concept_search.create_categorization_agent") as mock_create:
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = CategorizedConcepts(
            exact_matches=["RID5350"], should_include=[], marginal=[], rationale="Test"
        )
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_create.return_value = mock_agent

        # Run
        result = await search_ontology_concepts(finding_name="pneumonia", finding_description="lung infection")

        # Verify cleanup
        mock_client.connect.assert_called_once()
        mock_client.disconnect.assert_called_once()

        # Check result
        assert result is not None
        assert len(result.exact_matches) == 1


@patch("findingmodel.tools.ontology_concept_search.OntologySearchClient")
@pytest.mark.asyncio
async def test_search_cleanup_on_error(mock_client_class: Mock) -> None:
    """Test that client disconnects even on error."""
    # Setup mock client that fails
    mock_client = MagicMock()
    mock_client.connect = AsyncMock(side_effect=ConnectionError("Connection failed"))
    mock_client.disconnect = MagicMock()
    mock_client_class.return_value = mock_client

    # Run and expect exception
    with pytest.raises(ConnectionError, match="Connection failed"):
        await search_ontology_concepts("test")

    # Verify disconnect was still called
    mock_client.disconnect.assert_called_once()
