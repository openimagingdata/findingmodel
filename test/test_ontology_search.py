"""Tests for ontology search clients, Protocol compliance, and concept matching workflows."""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from findingmodel.config import settings
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
from findingmodel.tools.ontology_search import (
    BioOntologySearchClient,
    BioOntologySearchResult,
    OntologySearchResult,
    rerank_with_cohere,
)

# ==============================================================================
# BioOntology Protocol & Client Tests
# ==============================================================================


def test_bioontology_implements_protocol() -> None:
    """Test that BioOntologySearchClient implements OntologySearchProtocol."""
    # Skip if no API key
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    client = BioOntologySearchClient()

    # Verify required methods exist
    assert hasattr(client, "search")
    assert hasattr(client, "__aenter__")
    assert hasattr(client, "__aexit__")

    # Verify search method signature
    sig = inspect.signature(client.search)
    params = list(sig.parameters.keys())
    assert "queries" in params
    assert "max_results" in params
    assert "filter_anatomical" in params


@pytest.mark.asyncio
async def test_protocol_context_managers() -> None:
    """Test that BioOntology client works as async context manager."""
    # Skip if no API key
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    client = BioOntologySearchClient()
    async with client as ctx:
        assert ctx is client


def test_bioontology_search_result_from_api_response() -> None:
    """Test creating a search result from API response."""
    api_response = {
        "@id": "http://purl.bioontology.org/ontology/SNOMEDCT/233604007",
        "@type": "http://www.w3.org/2002/07/owl#Class",
        "prefLabel": "Pneumonia",
        "synonym": ["Pneumonia (disorder)", "Lung infection"],
        "definition": "Inflammation of lung tissue",
        "semanticType": ["T047"],
        "links": {
            "ontology": "https://data.bioontology.org/ontologies/SNOMEDCT",
            "ui": "http://bioportal.bioontology.org/ontologies/SNOMEDCT?p=classes&conceptid=233604007",
        },
    }

    result = BioOntologySearchResult.from_api_response(api_response)

    assert result.concept_id == "http://purl.bioontology.org/ontology/SNOMEDCT/233604007"
    assert result.ontology == "SNOMEDCT"
    assert result.pref_label == "Pneumonia"
    assert result.synonyms == ["Pneumonia (disorder)", "Lung infection"]
    assert result.definition == "Inflammation of lung tissue"
    assert result.semantic_types == ["T047"]
    assert "bioportal.bioontology.org" in result.ui_link


def test_bioontology_search_result_minimal() -> None:
    """Test creating a search result with minimal fields."""
    api_response = {
        "@id": "http://purl.bioontology.org/ontology/RADLEX/RID28836",
        "prefLabel": "Consolidation",
        "links": {
            "ontology": "https://data.bioontology.org/ontologies/RADLEX",
            "ui": "http://bioportal.bioontology.org/ontologies/RADLEX?p=classes&conceptid=RID28836",
        },
    }

    result = BioOntologySearchResult.from_api_response(api_response)

    assert result.concept_id == "http://purl.bioontology.org/ontology/RADLEX/RID28836"
    assert result.ontology == "RADLEX"
    assert result.pref_label == "Consolidation"
    assert result.synonyms == []
    assert result.definition is None
    assert result.semantic_types == []


def test_bioontology_client_no_api_key() -> None:
    """Test that client raises error without API key."""
    # Temporarily clear the API key if it exists
    original_key = getattr(settings, "bioontology_api_key", None)
    try:
        if original_key:
            settings.bioontology_api_key = None
        with pytest.raises(ValueError, match="API key is required"):
            BioOntologySearchClient(api_key=None)
    finally:
        if original_key:
            settings.bioontology_api_key = original_key


def test_default_ontologies_limited() -> None:
    """Test that BioOntologySearchClient only uses 3 core medical ontologies by default.

    This test verifies we only use SNOMEDCT, RADLEX, and LOINC as default ontologies,
    ensuring we don't regress back to including other ontologies like GAMUTS, ICD10CM,
    or CPT which were removed for performance and relevance reasons.
    """
    expected_ontologies = ["SNOMEDCT", "RADLEX", "LOINC"]
    assert expected_ontologies == BioOntologySearchClient.DEFAULT_ONTOLOGIES


# ==============================================================================
# Cohere Reranking Tests
# ==============================================================================


def test_rerank_with_cohere_no_api_key() -> None:
    """Test that rerank_with_cohere returns original order when no API key is configured."""
    import asyncio

    # Mock settings to have no Cohere API key
    with patch("findingmodel.tools.ontology_search.settings.cohere_api_key", None):
        # Create test documents
        docs = [
            OntologySearchResult(concept_id="1", concept_text="heart", score=0.5, table_name="test"),
            OntologySearchResult(concept_id="2", concept_text="lung", score=0.8, table_name="test"),
            OntologySearchResult(concept_id="3", concept_text="liver", score=0.3, table_name="test"),
        ]

        # Run rerank_with_cohere
        result = asyncio.run(rerank_with_cohere("cardiac", docs))

        # Should return original order
        assert result == docs
        assert [r.concept_id for r in result] == ["1", "2", "3"]


def test_rerank_with_cohere_empty_documents() -> None:
    """Test that rerank_with_cohere handles empty document list."""
    import asyncio

    result = asyncio.run(rerank_with_cohere("test query", []))
    assert result == []


@pytest.mark.asyncio
async def test_rerank_with_cohere_with_mock_client() -> None:
    """Test rerank_with_cohere with a mocked Cohere client."""
    # Create test documents
    docs = [
        OntologySearchResult(concept_id="1", concept_text="heart", score=0.5, table_name="test"),
        OntologySearchResult(concept_id="2", concept_text="lung", score=0.8, table_name="test"),
        OntologySearchResult(concept_id="3", concept_text="liver", score=0.3, table_name="test"),
    ]

    # Create mock Cohere client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    # Simulate reranking: put doc 2 first, then 1, then 3
    mock_response.results = [
        MagicMock(index=1),  # lung
        MagicMock(index=0),  # heart
        MagicMock(index=2),  # liver
    ]
    mock_client.rerank = AsyncMock(return_value=mock_response)

    # Run rerank with mock client
    result = await rerank_with_cohere("lung disease", docs, client=mock_client)

    # Check reordering happened
    assert len(result) == 3
    assert result[0].concept_id == "2"  # lung first
    assert result[1].concept_id == "1"  # heart second
    assert result[2].concept_id == "3"  # liver third

    # Verify client was called correctly
    mock_client.rerank.assert_called_once_with(
        model="rerank-v3.5", query="lung disease", documents=["1: heart", "2: lung", "3: liver"], top_n=None
    )


@pytest.mark.asyncio
async def test_rerank_with_cohere_top_n() -> None:
    """Test rerank_with_cohere with top_n parameter."""
    # Create test documents
    docs = [
        OntologySearchResult(concept_id=str(i), concept_text=f"concept_{i}", score=0.5, table_name="test")
        for i in range(5)
    ]

    # Create mock Cohere client
    mock_client = AsyncMock()
    mock_response = MagicMock()
    # Return only top 2
    mock_response.results = [
        MagicMock(index=2),
        MagicMock(index=4),
    ]
    mock_client.rerank = AsyncMock(return_value=mock_response)

    # Run rerank with top_n=2
    result = await rerank_with_cohere("test", docs, client=mock_client, top_n=2)

    # Should return only 2 results
    assert len(result) == 2
    assert result[0].concept_id == "2"
    assert result[1].concept_id == "4"

    # Verify top_n was passed to client
    mock_client.rerank.assert_called_once()
    assert mock_client.rerank.call_args.kwargs["top_n"] == 2


# ==============================================================================
# Query Generation Tests
# ==============================================================================


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


# ==============================================================================
# Search Execution Tests
# ==============================================================================


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


# ==============================================================================
# Exact Match Tests
# ==============================================================================


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


# ==============================================================================
# Categorization Tests
# ==============================================================================


def test_categorization_agent_creation() -> None:
    """Test that agent is created properly."""
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


# ==============================================================================
# Integration Tests (match_ontology_concepts)
# ==============================================================================


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


# ==============================================================================
# Callout/Integration Tests
# ==============================================================================


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_search_pneumonia() -> None:
    """Integration test: search for pneumonia concepts."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    async with BioOntologySearchClient() as client:
        results = await client.search_bioontology(
            query="pneumonia",
            ontologies=["SNOMEDCT", "RADLEX"],
            page_size=10,
        )

        assert results.query == "pneumonia"
        assert results.total_count > 0
        assert len(results.results) > 0

        # Check first result has expected fields
        first_result = results.results[0]
        assert first_result.concept_id
        assert first_result.ontology in ["SNOMEDCT", "RADLEX"]
        assert first_result.pref_label
        assert first_result.ui_link

        # Check that we get results from both ontologies if available
        ontologies_found = {r.ontology for r in results.results}
        assert len(ontologies_found) > 0  # At least one ontology represented


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_search_all_pages() -> None:
    """Integration test: search with pagination."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    async with BioOntologySearchClient() as client:
        results = await client.search_all_pages(
            query="fracture",
            ontologies=["SNOMEDCT"],
            max_results=25,  # Get more than one page
        )

        assert len(results) <= 25
        assert all(r.ontology == "SNOMEDCT" for r in results)
        assert all(
            "fracture" in r.pref_label.lower() or any("fracture" in s.lower() for s in r.synonyms) for r in results
        )


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_search_as_ontology_results() -> None:
    """Integration test: test conversion to OntologySearchResult format."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    async with BioOntologySearchClient() as client:
        results = await client.search_as_ontology_results(
            query="hepatic metastasis",
            ontologies=["SNOMEDCT", "RADLEX"],
            max_results=10,
        )

        assert len(results) <= 10
        # Check results are in OntologySearchResult format
        if results:
            first_result = results[0]
            assert hasattr(first_result, "concept_id")
            assert hasattr(first_result, "concept_text")
            assert hasattr(first_result, "table_name")
            assert hasattr(first_result, "score")


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_semantic_type_filter() -> None:
    """Integration test: test filtering by semantic type."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    async with BioOntologySearchClient() as client:
        results = await client.search_bioontology(
            query="liver",
            ontologies=["SNOMEDCT"],
            semantic_types=["T047"],  # Disease or syndrome
            page_size=10,
        )

        # Results should be disease-related liver concepts, not anatomical
        if results.results:
            # Check that results are disease-related
            for result in results.results[:5]:
                assert (
                    any(
                        keyword in result.pref_label.lower()
                        for keyword in ["disease", "disorder", "syndrome", "failure", "cirrhosis", "hepatitis"]
                    )
                    or "T047" in result.semantic_types
                )


@pytest.mark.callout
@pytest.mark.asyncio
async def test_rerank_with_cohere_integration() -> None:
    """Integration test for Cohere reranking (requires COHERE_API_KEY)."""
    if not getattr(settings, "cohere_api_key", None):
        pytest.skip("Cohere API key not configured")

    # Create test documents with intentionally mismatched order
    docs = [
        OntologySearchResult(concept_id="1", concept_text="liver disease", score=0.9, table_name="test"),
        OntologySearchResult(concept_id="2", concept_text="cardiac arrest", score=0.8, table_name="test"),
        OntologySearchResult(concept_id="3", concept_text="heart failure", score=0.7, table_name="test"),
        OntologySearchResult(concept_id="4", concept_text="myocardial infarction", score=0.6, table_name="test"),
    ]

    # Rerank with a cardiac-focused query
    result = await rerank_with_cohere("heart attack", docs, top_n=3)

    # Should return 3 results
    assert len(result) == 3

    # The cardiac-related concepts should rank higher than liver disease
    result_ids = [r.concept_id for r in result]
    assert "1" not in result_ids[:2]  # liver disease should not be in top 2


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_integration() -> None:
    """Integration test: Use real BioOntologySearchClient."""
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
