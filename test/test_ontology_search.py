"""Tests for ontology search clients and Protocol compliance."""

import inspect

import pytest
from pydantic_ai import models

from findingmodel.config import settings
from findingmodel.tools.ontology_search import (
    BioOntologySearchClient,
    BioOntologySearchResult,
)

# Prevent accidental API calls in tests
models.ALLOW_MODEL_REQUESTS = False


# Protocol Compliance Tests


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


# BioOntology Client Tests


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


# Integration Tests for BioOntology


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_search_pneumonia() -> None:
    """Integration test: search for pneumonia concepts."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

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

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_search_all_pages() -> None:
    """Integration test: search with pagination."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

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

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_search_as_ontology_results() -> None:
    """Integration test: test conversion to OntologySearchResult format."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

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

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow


@pytest.mark.callout
@pytest.mark.asyncio
async def test_bioontology_semantic_type_filter() -> None:
    """Integration test: test filtering by semantic type."""
    # Skip if no API key configured
    if not getattr(settings, "bioontology_api_key", None):
        pytest.skip("BioOntology API key not configured")

    # Temporarily enable model requests for integration test
    original_allow = models.ALLOW_MODEL_REQUESTS
    try:
        models.ALLOW_MODEL_REQUESTS = True

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

    finally:
        models.ALLOW_MODEL_REQUESTS = original_allow


# Cohere Reranking Tests


def test_rerank_with_cohere_no_api_key() -> None:
    """Test that rerank_with_cohere returns original order when no API key is configured."""
    import asyncio
    from unittest.mock import patch

    from findingmodel.tools.ontology_search import OntologySearchResult, rerank_with_cohere

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

    from findingmodel.tools.ontology_search import rerank_with_cohere

    result = asyncio.run(rerank_with_cohere("test query", []))
    assert result == []


@pytest.mark.asyncio
async def test_rerank_with_cohere_with_mock_client() -> None:
    """Test rerank_with_cohere with a mocked Cohere client."""
    from unittest.mock import AsyncMock, MagicMock

    from findingmodel.tools.ontology_search import OntologySearchResult, rerank_with_cohere

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
    from unittest.mock import AsyncMock, MagicMock

    from findingmodel.tools.ontology_search import OntologySearchResult, rerank_with_cohere

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


@pytest.mark.callout
@pytest.mark.asyncio
async def test_rerank_with_cohere_integration() -> None:
    """Integration test for Cohere reranking (requires COHERE_API_KEY)."""
    from findingmodel.tools.ontology_search import OntologySearchResult, rerank_with_cohere

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


def test_default_ontologies_limited() -> None:
    """Test that BioOntologySearchClient only uses 3 core medical ontologies by default.

    This test verifies we only use SNOMEDCT, RADLEX, and LOINC as default ontologies,
    ensuring we don't regress back to including other ontologies like GAMUTS, ICD10CM,
    or CPT which were removed for performance and relevance reasons.
    """
    expected_ontologies = ["SNOMEDCT", "RADLEX", "LOINC"]
    assert expected_ontologies == BioOntologySearchClient.DEFAULT_ONTOLOGIES
