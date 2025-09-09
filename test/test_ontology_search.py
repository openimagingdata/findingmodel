"""Tests for ontology search clients and Protocol compliance."""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import models

from findingmodel.config import settings
from findingmodel.tools.ontology_search import (
    BioOntologySearchClient,
    BioOntologySearchResult,
    LanceDBOntologySearchClient,
    OntologySearchResult,
)

# Prevent accidental API calls in tests
models.ALLOW_MODEL_REQUESTS = False


# Protocol Compliance Tests


def test_lancedb_implements_protocol() -> None:
    """Test that LanceDBOntologySearchClient implements OntologySearchProtocol."""
    client = LanceDBOntologySearchClient()

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
    """Test that both clients work as async context managers."""
    # Test LanceDBOntologySearchClient
    client = LanceDBOntologySearchClient()
    client.connect = AsyncMock()
    client.disconnect = MagicMock()

    async with client as ctx:
        assert ctx is client
        client.connect.assert_called_once()

    client.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_protocol_search_method_interface() -> None:
    """Test that search methods have compatible interfaces."""
    # Create mock client
    client = LanceDBOntologySearchClient()

    # Mock the underlying search_parallel method
    mock_results = [OntologySearchResult(concept_id="TEST1", concept_text="test concept", score=0.9, table_name="test")]
    client.search_parallel = AsyncMock(return_value=mock_results)

    # Call through Protocol interface
    results = await client.search(queries=["test"], max_results=10, filter_anatomical=True)

    # Verify it called search_parallel with mapped parameters
    client.search_parallel.assert_called_once()
    call_args = client.search_parallel.call_args
    assert call_args.kwargs["queries"] == ["test"]
    assert call_args.kwargs["filter_anatomical"] is True

    # Verify results
    assert results == mock_results


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
