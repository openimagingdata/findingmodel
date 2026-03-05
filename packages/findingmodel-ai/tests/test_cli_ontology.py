"""Tests for `findingmodel-ai ontology search` CLI command."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from findingmodel_ai.cli import cli
from findingmodel_ai.search.bioontology import BioOntologySearchResult

MOCK_RESULT = BioOntologySearchResult(
    concept_id="http://purl.bioontology.org/ontology/SNOMEDCT/233604007",
    ontology="SNOMEDCT",
    pref_label="Pneumothorax",
    synonyms=["Pneumothorax (disorder)"],
    definition="Air in the pleural space.",
    semantic_types=["T047"],
    ui_link="https://bioportal.bioontology.org/ontologies/SNOMEDCT?p=classes&conceptid=233604007",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_client_with_results() -> AsyncMock:
    """A mock BioOntologySearchClient that returns one result."""
    client = AsyncMock()
    client.search_all_pages = AsyncMock(return_value=[MOCK_RESULT])
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def patched_settings_with_key() -> Generator[MagicMock, None, None]:
    """Patch settings so bioontology_api_key is present."""
    mock_settings = MagicMock()
    mock_settings.bioontology_api_key = MagicMock()
    mock_settings.bioontology_api_key.get_secret_value.return_value = "fake-key"
    with patch("findingmodel_ai.cli.settings", mock_settings):
        yield mock_settings


def test_ontology_search_no_api_key(runner: CliRunner) -> None:
    """Command exits with error when API key is not configured."""
    with patch("findingmodel_ai.cli.settings") as mock_settings:
        mock_settings.bioontology_api_key = None
        result = runner.invoke(cli, ["ontology", "search", "pneumothorax"])

    assert result.exit_code == 1
    assert "BIOONTOLOGY_API_KEY" in result.output


def test_ontology_search_no_results(runner: CliRunner, patched_settings_with_key: MagicMock) -> None:
    """Command prints 'No results' message when API returns empty list."""
    mock_client = AsyncMock()
    mock_client.search_all_pages = AsyncMock(return_value=[])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("findingmodel_ai.search.bioontology.BioOntologySearchClient", return_value=mock_client):
        result = runner.invoke(cli, ["ontology", "search", "xyzzy"])

    assert result.exit_code == 0
    assert "No results" in result.output


def test_ontology_search_renders_table(
    runner: CliRunner, patched_settings_with_key: MagicMock, mock_client_with_results: AsyncMock
) -> None:
    """Command renders a Rich table with result rows."""
    with patch("findingmodel_ai.search.bioontology.BioOntologySearchClient", return_value=mock_client_with_results):
        result = runner.invoke(cli, ["ontology", "search", "pneumothorax"])

    assert result.exit_code == 0
    assert "Pneumothorax" in result.output
    assert "SNOMEDCT" in result.output


def test_ontology_search_custom_ontology(
    runner: CliRunner, patched_settings_with_key: MagicMock, mock_client_with_results: AsyncMock
) -> None:
    """--ontology option passes custom list to client."""
    with patch("findingmodel_ai.search.bioontology.BioOntologySearchClient", return_value=mock_client_with_results):
        result = runner.invoke(cli, ["ontology", "search", "pneumothorax", "--ontology", "RADLEX"])

    assert result.exit_code == 0
    mock_client_with_results.search_all_pages.assert_awaited_once()
    call_kwargs = mock_client_with_results.search_all_pages.call_args.kwargs
    assert call_kwargs["ontologies"] == ["RADLEX"]


def test_ontology_search_semantic_type(
    runner: CliRunner, patched_settings_with_key: MagicMock, mock_client_with_results: AsyncMock
) -> None:
    """--semantic-type option passes type list to client."""
    with patch("findingmodel_ai.search.bioontology.BioOntologySearchClient", return_value=mock_client_with_results):
        result = runner.invoke(
            cli,
            ["ontology", "search", "pneumothorax", "--semantic-type", "T047", "--semantic-type", "T048"],
        )

    assert result.exit_code == 0
    call_kwargs = mock_client_with_results.search_all_pages.call_args.kwargs
    assert call_kwargs["semantic_types"] == ["T047", "T048"]


def test_ontology_search_exact_flag(
    runner: CliRunner, patched_settings_with_key: MagicMock, mock_client_with_results: AsyncMock
) -> None:
    """--exact flag passes require_exact_match=True to client."""
    with patch("findingmodel_ai.search.bioontology.BioOntologySearchClient", return_value=mock_client_with_results):
        result = runner.invoke(cli, ["ontology", "search", "pneumothorax", "--exact"])

    assert result.exit_code == 0
    call_kwargs = mock_client_with_results.search_all_pages.call_args.kwargs
    assert call_kwargs["require_exact_match"] is True


@pytest.mark.callout
def test_ontology_search_live(runner: CliRunner) -> None:
    """Integration test — hits real BioOntology API. Requires BIOONTOLOGY_API_KEY."""
    result = runner.invoke(cli, ["ontology", "search", "pneumothorax", "--max-results", "5"])
    assert result.exit_code == 0
    assert "pneumothorax" in result.output.lower()


@pytest.mark.callout
def test_ontology_search_live_semantic_type(runner: CliRunner) -> None:
    """Integration test — semantic type filter reduces results to diseases."""
    result = runner.invoke(
        cli,
        ["ontology", "search", "pneumothorax", "--max-results", "5", "--semantic-type", "T047"],
    )
    assert result.exit_code == 0
    assert "pneumothorax" in result.output.lower()
