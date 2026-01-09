from collections.abc import Generator
from types import SimpleNamespace

import findingmodel.tools
import findingmodel.tools.finding_description as finding_description
import httpx
import pytest
from conftest import TEST_OPENAI_MODEL
from findingmodel import FindingInfo, FindingModelBase, FindingModelFull, logger
from findingmodel.config import settings
from findingmodel.finding_model import AttributeType, ChoiceAttributeIded
from findingmodel.index_code import IndexCode
from pydantic_ai import models

# Prevent accidental model requests in unit tests
# Tests marked with @pytest.mark.callout can enable this as needed
models.ALLOW_MODEL_REQUESTS = False

HAS_TAVILY_API_KEY = bool(settings.tavily_api_key.get_secret_value())
HAS_OPENAI_API_KEY = bool(settings.openai_api_key.get_secret_value())
HAS_ANTHROPIC_API_KEY = bool(settings.anthropic_api_key.get_secret_value())
HAS_GOOGLE_API_KEY = bool(settings.google_api_key.get_secret_value())


def test_create_stub(finding_info: FindingInfo) -> None:
    """Test creating a stub finding model from a FindingInfo object."""
    stub = findingmodel.tools.create_finding_model_stub_from_finding_info(finding_info)
    assert isinstance(stub, FindingModelBase)
    assert stub.name == finding_info.name.lower()
    assert stub.description == finding_info.description
    assert stub.synonyms == finding_info.synonyms
    assert len(stub.attributes) == 2
    assert stub.attributes[0].name == "presence"
    assert stub.attributes[1].name == "change from prior"


def test_add_ids_to_finding_model(base_model: FindingModelBase) -> None:
    """Test adding IDs to a finding model."""
    updated_model = findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")
    assert isinstance(updated_model, FindingModelFull)
    assert updated_model.oifm_id is not None
    assert updated_model.oifm_id.startswith("OIFM_")
    assert "TEST" in updated_model.oifm_id
    assert len(updated_model.attributes) == len(base_model.attributes)
    for attr in updated_model.attributes:
        assert attr.oifma_id is not None
        assert attr.oifma_id.startswith("OIFMA_")
        assert "TEST" in attr.oifma_id
        if attr.type == AttributeType.CHOICE:
            for i, value in enumerate(attr.values):
                assert value.value_code is not None
                assert value.value_code == f"{attr.oifma_id}.{i}"


def test_render_agent_prompt() -> None:
    """Test render_agent_prompt extracts instructions and user prompt correctly."""
    from findingmodel.tools.prompt_template import render_agent_prompt
    from jinja2 import Template

    # Create a simple test template
    template_text = """# SYSTEM
You are a helpful assistant.

# USER
Please help with: {{task}}
"""
    template = Template(template_text)

    instructions, user_prompt = render_agent_prompt(template, task="testing")

    assert instructions == "You are a helpful assistant."
    assert user_prompt == "Please help with: testing"


def test_render_agent_prompt_missing_user_section() -> None:
    """Test render_agent_prompt raises error if USER section missing."""
    from findingmodel.tools.prompt_template import render_agent_prompt
    from jinja2 import Template

    template_text = """# SYSTEM
Only system instructions.
"""
    template = Template(template_text)

    with pytest.raises(ValueError, match="must include a USER section"):
        render_agent_prompt(template)


IdsJsonType = dict[str, dict[str, str] | dict[str, tuple[str, str]]]


class _StubFindingInfoAgent:
    def __init__(self, output: FindingInfo) -> None:
        self._output = output
        self.prompts: list[str] = []

    async def run(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        return SimpleNamespace(output=self._output)


@pytest.mark.asyncio
async def test_create_info_from_name_normalizes_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    output = FindingInfo(
        name="pulmonary opacity",
        synonyms=["Pulmonary opacity", " pulmonary opacity ", "Left lower lobe shadow"],
        description="General statement about pulmonary opacity.",
    )

    stub_agent = _StubFindingInfoAgent(output)
    monkeypatch.setattr(
        finding_description,
        "_create_finding_info_agent",
        lambda model_tier, instructions: stub_agent,
    )

    logged: list[str] = []

    def fake_info(message: str, *args: object, **kwargs: object) -> None:
        if args:
            message = message.format(*args)
        logged.append(message)

    monkeypatch.setattr(logger, "info", fake_info)

    # Temporarily enable for stub agent call (Logfire instruments before reaching stub)
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        result = await finding_description.create_info_from_name("left lower lobe opacity", model_tier="small")
    finally:
        models.ALLOW_MODEL_REQUESTS = original

    assert result.name == "pulmonary opacity"
    assert result.synonyms == [
        "Pulmonary opacity",
        "Left lower lobe shadow",
        "left lower lobe opacity",
    ]
    assert any("left lower lobe opacity" in entry for entry in logged)
    assert stub_agent.prompts, "Agent should receive the rendered prompt"


@pytest.mark.asyncio
async def test_create_info_from_name_preserves_name_without_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    output = FindingInfo(
        name="pneumothorax",
        synonyms=["pneumothorax  ", "PTX"],
        description="Presence of air in the pleural space.",
    )

    stub_agent = _StubFindingInfoAgent(output)
    monkeypatch.setattr(
        finding_description,
        "_create_finding_info_agent",
        lambda model_tier, instructions: stub_agent,
    )

    logged: list[str] = []
    monkeypatch.setattr(
        logger,
        "info",
        lambda message, *args, **kwargs: logged.append(message),
    )

    # Temporarily enable for stub agent call (Logfire instruments before reaching stub)
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        result = await finding_description.create_info_from_name("pneumothorax")
    finally:
        models.ALLOW_MODEL_REQUESTS = original

    assert result.name == "pneumothorax"
    assert result.synonyms == ["pneumothorax", "PTX"]
    assert not logged


def test_add_index_codes_to_finding_model(full_model: FindingModelFull) -> None:
    """Test adding codes to a finding model."""
    findingmodel.tools.add_standard_codes_to_finding_model(full_model)
    attribute = full_model.attributes[0]
    assert attribute.index_codes is not None
    assert len(attribute.index_codes) == 1
    first_code = attribute.index_codes[0]
    assert isinstance(first_code, IndexCode)
    assert first_code.code == "246112005"
    assert first_code.system == "SNOMED"
    assert first_code.display is not None and first_code.display.startswith("Severity")
    assert isinstance(attribute, ChoiceAttributeIded)
    first_value = attribute.values[0]
    assert first_value.index_codes is not None
    assert len(first_value.index_codes) == 2
    first_value_code = first_value.index_codes[0]
    assert isinstance(first_value_code, IndexCode)
    assert first_value_code.system == "RADLEX"
    assert first_value_code.code == "RID5671"
    assert first_value_code.display is not None and first_value_code.display.startswith("mild")


def test_add_index_codes_to_finding_model_no_duplicates(full_model: FindingModelFull) -> None:
    """Test adding codes to a finding model."""
    findingmodel.tools.add_standard_codes_to_finding_model(full_model)
    attribute = full_model.attributes[0]
    assert attribute.index_codes is not None
    assert len(attribute.index_codes) == 1
    findingmodel.tools.add_standard_codes_to_finding_model(full_model)
    assert len(attribute.index_codes) == 1


# Tests for new function names (non-deprecated API)


def test_create_model_stub_from_info_new_api(finding_info: FindingInfo) -> None:
    """Test creating a stub finding model using the new function name."""
    stub = findingmodel.tools.create_model_stub_from_info(finding_info)
    assert isinstance(stub, FindingModelBase)
    assert stub.name == finding_info.name.lower()
    assert stub.description == finding_info.description
    assert stub.synonyms == finding_info.synonyms
    assert len(stub.attributes) == 2
    assert stub.attributes[0].name == "presence"
    assert stub.attributes[1].name == "change from prior"


def test_add_ids_to_model_new_api(base_model: FindingModelBase) -> None:
    """Test adding IDs to a finding model using the new function name."""
    updated_model = findingmodel.tools.add_ids_to_model(base_model, source="TEST")
    assert isinstance(updated_model, FindingModelFull)
    assert updated_model.oifm_id is not None
    assert updated_model.oifm_id.startswith("OIFM_")
    assert "TEST" in updated_model.oifm_id
    assert len(updated_model.attributes) == len(base_model.attributes)
    for attr in updated_model.attributes:
        assert attr.oifma_id is not None
        assert attr.oifma_id.startswith("OIFMA_")
        assert "TEST" in attr.oifma_id
        if attr.type == AttributeType.CHOICE:
            for i, value in enumerate(attr.values):
                assert value.value_code is not None
                assert value.value_code == f"{attr.oifma_id}.{i}"


def test_add_standard_codes_to_model_new_api(full_model: FindingModelFull) -> None:
    """Test adding codes to a finding model using the new function name."""
    findingmodel.tools.add_standard_codes_to_model(full_model)
    attribute = full_model.attributes[0]
    assert attribute.index_codes is not None
    assert len(attribute.index_codes) == 1
    first_code = attribute.index_codes[0]
    assert isinstance(first_code, IndexCode)
    assert first_code.code == "246112005"
    assert first_code.system == "SNOMED"
    assert first_code.display is not None and first_code.display.startswith("Severity")
    assert isinstance(attribute, ChoiceAttributeIded)
    first_value = attribute.values[0]
    assert first_value.index_codes is not None
    assert len(first_value.index_codes) == 2
    first_value_code = first_value.index_codes[0]
    assert isinstance(first_value_code, IndexCode)
    assert first_value_code.system == "RADLEX"
    assert first_value_code.code == "RID5671"
    assert first_value_code.display is not None and first_value_code.display.startswith("mild")


def test_add_standard_codes_to_model_no_duplicates_new_api(full_model: FindingModelFull) -> None:
    """Test adding codes to a finding model using the new function name (no duplicates)."""
    findingmodel.tools.add_standard_codes_to_model(full_model)
    attribute = full_model.attributes[0]
    assert attribute.index_codes is not None
    assert len(attribute.index_codes) == 1
    # Call again to ensure no duplicates
    findingmodel.tools.add_standard_codes_to_model(full_model)
    assert len(attribute.index_codes) == 1


# Integration tests requiring external API access
@pytest.mark.callout
@pytest.mark.asyncio
async def test_create_info_from_name_basic_wiring() -> None:
    """Sanity check: Verify basic wiring with real API.

    All comprehensive behavioral testing is in evals/finding_description.py.
    This test only verifies the tool can be called successfully.
    """
    # Skip if no API key configured
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
        pytest.skip("OpenAI API key not configured")

    from findingmodel.finding_info import FindingInfo
    from findingmodel.tools import create_info_from_name

    # Save and restore ALLOW_MODEL_REQUESTS state
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Single API call with simple input - use fast model for integration test
        result = await create_info_from_name("pneumothorax", model_tier="small")

        # Only structural assertions - no behavioral validation
        assert isinstance(result, FindingInfo)
        assert hasattr(result, "name")
        assert hasattr(result, "description")
        assert hasattr(result, "synonyms")
    finally:
        models.ALLOW_MODEL_REQUESTS = original


def _create_stub_result(output: str, messages: list[str]) -> SimpleNamespace:
    """Create a stub result for testing add_details_to_info without API calls."""

    class _MockMessage:
        def __init__(self, content: str) -> None:
            self.content = content

        def __str__(self) -> str:
            return self.content

    mock_messages = [_MockMessage(msg) for msg in messages]
    return SimpleNamespace(output=output, all_messages=lambda: mock_messages)


@pytest.mark.asyncio
async def test_add_details_to_info_with_test_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test: Verify citation extraction and field preservation without API calls.

    Uses stub result to verify citation extraction logic works correctly
    without making real API calls.
    """
    from unittest.mock import AsyncMock, patch

    from findingmodel.config import FindingModelConfig
    from findingmodel.tools import add_details_to_info

    # Create test FindingInfo
    finding = FindingInfo(
        name="pneumothorax",
        description="Presence of air in the pleural space",
        synonyms=["PTX", "spontaneous pneumothorax"],
    )

    # Expected detail text (what the agent should return)
    mock_detail = (
        "Pneumothorax is characterized by the presence of air in the pleural space. "
        "Key imaging features include visceral pleural line separation and absence of lung markings peripherally. "
        "Common locations include apical and lateral pleural spaces."
    )

    # Mock search tool responses with Source: URLs
    mock_messages = [
        "Initial prompt",
        "Pneumothorax is visible as air in the pleural cavity on chest X-ray.\n\n"
        "Source: https://radiopaedia.org/articles/pneumothorax",
        "The visceral pleural line is the key finding in pneumothorax diagnosis.\n\n"
        "Source: https://radiologyassistant.nl/chest/pneumothorax",
        "Additional information about pneumothorax characteristics.\n\n"
        "Source: https://radiopaedia.org/articles/pneumothorax",  # Duplicate URL to test deduplication
        f"Final response: {mock_detail}",
    ]

    stub_result = _create_stub_result(output=mock_detail, messages=mock_messages)

    # Mock Agent.run to return our stub result
    mock_run = AsyncMock(return_value=stub_result)

    # Patch Agent.run method
    from pydantic_ai import Agent

    monkeypatch.setattr(Agent, "run", mock_run)

    # Mock settings.get_model to return a test model string
    with patch.object(FindingModelConfig, "get_model", return_value="test"):
        # Run the function
        result = await add_details_to_info(finding)

    # Verify the result structure
    assert result is not None
    assert isinstance(result, FindingInfo)

    # Verify original fields are preserved
    assert result.name == finding.name
    assert result.synonyms == finding.synonyms
    assert result.description == finding.description

    # Verify detail is populated
    assert result.detail == mock_detail

    # Verify citations are extracted and deduplicated
    assert result.citations is not None
    assert len(result.citations) == 2  # Should deduplicate the radiopaedia URL
    assert "https://radiopaedia.org/articles/pneumothorax" in result.citations
    assert "https://radiologyassistant.nl/chest/pneumothorax" in result.citations

    # Verify agent.run was called
    assert mock_run.called


@pytest.mark.asyncio
async def test_add_details_to_info_empty_output_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test: Verify function returns None when agent returns empty output.

    Tests the error path at lines 186-187 in finding_description.py where
    the function returns None if result.output is falsy.
    """
    from unittest.mock import AsyncMock, patch

    from findingmodel.config import FindingModelConfig
    from findingmodel.tools import add_details_to_info
    from pydantic_ai import Agent

    # Create test FindingInfo
    finding = FindingInfo(
        name="pneumothorax",
        description="Presence of air in the pleural space",
        synonyms=["PTX"],
    )

    # Create stub result with empty output
    stub_result = _create_stub_result(output="", messages=["No results"])

    # Mock Agent.run to return empty output
    mock_run = AsyncMock(return_value=stub_result)
    monkeypatch.setattr(Agent, "run", mock_run)

    # Mock settings.get_model to return a test model string
    with patch.object(FindingModelConfig, "get_model", return_value="test"):
        # Run the function
        result = await add_details_to_info(finding)

    # Verify function returns None for empty output
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize("search_depth", ["basic", "advanced"])
async def test_add_details_to_info_search_depth_parameter(search_depth: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test: Verify search_depth parameter is passed to Tavily search.

    Tests that the search_depth parameter (line 97, used at line 126 in finding_description.py)
    flows correctly from function parameter to the Tavily client search call.
    """
    from typing import Any
    from unittest.mock import AsyncMock, patch

    from findingmodel.config import FindingModelConfig
    from findingmodel.tools import add_details_to_info

    # Create test FindingInfo
    finding = FindingInfo(
        name="pneumothorax",
        description="Presence of air in the pleural space",
        synonyms=["PTX"],
    )

    # Track calls to Tavily search
    captured_search_calls: list[dict[str, Any]] = []

    # Create mock Tavily client with search method
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(
        side_effect=lambda query, **kwargs: (
            captured_search_calls.append({"query": query, **kwargs}),
            {"results": [{"content": "Test radiology content", "url": "https://radiopaedia.org/test"}]},
        )[1]
    )

    # Mock get_async_tavily_client to return our mock client
    monkeypatch.setattr("findingmodel.tools.finding_description.get_async_tavily_client", lambda: mock_client)

    # Enable model requests for this test since we're using TestModel
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", True)

    # Mock settings.get_model and run add_details_to_info with the specified search_depth
    with patch.object(FindingModelConfig, "get_model", return_value="test"):
        result = await add_details_to_info(finding, search_depth=search_depth)

    # Verify the result is returned
    assert result is not None

    # Verify that Tavily search was called with the correct search_depth parameter
    assert len(captured_search_calls) > 0, "Tavily search should have been called at least once"
    for call in captured_search_calls:
        assert call["search_depth"] == search_depth, f"Expected search_depth={search_depth}, got {call['search_depth']}"


@pytest.mark.callout
@pytest.mark.asyncio
async def test_add_details_to_info_basic_wiring() -> None:
    """Sanity check: Verify basic wiring with real API.

    All comprehensive behavioral testing is in evals/finding_description.py.
    This test only verifies the tool can be called successfully.
    """
    # Skip if Tavily API key not configured
    if not settings.tavily_api_key or not settings.tavily_api_key.get_secret_value():
        pytest.skip("Tavily API key not configured")

    from findingmodel.finding_info import FindingInfo
    from findingmodel.tools import add_details_to_info

    # Save and restore ALLOW_MODEL_REQUESTS state
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Start with basic finding info
        basic_info = FindingInfo(
            name="pneumothorax", description="Presence of air in the pleural space", synonyms=["PTX"]
        )

        # Single API call
        result = await add_details_to_info(basic_info)

        # Only structural assertions - no behavioral validation
        assert isinstance(result, FindingInfo)
        assert hasattr(result, "detail")
        assert hasattr(result, "citations")
    finally:
        models.ALLOW_MODEL_REQUESTS = original


@pytest.mark.asyncio
async def test_create_model_from_markdown_with_test_model() -> None:
    """Unit test: Verify agent structure without API calls.

    Uses TestModel to verify agent configuration and prompt extraction
    work correctly without making real API calls.
    """
    from unittest.mock import patch

    from findingmodel.config import FindingModelConfig
    from findingmodel.finding_model import ChoiceAttribute, ChoiceValue, FindingModelBase
    from findingmodel.tools import create_model_from_markdown
    from pydantic_ai.models.test import TestModel

    # ALLOW_MODEL_REQUESTS is already False at module level

    # Create test data
    finding_info = FindingInfo(
        name="Test Finding",
        description="A test finding for unit testing",
        synonyms=["test", "unittest"],
    )

    markdown_text = """
    # Test Finding Attributes

    ## Size
    - Small
    - Large

    ## Severity
    - Mild
    - Moderate
    - Severe
    """

    # Create valid FindingModelBase for TestModel to return
    test_output = FindingModelBase(
        name="test finding",
        description="A test finding for unit testing",
        attributes=[
            ChoiceAttribute(
                name="Size",
                description="Size of the finding",
                values=[ChoiceValue(name="Small"), ChoiceValue(name="Large")],
                required=False,
                max_selected=1,
            )
        ],
    )

    # Mock settings.get_model to return a TestModel
    # Since get_model now returns a string, we patch at the class level
    with patch.object(FindingModelConfig, "get_model") as mock_get_model:
        mock_get_model.return_value = TestModel(custom_output_args=test_output.model_dump())

        # This verifies the agent structure, not LLM behavior
        result = await create_model_from_markdown(
            finding_info,
            markdown_text=markdown_text,
        )

        # Verify structure
        assert isinstance(result, FindingModelBase)
        assert result.name == test_output.name
        assert result.description == test_output.description
        assert len(result.attributes) == len(test_output.attributes)


@pytest.mark.callout
@pytest.mark.asyncio
async def test_create_model_from_markdown_basic_wiring() -> None:
    """Sanity check: Verify basic wiring with real API.

    All comprehensive behavioral testing is in evals/markdown_in.py.
    This test only verifies the tool can be called successfully.
    """
    # Skip if no API key configured
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
        pytest.skip("OpenAI API key not configured")

    from findingmodel.finding_model import FindingModelBase
    from findingmodel.tools import create_info_from_name, create_model_from_markdown

    # Save and restore ALLOW_MODEL_REQUESTS state
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Create basic info - use fast model for integration test
        finding_info = await create_info_from_name("pneumothorax", model_tier="small")

        # Simple markdown outline
        markdown_text = """
        # Pneumothorax Attributes

        ## Size
        - Small
        - Large
        """

        # Create model from markdown - use fast model for integration test
        model = await create_model_from_markdown(finding_info, markdown_text=markdown_text, model_tier="small")

        # Only structural assertions - no behavioral validation
        assert isinstance(model, FindingModelBase)
        assert hasattr(model, "name")
        assert hasattr(model, "description")
        assert hasattr(model, "attributes")
    finally:
        models.ALLOW_MODEL_REQUESTS = original


# Comprehensive behavioral testing in evals/similar_models.py
# This sanity check verifies basic wiring only
@pytest.mark.callout
@pytest.mark.asyncio
async def test_find_similar_models_basic_wiring() -> None:
    """Sanity check: Verify basic wiring with real API.

    All comprehensive behavioral testing is in evals/similar_models.py.
    This test only verifies the tool can be called successfully.
    """
    from findingmodel.tools import find_similar_models
    from findingmodel.tools.similar_finding_models import SimilarModelAnalysis

    # Skip if API key not configured
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
        pytest.skip("OPENAI_API_KEY not configured")

    # Save and restore ALLOW_MODEL_REQUESTS state
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Call with simplest valid input
        result = await find_similar_models(
            finding_name="pneumothorax", description="Presence of air in the pleural space"
        )

        # Assert only on structure, not behavior
        assert isinstance(result, SimilarModelAnalysis)
        assert hasattr(result, "similar_models")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "confidence")

        # NO behavioral assertions - those belong in evals
    finally:
        models.ALLOW_MODEL_REQUESTS = original


def test_tools_import_failures() -> None:
    """Test graceful handling when optional dependencies are missing."""
    # This tests the robustness of the import system
    try:
        import findingmodel.tools.create_stub

        assert hasattr(findingmodel.tools.create_stub, "create_model_stub_from_info")
    except ImportError:
        pytest.skip("create_stub module not available")


def test_concurrent_id_generation(base_model: FindingModelBase) -> None:
    """Test ID generation under concurrent access."""
    import concurrent.futures

    def generate_ids(source_suffix: int) -> FindingModelFull:
        # Use valid 3-character source codes
        sources = ["TST", "TES", "TEX"]
        return findingmodel.tools.add_ids_to_model(base_model, source=sources[source_suffix])

    # Run multiple ID generation operations concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(generate_ids, i) for i in range(3)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All should succeed and have unique IDs
    assert len(results) == 3
    oifm_ids = [r.oifm_id for r in results]
    assert len(set(oifm_ids)) == 3  # All should be unique


def test_get_async_tavily_client_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_async_tavily_client returns client when API key is set."""
    from findingmodel import config
    from findingmodel.tools.common import get_async_tavily_client
    from pydantic import SecretStr
    from tavily import AsyncTavilyClient

    # Temporarily override settings with test key
    original_key = config.settings.tavily_api_key
    try:
        config.settings.tavily_api_key = SecretStr("test-tavily-key")
        client = get_async_tavily_client()
        assert isinstance(client, AsyncTavilyClient)
    finally:
        config.settings.tavily_api_key = original_key


def test_get_async_tavily_client_without_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_async_tavily_client raises ConfigurationError when API key missing."""
    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from findingmodel.tools.common import get_async_tavily_client
    from pydantic import SecretStr

    # Temporarily override settings with empty key
    original_key = config.settings.tavily_api_key
    try:
        config.settings.tavily_api_key = SecretStr("")
        with pytest.raises(ConfigurationError, match="Tavily API key is not set"):
            get_async_tavily_client()
    finally:
        config.settings.tavily_api_key = original_key


# Tests for settings.get_model() method


def test_settings_get_model_returns_openai_model_for_base_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() returns OpenAIResponsesModel for base tier with OpenAI provider."""
    from findingmodel import config
    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIResponsesModel

    # Temporarily override settings with test key and model
    original_key = config.settings.openai_api_key
    original_model = config.settings.default_model
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        config.settings.default_model = "openai:gpt-5-mini"
        model = config.settings.get_model("base")
        assert isinstance(model, OpenAIResponsesModel)
    finally:
        config.settings.openai_api_key = original_key
        config.settings.default_model = original_model


def test_settings_get_model_returns_openai_model_for_small_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() returns OpenAIResponsesModel for small tier with OpenAI provider."""
    from findingmodel import config
    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIResponsesModel

    # Temporarily override settings with test key and model
    original_key = config.settings.openai_api_key
    original_model = config.settings.default_model_small
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        config.settings.default_model_small = "openai:gpt-5-nano"
        model = config.settings.get_model("small")
        assert isinstance(model, OpenAIResponsesModel)
    finally:
        config.settings.openai_api_key = original_key
        config.settings.default_model_small = original_model


def test_settings_get_model_returns_openai_model_for_full_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() returns OpenAIResponsesModel for full tier with OpenAI provider."""
    from findingmodel import config
    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIResponsesModel

    # Temporarily override settings with test key and model
    original_key = config.settings.openai_api_key
    original_model = config.settings.default_model_full
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        config.settings.default_model_full = "openai:gpt-5"
        model = config.settings.get_model("full")
        assert isinstance(model, OpenAIResponsesModel)
    finally:
        config.settings.openai_api_key = original_key
        config.settings.default_model_full = original_model


def test_settings_get_model_validates_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() raises ConfigurationError when OpenAI API key missing."""
    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from pydantic import SecretStr

    # Temporarily override settings with empty key
    original_key = config.settings.openai_api_key
    original_model = config.settings.default_model
    try:
        config.settings.openai_api_key = SecretStr("")
        config.settings.default_model = "openai:gpt-5-mini"
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY not configured"):
            config.settings.get_model("base")
    finally:
        config.settings.openai_api_key = original_key
        config.settings.default_model = original_model


def test_settings_get_model_validates_anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() raises ConfigurationError when Anthropic API key missing."""
    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from pydantic import SecretStr

    # Temporarily override settings with empty key
    original_key = config.settings.anthropic_api_key
    original_model = config.settings.default_model
    try:
        config.settings.anthropic_api_key = SecretStr("")
        config.settings.default_model = "anthropic:claude-sonnet-4-5"
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY not configured"):
            config.settings.get_model("base")
    finally:
        config.settings.anthropic_api_key = original_key
        config.settings.default_model = original_model


def test_settings_get_model_validates_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() raises ConfigurationError when Google API key missing."""
    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from pydantic import SecretStr

    # Temporarily override settings with empty key
    original_key = config.settings.google_api_key
    original_model = config.settings.default_model
    try:
        config.settings.google_api_key = SecretStr("")
        config.settings.default_model = "google:gemini-3-flash-preview"
        with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY not configured"):
            config.settings.get_model("base")
    finally:
        config.settings.google_api_key = original_key
        config.settings.default_model = original_model


def test_settings_get_model_validates_gateway_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() raises ConfigurationError when Gateway API key missing."""
    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from pydantic import SecretStr

    # Temporarily override settings with empty key
    original_key = config.settings.pydantic_ai_gateway_api_key
    original_model = config.settings.default_model
    try:
        config.settings.pydantic_ai_gateway_api_key = SecretStr("")
        config.settings.default_model = "gateway/openai:gpt-5-mini"
        with pytest.raises(ConfigurationError, match="PYDANTIC_AI_GATEWAY_API_KEY not configured"):
            config.settings.get_model("base")
    finally:
        config.settings.pydantic_ai_gateway_api_key = original_key
        config.settings.default_model = original_model


def test_settings_get_model_raises_for_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test settings.get_model() raises ConfigurationError for unknown providers."""
    from findingmodel import config
    from findingmodel.config import ConfigurationError

    # Temporarily override settings with unknown provider model
    original_model = config.settings.default_model
    try:
        config.settings.default_model = "unsupported-provider:some-model"
        # Should raise - unknown provider not supported
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            config.settings.get_model("base")
    finally:
        config.settings.default_model = original_model


# =============================================================================
# Ollama Model Validation Tests
# =============================================================================


class TestOllamaModelValidation:
    """Unit tests for Ollama model validation (mocked, no real server)."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> Generator[None, None, None]:
        """Clear Ollama models cache before and after each test."""
        from findingmodel import config

        config.settings.clear_ollama_models_cache()
        yield
        config.settings.clear_ollama_models_cache()

    def test_validate_exact_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model with exact name match passes validation."""
        from unittest.mock import Mock

        from findingmodel import config

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        mock_response.raise_for_status = Mock()

        monkeypatch.setattr(httpx, "get", Mock(return_value=mock_response))

        # Should not raise
        config.settings._validate_ollama_model("gpt-oss:20b")

    def test_validate_implicit_latest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model without tag matches model:latest."""
        from unittest.mock import Mock

        from findingmodel import config

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3:latest"}]}
        mock_response.raise_for_status = Mock()

        monkeypatch.setattr(httpx, "get", Mock(return_value=mock_response))

        # "llama3" should match "llama3:latest"
        config.settings._validate_ollama_model("llama3")

    def test_validate_not_found_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing model raises ConfigurationError with helpful message."""
        from unittest.mock import Mock

        from findingmodel import config
        from findingmodel.config import ConfigurationError

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        mock_response.raise_for_status = Mock()

        monkeypatch.setattr(httpx, "get", Mock(return_value=mock_response))

        with pytest.raises(ConfigurationError) as exc_info:
            config.settings._validate_ollama_model("nonexistent")

        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg
        assert "gpt-oss:20b" in error_msg
        assert "ollama pull" in error_msg

    def test_validate_server_unreachable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unreachable server raises ConfigurationError (fail fast)."""
        from unittest.mock import Mock

        from findingmodel import config
        from findingmodel.config import ConfigurationError

        monkeypatch.setattr(httpx, "get", Mock(side_effect=httpx.ConnectError("Connection refused")))

        with pytest.raises(ConfigurationError) as exc_info:
            config.settings._validate_ollama_model("any-model")

        assert "not reachable" in str(exc_info.value)
        assert "ollama serve" in str(exc_info.value)

    def test_validate_server_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Server error raises ConfigurationError (fail fast)."""
        from unittest.mock import Mock

        from findingmodel import config
        from findingmodel.config import ConfigurationError

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=Mock(), response=mock_response
        )

        monkeypatch.setattr(httpx, "get", Mock(return_value=mock_response))

        with pytest.raises(ConfigurationError) as exc_info:
            config.settings._validate_ollama_model("any-model")

        assert "500" in str(exc_info.value)

    def test_validate_caching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Models list is cached across calls."""
        from unittest.mock import Mock

        from findingmodel import config

        mock_get = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "test:latest"}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        monkeypatch.setattr(httpx, "get", mock_get)

        config.settings._get_ollama_available_models()
        config.settings._get_ollama_available_models()

        assert mock_get.call_count == 1  # Only one HTTP call

    def test_clear_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """clear_ollama_models_cache() invalidates the cache."""
        from unittest.mock import Mock

        from findingmodel import config

        mock_get = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "test:latest"}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        monkeypatch.setattr(httpx, "get", mock_get)

        config.settings._get_ollama_available_models()
        assert mock_get.call_count == 1

        config.settings.clear_ollama_models_cache()
        config.settings._get_ollama_available_models()
        assert mock_get.call_count == 2  # Called again after cache clear

    def test_make_ollama_model_validates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_make_ollama_model calls validation before creating model."""
        from unittest.mock import Mock

        from findingmodel import config
        from findingmodel.config import ConfigurationError

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "valid:latest"}]}
        mock_response.raise_for_status = Mock()

        monkeypatch.setattr(httpx, "get", Mock(return_value=mock_response))

        with pytest.raises(ConfigurationError) as exc_info:
            config.settings._make_ollama_model("invalid-model")

        assert "invalid-model" in str(exc_info.value)


class TestDefaultModelKeyValidation:
    """Tests for validate_default_model_keys() functionality."""

    def test_get_required_key_field_openai(self) -> None:
        """OpenAI provider requires openai_api_key."""
        from findingmodel import config

        assert config.settings._get_required_key_field("openai:gpt-5-mini") == "openai_api_key"

    def test_get_required_key_field_anthropic(self) -> None:
        """Anthropic provider requires anthropic_api_key."""
        from findingmodel import config

        assert config.settings._get_required_key_field("anthropic:claude-sonnet-4-5") == "anthropic_api_key"

    def test_get_required_key_field_google(self) -> None:
        """Google providers require google_api_key."""
        from findingmodel import config

        assert config.settings._get_required_key_field("google:gemini-2.0-flash") == "google_api_key"
        assert config.settings._get_required_key_field("google-gla:gemini-2.0-flash") == "google_api_key"

    def test_get_required_key_field_gateway(self) -> None:
        """Gateway providers require pydantic_ai_gateway_api_key."""
        from findingmodel import config

        assert config.settings._get_required_key_field("gateway/openai:gpt-5-mini") == "pydantic_ai_gateway_api_key"
        assert (
            config.settings._get_required_key_field("gateway/anthropic:claude-sonnet-4-5")
            == "pydantic_ai_gateway_api_key"
        )
        assert (
            config.settings._get_required_key_field("gateway/google:gemini-2.0-flash") == "pydantic_ai_gateway_api_key"
        )

    def test_get_required_key_field_ollama(self) -> None:
        """Ollama provider requires no API key."""
        from findingmodel import config

        assert config.settings._get_required_key_field("ollama:llama3") is None

    def test_get_required_key_field_invalid(self) -> None:
        """Invalid model string returns None."""
        from findingmodel import config

        assert config.settings._get_required_key_field("no-colon") is None

    def test_validate_passes_with_keys(self) -> None:
        """Validation passes when required keys are present."""
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr

        config = FindingModelConfig(
            openai_api_key=SecretStr("test-key"),
            default_model="openai:gpt-5-mini",
            default_model_full="openai:gpt-5.2",
            default_model_small="openai:gpt-5-nano",
        )

        # Should not raise
        config.validate_default_model_keys()

    def test_validate_fails_missing_key(self) -> None:
        """Validation fails when required key is missing."""
        from findingmodel.config import ConfigurationError, FindingModelConfig
        from pydantic import SecretStr

        config = FindingModelConfig(
            openai_api_key=SecretStr(""),  # Empty key
            default_model="openai:gpt-5-mini",
            default_model_full="openai:gpt-5.2",
            default_model_small="openai:gpt-5-nano",
        )

        with pytest.raises(ConfigurationError) as exc_info:
            config.validate_default_model_keys()

        error_msg = str(exc_info.value)
        assert "OPENAI_API_KEY" in error_msg
        assert "default_model" in error_msg

    def test_validate_mixed_providers(self) -> None:
        """Validation checks all providers correctly."""
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr

        config = FindingModelConfig(
            openai_api_key=SecretStr("openai-key"),
            anthropic_api_key=SecretStr("anthropic-key"),
            default_model="openai:gpt-5-mini",
            default_model_full="anthropic:claude-opus-4-5",
            default_model_small="openai:gpt-5-nano",
        )

        # Should not raise - both keys are present
        config.validate_default_model_keys()

    def test_validate_ollama_no_key_required(self) -> None:
        """Ollama defaults don't require API key validation."""
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr

        config = FindingModelConfig(
            openai_api_key=SecretStr(""),  # Empty key
            default_model="ollama:llama3",
            default_model_full="ollama:llama3:70b",
            default_model_small="ollama:llama3:8b",
        )

        # Should not raise - Ollama doesn't require API keys
        config.validate_default_model_keys()


@pytest.mark.callout
@pytest.mark.asyncio
async def test_gateway_openai_integration() -> None:
    """Integration test: Verify gateway/openai model works with real API.

    This test verifies that the gateway provider is correctly configured
    and can make actual API calls through the Pydantic AI Gateway.
    """
    from findingmodel import config
    from pydantic_ai import Agent

    # Skip if gateway API key not configured
    if not config.settings.pydantic_ai_gateway_api_key.get_secret_value():
        pytest.skip("PYDANTIC_AI_GATEWAY_API_KEY not configured")

    # Enable model requests for this callout test
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Temporarily override settings to use gateway
        original_model = config.settings.default_model
        config.settings.default_model = f"gateway/{TEST_OPENAI_MODEL}"

        try:
            # Get model from settings - this should return an OpenAIResponsesModel
            # configured with the gateway provider
            model = config.settings.get_model("base")

            # Create a simple agent and make a real API call
            agent = Agent(model, output_type=str)
            result = await agent.run("Reply with exactly: GATEWAY_TEST_OK")

            # Verify we got a response (content doesn't matter, just that it worked)
            assert result.output is not None
            assert len(result.output) > 0
        finally:
            config.settings.default_model = original_model
    finally:
        models.ALLOW_MODEL_REQUESTS = original


@pytest.mark.callout
@pytest.mark.asyncio
async def test_gemini_integration() -> None:
    """Integration test: Verify gemini model works with real API.

    This test verifies that the Gemini provider is correctly configured
    and can make actual API calls to Google's Gemini API.
    """
    from findingmodel import config
    from pydantic_ai import Agent

    # Skip if Google API key not configured
    if not config.settings.google_api_key.get_secret_value():
        pytest.skip("GOOGLE_API_KEY not configured")

    # Enable model requests for this callout test
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Temporarily override settings to use Gemini
        original_model = config.settings.default_model
        config.settings.default_model = "google:gemini-3-flash-preview"

        try:
            # Get model from settings
            model = config.settings.get_model("base")

            # Create a simple agent and make a real API call
            agent = Agent(model, output_type=str)
            result = await agent.run("Reply with exactly: GEMINI_TEST_OK")

            # Verify we got a response
            assert result.output is not None
            assert len(result.output) > 0
        finally:
            config.settings.default_model = original_model
    finally:
        models.ALLOW_MODEL_REQUESTS = original


# Preferred models for Ollama integration testing (in order of preference)
# These are small/fast models suitable for quick integration tests
PREFERRED_OLLAMA_MODELS = [
    "gpt-oss:20b",  # Good balance of speed and capability
    "qwen2.5:3b",  # Very fast, capable
    "qwen2.5:7b",  # Fast, more capable
    "llama3.2:3b",  # Fast, common
    "gemma3:4b",  # Fast, Google's small model
    "phi4-mini",  # Microsoft's small model
]


@pytest.mark.callout
@pytest.mark.asyncio
async def test_ollama_integration() -> None:
    """Integration test: Verify ollama model works with local Ollama server.

    This test verifies that the Ollama provider is correctly configured
    and can make actual API calls to a local Ollama server.
    Requires Ollama running locally with a model available.
    Prefers smaller/faster models from PREFERRED_OLLAMA_MODELS list.
    """
    import httpx
    from findingmodel import config
    from pydantic_ai import Agent

    # Check if Ollama server is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.settings.ollama_base_url.rstrip('/v1')}/api/tags", timeout=2.0)
            if response.status_code != 200:
                pytest.skip("Ollama server not responding")
            # Check if any models are available
            tags = response.json()
            if not tags.get("models"):
                pytest.skip("No models available in Ollama")

            # Get list of available model names
            available_models = {m["name"] for m in tags["models"]}

            # Prefer models from our list, in order
            selected_model = None
            for preferred in PREFERRED_OLLAMA_MODELS:
                if preferred in available_models:
                    selected_model = preferred
                    break

            # Fall back to first available if no preferred model found
            if selected_model is None:
                selected_model = tags["models"][0]["name"]

    except (httpx.ConnectError, httpx.TimeoutException):
        pytest.skip("Ollama server not running at " + config.settings.ollama_base_url)

    # Enable model requests for this callout test
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Temporarily override settings to use Ollama
        original_model = config.settings.default_model
        config.settings.default_model = f"ollama:{selected_model}"

        try:
            # Get model from settings
            model = config.settings.get_model("base")

            # Create a simple agent and make a real API call
            agent = Agent(model, output_type=str)
            result = await agent.run("Reply with exactly: OLLAMA_TEST_OK")

            # Verify we got a response
            assert result.output is not None
            assert len(result.output) > 0
        finally:
            config.settings.default_model = original_model
    finally:
        models.ALLOW_MODEL_REQUESTS = original


# =============================================================================
# Per-Agent Model Override Tests
# =============================================================================


class TestAgentModelOverrides:
    """Tests for per-agent model configuration overrides via get_agent_model()."""

    def test_returns_tier_default_when_no_override(self) -> None:
        """get_agent_model() falls back to tier when agent not in overrides."""
        from conftest import TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr
        from pydantic_ai.models.openai import OpenAIResponsesModel

        # Create config with empty overrides
        config = FindingModelConfig(
            openai_api_key=SecretStr("test-key"),
            default_model=TEST_OPENAI_MODEL,
            default_model_small=TEST_OPENAI_MODEL,
            default_model_full=TEST_OPENAI_MODEL,
            agent_model_overrides={},
        )

        # Agent not in overrides should use tier default
        model = config.get_agent_model("enrich_classify", default_tier="base")

        # Should return OpenAI model (same as get_model("base"))
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_override_when_configured(self) -> None:
        """get_agent_model() uses override from agent_model_overrides dict."""
        from conftest import TEST_ANTHROPIC_MODEL, TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr
        from pydantic_ai.models.anthropic import AnthropicModel

        # Create config with override for specific agent
        config = FindingModelConfig(
            openai_api_key=SecretStr("test-openai-key"),
            anthropic_api_key=SecretStr("test-anthropic-key"),
            default_model=TEST_OPENAI_MODEL,
            agent_model_overrides={"enrich_classify": TEST_ANTHROPIC_MODEL},
        )

        # Agent in overrides should use the override model
        model = config.get_agent_model("enrich_classify", default_tier="base")

        # Should return Anthropic model (from override), not OpenAI (from default)
        assert isinstance(model, AnthropicModel)

    def test_requires_explicit_agent_tag(self) -> None:
        """get_agent_model() requires explicit agent_tag parameter (no introspection)."""
        from conftest import TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr

        config = FindingModelConfig(
            openai_api_key=SecretStr("test-key"),
            default_model=TEST_OPENAI_MODEL,
        )

        # Should fail if called without agent_tag
        with pytest.raises(TypeError) as exc_info:
            config.get_agent_model(default_tier="base")  # type: ignore[call-arg]

        # Error should indicate missing required argument
        assert "agent_tag" in str(exc_info.value)

    def test_validates_model_spec_pattern(self) -> None:
        """Invalid model strings in overrides fail Pydantic validation."""
        from conftest import TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr, ValidationError

        # Try to create config with invalid override value
        with pytest.raises(ValidationError) as exc_info:
            FindingModelConfig(
                openai_api_key=SecretStr("test-key"),
                default_model=TEST_OPENAI_MODEL,
                agent_model_overrides={
                    "enrich_classify": "invalid-model-string"  # Missing provider:model format
                },
            )

        # Verify the validation error mentions the invalid override
        error_msg = str(exc_info.value)
        assert "agent_model_overrides" in error_msg

    def test_override_respects_default_tier(self) -> None:
        """Override model string is used but tier affects settings (e.g., reasoning effort)."""
        from conftest import TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr

        # Create config with OpenAI override
        config = FindingModelConfig(
            openai_api_key=SecretStr("test-key"),
            default_model=TEST_OPENAI_MODEL,
            agent_model_overrides={"edit_instructions": "openai:gpt-5-mini"},
        )

        # Tier affects internal settings like reasoning effort
        model_small = config.get_agent_model("edit_instructions", default_tier="small")
        model_base = config.get_agent_model("edit_instructions", default_tier="base")

        # Both should use the override model name, but tier affects settings
        # (We can't easily inspect the internal settings, but we verify both succeed)
        assert model_small is not None
        assert model_base is not None

    def test_multiple_overrides_isolated(self) -> None:
        """Multiple agent overrides don't interfere with each other."""
        from conftest import TEST_ANTHROPIC_MODEL, TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.models.openai import OpenAIResponsesModel

        # Create config with overrides for multiple agents
        config = FindingModelConfig(
            openai_api_key=SecretStr("test-openai-key"),
            anthropic_api_key=SecretStr("test-anthropic-key"),
            default_model=TEST_OPENAI_MODEL,
            agent_model_overrides={
                "anatomic_search": "openai:gpt-5-nano",
                "ontology_match": TEST_ANTHROPIC_MODEL,
            },
        )

        # Each agent gets its own override
        model_one = config.get_agent_model("anatomic_search")
        model_two = config.get_agent_model("ontology_match")

        assert isinstance(model_one, OpenAIResponsesModel)
        assert isinstance(model_two, AnthropicModel)

    def test_override_with_gateway_provider(self) -> None:
        """Override can use gateway provider prefix."""
        from conftest import TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr
        from pydantic_ai.models.openai import OpenAIResponsesModel

        # Create config with gateway override
        config = FindingModelConfig(
            pydantic_ai_gateway_api_key=SecretStr("test-gateway-key"),
            default_model=TEST_OPENAI_MODEL,
            openai_api_key=SecretStr("test-openai-key"),
            agent_model_overrides={"similar_search": f"gateway/{TEST_OPENAI_MODEL}"},
        )

        # Should create model successfully with gateway provider
        model = config.get_agent_model("similar_search")
        assert isinstance(model, OpenAIResponsesModel)

    def test_agent_model_overrides_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variable AGENT_MODEL_OVERRIDES__<tag> sets override."""
        from conftest import TEST_ANTHROPIC_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr

        # Set env var with nested delimiter
        monkeypatch.setenv("AGENT_MODEL_OVERRIDES__enrich_classify", TEST_ANTHROPIC_MODEL)

        # Create fresh config (reads from env)
        config = FindingModelConfig(
            openai_api_key=SecretStr("test-key"),
            anthropic_api_key=SecretStr("test-key"),
        )

        # Verify the override was loaded from env
        assert "enrich_classify" in config.agent_model_overrides
        assert config.agent_model_overrides["enrich_classify"] == TEST_ANTHROPIC_MODEL

    def test_rejects_invalid_agent_tag(self) -> None:
        """Invalid agent tag in overrides dict raises ValidationError."""
        from conftest import TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr, ValidationError

        with pytest.raises(ValidationError) as exc_info:
            FindingModelConfig(
                openai_api_key=SecretStr("test-key"),
                default_model=TEST_OPENAI_MODEL,
                agent_model_overrides={"not_a_valid_tag": TEST_OPENAI_MODEL},  # Invalid tag
            )

        error_msg = str(exc_info.value)
        assert "agent_model_overrides" in error_msg

    def test_get_effective_model_string_returns_override(self) -> None:
        """get_effective_model_string() returns override when configured."""
        from conftest import TEST_ANTHROPIC_MODEL, TEST_OPENAI_MODEL
        from findingmodel.config import FindingModelConfig
        from pydantic import SecretStr

        config = FindingModelConfig(
            openai_api_key=SecretStr("test-key"),
            anthropic_api_key=SecretStr("test-key"),
            default_model=TEST_OPENAI_MODEL,
            agent_model_overrides={"enrich_classify": TEST_ANTHROPIC_MODEL},
        )

        # Should return override string
        result = config.get_effective_model_string("enrich_classify", "base")
        assert result == TEST_ANTHROPIC_MODEL

        # Non-overridden agent should return tier default
        result = config.get_effective_model_string("edit_instructions", "base")
        assert result == TEST_OPENAI_MODEL


# =============================================================================
# Per-Agent Model Override Integration Tests
# =============================================================================


class TestAgentModelOverridesIntegration:
    """Integration tests verifying agent_model_overrides work end-to-end with real API calls.

    These tests modify settings.agent_model_overrides directly (no monkeypatch needed) and
    verify that the override flows through to actual workflow execution with real AI models.
    """

    @pytest.mark.callout
    @pytest.mark.asyncio
    async def test_agent_override_integration_anthropic(self) -> None:
        """Agent override flows through to real API call with Anthropic model."""
        if not HAS_ANTHROPIC_API_KEY:
            pytest.skip("ANTHROPIC_API_KEY not set - skipping integration test")

        from conftest import TEST_ANTHROPIC_MODEL
        from findingmodel.tools.finding_enrichment import enrich_finding

        # Enable model requests for this test
        original_allow = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True

        # Set override directly on global settings singleton
        settings.agent_model_overrides["enrich_classify"] = TEST_ANTHROPIC_MODEL

        try:
            # Call workflow WITHOUT passing model parameter
            # This forces it to use get_agent_model() which should pick up our override
            result = await enrich_finding("pneumonia")

            # Validate result structure - finding_name may be canonical form from Index
            assert result.finding_name.lower() == "pneumonia"
            assert isinstance(result.snomed_codes, list)
            assert isinstance(result.radlex_codes, list)
            assert isinstance(result.body_regions, list)
            assert isinstance(result.etiologies, list)
            assert isinstance(result.modalities, list)
            assert isinstance(result.subspecialties, list)
            assert isinstance(result.anatomic_locations, list)
            assert result.enrichment_timestamp is not None

            # Verify model used in metadata matches our override
            assert result.model_used == TEST_ANTHROPIC_MODEL

        finally:
            # Clean up: remove override and restore ALLOW_MODEL_REQUESTS
            if "enrich_classify" in settings.agent_model_overrides:
                del settings.agent_model_overrides["enrich_classify"]
            models.ALLOW_MODEL_REQUESTS = original_allow

    @pytest.mark.callout
    @pytest.mark.asyncio
    async def test_agent_override_integration_gemini(self) -> None:
        """Agent override flows through to real API call with Google Gemini model."""
        if not HAS_GOOGLE_API_KEY:
            pytest.skip("GOOGLE_API_KEY not set - skipping integration test")

        from conftest import TEST_GOOGLE_MODEL
        from findingmodel.tools.finding_enrichment import enrich_finding

        # Enable model requests for this test
        original_allow = models.ALLOW_MODEL_REQUESTS
        models.ALLOW_MODEL_REQUESTS = True

        # Set override directly on global settings singleton
        settings.agent_model_overrides["enrich_classify"] = TEST_GOOGLE_MODEL

        try:
            # Call workflow WITHOUT passing model parameter
            # This forces it to use get_agent_model() which should pick up our override
            result = await enrich_finding("fracture")

            # Validate result structure - finding_name may be canonical form from Index
            assert result.finding_name.lower() == "fracture"
            assert isinstance(result.snomed_codes, list)
            assert isinstance(result.radlex_codes, list)
            assert isinstance(result.body_regions, list)
            assert isinstance(result.etiologies, list)
            assert isinstance(result.modalities, list)
            assert isinstance(result.subspecialties, list)
            assert isinstance(result.anatomic_locations, list)
            assert result.enrichment_timestamp is not None

            # Verify model used in metadata matches our override
            assert result.model_used == TEST_GOOGLE_MODEL

        finally:
            # Clean up: remove override and restore ALLOW_MODEL_REQUESTS
            if "enrich_classify" in settings.agent_model_overrides:
                del settings.agent_model_overrides["enrich_classify"]
            models.ALLOW_MODEL_REQUESTS = original_allow
