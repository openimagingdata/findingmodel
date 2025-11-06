from types import SimpleNamespace

import pytest
from pydantic_ai import models

import findingmodel.tools
import findingmodel.tools.finding_description as finding_description
from findingmodel import FindingInfo, FindingModelBase, FindingModelFull, logger
from findingmodel.config import settings
from findingmodel.finding_model import AttributeType, ChoiceAttributeIded
from findingmodel.index_code import IndexCode

# Prevent accidental model requests in unit tests
# Tests marked with @pytest.mark.callout can enable this as needed
models.ALLOW_MODEL_REQUESTS = False

HAS_TAVILY_API_KEY = bool(settings.tavily_api_key.get_secret_value())
HAS_OPENAI_API_KEY = bool(settings.openai_api_key.get_secret_value())
HAS_ANTHROPIC_API_KEY = bool(settings.anthropic_api_key.get_secret_value())


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
    from jinja2 import Template

    from findingmodel.tools.prompt_template import render_agent_prompt

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
    from jinja2 import Template

    from findingmodel.tools.prompt_template import render_agent_prompt

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
        lambda model_tier, instructions, provider=None: stub_agent,
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
        lambda model_tier, instructions, provider=None: stub_agent,
    )
    monkeypatch.setattr(
        settings.__class__,
        "check_ready_for_openai",
        lambda self: True,
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
    from unittest.mock import AsyncMock

    from pydantic_ai.models.test import TestModel

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

    # Mock get_model to return TestModel
    monkeypatch.setattr("findingmodel.tools.finding_description.get_model", lambda model_tier: TestModel())

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
    from unittest.mock import AsyncMock

    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    from findingmodel.tools import add_details_to_info

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

    # Mock get_model
    monkeypatch.setattr("findingmodel.tools.finding_description.get_model", lambda model_tier: TestModel())

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
    from unittest.mock import AsyncMock

    from pydantic_ai.models.test import TestModel

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

    # Mock get_model to use TestModel that will call tools
    test_model = TestModel()
    monkeypatch.setattr("findingmodel.tools.finding_description.get_model", lambda model_tier: test_model)

    # Enable model requests for this test since we're using TestModel
    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", True)

    # Run add_details_to_info with the specified search_depth
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

    from pydantic_ai.models.test import TestModel

    from findingmodel.finding_model import ChoiceAttribute, ChoiceValue, FindingModelBase
    from findingmodel.tools import create_model_from_markdown

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

    # Mock get_model to return TestModel with controlled output
    # This prevents actual OpenAI client creation
    with patch("findingmodel.tools.markdown_in.get_model") as mock_model:
        # TestModel requires custom_output_args to be a dict, not a Pydantic model
        mock_model.return_value = TestModel(custom_output_args=test_output.model_dump())

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
    from pydantic import SecretStr
    from tavily import AsyncTavilyClient

    from findingmodel import config
    from findingmodel.tools.common import get_async_tavily_client

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
    from pydantic import SecretStr

    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from findingmodel.tools.common import get_async_tavily_client

    # Temporarily override settings with empty key
    original_key = config.settings.tavily_api_key
    try:
        config.settings.tavily_api_key = SecretStr("")
        with pytest.raises(ConfigurationError, match="Tavily API key is not set"):
            get_async_tavily_client()
    finally:
        config.settings.tavily_api_key = original_key


# Tests for get_model() function


def test_get_model_openai_base_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() with OpenAI provider and base tier."""
    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIModel

    from findingmodel import config
    from findingmodel.tools.common import get_model

    # Set environment variable for OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Temporarily override settings with test key
    original_key = config.settings.openai_api_key
    original_provider = config.settings.model_provider
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        config.settings.model_provider = "openai"
        model = get_model("base")
        assert isinstance(model, OpenAIModel)
    finally:
        config.settings.openai_api_key = original_key
        config.settings.model_provider = original_provider


def test_get_model_openai_small_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() with OpenAI provider and small tier."""
    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIModel

    from findingmodel import config
    from findingmodel.tools.common import get_model

    # Set environment variable for OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Temporarily override settings with test key
    original_key = config.settings.openai_api_key
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        model = get_model("small", provider="openai")
        assert isinstance(model, OpenAIModel)
    finally:
        config.settings.openai_api_key = original_key


def test_get_model_openai_full_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() with OpenAI provider and full tier."""
    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIModel

    from findingmodel import config
    from findingmodel.tools.common import get_model

    # Set environment variable for OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Temporarily override settings with test key
    original_key = config.settings.openai_api_key
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        model = get_model("full", provider="openai")
        assert isinstance(model, OpenAIModel)
    finally:
        config.settings.openai_api_key = original_key


def test_get_model_anthropic_base_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() with Anthropic provider and base tier."""
    from pydantic import SecretStr
    from pydantic_ai.models.anthropic import AnthropicModel

    from findingmodel import config
    from findingmodel.tools.common import get_model

    # Set environment variable for Anthropic client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    # Temporarily override settings with test key
    original_key = config.settings.anthropic_api_key
    try:
        config.settings.anthropic_api_key = SecretStr("test-anthropic-key")
        model = get_model("base", provider="anthropic")
        assert isinstance(model, AnthropicModel)
    finally:
        config.settings.anthropic_api_key = original_key


def test_get_model_anthropic_small_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() with Anthropic provider and small tier."""
    from pydantic import SecretStr
    from pydantic_ai.models.anthropic import AnthropicModel

    from findingmodel import config
    from findingmodel.tools.common import get_model

    # Set environment variable for Anthropic client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    # Temporarily override settings with test key
    original_key = config.settings.anthropic_api_key
    try:
        config.settings.anthropic_api_key = SecretStr("test-anthropic-key")
        model = get_model("small", provider="anthropic")
        assert isinstance(model, AnthropicModel)
    finally:
        config.settings.anthropic_api_key = original_key


def test_get_model_anthropic_full_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() with Anthropic provider and full tier."""
    from pydantic import SecretStr
    from pydantic_ai.models.anthropic import AnthropicModel

    from findingmodel import config
    from findingmodel.tools.common import get_model

    # Set environment variable for Anthropic client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    # Temporarily override settings with test key
    original_key = config.settings.anthropic_api_key
    try:
        config.settings.anthropic_api_key = SecretStr("test-anthropic-key")
        model = get_model("full", provider="anthropic")
        assert isinstance(model, AnthropicModel)
    finally:
        config.settings.anthropic_api_key = original_key


def test_get_model_uses_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() uses default provider from settings."""
    from pydantic import SecretStr
    from pydantic_ai.models.anthropic import AnthropicModel

    from findingmodel import config
    from findingmodel.tools.common import get_model

    # Set environment variable for Anthropic client
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    # Temporarily override settings with test key and provider
    original_key = config.settings.anthropic_api_key
    original_provider = config.settings.model_provider
    try:
        config.settings.anthropic_api_key = SecretStr("test-anthropic-key")
        config.settings.model_provider = "anthropic"
        model = get_model("base")
        assert isinstance(model, AnthropicModel)
    finally:
        config.settings.anthropic_api_key = original_key
        config.settings.model_provider = original_provider


def test_get_model_unsupported_provider_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() raises ConfigurationError for unsupported provider."""
    from pydantic import SecretStr

    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from findingmodel.tools.common import get_model

    # Set environment variable for OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Temporarily override settings with test key
    original_key = config.settings.openai_api_key
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        # Use type: ignore to bypass type checking for test
        with pytest.raises(ConfigurationError, match="Unsupported model provider"):
            get_model("base", provider="invalid")  # type: ignore
    finally:
        config.settings.openai_api_key = original_key


def test_get_model_missing_api_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_model() raises ConfigurationError when API key missing."""
    from pydantic import SecretStr

    from findingmodel import config
    from findingmodel.config import ConfigurationError
    from findingmodel.tools.common import get_model

    # Temporarily override settings with empty key
    original_key = config.settings.openai_api_key
    try:
        config.settings.openai_api_key = SecretStr("")
        with pytest.raises(ConfigurationError, match="OpenAI API key is not set"):
            get_model("base", provider="openai")
    finally:
        config.settings.openai_api_key = original_key


# Tests for get_openai_model() deprecation


def test_get_openai_model_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_openai_model() emits DeprecationWarning."""
    import warnings

    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIModel

    from findingmodel import config
    from findingmodel.tools.common import get_openai_model

    # Set environment variable for OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Temporarily override settings with test key
    original_key = config.settings.openai_api_key
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = get_openai_model("gpt-5-mini")
            # Check that at least one deprecation warning was emitted
            assert len(w) >= 1
            # Find our specific deprecation warning
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert any("get_openai_model() is deprecated" in str(warning.message) for warning in deprecation_warnings)
            assert isinstance(model, OpenAIModel)
    finally:
        config.settings.openai_api_key = original_key


def test_get_openai_model_none_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_openai_model(None) uses default model."""
    import warnings

    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIModel

    from findingmodel import config
    from findingmodel.tools.common import get_openai_model

    # Set environment variable for OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Temporarily override settings with test key
    original_key = config.settings.openai_api_key
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = get_openai_model(None)
            assert isinstance(model, OpenAIModel)
    finally:
        config.settings.openai_api_key = original_key


def test_get_openai_model_explicit_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_openai_model("specific-model") uses explicit name."""
    import warnings

    from pydantic import SecretStr
    from pydantic_ai.models.openai import OpenAIModel

    from findingmodel import config
    from findingmodel.tools.common import get_openai_model

    # Set environment variable for OpenAI client
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # Temporarily override settings with test key
    original_key = config.settings.openai_api_key
    try:
        config.settings.openai_api_key = SecretStr("test-openai-key")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model = get_openai_model("gpt-5-nano")
            assert isinstance(model, OpenAIModel)
    finally:
        config.settings.openai_api_key = original_key


# Integration tests for Anthropic models


@pytest.mark.callout
@pytest.mark.skipif(not HAS_ANTHROPIC_API_KEY, reason="Anthropic API key not set")
@pytest.mark.asyncio
async def test_create_info_from_name_anthropic() -> None:
    """Test finding info creation with Anthropic model."""
    # Save and restore ALLOW_MODEL_REQUESTS state
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        result = await finding_description.create_info_from_name(
            "pneumothorax",
            model_tier="small",
            provider="anthropic",
        )
        assert isinstance(result, FindingInfo)
        assert result.name
        assert result.description
    finally:
        models.ALLOW_MODEL_REQUESTS = original


@pytest.mark.callout
@pytest.mark.asyncio
async def test_model_provider_switching() -> None:
    """Test switching between OpenAI and Anthropic."""
    if not (HAS_OPENAI_API_KEY and HAS_ANTHROPIC_API_KEY):
        pytest.skip("Both API keys required")

    # Save and restore ALLOW_MODEL_REQUESTS state
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True

    try:
        # Test OpenAI
        result_openai = await finding_description.create_info_from_name("pneumonia", provider="openai")
        assert isinstance(result_openai, FindingInfo)

        # Test Anthropic
        result_anthropic = await finding_description.create_info_from_name("pneumonia", provider="anthropic")
        assert isinstance(result_anthropic, FindingInfo)
    finally:
        models.ALLOW_MODEL_REQUESTS = original


def test_model_tier_selection() -> None:
    """Test that model tiers select correct models."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIModel

    from findingmodel.tools.common import get_model

    # OpenAI tiers
    model = get_model("small", provider="openai")
    assert isinstance(model, OpenAIModel)

    # Anthropic tiers (skip if no key)
    if HAS_ANTHROPIC_API_KEY:
        model = get_model("small", provider="anthropic")
        assert isinstance(model, AnthropicModel)
