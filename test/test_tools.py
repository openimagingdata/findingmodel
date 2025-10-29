from pathlib import Path
from types import SimpleNamespace

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

import findingmodel.tools
import findingmodel.tools.finding_description as finding_description
from findingmodel import FindingInfo, FindingModelBase, FindingModelFull, logger
from findingmodel.config import settings
from findingmodel.finding_model import AttributeType, ChoiceAttribute, ChoiceAttributeIded
from findingmodel.index_code import IndexCode

HAS_PERPLEXITY_API_KEY = bool(settings.perplexity_api_key.get_secret_value())


def _mongodb_available(uri: str) -> bool:
    client: MongoClient | None = None  # type: ignore[type-arg]
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=1000, connectTimeoutMS=1000)
        client.admin.command("ping")
        return True
    except (PyMongoError, OSError):
        return False
    finally:
        if client is not None:
            client.close()


HAS_MONGODB = _mongodb_available("mongodb://localhost:27017")


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
        lambda model_name, instructions: stub_agent,
    )
    monkeypatch.setattr(
        settings.__class__,
        "check_ready_for_openai",
        lambda self: True,
    )

    logged: list[str] = []

    def fake_info(message: str, *args: object, **kwargs: object) -> None:
        if args:
            message = message.format(*args)
        logged.append(message)

    monkeypatch.setattr(logger, "info", fake_info)

    result = await finding_description.create_info_from_name("left lower lobe opacity", model_name="stub-model")

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
        lambda model_name, instructions: stub_agent,
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

    result = await finding_description.create_info_from_name("pneumothorax")

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
async def test_create_info_from_name_integration() -> None:
    """Integration test for create_info_from_name with real OpenAI API."""
    from findingmodel.finding_info import FindingInfo
    from findingmodel.tools import create_info_from_name

    # Test with a common medical finding
    result = await create_info_from_name("pneumothorax")

    assert isinstance(result, FindingInfo)
    assert result.name.lower() == "pneumothorax"
    assert result.description is not None
    assert len(result.description) > 10  # Should have meaningful description

    # Should typically have synonyms for pneumothorax
    assert result.synonyms is not None
    assert len(result.synonyms) > 0

    # Common synonyms for pneumothorax - accept any of these valid alternatives
    synonyms_lower = [s.lower() for s in result.synonyms]
    common_synonyms = ["ptx", "collapsed lung", "pneumo"]
    assert any(syn in synonyms_lower for syn in common_synonyms), (
        f"Expected one of {common_synonyms}, got: {synonyms_lower}"
    )


@pytest.mark.callout
@pytest.mark.asyncio
async def test_create_info_from_name_edge_cases() -> None:
    """Test create_info_from_name with edge cases."""
    from findingmodel.tools import create_info_from_name

    # Test with less common finding
    result = await create_info_from_name("thyroid nodule")
    assert result.name.lower() == "thyroid nodule"
    assert result.description is not None

    # Test with very specific finding
    result = await create_info_from_name("pulmonary embolism")
    assert result.name.lower() == "pulmonary embolism"
    assert result.description is not None


@pytest.mark.callout
@pytest.mark.skipif(not HAS_PERPLEXITY_API_KEY, reason="Perplexity API key not configured")
@pytest.mark.asyncio
async def test_add_details_to_info_integration() -> None:
    """Integration test for add_details_to_info with real Perplexity API."""
    from findingmodel.finding_info import FindingInfo
    from findingmodel.tools import add_details_to_info

    # Start with basic finding info
    basic_info = FindingInfo(name="pneumothorax", description="Presence of air in the pleural space", synonyms=["PTX"])

    # Add detailed information
    detailed_info = await add_details_to_info(basic_info)

    assert isinstance(detailed_info, FindingInfo)
    assert detailed_info.name == basic_info.name
    assert detailed_info.description == basic_info.description
    assert detailed_info.synonyms == basic_info.synonyms

    # Should have added detailed information
    assert detailed_info.detail is not None
    assert len(detailed_info.detail) > len(basic_info.description or "")

    # Should have citations
    assert detailed_info.citations is not None
    assert len(detailed_info.citations) > 0

    # Citations should be valid URLs
    for citation in detailed_info.citations:
        assert citation.startswith(("http://", "https://")) or "http" in citation


@pytest.mark.callout
@pytest.mark.asyncio
async def test_create_model_from_markdown_integration() -> None:
    """Integration test for create_model_from_markdown with real OpenAI API."""
    from findingmodel.finding_model import FindingModelBase
    from findingmodel.tools import create_info_from_name, create_model_from_markdown

    # First create basic info
    finding_info = await create_info_from_name("pneumothorax")

    # Define markdown outline
    markdown_text = """
    # Pneumothorax Attributes
    
    ## Size
    - Small: Less than 2cm
    - Moderate: 2-4cm  
    - Large: Greater than 4cm
    
    ## Location
    - Apical
    - Basilar
    - Complete
    
    ## Tension
    - Present
    - Absent
    """

    # Create model from markdown
    model = await create_model_from_markdown(finding_info, markdown_text=markdown_text)

    assert isinstance(model, FindingModelBase)
    # Compare names case-insensitively to account for LLM casing variability
    assert model.name.strip().lower() == finding_info.name.strip().lower()
    # AI may generate slightly different descriptions, so check that both contain key concepts
    assert model.description is not None
    assert finding_info.description is not None
    # Both should mention pneumothorax and lung/pleural concepts
    model_desc_lower = model.description.lower()
    # info_desc_lower = finding_info.description.lower()
    assert "pneumothorax" in model_desc_lower or "air" in model_desc_lower
    assert "lung" in model_desc_lower or "pleural" in model_desc_lower
    assert model.synonyms == finding_info.synonyms

    # Should have created attributes
    assert len(model.attributes) >= 3  # Size, Location, Tension

    # Check attribute names - use flexible matching for semantically equivalent terms
    attr_names = [attr.name.lower() for attr in model.attributes]
    assert any("size" in name for name in attr_names), f"Expected size-related attribute, got: {attr_names}"
    assert any("location" in name for name in attr_names), f"Expected location-related attribute, got: {attr_names}"
    assert any("tension" in name for name in attr_names), f"Expected tension-related attribute, got: {attr_names}"

    # Check that choice attributes have values
    for attr in model.attributes:
        if attr.type.value == "choice":
            assert isinstance(attr, ChoiceAttribute)
            assert len(attr.values) > 0


@pytest.mark.callout
@pytest.mark.asyncio
async def test_create_model_from_markdown_file_integration(tmp_path: Path) -> None:
    """Integration test for create_model_from_markdown using file input."""
    from findingmodel.tools import create_info_from_name, create_model_from_markdown

    # Create markdown file
    markdown_content = """
    # Heart Murmur Assessment
    
    ## Intensity
    - Grade 1: Very faint
    - Grade 2: Soft but audible
    - Grade 3: Moderately loud
    - Grade 4: Loud with thrill
    
    ## Timing
    - Systolic
    - Diastolic
    - Continuous
    """

    markdown_file = tmp_path / "heart_murmur.md"
    markdown_file.write_text(markdown_content)

    # Create finding info
    finding_info = await create_info_from_name("heart murmur")

    # Create model from file
    model = await create_model_from_markdown(finding_info, markdown_path=markdown_file)

    # Compare names case-insensitively (AI might normalize case differently)
    assert model.name.lower() == finding_info.name.lower()
    assert len(model.attributes) >= 2  # Intensity, Timing

    # Check for specific attributes
    attr_names = [attr.name.lower() for attr in model.attributes]
    assert "intensity" in attr_names
    assert "timing" in attr_names


@pytest.mark.callout
@pytest.mark.asyncio
async def test_create_info_from_name_integration_normalizes_output() -> None:
    """Ensure create_info_from_name returns normalized data when using the real API."""
    from findingmodel.tools import create_info_from_name

    raw_input = " PCL tear "
    result = await create_info_from_name(raw_input)

    assert result.name == result.name.strip()
    assert result.name, "Finding name should not be empty"

    synonyms = result.synonyms or []
    assert all(syn == syn.strip() for syn in synonyms)
    assert len({syn.casefold() for syn in synonyms}) == len(synonyms)

    original_term = raw_input.strip()
    if result.name.casefold() != original_term.casefold():
        assert original_term in synonyms


@pytest.mark.callout
@pytest.mark.skipif(not HAS_PERPLEXITY_API_KEY, reason="Perplexity API key not configured")
@pytest.mark.asyncio
async def test_ai_tools_error_handling() -> None:
    """Test AI tools error handling with invalid inputs."""
    from findingmodel.finding_info import FindingInfo
    from findingmodel.tools import add_details_to_info, create_info_from_name

    # Test with very unusual/nonsensical input
    result = await create_info_from_name("xyznonsensicalmedicalterm")
    # Should still return a FindingInfo object, even if description is generic
    assert isinstance(result, FindingInfo)
    assert result.name == "xyznonsensicalmedicalterm"

    # Test add_details_to_info with minimal input
    minimal_info = FindingInfo(name="test", description="test description")
    detailed = await add_details_to_info(minimal_info)
    assert isinstance(detailed, FindingInfo)
    assert detailed.name == minimal_info.name


# Tests for find_similar_models function
@pytest.mark.skipif(not HAS_MONGODB, reason="MongoDB not available for find_similar_models tests")
def test_find_similar_models_basic_functionality() -> None:
    """Test basic functionality of find_similar_models without API calls."""
    from findingmodel.tools import find_similar_models

    # Test the function exists and has the correct signature
    # This test will skip if it requires API calls or database doesn't exist
    try:
        import asyncio

        # Call with minimal parameters to test existence
        result = asyncio.run(find_similar_models("pneumothorax"))

        # Should return a SimilarModelAnalysis object
        from findingmodel.tools.similar_finding_models import SimilarModelAnalysis

        assert isinstance(result, SimilarModelAnalysis)

    except Exception as e:
        # If it requires API calls or database doesn't exist, we'll handle that in the callout tests
        if (
            "API" in str(e)
            or "key" in str(e).lower()
            or "openai" in str(e).lower()
            or "database does not exist" in str(e).lower()
        ):
            pytest.skip(f"Skipping test that requires API access or database: {e}")
        else:
            raise


@pytest.mark.callout
@pytest.mark.skipif(not HAS_MONGODB, reason="MongoDB not available for find_similar_models tests")
@pytest.mark.asyncio
async def test_find_similar_models_integration() -> None:
    """Integration test for find_similar_models with real OpenAI API."""
    from findingmodel.tools import find_similar_models
    from findingmodel.tools.similar_finding_models import SimilarModelAnalysis

    # Test with a common medical finding
    result = await find_similar_models(
        finding_name="pneumothorax", description="Presence of air in the pleural space causing lung collapse"
    )

    # Verify structure
    assert isinstance(result, SimilarModelAnalysis)

    # Should have the required fields
    assert hasattr(result, "similar_models")
    assert hasattr(result, "recommendation")
    assert hasattr(result, "confidence")

    # The result should be valid
    assert result.recommendation in ["edit_existing", "create_new"]
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.similar_models, list)


@pytest.mark.callout
@pytest.mark.skipif(not HAS_MONGODB, reason="MongoDB not available for find_similar_models tests")
@pytest.mark.asyncio
async def test_find_similar_models_edge_cases() -> None:
    """Test find_similar_models with edge cases."""
    from findingmodel.tools import find_similar_models
    from findingmodel.tools.similar_finding_models import SimilarModelAnalysis

    # Use faster models for edge case testing
    small_model = "gpt-4o-mini"  # Use faster model for edge cases
    analysis_model = "gpt-4o-mini"  # Use faster model for analysis too

    # Test with minimal input
    result = await find_similar_models(
        finding_name="test finding", search_model=small_model, analysis_model=analysis_model
    )
    assert isinstance(result, SimilarModelAnalysis)

    # Test with empty description
    result_empty_desc = await find_similar_models(
        finding_name="test", description="", search_model=small_model, analysis_model=analysis_model
    )
    assert isinstance(result_empty_desc, SimilarModelAnalysis)

    # Test with very long finding name
    long_name = "very long finding name " * 20
    result_long = await find_similar_models(
        finding_name=long_name, search_model=small_model, analysis_model=analysis_model
    )
    assert isinstance(result_long, SimilarModelAnalysis)


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
