from unittest.mock import MagicMock, patch

import httpx

import findingmodel.tools
from findingmodel import FindingInfo, FindingModelBase, FindingModelFull
from findingmodel.finding_model import AttributeType, ChoiceAttributeIded
from findingmodel.index_code import IndexCode


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


def test_add_ids_with_empty_cache(base_model: FindingModelBase) -> None:
    """Test adding IDs when cache is empty (first call)."""
    mock_data: IdsJsonType = {"oifm_ids": {}, "attribute_ids": {}}

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Clear the cache before testing
        findingmodel.tools.OIFM_IDS.clear()
        findingmodel.tools.ATTRIBUTE_IDS.clear()

        updated_model = findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")

        assert isinstance(updated_model, FindingModelFull)
        assert updated_model.oifm_id is not None
        assert updated_model.oifm_id.startswith("OIFM_")
        assert "TEST" in updated_model.oifm_id

        # Verify HTTP call was made
        mock_client.return_value.__enter__.return_value.get.assert_called_once()


def test_add_ids_with_populated_cache(base_model: FindingModelBase) -> None:
    """Test adding IDs when cache already has data (avoids duplicate IDs)."""
    existing_oifm_id = "OIFM_TEST_12345"
    existing_attr_id = "OIFMA_TEST_67890"

    mock_data = {
        "oifm_ids": {existing_oifm_id: "Existing Model"},
        "attribute_ids": {existing_attr_id: ("OIFM_TEST_12345", "existing_attr")},
    }

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Clear and populate cache
        findingmodel.tools.OIFM_IDS.clear()
        findingmodel.tools.ATTRIBUTE_IDS.clear()

        updated_model = findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")

        # Ensure new IDs don't conflict with existing ones
        assert updated_model.oifm_id != existing_oifm_id
        for attr in updated_model.attributes:
            assert attr.oifma_id != existing_attr_id


def test_add_ids_uses_cache_on_second_call(base_model: FindingModelBase) -> None:
    """Test that second call uses cache and doesn't make HTTP request."""
    mock_data: IdsJsonType = {"oifm_ids": {}, "attribute_ids": {}}

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Clear cache first
        findingmodel.tools.OIFM_IDS.clear()
        findingmodel.tools.ATTRIBUTE_IDS.clear()

        # First call - should make HTTP request
        findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")
        assert mock_client.return_value.__enter__.return_value.get.call_count == 1

        # Second call - should use cache
        findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")
        assert mock_client.return_value.__enter__.return_value.get.call_count == 1  # Still 1


def test_add_ids_handles_http_timeout(base_model: FindingModelBase) -> None:
    """Test that function handles HTTP timeout gracefully."""
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = httpx.TimeoutException("Timeout")

        # Clear cache
        findingmodel.tools.OIFM_IDS.clear()
        findingmodel.tools.ATTRIBUTE_IDS.clear()

        # Should still work with empty cache when HTTP fails
        updated_model = findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")

        assert isinstance(updated_model, FindingModelFull)
        assert updated_model.oifm_id is not None


def test_add_ids_handles_http_error(base_model: FindingModelBase) -> None:
    """Test that function handles HTTP errors gracefully."""
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = httpx.HTTPError("HTTP Error")

        # Clear cache
        findingmodel.tools.OIFM_IDS.clear()
        findingmodel.tools.ATTRIBUTE_IDS.clear()

        # Should still work with empty cache when HTTP fails
        updated_model = findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")

        assert isinstance(updated_model, FindingModelFull)
        assert updated_model.oifm_id is not None


def test_add_ids_refresh_cache(base_model: FindingModelBase) -> None:
    """Test forcing cache refresh."""
    mock_data = {"oifm_ids": {"OIFM_REFRESH_TEST": "Test Model"}, "attribute_ids": {}}

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Populate cache first
        findingmodel.tools.OIFM_IDS = {"existing": "model"}
        findingmodel.tools.ATTRIBUTE_IDS = {"existing": ("oifm", "attr")}

        # Force refresh
        with patch("findingmodel.tools.load_used_ids_from_github") as mock_load:
            findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")
            mock_load.assert_called_once()


def test_load_used_ids_from_github_directly() -> None:
    """Test the load_used_ids_from_github function directly."""
    mock_data = {
        "oifm_ids": {"OIFM_TEST_123": "Test Model"},
        "attribute_ids": {"OIFMA_TEST_456": ["OIFM_TEST_123", "test_attr"]},
    }

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        # Clear cache
        findingmodel.tools.OIFM_IDS.clear()
        findingmodel.tools.ATTRIBUTE_IDS.clear()

        findingmodel.tools.load_used_ids_from_github()

        assert "OIFM_TEST_123" in findingmodel.tools.OIFM_IDS
        assert "OIFMA_TEST_456" in findingmodel.tools.ATTRIBUTE_IDS
        assert findingmodel.tools.ATTRIBUTE_IDS["OIFMA_TEST_456"] == ("OIFM_TEST_123", "test_attr")


def test_load_used_ids_with_custom_url() -> None:
    """Test load_used_ids_from_github with custom URL."""
    custom_url = "https://example.com/custom-ids.json"
    mock_data: IdsJsonType = {"oifm_ids": {}, "attribute_ids": {}}

    original_oifm_ids = findingmodel.tools.OIFM_IDS.copy()
    original_attribute_ids = findingmodel.tools.ATTRIBUTE_IDS.copy()
    findingmodel.tools.OIFM_IDS.clear()
    findingmodel.tools.ATTRIBUTE_IDS.clear()
    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        findingmodel.tools.load_used_ids_from_github(url=custom_url)

        mock_client.return_value.__enter__.return_value.get.assert_called_with(custom_url, timeout=5.0)
    findingmodel.tools.OIFM_IDS = original_oifm_ids
    findingmodel.tools.ATTRIBUTE_IDS = original_attribute_ids


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
