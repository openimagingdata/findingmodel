from unittest.mock import MagicMock, patch

import httpx

import findingmodel.tools
from findingmodel import FindingInfo, FindingModelBase, FindingModelFull
from findingmodel.finding_model import AttributeType, ChoiceAttributeIded
from findingmodel.index_code import IndexCode
from findingmodel.tools.add_ids import IdManager


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
        findingmodel.tools.id_manager.oifm_ids.clear()
        findingmodel.tools.id_manager.attribute_ids.clear()

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
        findingmodel.tools.id_manager.oifm_ids.clear()
        findingmodel.tools.id_manager.attribute_ids.clear()

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
        findingmodel.tools.id_manager.oifm_ids.clear()
        findingmodel.tools.id_manager.attribute_ids.clear()

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
        findingmodel.tools.id_manager.oifm_ids.clear()
        findingmodel.tools.id_manager.attribute_ids.clear()

        # Should still work with empty cache when HTTP fails
        updated_model = findingmodel.tools.add_ids_to_finding_model(base_model, source="TEST")

        assert isinstance(updated_model, FindingModelFull)
        assert updated_model.oifm_id is not None


def test_add_ids_handles_http_error(base_model: FindingModelBase) -> None:
    """Test that function handles HTTP errors gracefully."""
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = httpx.HTTPError("HTTP Error")

        # Clear cache
        findingmodel.tools.id_manager.oifm_ids.clear()
        findingmodel.tools.id_manager.attribute_ids.clear()

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
        findingmodel.tools.id_manager.oifm_ids.clear()
        findingmodel.tools.id_manager.oifm_ids.update({"existing": "model"})
        findingmodel.tools.id_manager.attribute_ids.clear()
        findingmodel.tools.id_manager.attribute_ids.update({"existing": ("oifm", "attr")})

        # Force refresh by calling load_used_ids_from_github with refresh_cache=True
        findingmodel.tools.id_manager.load_used_ids_from_github(refresh_cache=True)
        assert "OIFM_REFRESH_TEST" in findingmodel.tools.id_manager.oifm_ids


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
        findingmodel.tools.id_manager.oifm_ids.clear()
        findingmodel.tools.id_manager.attribute_ids.clear()

        findingmodel.tools.id_manager.load_used_ids_from_github(refresh_cache=True)

        assert "OIFM_TEST_123" in findingmodel.tools.id_manager.oifm_ids
        assert "OIFMA_TEST_456" in findingmodel.tools.id_manager.attribute_ids
        assert findingmodel.tools.id_manager.attribute_ids["OIFMA_TEST_456"] == ("OIFM_TEST_123", "test_attr")


def test_load_used_ids_with_custom_url() -> None:
    """Test load_used_ids_from_github with custom URL."""
    custom_url = "https://example.com/custom-ids.json"
    mock_data: IdsJsonType = {"oifm_ids": {}, "attribute_ids": {}}

    custom_id_manager = IdManager(url=custom_url)

    with patch("httpx.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response

        custom_id_manager.load_used_ids_from_github(refresh_cache=True)

        mock_client.return_value.__enter__.return_value.get.assert_called_with(custom_url, timeout=5.0)


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
