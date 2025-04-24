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
