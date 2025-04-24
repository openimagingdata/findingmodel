from findingmodel.finding_model import (
    AttributeType,
    ChoiceAttribute,
    ChoiceValue,
    FindingModelBase,
    NumericAttribute,
)


def test_choice_value() -> None:
    value = ChoiceValue(name="Severe", description="A severe finding")
    assert value.name == "Severe"
    assert value.description == "A severe finding"


def test_choice_attribute() -> None:
    value1 = ChoiceValue(name="Mild")
    value2 = ChoiceValue(name="Severe")
    attribute = ChoiceAttribute(
        name="Severity",
        values=[value1, value2],
        required=True,
    )
    assert attribute.name == "Severity"
    assert attribute.type == AttributeType.CHOICE
    assert len(attribute.values) == 2
    assert attribute.required is True
    assert attribute.max_selected == 1


def test_numeric_attribute() -> None:
    attribute = NumericAttribute(
        name="Size",
        minimum=0,
        maximum=10,
        unit="cm",
    )
    assert attribute.name == "Size"
    assert attribute.type == AttributeType.NUMERIC
    assert attribute.minimum == 0
    assert attribute.maximum == 10
    assert attribute.unit == "cm"
    assert attribute.required is False


def test_multichoice_attribute() -> None:
    value1 = ChoiceValue(name="Mild")
    value2 = ChoiceValue(name="Moderate")
    value3 = ChoiceValue(name="Severe")
    attribute = ChoiceAttribute(name="Severity", values=[value1, value2, value3], max_selected=2)
    assert attribute.name == "Severity"
    assert attribute.type == AttributeType.CHOICE
    assert len(attribute.values) == 3
    assert attribute.max_selected == 2
    assert attribute.required is False


def test_finding_model_base(tmp_path) -> None:  # type: ignore # noqa: ANN001
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()
    choice_values = [ChoiceValue(name="Mild"), ChoiceValue(name="Moderate"), ChoiceValue(name="Severe")]
    choice_attribute = ChoiceAttribute(
        name="Severity",
        values=choice_values,
        required=True,
    )
    numeric_attribute = NumericAttribute(
        name="Size",
        minimum=0,
        maximum=10,
        unit="cm",
        required=False,
    )
    finding_model = FindingModelBase(
        name="ExampleFinding",
        description="An example finding for testing.",
        attributes=[choice_attribute, numeric_attribute],
    )
    json_file = test_data_dir / "example_finding_model.json"
    with open(json_file, "w") as f:
        f.write(finding_model.model_dump_json(indent=2, exclude_none=True))
    with open(json_file) as f:
        json_text = f.read()
    loaded_finding_model = FindingModelBase.model_validate_json(json_text)
    assert loaded_finding_model.name == finding_model.name
    assert loaded_finding_model.description == finding_model.description
    assert len(loaded_finding_model.attributes) == len(finding_model.attributes)


def test_load_finding_model(pe_fm_json: str) -> None:
    # Test loading a finding model from a JSON file
    loaded_model = FindingModelBase.model_validate_json(pe_fm_json)
    assert loaded_model.name == "pulmonary embolism"
    assert len(loaded_model.attributes) == 4
    assert loaded_model.attributes[0].name == "presence"
    assert loaded_model.attributes[1].name == "change from prior"
    assert loaded_model.attributes[2].name == "other presence"
    assert loaded_model.attributes[3].name == "size"
