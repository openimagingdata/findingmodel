"""Tests for core Pydantic models: FindingModel, Contributor classes."""

from pathlib import Path
from typing import Generator

import pytest
from findingmodel.contributor import Organization, Person
from findingmodel.finding_model import (
    AttributeType,
    ChoiceAttribute,
    ChoiceValue,
    FindingModelBase,
    FindingModelFull,
    NumericAttribute,
)
from pydantic import ValidationError

# ============================================================================
# Core Model Tests
# ============================================================================


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


def test_load_finding_model_with_codes(tn_fm_json: str) -> None:
    # Test loading a finding model with codes
    loaded_model = FindingModelFull.model_validate_json(tn_fm_json)
    assert loaded_model.name == "thyroid nodule"
    assert loaded_model.index_codes is not None
    assert len(loaded_model.index_codes) == 4
    radlex_code, snomed_code, loinc_code, icd10_code = loaded_model.index_codes
    assert radlex_code.system == "RADLEX"
    assert radlex_code.code.startswith("RID")
    assert snomed_code.system == "SNOMED"
    assert snomed_code.code == "237495005"
    assert loinc_code.system == "LOINC"
    assert loinc_code.code.startswith("LA")
    assert icd10_code.system == "ICD10CM"
    assert icd10_code.code == "E04.1"
    assert len(loaded_model.attributes) == 3
    presence_attribute, *_ = loaded_model.attributes
    assert presence_attribute.index_codes is not None
    assert len(presence_attribute.index_codes) == 1
    code = presence_attribute.index_codes[0]
    assert code.system == "SNOMED"
    assert code.code == "705057003"
    assert presence_attribute.type == AttributeType.CHOICE
    assert len(presence_attribute.values) == 4
    absent_value, *_ = presence_attribute.values
    assert absent_value.index_codes is not None
    assert len(absent_value.index_codes) == 2
    assert absent_value.index_codes[0].system == "RADLEX"
    assert absent_value.index_codes[0].code == "RID28473"
    assert absent_value.index_codes[1].system == "SNOMED"
    assert absent_value.index_codes[1].code == "2667000"


def test_load_finding_model_with_contributors(tn_fm_json: str) -> None:
    # Test loading a finding model with contributors
    Organization.model_validate({"code": "ACR", "name": "American College of Radiology"})
    Organization.model_validate({"code": "OIDM", "name": "Open Imaging Data Model"})
    loaded_model = FindingModelFull.model_validate_json(tn_fm_json)
    assert loaded_model.name == "thyroid nodule"
    assert loaded_model.contributors is not None
    assert len(loaded_model.contributors) == 3
    john, jane, oidm = loaded_model.contributors
    assert isinstance(john, Person)
    assert john.github_username == "johndoe"
    assert john.organization_code == "OIDM"
    assert isinstance(jane, Person)
    assert jane.github_username == "janedoe"
    assert jane.organization_code == "ACR"
    assert isinstance(oidm, Organization)
    assert oidm.code == "OIDM"


# ============================================================================
# Markdown Conversion Tests
# ============================================================================

BASE_MODEL_MARKDOWN = """
# Test model

**Synonyms:** Test Synonym

**Tags:** tag1, tag2

A simple test finding model.

## Attributes

### Severity

How severe is the finding?
*(Select one)*

- **Mild**
- **Severe**

### Size

Size of the finding.
Mininum: 1
Maximum: 10
Unit: cm""".strip()


def test_base_model_markdown(base_model: FindingModelBase) -> None:
    md = base_model.as_markdown()
    # Compare line by line after stripping trailing spaces (markdown uses double-space line breaks)
    md_lines = [line.rstrip() for line in md.strip().splitlines()]
    expected_lines = [line.rstrip() for line in BASE_MODEL_MARKDOWN.splitlines()]
    assert md_lines == expected_lines
    # Spacing: no double blank lines between sections
    assert "\n\n\n" not in md


FULL_MODEL_MARKDOWN = """
# Test model—`OIFM_TEST_123456`

**Synonyms:** Test Synonym

**Tags:** tag1, tag2

A simple test finding model.

## Attributes

### Severity—`OIFMA_TEST_123456`

How severe is the finding?
*(Select one)*

- **Mild**
- **Severe**

### Size—`OIFMA_TEST_654321`

Size of the finding.
Mininum: 1
Maximum: 10
Unit: cm""".strip()


def test_full_model_markdown_with_ids(full_model: FindingModelFull) -> None:
    md = full_model.as_markdown()
    # Compare line by line after stripping trailing spaces (markdown uses double-space line breaks)
    md_lines = [line.rstrip() for line in md.strip().splitlines()]
    expected_lines = [line.rstrip() for line in FULL_MODEL_MARKDOWN.splitlines()]
    assert md_lines == expected_lines
    # Spacing: no double blank lines between sections
    assert "\n\n\n" not in md


def test_full_model_markdown_hide_ids(full_model: FindingModelFull) -> None:
    md = full_model.as_markdown(hide_ids=True)
    # IDs should not appear
    assert "OIFM_TEST_123456" not in md
    assert "OIFMA_TEST_123456" not in md
    assert "OIFMA_TEST_654321" not in md
    # Spacing: no double blank lines between sections
    assert "\n\n\n" not in md


def test_real_model_markdown(real_model: FindingModelFull, real_model_markdown: str) -> None:
    md = real_model.as_markdown()
    for generated_line, expected_line in zip(md.splitlines(), real_model_markdown.splitlines(), strict=True):
        assert generated_line.strip() == expected_line.strip()


def test_real_model_markdown_with_codes(tn_fm_json: str, tn_markdown: str) -> None:
    tn_model = FindingModelFull.model_validate_json(tn_fm_json)
    md = tn_model.as_markdown()
    for generated_line, expected_line in zip(md.splitlines(), tn_markdown.splitlines(), strict=True):
        assert generated_line.strip() == expected_line.strip()


# ============================================================================
# Contributor Tests
# ============================================================================


@pytest.fixture(autouse=True)
def clear_registries() -> Generator[None, None, None]:
    """Clears the registries before each test to ensure isolation."""
    Organization._org_registry.clear()
    Person._person_registry.clear()
    yield  # Test runs here
    Organization._org_registry.clear()
    Person._person_registry.clear()


@pytest.fixture
def sample_org_oidm() -> dict[str, str]:
    return {"name": "Open Imaging Data Model Project", "code": "OIDM"}


@pytest.fixture
def sample_org_acr() -> dict[str, str]:
    return {"name": "American College of Radiology", "code": "ACR", "url": "https://acr.org"}


@pytest.fixture
def sample_person_john(sample_org_oidm: dict[str, str]) -> dict[str, str]:
    # Ensure the org is created for this person
    Organization(**sample_org_oidm)  # type: ignore
    return {
        "github_username": "johndoe",
        "email": "john.doe@example.com",
        "name": "John Doe",
        "organization_code": "OIDM",
    }


@pytest.fixture
def sample_person_jane(sample_org_acr: dict[str, str]) -> dict[str, str]:
    # Ensure the org is created for this person
    Organization(**sample_org_acr)  # type: ignore
    return {
        "github_username": "janedoe",
        "email": "jane.doe@example.com",
        "name": "Jane Doe",
        "organization_code": "ACR",
        "url": "https://janedoe.example.com",
    }


# --- Organization Tests ---


def test_organization_creation(sample_org_oidm: dict[str, str]) -> None:
    org = Organization(**sample_org_oidm)  # type: ignore
    assert org.name == sample_org_oidm["name"]
    assert org.code == sample_org_oidm["code"]
    assert Organization.get(sample_org_oidm["code"]) == org
    assert len(Organization.organizations()) == 1


def test_organization_duplicate_code(sample_org_oidm: dict[str, str]) -> None:
    Organization(**sample_org_oidm)  # type: ignore
    assert len(Organization.organizations()) == 1
    Organization(name="Another Org", code=sample_org_oidm["code"])
    assert len(Organization.organizations()) == 1
    got_org = Organization.get(sample_org_oidm["code"])
    assert got_org is not None
    assert got_org.name == "Another Org"


def test_organization_invalid_code_pattern() -> None:
    with pytest.raises(ValidationError):
        Organization(name="Test Org", code="INVALID")  # Too long
    with pytest.raises(ValidationError):
        Organization(name="Test Org", code="AB")  # Too short
    with pytest.raises(ValidationError):
        Organization(name="Test Org", code="ab12")  # Invalid characters


def test_organization_organizations_list(sample_org_oidm: dict[str, str], sample_org_acr: dict[str, str]) -> None:
    org1 = Organization(**sample_org_oidm)  # type: ignore
    org2 = Organization(**sample_org_acr)  # type: ignore
    orgs = Organization.organizations()
    assert len(orgs) == 2
    assert org1 in orgs
    assert org2 in orgs
    org_gotten = Organization.get(org1.code)
    assert org_gotten is not None
    assert org_gotten.name == org1.name
    assert Organization.get("NONE") is None


def test_organization_save_and_load_jsonl(
    tmp_path: Path, sample_org_oidm: dict[str, str], sample_org_acr: dict[str, str]
) -> None:
    Organization(**sample_org_oidm)  # type: ignore
    Organization(**sample_org_acr)  # type: ignore

    jsonl_file = tmp_path / "orgs.jsonl"
    Organization.save_jsonl(jsonl_file)
    assert jsonl_file.exists()

    # Clear registry and load
    Organization._org_registry.clear()
    assert len(Organization.organizations()) == 0

    Organization.load_jsonl(jsonl_file)
    assert len(Organization.organizations()) == 2
    org1_loaded = Organization.get(sample_org_oidm["code"])
    org2_loaded = Organization.get(sample_org_acr["code"])
    assert org1_loaded is not None
    assert org1_loaded.name == sample_org_oidm["name"]
    assert org2_loaded is not None
    assert org2_loaded.name == sample_org_acr["name"]
    assert str(org2_loaded.url).startswith(sample_org_acr["url"])


def test_organization_load_jsonl_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Organization.load_jsonl(tmp_path / "non_existent.jsonl")


# --- Person Tests ---


def test_person_creation(sample_person_john: dict[str, str], sample_org_oidm: dict[str, str]) -> None:
    person = Person(**sample_person_john)  # type: ignore
    assert person.github_username == sample_person_john["github_username"]
    assert person.organization_code == sample_org_oidm["code"]
    assert Person.get(sample_person_john["github_username"]) == person
    assert len(Person.people()) == 1


def test_person_duplicate_github_username(sample_person_john: dict[str, str]) -> None:
    Person(**sample_person_john)  # type: ignore
    assert len(Person.people()) == 1
    Person(
        github_username=sample_person_john["github_username"],
        email="another.email@example.com",
        name="Another Name",
        organization_code=sample_person_john["organization_code"],
    )
    assert len(Person.people()) == 1
    got_person = Person.get(sample_person_john["github_username"])
    assert got_person is not None
    assert got_person.name == "Another Name"


def test_person_organization_property(sample_person_john: dict[str, str], sample_org_oidm: dict[str, str]) -> None:
    person = Person(**sample_person_john)  # type: ignore
    org_obj = person.organization
    assert isinstance(org_obj, Organization)
    assert org_obj.code == sample_org_oidm["code"]
    assert org_obj.name == sample_org_oidm["name"]


def test_person_organization_property_org_deleted_after_person_creation(sample_person_john: dict[str, str]) -> None:
    # This tests a more complex scenario where the org might be removed from registry
    # after the person is created but before the property is accessed.
    person = Person(**sample_person_john)  # type: ignore
    Organization._org_registry.clear()  # Simulate org being removed
    with pytest.raises(ValueError) as excinfo:
        _ = person.organization
    assert f"Organization {sample_person_john['organization_code']} not found in registry" in str(excinfo.value)


def test_person_get_non_existent() -> None:
    assert Person.get("ghost") is None


def test_person_people_list(sample_person_john: dict[str, str], sample_person_jane: dict[str, str]) -> None:
    person1 = Person(**sample_person_john)  # type: ignore
    person2 = Person(**sample_person_jane)  # type: ignore
    people_list = Person.people()
    assert len(people_list) == 2
    assert person1 in people_list
    assert person2 in people_list


def test_person_save_and_load_jsonl(
    tmp_path: Path,
    sample_person_john: dict[str, str],
    sample_person_jane: dict[str, str],
    sample_org_oidm: dict[str, str],
    sample_org_acr: dict[str, str],
) -> None:
    # Organizations need to be present for Person.load_jsonl to validate organization_code
    # Re-create them or load them if they were also saved/cleared
    Organization(**{"name": "Open Health Imaging Foundation", "code": "OHIF"})  # type: ignore
    Organization(**{"name": "Radiological Society of North America", "code": "RSNA", "url": "https://rsna.org"})  # type: ignore
    Person(**sample_person_john)  # type: ignore
    Person(**sample_person_jane)  # type: ignore

    jsonl_file = tmp_path / "people.jsonl"
    Person.save_jsonl(jsonl_file)
    assert jsonl_file.exists()

    # Clear registry and load
    Person._person_registry.clear()
    assert len(Person.people()) == 0

    Person.load_jsonl(jsonl_file)
    assert len(Person.people()) == 2
    person1_loaded = Person.get(sample_person_john["github_username"])
    person2_loaded = Person.get(sample_person_jane["github_username"])

    assert person1_loaded is not None
    assert person1_loaded.name == sample_person_john["name"]
    assert person1_loaded.organization_code == sample_person_john["organization_code"]

    assert person2_loaded is not None
    assert person2_loaded.name == sample_person_jane["name"]
    assert str(person2_loaded.url).startswith(sample_person_jane["url"])


def test_person_load_jsonl_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Person.load_jsonl(tmp_path / "non_existent_people.jsonl")
