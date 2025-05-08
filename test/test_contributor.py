from pathlib import Path
from typing import Generator

import pytest
from pydantic import ValidationError

from findingmodel.contributor import Organization, Person

# --- Fixtures ---


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
