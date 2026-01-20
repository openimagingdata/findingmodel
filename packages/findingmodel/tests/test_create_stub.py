"""Tests for create_stub module."""

import pytest
from findingmodel.create_stub import create_model_stub_from_info
from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import FindingModelBase


@pytest.fixture
def finding_info() -> FindingInfo:
    """Create a FindingInfo fixture for testing."""
    return FindingInfo(
        name="test finding",
        description="A test finding for testing.",
        synonyms=["test", "finding"],
    )


def test_create_stub(finding_info: FindingInfo) -> None:
    """Test creating a stub finding model from a FindingInfo object."""
    stub = create_model_stub_from_info(finding_info)
    assert isinstance(stub, FindingModelBase)
    assert stub.name == finding_info.name.lower()
    assert stub.description == finding_info.description
    assert stub.synonyms == finding_info.synonyms
    assert len(stub.attributes) == 2
    assert stub.attributes[0].name == "presence"
    assert stub.attributes[1].name == "change from prior"


def test_create_stub_with_tags(finding_info: FindingInfo) -> None:
    """Test creating a stub finding model with tags."""
    tags = ["tag1", "tag2"]
    stub = create_model_stub_from_info(finding_info, tags=tags)
    assert stub.tags == tags


def test_create_stub_presence_values(finding_info: FindingInfo) -> None:
    """Test that presence attribute has correct values."""
    stub = create_model_stub_from_info(finding_info)
    presence = stub.attributes[0]
    assert presence.name == "presence"
    value_names = [v.name for v in presence.values]
    assert "absent" in value_names
    assert "present" in value_names
    assert "indeterminate" in value_names
    assert "unknown" in value_names


def test_create_stub_change_values(finding_info: FindingInfo) -> None:
    """Test that change from prior attribute has correct values."""
    stub = create_model_stub_from_info(finding_info)
    change = stub.attributes[1]
    assert change.name == "change from prior"
    value_names = [v.name for v in change.values]
    assert "unchanged" in value_names
    assert "new" in value_names
    assert "resolved" in value_names
    assert "increased" in value_names
    assert "decreased" in value_names
