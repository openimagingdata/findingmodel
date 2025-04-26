from pathlib import Path

import pytest

from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import (
    ChoiceAttribute,
    ChoiceAttributeIded,
    ChoiceValue,
    ChoiceValueIded,
    FindingModelBase,
    FindingModelFull,
    NumericAttribute,
    NumericAttributeIded,
)


@pytest.fixture
def base_model() -> FindingModelBase:
    return FindingModelBase(
        name="Test Model",
        description="A simple test finding model.",
        synonyms=["Test Synonym"],
        tags=["tag1", "tag2"],
        attributes=[
            ChoiceAttribute(
                name="Severity",
                description="How severe is the finding?",
                values=[ChoiceValue(name="Mild"), ChoiceValue(name="Severe")],
                required=True,
                max_selected=1,
            ),
            NumericAttribute(
                name="Size",
                description="Size of the finding.",
                minimum=1,
                maximum=10,
                unit="cm",
                required=False,
            ),
        ],
    )


@pytest.fixture
def full_model() -> FindingModelFull:
    return FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Model",
        description="A simple test finding model.",
        synonyms=["Test Synonym"],
        tags=["tag1", "tag2"],
        attributes=[
            ChoiceAttributeIded(
                oifma_id="OIFMA_TEST_123456",
                name="Severity",
                description="How severe is the finding?",
                values=[
                    ChoiceValueIded(value_code="OIFMA_TEST_123456.0", name="Mild"),
                    ChoiceValueIded(value_code="OIFMA_TEST_123456.1", name="Severe"),
                ],
                required=True,
                max_selected=1,
            ),
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_654321",
                name="Size",
                description="Size of the finding.",
                minimum=1,
                maximum=10,
                unit="cm",
                required=False,
            ),
        ],
    )


@pytest.fixture
def real_model() -> FindingModelFull:
    # Get the path to the data directory
    data_dir = Path(__file__).parent / "data"
    data = (data_dir / "pulmonary_embolism.fm.json").read_text()
    return FindingModelFull.model_validate_json(data)


@pytest.fixture
def real_model_markdown() -> str:
    data_dir = Path(__file__).parent / "data"
    data = (data_dir / "pulmonary_embolism.md").read_text()
    return data.strip()


@pytest.fixture
def pe_fm_json() -> str:
    test_data_dir = Path(__file__).parent / "data"
    json = (test_data_dir / "pulmonary_embolism.fm.json").read_text()
    return json.strip()


@pytest.fixture
def tn_fm_json() -> str:
    test_data_dir = Path(__file__).parent / "data"
    json = (test_data_dir / "thyroid_nodule_codes.fm.json").read_text()
    return json.strip()


@pytest.fixture
def tn_markdown() -> str:
    test_data_dir = Path(__file__).parent / "data"
    md = (test_data_dir / "thyroid_nodule_codes.md").read_text()
    return md.strip()


@pytest.fixture
def finding_info() -> FindingInfo:
    return FindingInfo(
        name="test finding",
        description="A test finding for testing.",
        synonyms=["test", "finding"],
    )
