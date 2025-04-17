import pytest

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


BASE_MODEL_MARKDOWN = """
# Test model

**Synonyms:** Test Synonym

**Tags:** tag1, tag2

A simple test finding model.

## Attributes

### Severity
How severe is the finding? *(Select one)*

- **Mild**: None
- **Severe**: None

### Size
Size of the finding.
Mininum: 1
Maximum: 10
Unit: cm""".strip()


def test_base_model_markdown(base_model: FindingModelBase) -> None:
    md = base_model.as_markdown()
    assert md.strip() == BASE_MODEL_MARKDOWN
    # Spacing: no double blank lines between sections
    assert "\n\n\n" not in md


FULL_MODEL_MARKDOWN = """
# Test model—`OIFM_TEST_123456`

**Synonyms:** Test Synonym

**Tags:** tag1, tag2

A simple test finding model.

## Attributes

### Severity—`OIFMA_TEST_123456`
How severe is the finding? *(Select one)*

- **Mild**: None
- **Severe**: None

### Size—`OIFMA_TEST_654321`
Size of the finding.
Mininum: 1
Maximum: 10
Unit: cm""".strip()


def test_full_model_markdown_with_ids(full_model: FindingModelFull) -> None:
    md = full_model.as_markdown()
    assert md.strip() == FULL_MODEL_MARKDOWN
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
