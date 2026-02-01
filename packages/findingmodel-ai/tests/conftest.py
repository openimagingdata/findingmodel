from pathlib import Path

import pytest
from findingmodel import Index, logger
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

# =============================================================================
# Test Model Constants
# =============================================================================
# Use cheapest models for test fixtures to minimize cost during testing.
# These should be valid ModelSpec strings (see config.MODEL_SPEC_PATTERN).
# Update when model families change.

TEST_OPENAI_MODEL = "openai:gpt-5-nano"
TEST_ANTHROPIC_MODEL = "anthropic:claude-haiku-4-5"
TEST_GOOGLE_MODEL = "google:gemini-3-flash-preview"


@pytest.fixture(scope="session", autouse=True)
def configure_test_logging() -> None:
    """Configure logging for test session - runs once at start of session."""
    # Enable logging for the findingmodel module
    logger.enable("findingmodel")
    # Add a file handler to the findingmodel logger
    logger.add("test.log", level="INFO", rotation="10 MB")


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
def finding_info() -> FindingInfo:
    return FindingInfo(
        name="test finding",
        description="A test finding for testing.",
        synonyms=["test", "finding"],
    )


@pytest.fixture
def index_with_test_db() -> Index:
    """Provide a Index with prebuilt test database (no network calls)."""
    # Use the prebuilt test database from findingmodel package
    db_path = Path(__file__).parent.parent.parent / "findingmodel" / "tests" / "data" / "test_index.duckdb"
    if not db_path.exists():
        pytest.skip(
            "Pre-built test database not found. Run: uv run python packages/oidm-maintenance/scripts/build_test_fixtures.py"
        )

    index = Index(db_path)
    index._ensure_connection()
    return index
