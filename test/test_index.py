from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import pytest
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from findingmodel.finding_model import ChoiceAttributeIded, ChoiceValueIded, FindingModelFull, NumericAttributeIded
from findingmodel.index import Index

TEST_DB_NAME = "test_findingmodel_db"
TEST_COLLECTION_NAME = "test_findingmodel_collection"


@pytest.fixture(scope="session")
def mongo_client() -> Iterator[AsyncIOMotorClient[Any]]:
    client: AsyncIOMotorClient[Any] = AsyncIOMotorClient("mongodb://localhost:27017")
    yield client
    client.close()


@pytest.fixture(scope="session")
async def test_db(mongo_client: AsyncIOMotorClient[Any]) -> AsyncIterator[AsyncIOMotorDatabase[Any]]:
    db = mongo_client[TEST_DB_NAME]
    yield db
    await mongo_client.drop_database(TEST_DB_NAME)


@pytest.fixture
async def index() -> AsyncIterator[Index]:
    idx = Index()
    await idx.setup_indexes()
    yield idx
    # Clean up after test
    await idx.collection.drop()


@pytest.fixture
def sample_model() -> FindingModelFull:
    return FindingModelFull(
        oifm_id="OIFM_TEST_000001",
        name="Test Model",
        description="A test finding model.",
        attributes=[
            ChoiceAttributeIded(
                oifma_id="OIFMA_TEST_000001",
                name="Severity",
                description="Severity of the finding",
                values=[
                    ChoiceValueIded(value_code="OIFMA_TEST_000001.0", name="Mild"),
                    ChoiceValueIded(value_code="OIFMA_TEST_000001.1", name="Moderate"),
                ],
                required=True,
                max_selected=1,
            ),
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_000002",
                name="Size",
                description="Size of the finding",
                minimum=1,
                maximum=10,
                unit="cm",
                required=False,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_add_and_retrieve_model(index: Index, sample_model: FindingModelFull, tmp_path: Path) -> None:
    # Create a test file
    test_file = tmp_path / "test_model.fm.json"
    test_file.write_text(sample_model.model_dump_json())

    # Add the model to the index
    result = await index.add_or_update_entry_from_file(test_file, sample_model)
    assert result in ["added", "updated"]

    # Retrieve the entry
    retrieved_entry = await index.get(sample_model.oifm_id)
    assert retrieved_entry is not None
    assert retrieved_entry.oifm_id == sample_model.oifm_id
    assert retrieved_entry.name == sample_model.name


@pytest.mark.asyncio
async def test_validate_model_no_duplicates(index: Index, sample_model: FindingModelFull) -> None:
    errors = await index.validate_model(sample_model)
    assert errors == []


@pytest.mark.asyncio
async def test_validate_model_with_duplicates(index: Index, sample_model: FindingModelFull, tmp_path: Path) -> None:
    # First add the model
    test_file = tmp_path / "duplicate_model.fm.json"
    test_file.write_text(sample_model.model_dump_json())
    await index.add_or_update_entry_from_file(test_file, sample_model)

    # Try to validate the same model again (should have errors)
    errors = await index.validate_model(sample_model)
    assert len(errors) > 0
    assert any("Duplicate ID or name" in error for error in errors)


@pytest.mark.asyncio
async def test_contains_method(index: Index, sample_model: FindingModelFull, tmp_path: Path) -> None:
    # Model should not exist initially
    assert await index.contains(sample_model.oifm_id) is False
    assert await index.contains(sample_model.name) is False

    # Add the model
    test_file = tmp_path / "contains_test.fm.json"
    test_file.write_text(sample_model.model_dump_json())
    await index.add_or_update_entry_from_file(test_file, sample_model)

    # Now it should exist
    assert await index.contains(sample_model.oifm_id) is True
    assert await index.contains(sample_model.name) is True


@pytest.mark.asyncio
async def test_count_method(index: Index, sample_model: FindingModelFull, tmp_path: Path) -> None:
    # Initially empty
    initial_count = await index.count()
    assert initial_count == 0

    # Add a model
    test_file = tmp_path / "count_test.fm.json"
    test_file.write_text(sample_model.model_dump_json())
    await index.add_or_update_entry_from_file(test_file, sample_model)

    # Count should increase
    new_count = await index.count()
    assert new_count == initial_count + 1
