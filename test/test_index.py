import shutil
import socket
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import pytest
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError

from findingmodel.finding_model import ChoiceAttributeIded, ChoiceValueIded, FindingModelFull, NumericAttributeIded
from findingmodel.index import Index

TEST_MONGODB_URI = "mongodb://localhost:27017"
TEST_MONGODB_DB = "test_findingmodels_db"


# Synchronous check for MongoDB availability at module import time
def is_mongodb_running() -> bool:
    try:
        # First try a quick TCP connection to see if anything is listening
        with socket.create_connection(("localhost", 27017), timeout=1.0):
            pass

        # If that succeeds, try a quick MongoDB client connection
        client: AsyncIOMotorClient[Any] = AsyncIOMotorClient(TEST_MONGODB_URI, serverSelectionTimeoutMS=1000)
        # Close the client after we're done
        client.close()
        return True
    except (socket.error, ServerSelectionTimeoutError):
        return False


if not is_mongodb_running():
    pytest.skip("MongoDB not available - skipping MongoDB-dependent tests", allow_module_level=True)


@pytest.fixture(scope="session")
def mongo_client() -> Iterator[AsyncIOMotorClient[Any]]:
    client: AsyncIOMotorClient[Any] = AsyncIOMotorClient(TEST_MONGODB_URI)
    yield client
    client.close()


@pytest.fixture(scope="session")
async def test_db(mongo_client: AsyncIOMotorClient[Any]) -> AsyncIterator[AsyncIOMotorDatabase[Any]]:
    db = mongo_client[TEST_MONGODB_DB]
    yield db
    await mongo_client.drop_database(TEST_MONGODB_DB)


@pytest.fixture
async def index() -> AsyncIterator[Index]:
    index = Index(mongodb_uri=TEST_MONGODB_URI, db_name=TEST_MONGODB_DB)
    await index.setup_indexes()
    yield index
    # Clean up after test
    await index.index_collection.drop()


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


@pytest.fixture
def tmp_defs_path(tmp_path: Path) -> Path:
    """Create a temporary path for test files."""
    # Copy everyhing from the test/data/defs directory to the tmp_path
    data_dir = Path(__file__).parent / "data" / "defs"
    shutil.copytree(data_dir, tmp_path, dirs_exist_ok=True)
    return tmp_path


@pytest.fixture
async def populated_index(index: Index, tmp_defs_path: Path) -> AsyncIterator[Index]:
    """Populate the index with all *.fm.json files from test/data and test/data/defs."""
    # for fm_file in tmp_defs_path.glob("*.fm.json"):
    #     await index.add_or_update_entry_from_file(fm_file)
    await index.update_from_directory(tmp_defs_path)
    yield index
    # Clean up after test
    await index.index_collection.drop()


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


OIFM_IDS_IN_DEFS_DIR = [
    "OIFM_MSFT_573630",
    "OIFM_MSFT_356221",
    "OIFM_MSFT_156954",
    "OIFM_MSFT_367670",
    "OIFM_MSFT_932618",
    "OIFM_MSFT_134126",
]


@pytest.mark.asyncio
async def test_populated_index_count(populated_index: Index) -> None:
    count = await populated_index.count()
    # Should be at least as many as the number of *.fm.json files
    assert count >= len(OIFM_IDS_IN_DEFS_DIR)


@pytest.mark.asyncio
async def test_populated_index_retrieval(populated_index: Index) -> None:
    # Try to retrieve all models by oifm_id

    for oifm_id in OIFM_IDS_IN_DEFS_DIR:
        entry = await populated_index.get(oifm_id)
        assert entry is not None
        assert entry.oifm_id == oifm_id


@pytest.mark.asyncio
async def test_add_already_existing_model_unchanged(populated_index: Index) -> None:
    model_file = Path(__file__).parent / "data" / "defs" / "abdominal_aortic_aneurysm.fm.json"
    result = await populated_index.add_or_update_entry_from_file(model_file)
    assert result == "unchanged"
    count = await populated_index.count()
    assert count == len(OIFM_IDS_IN_DEFS_DIR), "Count should not change when adding an existing model"


@pytest.mark.asyncio
async def test_add_new_model(populated_index: Index, tmp_defs_path: Path) -> None:
    shutil.copy(Path(__file__).parent / "data" / "thyroid_nodule_codes.fm.json", tmp_defs_path)
    result = await populated_index.add_or_update_entry_from_file(tmp_defs_path / "thyroid_nodule_codes.fm.json")
    assert result == "added"
    count = await populated_index.count()
    assert count == len(OIFM_IDS_IN_DEFS_DIR) + 1, "Count should increase when adding a new model"
    entry = await populated_index.get("thyroid nodule")
    assert entry is not None


@pytest.mark.asyncio
async def test_add_updated_model_file(populated_index: Index, tmp_defs_path: Path) -> None:
    # Open "abdominal_aortic_aneurysm.fm.json" and change the description
    model_file = tmp_defs_path / "abdominal_aortic_aneurysm.fm.json"
    model = FindingModelFull.model_validate_json(model_file.read_text())
    entry = await populated_index.get(model.oifm_id)
    assert entry is not None
    current_hash = entry.file_hash_sha256
    model.description = "Updated description for abdominal aortic aneurysm."
    model_file.write_text(model.model_dump_json(indent=2, exclude_none=True))
    result = await populated_index.add_or_update_entry_from_file(model_file)
    assert result == "updated"
    count = await populated_index.count()
    assert count == len(OIFM_IDS_IN_DEFS_DIR), "Count should not change when updating an existing model"
    entry = await populated_index.get(model.oifm_id)
    assert entry is not None
    assert entry.file_hash_sha256 != current_hash, "File hash should change when the model is updated"


@pytest.mark.asyncio
async def test_remove_not_found_model(populated_index: Index, tmp_defs_path: Path) -> None:
    filepaths = sorted([f.name for f in tmp_defs_path.glob("*.fm.json")])
    assert len(filepaths) == len(OIFM_IDS_IN_DEFS_DIR), (
        "There should be as many files as OIFM IDs in the defs directory"
    )
    assert filepaths[0] == "abdominal_aortic_aneurysm.fm.json", "First file should be abdominal_aortic_aneurysm"
    aaa_entry = await populated_index.get("abdominal aortic aneurysm")
    assert aaa_entry is not None, "Should find the abdominal aortic aneurysm entry"
    assert aaa_entry.filename == "abdominal_aortic_aneurysm.fm.json", "Filename should match the entry"
    SKIP_ENTRIES = 2
    removed_entries = list(await populated_index.remove_unused_entries(filepaths[SKIP_ENTRIES:]))
    assert len(removed_entries) == SKIP_ENTRIES, "Should remove the first two entries"
    new_count = await populated_index.count()
    assert new_count == len(OIFM_IDS_IN_DEFS_DIR) - SKIP_ENTRIES, (
        "Count should decrease by the number of removed entries"
    )
    aaa_entry = await populated_index.get("abdominal aortic aneurysm")
    assert aaa_entry is None, "Should not find the abdominal aortic aneurysm entry after removal"


@pytest.mark.asyncio
async def test_duplicate_oifm_id_fails_validation(populated_index: Index, sample_model: FindingModelFull) -> None:
    sample_model.oifm_id = OIFM_IDS_IN_DEFS_DIR[0]  # Set to an existing OIFM ID
    errors = await populated_index.validate_model(sample_model)
    assert len(errors) > 0, "Validation should fail due to duplicate OIFM ID"
    assert errors[0].startswith("Duplicate ID"), "Error message should indicate duplicate ID or name"


@pytest.mark.asyncio
async def test_duplicate_name_fails_validation(populated_index: Index, sample_model: FindingModelFull) -> None:
    sample_model.name = "Abdominal Aortic Aneurysm"
    errors = await populated_index.validate_model(sample_model)
    assert len(errors) > 0, "Validation should fail due to duplicate name"
    assert errors[0].startswith("Duplicate name"), "Error message should indicate duplicate ID or name"


@pytest.mark.asyncio
async def test_duplicate_attribute_id_fails_validation(populated_index: Index, sample_model: FindingModelFull) -> None:
    EXISTING_ATTRIBUTE_ID = "OIFMA_MSFT_898601"  # Use an existing attribute ID
    sample_model.attributes[1].oifma_id = EXISTING_ATTRIBUTE_ID

    errors = await populated_index.validate_model(sample_model)
    assert len(errors) > 0, "Validation should fail due to duplicate attribute ID"
    assert errors[0].startswith("Attribute ID conflict"), "Error message should indicate duplicate attribute ID"


@pytest.mark.asyncio
async def test_update_from_directory(populated_index: Index, tmp_defs_path: Path) -> None:
    """Test the update_from_directory method with add, modify, and delete operations."""
    initial_count = await populated_index.count()
    assert initial_count == len(OIFM_IDS_IN_DEFS_DIR)
    # 1. First, identify existing entries to modify (before adding new file)
    files_to_modify = list(tmp_defs_path.glob("*.fm.json"))[:3]

    # 2. Add a new entry by copying a file to the directory
    new_file_path = tmp_defs_path / "thyroid_nodule_codes.fm.json"
    shutil.copy(Path(__file__).parent / "data" / "thyroid_nodule_codes.fm.json", new_file_path)

    # 3. Modify 3 existing entries
    modified_models = []
    for i, file_path in enumerate(files_to_modify):
        model = FindingModelFull.model_validate_json(file_path.read_text())
        model.description = f"Modified description {i + 1} for testing update_from_directory"
        modified_models.append(model)
        file_path.write_text(model.model_dump_json(indent=2, exclude_none=True))

    # 4. Delete 2 entries by removing their files
    files_to_delete = list(tmp_defs_path.glob("*.fm.json"))[-2:]
    deleted_oifm_ids = []
    for file_path in files_to_delete:
        model = FindingModelFull.model_validate_json(file_path.read_text())
        deleted_oifm_ids.append(model.oifm_id)
        file_path.unlink()  # Delete the file

    # Run update_from_directory
    added, updated, removed = await populated_index.update_from_directory(tmp_defs_path)

    # Verify the return values
    assert added == 1, f"Expected 1 added, got {added}"
    assert updated == 3, f"Expected 3 updated, got {updated}"
    assert removed == 2, f"Expected 2 removed, got {removed}"

    # Verify the actual changes in the index
    final_count = await populated_index.count()
    expected_count = initial_count + added - removed
    assert final_count == expected_count, f"Expected count {expected_count}, got {final_count}"

    # Verify the new entry was added
    thyroid_entry = await populated_index.get("thyroid nodule")
    assert thyroid_entry is not None, "New thyroid nodule entry should be found"
    assert thyroid_entry.filename == "thyroid_nodule_codes.fm.json"

    # Verify the modified entries have updated descriptions
    for modified_model in modified_models:
        entry = await populated_index.get(modified_model.oifm_id)
        assert entry is not None, f"Modified entry {modified_model.oifm_id} should still exist"
        # Note: The description is not stored in IndexEntry, but we can verify it was processed
        # by checking that the file hash changed (since we modified the file)

    # Verify the deleted entries are gone
    for deleted_oifm_id in deleted_oifm_ids:
        entry = await populated_index.get(deleted_oifm_id)
        assert entry is None, f"Deleted entry {deleted_oifm_id} should not be found"


@pytest.mark.asyncio
async def test_update_from_directory_empty_directory(populated_index: Index, tmp_path: Path) -> None:
    """Test update_from_directory with an empty directory removes all entries."""
    initial_count = await populated_index.count()
    assert initial_count > 0

    # Create an empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Run update_from_directory on empty directory
    added, updated, removed = await populated_index.update_from_directory(empty_dir)

    # Should remove all existing entries
    assert added == 0
    assert updated == 0
    assert removed == initial_count

    # Index should now be empty
    final_count = await populated_index.count()
    assert final_count == 0


@pytest.mark.asyncio
async def test_update_from_directory_nonexistent_directory(populated_index: Index, tmp_path: Path) -> None:
    """Test update_from_directory with a nonexistent directory raises ValueError."""
    nonexistent_dir = tmp_path / "nonexistent"

    with pytest.raises(ValueError, match="is not a valid directory"):
        await populated_index.update_from_directory(nonexistent_dir)
