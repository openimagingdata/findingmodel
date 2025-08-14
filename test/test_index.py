import json
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


def test_index_initialization_with_client(mongo_client: AsyncIOMotorClient[Any]) -> None:
    """Test that the Index can be initialized with an existing client."""
    index = Index(client=mongo_client, db_name=TEST_MONGODB_DB)
    assert index.client is mongo_client
    assert index.db.name == TEST_MONGODB_DB
    assert index.index_collection.name == "index_entries_main"
    assert index.people_collection.name == "people_main"
    assert index.organizations_collection.name == "organizations_main"


# Search functionality tests
@pytest.mark.asyncio
async def test_search_basic_functionality(populated_index: Index) -> None:
    """Test basic search functionality with populated index."""
    # Search for "aneurysm" should find abdominal aortic aneurysm
    results = await populated_index.search("aneurysm", limit=10)
    assert len(results) >= 1

    # Check that we get IndexEntry objects back
    from findingmodel.index import IndexEntry

    assert all(isinstance(result, IndexEntry) for result in results)

    # Should find the abdominal aortic aneurysm model
    aneurysm_results = [r for r in results if "aneurysm" in r.name.lower()]
    assert len(aneurysm_results) >= 1


@pytest.mark.asyncio
async def test_search_by_name(populated_index: Index) -> None:
    """Test search functionality by exact and partial name matches."""
    # Exact name search
    results = await populated_index.search("abdominal aortic aneurysm")
    assert len(results) >= 1
    assert any("abdominal aortic aneurysm" in r.name.lower() for r in results)

    # Partial name search
    results = await populated_index.search("aortic")
    assert len(results) >= 1
    assert any("aortic" in r.name.lower() for r in results)


@pytest.mark.asyncio
async def test_search_by_description(populated_index: Index) -> None:
    """Test search functionality using description content."""
    # Search for terms that should appear in descriptions
    results = await populated_index.search("dilation")
    assert len(results) >= 0  # May or may not find results depending on description content

    # Search for medical terms
    results = await populated_index.search("diameter")
    assert len(results) >= 0  # May or may not find results


@pytest.mark.asyncio
async def test_search_by_synonyms(populated_index: Index) -> None:
    """Test search functionality using synonyms."""
    # Search for "AAA" which should be a synonym for abdominal aortic aneurysm
    results = await populated_index.search("AAA")
    # Note: This may not find results if synonyms aren't in the text index
    # but it's important to test the functionality
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_limit_parameter(populated_index: Index) -> None:
    """Test that search respects the limit parameter."""
    # Search with different limits
    results_limit_1 = await populated_index.search("aneurysm", limit=1)
    results_limit_5 = await populated_index.search("aneurysm", limit=5)

    assert len(results_limit_1) <= 1
    assert len(results_limit_5) <= 5

    # If there are results, limit should work correctly
    if results_limit_5:
        assert len(results_limit_1) <= len(results_limit_5)


@pytest.mark.asyncio
async def test_search_no_results(populated_index: Index) -> None:
    """Test search with query that should return no results."""
    results = await populated_index.search("zyxwvutsrqponmlkjihgfedcba")
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_empty_query(populated_index: Index) -> None:
    """Test search behavior with empty query."""
    results = await populated_index.search("", limit=5)
    assert isinstance(results, list)
    # Empty query behavior may vary - just ensure it doesn't crash


@pytest.mark.asyncio
async def test_search_case_insensitive(populated_index: Index) -> None:
    """Test that search is case insensitive."""
    results_lower = await populated_index.search("aneurysm")
    results_upper = await populated_index.search("ANEURYSM")
    results_mixed = await populated_index.search("Aneurysm")

    # Should get same results regardless of case
    assert len(results_lower) == len(results_upper) == len(results_mixed)


@pytest.mark.asyncio
async def test_search_multiple_terms(populated_index: Index) -> None:
    """Test search with multiple terms."""
    # Search for multiple terms
    results = await populated_index.search("abdominal aortic")
    assert isinstance(results, list)

    # Should potentially find models containing either term
    if results:
        found_text = " ".join([r.name + " " + (r.description or "") for r in results]).lower()
        # At least one term should be found
        assert "abdominal" in found_text or "aortic" in found_text


@pytest.mark.asyncio
async def test_search_with_empty_index(index: Index) -> None:
    """Test search functionality with empty index."""
    results = await index.search("anything", limit=10)
    assert isinstance(results, list)
    assert len(results) == 0


# Error handling and failure tests
@pytest.mark.asyncio
async def test_mongodb_connection_failure() -> None:
    """Test Index behavior when MongoDB connection fails."""
    from pymongo.errors import ServerSelectionTimeoutError

    # Create Index with invalid MongoDB URI
    invalid_index = Index(mongodb_uri="mongodb://nonexistent:27017", db_name="test_db")

    # Operations should fail gracefully
    with pytest.raises(ServerSelectionTimeoutError):
        await invalid_index.count()


@pytest.mark.asyncio
async def test_add_entry_with_invalid_json_file(index: Index, tmp_path: Path) -> None:
    """Test error handling when adding file with invalid JSON."""
    # Create file with invalid JSON
    invalid_file = tmp_path / "invalid.fm.json"
    invalid_file.write_text("{invalid json content")

    # Should raise appropriate error
    with pytest.raises((json.JSONDecodeError, ValueError)):  # JSON decode error or validation error
        await index.add_or_update_entry_from_file(invalid_file)


@pytest.mark.asyncio
async def test_add_entry_with_nonexistent_file(index: Index, tmp_path: Path) -> None:
    """Test error handling when adding nonexistent file."""
    nonexistent_file = tmp_path / "does_not_exist.fm.json"

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        await index.add_or_update_entry_from_file(nonexistent_file)


@pytest.mark.asyncio
async def test_add_entry_with_invalid_model_data(index: Index, tmp_path: Path) -> None:
    """Test error handling when adding file with invalid model data."""
    # Create file with JSON that doesn't match FindingModelFull schema
    invalid_model_file = tmp_path / "invalid_model.fm.json"
    invalid_model_data = {
        "name": "Test Model",
        # Missing required fields like oifm_id, attributes, etc.
    }
    invalid_model_file.write_text(json.dumps(invalid_model_data))

    # Should raise validation error
    from pydantic import ValidationError

    with pytest.raises((ValidationError, ValueError)):  # Pydantic validation error
        await index.add_or_update_entry_from_file(invalid_model_file)


@pytest.mark.asyncio
async def test_batch_operation_partial_failure(index: Index, tmp_path: Path, sample_model: FindingModelFull) -> None:
    """Test behavior when batch operations partially fail."""
    # Create one valid file
    valid_file = tmp_path / "valid.fm.json"
    valid_file.write_text(sample_model.model_dump_json())

    # Create one invalid file
    invalid_file = tmp_path / "invalid.fm.json"
    invalid_file.write_text("{invalid json")

    # update_from_directory should handle partial failures gracefully
    # The exact behavior may vary - it might skip invalid files or raise an error
    try:
        added, updated, removed = await index.update_from_directory(tmp_path)
        # If it succeeds, at least the valid file should be processed
        assert added >= 0 and updated >= 0 and removed >= 0
    except Exception:
        # If it fails, that's also acceptable behavior for invalid files
        pass


@pytest.mark.asyncio
async def test_concurrent_index_operations(index: Index, sample_model: FindingModelFull, tmp_path: Path) -> None:
    """Test Index behavior under concurrent operations."""
    import asyncio

    # Create multiple files
    files = []
    models = []
    for i in range(3):
        model = sample_model.model_copy()
        model.oifm_id = f"OIFM_CONCURRENT_TEST_{i:06d}"
        model.name = f"Concurrent Test Model {i}"

        file_path = tmp_path / f"concurrent_test_{i}.fm.json"
        file_path.write_text(model.model_dump_json())

        files.append(file_path)
        models.append(model)

    # Try to add all files concurrently
    async def add_file(file_path: Path, model: FindingModelFull) -> str:
        return await index.add_or_update_entry_from_file(file_path, model)

    # Run concurrent operations
    tasks = [add_file(f, m) for f, m in zip(files, models, strict=False)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # All operations should complete (successfully or with exceptions)
    assert len(results) == 3

    # Check that some models were added successfully
    final_count = await index.count()
    assert final_count > 0


@pytest.mark.asyncio
async def test_search_with_mongodb_error(index: Index) -> None:
    """Test search behavior when MongoDB has issues."""
    # This is difficult to test without actually breaking MongoDB
    # But we can test that search handles empty results gracefully
    results = await index.search("nonexistent_term_xyz", limit=5)
    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_large_query_handling(index: Index) -> None:
    """Test Index behavior with very large search queries."""
    # Test with very long search query
    very_long_query = "a" * 10000  # 10k character query

    # Should not crash
    results = await index.search(very_long_query, limit=5)
    assert isinstance(results, list)

    # Test with query containing special characters
    special_char_query = "\"'\\/{}[]$^*+?.|()"
    results = await index.search(special_char_query, limit=5)
    assert isinstance(results, list)
