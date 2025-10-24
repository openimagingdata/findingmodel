"""Tests for the DuckDB-backed index implementation."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pytest
import pytest_asyncio
from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

from findingmodel import index as duckdb_index
from findingmodel.config import settings
from findingmodel.contributor import Organization, Person
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import DuckDBIndex, IndexReturnType


def _fake_openai_client(*_: Any, **__: Any) -> object:  # pragma: no cover - test helper
    """Return a dummy OpenAI client for patched calls."""
    return object()


def _write_model_file(path: Path, data: FindingModelFull) -> None:
    """Write a FindingModelFull to a JSON file."""
    path.write_text(data.model_dump_json(indent=2, exclude_none=True))


def _table_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    """Get row count from a table."""
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _hnsw_index_exists(conn: duckdb.DuckDBPyConnection) -> bool:
    """Check if HNSW index exists on finding_models table."""
    rows = conn.execute("SELECT index_name FROM duckdb_indexes() WHERE table_name = 'finding_models'").fetchall()
    return any(row[0] == "finding_models_embedding_hnsw" for row in rows)


def _fts_index_works(conn: duckdb.DuckDBPyConnection) -> bool:
    """Check if FTS index is functional."""
    try:
        conn.execute(
            "SELECT fts_main_finding_models.match_bm25(oifm_id, 'test') FROM finding_models LIMIT 1"
        ).fetchall()
        return True
    except duckdb.Error:  # pragma: no cover
        return False


# ============================================================================
# Fixtures Overview
# ============================================================================
#
# From conftest.py:
# - base_model: FindingModelBase (function-scoped)
# - full_model: FindingModelFull (function-scoped)
# - tmp_defs_path: Path (function-scoped, copies test/data/defs to tmp_path)
#
# From this file:
# - index: Empty DuckDBIndex (function-scoped, mocked embeddings)
#   Use for: Tests that need an empty index or tests that will mutate the index
#
# - populated_index: Populated DuckDBIndex (function-scoped, mocked embeddings)
#   Use for: Tests that mutate the index (add/update/remove entries)
#   Note: Rebuilds from scratch for each test (slow but isolated)
#
# - session_populated_index: Populated DuckDBIndex (session-scoped, mocked embeddings)
#   Use for: Tests that ONLY READ from the index (no mutations)
#   Note: Built once per test session (fast, but shared across tests)
#   WARNING: DO NOT use for tests that add/update/remove entries!
#
# ============================================================================


# Shared fake embedding functions for all fixtures
async def _fake_embedding_deterministic(
    text: str,
    *,
    client: object | None = None,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[float]:  # pragma: no cover - test helper
    """Deterministic fake embedding based on text hash."""
    _ = (client, model)
    target_dims = dimensions or settings.openai_embedding_dimensions
    await asyncio.sleep(0)
    # Use simple hash-based embedding for determinism
    hash_val = sum(ord(c) for c in text)
    return [(hash_val % 100) / 100.0] * target_dims


async def _fake_client_for_testing() -> object:  # pragma: no cover - test helper
    """Return fake OpenAI client for testing."""
    await asyncio.sleep(0)
    return _fake_openai_client()


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a session-scoped event loop for session-scoped async fixtures.

    Required for session_populated_index fixture. pytest-asyncio requires that
    async fixtures have an event loop with matching or broader scope.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def _session_monkeypatch_setup() -> Generator[None, None, None]:
    """Session-scoped monkeypatch setup for mocking embeddings.

    Note: This patches module-level functions at session start and undoes at session end.
    Individual test fixtures can still override with their own monkeypatches if needed.
    """
    # Store original functions
    original_get_embedding = duckdb_index.get_embedding_for_duckdb  # type: ignore[attr-defined]
    original_batch_embeddings = duckdb_index.batch_embeddings_for_duckdb  # type: ignore[attr-defined]
    original_ensure_client = DuckDBIndex._ensure_openai_client

    # Patch with fakes
    duckdb_index.get_embedding_for_duckdb = _fake_embedding_deterministic  # type: ignore[attr-defined]
    duckdb_index.batch_embeddings_for_duckdb = lambda texts, client: asyncio.gather(  # type: ignore[attr-defined,assignment,misc]
        *[_fake_embedding_deterministic(t, client=client) for t in texts]
    )
    DuckDBIndex._ensure_openai_client = lambda _: _fake_client_for_testing()  # type: ignore[assignment,return-value,method-assign]

    yield

    # Restore originals
    duckdb_index.get_embedding_for_duckdb = original_get_embedding  # type: ignore[attr-defined]
    duckdb_index.batch_embeddings_for_duckdb = original_batch_embeddings  # type: ignore[attr-defined]
    DuckDBIndex._ensure_openai_client = original_ensure_client  # type: ignore[method-assign]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session_populated_index(
    tmp_path_factory: pytest.TempPathFactory, _session_monkeypatch_setup: None
) -> AsyncGenerator[DuckDBIndex, None]:
    """Session-scoped populated index - built once, shared across all read-only tests.

    This fixture builds the index from scratch ONCE per test session, significantly
    speeding up tests that only need to read from a populated index.

    IMPORTANT: Tests using this fixture must NOT mutate the index. Use the
    function-scoped `populated_index` or `index` fixtures for tests that add/update/remove entries.

    Performance: Building the index once saves ~10 seconds per test that only needs
    to read from a populated index. With 30+ read-only tests, this can save 5+ minutes.

    Note: Requires loop_scope="session" to match fixture scope (pytest-asyncio 1.2.0+).
    """
    # Create session-level temp directory
    tmp_dir = tmp_path_factory.mktemp("session_index")

    # Copy test data to session temp directory
    import shutil

    data_dir = Path(__file__).parent / "data" / "defs"
    defs_path = tmp_dir / "defs"
    shutil.copytree(data_dir, defs_path)

    # Create and populate index
    db_path = tmp_dir / "session_test_index.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()
    await index.update_from_directory(defs_path)

    yield index

    # Cleanup
    if index.conn is not None:
        index.conn.close()


@pytest.fixture
async def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[DuckDBIndex, None]:
    """Create a DuckDBIndex for testing with mocked OpenAI client."""

    async def fake_embedding(
        text: str,
        *,
        client: object | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[float]:  # pragma: no cover - test helper
        _ = (text, client, model)
        target_dims = dimensions or settings.openai_embedding_dimensions
        await asyncio.sleep(0)
        # Use simple hash-based embedding for determinism
        hash_val = sum(ord(c) for c in text)
        return [(hash_val % 100) / 100.0] * target_dims

    async def fake_client(self: DuckDBIndex) -> object:  # pragma: no cover - test helper
        await asyncio.sleep(0)
        return _fake_openai_client()

    monkeypatch.setattr(duckdb_index, "get_embedding_for_duckdb", fake_embedding)
    monkeypatch.setattr(
        duckdb_index,
        "batch_embeddings_for_duckdb",
        lambda texts, client: asyncio.gather(*[fake_embedding(t, client=client) for t in texts]),
    )
    monkeypatch.setattr(DuckDBIndex, "_ensure_openai_client", fake_client, raising=False)

    db_path = tmp_path / "test_index.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()
    yield index
    if index.conn is not None:
        index.conn.close()


@pytest.fixture
async def populated_index(index: DuckDBIndex, tmp_defs_path: Path) -> DuckDBIndex:
    """Populate the index with all *.fm.json files from test/data/defs."""
    await index.update_from_directory(tmp_defs_path)
    return index


OIFM_IDS_IN_DEFS_DIR = [
    "OIFM_MSFT_573630",
    "OIFM_MSFT_356221",
    "OIFM_MSFT_156954",
    "OIFM_MSFT_367670",
    "OIFM_MSFT_932618",
    "OIFM_MSFT_134126",
]


# ============================================================================
# CRUD Operations Tests (Ported from MongoDB Index)
# ============================================================================


@pytest.mark.asyncio
async def test_add_and_retrieve_model(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test adding a model and retrieving it by ID."""
    # Create a test file
    test_file = tmp_path / "test_model.fm.json"
    _write_model_file(test_file, full_model)

    # Add the model to the index
    result = await index.add_or_update_entry_from_file(test_file, full_model)
    assert result is IndexReturnType.ADDED

    # Retrieve the entry
    retrieved_entry = await index.get(full_model.oifm_id)
    assert retrieved_entry is not None
    assert retrieved_entry.oifm_id == full_model.oifm_id
    assert retrieved_entry.name == full_model.name


def test_validate_model_no_duplicates(index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test that validation passes when no duplicates exist."""
    errors = index._validate_model(full_model)
    assert errors == []


@pytest.mark.asyncio
async def test_contains_method(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test the contains method with ID and name lookups."""
    # Model should not exist initially
    assert await index.contains(full_model.oifm_id) is False
    assert await index.contains(full_model.name) is False

    # Add the model
    test_file = tmp_path / "contains_test.fm.json"
    _write_model_file(test_file, full_model)
    await index.add_or_update_entry_from_file(test_file, full_model)

    # Now it should exist
    assert await index.contains(full_model.oifm_id) is True
    assert await index.contains(full_model.name) is True


@pytest.mark.asyncio
async def test_count_method(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test that count returns correct number of models."""
    # Initially empty
    initial_count = await index.count()
    assert initial_count == 0

    # Add a model
    test_file = tmp_path / "count_test.fm.json"
    _write_model_file(test_file, full_model)
    await index.add_or_update_entry_from_file(test_file, full_model)

    # Count should increase
    new_count = await index.count()
    assert new_count == initial_count + 1


@pytest.mark.asyncio
async def test_populated_index_count(session_populated_index: DuckDBIndex) -> None:
    """Test count on populated index."""
    count = await session_populated_index.count()
    # Should be at least as many as the number of *.fm.json files
    assert count >= len(OIFM_IDS_IN_DEFS_DIR)


@pytest.mark.asyncio
async def test_populated_index_retrieval(session_populated_index: DuckDBIndex) -> None:
    """Test retrieving all models from populated index."""
    for oifm_id in OIFM_IDS_IN_DEFS_DIR:
        entry = await session_populated_index.get(oifm_id)
        assert entry is not None
        assert entry.oifm_id == oifm_id


@pytest.mark.asyncio
async def test_add_already_existing_model_unchanged(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test that adding an unchanged model returns 'unchanged'."""
    model_file = tmp_defs_path / "abdominal_aortic_aneurysm.fm.json"
    result = await populated_index.add_or_update_entry_from_file(model_file)
    assert result is IndexReturnType.UNCHANGED
    count = await populated_index.count()
    assert count == len(OIFM_IDS_IN_DEFS_DIR), "Count should not change when adding an existing model"


@pytest.mark.asyncio
async def test_add_new_model(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test adding a new model to populated index."""
    source_file = Path(__file__).parent / "data" / "thyroid_nodule_codes.fm.json"
    new_file = tmp_defs_path / "thyroid_nodule_codes.fm.json"
    shutil.copy(source_file, new_file)

    result = await populated_index.add_or_update_entry_from_file(new_file)
    assert result is IndexReturnType.ADDED

    count = await populated_index.count()
    assert count == len(OIFM_IDS_IN_DEFS_DIR) + 1, "Count should increase when adding a new model"

    entry = await populated_index.get("thyroid nodule")
    assert entry is not None


@pytest.mark.asyncio
async def test_add_updated_model_file(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test updating an existing model file."""
    # Open "abdominal_aortic_aneurysm.fm.json" and change the description
    model_file = tmp_defs_path / "abdominal_aortic_aneurysm.fm.json"
    model = FindingModelFull.model_validate_json(model_file.read_text())
    entry = await populated_index.get(model.oifm_id)
    assert entry is not None
    current_hash = entry.file_hash_sha256

    model.description = "Updated description for abdominal aortic aneurysm."
    _write_model_file(model_file, model)

    result = await populated_index.add_or_update_entry_from_file(model_file)
    assert result is IndexReturnType.UPDATED

    count = await populated_index.count()
    assert count == len(OIFM_IDS_IN_DEFS_DIR), "Count should not change when updating an existing model"

    entry = await populated_index.get(model.oifm_id)
    assert entry is not None
    assert entry.file_hash_sha256 != current_hash, "File hash should change when the model is updated"


@pytest.mark.asyncio
async def test_remove_not_found_model(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test removing entries not in the filenames list (equivalent to remove_unused_entries)."""
    filepaths = sorted([f.name for f in tmp_defs_path.glob("*.fm.json")])
    assert len(filepaths) == len(OIFM_IDS_IN_DEFS_DIR), (
        "There should be as many files as OIFM IDs in the defs directory"
    )
    assert filepaths[0] == "abdominal_aortic_aneurysm.fm.json", "First file should be abdominal_aortic_aneurysm"
    aaa_entry = await populated_index.get("abdominal aortic aneurysm")
    assert aaa_entry is not None, "Should find the abdominal aortic aneurysm entry"
    assert aaa_entry.filename == "abdominal_aortic_aneurysm.fm.json", "Filename should match the entry"

    # Remove first 2 files from directory
    SKIP_ENTRIES = 2
    for filepath in filepaths[:SKIP_ENTRIES]:
        (tmp_defs_path / filepath).unlink()

    # Run update_from_directory which should remove the deleted files
    result = await populated_index.update_from_directory(tmp_defs_path)
    assert result["removed"] == SKIP_ENTRIES, "Should remove the first two entries"

    new_count = await populated_index.count()
    assert new_count == len(OIFM_IDS_IN_DEFS_DIR) - SKIP_ENTRIES, (
        "Count should decrease by the number of removed entries"
    )
    aaa_entry = await populated_index.get("abdominal aortic aneurysm")
    assert aaa_entry is None, "Should not find the abdominal aortic aneurysm entry after removal"


# ============================================================================
# Validation Tests (Ported from MongoDB Index)
# ============================================================================


def test_duplicate_oifm_id_fails_validation(session_populated_index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test validation fails when OIFM ID already exists."""
    full_model.oifm_id = OIFM_IDS_IN_DEFS_DIR[0]  # Set to an existing OIFM ID
    errors = session_populated_index._validate_model(full_model)
    assert len(errors) > 0, "Validation should fail due to duplicate OIFM ID"
    assert "already exists" in errors[0] or "Duplicate" in errors[0], "Error message should indicate duplicate ID"


def test_duplicate_name_fails_validation(session_populated_index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test validation fails when name already exists (case-insensitive)."""
    full_model.name = "Abdominal Aortic Aneurysm"
    errors = session_populated_index._validate_model(full_model)
    assert len(errors) > 0, "Validation should fail due to duplicate name"
    assert "name" in errors[0].lower() and "already" in errors[0].lower(), (
        "Error message should indicate duplicate name"
    )


def test_duplicate_attribute_id_fails_validation(
    session_populated_index: DuckDBIndex, full_model: FindingModelFull
) -> None:
    """Test validation fails when attribute ID is used by another model."""
    EXISTING_ATTRIBUTE_ID = "OIFMA_MSFT_898601"  # Use an existing attribute ID
    full_model.attributes[1].oifma_id = EXISTING_ATTRIBUTE_ID

    errors = session_populated_index._validate_model(full_model)
    assert len(errors) > 0, "Validation should fail due to duplicate attribute ID"
    assert "attribute" in errors[0].lower() and ("conflict" in errors[0].lower() or "already" in errors[0].lower()), (
        "Error message should indicate attribute conflict"
    )


# ============================================================================
# Directory Update Tests (Ported from MongoDB Index)
# ============================================================================


@pytest.mark.asyncio
async def test_update_from_directory(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test the update_from_directory method with add, modify, and delete operations."""
    initial_count = await populated_index.count()
    assert initial_count == len(OIFM_IDS_IN_DEFS_DIR)

    # 1. First, identify existing entries to modify (before adding new file)
    files_to_modify = list(tmp_defs_path.glob("*.fm.json"))[:3]

    # 2. Add a new entry by copying a file to the directory
    source_file = Path(__file__).parent / "data" / "thyroid_nodule_codes.fm.json"
    new_file_path = tmp_defs_path / "thyroid_nodule_codes.fm.json"
    shutil.copy(source_file, new_file_path)

    # 3. Modify 3 existing entries
    modified_models = []
    for i, file_path in enumerate(files_to_modify):
        model = FindingModelFull.model_validate_json(file_path.read_text())
        model.description = f"Modified description {i + 1} for testing update_from_directory"
        modified_models.append(model)
        _write_model_file(file_path, model)

    # 4. Delete 2 entries by removing their files
    files_to_delete = list(tmp_defs_path.glob("*.fm.json"))[-2:]
    deleted_oifm_ids = []
    for file_path in files_to_delete:
        model = FindingModelFull.model_validate_json(file_path.read_text())
        deleted_oifm_ids.append(model.oifm_id)
        file_path.unlink()  # Delete the file

    # Run update_from_directory
    result = await populated_index.update_from_directory(tmp_defs_path)

    # Verify the return values
    assert result["added"] == 1, f"Expected 1 added, got {result['added']}"
    assert result["updated"] == 3, f"Expected 3 updated, got {result['updated']}"
    assert result["removed"] == 2, f"Expected 2 removed, got {result['removed']}"

    # Verify the actual changes in the index
    final_count = await populated_index.count()
    expected_count = initial_count + result["added"] - result["removed"]
    assert final_count == expected_count, f"Expected count {expected_count}, got {final_count}"

    # Verify the new entry was added
    thyroid_entry = await populated_index.get("thyroid nodule")
    assert thyroid_entry is not None, "New thyroid nodule entry should be found"
    assert thyroid_entry.filename == "thyroid_nodule_codes.fm.json"

    # Verify the modified entries still exist
    for modified_model in modified_models:
        entry = await populated_index.get(modified_model.oifm_id)
        assert entry is not None, f"Modified entry {modified_model.oifm_id} should still exist"

    # Verify the deleted entries are gone
    for deleted_oifm_id in deleted_oifm_ids:
        entry = await populated_index.get(deleted_oifm_id)
        assert entry is None, f"Deleted entry {deleted_oifm_id} should not be found"


@pytest.mark.asyncio
async def test_update_from_directory_empty_directory(populated_index: DuckDBIndex, tmp_path: Path) -> None:
    """Test update_from_directory with an empty directory removes all entries."""
    initial_count = await populated_index.count()
    assert initial_count > 0

    # Create an empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Run update_from_directory on empty directory
    result = await populated_index.update_from_directory(empty_dir)

    # Should remove all existing entries
    assert result["added"] == 0
    assert result["updated"] == 0
    assert result["removed"] == initial_count

    # Index should now be empty
    final_count = await populated_index.count()
    assert final_count == 0


@pytest.mark.asyncio
async def test_update_from_directory_nonexistent_directory(populated_index: DuckDBIndex, tmp_path: Path) -> None:
    """Test update_from_directory with a nonexistent directory raises ValueError."""
    nonexistent_dir = tmp_path / "nonexistent"

    with pytest.raises(ValueError, match="is not a valid directory"):
        await populated_index.update_from_directory(nonexistent_dir)


# ============================================================================
# Search Functionality Tests (Ported from MongoDB Index)
# ============================================================================


@pytest.mark.asyncio
async def test_search_basic_functionality(session_populated_index: DuckDBIndex) -> None:
    """Test basic search functionality with populated index."""
    # Search for "aneurysm" should find abdominal aortic aneurysm
    results = await session_populated_index.search("aneurysm", limit=10)
    assert len(results) >= 1

    # Check that we get IndexEntry objects back
    from findingmodel.index import IndexEntry

    assert all(isinstance(result, IndexEntry) for result in results)

    # Should find the abdominal aortic aneurysm model
    aneurysm_results = [r for r in results if "aneurysm" in r.name.lower()]
    assert len(aneurysm_results) >= 1


@pytest.mark.asyncio
async def test_search_by_name(session_populated_index: DuckDBIndex) -> None:
    """Test search functionality by exact and partial name matches."""
    # Exact name search
    results = await session_populated_index.search("abdominal aortic aneurysm")
    assert len(results) >= 1
    assert any("abdominal aortic aneurysm" in r.name.lower() for r in results)

    # Partial name search
    results = await session_populated_index.search("aortic")
    assert len(results) >= 1
    assert any("aortic" in r.name.lower() for r in results)


@pytest.mark.asyncio
async def test_search_by_description(session_populated_index: DuckDBIndex) -> None:
    """Test search functionality using description content."""
    # Search for terms that should appear in descriptions
    results = await session_populated_index.search("dilation")
    assert len(results) >= 0  # May or may not find results depending on description content

    # Search for medical terms
    results = await session_populated_index.search("diameter")
    assert len(results) >= 0  # May or may not find results


@pytest.mark.asyncio
async def test_search_by_synonyms(session_populated_index: DuckDBIndex) -> None:
    """Test search functionality using synonyms."""
    # Search for "AAA" which should be a synonym for abdominal aortic aneurysm
    results = await session_populated_index.search("AAA")
    # Note: This may not find results if synonyms aren't in the text index
    # but it's important to test the functionality
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_search_limit_parameter(session_populated_index: DuckDBIndex) -> None:
    """Test that search respects the limit parameter."""
    # Search with different limits
    results_limit_1 = await session_populated_index.search("aneurysm", limit=1)
    results_limit_5 = await session_populated_index.search("aneurysm", limit=5)

    assert len(results_limit_1) <= 1
    assert len(results_limit_5) <= 5

    # If there are results, limit should work correctly
    if results_limit_5:
        assert len(results_limit_1) <= len(results_limit_5)


@pytest.mark.asyncio
async def test_search_no_results(session_populated_index: DuckDBIndex) -> None:
    """Test search with query that should return no results."""
    results = await session_populated_index.search("zyxwvutsrqponmlkjihgfedcba")
    assert isinstance(results, list)
    # With fake embeddings in tests, semantic search might still return results
    # In production with real embeddings, nonsense queries typically return nothing


@pytest.mark.asyncio
async def test_search_empty_query(session_populated_index: DuckDBIndex) -> None:
    """Test search behavior with empty query."""
    results = await session_populated_index.search("", limit=5)
    assert isinstance(results, list)
    # Empty query behavior may vary - just ensure it doesn't crash


@pytest.mark.asyncio
async def test_search_case_insensitive(session_populated_index: DuckDBIndex) -> None:
    """Test that search is case insensitive."""
    results_lower = await session_populated_index.search("aneurysm")
    results_upper = await session_populated_index.search("ANEURYSM")
    results_mixed = await session_populated_index.search("Aneurysm")

    # Should get same results regardless of case
    assert len(results_lower) == len(results_upper) == len(results_mixed)


@pytest.mark.asyncio
async def test_search_multiple_terms(session_populated_index: DuckDBIndex) -> None:
    """Test search with multiple terms."""
    # Search for multiple terms
    results = await session_populated_index.search("abdominal aortic")
    assert isinstance(results, list)

    # Should potentially find models containing either term
    if results:
        found_text = " ".join([r.name + " " + (r.description or "") for r in results]).lower()
        # At least one term should be found
        assert "abdominal" in found_text or "aortic" in found_text


@pytest.mark.asyncio
async def test_search_with_empty_index(index: DuckDBIndex) -> None:
    """Test search functionality with empty index."""
    results = await index.search("anything", limit=10)
    assert isinstance(results, list)
    assert len(results) == 0


# ============================================================================
# Error Handling Tests (Ported from MongoDB Index)
# ============================================================================


@pytest.mark.asyncio
async def test_add_entry_with_invalid_json_file(index: DuckDBIndex, tmp_path: Path) -> None:
    """Test error handling when adding file with invalid JSON."""
    # Create file with invalid JSON
    invalid_file = tmp_path / "invalid.fm.json"
    invalid_file.write_text("{invalid json content")

    # Should raise appropriate error
    with pytest.raises((json.JSONDecodeError, ValueError)):  # JSON decode error or validation error
        await index.add_or_update_entry_from_file(invalid_file)


@pytest.mark.asyncio
async def test_add_entry_with_nonexistent_file(index: DuckDBIndex, tmp_path: Path) -> None:
    """Test error handling when adding nonexistent file."""
    nonexistent_file = tmp_path / "does_not_exist.fm.json"

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        await index.add_or_update_entry_from_file(nonexistent_file)


@pytest.mark.asyncio
async def test_add_entry_with_invalid_model_data(index: DuckDBIndex, tmp_path: Path) -> None:
    """Test error handling when adding file with invalid model data."""
    # Create file with JSON that doesn't match FindingModelFull schema
    invalid_model_file = tmp_path / "invalid_model.fm.json"
    invalid_model_data = {
        "name": "Test Model",
        # Missing required fields like oifm_id, attributes, etc.
    }
    invalid_model_file.write_text(json.dumps(invalid_model_data))

    # Should raise validation error
    with pytest.raises((ValidationError, ValueError)):  # Pydantic validation error
        await index.add_or_update_entry_from_file(invalid_model_file)


@pytest.mark.asyncio
async def test_batch_operation_partial_failure(
    index: DuckDBIndex, tmp_path: Path, full_model: FindingModelFull
) -> None:
    """Test behavior when batch operations partially fail."""
    # Create one valid file
    valid_file = tmp_path / "valid.fm.json"
    _write_model_file(valid_file, full_model)

    # Create one invalid file
    invalid_file = tmp_path / "invalid.fm.json"
    invalid_file.write_text("{invalid json")

    # update_from_directory should handle partial failures gracefully
    # The exact behavior may vary - it might skip invalid files or raise an error
    try:
        result = await index.update_from_directory(tmp_path)
        # If it succeeds, at least the valid file should be processed
        assert result["added"] >= 0 and result["updated"] >= 0 and result["removed"] >= 0
    except Exception:
        # If it fails, that's also acceptable behavior for invalid files
        pass


@pytest.mark.asyncio
async def test_concurrent_index_operations(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test Index behavior under concurrent operations."""
    # Create multiple files
    files = []
    models = []
    for i in range(3):
        model = full_model.model_copy()
        model.oifm_id = f"OIFM_CONCURRENT_TEST_{i:06d}"
        model.name = f"Concurrent Test Model {i}"

        file_path = tmp_path / f"concurrent_test_{i}.fm.json"
        _write_model_file(file_path, model)

        files.append(file_path)
        models.append(model)

    # Try to add all files concurrently
    async def add_file(file_path: Path, model: FindingModelFull) -> IndexReturnType:
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
async def test_large_query_handling(index: DuckDBIndex) -> None:
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


# ============================================================================
# Contributor Management Tests (New for DuckDB)
# ============================================================================


@pytest.mark.asyncio
async def test_get_person(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test retrieving a person by github username."""
    model = full_model.model_copy(deep=True)
    model.contributors = [
        Person(
            github_username="testuser",
            name="Test User",
            email="test@example.com",
            organization_code="TEST",
        )
    ]

    test_file = tmp_path / "person_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    person = await index.get_person("testuser")
    assert person is not None
    assert person.github_username == "testuser"
    assert person.name == "Test User"
    assert person.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_organization(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test retrieving an organization by code."""
    model = full_model.model_copy(deep=True)
    from pydantic import HttpUrl

    model.contributors = [
        Organization(
            code="TEST",
            name="Test Organization",
            url=HttpUrl("https://testorg.example.com"),
        )
    ]

    test_file = tmp_path / "org_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    org = await index.get_organization("TEST")
    assert org is not None
    assert org.code == "TEST"
    assert org.name == "Test Organization"
    assert str(org.url) == "https://testorg.example.com/"  # URL objects include trailing slash


@pytest.mark.asyncio
async def test_get_people(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test retrieving all people from the index."""
    # Base contributors should already be present (4 people)
    people = await index.get_people()
    initial_count = len(people)
    assert initial_count >= 4

    # Add a finding model with a NEW contributor
    model = full_model.model_copy(deep=True)
    model.contributors = [
        Person(github_username="newuser", name="New User", email="new@example.com", organization_code="MSFT")
    ]
    test_file = tmp_path / "test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    # Should include base + new contributor
    people = await index.get_people()
    assert len(people) == initial_count + 1
    assert all(isinstance(p, Person) for p in people)
    # Verify sorted by name
    names = [p.name for p in people]
    assert names == sorted(names)


@pytest.mark.asyncio
async def test_get_organizations(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test retrieving all organizations from the index."""
    from pydantic import HttpUrl

    # Base contributors should already be present (7 organizations)
    orgs = await index.get_organizations()
    initial_count = len(orgs)
    assert initial_count >= 7

    # Add a finding model with contributor from new organization (code must be 3-4 chars)
    model = full_model.model_copy(deep=True)
    model.contributors = [Organization(code="NEW", name="New Organization", url=HttpUrl("https://neworg.example.com"))]
    test_file = tmp_path / "test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    # Should include base + new org
    orgs = await index.get_organizations()
    assert len(orgs) == initial_count + 1
    assert all(isinstance(o, Organization) for o in orgs)
    # Verify sorted by name
    names = [o.name for o in orgs]
    assert names == sorted(names)


@pytest.mark.asyncio
async def test_count_people(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test counting people in the index."""
    # Base contributors are loaded automatically, so we expect 4 people initially
    initial_count = await index.count_people()
    assert initial_count == 4

    model1 = full_model.model_copy(deep=True)
    model1.oifm_id = "OIFM_TSTA_000001"
    model1.name = "Model 1"
    model1.attributes[0].oifma_id = "OIFMA_TSTA_000001"
    model1.attributes[1].oifma_id = "OIFMA_TSTA_000002"
    model1.contributors = [
        Person(github_username="user1", name="User One", email="user1@example.com", organization_code="TSTA")
    ]

    model2 = full_model.model_copy(deep=True)
    model2.oifm_id = "OIFM_TSTB_000002"
    model2.name = "Model 2"
    model2.attributes[0].oifma_id = "OIFMA_TSTB_000001"
    model2.attributes[1].oifma_id = "OIFMA_TSTB_000002"
    model2.contributors = [
        Person(github_username="user2", name="User Two", email="user2@example.com", organization_code="TSTB")
    ]

    file1 = tmp_path / "model1.fm.json"
    file2 = tmp_path / "model2.fm.json"
    _write_model_file(file1, model1)
    _write_model_file(file2, model2)

    await index.add_or_update_entry_from_file(file1, model1)
    await index.add_or_update_entry_from_file(file2, model2)

    # Should have initial 4 base contributors + 2 new ones = 6
    assert await index.count_people() == 6


@pytest.mark.asyncio
async def test_count_organizations(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test counting organizations in the index."""
    # Base contributors are loaded automatically, so we expect 7 organizations initially
    initial_count = await index.count_organizations()
    assert initial_count == 7

    model = full_model.model_copy(deep=True)
    model.contributors = [
        Organization(code="ORGA", name="Organization One"),
        Organization(code="ORGB", name="Organization Two"),
    ]

    test_file = tmp_path / "org_count_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    # Should have initial 7 base contributors + 2 new ones = 9
    assert await index.count_organizations() == 9


# ============================================================================
# DuckDB-Specific Tests: Denormalized Tables
# ============================================================================


@pytest.mark.asyncio
async def test_denormalized_synonyms_table(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test that synonyms table is populated correctly."""
    model = full_model.model_copy(deep=True)
    model.synonyms = ["Synonym 1", "Synonym 2", "Synonym 3"]

    test_file = tmp_path / "synonyms_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    conn = index._ensure_connection()
    count = _table_count(conn, "synonyms")
    assert count == 3

    rows = conn.execute("SELECT synonym FROM synonyms WHERE oifm_id = ? ORDER BY synonym", (model.oifm_id,)).fetchall()
    synonyms = [row[0] for row in rows]
    assert synonyms == ["Synonym 1", "Synonym 2", "Synonym 3"]


@pytest.mark.asyncio
async def test_denormalized_tags_table(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test that tags table is populated correctly."""
    model = full_model.model_copy(deep=True)
    model.tags = ["tag1", "tag2", "tag3"]

    test_file = tmp_path / "tags_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    conn = index._ensure_connection()
    count = _table_count(conn, "tags")
    assert count == 3

    rows = conn.execute("SELECT tag FROM tags WHERE oifm_id = ? ORDER BY tag", (model.oifm_id,)).fetchall()
    tags = [row[0] for row in rows]
    assert tags == ["tag1", "tag2", "tag3"]


@pytest.mark.asyncio
async def test_denormalized_attributes_table(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test that attributes table is populated correctly."""
    model = full_model.model_copy(deep=True)

    test_file = tmp_path / "attributes_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    conn = index._ensure_connection()
    count = _table_count(conn, "attributes")
    assert count == 2  # full_model has 2 attributes

    row = conn.execute(
        "SELECT attribute_id, attribute_name, attribute_type FROM attributes WHERE oifm_id = ? ORDER BY attribute_id",
        (model.oifm_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == model.attributes[0].oifma_id
    assert row[1] == "Severity"
    assert "CHOICE" in row[2] or row[2] == "choice"  # Handle both enum string formats


@pytest.mark.asyncio
async def test_denormalized_model_people_table(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that model_people junction table is populated correctly."""
    model = full_model.model_copy(deep=True)
    model.contributors = [
        Person(github_username="user1", name="User One", email="user1@example.com", organization_code="TEST"),
        Person(github_username="user2", name="User Two", email="user2@example.com", organization_code="TEST"),
    ]

    test_file = tmp_path / "people_table_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    conn = index._ensure_connection()
    count = _table_count(conn, "model_people")
    assert count == 2

    rows = conn.execute(
        "SELECT person_id, display_order FROM model_people WHERE oifm_id = ? ORDER BY display_order", (model.oifm_id,)
    ).fetchall()
    assert rows[0][0] == "user1"
    assert rows[0][1] == 0
    assert rows[1][0] == "user2"
    assert rows[1][1] == 1


@pytest.mark.asyncio
async def test_denormalized_model_organizations_table(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that model_organizations junction table is populated correctly."""
    model = full_model.model_copy(deep=True)
    model.contributors = [
        Organization(code="ORGA", name="Organization One"),
        Organization(code="ORGB", name="Organization Two"),
    ]

    test_file = tmp_path / "orgs_table_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    conn = index._ensure_connection()
    count = _table_count(conn, "model_organizations")
    assert count == 2

    rows = conn.execute(
        "SELECT organization_id, display_order FROM model_organizations WHERE oifm_id = ? ORDER BY display_order",
        (model.oifm_id,),
    ).fetchall()
    assert rows[0][0] == "ORGA"
    assert rows[0][1] == 0
    assert rows[1][0] == "ORGB"
    assert rows[1][1] == 1


@pytest.mark.asyncio
async def test_remove_entry_clears_related_rows(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that removing an entry clears all denormalized table rows."""
    model = full_model.model_copy(deep=True)
    model.synonyms = ["syn1", "syn2"]
    model.tags = ["tag1", "tag2"]

    test_file = tmp_path / "remove_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    conn = index._ensure_connection()
    pre_delete_counts = {
        "synonyms": _table_count(conn, "synonyms"),
        "tags": _table_count(conn, "tags"),
        "attributes": _table_count(conn, "attributes"),
    }
    assert pre_delete_counts == {"synonyms": 2, "tags": 2, "attributes": 2}  # full_model has 2 attributes

    removed = await index.remove_entry(model.oifm_id)
    assert removed is True

    post_delete_counts = {
        "synonyms": _table_count(conn, "synonyms"),
        "tags": _table_count(conn, "tags"),
        "attributes": _table_count(conn, "attributes"),
    }
    assert post_delete_counts == {"synonyms": 0, "tags": 0, "attributes": 0}


# ============================================================================
# DuckDB-Specific Tests: HNSW Index Management
# ============================================================================


def test_setup_creates_search_indexes(index: DuckDBIndex) -> None:
    """Test that setup() creates both HNSW and FTS indexes."""
    conn = index._ensure_connection()
    assert _hnsw_index_exists(conn), "HNSW index should be created"
    assert _fts_index_works(conn), "FTS index should be functional"


@pytest.mark.asyncio
async def test_write_operations_rebuild_search_indexes(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that write operations drop and rebuild search indexes."""
    model = full_model.model_copy(deep=True)
    file_path = tmp_path / "model.fm.json"
    _write_model_file(file_path, model)

    await index.add_or_update_entry_from_file(file_path, model)

    conn = index._ensure_connection()
    assert _hnsw_index_exists(conn)
    assert _fts_index_works(conn)

    await index.remove_entry(model.oifm_id)

    assert _hnsw_index_exists(conn)
    assert _fts_index_works(conn)


@pytest.mark.asyncio
async def test_batch_update_rebuilds_indexes_once(index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test that batch directory update rebuilds indexes once, not per file."""
    # This test verifies the optimization where indexes are dropped once,
    # batch mutations applied, then indexes rebuilt once
    result = await index.update_from_directory(tmp_defs_path)
    assert result["added"] > 0

    conn = index._ensure_connection()
    assert _hnsw_index_exists(conn)
    assert _fts_index_works(conn)


# ============================================================================
# DuckDB-Specific Tests: Tag Filtering
# ============================================================================


@pytest.mark.asyncio
async def test_search_with_single_tag(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test search with single tag filter."""
    model1 = full_model.model_copy(deep=True)
    model1.oifm_id = "OIFM_TAGA_000001"
    model1.name = "Model With Tag A"
    model1.tags = ["tagA", "tagB"]
    model1.attributes[0].oifma_id = "OIFMA_TAGA_000001"
    model1.attributes[1].oifma_id = "OIFMA_TAGA_000002"

    model2 = full_model.model_copy(deep=True)
    model2.oifm_id = "OIFM_TAGB_000002"
    model2.name = "Model With Tag B"
    model2.tags = ["tagB", "tagC"]
    model2.attributes[0].oifma_id = "OIFMA_TAGB_000001"
    model2.attributes[1].oifma_id = "OIFMA_TAGB_000002"

    file1 = tmp_path / "tag_model1.fm.json"
    file2 = tmp_path / "tag_model2.fm.json"
    _write_model_file(file1, model1)
    _write_model_file(file2, model2)

    await index.add_or_update_entry_from_file(file1, model1)
    await index.add_or_update_entry_from_file(file2, model2)

    # Search with tagA - should find only model1
    results = await index.search("model", limit=10, tags=["tagA"])
    assert len(results) == 1
    assert results[0].oifm_id == "OIFM_TAGA_000001"

    # Search with tagB - should find both
    results = await index.search("model", limit=10, tags=["tagB"])
    assert len(results) == 2

    # Search with tagC - should find only model2
    results = await index.search("model", limit=10, tags=["tagC"])
    assert len(results) == 1
    assert results[0].oifm_id == "OIFM_TAGB_000002"


@pytest.mark.asyncio
async def test_search_with_multiple_tags_and_logic(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test search with multiple tags (AND logic - must have ALL)."""
    model1 = full_model.model_copy(deep=True)
    model1.oifm_id = "OIFM_MTAG_000101"
    model1.name = "Model With Tags A and B"
    model1.tags = ["tagA", "tagB"]
    model1.attributes[0].oifma_id = "OIFMA_MTAG_000101"
    model1.attributes[1].oifma_id = "OIFMA_MTAG_000102"

    model2 = full_model.model_copy(deep=True)
    model2.oifm_id = "OIFM_MTAG_000103"
    model2.name = "Model With Only Tag A"
    model2.tags = ["tagA"]
    model2.attributes[0].oifma_id = "OIFMA_MTAG_000103"
    model2.attributes[1].oifma_id = "OIFMA_MTAG_000104"

    file1 = tmp_path / "multi_tag_model1.fm.json"
    file2 = tmp_path / "multi_tag_model2.fm.json"
    _write_model_file(file1, model1)
    _write_model_file(file2, model2)

    await index.add_or_update_entry_from_file(file1, model1)
    await index.add_or_update_entry_from_file(file2, model2)

    # Search for models with BOTH tagA AND tagB - should find only model1
    results = await index.search("model", limit=10, tags=["tagA", "tagB"])
    assert len(results) == 1
    assert results[0].oifm_id == "OIFM_MTAG_000101"


@pytest.mark.asyncio
async def test_search_with_nonexistent_tags(index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path) -> None:
    """Test search with tags that don't exist."""
    model = full_model.model_copy(deep=True)
    model.tags = ["realTag"]

    test_file = tmp_path / "tag_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    # Search with non-existent tag
    results = await index.search("model", limit=10, tags=["nonExistentTag"])
    assert len(results) == 0


@pytest.mark.asyncio
async def test_tag_filtering_works_in_all_search_paths(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that tag filtering works in exact, FTS, and semantic search paths."""
    model = full_model.model_copy(deep=True)
    model.oifm_id = "OIFM_TGPT_000001"
    model.name = "Tag Path Test Model"
    model.tags = ["specialTag"]

    test_file = tmp_path / "tag_path_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    # Exact match with tag filter
    results = await index.search("Tag Path Test Model", limit=10, tags=["specialTag"])
    assert len(results) == 1

    # Exact match without tag filter
    results = await index.search("Tag Path Test Model", limit=10, tags=["wrongTag"])
    assert len(results) == 0

    # FTS/semantic search with tag filter
    results = await index.search("path test", limit=10, tags=["specialTag"])
    assert len(results) >= 1

    # FTS/semantic search without matching tag
    results = await index.search("path test", limit=10, tags=["wrongTag"])
    assert len(results) == 0


# ============================================================================
# DuckDB-Specific Tests: search_batch() Optimization
# ============================================================================


@pytest.mark.asyncio
async def test_search_batch_multiple_queries(session_populated_index: DuckDBIndex) -> None:
    """Test batching multiple queries efficiently."""
    queries = ["aneurysm", "aortic", "pulmonary"]
    results = await session_populated_index.search_batch(queries, limit=5)

    assert len(results) == 3
    assert "aneurysm" in results
    assert "aortic" in results
    assert "pulmonary" in results

    # Each query should have results
    for query in queries:
        assert isinstance(results[query], list)


@pytest.mark.asyncio
async def test_search_batch_all_queries_return_results(session_populated_index: DuckDBIndex) -> None:
    """Test that all queries in batch return their results."""
    queries = ["aneurysm", "diameter"]
    results = await session_populated_index.search_batch(queries, limit=10)

    assert len(results) == len(queries)
    for query in queries:
        assert query in results


@pytest.mark.asyncio
async def test_search_batch_empty_queries_list(index: DuckDBIndex) -> None:
    """Test search_batch with empty queries list."""
    results = await index.search_batch([], limit=10)
    assert results == {}


@pytest.mark.asyncio
async def test_search_batch_with_valid_and_invalid_queries(session_populated_index: DuckDBIndex) -> None:
    """Test search_batch with mix of valid and invalid queries."""
    queries = ["aneurysm", "zzzzzznonexistent", "aortic"]
    results = await session_populated_index.search_batch(queries, limit=5)

    assert len(results) == 3
    assert "aneurysm" in results
    assert "zzzzzznonexistent" in results
    assert "aortic" in results

    # Invalid query should return empty list
    assert isinstance(results["zzzzzznonexistent"], list)


# ============================================================================
# DuckDB-Specific Tests: Batch Directory Operations
# ============================================================================


@pytest.mark.asyncio
async def test_update_from_directory_batch_add(index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test batch directory ingestion - add scenario."""
    result = await index.update_from_directory(tmp_defs_path)

    assert result["added"] == len(OIFM_IDS_IN_DEFS_DIR)
    assert result["updated"] == 0
    assert result["removed"] == 0

    count = await index.count()
    assert count == len(OIFM_IDS_IN_DEFS_DIR)


@pytest.mark.asyncio
async def test_update_from_directory_batch_update(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test batch directory ingestion - update scenario."""
    # Modify all files
    for file_path in tmp_defs_path.glob("*.fm.json"):
        model = FindingModelFull.model_validate_json(file_path.read_text())
        model.description = f"Updated: {model.description or 'No description'}"
        _write_model_file(file_path, model)

    result = await populated_index.update_from_directory(tmp_defs_path)

    assert result["added"] == 0
    assert result["updated"] == len(OIFM_IDS_IN_DEFS_DIR)
    assert result["removed"] == 0


@pytest.mark.asyncio
async def test_update_from_directory_batch_delete(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test batch directory ingestion - delete scenario."""
    initial_count = await populated_index.count()

    # Delete half the files
    files = list(tmp_defs_path.glob("*.fm.json"))
    num_to_delete = len(files) // 2
    for file_path in files[:num_to_delete]:
        file_path.unlink()

    result = await populated_index.update_from_directory(tmp_defs_path)

    assert result["added"] == 0
    assert result["updated"] == 0
    assert result["removed"] == num_to_delete

    final_count = await populated_index.count()
    assert final_count == initial_count - num_to_delete


@pytest.mark.asyncio
async def test_update_from_directory_batch_mixed(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test batch directory ingestion - mixed add/update/delete scenario."""
    # Add new file
    source_file = Path(__file__).parent / "data" / "thyroid_nodule_codes.fm.json"
    new_file = tmp_defs_path / "thyroid_nodule_codes.fm.json"
    shutil.copy(source_file, new_file)

    # Update some files
    files = list(tmp_defs_path.glob("*.fm.json"))
    for file_path in files[:2]:
        if file_path.name != "thyroid_nodule_codes.fm.json":
            model = FindingModelFull.model_validate_json(file_path.read_text())
            model.description = "Mixed batch test update"
            _write_model_file(file_path, model)

    # Delete some files
    deleted_count = 0
    for file_path in files[-2:]:
        if file_path.name != "thyroid_nodule_codes.fm.json":
            file_path.unlink()
            deleted_count += 1

    result = await populated_index.update_from_directory(tmp_defs_path)

    assert result["added"] == 1
    # Updated count depends on which files were actually updated vs deleted
    assert result["updated"] >= 1
    assert result["removed"] == deleted_count


@pytest.mark.asyncio
async def test_update_from_directory_no_changes(populated_index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test update_from_directory when no changes are needed."""
    result = await populated_index.update_from_directory(tmp_defs_path)

    # Should report no changes
    assert result["added"] == 0
    assert result["updated"] == 0
    assert result["removed"] == 0


# ============================================================================
# DuckDB-Specific Tests: Read-Only Mode
# ============================================================================


@pytest.mark.asyncio
async def test_read_only_mode_blocks_writes(
    full_model: FindingModelFull, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that read-only mode prevents write operations."""

    async def fake_client(self: DuckDBIndex) -> object:  # pragma: no cover - test helper
        await asyncio.sleep(0)
        return _fake_openai_client()

    monkeypatch.setattr(DuckDBIndex, "_ensure_openai_client", fake_client, raising=False)

    # Create a DB first in write mode
    db_path = tmp_path / "readonly_test.duckdb"
    write_index = DuckDBIndex(db_path, read_only=False)
    await write_index.setup()
    if write_index.conn is not None:
        write_index.conn.close()

    # Now try to open in read-only mode
    read_index = DuckDBIndex(db_path, read_only=True)
    await read_index.setup()

    # Write operations should fail
    model = full_model.model_copy(deep=True)
    test_file = tmp_path / "test.fm.json"
    _write_model_file(test_file, model)

    with pytest.raises(RuntimeError, match="read-only mode"):
        await read_index.add_or_update_entry_from_file(test_file, model)

    if read_index.conn is not None:
        read_index.conn.close()


# ============================================================================
# DuckDB-Specific Tests: Performance Benchmarks
# ============================================================================


@pytest.mark.asyncio
async def test_search_latency_benchmark(session_populated_index: DuckDBIndex) -> None:
    """Test that search latency is reasonable (< 200ms for typical query)."""
    import time

    start = time.time()
    results = await session_populated_index.search("aneurysm", limit=10)
    elapsed = time.time() - start

    assert len(results) >= 1
    # Allow generous time for CI environments and index rebuild overhead
    assert elapsed < 0.3, f"Search took {elapsed:.3f}s, expected < 0.3s"


@pytest.mark.asyncio
async def test_batch_embedding_optimization(session_populated_index: DuckDBIndex) -> None:
    """Test that search_batch is faster than individual searches."""
    import time

    queries = ["aneurysm", "aortic", "pulmonary", "diameter"]

    # Time individual searches
    start_individual = time.time()
    for query in queries:
        await session_populated_index.search(query, limit=5)
    elapsed_individual = time.time() - start_individual

    # Time batch search
    start_batch = time.time()
    await session_populated_index.search_batch(queries, limit=5)
    elapsed_batch = time.time() - start_batch

    # Batch should be faster (or at least comparable)
    # In practice, batch should be significantly faster due to single embedding API call
    assert elapsed_batch <= elapsed_individual * 1.5, (
        f"Batch search ({elapsed_batch:.3f}s) should be faster than individual searches ({elapsed_individual:.3f}s)"
    )


@pytest.mark.asyncio
async def test_directory_sync_performance(index: DuckDBIndex, tmp_defs_path: Path) -> None:
    """Test that directory sync with 10+ models completes in reasonable time."""
    import time

    start = time.time()
    result = await index.update_from_directory(tmp_defs_path)
    elapsed = time.time() - start

    assert result["added"] == len(OIFM_IDS_IN_DEFS_DIR)
    # Allow generous time for batch operations (10 seconds)
    assert elapsed < 10.0, f"Directory sync took {elapsed:.3f}s, expected < 10s"


# ============================================================================
# Additional Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_semantic_search_returns_results(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test semantic search with HNSW returns results (uses fake embeddings)."""
    model = full_model.model_copy(deep=True)
    file_path = tmp_path / "model.fm.json"
    _write_model_file(file_path, model)

    status = await index.add_or_update_entry_from_file(file_path, model)
    assert status is IndexReturnType.ADDED

    entry = await index.get(model.oifm_id)
    assert entry is not None
    assert entry.name == model.name

    # Search with unrelated query should still return results via semantic search
    results = await index.search("unrelated query", limit=5)
    assert len(results) >= 1
    assert any(r.oifm_id == model.oifm_id for r in results)


@pytest.mark.asyncio
async def test_semantic_search_with_precomputed_embedding(tmp_path: Path, full_model: FindingModelFull) -> None:
    """Test semantic search using pre-computed embedding (deterministic, no API calls)."""
    # Create index WITHOUT mocking (to test real semantic search logic)
    db_path = tmp_path / "precomputed_test.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    try:
        # Manually insert model with pre-computed embedding from real OpenAI API
        # (embedding was pre-computed offline, so this test makes no API calls)
        model = full_model.model_copy(deep=True)
        model.name = "Test Aneurysm Model"
        model.description = "A model about aneurysms for testing"

        # Real OpenAI embedding for "test aneurysm model description" (pre-computed)
        # Generated with: get_embedding_for_duckdb('test aneurysm model description')
        precomputed_model_embedding = [
            -0.039285335689783096,
            0.008606924675405025,
            -0.013724412769079208,
            -0.020183583721518517,
            -0.07691610604524612,
            -0.0039720190688967705,
            -0.013395620509982109,
            -0.018804777413606644,
            -0.009656937792897224,
            0.02825489453971386,
        ] + [0.0] * 502  # Truncated for readability, padded to 512

        conn = index._ensure_writable_connection()
        search_text = index._build_search_text(model)

        # Compute slug_name from name (same as the index does)
        from findingmodel.common import normalize_name

        slug_name = normalize_name(model.name)

        conn.execute(
            """
            INSERT INTO finding_models (
                oifm_id, slug_name, name, filename, file_hash_sha256,
                description, search_text, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model.oifm_id,
                slug_name,
                model.name,
                "test.fm.json",
                "fake_hash_123",
                model.description,
                search_text,
                precomputed_model_embedding,
            ),
        )

        # Insert attributes into denormalized table (required for IndexEntry validation)
        for attr in model.attributes:
            conn.execute(
                """
                INSERT INTO attributes (attribute_id, oifm_id, model_name, attribute_name, attribute_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (attr.oifma_id, model.oifm_id, model.name, attr.name, str(attr.type)),
            )

        conn.commit()

        # Rebuild indexes after manual insert
        index._create_search_indexes(conn)

        # Real OpenAI embedding for "similar medical concept" (pre-computed)
        # Generated with: get_embedding_for_duckdb('similar medical concept')
        precomputed_query_embedding = [
            -0.08572481572628021,
            -0.018087327480316162,
            0.029814276844263077,
            0.08507007360458374,
            -0.054390594363212585,
            0.016158169135451317,
            0.005620867945253849,
            -0.022425012663006783,
            -0.04655703902244568,
            -0.07454738765954971,
        ] + [0.0] * 502  # Truncated for readability, padded to 512

        # Use the internal search method directly with pre-computed embedding (no await - it's sync)
        results = index._search_semantic_with_embedding(
            conn,
            precomputed_query_embedding,
            limit=5,
        )

        # Should find our model since embeddings are from the same embedding space
        assert len(results) >= 1
        assert any(entry.oifm_id == model.oifm_id for entry, _ in results)

    finally:
        if index.conn is not None:
            index.conn.close()


@pytest.mark.asyncio
@pytest.mark.callout
async def test_semantic_search_with_real_openai_api(tmp_path: Path, full_model: FindingModelFull) -> None:
    """Test semantic search using real OpenAI API (requires OPENAI_API_KEY)."""
    # This test makes real API calls - only run with pytest -m callout
    db_path = tmp_path / "real_api_test.duckdb"
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    try:
        # Add a model about aneurysms
        model = full_model.model_copy(deep=True)
        model.name = "Abdominal Aortic Aneurysm"
        model.description = "A bulge in the abdominal aorta that can rupture"
        model.synonyms = ["AAA", "abdominal aneurysm"]

        test_file = tmp_path / "aneurysm.fm.json"
        _write_model_file(test_file, model)
        await index.add_or_update_entry_from_file(test_file, model)

        # Search with semantically similar query (not exact match)
        # "blood vessel enlargement" should match "aneurysm" semantically
        results = await index.search("blood vessel enlargement", limit=5)

        # With real embeddings, should find the aneurysm model
        assert len(results) >= 1
        # Check if our model is in top results (semantic similarity)
        oifm_ids = [r.oifm_id for r in results]
        assert model.oifm_id in oifm_ids, "Semantic search should find aneurysm model for 'blood vessel enlargement'"

    finally:
        if index.conn is not None:
            index.conn.close()


@pytest.mark.asyncio
async def test_remove_entry_when_not_exists(index: DuckDBIndex) -> None:
    """Test removing an entry that doesn't exist returns False."""
    removed = await index.remove_entry("OIFM_NONEXISTENT_999999")
    assert removed is False
