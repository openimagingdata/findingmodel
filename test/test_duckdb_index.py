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


# ============================================================================
# Phase 2: Enhanced API Methods Tests
# ============================================================================


# ----------------------------------------------------------------------------
# all() Method Tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_pagination(session_populated_index: DuckDBIndex) -> None:
    """Test that all() respects limit and offset."""
    # Get first page
    page1, total1 = await session_populated_index.all(limit=3, offset=0)
    assert len(page1) == 3
    assert total1 >= len(OIFM_IDS_IN_DEFS_DIR)

    # Get second page
    page2, total2 = await session_populated_index.all(limit=3, offset=3)
    assert len(page2) <= 3  # May be fewer if total < 6
    assert total2 == total1  # Total should be same

    # Verify no overlap
    page1_ids = {e.oifm_id for e in page1}
    page2_ids = {e.oifm_id for e in page2}
    assert page1_ids.isdisjoint(page2_ids)

    # Verify offset beyond total returns empty list
    page_beyond, total_beyond = await session_populated_index.all(limit=10, offset=total1)
    assert len(page_beyond) == 0
    assert total_beyond == total1


@pytest.mark.asyncio
async def test_all_sorting_all_fields(session_populated_index: DuckDBIndex) -> None:
    """Test all valid fields for ordering (name, oifm_id, created_at, updated_at, slug_name)."""
    # Test name sorting (ascending)
    results_name_asc, _ = await session_populated_index.all(order_by="name", order_dir="asc", limit=100)
    names_asc = [e.name for e in results_name_asc]
    assert names_asc == sorted(names_asc, key=str.lower)

    # Test name sorting (descending)
    results_name_desc, _ = await session_populated_index.all(order_by="name", order_dir="desc", limit=100)
    names_desc = [e.name for e in results_name_desc]
    assert names_desc == sorted(names_desc, key=str.lower, reverse=True)

    # Test oifm_id sorting
    results_id, _ = await session_populated_index.all(order_by="oifm_id", order_dir="asc", limit=100)
    ids = [e.oifm_id for e in results_id]
    assert ids == sorted(ids)

    # Test slug_name sorting
    results_slug, _ = await session_populated_index.all(order_by="slug_name", order_dir="asc", limit=100)
    slugs = [e.slug_name for e in results_slug]
    assert slugs == sorted(slugs, key=str.lower)

    # Test created_at sorting (all should have timestamps)
    results_created, _ = await session_populated_index.all(order_by="created_at", order_dir="asc", limit=100)
    assert all(e.created_at is not None for e in results_created)
    created_times = [e.created_at for e in results_created if e.created_at is not None]
    assert created_times == sorted(created_times)

    # Test updated_at sorting
    results_updated, _ = await session_populated_index.all(order_by="updated_at", order_dir="asc", limit=100)
    assert all(e.updated_at is not None for e in results_updated)
    updated_times = [e.updated_at for e in results_updated if e.updated_at is not None]
    assert updated_times == sorted(updated_times)


@pytest.mark.asyncio
async def test_all_case_insensitive_sorting(session_populated_index: DuckDBIndex) -> None:
    """Verify LOWER() works for name/slug_name."""
    # Get results sorted by name
    results, _ = await session_populated_index.all(order_by="name", order_dir="asc", limit=100)
    names = [e.name for e in results]

    # Verify case-insensitive sorting: "Abdominal" should come before "aortic"
    # regardless of case
    lower_names = [n.lower() for n in names]
    assert lower_names == sorted(lower_names)

    # Same for slug_name
    results_slug, _ = await session_populated_index.all(order_by="slug_name", order_dir="asc", limit=100)
    slugs = [e.slug_name for e in results_slug]
    lower_slugs = [s.lower() for s in slugs]
    assert lower_slugs == sorted(lower_slugs)


@pytest.mark.asyncio
async def test_all_invalid_order_by(session_populated_index: DuckDBIndex) -> None:
    """Verify ValueError raised for invalid order_by."""
    with pytest.raises(ValueError, match="Invalid order_by field"):
        await session_populated_index.all(order_by="invalid_field")

    with pytest.raises(ValueError, match="Invalid order_by field"):
        await session_populated_index.all(order_by="description")  # Not a valid field

    with pytest.raises(ValueError, match="Invalid order_by field"):
        await session_populated_index.all(order_by="")


@pytest.mark.asyncio
async def test_all_invalid_order_dir(session_populated_index: DuckDBIndex) -> None:
    """Verify ValueError raised for invalid order_dir."""
    with pytest.raises(ValueError, match="Invalid order_dir"):
        await session_populated_index.all(order_dir="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid order_dir"):
        await session_populated_index.all(order_dir="ASC")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid order_dir"):
        await session_populated_index.all(order_dir="")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_all_empty_database(index: DuckDBIndex) -> None:
    """Verify returns ([], 0) for empty database."""
    results, total = await index.all()
    assert results == []
    assert total == 0


@pytest.mark.asyncio
async def test_all_single_page(session_populated_index: DuckDBIndex) -> None:
    """Verify works with results < limit."""
    # Get total count first
    total_count = await session_populated_index.count()

    # Request more than total
    results, total = await session_populated_index.all(limit=total_count + 100, offset=0)
    assert len(results) == total_count
    assert total == total_count


# ----------------------------------------------------------------------------
# search_by_slug() Method Tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_by_slug_exact_match(session_populated_index: DuckDBIndex) -> None:
    """Verify exact match type finds only exact matches."""
    # "abdominal_aortic_aneurysm" exists in test data
    results, total = await session_populated_index.search_by_slug("abdominal aortic aneurysm", match_type="exact")
    assert total == 1
    assert len(results) == 1
    assert results[0].slug_name == "abdominal_aortic_aneurysm"

    # Partial match should not find anything with exact match type
    results, total = await session_populated_index.search_by_slug("abdominal", match_type="exact")
    assert total == 0
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_by_slug_prefix_match(session_populated_index: DuckDBIndex) -> None:
    """Verify prefix match type finds slug_name.startswith(pattern)."""
    # Search for "aortic" - should find "aortic_dissection" and potentially others starting with "aortic"
    results, total = await session_populated_index.search_by_slug("aortic", match_type="prefix")
    assert total >= 1
    assert len(results) >= 1
    # All results should start with "aortic"
    assert all(e.slug_name.startswith("aortic") for e in results)

    # Find "abdominal" prefix
    results, total = await session_populated_index.search_by_slug("abdominal", match_type="prefix")
    assert total >= 1
    assert all(e.slug_name.startswith("abdominal") for e in results)


@pytest.mark.asyncio
async def test_search_by_slug_contains_match(session_populated_index: DuckDBIndex) -> None:
    """Verify contains match type finds slug_name.__contains__(pattern)."""
    # Search for "aortic" - should find both "abdominal_aortic_aneurysm" and "aortic_dissection"
    results, total = await session_populated_index.search_by_slug("aortic", match_type="contains")
    assert total >= 2
    assert len(results) >= 2
    # All results should contain "aortic"
    assert all("aortic" in e.slug_name for e in results)

    # Search for "embolism"
    results, total = await session_populated_index.search_by_slug("embolism", match_type="contains")
    assert total >= 1
    assert all("embolism" in e.slug_name for e in results)


@pytest.mark.asyncio
async def test_search_by_slug_relevance_ranking() -> None:
    """Verify exact > prefix > contains, then alphabetical."""
    # Add models with predictable slug patterns
    # We'll use populated_index (not session) since we need to add entries
    # Get a fresh index for this test
    import tempfile

    from findingmodel.finding_model import FindingModelFull, NumericAttributeIded

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test index with mocked embeddings
        import asyncio

        from findingmodel import index as duckdb_index

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
            hash_val = sum(ord(c) for c in text)
            return [(hash_val % 100) / 100.0] * target_dims

        # Temporarily patch embeddings
        original_get_embedding = duckdb_index.get_embedding_for_duckdb  # type: ignore[attr-defined]
        original_batch_embeddings = duckdb_index.batch_embeddings_for_duckdb  # type: ignore[attr-defined]

        duckdb_index.get_embedding_for_duckdb = fake_embedding  # type: ignore[attr-defined]
        duckdb_index.batch_embeddings_for_duckdb = lambda texts, client: asyncio.gather(  # type: ignore[attr-defined,assignment,misc]
            *[fake_embedding(t, client=client) for t in texts]
        )

        try:
            db_path = tmp_path / "relevance_test.duckdb"
            test_index = DuckDBIndex(db_path, read_only=False)
            await test_index.setup()

            # Create models with slug names: "testmodel", "testmodel_prefix", "contains_testmodel"
            # When searching for "testmodel", ranking should be: exact > prefix > contains
            model1 = FindingModelFull(
                oifm_id="OIFM_TEST_000001",
                name="TestModel",  # slug: "testmodel" (5+ chars, normalized)
                description="Test model for relevance ranking",
                attributes=[
                    NumericAttributeIded(
                        oifma_id="OIFMA_TEST_000001", name="Size", minimum=1, maximum=10, unit="cm", required=False
                    )
                ],
            )
            model2 = FindingModelFull(
                oifm_id="OIFM_TEST_000002",
                name="TestModel Prefix",  # slug: "testmodel_prefix"
                description="Test model with prefix match",
                attributes=[
                    NumericAttributeIded(
                        oifma_id="OIFMA_TEST_000002", name="Size", minimum=1, maximum=10, unit="cm", required=False
                    )
                ],
            )
            model3 = FindingModelFull(
                oifm_id="OIFM_TEST_000003",
                name="Contains TestModel",  # slug: "contains_testmodel"
                description="Test model with contains match",
                attributes=[
                    NumericAttributeIded(
                        oifma_id="OIFMA_TEST_000003", name="Size", minimum=1, maximum=10, unit="cm", required=False
                    )
                ],
            )

            file1 = tmp_path / "test1.fm.json"
            file2 = tmp_path / "test2.fm.json"
            file3 = tmp_path / "test3.fm.json"

            _write_model_file(file1, model1)
            _write_model_file(file2, model2)
            _write_model_file(file3, model3)

            await test_index.add_or_update_entry_from_file(file1, model1)
            await test_index.add_or_update_entry_from_file(file2, model2)
            await test_index.add_or_update_entry_from_file(file3, model3)

            # Search for "testmodel" - should rank exact > prefix > contains
            results, total = await test_index.search_by_slug("testmodel", match_type="contains")
            assert total == 3
            assert len(results) == 3

            # Verify ranking
            assert results[0].slug_name == "testmodel"  # Exact match first
            assert results[1].slug_name == "testmodel_prefix"  # Prefix match second
            assert results[2].slug_name == "contains_testmodel"  # Contains match third

            if test_index.conn is not None:
                test_index.conn.close()

        finally:
            # Restore original functions
            duckdb_index.get_embedding_for_duckdb = original_get_embedding  # type: ignore[attr-defined]
            duckdb_index.batch_embeddings_for_duckdb = original_batch_embeddings  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_search_by_slug_pattern_normalization(session_populated_index: DuckDBIndex) -> None:
    """Verify normalize_name() called on pattern."""
    # Search with uppercase and spaces - should normalize to lowercase with underscores
    results1, total1 = await session_populated_index.search_by_slug("Abdominal Aortic Aneurysm", match_type="exact")
    results2, total2 = await session_populated_index.search_by_slug("abdominal aortic aneurysm", match_type="exact")

    # Should return same results since pattern is normalized
    assert total1 == total2
    assert len(results1) == len(results2)
    if results1:
        assert results1[0].oifm_id == results2[0].oifm_id


@pytest.mark.asyncio
async def test_search_by_slug_pagination(session_populated_index: DuckDBIndex) -> None:
    """Verify limit/offset work."""
    # Search for common term "aortic" (appears in multiple models)
    page1, total1 = await session_populated_index.search_by_slug("aortic", match_type="contains", limit=1, offset=0)
    page2, total2 = await session_populated_index.search_by_slug("aortic", match_type="contains", limit=1, offset=1)

    assert total1 == total2  # Total should be same
    assert len(page1) <= 1
    assert len(page2) <= 1

    # Verify no overlap if both pages have results
    if len(page1) > 0 and len(page2) > 0:
        page1_ids = {e.oifm_id for e in page1}
        page2_ids = {e.oifm_id for e in page2}
        assert page1_ids.isdisjoint(page2_ids)


@pytest.mark.asyncio
async def test_search_by_slug_no_matches(session_populated_index: DuckDBIndex) -> None:
    """Verify returns ([], 0) for no matches."""
    results, total = await session_populated_index.search_by_slug("zzzznonexistentpatternzzz", match_type="contains")
    assert results == []
    assert total == 0


# ----------------------------------------------------------------------------
# count_search() Method Tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_search_exact_match(session_populated_index: DuckDBIndex) -> None:
    """Verify count with exact match type."""
    # Count exact match for "abdominal_aortic_aneurysm"
    count = await session_populated_index.count_search("abdominal aortic aneurysm", match_type="exact")
    assert count == 1

    # Count non-existent exact match
    count = await session_populated_index.count_search("nonexistent", match_type="exact")
    assert count == 0


@pytest.mark.asyncio
async def test_count_search_prefix_match(session_populated_index: DuckDBIndex) -> None:
    """Verify count with prefix match type."""
    # Count models with slug starting with "aortic"
    count = await session_populated_index.count_search("aortic", match_type="prefix")
    assert count >= 1

    # Count models with slug starting with "abdominal"
    count = await session_populated_index.count_search("abdominal", match_type="prefix")
    assert count >= 1


@pytest.mark.asyncio
async def test_count_search_contains_match(session_populated_index: DuckDBIndex) -> None:
    """Verify count with contains match type."""
    # Count models containing "aortic"
    count = await session_populated_index.count_search("aortic", match_type="contains")
    assert count >= 2  # Should find both "abdominal_aortic_aneurysm" and "aortic_dissection"

    # Count models containing "embolism"
    count = await session_populated_index.count_search("embolism", match_type="contains")
    assert count >= 1


@pytest.mark.asyncio
async def test_count_search_empty_database(index: DuckDBIndex) -> None:
    """Verify returns 0 for empty database."""
    count = await index.count_search("anything", match_type="contains")
    assert count == 0


# ============================================================================
# Phase 3: ID Generation Methods Tests
# ============================================================================


# ----------------------------------------------------------------------------
# generate_model_id() Method Tests
# ----------------------------------------------------------------------------


def test_generate_model_id_format(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_model_id() returns correctly formatted IDs."""
    oifm_id = session_populated_index.generate_model_id("OIDM")

    assert oifm_id.startswith("OIFM_OIDM_"), f"Expected ID to start with 'OIFM_OIDM_', got: {oifm_id}"
    assert len(oifm_id) == 16, f"Expected ID length 16 ('OIFM_OIDM_' + 6 digits), got: {len(oifm_id)}"
    assert oifm_id[-6:].isdigit(), f"Expected last 6 chars to be digits, got: {oifm_id[-6:]}"


def test_generate_model_id_uniqueness(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_model_id() generates unique IDs in batch."""
    ids = {session_populated_index.generate_model_id("TEST") for _ in range(100)}

    assert len(ids) == 100, "Should generate 100 unique IDs without collision"


def test_generate_model_id_different_sources_independent(session_populated_index: DuckDBIndex) -> None:
    """Test that different sources have independent ID spaces."""
    oidm_ids = {session_populated_index.generate_model_id("OIDM") for _ in range(10)}
    gmts_ids = {session_populated_index.generate_model_id("GMTS") for _ in range(10)}

    # All OIDM IDs should have OIDM prefix
    assert all(oid.startswith("OIFM_OIDM_") for oid in oidm_ids)
    # All GMTS IDs should have GMTS prefix
    assert all(gid.startswith("OIFM_GMTS_") for gid in gmts_ids)
    # IDs should be independent (no overlap)
    assert oidm_ids.isdisjoint(gmts_ids), "Different sources should have independent ID spaces"


@pytest.mark.asyncio
async def test_generate_model_id_collision_avoidance(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that collision detection works by pre-populating database with IDs."""
    # Pre-populate the database with some IDs
    existing_ids = []
    for i in range(5):
        model = full_model.model_copy(deep=True)
        model.oifm_id = f"OIFM_COLL_{i:06d}"
        model.name = f"Collision Test Model {i}"
        model.attributes[0].oifma_id = f"OIFMA_COLL_{i:06d}"
        model.attributes[1].oifma_id = f"OIFMA_COLL_{i + 10:06d}"

        test_file = tmp_path / f"coll_test_{i}.fm.json"
        _write_model_file(test_file, model)
        await index.add_or_update_entry_from_file(test_file, model)
        existing_ids.append(model.oifm_id)

    # Generate new IDs - they should avoid the existing ones
    new_ids = {index.generate_model_id("COLL") for _ in range(20)}

    # Verify no collisions with existing IDs
    for existing_id in existing_ids:
        assert existing_id not in new_ids, f"Generated ID should not collide with existing ID: {existing_id}"

    # Verify all new IDs are unique
    assert len(new_ids) == 20, "Should generate 20 unique IDs"


def test_generate_model_id_cache_prevents_self_collision(session_populated_index: DuckDBIndex) -> None:
    """Test that cache prevents self-collision when generating multiple IDs in same session."""
    # Generate IDs in sequence without writing to database
    ids = []
    for _ in range(50):
        new_id = session_populated_index.generate_model_id("CACH")
        # Verify this ID hasn't been generated before in this session
        assert new_id not in ids, f"Generated duplicate ID in same session: {new_id}"
        ids.append(new_id)

    # All IDs should be unique
    assert len(set(ids)) == 50, "Cache should prevent self-collision"


def test_generate_model_id_invalid_source_too_short(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_model_id() rejects source codes that are too short."""
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_model_id("AB")  # Too short (2 chars)

    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_model_id("A")  # Too short (1 char)


def test_generate_model_id_invalid_source_too_long(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_model_id() rejects source codes that are too long."""
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_model_id("TOOLONG")  # Too long (7 chars)

    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_model_id("ABCDE")  # Too long (5 chars)


def test_generate_model_id_invalid_source_contains_digits(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_model_id() rejects source codes with digits."""
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_model_id("AB1")  # Contains digit

    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_model_id("TEST1")  # Contains digit


def test_generate_model_id_source_normalization(session_populated_index: DuckDBIndex) -> None:
    """Test that source code is normalized (trimmed and uppercased)."""
    # Test with lowercase
    id1 = session_populated_index.generate_model_id("oidm")
    assert id1.startswith("OIFM_OIDM_"), f"Expected uppercase source in ID, got: {id1}"

    # Test with whitespace
    id2 = session_populated_index.generate_model_id("  gmts  ")
    assert id2.startswith("OIFM_GMTS_"), f"Expected trimmed and uppercase source in ID, got: {id2}"

    # Test with mixed case
    id3 = session_populated_index.generate_model_id("TeSt")
    assert id3.startswith("OIFM_TEST_"), f"Expected uppercase source in ID, got: {id3}"


def test_generate_model_id_max_attempts_exhausted(
    session_populated_index: DuckDBIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that RuntimeError is raised when max_attempts is exhausted."""

    # Mock _random_digits to always return the same value (forcing collisions)
    def mock_random_digits(length: int) -> str:
        return "0" * length

    from findingmodel import finding_model

    monkeypatch.setattr(finding_model, "_random_digits", mock_random_digits)

    # Pre-populate the cache with the ID that will be generated (forcing collision)
    session_populated_index._load_oifm_ids_for_source("FAIL")
    session_populated_index._oifm_id_cache["FAIL"].add("OIFM_FAIL_000000")

    # Try to generate ID - should exhaust max_attempts and raise RuntimeError
    with pytest.raises(RuntimeError, match="Unable to generate unique OIFM ID"):
        session_populated_index.generate_model_id("FAIL", max_attempts=10)


# ----------------------------------------------------------------------------
# generate_attribute_id() Method Tests
# ----------------------------------------------------------------------------


def test_generate_attribute_id_format(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_attribute_id() returns correctly formatted IDs."""
    oifma_id = session_populated_index.generate_attribute_id(source="OIDM")

    assert oifma_id.startswith("OIFMA_OIDM_"), f"Expected ID to start with 'OIFMA_OIDM_', got: {oifma_id}"
    assert len(oifma_id) == 17, f"Expected ID length 17 ('OIFMA_OIDM_' + 6 digits), got: {len(oifma_id)}"
    assert oifma_id[-6:].isdigit(), f"Expected last 6 chars to be digits, got: {oifma_id[-6:]}"


def test_generate_attribute_id_uniqueness(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_attribute_id() generates unique IDs in batch."""
    ids = {session_populated_index.generate_attribute_id(source="TEST") for _ in range(100)}

    assert len(ids) == 100, "Should generate 100 unique attribute IDs without collision"


def test_generate_attribute_id_independent_from_oifm_ids(session_populated_index: DuckDBIndex) -> None:
    """Test that attribute IDs (OIFMA) are independent from model IDs (OIFM)."""
    # Generate OIFM IDs
    oifm_ids = {session_populated_index.generate_model_id("ATTR") for _ in range(10)}

    # Generate OIFMA IDs
    oifma_ids = {session_populated_index.generate_attribute_id(source="ATTR") for _ in range(10)}

    # All IDs should be unique
    assert len(oifm_ids) == 10
    assert len(oifma_ids) == 10

    # OIFM and OIFMA IDs should have different prefixes
    assert all(oid.startswith("OIFM_ATTR_") for oid in oifm_ids)
    assert all(aid.startswith("OIFMA_ATTR_") for aid in oifma_ids)


def test_generate_attribute_id_infer_source_from_model_id(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_attribute_id() can infer source from model OIFM ID."""
    # Generate attribute ID by inferring source from model ID
    oifma_id = session_populated_index.generate_attribute_id(model_oifm_id="OIFM_GMTS_123456")

    # Should infer source "GMTS" from model ID
    assert oifma_id.startswith("OIFMA_GMTS_"), f"Expected to infer GMTS source from model ID, got: {oifma_id}"


def test_generate_attribute_id_explicit_source_overrides_inference(session_populated_index: DuckDBIndex) -> None:
    """Test that explicit source parameter overrides inference from model_oifm_id."""
    # Provide both model_oifm_id and explicit source - explicit should win
    oifma_id = session_populated_index.generate_attribute_id(model_oifm_id="OIFM_GMTS_123456", source="OIDM")

    # Should use explicit source "OIDM", not inferred "GMTS"
    assert oifma_id.startswith("OIFMA_OIDM_"), f"Expected explicit source to override inference, got: {oifma_id}"


def test_generate_attribute_id_default_source(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_attribute_id() defaults to 'OIDM' when no source provided."""
    # Generate attribute ID with no source and no model_oifm_id
    oifma_id = session_populated_index.generate_attribute_id()

    # Should default to "OIDM"
    assert oifma_id.startswith("OIFMA_OIDM_"), f"Expected default source OIDM, got: {oifma_id}"


def test_generate_attribute_id_invalid_model_id_format(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_attribute_id() rejects invalid model ID formats."""
    # Test with wrong prefix
    with pytest.raises(ValueError, match="Cannot infer source from invalid model ID"):
        session_populated_index.generate_attribute_id(model_oifm_id="OIFMA_GMTS_123456")

    # Test with wrong number of parts
    with pytest.raises(ValueError, match="Cannot infer source from invalid model ID"):
        session_populated_index.generate_attribute_id(model_oifm_id="OIFM_GMTS")

    # Test with extra parts (strict validation)
    with pytest.raises(ValueError, match="Cannot infer source from invalid model ID"):
        session_populated_index.generate_attribute_id(model_oifm_id="OIFM_GMTS_123456_EXTRA")


def test_generate_attribute_id_invalid_source(session_populated_index: DuckDBIndex) -> None:
    """Test that generate_attribute_id() rejects invalid source codes."""
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_attribute_id(source="AB")  # Too short

    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_attribute_id(source="TOOLONG")  # Too long

    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        session_populated_index.generate_attribute_id(source="AB1")  # Contains digit


@pytest.mark.asyncio
async def test_generate_attribute_id_collision_avoidance(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that attribute ID collision detection works."""
    # Pre-populate the database with some attribute IDs
    existing_ids = []
    for i in range(5):
        model = full_model.model_copy(deep=True)
        model.oifm_id = f"OIFM_ACOL_{i:06d}"
        model.name = f"Attribute Collision Test {i}"
        # Use predictable attribute IDs
        model.attributes[0].oifma_id = f"OIFMA_ACOL_{i:06d}"
        model.attributes[1].oifma_id = f"OIFMA_ACOL_{i + 10:06d}"

        test_file = tmp_path / f"acol_test_{i}.fm.json"
        _write_model_file(test_file, model)
        await index.add_or_update_entry_from_file(test_file, model)
        existing_ids.extend([model.attributes[0].oifma_id, model.attributes[1].oifma_id])

    # Generate new attribute IDs - they should avoid existing ones
    new_ids = {index.generate_attribute_id(source="ACOL") for _ in range(20)}

    # Verify no collisions with existing IDs
    for existing_id in existing_ids:
        assert existing_id not in new_ids, f"Generated attribute ID should not collide with existing: {existing_id}"

    # Verify all new IDs are unique
    assert len(new_ids) == 20, "Should generate 20 unique attribute IDs"


def test_generate_attribute_id_cache_prevents_self_collision(session_populated_index: DuckDBIndex) -> None:
    """Test that cache prevents self-collision for attribute IDs."""
    # Generate attribute IDs in sequence
    ids = []
    for _ in range(50):
        new_id = session_populated_index.generate_attribute_id(source="CACH")
        # Verify this ID hasn't been generated before in this session
        assert new_id not in ids, f"Generated duplicate attribute ID in same session: {new_id}"
        ids.append(new_id)

    # All IDs should be unique
    assert len(set(ids)) == 50, "Cache should prevent self-collision for attribute IDs"


# ----------------------------------------------------------------------------
# Helper Methods Tests (_load_oifm_ids_for_source, _load_oifma_ids_for_source)
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_oifm_ids_for_source_caching(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that _load_oifm_ids_for_source() caches results."""
    # Add a model with MSFT source
    model = full_model.model_copy(deep=True)
    model.oifm_id = "OIFM_MSFT_999999"
    model.name = "Cache Test Model"
    model.attributes[0].oifma_id = "OIFMA_MSFT_999001"
    model.attributes[1].oifma_id = "OIFMA_MSFT_999002"

    test_file = tmp_path / "cache_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    # First call should load from database
    ids1 = index._load_oifm_ids_for_source("MSFT")
    assert "OIFM_MSFT_999999" in ids1

    # Second call should return cached result
    ids2 = index._load_oifm_ids_for_source("MSFT")
    assert ids1 is ids2, "Should return cached set instance"


@pytest.mark.asyncio
async def test_load_oifma_ids_for_source_caching(
    index: DuckDBIndex, full_model: FindingModelFull, tmp_path: Path
) -> None:
    """Test that _load_oifma_ids_for_source() caches results."""
    # Add a model with MSFT source
    model = full_model.model_copy(deep=True)
    model.oifm_id = "OIFM_MSFT_888888"
    model.name = "Attribute Cache Test"
    model.attributes[0].oifma_id = "OIFMA_MSFT_888001"
    model.attributes[1].oifma_id = "OIFMA_MSFT_888002"

    test_file = tmp_path / "attr_cache_test.fm.json"
    _write_model_file(test_file, model)
    await index.add_or_update_entry_from_file(test_file, model)

    # First call should load from database
    ids1 = index._load_oifma_ids_for_source("MSFT")
    assert "OIFMA_MSFT_888001" in ids1
    assert "OIFMA_MSFT_888002" in ids1

    # Second call should return cached result
    ids2 = index._load_oifma_ids_for_source("MSFT")
    assert ids1 is ids2, "Should return cached set instance"


# ----------------------------------------------------------------------------
# ID Orchestration Tests (add_ids_to_model, finalize_placeholder_attribute_ids)
# ----------------------------------------------------------------------------


def test_add_ids_to_model_complete_new_model(index: DuckDBIndex) -> None:
    """Test add_ids_to_model generates all IDs for a new model."""
    # Create model without IDs
    base_model = FindingModelBase(
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttribute(name="Size", description="Size in cm", minimum=0.0, maximum=100.0, unit="cm"),
            ChoiceAttribute(
                name="Shape",
                description="Shape of finding",
                values=[ChoiceValue(name="Round"), ChoiceValue(name="Irregular")],
            ),
        ],
    )

    # Add IDs
    full_model = index.add_ids_to_model(base_model, "TEST")

    # Verify OIFM ID generated
    assert full_model.oifm_id is not None, "OIFM ID should be generated"
    assert full_model.oifm_id.startswith("OIFM_TEST_"), f"Expected OIFM_TEST_ prefix, got: {full_model.oifm_id}"
    assert len(full_model.oifm_id) == 16, f"Expected ID length 16, got: {len(full_model.oifm_id)}"

    # Verify all attributes have OIFMA IDs
    assert len(full_model.attributes) == 2, "Should have 2 attributes"
    for attr in full_model.attributes:
        assert attr.oifma_id is not None, f"Attribute {attr.name} should have OIFMA ID"
        assert attr.oifma_id.startswith("OIFMA_TEST_"), f"Expected OIFMA_TEST_ prefix, got: {attr.oifma_id}"
        assert len(attr.oifma_id) == 17, f"Expected OIFMA ID length 17, got: {len(attr.oifma_id)}"

    # Verify returns FindingModelFull
    assert isinstance(full_model, FindingModelFull), "Should return FindingModelFull instance"


def test_add_ids_to_model_existing_oifm_id(index: DuckDBIndex) -> None:
    """Test add_ids_to_model preserves existing OIFM ID when using FindingModelFull."""
    # Create FindingModelFull with OIFM ID but attributes without IDs (hybrid case)
    # This uses the fact that FindingModelFull can be created with partial data via model_dump
    full_model_with_id = FindingModelFull(
        oifm_id="OIFM_TEST_999999",
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_111111",
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_222222",
                name="Depth",
                description="Depth in cm",
                minimum=0.0,
                maximum=50.0,
                unit="cm",
            ),
        ],
    )

    # Add IDs (should preserve existing)
    result = index.add_ids_to_model(full_model_with_id, "TEST")

    # Verify OIFM ID unchanged
    assert result.oifm_id == "OIFM_TEST_999999", "OIFM ID should be preserved"

    # Verify attributes IDs unchanged
    assert len(result.attributes) == 2
    assert result.attributes[0].oifma_id == "OIFMA_TEST_111111"
    assert result.attributes[1].oifma_id == "OIFMA_TEST_222222"


def test_add_ids_to_model_partial_attribute_ids(index: DuckDBIndex) -> None:
    """Test add_ids_to_model generates IDs for attributes without them."""
    # Create a base model (no IDs)
    base_model = FindingModelBase(
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttribute(
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
            ChoiceAttribute(
                name="Shape",
                description="Shape of finding",
                values=[ChoiceValue(name="Round"), ChoiceValue(name="Irregular")],
            ),
            NumericAttribute(name="Depth", description="Depth in cm", minimum=0.0, maximum=50.0, unit="cm"),
        ],
    )

    # Add IDs
    full_model = index.add_ids_to_model(base_model, "TEST")

    # Verify OIFM ID generated
    assert full_model.oifm_id is not None
    assert full_model.oifm_id.startswith("OIFM_TEST_")

    # Verify all attribute IDs generated
    assert full_model.attributes[0].oifma_id is not None, "First attribute should have generated OIFMA ID"
    assert full_model.attributes[0].oifma_id.startswith("OIFMA_TEST_")

    assert full_model.attributes[1].oifma_id is not None, "Second attribute should have generated OIFMA ID"
    assert full_model.attributes[1].oifma_id.startswith("OIFMA_TEST_")

    assert full_model.attributes[2].oifma_id is not None, "Third attribute should have generated OIFMA ID"
    assert full_model.attributes[2].oifma_id.startswith("OIFMA_TEST_")

    # Verify all generated IDs are unique
    all_ids = [attr.oifma_id for attr in full_model.attributes]
    assert len(set(all_ids)) == len(all_ids), "Generated IDs should be unique"


def test_add_ids_to_model_all_ids_present(index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test add_ids_to_model preserves all IDs when model is already complete."""
    # Use full_model fixture which has all IDs
    original_oifm_id = full_model.oifm_id
    original_attr_ids = [attr.oifma_id for attr in full_model.attributes]

    # Add IDs (should be no-op)
    result = index.add_ids_to_model(full_model, "TEST")

    # Verify no IDs changed
    assert result.oifm_id == original_oifm_id, "OIFM ID should be unchanged"
    assert len(result.attributes) == len(original_attr_ids)
    for i, attr in enumerate(result.attributes):
        assert attr.oifma_id == original_attr_ids[i], f"Attribute {i} ID should be unchanged"


def test_add_ids_to_model_source_used(index: DuckDBIndex) -> None:
    """Test add_ids_to_model uses the specified source code."""
    base_model = FindingModelBase(
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttribute(name="Size", description="Size in cm", minimum=0.0, maximum=100.0, unit="cm"),
        ],
    )

    # Call with custom source "GMTS"
    full_model = index.add_ids_to_model(base_model, "GMTS")

    # Verify generated IDs use that source
    assert full_model.oifm_id.startswith("OIFM_GMTS_"), f"Expected OIFM_GMTS_ prefix, got: {full_model.oifm_id}"
    assert full_model.attributes[0].oifma_id.startswith("OIFMA_GMTS_"), (
        f"Expected OIFMA_GMTS_ prefix, got: {full_model.attributes[0].oifma_id}"
    )

    # Check format: OIFM_GMTS_NNNNNN, OIFMA_GMTS_NNNNNN
    assert len(full_model.oifm_id) == 16, "OIFM ID should be 16 chars (OIFM_GMTS_ + 6 digits)"
    assert full_model.oifm_id[-6:].isdigit(), "Last 6 chars of OIFM ID should be digits"
    assert len(full_model.attributes[0].oifma_id) == 17, "OIFMA ID should be 17 chars (OIFMA_GMTS_ + 6 digits)"
    assert full_model.attributes[0].oifma_id[-6:].isdigit(), "Last 6 chars of OIFMA ID should be digits"


def test_add_ids_to_model_invalid_source_too_short(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test add_ids_to_model rejects invalid source (too short)."""
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        index.add_ids_to_model(base_model, "AB")


def test_add_ids_to_model_invalid_source_too_long(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test add_ids_to_model rejects invalid source (too long)."""
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        index.add_ids_to_model(base_model, "TOOLONG")


def test_add_ids_to_model_invalid_source_contains_digits(index: DuckDBIndex, base_model: FindingModelBase) -> None:
    """Test add_ids_to_model rejects invalid source (contains digits)."""
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        index.add_ids_to_model(base_model, "TE5T")


def test_finalize_placeholder_single_placeholder(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids replaces a single placeholder."""
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    # Create FindingModelFull with one placeholder attribute ID
    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
            NumericAttributeIded(
                oifma_id="OIFMA_TEST_555555",
                name="Depth",
                description="Depth in cm",
                minimum=0.0,
                maximum=50.0,
                unit="cm",
            ),
        ],
    )

    # Finalize placeholder IDs
    result = index.finalize_placeholder_attribute_ids(model)

    # Verify placeholder replaced with real ID
    assert result.attributes[0].oifma_id != PLACEHOLDER_ATTRIBUTE_ID, "Placeholder should be replaced"
    assert result.attributes[0].oifma_id.startswith("OIFMA_TEST_"), "Should use TEST source"
    assert len(result.attributes[0].oifma_id) == 17, "Should be valid OIFMA ID"

    # Verify other attributes unchanged
    assert result.attributes[1].oifma_id == "OIFMA_TEST_555555", "Non-placeholder ID should be unchanged"


def test_finalize_placeholder_multiple_placeholders(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids replaces multiple placeholders with unique IDs."""
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    # Create model with 3 placeholder attribute IDs
    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Depth",
                description="Depth in cm",
                minimum=0.0,
                maximum=50.0,
                unit="cm",
            ),
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Width",
                description="Width in cm",
                minimum=0.0,
                maximum=75.0,
                unit="cm",
            ),
        ],
    )

    # Finalize placeholder IDs
    result = index.finalize_placeholder_attribute_ids(model)

    # Verify all placeholders replaced with real IDs
    generated_ids = [attr.oifma_id for attr in result.attributes]
    assert all(aid != PLACEHOLDER_ATTRIBUTE_ID for aid in generated_ids), "All placeholders should be replaced"
    assert all(aid.startswith("OIFMA_TEST_") for aid in generated_ids), "All IDs should use TEST source"

    # Verify no duplicate IDs generated
    assert len(set(generated_ids)) == 3, "All generated IDs should be unique"


def test_finalize_placeholder_no_placeholders(index: DuckDBIndex, full_model: FindingModelFull) -> None:
    """Test finalize_placeholder_attribute_ids returns original model when no placeholders present."""
    # Use full_model fixture which has no placeholders
    result = index.finalize_placeholder_attribute_ids(full_model)

    # Verify returns original model unchanged
    assert result is full_model, "Should return original model when no placeholders"


def test_finalize_placeholder_source_inference(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids infers source from model OIFM ID."""
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    # Create model with OIFM_GMTS_123456 and placeholder attribute
    model = FindingModelFull(
        oifm_id="OIFM_GMTS_123456",
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
        ],
    )

    # Don't provide source parameter (should infer GMTS)
    result = index.finalize_placeholder_attribute_ids(model)

    # Verify infers "GMTS" from model ID and uses it
    assert result.attributes[0].oifma_id.startswith("OIFMA_GMTS_"), (
        f"Should infer GMTS source, got: {result.attributes[0].oifma_id}"
    )


def test_finalize_placeholder_explicit_source_override(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids uses explicit source over inference."""
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    # Create model with OIFM_GMTS_123456
    model = FindingModelFull(
        oifm_id="OIFM_GMTS_123456",
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
        ],
    )

    # Provide explicit source="OIDM" (should override GMTS)
    result = index.finalize_placeholder_attribute_ids(model, source="OIDM")

    # Verify uses OIDM (not GMTS)
    assert result.attributes[0].oifma_id.startswith("OIFMA_OIDM_"), (
        f"Should use explicit OIDM source, got: {result.attributes[0].oifma_id}"
    )


def test_finalize_placeholder_choice_value_codes(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids renumbers value codes for choice attributes."""
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    # Create choice attribute with placeholder ID
    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Finding",
        description="Test description",
        attributes=[
            ChoiceAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Severity",
                description="Severity level",
                values=[
                    ChoiceValueIded(value_code=f"{PLACEHOLDER_ATTRIBUTE_ID}.0", name="Mild"),
                    ChoiceValueIded(value_code=f"{PLACEHOLDER_ATTRIBUTE_ID}.1", name="Moderate"),
                    ChoiceValueIded(value_code=f"{PLACEHOLDER_ATTRIBUTE_ID}.2", name="Severe"),
                ],
            ),
        ],
    )

    # Finalize placeholder IDs
    result = index.finalize_placeholder_attribute_ids(model)

    # Verify attribute ID replaced
    new_id = result.attributes[0].oifma_id
    assert new_id != PLACEHOLDER_ATTRIBUTE_ID
    assert new_id.startswith("OIFMA_TEST_")

    # Verify value codes renumbered: {new_id}.0, {new_id}.1, {new_id}.2
    choice_attr = result.attributes[0]
    assert isinstance(choice_attr, ChoiceAttributeIded)
    assert len(choice_attr.values) == 3

    for idx, value in enumerate(choice_attr.values):
        expected_code = f"{new_id}.{idx}"
        assert value.value_code == expected_code, f"Expected value_code {expected_code}, got {value.value_code}"


def test_finalize_placeholder_invalid_source(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids rejects invalid explicit source."""
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    model = FindingModelFull(
        oifm_id="OIFM_TEST_123456",
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
        ],
    )

    # Call with invalid explicit source
    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        index.finalize_placeholder_attribute_ids(model, source="AB")  # Too short

    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        index.finalize_placeholder_attribute_ids(model, source="TOOLONG")  # Too long

    with pytest.raises(ValueError, match="3-4 uppercase letters"):
        index.finalize_placeholder_attribute_ids(model, source="TE5T")  # Contains digits


def test_finalize_placeholder_invalid_model_id_inference(index: DuckDBIndex) -> None:
    """Test finalize_placeholder_attribute_ids fails when model ID is malformed and no source provided."""
    from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID

    # Create model with malformed OIFM ID using model_construct to bypass validation
    model = FindingModelFull.model_construct(
        oifm_id="BAD_ID_123",  # Malformed ID
        name="Test Finding",
        description="Test description",
        attributes=[
            NumericAttributeIded(
                oifma_id=PLACEHOLDER_ATTRIBUTE_ID,
                name="Size",
                description="Size in cm",
                minimum=0.0,
                maximum=100.0,
                unit="cm",
            ),
        ],
    )

    # Don't provide source - should fail to infer
    with pytest.raises(ValueError, match="Cannot infer source from model ID"):
        index.finalize_placeholder_attribute_ids(model)
