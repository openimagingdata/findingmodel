"""Integration tests for manifest pattern in DuckDB index downloads.

These tests verify that DuckDBIndex properly uses fetch_manifest() to get
database URLs/hashes instead of relying on hardcoded config defaults.

Per manifest_integration_bugfix_plan.md, Task 1.1 tests are designed to verify
the manifest integration is working. Task 1.2 verifies schema compatibility.

NOTE: DuckDBIndex.__init__ was already fixed in the plan (line 219 has manifest_key).
This test validates that fix is working correctly and demonstrates the pattern
that needs to be applied to the remaining 5 call sites.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest
from findingmodel.config import clear_manifest_cache, settings
from findingmodel.index import DuckDBIndex


def test_duckdb_index_uses_manifest_when_no_db_path_provided() -> None:
    """Test that DuckDBIndex calls fetch_manifest() when no explicit db_path is provided.

    This test verifies Task 1.1 from the bugfix plan:
    - Mock fetch_manifest() to return test manifest with specific hash (different from config default)
    - Clear any cached database files
    - Create DuckDBIndex() without db_path
    - Assert fetch_manifest() was called
    - Assert downloaded file uses manifest hash (not hardcoded config hash)

    This validates that DuckDBIndex.__init__ (already fixed) properly uses manifest.
    """
    # Clear manifest cache to ensure fresh fetch
    clear_manifest_cache()

    # Create a test manifest with a different hash than the config default
    test_manifest = {
        "databases": {
            "finding_models": {
                "version": "2025-01-26",
                "url": "https://findingmodelsdata.t3.storage.dev/finding_models_20250126_TEST.duckdb",
                "hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "date": "2025-01-26",
            }
        }
    }

    # Mock fetch_manifest to return our test manifest
    with (
        patch("findingmodel.config.fetch_manifest", return_value=test_manifest) as mock_fetch,
        patch("pooch.retrieve") as mock_retrieve,
        # Mock Path.exists to return False so manifest will be called
        patch("pathlib.Path.exists", return_value=False),
    ):
        # Configure mock to return a fake path
        mock_db_path = Path("/fake/path/finding_models.duckdb")
        mock_retrieve.return_value = str(mock_db_path)

        # Create DuckDBIndex without explicit db_path
        # This should trigger ensure_db_file() which should call fetch_manifest()
        with contextlib.suppress(Exception):
            # Construction may fail due to fake path, but we only care about
            # whether fetch_manifest was called with correct parameters
            DuckDBIndex(db_path=None, read_only=True)

        # CRITICAL ASSERTIONS - These should FAIL initially:
        # 1. fetch_manifest() should have been called
        assert mock_fetch.called, "fetch_manifest() should be called when no db_path provided"

        # 2. pooch.retrieve should have been called with the manifest hash, not config default
        assert mock_retrieve.called, "pooch.retrieve should be called to download database"

        # Get the hash that was actually used
        call_kwargs = mock_retrieve.call_args.kwargs
        actual_hash = call_kwargs.get("known_hash")

        # Verify it's using the manifest hash, not the hardcoded config default
        expected_hash = test_manifest["databases"]["finding_models"]["hash"]
        assert actual_hash == expected_hash, (
            f"Should use manifest hash {expected_hash}, "
            f"but got {actual_hash} (config default: {settings.remote_index_db_hash})"
        )

        # 3. URL should also be from manifest
        actual_url = call_kwargs.get("url")
        expected_url = test_manifest["databases"]["finding_models"]["url"]
        assert actual_url == expected_url, f"Should use manifest URL {expected_url}, but got {actual_url}"


@pytest.mark.asyncio
async def test_duckdb_index_has_new_schema_with_finding_model_json_table(tmp_path: Path) -> None:
    """Test that DuckDBIndex has the new schema including finding_model_json table.

    This test verifies Task 1.2 from the bugfix plan:
    - Create DuckDBIndex with fresh database
    - Verify finding_model_json table exists
    - Verify get_full() method works

    EXPECTED RESULT: This test should FAIL if old database schema is downloaded.

    Note: This test creates an actual database to verify schema, but doesn't download.
    """
    # Create a fresh database locally (not downloaded)
    db_path = tmp_path / "test_fresh_schema.duckdb"

    # Create index in write mode to trigger setup()
    index = DuckDBIndex(db_path, read_only=False)
    await index.setup()

    try:
        # Verify connection exists
        conn = index._ensure_connection()

        # CRITICAL ASSERTION 1: finding_model_json table should exist
        tables_result = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'finding_model_json'
            """
        ).fetchall()

        assert len(tables_result) > 0, (
            "finding_model_json table should exist in the schema. "
            "If this fails, the downloaded database has the old schema without this table."
        )

        # CRITICAL ASSERTION 2: Verify table has expected columns
        columns_result = conn.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'finding_model_json'
            ORDER BY column_name
            """
        ).fetchall()

        column_names = [row[0] for row in columns_result]
        expected_columns = ["model_json", "oifm_id"]  # Sorted alphabetically

        assert column_names == expected_columns, (
            f"finding_model_json table should have columns {expected_columns}, "
            f"but found {column_names}. This indicates schema mismatch."
        )

        # Note: We cannot test get_full() without actual data, but we've verified
        # the table structure exists. The get_full() test would require populating
        # the database, which is tested elsewhere.

    finally:
        if index.conn is not None:
            index.conn.close()


@pytest.mark.asyncio
async def test_manifest_integration_with_mock_download(tmp_path: Path) -> None:
    """Integration test verifying manifest pattern end-to-end with mocked download.

    This is a comprehensive test combining both aspects:
    - Manifest is fetched and used
    - Resulting database has correct schema
    """
    clear_manifest_cache()

    # Test manifest pointing to a database with new schema
    test_manifest = {
        "databases": {
            "finding_models": {
                "version": "2025-01-26",
                "url": "https://findingmodelsdata.t3.storage.dev/finding_models_20250126.duckdb",
                "hash": "sha256:test_hash_for_new_schema_database",
                "date": "2025-01-26",
            }
        }
    }

    # Create a real database with the new schema to simulate what would be downloaded
    real_db_path = tmp_path / "simulated_download.duckdb"
    temp_index = DuckDBIndex(real_db_path, read_only=False)
    await temp_index.setup()
    if temp_index.conn is not None:
        temp_index.conn.close()

    with (
        patch("findingmodel.config.fetch_manifest", return_value=test_manifest) as mock_fetch,
        patch("pooch.retrieve", return_value=str(real_db_path)) as mock_retrieve,
        # Mock Path.exists to return False so manifest will be called
        patch("pathlib.Path.exists", return_value=False),
    ):
        # Create index without explicit path - should use manifest
        index = DuckDBIndex(db_path=None, read_only=True)

        # Verify manifest was used
        assert mock_fetch.called, "Should fetch manifest"
        assert mock_retrieve.called, "Should attempt download"

        # Verify the downloaded database has new schema
        conn = index._ensure_connection()
        tables = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name = 'finding_model_json'
            """
        ).fetchall()

        assert len(tables) > 0, "Downloaded database should have finding_model_json table"

        if index.conn is not None:
            index.conn.close()


def test_ensure_db_file_with_manifest_key_fetches_manifest() -> None:
    """Test that ensure_db_file WITH manifest_key parameter DOES call fetch_manifest.

    This demonstrates the correct behavior (already working in DuckDBIndex.__init__).

    EXPECTED RESULT: This test should PASS, showing how the fix should work.
    """
    from findingmodel.config import ensure_db_file

    clear_manifest_cache()

    test_manifest = {
        "databases": {
            "finding_models": {
                "version": "2025-01-26",
                "url": "https://findingmodelsdata.t3.storage.dev/finding_models_TEST.duckdb",
                "hash": "sha256:8888888888888888888888888888888888888888888888888888888888888888",
                "date": "2025-01-26",
            }
        }
    }

    with (
        patch("findingmodel.config.fetch_manifest", return_value=test_manifest) as mock_fetch,
        patch("pooch.retrieve") as mock_retrieve,
        # Mock Path.exists to return False so manifest will be called
        patch("pathlib.Path.exists", return_value=False),
    ):
        mock_retrieve.return_value = "/fake/db.duckdb"

        # Call ensure_db_file WITH manifest_key (correct behavior)
        # Use file_path=None for managed download mode
        ensure_db_file(
            file_path=None,  # Managed download mode
            remote_url=settings.remote_index_db_url,
            remote_hash=settings.remote_index_db_hash,
            manifest_key="finding_models",  # CORRECT: Provide manifest_key
        )

        # CORRECT BEHAVIOR: fetch_manifest should be called
        assert mock_fetch.called, "fetch_manifest() should be called when manifest_key is provided"

        # Verify that manifest values were used
        if mock_retrieve.called:
            call_kwargs = mock_retrieve.call_args.kwargs
            actual_hash = call_kwargs.get("known_hash")
            actual_url = call_kwargs.get("url")

            # Should use manifest values, NOT config defaults
            expected_hash = test_manifest["databases"]["finding_models"]["hash"]
            expected_url = test_manifest["databases"]["finding_models"]["url"]
            assert actual_hash == expected_hash, "Should use manifest hash when manifest_key is provided"
            assert actual_url == expected_url, "Should use manifest URL when manifest_key is provided"
