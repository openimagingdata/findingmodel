"""Integration tests for manifest pattern in DuckDB index downloads.

These tests verify that FindingModelIndex properly uses fetch_manifest() to get
database URLs/hashes instead of relying on hardcoded config defaults.

NOTE: Schema creation tests (finding_model_json table, etc.) are now in
oidm-maintenance/tests/test_findingmodel_build.py since FindingModelIndex is read-only.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import patch

from findingmodel.config import get_settings
from findingmodel.index import FindingModelIndex
from oidm_common.distribution import clear_manifest_cache


def test_duckdb_index_uses_manifest_when_no_db_path_provided() -> None:
    """Test that FindingModelIndex calls fetch_manifest() when no explicit db_path is provided.

    This test verifies Task 1.1 from the bugfix plan:
    - Mock fetch_manifest() to return test manifest with specific hash (different from config default)
    - Clear any cached database files
    - Create FindingModelIndex() without db_path
    - Assert fetch_manifest() was called
    - Assert downloaded file uses manifest hash (not hardcoded config hash)

    This validates that FindingModelIndex.__init__ (already fixed) properly uses manifest.
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
        patch("oidm_common.distribution.manifest.fetch_manifest", return_value=test_manifest) as mock_fetch,
        patch("pooch.retrieve") as mock_retrieve,
        # Mock Path.exists to return False so manifest will be called
        patch("pathlib.Path.exists", return_value=False),
    ):
        # Configure mock to return a fake path
        mock_db_path = Path("/fake/path/finding_models.duckdb")
        mock_retrieve.return_value = str(mock_db_path)

        # Create FindingModelIndex without explicit db_path
        # This should trigger ensure_db_file() which should call fetch_manifest()
        with contextlib.suppress(Exception):
            # Construction may fail due to fake path, but we only care about
            # whether fetch_manifest was called with correct parameters
            FindingModelIndex(db_path=None)

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
            f"but got {actual_hash} (config default: {get_settings().remote_db_hash})"
        )

        # 3. URL should also be from manifest
        actual_url = call_kwargs.get("url")
        expected_url = test_manifest["databases"]["finding_models"]["url"]
        assert actual_url == expected_url, f"Should use manifest URL {expected_url}, but got {actual_url}"


def test_ensure_db_file_with_manifest_key_fetches_manifest() -> None:
    """Test that ensure_db_file WITH manifest_key parameter DOES call fetch_manifest.

    This demonstrates the correct behavior (already working in FindingModelIndex.__init__).

    EXPECTED RESULT: This test should PASS, showing how the fix should work.
    """
    from oidm_common.distribution import ensure_db_file

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
        patch("oidm_common.distribution.manifest.fetch_manifest", return_value=test_manifest) as mock_fetch,
        patch("pooch.retrieve") as mock_retrieve,
        # Mock Path.exists to return False so manifest will be called
        patch("pathlib.Path.exists", return_value=False),
    ):
        mock_retrieve.return_value = "/fake/db.duckdb"

        # Call ensure_db_file WITH manifest_key (correct behavior)
        # Use file_path=None for managed download mode
        ensure_db_file(
            file_path=None,  # Managed download mode
            remote_url=get_settings().remote_db_url,
            remote_hash=get_settings().remote_db_hash,
            manifest_key="finding_models",  # CORRECT: Provide manifest_key
            manifest_url=get_settings().manifest_url,  # Required for managed downloads
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
