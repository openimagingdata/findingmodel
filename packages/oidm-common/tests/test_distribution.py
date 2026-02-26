"""Tests for oidm_common.distribution.manifest caching behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from oidm_common.distribution import DistributionError, ensure_db_file
from oidm_common.distribution.manifest import clear_manifest_cache, fetch_manifest

MANIFEST_URL_A = "https://findingmodelsdata.t3.storage.dev/manifest.json"
MANIFEST_URL_B = "https://anatomiclocationsdata.t3.storage.dev/manifest.json"

MANIFEST_A: dict[str, Any] = {
    "databases": {
        "finding_models": {"version": "2025-01-24", "url": "https://a.example.com/fm.duckdb", "hash": "sha256:aaa"},
        "anatomic_locations": {
            "version": "2024-06-01",
            "url": "https://a.example.com/al.duckdb",
            "hash": "sha256:stale",
        },
    }
}

MANIFEST_B: dict[str, Any] = {
    "databases": {
        "anatomic_locations": {"version": "2025-01-20", "url": "https://b.example.com/al.duckdb", "hash": "sha256:bbb"},
    }
}


def _mock_response(data: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Ensure a clean manifest cache for every test."""
    clear_manifest_cache()


class TestFetchManifestCaching:
    """Verify that manifests are cached per-URL, not globally."""

    @patch("oidm_common.distribution.manifest.httpx.get")
    def test_caches_by_url(self, mock_get: MagicMock) -> None:
        """Fetching from URL A then URL B should make two HTTP calls."""
        mock_get.side_effect = [_mock_response(MANIFEST_A), _mock_response(MANIFEST_B)]

        fetch_manifest(MANIFEST_URL_A)
        fetch_manifest(MANIFEST_URL_B)

        assert mock_get.call_count == 2

    @patch("oidm_common.distribution.manifest.httpx.get")
    def test_returns_cached_for_same_url(self, mock_get: MagicMock) -> None:
        """Fetching the same URL twice should make only one HTTP call."""
        mock_get.return_value = _mock_response(MANIFEST_A)

        result1 = fetch_manifest(MANIFEST_URL_A)
        result2 = fetch_manifest(MANIFEST_URL_A)

        assert mock_get.call_count == 1
        assert result1 is result2

    @patch("oidm_common.distribution.manifest.httpx.get")
    def test_different_urls_return_different_data(self, mock_get: MagicMock) -> None:
        """Each URL must return its own manifest data."""
        mock_get.side_effect = [_mock_response(MANIFEST_A), _mock_response(MANIFEST_B)]

        result_a = fetch_manifest(MANIFEST_URL_A)
        result_b = fetch_manifest(MANIFEST_URL_B)

        assert result_a == MANIFEST_A
        assert result_b == MANIFEST_B
        assert result_a is not result_b

    @patch("oidm_common.distribution.manifest.httpx.get")
    def test_clear_manifest_cache_clears_all(self, mock_get: MagicMock) -> None:
        """After clearing, both URLs should be re-fetched."""
        mock_get.side_effect = [
            _mock_response(MANIFEST_A),
            _mock_response(MANIFEST_B),
            _mock_response(MANIFEST_A),
            _mock_response(MANIFEST_B),
        ]

        fetch_manifest(MANIFEST_URL_A)
        fetch_manifest(MANIFEST_URL_B)
        assert mock_get.call_count == 2

        clear_manifest_cache()

        fetch_manifest(MANIFEST_URL_A)
        fetch_manifest(MANIFEST_URL_B)
        assert mock_get.call_count == 4

    @patch("oidm_common.distribution.manifest.httpx.get")
    def test_multi_package_manifest_isolation(self, mock_get: MagicMock) -> None:
        """Reproduce the actual bug: two packages with different manifest URLs.

        findingmodel fetches first with a manifest containing a stale
        anatomic_locations entry. anatomic-locations fetches second with its
        own manifest containing the correct entry. Each must get its own data.
        """
        mock_get.side_effect = [_mock_response(MANIFEST_A), _mock_response(MANIFEST_B)]

        # Simulate findingmodel fetching its manifest first
        fm_manifest = fetch_manifest(MANIFEST_URL_A)
        # Simulate anatomic-locations fetching its own manifest second
        al_manifest = fetch_manifest(MANIFEST_URL_B)

        # findingmodel manifest should have the stale anatomic_locations entry
        assert fm_manifest["databases"]["anatomic_locations"]["hash"] == "sha256:stale"

        # anatomic-locations manifest must have the correct entry, not the stale one
        assert al_manifest["databases"]["anatomic_locations"]["hash"] == "sha256:bbb"
        assert al_manifest["databases"]["anatomic_locations"]["version"] == "2025-01-20"


class TestEnsureDbFileValidation:
    """Tests for ensure_db_file URL/hash pair validation."""

    @patch("oidm_common.distribution.paths._resolve_target_path")
    def test_raises_error_when_url_without_hash(self, mock_resolve: MagicMock) -> None:
        """Test that providing URL without hash raises DistributionError."""
        mock_resolve.return_value = MagicMock()

        with pytest.raises(DistributionError, match="Must provide both remote_url and remote_hash"):
            ensure_db_file(
                file_path=None,
                remote_url="https://example.com/db.duckdb",
                remote_hash=None,
                manifest_key="finding_models",
                manifest_url="https://example.com/manifest.json",
            )

    @patch("oidm_common.distribution.paths._resolve_target_path")
    def test_raises_error_when_hash_without_url(self, mock_resolve: MagicMock) -> None:
        """Test that providing hash without URL raises DistributionError."""
        mock_resolve.return_value = MagicMock()

        with pytest.raises(DistributionError, match="Must provide both remote_url and remote_hash"):
            ensure_db_file(
                file_path=None,
                remote_url=None,
                remote_hash="sha256:abc123",
                manifest_key="finding_models",
                manifest_url="https://example.com/manifest.json",
            )

    @patch("oidm_common.distribution.paths._resolve_target_path")
    @patch("oidm_common.distribution.manifest.fetch_manifest")
    @patch("oidm_common.distribution.download.download_file")
    def test_succeeds_when_both_url_and_hash_provided(
        self, mock_download: MagicMock, mock_fetch: MagicMock, mock_resolve: MagicMock
    ) -> None:
        """Test that providing both URL and hash succeeds."""
        target_path = MagicMock()
        mock_resolve.return_value = target_path
        mock_download.return_value = target_path

        result = ensure_db_file(
            file_path=None,
            remote_url="https://example.com/db.duckdb",
            remote_hash="sha256:abc123",
            manifest_key="finding_models",
            manifest_url="https://example.com/manifest.json",
        )

        assert result == target_path
        mock_download.assert_called_once_with(target_path, "https://example.com/db.duckdb", "sha256:abc123")

    @patch("oidm_common.distribution.paths._resolve_target_path")
    @patch("oidm_common.distribution.manifest.fetch_manifest")
    def test_succeeds_when_neither_url_nor_hash_provided(self, mock_fetch: MagicMock, mock_resolve: MagicMock) -> None:
        """Test that providing neither URL nor hash succeeds (uses manifest)."""
        target_path = MagicMock()
        target_path.exists.return_value = True
        mock_resolve.return_value = target_path
        mock_fetch.return_value = {
            "databases": {
                "finding_models": {
                    "url": "https://example.com/db.duckdb",
                    "hash": "sha256:abc123",
                    "version": "2025-01-01",
                }
            }
        }

        with patch("oidm_common.distribution.download.download_file") as mock_download:
            mock_download.return_value = target_path

            result = ensure_db_file(
                file_path=None,
                remote_url=None,
                remote_hash=None,
                manifest_key="finding_models",
                manifest_url="https://example.com/manifest.json",
            )

            assert result == target_path
