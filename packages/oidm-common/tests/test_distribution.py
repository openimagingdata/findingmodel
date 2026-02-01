"""Tests for oidm_common.distribution.manifest caching behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from oidm_common.distribution.manifest import clear_manifest_cache, fetch_manifest

MANIFEST_URL_A = "https://findingmodelsdata.t3.storage.dev/manifest.json"
MANIFEST_URL_B = "https://anatomiclocationsdata.t3.storage.dev/manifest.json"

MANIFEST_A: dict[str, Any] = {
    "databases": {
        "finding_models": {"version": "2025-01-24", "url": "https://a.example.com/fm.duckdb", "hash": "sha256:aaa"},
        "anatomic_locations": {"version": "2024-06-01", "url": "https://a.example.com/al.duckdb", "hash": "sha256:stale"},
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
