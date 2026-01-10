"""Manifest fetching and caching utilities."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Module-level cache for manifest (cleared on process restart)
_manifest_cache: dict[str, Any] | None = None


def fetch_manifest(manifest_url: str) -> dict[str, Any]:
    """Fetch and parse the remote manifest JSON with session caching.

    Args:
        manifest_url: URL to manifest JSON file

    Returns:
        Parsed manifest with database version info

    Raises:
        ValueError: If manifest URL not provided
        httpx.HTTPError: If fetch fails

    Example:
        manifest = fetch_manifest("https://example.com/manifest.json")
        db_info = manifest["databases"]["finding_models"]
        # {"version": "2025-01-24", "url": "...", "hash": "sha256:..."}
    """
    global _manifest_cache

    # Return cached manifest if available
    if _manifest_cache is not None:
        logger.debug("Using cached manifest")
        return _manifest_cache

    if not manifest_url:
        raise ValueError("Manifest URL not provided")

    logger.info(f"Fetching manifest from {manifest_url}")
    response = httpx.get(manifest_url, timeout=2.0)
    response.raise_for_status()

    manifest_data: dict[str, Any] = response.json()
    _manifest_cache = manifest_data
    logger.debug(f"Manifest cached with keys: {list(manifest_data.keys())}")
    return manifest_data


def clear_manifest_cache() -> None:
    """Clear the manifest cache (for testing)."""
    global _manifest_cache
    _manifest_cache = None
