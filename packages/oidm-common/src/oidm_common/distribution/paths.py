"""Path resolution utilities for database files."""

import logging
from pathlib import Path

from platformdirs import user_data_dir

logger = logging.getLogger(__name__)


class DistributionError(RuntimeError):
    """Error during database distribution operations."""

    pass


def _resolve_target_path(file_path: str | Path | None, manifest_key: str, app_name: str = "oidm") -> Path:
    """Resolve database file path to absolute Path.

    Args:
        file_path: User-specified path (absolute, relative, or None)
        manifest_key: Key in manifest for default filename (e.g., 'finding_models')
        app_name: Application name for user data directory (default: "oidm")

    Returns:
        Resolved absolute Path
    """
    data_dir = Path(user_data_dir(appname=app_name, appauthor="openimagingdata", ensure_exists=True))

    if file_path is None:
        # Default: {manifest_key}.duckdb in user data dir
        return data_dir / f"{manifest_key}.duckdb"

    path = Path(file_path)
    if path.is_absolute():
        return path
    else:
        # Relative path: resolve to user_data_dir
        return data_dir / path


def ensure_db_file(
    file_path: str | Path | None,
    remote_url: str | None,
    remote_hash: str | None,
    manifest_key: str,
    manifest_url: str | None = None,
    app_name: str = "oidm",
) -> Path:
    """Ensure database file is available.

    Two modes:
        1. Explicit path: User specified exact file to use (via file_path parameter)
           - Validates file exists, returns path
           - No downloads or hash verification (user's responsibility)

        2. Managed download: Use Pooch for automatic download/caching/updates
           - Gets URL/hash from explicit config or manifest
           - Pooch handles: download if missing, hash verification, re-download if hash mismatch
           - Automatic updates when manifest changes

    Args:
        file_path: Database file path (absolute, relative to user data dir, or None for default)
            - If None: uses managed download with automatic updates
            - If set: uses explicit path, no automatic updates
        remote_url: Optional explicit download URL (must provide both URL and hash, or neither)
        remote_hash: Optional explicit hash for verification (e.g., 'sha256:abc...')
        manifest_key: Key in manifest JSON databases section (e.g., 'finding_models', 'anatomic_locations')
        manifest_url: URL to manifest JSON (required if using manifest mode)
        app_name: Application name for user data directory (default: "oidm")

    Returns:
        Path to the database file

    Raises:
        DistributionError: If explicit file doesn't exist or download fails

    Examples:
        # Explicit path (Docker/production): use pre-mounted file
        db_path = ensure_db_file("/mnt/data/finding_models.duckdb", None, None, "finding_models")

        # Managed download with automatic updates (default behavior)
        db_path = ensure_db_file(None, None, None, "finding_models",
                                 manifest_url="https://example.com/manifest.json")

        # Explicit remote URL/hash (overrides manifest)
        db_path = ensure_db_file(None, "https://example.com/db.duckdb", "sha256:abc123...",
                                 "finding_models")
    """
    from oidm_common.distribution.download import download_file
    from oidm_common.distribution.manifest import fetch_manifest

    # Case 1: User specified explicit path - validate and return
    if file_path is not None:
        target = _resolve_target_path(file_path, manifest_key, app_name)
        if not target.exists():
            raise DistributionError(
                f"Explicit database file not found: {target}. "
                f"Either provide the file or unset the path to enable automatic downloads."
            )
        logger.debug(f"Using explicit database file: {target}")
        return target

    # Case 2: Managed download - let Pooch handle download/caching/updates
    target = _resolve_target_path(None, manifest_key, app_name)

    # Get URL and hash (from explicit config or manifest)
    if remote_url is not None and remote_hash is not None:
        url, hash_value = remote_url, remote_hash
        logger.debug(f"Using explicit remote URL: {url}")
    else:
        # Fetch from manifest with graceful fallback
        try:
            if not manifest_url:
                raise DistributionError("Manifest URL required for managed downloads")
            manifest = fetch_manifest(manifest_url)
            db_info = manifest["databases"][manifest_key]
            url = db_info["url"]
            hash_value = db_info["hash"]
            version = db_info.get("version", "unknown")
            logger.debug(f"Using manifest version {version}: {url}")
        except Exception as e:
            # If manifest fetch fails but we have a local file, use it
            if target.exists():
                logger.warning(
                    f"Cannot fetch manifest ({e}), but using existing local file: {target}. Database may be outdated."
                )
                return target
            else:
                # No local file and can't fetch manifest - fail
                raise DistributionError(
                    f"Cannot fetch manifest for {manifest_key}: {e}. "
                    f"No local database file exists at {target}. "
                    f"Either fix network connectivity or set explicit file path and remote URL/hash."
                ) from e

    # Pooch handles: exists check, hash verification, re-download if needed
    return download_file(target, url, hash_value)
