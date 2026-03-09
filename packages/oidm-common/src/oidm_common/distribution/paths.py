"""Path resolution utilities for database files."""

import logging
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from oidm_common.distribution.profiles import (
    OPENAI_PROFILE,
    build_profile_descriptor,
    build_profile_manifest_key,
    read_embedding_profile_from_db,
    resolve_profile_name,
)

logger = logging.getLogger(__name__)
_OPENAI_BASE_COMPAT_DESCRIPTOR = build_profile_descriptor(
    OPENAI_PROFILE.provider, OPENAI_PROFILE.model, OPENAI_PROFILE.dimensions
)


class DistributionError(RuntimeError):
    """Error during database distribution operations."""

    pass


def _validate_databases(manifest: dict[str, Any]) -> dict[str, Any]:
    databases = manifest.get("databases")
    if not isinstance(databases, dict):
        raise DistributionError("Manifest is missing 'databases' object")
    return databases


def _get_base_entry(databases: dict[str, Any], base_key: str) -> tuple[str, dict[str, Any]]:
    if base_key not in databases:
        raise DistributionError(f"Manifest key not found: {base_key}")
    db_info = databases[base_key]
    if not isinstance(db_info, dict):
        raise DistributionError(f"Manifest entry for '{base_key}' is invalid")
    return base_key, db_info


def _entry_from_aliases(
    manifest: dict[str, Any],
    databases: dict[str, Any],
    base_key: str,
    descriptor: str,
) -> tuple[str, dict[str, Any]] | None:
    profile_aliases = manifest.get("profile_aliases", {})
    if not isinstance(profile_aliases, dict):
        return None
    aliases_for_base = profile_aliases.get(base_key, {})
    if not isinstance(aliases_for_base, dict):
        return None
    aliased_key = aliases_for_base.get(descriptor)
    if not isinstance(aliased_key, str):
        return None
    aliased_entry = databases.get(aliased_key)
    if not isinstance(aliased_entry, dict):
        return None
    return aliased_key, aliased_entry


def _entry_from_generated_key(
    databases: dict[str, Any],
    generated_key: str,
) -> tuple[str, dict[str, Any]] | None:
    generated_entry = databases.get(generated_key)
    if not isinstance(generated_entry, dict):
        return None
    return generated_key, generated_entry


def _entry_from_metadata_scan(
    databases: dict[str, Any],
    base_key: str,
    descriptor: str,
) -> tuple[str, dict[str, Any]] | None:
    for key, value in databases.items():
        if not isinstance(key, str) or not key.startswith(f"{base_key}__"):
            continue
        if not isinstance(value, dict):
            continue
        provider = value.get("embedding_provider")
        model = value.get("embedding_model")
        dims = value.get("embedding_dimensions")
        if not (isinstance(provider, str) and isinstance(model, str) and isinstance(dims, int)):
            continue
        if build_profile_descriptor(provider, model, dims) == descriptor:
            return key, value
    return None


def _find_existing_local_candidate(
    *,
    error: Exception,
    target: Path,
    base_target: Path,
    profile_target: Path | None,
    allow_base_target: bool = True,
) -> Path | None:
    """Return the best existing local fallback file, if any."""
    candidates: list[Path] = []
    if profile_target is not None:
        candidates.append(profile_target)
    if allow_base_target:
        candidates.extend([target, base_target])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            logger.warning(
                "Cannot fetch manifest (%s), but using existing local file: %s. Database may be outdated.",
                error,
                candidate,
            )
            return candidate
    return None


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


def _select_manifest_entry(
    manifest: dict[str, Any],
    base_key: str,
    *,
    embedding_provider: str | None,
    embedding_model: str | None,
    embedding_dimensions: int | None,
) -> tuple[str, dict[str, Any]]:
    """Resolve manifest entry for base key with optional embedding profile."""
    databases = _validate_databases(manifest)

    if embedding_provider is None or embedding_model is None or embedding_dimensions is None:
        return _get_base_entry(databases, base_key)

    descriptor = build_profile_descriptor(embedding_provider, embedding_model, embedding_dimensions)

    # 1) Explicit alias map
    by_alias = _entry_from_aliases(manifest, databases, base_key, descriptor)
    if by_alias is not None:
        return by_alias

    # 2) Deterministic generated key
    generated_key = build_profile_manifest_key(base_key, embedding_provider, embedding_model, embedding_dimensions)
    by_generated_key = _entry_from_generated_key(databases, generated_key)
    if by_generated_key is not None:
        return by_generated_key

    # 3) Scan profile entries with explicit metadata
    by_metadata = _entry_from_metadata_scan(databases, base_key, descriptor)
    if by_metadata is not None:
        return by_metadata

    # 4) Base-key fallback only when it is known to match the requested profile
    default_entry = databases.get(base_key)
    if isinstance(default_entry, dict):
        base_provider = default_entry.get("embedding_provider")
        base_model = default_entry.get("embedding_model")
        base_dims = default_entry.get("embedding_dimensions")

        if isinstance(base_provider, str) and isinstance(base_model, str) and isinstance(base_dims, int):
            base_descriptor = build_profile_descriptor(base_provider, base_model, base_dims)
            if base_descriptor == descriptor:
                logger.warning(
                    "No profile-specific manifest entry found for %s (%s); using base key '%s' with matching metadata",
                    base_key,
                    descriptor,
                    base_key,
                )
                return base_key, default_entry
            raise DistributionError(
                f"Manifest base entry for '{base_key}' uses '{base_descriptor}', "
                f"which does not match requested profile '{descriptor}'."
            )

        if descriptor == _OPENAI_BASE_COMPAT_DESCRIPTOR:
            logger.warning(
                "No profile-specific manifest entry found for %s (%s); using base key '%s' compatibility entry",
                base_key,
                descriptor,
                base_key,
            )
            return base_key, default_entry

        raise DistributionError(
            f"No manifest entry found for requested profile '{descriptor}' under '{base_key}'. "
            "Base entry is missing embedding metadata and cannot be safely matched."
        )

    raise DistributionError(f"Manifest key not found: {base_key} (including profile variants)")


def ensure_db_file(  # noqa: C901
    file_path: str | Path | None,
    remote_url: str | None,
    remote_hash: str | None,
    manifest_key: str,
    manifest_url: str | None = None,
    app_name: str = "oidm",
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
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

    # Validate URL/hash pair: must provide both or neither
    if (remote_url is not None) != (remote_hash is not None):
        raise DistributionError(
            "Must provide both remote_url and remote_hash, or neither. "
            f"Got url={'set' if remote_url else 'unset'}, "
            f"hash={'set' if remote_hash else 'unset'}"
        )

    # Default target path (may be overridden by profile-specific manifest key below)
    target = _resolve_target_path(None, manifest_key, app_name)
    base_target = target
    profile_target: Path | None = None
    requested_descriptor: str | None = None
    allow_base_fallback_candidate = True
    if embedding_provider is not None and embedding_model is not None and embedding_dimensions is not None:
        requested_descriptor = build_profile_descriptor(embedding_provider, embedding_model, embedding_dimensions)
        profile_key = build_profile_manifest_key(
            manifest_key, embedding_provider, embedding_model, embedding_dimensions
        )
        profile_target = _resolve_target_path(None, profile_key, app_name)
        allow_base_fallback_candidate = (
            resolve_profile_name(embedding_provider, embedding_model, embedding_dimensions) == OPENAI_PROFILE.name
        )

    # Get URL and hash (from explicit config or manifest)
    if remote_url is not None and remote_hash is not None:
        url, hash_value = remote_url, remote_hash
        # Keep explicit downloads profile-separated when caller provides profile tuple.
        if profile_target is not None:
            target = profile_target
        logger.debug(f"Using explicit remote URL: {url}")
    else:
        # Fetch from manifest with graceful fallback
        try:
            if not manifest_url:
                raise DistributionError("Manifest URL required for managed downloads")
            manifest = fetch_manifest(manifest_url)
            resolved_key, db_info = _select_manifest_entry(
                manifest,
                manifest_key,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimensions=embedding_dimensions,
            )
            url = db_info["url"]
            hash_value = db_info["hash"]
            version = db_info.get("version", "unknown")
            # Store each profile under its own file name so profiles can coexist locally.
            target = _resolve_target_path(None, resolved_key, app_name)
            logger.debug(f"Using manifest version {version} ({resolved_key}): {url}")
        except Exception as e:
            fallback = _find_existing_local_candidate(
                error=e,
                target=target,
                base_target=base_target,
                profile_target=profile_target,
                allow_base_target=allow_base_fallback_candidate,
            )
            if fallback is not None:
                return fallback

            # No local file and can't fetch manifest - fail
            missing_path = profile_target if profile_target is not None else base_target
            profile_hint = f" for profile '{requested_descriptor}'" if requested_descriptor is not None else ""
            raise DistributionError(
                f"Cannot fetch manifest for {manifest_key}: {e}. "
                f"No local database file exists at {missing_path}{profile_hint}. "
                f"Either fix network connectivity or set explicit file path and remote URL/hash."
            ) from e

    # Pooch handles: exists check, hash verification, re-download if needed
    downloaded = download_file(target, url, hash_value)

    # For explicit URL/hash mode, detect embedded profile metadata and normalize the local cache path
    # so openai/local artifacts can coexist even if caller passed a mismatched profile tuple.
    if remote_url is not None and remote_hash is not None:
        detected_profile = read_embedding_profile_from_db(downloaded)
        if detected_profile is not None:
            detected_provider, detected_model, detected_dims = detected_profile
            detected_key = build_profile_manifest_key(
                manifest_key,
                detected_provider,
                detected_model,
                detected_dims,
            )
            detected_target = _resolve_target_path(None, detected_key, app_name)
            if detected_target != downloaded:
                detected_target.parent.mkdir(parents=True, exist_ok=True)
                downloaded.replace(detected_target)
                logger.debug("Re-homed downloaded DB to profile-specific path: %s", detected_target)
                downloaded = detected_target
            if requested_descriptor is not None:
                detected_descriptor = build_profile_descriptor(detected_provider, detected_model, detected_dims)
                if detected_descriptor != requested_descriptor:
                    logger.warning(
                        "Explicit DB profile '%s' differs from requested '%s'; using downloaded DB profile metadata.",
                        detected_descriptor,
                        requested_descriptor,
                    )

    return downloaded
