"""Package distribution utilities for OIDM packages.

This module provides utilities for downloading and managing database files:
- Manifest fetching and caching
- File downloads with hash verification
- Path resolution and database file management
"""

from oidm_common.distribution.download import download_file
from oidm_common.distribution.manifest import clear_manifest_cache, fetch_manifest
from oidm_common.distribution.paths import DistributionError, ensure_db_file
from oidm_common.distribution.profiles import (
    LOCAL_PROFILE,
    OPENAI_PROFILE,
    SUPPORTED_EMBEDDING_PROFILES,
    build_named_profile_manifest_key,
    build_profile_descriptor,
    build_profile_manifest_key,
    read_embedding_profile_from_db,
    resolve_profile_name,
    slugify_profile_part,
)

__all__ = [
    "LOCAL_PROFILE",
    "OPENAI_PROFILE",
    "SUPPORTED_EMBEDDING_PROFILES",
    "DistributionError",
    "build_named_profile_manifest_key",
    "build_profile_descriptor",
    "build_profile_manifest_key",
    "clear_manifest_cache",
    "download_file",
    "ensure_db_file",
    "fetch_manifest",
    "read_embedding_profile_from_db",
    "resolve_profile_name",
    "slugify_profile_part",
]
