"""Shared infrastructure for Open Imaging Data Model (OIDM) packages.

This package provides common models, utilities, and infrastructure used across
OIDM packages including findingmodel and anatomic-locations.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__ or __name__)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

from oidm_common.distribution import DistributionError, ensure_db_file, fetch_manifest
from oidm_common.embeddings import EmbeddingCache
from oidm_common.models import IndexCode, WebReference

__all__ = [
    "DistributionError",
    "EmbeddingCache",
    "IndexCode",
    "WebReference",
    "__version__",
    "ensure_db_file",
    "fetch_manifest",
]
