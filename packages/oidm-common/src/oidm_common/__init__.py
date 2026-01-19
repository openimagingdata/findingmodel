"""Shared infrastructure for Open Imaging Data Model (OIDM) packages.

This package provides common models, utilities, and infrastructure used across
OIDM packages including findingmodel and anatomic-locations.
"""

from oidm_common.distribution import DistributionError, ensure_db_file, fetch_manifest
from oidm_common.embeddings import EmbeddingCache
from oidm_common.models import IndexCode, WebReference
from oidm_common.utils import strip_quotes, strip_quotes_secret

__version__ = "0.1.0"

__all__ = [
    "DistributionError",
    "EmbeddingCache",
    "IndexCode",
    "WebReference",
    "__version__",
    "ensure_db_file",
    "fetch_manifest",
    "strip_quotes",
    "strip_quotes_secret",
]
