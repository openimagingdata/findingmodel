"""DuckDB utilities and helpers for OIDM packages."""

from oidm_common.duckdb.connection import setup_duckdb_connection
from oidm_common.duckdb.indexes import create_fts_index, create_hnsw_index, drop_search_indexes
from oidm_common.duckdb.search import (
    ScoreTuple,
    l2_to_cosine_similarity,
    normalize_scores,
    rrf_fusion,
    weighted_fusion,
)

__all__ = [
    # Search utilities
    "ScoreTuple",
    # Index management
    "create_fts_index",
    "create_hnsw_index",
    "drop_search_indexes",
    "l2_to_cosine_similarity",
    "normalize_scores",
    "rrf_fusion",
    # Connection utilities
    "setup_duckdb_connection",
    "weighted_fusion",
]
