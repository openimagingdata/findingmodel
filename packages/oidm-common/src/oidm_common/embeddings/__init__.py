"""Embedding generation utilities for OIDM packages.

This module provides shared embedding cache functionality across OIDM packages.
"""

from oidm_common.embeddings.cache import EmbeddingCache
from oidm_common.embeddings.config import ACTIVE_EMBEDDING_CONFIG, EmbeddingProfileSpec, read_embedding_profile_from_db
from oidm_common.embeddings.generation import (
    # Low-level API (advanced use)
    create_openai_client,
    generate_embedding,
    generate_embeddings_batch,
    # High-level API (preferred)
    get_embedding,
    get_embeddings_batch,
)

__all__ = [
    "ACTIVE_EMBEDDING_CONFIG",
    "EmbeddingCache",
    "EmbeddingProfileSpec",
    "create_openai_client",
    "generate_embedding",
    "generate_embeddings_batch",
    "get_embedding",
    "get_embeddings_batch",
    "read_embedding_profile_from_db",
]
