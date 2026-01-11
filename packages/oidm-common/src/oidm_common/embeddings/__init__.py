"""Embedding generation utilities for OIDM packages.

This module provides shared embedding cache functionality across OIDM packages.
"""

from oidm_common.embeddings.cache import EmbeddingCache
from oidm_common.embeddings.generation import generate_embedding, generate_embeddings_batch

__all__ = ["EmbeddingCache", "generate_embedding", "generate_embeddings_batch"]
