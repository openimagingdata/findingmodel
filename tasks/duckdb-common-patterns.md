# DuckDB Common Patterns - Code Consolidation Plan

## Goal
Extract shared DuckDB utilities to avoid code duplication between:
- `src/findingmodel/tools/duckdb_search.py` (anatomic location search)
- `src/findingmodel/duckdb_index.py` (new finding model index)

## Analysis: Common Patterns

### 1. Connection Management ✅ **EXTRACT**

**Current Pattern** (both implementations):
```python
async def __aenter__(self):
    if self.conn is None:
        self.conn = duckdb.connect(str(self.db_path), read_only=True)
        self.conn.execute("LOAD fts")
        self.conn.execute("LOAD vss")
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.conn:
        self.conn.close()
        self.conn = None
```

**Differences**:
- Anatomic search: `read_only=True` (doesn't modify data)
- Index: `read_only=False` + needs `SET hnsw_enable_experimental_persistence = true`

**Proposed Utility**:
```python
# src/findingmodel/tools/duckdb_utils.py

import duckdb
from pathlib import Path
from typing import Protocol

class DuckDBConnection(Protocol):
    """Protocol for classes that need DuckDB connection management."""
    db_path: Path
    conn: duckdb.DuckDBPyConnection | None
    
async def setup_duckdb_connection(
    db_path: Path | str,
    read_only: bool = True,
    enable_hnsw_persistence: bool = False
) -> duckdb.DuckDBPyConnection:
    """Setup DuckDB connection with extensions loaded.
    
    Args:
        db_path: Path to .duckdb file
        read_only: Whether to open in read-only mode
        enable_hnsw_persistence: Enable experimental HNSW persistence (for Index)
    
    Returns:
        Connected DuckDB connection with FTS and VSS loaded
    """
    conn = duckdb.connect(str(db_path), read_only=read_only)
    conn.execute("LOAD fts")
    conn.execute("LOAD vss")
    
    if enable_hnsw_persistence:
        conn.execute("SET hnsw_enable_experimental_persistence = true")
    
    return conn
```

**Usage**:
```python
# In DuckDBOntologySearchClient
async def __aenter__(self):
    if self.conn is None:
        self.conn = await setup_duckdb_connection(self.db_path, read_only=True)
    return self

# In DuckDBIndex
async def __aenter__(self):
    if self.conn is None:
        self.conn = await setup_duckdb_connection(
            self.db_path, 
            read_only=False, 
            enable_hnsw_persistence=True
        )
    return self
```

---

### 2. Embedding Generation ✅ **EXTRACT**

**Current Pattern** (anatomic search):
```python
async def _get_embedding(self, text: str) -> list[float] | None:
    return await get_embedding(text, client=self._openai_client, dimensions=512)
```

**Planned Pattern** (Index):
```python
async def _get_embedding(self, text: str) -> list[float] | None:
    embedding = await get_embedding(
        text, 
        client=self._openai_client,
        dimensions=settings.openai_embedding_dimensions
    )
    # Convert to float32 for DuckDB
    if embedding:
        import numpy as np
        return np.array(embedding, dtype=np.float32).tolist()
    return None
```

**Problem**: 
- Both need float64 → float32 conversion for DuckDB
- Both should use config settings
- Current anatomic search doesn't do float32 conversion (but should!)

**Proposed Utility**:
```python
# src/findingmodel/tools/duckdb_utils.py

import numpy as np
from openai import AsyncOpenAI
from findingmodel.config import settings
from findingmodel.tools.common import get_embedding

async def get_embedding_for_duckdb(
    text: str,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dimensions: int | None = None
) -> list[float] | None:
    """Get embedding optimized for DuckDB storage.
    
    Generates embedding and converts to FLOAT (32-bit) for DuckDB compatibility.
    Uses config settings by default.
    
    Args:
        text: Text to embed
        client: OpenAI client (optional, created if not provided)
        model: Embedding model (defaults to config setting)
        dimensions: Embedding dimensions (defaults to config setting)
    
    Returns:
        Float32 embedding vector or None if generation fails
    """
    model = model or settings.openai_embedding_model
    dimensions = dimensions or settings.openai_embedding_dimensions
    
    # Get embedding (float64)
    embedding = await get_embedding(text, client=client, dimensions=dimensions)
    
    # Convert to float32 for DuckDB FLOAT type
    if embedding:
        return np.array(embedding, dtype=np.float32).tolist()
    
    return None

async def batch_embeddings_for_duckdb(
    texts: list[str],
    client: AsyncOpenAI,
    model: str | None = None,
    dimensions: int | None = None
) -> list[list[float]]:
    """Generate embeddings for multiple texts in one API call.
    
    Optimized for batch operations - single OpenAI API call.
    Converts all embeddings to float32 for DuckDB.
    
    Args:
        texts: List of texts to embed
        client: OpenAI client
        model: Embedding model (defaults to config setting)
        dimensions: Embedding dimensions (defaults to config setting)
    
    Returns:
        List of float32 embedding vectors
    """
    model = model or settings.openai_embedding_model
    dimensions = dimensions or settings.openai_embedding_dimensions
    
    response = await client.embeddings.create(
        model=model,
        input=texts,
        dimensions=dimensions
    )
    
    # Convert all to float32
    return [
        np.array(e.embedding, dtype=np.float32).tolist()
        for e in response.data
    ]
```

**Usage**:
```python
# Simple single embedding
embedding = await get_embedding_for_duckdb(query)

# Batch embeddings (for search_batch)
embeddings = await batch_embeddings_for_duckdb(queries, self._openai_client)
```

---

### 3. Score Normalization ✅ **EXTRACT**

**Current Pattern** (anatomic search uses RRF, Index will use weighted fusion):

Both need min-max normalization for BM25 scores before fusion.

**Proposed Utility**:
```python
# src/findingmodel/tools/duckdb_utils.py

def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1] range.
    
    Args:
        scores: List of scores to normalize
    
    Returns:
        Normalized scores in [0, 1] range
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]

def weighted_fusion(
    results_a: list[tuple[str, float]],
    results_b: list[tuple[str, float]],
    weight_a: float = 0.3,
    weight_b: float = 0.7,
    normalize: bool = True
) -> list[tuple[str, float]]:
    """Combine two result sets using weighted score fusion.
    
    Args:
        results_a: List of (id, score) tuples from first search
        results_b: List of (id, score) tuples from second search
        weight_a: Weight for first result set (default 0.3 for BM25)
        weight_b: Weight for second result set (default 0.7 for semantic)
        normalize: Whether to normalize scores first (recommended)
    
    Returns:
        Combined results sorted by weighted score descending
    """
    # Build score dictionaries
    scores_a = dict(results_a)
    scores_b = dict(results_b)
    
    # Normalize if requested
    if normalize and scores_a:
        norm_a = normalize_scores(list(scores_a.values()))
        scores_a = dict(zip(scores_a.keys(), norm_a))
    
    if normalize and scores_b:
        norm_b = normalize_scores(list(scores_b.values()))
        scores_b = dict(zip(scores_b.keys(), norm_b))
    
    # Combine scores
    all_ids = set(scores_a.keys()) | set(scores_b.keys())
    combined = []
    
    for id in all_ids:
        score = (
            weight_a * scores_a.get(id, 0.0) +
            weight_b * scores_b.get(id, 0.0)
        )
        combined.append((id, score))
    
    # Sort by score descending
    return sorted(combined, key=lambda x: x[1], reverse=True)
```

---

### 4. RRF Fusion ✅ **EXTRACT** (Currently in anatomic search)

**Current Pattern** (anatomic search):
```python
def _apply_rrf_fusion(self, fts_results, vector_results, limit):
    # ... 30 lines of RRF logic ...
```

**Proposed Utility**:
```python
# src/findingmodel/tools/duckdb_utils.py

def rrf_fusion(
    results_a: list[tuple[str, float]],
    results_b: list[tuple[str, float]],
    k: int = 60,
    weight_a: float = 0.5,
    weight_b: float = 0.5
) -> list[tuple[str, float]]:
    """Combine two result sets using Reciprocal Rank Fusion.
    
    RRF score for item = weight_a / (k + rank_a) + weight_b / (k + rank_b)
    
    Args:
        results_a: List of (id, score) tuples from first search (order matters)
        results_b: List of (id, score) tuples from second search (order matters)
        k: RRF constant (default 60)
        weight_a: Weight for first result set (default 0.5)
        weight_b: Weight for second result set (default 0.5)
    
    Returns:
        Combined results sorted by RRF score descending
    """
    # Build rank dictionaries (1-indexed)
    ranks_a = {id: i + 1 for i, (id, _) in enumerate(results_a)}
    ranks_b = {id: i + 1 for i, (id, _) in enumerate(results_b)}
    
    # Combine using RRF
    all_ids = set(ranks_a.keys()) | set(ranks_b.keys())
    combined = []
    
    for id in all_ids:
        rank_a = ranks_a.get(id, len(results_a) + 1)
        rank_b = ranks_b.get(id, len(results_b) + 1)
        
        rrf_score = (
            weight_a / (k + rank_a) +
            weight_b / (k + rank_b)
        )
        combined.append((id, rrf_score))
    
    # Sort by RRF score descending
    return sorted(combined, key=lambda x: x[1], reverse=True)
```

---

### 5. L2 to Cosine Conversion ✅ **EXTRACT**

**Needed by**: Index (uses HNSW with L2 distance)
**Not needed by**: Anatomic search (uses `array_cosine_distance` directly)

**Proposed Utility**:
```python
# src/findingmodel/tools/duckdb_utils.py

def l2_to_cosine_similarity(l2_distance: float) -> float:
    """Convert L2 distance to cosine similarity.
    
    For normalized vectors: cosine_similarity ≈ 1 - (l2_distance / 2)
    
    Args:
        l2_distance: L2 (Euclidean) distance between vectors
    
    Returns:
        Approximate cosine similarity in [-1, 1] range
    """
    return 1.0 - (l2_distance / 2.0)
```

---

## What NOT to Extract ❌

### Schema-Specific Logic
- Exact match queries (different for name vs description)
- Table structures (finding_models vs anatomic_locations)
- Result object construction (IndexEntry vs OntologySearchResult)
- Filtering logic (tags vs region/sided)

### High-Level Search Patterns
- Too different between use cases
- Better to keep implementations focused and readable
- Common utilities for low-level operations only

---

## Implementation Plan

### Phase 1: Create Utility Module ✅
**File**: `src/findingmodel/tools/duckdb_utils.py`

**Contents**:
1. `setup_duckdb_connection()` - connection with extensions
2. `get_embedding_for_duckdb()` - single embedding with float32 conversion
3. `batch_embeddings_for_duckdb()` - batch embeddings for efficiency
4. `normalize_scores()` - min-max normalization
5. `weighted_fusion()` - weighted score combination
6. `rrf_fusion()` - reciprocal rank fusion
7. `l2_to_cosine_similarity()` - distance metric conversion

**Dependencies**:
- numpy (for float32 conversion)
- duckdb
- openai
- findingmodel.config (settings)
- findingmodel.tools.common (get_embedding)

### Phase 2: Update Anatomic Search ⚙️
**File**: `src/findingmodel/tools/duckdb_search.py`

**Changes**:
1. Use `setup_duckdb_connection()` in `__aenter__`
2. Replace `_get_embedding()` with `get_embedding_for_duckdb()`
3. Replace `_apply_rrf_fusion()` with `rrf_fusion()` utility
4. Fix hardcoded dimensions to use `settings.openai_embedding_dimensions`

**Impact**: Minimal - internal refactoring only, same API

### Phase 3: Use in New Index Implementation ⚙️
**File**: `src/findingmodel/duckdb_index.py`

**Usage**:
1. Use `setup_duckdb_connection()` with `enable_hnsw_persistence=True`
2. Use `batch_embeddings_for_duckdb()` in `search_batch()`
3. Use `weighted_fusion()` for hybrid search
4. Use `l2_to_cosine_similarity()` for HNSW results
5. Use `normalize_scores()` for BM25 before fusion

**Benefit**: Shared, tested utilities from day 1

### Phase 4: Testing ✅
**File**: `test/test_duckdb_utils.py`

**Test Coverage**:
- Connection setup with different flags
- Embedding generation and float32 conversion
- Batch embedding generation
- Score normalization edge cases (empty, single value, all same)
- Weighted fusion with various weights
- RRF fusion correctness
- L2 to cosine conversion accuracy

---

## Benefits

✅ **Code Reuse**: ~200 lines of utility code shared across 2+ implementations  
✅ **Consistency**: Same embedding handling, same fusion algorithms  
✅ **Testability**: Utilities tested independently  
✅ **Maintainability**: Fix bugs once, benefit everywhere  
✅ **Flexibility**: Easy to add new DuckDB-based features  
✅ **Configuration**: All use same config settings  

## Risks & Mitigation

⚠️ **Risk**: Over-abstraction makes code harder to understand  
✅ **Mitigation**: Keep utilities low-level and focused, document well

⚠️ **Risk**: Breaking anatomic search during refactor  
✅ **Mitigation**: Write tests first, refactor incrementally, keep same API

⚠️ **Risk**: Numpy dependency just for float32 conversion  
✅ **Mitigation**: Numpy already used elsewhere, conversion is critical for correctness

---

## Timeline

1. **During Index Implementation**: Create `duckdb_utils.py` as needed
2. **After Index Working**: Refactor anatomic search to use utilities
3. **Continuous**: Add more utilities as patterns emerge

---

## Success Criteria

- [ ] `duckdb_utils.py` module created with 7 core utilities
- [ ] All utilities have docstrings and type hints
- [ ] Test coverage > 90% for utilities
- [ ] Anatomic search refactored to use utilities (same API)
- [ ] Index implementation uses utilities from start
- [ ] Both implementations use `settings.openai_embedding_dimensions`
- [ ] Both implementations convert embeddings to float32
- [ ] No code duplication for connection, embedding, fusion logic

---

## Related Files

- Plan: `tasks/index-duckdb-migration.md`
- Existing: `src/findingmodel/tools/duckdb_search.py`
- New: `src/findingmodel/duckdb_index.py`
- Utilities: `src/findingmodel/tools/duckdb_utils.py` (to create)
- Tests: `test/test_duckdb_utils.py` (to create)
- Config: `src/findingmodel/config.py`
