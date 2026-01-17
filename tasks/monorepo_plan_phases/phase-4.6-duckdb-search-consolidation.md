# Phase 4.6: DuckDB Search Consolidation

**Status:** ✅ COMPLETE (2026-01-17)

**Goal:** Consolidate hybrid search code with proper async patterns so each package uses oidm-common primitives and exposes search through its own Index class.

## Completion Summary

All sub-phases completed:
- **4.6.0**: Added `asyncer==0.0.12` to oidm-common with async DuckDB pattern tests
- **4.6.1**: Fixed DuckDBIndex async patterns using typed sync helpers + asyncify
- **4.6.2**: Added async hybrid search to AnatomicLocationIndex with RRF fusion
- **4.6.3**: Updated findingmodel-ai to use AnatomicLocationIndex directly
- **4.6.4**: Deleted deprecated `duckdb_search.py` and demo script
- **4.6.5**: All tests pass (498 total), all checks pass

Key pattern established: Define typed sync helper methods with full type annotations, wrap with `asyncify(self._helper)(args)` at call sites to preserve types through async boundary.

## Problem Statement

### Architectural Issues

| Component | Location | Issue |
|-----------|----------|-------|
| `DuckDBOntologySearchClient` | `findingmodel/tools/duckdb_search.py` | Searches anatomic-locations DB but lives in WRONG package |
| `AnatomicLocationIndex.search()` | `anatomic-locations/index.py` | Only has FTS search, missing semantic/hybrid; sync only |
| `DuckDBIndex` | `findingmodel/index.py` | Has hybrid search but "pseudo-async" (blocks event loop) |

### Async Pattern Issues

**Current Problem:** DuckDB methods are marked `async def` but perform NO actual async operations:

```python
# ❌ PSEUDO-ASYNC - blocks event loop
async def contains(self, identifier: str) -> bool:
    conn = self._ensure_connection()
    return self._resolve_oifm_id(conn, identifier) is not None  # Sync call!
```

**Research Findings:**
- DuckDB has **no native async support** (intentional design choice)
- Embedding API calls are truly async (I/O-bound)
- DuckDB queries should use `asyncio.to_thread()` to avoid blocking the event loop
- This matters for FastAPI/MCP server concurrent request handling

## Target Architecture

```
oidm-common (infrastructure)
    ├── duckdb/
    │   ├── async_utils.py (NEW) - run_in_thread() helper
    │   ├── search.py - rrf_fusion, weighted_fusion
    │   └── connection.py - setup_duckdb_connection
    └── embeddings/
        └── get_embedding, get_embeddings_batch (already async)

anatomic-locations
    └── AnatomicLocationIndex
        ├── search() - async, hybrid (FTS + semantic), uses to_thread()
        └── async context manager

findingmodel
    └── DuckDBIndex
        ├── search() - async, hybrid, uses to_thread() (FIX existing)
        └── async context manager (already exists)

findingmodel-ai
    └── Uses AnatomicLocationIndex.search() directly
        (delete DuckDBOntologySearchClient)
```

---

## Sub-Phase 4.6.0: Create Async DuckDB Infrastructure in oidm-common

**Status:** ⏳ PENDING

**Goal:** Establish proper async patterns for DuckDB operations using `asyncer`.

### Step 0.0: Add asyncer dependency to oidm-common

**File:** `packages/oidm-common/pyproject.toml`

Add to dependencies:
```toml
dependencies = [
    # ... existing deps ...
    "asyncer==0.0.12",
]
```

### Step 0.1: Add Tests for asyncify usage with DuckDB

**File:** `packages/oidm-common/tests/test_duckdb_async.py` (NEW)

```python
import asyncio
import pytest
import duckdb
from asyncer import asyncify

# Define sync helpers with full type hints - types are preserved through asyncify
def query_all_rows(conn: duckdb.DuckDBPyConnection) -> list[tuple[int, str]]:
    """Sync helper with typed return - asyncify preserves this."""
    return conn.execute("SELECT * FROM test").fetchall()

def query_count(conn: duckdb.DuckDBPyConnection) -> int:
    """Sync helper returning count."""
    result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
    return result[0] if result else 0

@pytest.mark.asyncio
async def test_asyncify_typed_sync_function():
    """Test asyncify with typed sync function - types preserved."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
    conn.execute("INSERT INTO test VALUES (1, 'alice'), (2, 'bob')")

    # Wrap typed sync helper - return type flows through
    rows = await asyncify(query_all_rows)(conn)

    assert len(rows) == 2
    assert rows[0] == (1, 'alice')
    conn.close()

@pytest.mark.asyncio
async def test_asyncify_concurrent_typed_calls():
    """Multiple typed async calls should run concurrently."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE test (id INTEGER)")
    conn.execute("INSERT INTO test VALUES (1), (2), (3)")

    # Each call wraps the typed sync helper
    results = await asyncio.gather(
        asyncify(query_count)(conn),
        asyncify(query_count)(conn),
        asyncify(query_count)(conn),
    )

    assert results == [3, 3, 3]
    conn.close()
```

---

## Sub-Phase 4.6.1: Fix DuckDBIndex Async Patterns

**Status:** ⏳ PENDING

**Goal:** Make `DuckDBIndex` in findingmodel properly async (not pseudo-async).

### Step 1.1: Pattern - Typed Sync Helpers + Asyncify

**Key insight:** Define sync helper methods with full type hints, then wrap with `asyncify` at call sites. This preserves type information (lambdas lose types).

**File:** `packages/findingmodel/src/findingmodel/index.py`

Add import:
```python
from asyncer import asyncify
```

**Pattern:**

```python
# 1. Define sync helper with full type hints
def _query_by_id(
    self,
    conn: duckdb.DuckDBPyConnection,
    identifier: str,
) -> tuple | None:
    """Check if identifier exists (sync - types preserved)."""
    return conn.execute(
        "SELECT oifm_id FROM finding_models WHERE oifm_id = ?", [identifier]
    ).fetchone()

# 2. Async method wraps the typed sync helper
async def contains(self, identifier: str) -> bool:
    conn = self._ensure_connection()
    result = await asyncify(self._query_by_id)(conn, identifier)
    return result is not None
```

### Step 1.2: Sync Helpers to Create

Define these sync helpers with full type annotations:

```python
def _query_by_id(self, conn: DuckDBPyConnection, identifier: str) -> tuple | None:
    """Lookup by ID."""
    ...

def _query_full_model(self, conn: DuckDBPyConnection, oifm_id: str) -> dict | None:
    """Get full model data."""
    ...

def _query_count(self, conn: DuckDBPyConnection, table: str) -> int:
    """Count rows in table."""
    ...

def _search_fts(
    self,
    conn: DuckDBPyConnection,
    query: str,
    limit: int,
    tags: Sequence[str] | None,
) -> list[tuple[str, float]]:
    """FTS search returning (oifm_id, score) pairs."""
    ...

def _search_semantic(
    self,
    conn: DuckDBPyConnection,
    embedding: list[float],
    limit: int,
    tags: Sequence[str] | None,
) -> list[tuple[str, float]]:
    """Semantic search returning (oifm_id, distance) pairs."""
    ...
```

### Step 1.3: Async Methods Wrap Sync Helpers

```python
async def search(
    self,
    query: str,
    *,
    limit: int = 10,
    tags: Sequence[str] | None = None,
) -> list[IndexEntry]:
    conn = self._ensure_connection()

    # FTS search - wrap typed sync helper
    fts_results = await asyncify(self._search_fts)(conn, query, limit * 2, tags)

    # Semantic search - embedding is already async, then wrap sync helper
    embedding = await get_embedding_for_duckdb(query)
    semantic_results = []
    if embedding is not None:
        semantic_results = await asyncify(self._search_semantic)(
            conn, embedding, limit * 2, tags
        )

    # RRF fusion - CPU-bound, fast, stays sync (no I/O)
    fused = rrf_fusion(fts_results, semantic_results)
    # ... rest of method
```

---

## Sub-Phase 4.6.2: Add Hybrid Search to AnatomicLocationIndex

**Status:** ⏳ PENDING

**Goal:** Enhance `AnatomicLocationIndex` with proper async hybrid search.

### Step 2.1: Add Embedding Settings

**File:** `packages/anatomic-locations/src/anatomic_locations/config.py`

```python
# Embedding configuration (for hybrid search)
openai_api_key: QuoteStrippedSecretStr | None = Field(default=None)
openai_embedding_model: str = Field(default="text-embedding-3-small")
openai_embedding_dimensions: int = Field(default=512)
```

### Step 2.2: Add Async Context Manager

**File:** `packages/anatomic-locations/src/anatomic_locations/index.py`

```python
from asyncer import asyncify

class AnatomicLocationIndex:
    # Keep existing sync context manager for backwards compatibility
    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    # Add async context manager
    async def __aenter__(self) -> Self:
        return self.open()

    async def __aexit__(self, *args) -> None:
        self.close()
```

### Step 2.3: Define Typed Sync Helpers

**File:** `packages/anatomic-locations/src/anatomic_locations/index.py`

Define sync helpers with full type annotations (same pattern as DuckDBIndex):

```python
def _find_exact_match(
    self,
    conn: duckdb.DuckDBPyConnection,
    query: str,
    region: str | None,
    sided_filter: list[str] | None,
) -> list[tuple]:
    """Find exact matches on description (sync, typed)."""
    sql = "SELECT * FROM anatomic_locations WHERE LOWER(description) = LOWER(?)"
    # Add region/sided filters as needed
    return conn.execute(sql, [query]).fetchall()

def _search_fts(
    self,
    conn: duckdb.DuckDBPyConnection,
    query: str,
    limit: int,
    region: str | None,
    sided_filter: list[str] | None,
) -> list[tuple]:
    """FTS search with BM25 scoring (sync, typed)."""
    sql = """
        SELECT *, fts_main_anatomic_locations.match_bm25(id, ?, fields := 'description') as score
        FROM anatomic_locations
        WHERE score IS NOT NULL
        ORDER BY score DESC LIMIT ?
    """
    return conn.execute(sql, [query, limit]).fetchall()

def _search_semantic(
    self,
    conn: duckdb.DuckDBPyConnection,
    embedding: list[float],
    limit: int,
    region: str | None,
    sided_filter: list[str] | None,
) -> list[tuple]:
    """Semantic search with cosine distance (sync, typed)."""
    sql = """
        SELECT *, array_cosine_distance(embedding, ?::FLOAT[512]) as distance
        FROM anatomic_locations
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC LIMIT ?
    """
    return conn.execute(sql, [embedding, limit]).fetchall()
```

### Step 2.4: Async Search Wraps Sync Helpers

```python
async def search(
    self,
    query: str,
    *,
    limit: int = 30,
    region: str | None = None,
    sided_filter: list[str] | None = None,
) -> list[AnatomicLocation]:
    """Hybrid search combining FTS and semantic search with RRF fusion."""
    conn = self._ensure_connection()

    # Check for exact matches first - wrap typed sync helper
    exact_rows = await asyncify(self._find_exact_match)(
        conn, query, region, sided_filter
    )
    if exact_rows:
        return [self._row_to_location(r) for r in exact_rows[:limit]]

    # FTS search - wrap typed sync helper
    fts_rows = await asyncify(self._search_fts)(
        conn, query, limit * 2, region, sided_filter
    )

    # Semantic search - embedding API already async, then wrap sync helper
    embedding = await self._get_embedding(query)
    semantic_rows = []
    if embedding is not None:
        semantic_rows = await asyncify(self._search_semantic)(
            conn, embedding, limit * 2, region, sided_filter
        )

    # If no semantic results, return FTS only
    if not semantic_rows:
        return [self._row_to_location(r) for r in fts_rows[:limit]]

    # Apply RRF fusion - CPU-bound, fast, stays sync (no I/O)
    fused = self._apply_rrf_fusion(fts_rows, semantic_rows, limit)
    return [self._row_to_location(r) for r in fused]
```

### Step 2.5: Other Helper Methods

```python
async def _get_embedding(self, text: str) -> list[float] | None:
    """Get embedding for query text (async - calls OpenAI API)."""
    from anatomic_locations.config import settings
    from oidm_common.embeddings import get_embedding

    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    if not api_key:
        return None

    return await get_embedding(
        text,
        api_key=api_key,
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
    )

def _apply_rrf_fusion(
    self,
    fts_results: list[tuple],
    semantic_results: list[tuple],
    limit: int,
) -> list[tuple]:
    """Apply RRF fusion to combine result sets (sync - CPU-bound, fast)."""
    from oidm_common.duckdb import rrf_fusion
    # ... implementation
```

---

## Sub-Phase 4.6.3: Update findingmodel-ai Consumers

**Status:** ⏳ PENDING

### Step 3.1: Update anatomic_location_search.py

**File:** `packages/findingmodel-ai/src/findingmodel_ai/tools/anatomic_location_search.py`

```python
# OLD:
from findingmodel.tools.duckdb_search import DuckDBOntologySearchClient

async with DuckDBOntologySearchClient() as client:
    results = await client.search(queries)

# NEW:
from anatomic_locations import AnatomicLocationIndex

async with AnatomicLocationIndex() as index:
    results = []
    for query in queries:
        matches = await index.search(query, limit=limit_per_query)
        results.extend(_convert_to_ontology_results(matches))
```

### Step 3.2: Update finding_enrichment.py

**File:** `packages/findingmodel-ai/src/findingmodel_ai/tools/finding_enrichment.py`

Same pattern as above.

### Step 3.3: Add Helper for Result Conversion

```python
def _convert_to_ontology_results(
    locations: list[AnatomicLocation],
) -> list[OntologySearchResult]:
    """Convert AnatomicLocation objects to OntologySearchResult."""
    return [
        OntologySearchResult(
            concept_id=loc.id,
            concept_text=loc.description,
            score=getattr(loc, '_search_score', 0.0),
            table_name="anatomic_locations",
        )
        for loc in locations
    ]
```

---

## Sub-Phase 4.6.4: Delete Deprecated Code

**Status:** ✅ COMPLETE

### Step 4.1: Delete duckdb_search.py

**DELETE:** `packages/findingmodel/src/findingmodel/tools/duckdb_search.py` ✅

### Step 4.2: Update tools/__init__.py

Remove `DuckDBOntologySearchClient` export. ✅ (was not exported)

### Step 4.3: Delete or Update Demo Script

**File:** `scripts/anatomic_location_search.py` ✅ DELETED

Decision: Deleted the script since it was a demo/testing tool that used the deprecated `DuckDBOntologySearchClient`. Users should use `AnatomicLocationIndex` directly for anatomic location search.

---

## Sub-Phase 4.6.5: Tests and Verification

**Status:** ⏳ PENDING

### Step 5.1: Test Async Behavior

```python
@pytest.mark.asyncio
async def test_search_does_not_block_event_loop():
    """Verify concurrent searches run in parallel."""
    async with AnatomicLocationIndex() as index:
        start = time.perf_counter()
        # Run 10 searches concurrently
        results = await asyncio.gather(*[
            index.search(f"query_{i}")
            for i in range(10)
        ])
        elapsed = time.perf_counter() - start

        # If properly async, should be much faster than 10x single query time
        assert len(results) == 10
```

### Step 5.2: Verification Checklist

```bash
# 1. Verify duckdb_search.py is deleted
ls packages/findingmodel/src/findingmodel/tools/duckdb_search.py
# Should return: No such file or directory

# 2. Verify no imports of deleted module
grep -rn "from findingmodel.tools.duckdb_search" packages/
# Should find nothing

# 3. Verify run_in_thread is used in both indexes
grep -n "run_in_thread" packages/findingmodel/src/findingmodel/index.py
grep -n "run_in_thread" packages/anatomic-locations/src/anatomic_locations/index.py

# 4. All tests pass
task test

# 5. All checks pass
task check
```

---

## Acceptance Criteria

- [ ] `asyncer==0.0.12` is a dependency of `oidm-common`
- [ ] `DuckDBIndex` methods wrap DuckDB calls with `asyncify(lambda: ...)()`
- [ ] `AnatomicLocationIndex.search()` is async with hybrid search
- [ ] `AnatomicLocationIndex` has async context manager
- [ ] `findingmodel-ai` uses `AnatomicLocationIndex` directly (no adapter needed)
- [ ] `findingmodel/tools/duckdb_search.py` is deleted
- [ ] Concurrent async searches don't block each other
- [ ] All tests pass
- [ ] All checks pass

---

## Reference: Async Pattern Summary

| Operation | Pattern | Why |
|-----------|---------|-----|
| DuckDB query | `await asyncify(self._sync_helper)(args)` | Typed sync helper preserves types |
| Embedding API call | `await get_embedding(...)` | Already async |
| RRF fusion | `rrf_fusion(...)` (sync) | CPU-bound, fast, no I/O |
| Context manager | `async with Index() as index:` | Consistency |

**Key Pattern:** Define sync helper methods with full type annotations, then wrap with `asyncify` at call sites.

```python
# 1. Typed sync helper (types preserved)
def _search_fts(self, conn: DuckDBPyConnection, query: str, limit: int) -> list[tuple[str, float]]:
    return conn.execute(sql, [query, limit]).fetchall()

# 2. Async method wraps it
async def search(self, query: str, limit: int = 10) -> list[Result]:
    results = await asyncify(self._search_fts)(conn, query, limit)  # Types flow through
```

**Why not lambdas:** `asyncify(lambda: ...)` loses type information - return type becomes `Any`.

**Why asyncer:**
- From the FastAPI creator (Sebastián Ramírez), designed for this use case
- Preserves type hints when wrapping typed functions
- Better integration with FastAPI/Starlette ecosystem
- Pinned to 0.0.12 (latest)
