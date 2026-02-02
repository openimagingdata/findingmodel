# Consolidate Embedding Infrastructure

**Status: COMPLETE** — Implemented 2026-02-02

Date: 2026-02-02
Related issues: #3 (embedding helper duplication), #8 (database builds don't use embedding cache)

## Implementation Notes

All 6 steps completed as planned:
- **oidm-common 0.2.2**: `get_embedding()` and `get_embeddings_batch()` now have transparent always-on caching via module-level `EmbeddingCache` singleton. `_UNSET` sentinel distinguishes "not provided" (use singleton) from `cache=None` (disable). Partial-hit optimization in batch calls generates only for cache misses.
- **findingmodel 1.0.1**: Deleted `get_embedding_for_duckdb()` and `batch_embeddings_for_duckdb()` from `duckdb_utils.py` (file fully removed). `DuckDBIndex` calls oidm-common directly.
- **anatomic-locations 0.2.1**: Deleted `_get_embedding()` wrapper. `search()` calls oidm-common directly.
- **oidm-maintenance 0.2.1**: Deleted `_generate_embeddings_async()` from findingmodel build. Replaced inline `AsyncOpenAI` client creation in anatomic build. Both use `get_embeddings_batch()` from oidm-common.
- **Tests**: 5 new cache-aware tests in oidm-common. `test_embedding_cache.py` removed from findingmodel (consolidated). 7 mock targets updated in oidm-maintenance tests.

## Problem

`oidm-common` has both embedding generation (`generation.py`) and caching (`cache.py`), but they're completely separate. Every downstream package wrote its own wrapper to resolve config → call generation, and none of them use the cache. Result: duplicated wrappers everywhere, expensive rebuilds regenerating all embeddings from scratch.

## Approach: Always-on transparent caching in oidm-common

The high-level functions `get_embedding()` and `get_embeddings_batch()` gain built-in caching using a module-level `EmbeddingCache` singleton at the default platformdirs path (`~/.cache/findingmodel/embeddings.duckdb` — already defined in `cache.py`). Callers don't see or manage the cache. Cache is always on.

A `cache: EmbeddingCache | None` parameter is still exposed for override/testing (pass explicit cache or `None` to disable), but defaults to the module-level singleton.

Then kill every wrapper in downstream packages and have callers use oidm-common directly.

---

## Reference: Current Code (read this before making changes)

### generation.py — current full source

**File**: `packages/oidm-common/src/oidm_common/embeddings/generation.py`

```python
"""Embedding generation utilities for OpenAI models."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Module-level client cache for connection reuse
_client_cache: dict[str, "AsyncOpenAI"] = {}


def _to_float32(embedding: list[float]) -> list[float]:
    """Convert embedding to 32-bit floats for DuckDB compatibility."""
    return list(array("f", embedding))


def _get_or_create_client(api_key: str) -> "AsyncOpenAI | None":
    """Get or create an AsyncOpenAI client, caching by API key.
    Returns None if openai is not installed (graceful degradation).
    """
    if not api_key:
        return None
    if api_key in _client_cache:
        return _client_cache[api_key]
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.debug("openai not installed - semantic search disabled")
        return None
    client = AsyncOpenAI(api_key=api_key)
    _client_cache[api_key] = client
    return client


async def get_embedding(
    text: str,
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
) -> list[float] | None:
    """Generate a single embedding vector for text."""
    client = _get_or_create_client(api_key)
    if client is None:
        return None
    return await generate_embedding(text, client, model, dimensions)


async def get_embeddings_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
) -> list[list[float] | None]:
    """Generate embeddings for multiple texts in a single API call."""
    if not texts:
        return []
    client = _get_or_create_client(api_key)
    if client is None:
        return [None] * len(texts)
    return await generate_embeddings_batch(texts, client, model, dimensions)


# Low-level functions (generate_embedding, generate_embeddings_batch, create_openai_client)
# stay unchanged — they take a client and have no cache involvement.

__all__ = [
    "create_openai_client",
    "generate_embedding",
    "generate_embeddings_batch",
    "get_embedding",
    "get_embeddings_batch",
]
```

### EmbeddingCache — key details

**File**: `packages/oidm-common/src/oidm_common/embeddings/cache.py`

- Default path: `Path(user_cache_dir(appname="findingmodel", appauthor="openimagingdata", ensure_exists=True)) / "embeddings.duckdb"`
- `__init__(self, db_path: Path | None = None)` — defaults to above path
- `async def setup(self)` — creates schema (CREATE TABLE IF NOT EXISTS), idempotent, fail-safe (logs warning, doesn't raise)
- `async def get_embedding(self, text, model, dimensions) -> list[float] | None`
- `async def store_embedding(self, text, model, dimensions, embedding) -> None`
- `async def get_embeddings_batch(self, texts, model, dimensions) -> list[list[float] | None]`
- `async def store_embeddings_batch(self, texts, model, dimensions, embeddings) -> None`
  - **IMPORTANT**: `embeddings` param is `list[list[float]]` — NO None values allowed. Must filter before calling.
- Async context manager: `__aenter__` calls `setup()`, `__aexit__` closes connection
- All operations are fail-safe (catch exceptions, log, don't raise)

### Wrapper functions to DELETE

**1. `findingmodel/tools/duckdb_utils.py`** — `get_embedding_for_duckdb()` and `batch_embeddings_for_duckdb()`

These just read `findingmodel.config.settings` for api_key/model/dimensions, then call `oidm_common.embeddings.get_embedding()` / `get_embeddings_batch()`. The file also contains FTS/HNSW index utilities (`create_fts_index`, `create_hnsw_index`, `drop_search_indexes`) which must be kept.

**2. `anatomic-locations/index.py`** — `AnatomicLocationIndex._get_embedding()`

```python
async def _get_embedding(self, text: str) -> list[float] | None:
    from anatomic_locations.config import get_settings
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    if not api_key:
        return None
    return await get_embedding(
        text, api_key=api_key,
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
    )
```

Called at one place in `search()` method: `embedding = await self._get_embedding(query)`

**3. `oidm-maintenance/findingmodel/build.py`** — `_generate_embeddings_async()`

```python
async def _generate_embeddings_async(embedding_texts: Sequence[str]) -> list[list[float]]:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not configured in environment")
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    raw_embeddings = await generate_embeddings_batch(
        texts=list(embedding_texts), client=client,
        model=settings.openai_embedding_model,
        dimensions=settings.openai_embedding_dimensions,
    )
    embeddings: list[list[float]] = []
    for i, embedding in enumerate(raw_embeddings):
        if embedding is None:
            raise RuntimeError(f"Failed to generate embedding for model at index {i}")
        embeddings.append(embedding)
    return embeddings
```

Note: This uses the LOW-LEVEL `generate_embeddings_batch(texts, client, ...)` — creates its own client. Should switch to high-level `get_embeddings_batch(texts, api_key=..., ...)`.

**4. `oidm-maintenance/anatomic/build.py`** — inline in `build_anatomic_database()`

The embedding block (inside `if generate_embeddings:`) at lines ~1041-1072:
```python
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
embeddings: list[list[float] | None] = []
for i in range(0, len(searchable_texts), batch_size):
    batch = searchable_texts[i : i + batch_size]
    batch_embeddings = await generate_embeddings_batch(
        batch, client,
        model=settings.openai_embedding_model, dimensions=dimensions,
    )
    embeddings.extend(batch_embeddings)
```

This also uses the low-level API and manages its own client + batch loop. Replace entirely.

### Callers to update in `findingmodel/index.py`

**`DuckDBIndex._search_semantic()`** (line 1126):
```python
async def _search_semantic(self, conn, query, *, limit, tags=None):
    if limit <= 0:
        return []
    trimmed_query = query.strip()
    if not trimmed_query:
        return []
    embedding = await get_embedding_for_duckdb(trimmed_query)  # ← REPLACE THIS
    if embedding is None:
        return []
    return await asyncify(self._search_semantic_with_embedding)(conn, embedding, limit=limit, tags=tags)
```

**`DuckDBIndex.search_batch()`** (line 485):
```python
# Line 515:
embeddings = await batch_embeddings_for_duckdb(valid_queries)  # ← REPLACE THIS
```

Import to change: line 26 currently has `from findingmodel.tools.duckdb_utils import batch_embeddings_for_duckdb, get_embedding_for_duckdb`

### Caller to update in `anatomic-locations/index.py`

**`AnatomicLocationIndex.search()`** (line 176):
```python
# Line 207:
embedding = await self._get_embedding(query)  # ← REPLACE with direct oidm-common call
```

### Mock targets in oidm-maintenance tests that need updating

**`test_findingmodel_build.py`** — 4 places mock the deleted function:
```python
patch("oidm_maintenance.findingmodel.build._generate_embeddings_async", new_callable=AsyncMock, ...)
```
Lines: 61, 138, 494, 582. These need to change to mock `oidm_common.embeddings.generation.get_embeddings_batch` (or wherever the import resolves).

There's also a low-level mock at line 750:
```python
patch("oidm_maintenance.findingmodel.build.generate_embeddings_batch", side_effect=mock_batch_embeddings_with_none)
```

**`test_cli.py`** — 2 places mock the low-level function in anatomic build:
```python
patch("oidm_maintenance.anatomic.build.generate_embeddings_batch", side_effect=fake_batch_embeddings)
```
Lines: 495, 592. These need to change to mock the high-level function instead.

---

## Step 1: Add transparent caching to `get_embedding()` and `get_embeddings_batch()`

**File**: `packages/oidm-common/src/oidm_common/embeddings/generation.py`

Add module-level singleton and sentinel:

```python
from oidm_common.embeddings.cache import EmbeddingCache

_UNSET = object()  # sentinel to distinguish "not provided" from None
_default_cache: EmbeddingCache | None = None

def _get_default_cache() -> EmbeddingCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = EmbeddingCache()  # uses platformdirs default path
    return _default_cache
```

Update both high-level functions to add `cache` parameter with `_UNSET` default:

```python
async def get_embedding(
    text: str,
    *,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
    cache: EmbeddingCache | None | object = _UNSET,
) -> list[float] | None:
```

- `cache=_UNSET` (default) → use module-level singleton, always-on
- `cache=None` → explicitly disable caching (useful for tests)
- `cache=<instance>` → use provided cache (useful for custom path)

**`get_embedding()`** logic:
1. Resolve cache (singleton if _UNSET, None if None, provided if explicit)
2. If cache: `await cache.setup()` (idempotent), check cache
3. If hit, return it
4. Generate via OpenAI (existing logic: `_get_or_create_client` → `generate_embedding`)
5. If cache and result not None: store in cache
6. Return result

**`get_embeddings_batch()`** logic:
1. Resolve cache same way
2. If cache: batch check all texts
3. Collect miss indices/texts
4. Generate batch for misses only (saves API calls)
5. Filter out None results before `cache.store_embeddings_batch()` (it requires `list[list[float]]`, no None allowed)
6. Merge generated into cached results, return

**`__init__.py`**: No export changes needed — same function names, just new optional param.

**Low-level functions unchanged**: `generate_embedding()` and `generate_embeddings_batch()` (client-based) stay cache-free.

## Step 2: Add tests for cache-aware behavior in oidm-common

**File**: `packages/oidm-common/tests/test_embeddings.py` (add to existing)

Tests (mock OpenAI calls):
- `test_get_embedding_cache_none_passthrough` — cache=None behaves as before
- `test_get_embedding_cache_hit_no_api_call` — cache hit skips OpenAI
- `test_get_embedding_cache_miss_generates_and_stores` — miss → generate → store
- `test_get_embeddings_batch_partial_cache_hits` — 3 texts, 1 cached, 2 generated
- `test_get_embeddings_batch_filters_none_before_storing` — None results from generation not passed to store_embeddings_batch

## Step 3: Delete wrappers in findingmodel, update callers

### 3a. Delete embedding functions from `duckdb_utils.py`

**File**: `packages/findingmodel/src/findingmodel/tools/duckdb_utils.py`

Remove `get_embedding_for_duckdb()` and `batch_embeddings_for_duckdb()`. Remove from `__all__`. File keeps its FTS/HNSW index utility functions (`create_fts_index`, `create_hnsw_index`, `drop_search_indexes`).

### 3b. Update `DuckDBIndex` to call oidm-common directly

**File**: `packages/findingmodel/src/findingmodel/index.py`

Change import on line 26 from `findingmodel.tools.duckdb_utils` to `oidm_common.embeddings`.

In `_search_semantic()` (line 1143): replace `await get_embedding_for_duckdb(trimmed_query)` with:
```python
api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
embedding = await get_embedding(
    trimmed_query,
    api_key=api_key,
    model=settings.openai_embedding_model,
    dimensions=settings.openai_embedding_dimensions,
)
```

In `search_batch()` (line 515): replace `await batch_embeddings_for_duckdb(valid_queries)` with:
```python
api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
embeddings = await get_embeddings_batch(
    valid_queries,
    api_key=api_key,
    model=settings.openai_embedding_model,
    dimensions=settings.openai_embedding_dimensions,
)
```

Caching is automatic via the default singleton.

## Step 4: Delete wrapper in anatomic-locations, update caller

**File**: `packages/anatomic-locations/src/anatomic_locations/index.py`

Delete `_get_embedding()` method (lines 581-602).

In `search()` (line 207), replace `await self._get_embedding(query)` with:
```python
from anatomic_locations.config import get_settings as get_anatomic_settings
anatomic_settings = get_anatomic_settings()
api_key = anatomic_settings.openai_api_key.get_secret_value() if anatomic_settings.openai_api_key else ""
embedding = await get_embedding(
    query, api_key=api_key,
    model=anatomic_settings.openai_embedding_model,
    dimensions=anatomic_settings.openai_embedding_dimensions,
) if api_key else None
```

Import `get_embedding` from `oidm_common.embeddings` at top of file.

## Step 5: Simplify oidm-maintenance build pipelines

### 5a. findingmodel build

**File**: `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`

- Delete `_generate_embeddings_async()` (lines 278-312)
- Replace its call site with direct call to `get_embeddings_batch()`:

```python
from oidm_common.embeddings import get_embeddings_batch

embeddings_raw = await get_embeddings_batch(
    list(embedding_texts),
    api_key=settings.openai_api_key.get_secret_value(),
    model=settings.openai_embedding_model,
    dimensions=settings.openai_embedding_dimensions,
)
# Validate no None results (preserve existing error behavior)
embeddings: list[list[float]] = []
for i, emb in enumerate(embeddings_raw):
    if emb is None:
        raise RuntimeError(f"Failed to generate embedding for model at index {i}")
    embeddings.append(emb)
```

- Remove `AsyncOpenAI` import (no longer creating client directly)
- Caching is automatic via the default singleton

### 5b. anatomic build

**File**: `packages/oidm-maintenance/src/oidm_maintenance/anatomic/build.py`

Replace the entire `if generate_embeddings:` block (lines ~1041-1072) that creates an AsyncOpenAI client and loops in batches. Replace with:

```python
from oidm_common.embeddings import get_embeddings_batch

embeddings = await get_embeddings_batch(
    searchable_texts,
    api_key=settings.openai_api_key.get_secret_value(),
    model=settings.openai_embedding_model,
    dimensions=dimensions,
)
```

Note: The old code batched in groups of `batch_size` for rate limiting. The high-level `get_embeddings_batch()` sends all texts in a single OpenAI API call. If rate limiting is needed, that should be handled at a lower level (the OpenAI client has built-in retry). If the batch is very large (1000+ texts), consider whether the OpenAI API has input limits — but the current anatomic dataset is ~3000 records, well within limits.

### 5c. CLI

**File**: `packages/oidm-maintenance/src/oidm_maintenance/cli.py`

No CLI changes needed — caching is always-on via the default path.

## Step 6: Consolidate tests

### 6a. Move EmbeddingCache tests to oidm-common

- `packages/findingmodel/tests/test_embedding_cache.py` (36 tests) substantially duplicates `packages/oidm-common/tests/test_embeddings.py`
- Merge any unique tests from findingmodel's file into oidm-common's test file
- Delete `packages/findingmodel/tests/test_embedding_cache.py`
- `findingmodel/__init__.py` already re-exports `EmbeddingCache` — keep for backward compat

### 6b. Update mock targets in oidm-maintenance tests

**`test_findingmodel_build.py`** — 4 patches at lines 61, 138, 494, 582 currently target:
```python
"oidm_maintenance.findingmodel.build._generate_embeddings_async"
```
Change to target the high-level function imported into build.py:
```python
"oidm_maintenance.findingmodel.build.get_embeddings_batch"
```
Note: mock return type changes from `list[list[float]]` to `list[list[float] | None]` — the build code now handles the None filtering itself.

Also line 750 targets:
```python
"oidm_maintenance.findingmodel.build.generate_embeddings_batch"
```
This should change to target the same high-level function.

**`test_cli.py`** — 2 patches at lines 495, 592 currently target:
```python
"oidm_maintenance.anatomic.build.generate_embeddings_batch"
```
Change to:
```python
"oidm_maintenance.anatomic.build.get_embeddings_batch"
```

---

## Verification

1. `task check` — ruff + mypy clean
2. `task test:oidm-common` — new cache-aware tests pass
3. `task test:findingmodel` — no regressions (wrapper deletion + caller updates)
4. `task test:anatomic-locations` — no regressions
5. `task test:oidm-maintenance` — build tests pass with updated mock targets
6. `task test` — full suite green

## Files Modified

| File | Change |
|------|--------|
| `oidm-common/embeddings/generation.py` | Add singleton cache + `cache` param to `get_embedding`, `get_embeddings_batch` |
| `oidm-common/tests/test_embeddings.py` | Add cache-aware tests |
| `findingmodel/tools/duckdb_utils.py` | Delete `get_embedding_for_duckdb`, `batch_embeddings_for_duckdb` |
| `findingmodel/index.py` | Update imports + 2 call sites to use oidm-common directly |
| `findingmodel/tests/test_embedding_cache.py` | Delete (merged into oidm-common) |
| `anatomic-locations/index.py` | Delete `_get_embedding`, update `search()` |
| `oidm-maintenance/findingmodel/build.py` | Delete `_generate_embeddings_async`, use `get_embeddings_batch` from oidm-common |
| `oidm-maintenance/anatomic/build.py` | Remove inline client creation + batch loop, use `get_embeddings_batch` from oidm-common |
| `oidm-maintenance/tests/test_findingmodel_build.py` | Update 5 mock targets from deleted functions to `get_embeddings_batch` |
| `oidm-maintenance/tests/test_cli.py` | Update 2 mock targets from `generate_embeddings_batch` to `get_embeddings_batch` |

## Coding conventions

- 120 char line limit, ruff formatting
- Use `from __future__ import annotations` in all files
- Strict typing: annotate everything
- Don't add `# noqa` comments without explicit approval
- Use `Index` not `DuckDBIndex` for public references (DuckDBIndex is being removed from public API)
- Run `task check` (ruff + mypy) and `task test` to verify
