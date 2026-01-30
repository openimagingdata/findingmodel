# Phase 3.5: Embedding Utilities Cleanup

**Status:** ✅ COMPLETE

**Goal:** Consolidate embedding generation code in oidm-common

## Completed Work

- Created `oidm_common/embeddings/generation.py` with:
  - `generate_embedding(text, client, model, dimensions)` → single embedding
  - `generate_embeddings_batch(texts, client, model, dimensions)` → batch embeddings
  - Both return float32 precision for DuckDB compatibility
- Updated `anatomic_locations/migration.py` to import from oidm-common (removed duplicate)
- Updated `findingmodel/tools/duckdb_utils.py` to use oidm-common functions
- Updated test mocks in anatomic-locations tests

## Design Decisions

- Standalone async functions (YAGNI - no protocol abstraction needed)
- Functions take explicit `client: AsyncOpenAI` parameter (no config dependency in oidm-common)
- Float32 conversion happens in oidm-common; consumers don't need to handle it
- `duckdb_utils.py` remains as convenience wrappers with config-based defaults

## Verification

```bash
uv run --package oidm-common pytest         # ✅ 76 passed
uv run --package anatomic-locations pytest  # ✅ 80 passed
uv run --package findingmodel pytest        # ✅ 463 passed
anatomic --help                             # ✅ Works
ruff check packages/                        # ✅ All checks passed
```
