# DuckDB Architecture

Consolidated reference for DuckDB patterns in the findingmodel project.

## Base Class: ReadOnlyDuckDBIndex

`oidm_common.duckdb.base.ReadOnlyDuckDBIndex` is the shared base for all read-only DuckDB indexes.

**Provides:**
- `db_path` resolution (explicit path or callable `ensure_db`)
- `open()` / `close()` for explicit lifecycle management
- Sync (`with`) and async (`async with`) context managers
- `_ensure_connection()` — auto-opens the connection on first use (no RuntimeError)

**Subclasses:**
- `AnatomicLocationIndex` (`packages/anatomic-locations/src/anatomic_locations/index.py`)
- `FindingModelIndex` (`packages/findingmodel/src/findingmodel/index.py`) — also exported as `Index`

**Usage patterns:**
```python
# Auto-open (simplest — connection opens on first use)
index = AnatomicLocationIndex()
loc = index.get("kidney")

# Context manager (explicit cleanup)
with AnatomicLocationIndex() as index:
    loc = index.get("RID2772")

# Explicit open/close (FastAPI lifespan)
index = AnatomicLocationIndex()
index.open()
# ...
index.close()
```

## Design Decisions

### Drop/Rebuild HNSW Strategy
- Drop HNSW and FTS indexes before batch writes, rebuild after
- Avoids experimental HNSW persistence flag (safer, simpler)
- No corruption risk from unexpected shutdown
- Search unavailable for ~5 seconds during batch updates (acceptable - infrequent)

### No Foreign Key Constraints
- Simplifies drop/rebuild strategy
- Integrity enforced by application during rebuild process
- Denormalized tables always refreshed completely before recreating indexes
- Manual cleanup via `_delete_denormalized_records()` before mutations

### Semantic Search Always Enabled
- Embeddings NOT NULL, OpenAI key REQUIRED
- FLOAT[512] embeddings from text-embedding-3-small
- Consistent with anatomic location search pattern

### Remote Database Downloads
- Optional automatic download of pre-built DuckDB files
- Flexible configuration priority:
  1. If `*_DB_PATH` exists (no URL/hash): use file directly
  2. If `*_DB_PATH` exists AND URL/hash set: verify hash
  3. If `*_DB_PATH` doesn't exist AND URL/hash set: download from URL
  4. If nothing specified: download from manifest.json (fallback)
- SHA256 verification via Pooch library

## Patterns

### Connection Lifecycle
- Auto-open via `_ensure_connection()` (base class behavior)
- Sync and async context managers both call `open()` / `close()`
- Read-only by default, explicit writable mode for updates
- Single writer pattern (DuckDB limitation)
- Connection cleanup with try/finally

### Bulk Loading
- `read_json()` for FLOAT[]/STRUCT[] (1000x faster than executemany)
- Hash-based diffing with temp table + full outer join for batch updates
- Directory ingestion with `update_from_directory()`

### Hybrid Search
- Exact match check first → FTS + semantic → weighted fusion
- FTS: BM25 scoring on search_text (name + description + synonyms)
- HNSW: Vector similarity on embeddings (L2 distance, convert to cosine)
- Fusion: `0.3 * normalized_bm25 + 0.7 * cosine_similarity`
- L2→Cosine conversion: `cosine_sim = 1 - (l2_distance / 2)`

### Embedding Format
- `get_embedding()` / `get_embeddings_batch()` from oidm-common (with transparent caching)
- Store as DOUBLE[] array columns
- text-embedding-3-small (512 dimensions)

## Configuration

Environment variables:
- `FINDINGMODEL_DB_PATH` – path to finding models database
- `ANATOMIC_DB_PATH` – path to anatomic locations database
- `FINDINGMODEL_REMOTE_DB_URL`, `FINDINGMODEL_REMOTE_DB_HASH` – download config for index
- `ANATOMIC_REMOTE_DB_URL`, `ANATOMIC_REMOTE_DB_HASH` – download config for anatomic
- `FINDINGMODEL_MANIFEST_URL` – fallback manifest URL

Path resolution:
- `None` → `{user_data_dir}/{manifest_key}.duckdb`
- Relative path → `{user_data_dir}/{relative_path}`
- Absolute path → used as-is

## Schema (8 Tables)

1. **finding_models** – main metadata with embeddings (NOT NULL)
2. **people** – normalized person master data
3. **organizations** – normalized organization master data
4. **model_people** – denormalized model→person links (junction)
5. **model_organizations** – denormalized model→organization links (junction)
6. **synonyms** – denormalized synonym storage
7. **attributes** – denormalized attribute storage
8. **tags** – denormalized tag storage

## Common Pitfalls

- Unquoted column types in `read_json()`
- Missing float32 conversion for embeddings
- HNSW on read-only connections (must be writable during index creation)
- `CURRENT_TIMESTAMP()` syntax (use `now()` instead)
