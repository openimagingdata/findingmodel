---
paths:
  - "packages/findingmodel/**"
  - "packages/anatomic-locations/**"
  - "packages/oidm-common/**"
---

# DuckDB Index Patterns

Shared patterns for packages using DuckDB-based indexes (finding models, anatomic locations).

## Base Class: `ReadOnlyDuckDBIndex`

`oidm_common.duckdb.base.ReadOnlyDuckDBIndex` is the shared base for all read-only DuckDB indexes:
- `AnatomicLocationIndex` and `FindingModelIndex` (also exported as `Index`) both inherit from it.
- Provides: `db_path` resolution, `open()` / `close()`, sync and async context managers.
- `_ensure_connection()` auto-opens the connection on first use — explicit `open()` is optional.

## Database Lifecycle

- **Auto-download**: Databases download automatically on first use via pooch with checksum verification.
- **Auto-open**: `_ensure_connection()` opens the connection implicitly; no need to call `open()` for simple queries.
- **Context managers**: Both sync (`with`) and async (`async with`) context managers are supported.
- **Connection cleanup**: Always close explicitly (`close()`) or use a context manager in long-lived processes.

## Class Names

- **`FindingModelIndex`** – canonical name in `findingmodel.index` (previously `DuckDBIndex`)
- **`Index`** – public alias exported from `findingmodel` package for backward compatibility
- **`AnatomicLocationIndex`** – in `anatomic_locations.index`

## Index Rebuild Strategy

See Serena `duckdb_architecture` for the 3-step process:
1. Drop HNSW/FTS indexes before bulk writes
2. Clear denormalized tables manually
3. Rebuild indexes afterward

## Search Patterns

- **Hybrid search**: Combines FTS (full-text) + vector (HNSW) with configurable weights
- **Embedding generation**: Uses OpenAI embeddings via oidm-common client
- **Result ranking**: FTS score + cosine similarity, weighted by config

## Configuration

- `FINDINGMODEL_DB_PATH` – override auto-download location for finding models
- `ANATOMIC_DB_PATH` – override for anatomic locations database

## Row Hydration Pattern

Both `AnatomicLocationIndex` and `FindingModelIndex` use **named dict access** for row hydration, provided by two helpers on `ReadOnlyDuckDBIndex`:

```python
# Single row — returns dict[str, object] | None
row = self._execute_one(conn, "SELECT * EXCLUDE (search_text, vector) FROM anatomic_locations WHERE id = ?", [id])
if row is not None:
    description = str(row["description"])  # Named access — schema-change safe

# Multiple rows — returns list[dict[str, object]]
rows = self._execute_all(conn, "SELECT * EXCLUDE (search_text, vector) FROM anatomic_locations WHERE region = ?", [region])
locations = [self._row_to_location(row) for row in rows]
```

**Why this matters:** Positional indexing (`row[N]`) breaks silently when columns are added, removed, or reordered. Named dict access via `cursor.description` is immune to schema evolution.

**`SELECT * EXCLUDE (search_text, vector)` pattern:**
- EXCLUDE `search_text` and `vector` — large columns not needed for object construction; exist in **all** schema versions
- Do NOT EXCLUDE `synonyms_text` — it was added in v0.2.3 and is absent from older production DBs; EXCLUDEing it would raise a `Binder Error` on old schemas. Named dict access simply won't find the key when it's absent.

**Bulk methods:** Use `_execute_all()` + direct `_row_to_location()` (single query per method, no N+1 per-row re-fetch).

## Serena References

- `duckdb_architecture` – consolidated design decisions and patterns
