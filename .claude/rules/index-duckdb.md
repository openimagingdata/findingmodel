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

## Serena References

- `duckdb_architecture` – consolidated design decisions and patterns
