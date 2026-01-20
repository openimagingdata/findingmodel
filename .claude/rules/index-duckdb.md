---
paths:
  - "packages/findingmodel/**"
  - "packages/anatomic-locations/**"
  - "packages/oidm-common/**"
---

# DuckDB Index Patterns

Shared patterns for packages using DuckDB-based indexes (finding models, anatomic locations).

## Database Lifecycle

- **Auto-download**: Databases download automatically on first use via pooch with checksum verification.
- **Async context manager**: Always use `async with Index() as index:` pattern.
- **Connection cleanup**: Ensure connections close properly with try/finally.

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

- `DUCKDB_INDEX_PATH` – override auto-download location for finding models
- `ANATOMIC_DB_PATH` – override for anatomic locations database

## Serena References

- `duckdb_architecture` – consolidated design decisions and patterns
