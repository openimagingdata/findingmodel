---
paths: "packages/oidm-maintenance/**"
---

# oidm-maintenance Package Rules

## Purpose
Maintainer-only tools for building and publishing OIDM databases (finding model index, anatomic locations).

## Constraints
- **Not user-facing**: This package is for OIDM maintainers only.
- **AWS credentials required**: S3 upload needs proper IAM configuration.
- **Large data operations**: Database builds may process thousands of records.

## CLI
- `oidm-maintain build-index` – Build finding model DuckDB index
- `oidm-maintain build-anatomic` – Build anatomic location DuckDB database
- `oidm-maintain publish` – Upload databases to S3

## Key Patterns
- Embedding generation for semantic search
- DuckDB index building with HNSW/FTS
- Checksum generation for pooch auto-download

## Serena References
- `index_duckdb_migration_status_2025` – index build pipeline
- `duckdb_development_patterns` – DuckDB conventions
