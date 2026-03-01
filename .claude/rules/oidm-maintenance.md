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

## Schema Change Checklist
When modifying the `anatomic_locations` or `finding_models` table schema in `build.py`:
1. **No positional index updates needed** — both `AnatomicLocationIndex._row_to_location()` and `FindingModelIndex._fetch_index_entry()` use named dict access via `_execute_one`/`_execute_all` from `ReadOnlyDuckDBIndex`. New columns are automatically available by name; removed columns will raise `KeyError` only if code references them.
2. Rebuild the test fixture: `uv run python packages/oidm-maintenance/scripts/build_anatomic_test_fixture.py`
3. Run `task test` and verify all tests pass.

## Serena References
- `index_duckdb_migration_status_2025` – index build pipeline
- `duckdb_development_patterns` – DuckDB conventions
