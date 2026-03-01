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

Anatomic database:
- `oidm-maintain anatomic validate SOURCE_JSON` – Validate source JSON before building
- `oidm-maintain anatomic build --source FILE --output PATH` – Build DuckDB from source JSON
- `oidm-maintain anatomic publish PATH` – Upload to S3 and update manifest

FindingModel database:
- `oidm-maintain findingmodel build --source DIR --output PATH` – Build DuckDB index
- `oidm-maintain findingmodel publish PATH` – Upload to S3 and update manifest

## Key Patterns
- Embedding generation for semantic search
- DuckDB index building with HNSW/FTS
- SHA256 checksum + S3 manifest for pooch auto-download

## Schema Change Checklist
When modifying the `anatomic_locations` or `finding_models` table schema in `build.py`:
1. **No positional index updates needed** — both `AnatomicLocationIndex._build_location()` and `FindingModelIndex._fetch_index_entry()` use named dict access via `_execute_one`/`_execute_all` from `ReadOnlyDuckDBIndex`. New columns are automatically available by name; removed columns will raise `KeyError` only if code references them.
2. Rebuild the test fixture: `uv run python packages/oidm-maintenance/scripts/build_anatomic_test_fixture.py`
3. Run `task test` and verify all tests pass.

## Serena References
- `duckdb_architecture` – DuckDB conventions and hydration pipeline
