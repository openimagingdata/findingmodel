# Anatomic Location Search Cleanup

**Date**: 2025-10-12
**Status**: Ready for Implementation

## Goal

Refactor anatomic location code to eliminate duplication and move migration from notebooks to CLI commands.

**Success Criteria**:
- CLI commands for building/managing anatomic location database (`fm anatomic build/validate/stats`)
- Search client uses common DuckDB utilities (no duplicated connection/embedding/RRF logic)
- No hardcoded config values
- Migration script removed from `notebooks/`
- Tests pass with >90% coverage for new code

## Implementation

### Phase 1: Refactor Search Client

**File**: `src/findingmodel/tools/duckdb_search.py`

Fix hardcoded config and use common utilities:
- Line 490: Change `dimensions=512` to `dimensions=settings.openai_embedding_dimensions`
- `__aenter__()`: Use `setup_duckdb_connection(self.db_path, read_only=True)` from duckdb_utils
- `_get_embedding()`: Use `get_embedding_for_duckdb()` from duckdb_utils
- `_apply_rrf_fusion()`: Use `rrf_fusion()` from duckdb_utils

**Validation**: Existing tests pass without modification.

---

### Phase 2: Create CLI Commands

**Files**:
- `src/findingmodel/cli.py` - Add `anatomic` command group
- `src/findingmodel/anatomic_migration.py` - New module with migration functions

**Commands**:
```bash
fm anatomic build [--source URL|FILE] [--output PATH] [--force]
fm anatomic validate [--source URL|FILE]
fm anatomic stats [--db-path PATH]
```

**Migration module functions** (extract from `notebooks/migrate_anatomic_to_duckdb.py`):
- `create_searchable_text(record)` - Combine description/synonyms/definition
- `determine_sided(record)` - Return "generic", "left", "right", "unsided", or "nonlateral"
- `async load_anatomic_data(source)` - Load from URL (httpx) or file
- `validate_anatomic_record(record)` - Check required fields
- `async create_anatomic_database(db_path, records, client)` - Build database using common utilities

**Use common utilities**:
- `setup_duckdb_connection()` for connection management
- `batch_embeddings_for_duckdb()` for embedding generation
- Standard index creation patterns

**Dependencies**: Requires `httpx` for async URL downloads.

**Validation**: Can build database from local file and URL, validate detects errors, stats shows correct info.

---

### Phase 3: Add Tests

**New files**:
- `test/test_anatomic_migration.py` - Test migration functions
- `test/test_cli_anatomic.py` - Test CLI commands

**Coverage**:
- Migration functions with valid/invalid data
- CLI commands with mocked OpenAI/httpx
- Search client uses common utilities (verify mocks called)
- Integration test with `test/data/anatomic_locations_test.json` (100 records)

**Test data**: Use existing `test/data/anatomic_locations_test.json` (already created).

**Validation**: All tests pass, coverage >90% for new code.

---

### Phase 4: Cleanup

- Delete `notebooks/migrate_anatomic_to_duckdb.py`
- Update README.md CLI section with `fm anatomic` commands
- Update CHANGELOG.md with user-facing changes
- Update Serena memories (anatomic_location_search_implementation, suggested_commands)

**Validation**: No broken references, docs accurate.

---

## Key Files

**Existing**:
- `notebooks/migrate_anatomic_to_duckdb.py` - 458 lines to migrate
- `src/findingmodel/tools/duckdb_search.py` - Search client to refactor
- `src/findingmodel/tools/duckdb_utils.py` - Common utilities to use
- `src/findingmodel/cli.py` - CLI framework

**New**:
- `src/findingmodel/anatomic_migration.py` - Extracted migration logic
- `test/test_anatomic_migration.py` - Migration tests
- `test/test_cli_anatomic.py` - CLI tests

**Test data**:
- `test/data/anatomic_locations_test.json` - 100 records for testing

---

## Notes

- Each phase can be tested independently
- Phase 1 is low-risk (internal refactoring only)
- Phase 2 adds new functionality without breaking existing
- Phases can be implemented sequentially or Phase 1 can run in parallel with Phase 2
- Standard git workflow for rollback if needed
