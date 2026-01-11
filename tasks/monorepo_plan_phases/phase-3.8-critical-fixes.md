# Phase 3.8: Complete Read-Only Migration for findingmodel

## Overview

Phase 3.7 created the `oidm-maintenance` package with database build/publish functionality. This phase completes the migration by:

1. Stripping `DuckDBIndex` to read-only (query/search only)
2. Removing write-related CLI commands from findingmodel
3. Deleting `db_publish.py` from findingmodel (now lives in oidm-maintenance)
4. Fixing tests to use the pre-built test fixture

**Guiding principle:** findingmodel is a READ-ONLY package. All database creation, modification, and publishing is handled by oidm-maintenance.

## Pre-Conditions

- Phase 3.7 complete (oidm-maintenance package exists with build/publish CLI)
- All oidm-maintenance tests pass
- Pre-built test fixture exists: `packages/findingmodel/tests/data/test_index.duckdb`

---

## Sub-Phase 3.8.1: Strip DuckDBIndex to Read-Only

**Goal:** Remove all write methods from `DuckDBIndex`. It should only support search, query, and read operations.

**File:** `packages/findingmodel/src/findingmodel/index.py`

### Remove these methods entirely:

Database setup and schema:
- `setup()` - creates tables and indexes
- `_create_search_indexes()` - creates HNSW/FTS indexes
- `_load_base_contributors()` - loads initial contributor data

Write operations:
- `add_or_update_entry_from_file()` - adds/updates entries from files
- `update_from_directory()` - batch sync from directory
- `remove_entry()` - deletes entries

Validation (write-time):
- `_validate_model()` - validates during writes

Internal write helpers:
- `_populate_finding_models()` - inserts model data
- `_populate_finding_model_json()` - inserts JSON data
- `_populate_denormalized_tables()` - populates synonym/tag/attribute tables
- `_populate_contributors()` - populates contributor tables
- `_clear_denormalized_tables()` - clears tables before rebuild
- `_drop_search_indexes()` - drops indexes before write
- `_rebuild_search_indexes()` - rebuilds indexes after write
- `_load_models_metadata()` - loads metadata for validation
- Any other method that writes to the database

### Modify `__init__`:

Remove the `read_only` parameter entirely. The connection is always read-only.

```python
# Change from:
def __init__(self, db_path: str | Path | None = None, *, read_only: bool = True) -> None:
    ...
    self.read_only = read_only

# To:
def __init__(self, db_path: str | Path | None = None) -> None:
    ...
    # No read_only attribute - always read-only
```

### Modify `_ensure_connection`:

Always open in read-only mode:

```python
# Change from:
self.conn = setup_duckdb_connection(self.db_path, read_only=self.read_only)

# To:
self.conn = setup_duckdb_connection(self.db_path, read_only=True)
```

### Methods to KEEP (read-only operations):

Context management:
- `__init__`, `__aenter__`, `__aexit__`, `close`
- `_ensure_connection` (modified as above)

Search and query:
- `search()`, `search_batch()` - hybrid search
- `get()`, `get_full()` - retrieve by ID/name/slug
- `all()` - list with pagination
- `contains()` - check existence
- `count()`, `count_people()`, `count_organizations()` - counts
- `count_search_results()` - count matching search

Contributor lookups:
- `get_person()`, `get_organization()`
- `get_people()`, `get_organizations()`

ID generation (reads existing IDs to avoid collisions):
- `generate_model_id()`, `generate_attribute_id()`
- `_get_existing_model_ids()`, `_get_existing_attribute_ids()`
- `_build_oifm_id_cache()`, `_build_oifma_id_cache()`

Internal read helpers:
- `_ensure_openai_client()` - for embeddings during search
- `_get_embedding()` - generates query embeddings
- Any method that only reads

### Acceptance Criteria

```bash
# Verify no write methods remain
grep -n "def setup\|def add_or_update\|def update_from_directory\|def remove_entry" \
  packages/findingmodel/src/findingmodel/index.py
# Should return empty

# Verify read_only parameter removed from __init__
grep -n "read_only" packages/findingmodel/src/findingmodel/index.py
# Should only show setup_duckdb_connection call with read_only=True

# Verify module imports/runs
uv run python -c "from findingmodel.index import DuckDBIndex; print('OK')"
```

---

## Sub-Phase 3.8.2: Remove Write CLI Commands from findingmodel

**Goal:** Remove CLI commands that called the now-removed write methods.

**File:** `packages/findingmodel/src/findingmodel/cli.py`

### Remove entirely:

Search for and delete these command functions:
- `build` command - calls removed `setup()`, `update_from_directory()`
- `update` command - calls removed `update_from_directory()`
- `validate` command - calls removed `setup()`, `_validate_model()`
- `_validate_single_file` helper - used by validate command

Keep the `@index` group decorator but it will only contain `stats`.

### Fix `stats` command:

The stats command currently creates a database if it doesn't exist. Change it to error instead:

```python
# Remove database creation logic. Replace with:
if not db_path.exists():
    console.print(f"[bold red]Error: Database not found: {db_path}[/bold red]")
    console.print("[yellow]Hint: Use 'oidm-maintain findingmodel build' to create a database.[/yellow]")
    raise SystemExit(1)
```

Also remove the `read_only` parameter from DuckDBIndex instantiation:

```python
# Change from:
async with DuckDBIndex(db_path=db_path, read_only=True) as idx:

# To:
async with DuckDBIndex(db_path=db_path) as idx:
```

### Acceptance Criteria

```bash
# Verify commands removed
uv run findingmodel index --help
# Should show only: stats

# Verify no read_only=False in CLI
grep -n "read_only=False" packages/findingmodel/src/findingmodel/cli.py
# Should return empty

# Verify stats command works with existing database
uv run findingmodel index stats --index packages/findingmodel/tests/data/test_index.duckdb

# Verify stats command errors on missing database
uv run findingmodel index stats --index /nonexistent/path.duckdb
# Should print error message, not create database
```

---

## Sub-Phase 3.8.3: Remove read_only Parameter from Tools

**Goal:** Update tool files that pass `read_only=True` to DuckDBIndex (parameter no longer exists).

**Files to update:**

1. `packages/findingmodel/src/findingmodel/tools/finding_enrichment.py`
   - Find: `DuckDBIndex(read_only=True)`
   - Replace with: `DuckDBIndex()`

2. `packages/findingmodel/src/findingmodel/tools/finding_enrichment_agentic.py`
   - Find: `DuckDBIndex(read_only=True)`
   - Replace with: `DuckDBIndex()`

### Acceptance Criteria

```bash
# Verify no read_only parameter usage in tools
grep -rn "read_only" packages/findingmodel/src/findingmodel/tools/
# Should return empty (or only comments)

# Verify no read_only in entire findingmodel src except internal duckdb setup
grep -rn "read_only" packages/findingmodel/src/findingmodel/
# Should only show setup_duckdb_connection calls in index.py and duckdb_search.py
```

---

## Sub-Phase 3.8.4: Delete db_publish.py from findingmodel

**Goal:** Remove `db_publish.py` from findingmodel. This functionality now lives in `oidm-maintenance`.

**File to delete:** `packages/findingmodel/src/findingmodel/db_publish.py`

### Also update:

1. Remove any imports of `db_publish` in `packages/findingmodel/src/findingmodel/__init__.py`
2. Remove any CLI commands that reference `db_publish` (likely already handled in 3.8.2)

### Acceptance Criteria

```bash
# Verify file deleted
ls packages/findingmodel/src/findingmodel/db_publish.py
# Should return "No such file or directory"

# Verify no imports of db_publish
grep -rn "db_publish" packages/findingmodel/src/
# Should return empty

# Verify findingmodel still imports cleanly
uv run python -c "import findingmodel; print('OK')"
```

---

## Sub-Phase 3.8.5: Fix Tests

**Goal:** Update tests to work with read-only DuckDBIndex using the pre-built test fixture.

**Note:** Sub-phase 3.8.5a (create test fixture) is already complete. The file `packages/findingmodel/tests/data/test_index.duckdb` exists.

### 3.8.5a: Test Fixture ✅ COMPLETE

The pre-built test database and build script already exist:
- `packages/findingmodel/tests/data/test_index.duckdb`
- `packages/oidm-maintenance/scripts/build_test_fixtures.py`

### 3.8.5b: Update findingmodel Test Fixtures

**File:** `packages/findingmodel/tests/test_duckdb_index.py`

Replace write-based fixtures with simple read-only fixtures:

```python
@pytest.fixture(scope="session")
def prebuilt_db_path() -> Path:
    """Path to pre-built test database (committed to repo)."""
    db_path = Path(__file__).parent / "data" / "test_index.duckdb"
    if not db_path.exists():
        pytest.skip(
            "Pre-built test database not found. "
            "Run: uv run python packages/oidm-maintenance/scripts/build_test_fixtures.py"
        )
    return db_path


@pytest.fixture
async def index(prebuilt_db_path: Path) -> AsyncGenerator[DuckDBIndex, None]:
    """Load the pre-built test database (read-only)."""
    async with DuckDBIndex(prebuilt_db_path) as idx:
        yield idx
```

Remove the complex session-scoped fixtures that used monkeypatching for write operations.

### 3.8.5c: Delete Write Operation Tests from findingmodel

Tests that exercise write operations should be deleted from findingmodel. These test the build functionality which is now in oidm-maintenance.

Delete tests matching these patterns:
- `test_add_*` - tests for adding entries
- `test_update_*` - tests for updating entries
- `test_remove_*` - tests for removing entries
- `test_setup_*` - tests for setup/schema creation
- `test_batch_*` - tests for batch operations
- `test_validate_*` and `test_duplicate_*` - write-time validation tests
- `test_denormalized_*` - tests for populating denormalized tables
- `test_read_only_mode_*` - tests for read_only parameter (no longer exists)

### 3.8.5d: Keep and Fix Read Operation Tests

Tests that exercise read operations should remain and use the new `index` fixture:

Keep tests matching these patterns:
- `test_search_*` - search functionality
- `test_count_*` - count operations
- `test_get_*` - retrieval operations
- `test_all_*` - pagination/listing
- `test_contains_*` - existence checks
- `test_generate_*_id_*` - ID generation (reads to check collisions)
- `test_populated_index_*` - tests using populated data

Update these tests to:
1. Use the `index` fixture instead of `session_populated_index` or `populated_index`
2. Remove any `read_only` parameter assertions

### 3.8.5e: Ensure oidm-maintenance Has Build Tests

**File:** `packages/oidm-maintenance/tests/test_findingmodel_build.py`

Verify this file has comprehensive tests for:
- Building database from source directory
- Incremental builds (add/update/remove detection)
- Index creation (HNSW, FTS)
- Validation during build
- Error handling

If tests are missing, add them. These replace the write operation tests removed from findingmodel.

### Acceptance Criteria

```bash
# All findingmodel tests pass
uv run pytest packages/findingmodel/tests/ -v

# All oidm-maintenance tests pass
uv run pytest packages/oidm-maintenance/tests/ -v

# No skipped tests with "write" related messages
uv run pytest packages/findingmodel/tests/test_duckdb_index.py -v 2>&1 | grep -i skip
# Should show minimal/no skips

# Verify no references to removed methods in tests
grep -n "\.setup()\|\.add_or_update\|\.update_from_directory\|\.remove_entry" \
  packages/findingmodel/tests/test_duckdb_index.py
# Should return empty
```

---

## Sub-Phase 3.8.6: Verification

**Goal:** Confirm all changes work together.

### Run full verification:

```bash
# 1. Lint and type check
task check

# 2. Run all tests
task test

# 3. Verify CLI help shows correct commands
uv run findingmodel --help
uv run findingmodel index --help
# Should show: stats (only)

uv run oidm-maintain --help
uv run oidm-maintain findingmodel --help
# Should show: build, publish

# 4. Verify no read_only parameter leaks
grep -rn "read_only" packages/findingmodel/src/findingmodel/
# Should only show internal setup_duckdb_connection usage

# 5. Verify db_publish.py only in oidm-maintenance
find packages/ -name "db_publish.py" -o -name "*publish*.py" | grep -v oidm-maintenance
# Should return empty (no publish code outside oidm-maintenance)

# 6. Verify DuckDBIndex has no write methods
grep -n "def setup\|def add_or_update\|def update_from_directory\|def remove_entry\|def _populate\|def _create_search_indexes" \
  packages/findingmodel/src/findingmodel/index.py
# Should return empty

# 7. Test the actual workflow
# Build a test database with oidm-maintenance
uv run oidm-maintain findingmodel build \
  --source packages/findingmodel/tests/data/defs \
  --output /tmp/test_build.duckdb

# Query it with findingmodel (read-only)
uv run findingmodel index stats --index /tmp/test_build.duckdb
```

### Acceptance Criteria

- `task check` passes (no lint/type errors)
- `task test` passes (all tests green)
- findingmodel CLI only has read commands
- oidm-maintenance CLI has build/publish commands
- No `db_publish.py` in findingmodel
- No write methods in `DuckDBIndex`
- End-to-end workflow works: build with oidm-maintain, query with findingmodel

---

## Post-Conditions

After Phase 3.8:

- **findingmodel** is a read-only package:
  - `DuckDBIndex` only supports search/query/read operations
  - CLI only has `stats` command under `index` group
  - No `db_publish.py`
  - Tests use pre-built fixture

- **oidm-maintenance** handles all write operations:
  - Database building via `build_findingmodel_database()`
  - Database publishing via `publish_findingmodel_database()`
  - CLI commands: `oidm-maintain findingmodel build/publish`
  - Comprehensive build tests

---

## Files Summary

| File | Action |
|------|--------|
| `packages/findingmodel/src/findingmodel/index.py` | Remove write methods, remove read_only parameter |
| `packages/findingmodel/src/findingmodel/cli.py` | Remove build/update/validate commands, fix stats |
| `packages/findingmodel/src/findingmodel/db_publish.py` | **DELETE** |
| `packages/findingmodel/src/findingmodel/tools/finding_enrichment.py` | Remove read_only parameter |
| `packages/findingmodel/src/findingmodel/tools/finding_enrichment_agentic.py` | Remove read_only parameter |
| `packages/findingmodel/tests/test_duckdb_index.py` | Update fixtures, remove write tests, keep read tests |
| `packages/findingmodel/tests/conftest.py` | May need fixture updates |
