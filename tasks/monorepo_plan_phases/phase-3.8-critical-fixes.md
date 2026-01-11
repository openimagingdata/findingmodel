# Phase 3.8: Critical Fixes for oidm-maintenance Migration

## Overview

Phase 3.7 created the `oidm-maintenance` package but left breaking changes in the codebase. This phase fixes those issues to restore a working state.

## Pre-Conditions

- Phase 3.7 complete (oidm-maintenance package exists)
- All oidm-maintenance tests pass
- All anatomic-locations tests pass

## Sub-Phases

---

### 3.8.1: Remove Broken CLI Commands from findingmodel

**Goal**: Remove CLI commands that call removed methods from `DuckDBIndex`.

**File**: `packages/findingmodel/src/findingmodel/cli.py`

**Remove entirely** (lines 194-346):
1. `build` command (lines 194-227) - calls `idx.setup()`, `idx.update_from_directory()`
2. `update` command (lines 230-263) - calls `idx.update_from_directory()`
3. `_validate_single_file` helper function (lines 266-289) - calls `idx._validate_model()`, `idx.add_or_update_entry_from_file()`
4. `validate` command (lines 292-346) - calls `idx.setup()`, uses `_validate_single_file`

**Keep but fix**:
- `stats` command (handled in 3.8.2)
- `index` group decorator (keep the group, it will only have `stats`)

**Acceptance Criteria**:
- Commands `build`, `update`, `validate` removed from CLI
- Running `findingmodel index --help` shows only `stats` subcommand
- No `read_only=False` usages remain in cli.py
- `task check` passes for cli.py

---

### 3.8.2: Fix stats Command

**Goal**: Make `stats` command work with read-only `DuckDBIndex`.

**File**: `packages/findingmodel/src/findingmodel/cli.py`

**Changes**:
1. **Remove database creation logic** (lines 362-365):
   ```python
   # REMOVE this block:
   if not db_path.exists():
       async with DuckDBIndex(db_path=db_path, read_only=False) as temp_idx:
           await temp_idx.setup()
   ```
   Replace with error if database doesn't exist:
   ```python
   if not db_path.exists():
       console.print(f"[bold red]Database not found: {db_path}")
       console.print("[yellow]Use 'oidm-maintain findingmodel build' to create a database.")
       sys.exit(1)
   ```

2. **Remove read_only parameter** (line 372):
   ```python
   # Change from:
   async with DuckDBIndex(db_path=db_path, read_only=True) as idx:
   # To:
   async with DuckDBIndex(db_path=db_path) as idx:
   ```

**Acceptance Criteria**:
- `stats` command works with existing databases
- `stats` command gives helpful error for missing databases
- No `read_only` parameter usage in stats command
- `task check` passes

---

### 3.8.3: Fix read_only Parameter in Tools

**Goal**: Remove `read_only=True` parameter from tool files (DuckDBIndex is now always read-only).

**Files and changes**:

1. **`packages/findingmodel/src/findingmodel/tools/finding_enrichment.py`**
   - Line 770: Change `DuckDBIndex(read_only=True)` to `DuckDBIndex()`
   - Line 882: Change `DuckDBIndex(read_only=True)` to `DuckDBIndex()`

2. **`packages/findingmodel/src/findingmodel/tools/finding_enrichment_agentic.py`**
   - Line 244: Change `DuckDBIndex(read_only=True)` to `DuckDBIndex()`

**Acceptance Criteria**:
- No `read_only` parameter usages in tools/ directory
- `task check` passes for modified files
- `grep -r "read_only" packages/findingmodel/src/` shows only internal usages (setup_duckdb_connection)

---

### 3.8.4: Make findingmodel Build Function Async

**Goal**: Align API consistency - both build functions should be async.

**File**: `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`

**Changes**:
1. Change function signature from:
   ```python
   def build_findingmodel_database(
       source_dir: Path,
       output_path: Path,
       generate_embeddings: bool = True,
   ) -> Path:
   ```
   To:
   ```python
   async def build_findingmodel_database(
       source_dir: Path,
       output_path: Path,
       generate_embeddings: bool = True,
   ) -> Path:
   ```

2. Update any synchronous I/O operations to be properly awaited if needed (review function body)

3. **Update CLI caller** in `packages/oidm-maintenance/src/oidm_maintenance/cli.py`:
   - The `build` command under `findingmodel` group needs to use `asyncio.run()` like anatomic does

4. **Update tests** if any call the build function synchronously

**Acceptance Criteria**:
- `build_findingmodel_database` is async
- Both build functions have consistent signatures
- CLI works correctly with async function
- All oidm-maintenance tests pass

---

### 3.8.5: Migrate Tests from findingmodel to oidm-maintenance

**Goal**: Move the 104 skipped tests from `test_duckdb_index.py` to oidm-maintenance, properly adapted to the new build architecture.

**Source file**: `packages/findingmodel/tests/test_duckdb_index.py`

#### Test Categories and Migration Strategy

| Category | Count | Destination | Notes |
|----------|-------|-------------|-------|
| Write operations | ~20 | oidm-maintenance | `test_add_*`, `test_update_*`, `test_remove_*` |
| Validation | ~4 | oidm-maintenance | `test_validate_*`, `test_duplicate_*` |
| Index setup | ~3 | oidm-maintenance | `test_setup_creates_*`, `test_write_operations_rebuild_*` |
| Search (read-only) | ~25 | Keep in findingmodel | `test_search_*`, `test_count_search_*` |
| CRUD reads | ~15 | Keep in findingmodel | `test_get_*`, `test_count_*`, `test_all_*` |
| ID generation | ~30 | Keep in findingmodel | `test_generate_*_id_*`, `test_add_ids_*`, `test_finalize_*` |

#### 3.8.5a: Create Pre-Built Test Database Fixture

**Goal**: Build a test database ONCE and commit it, so tests use it directly without rebuilding.

**New fixture file**: `packages/findingmodel/tests/data/test_index.duckdb`

**Build process** (run once, commit result):
```bash
# Build test database from sample data with mocked embeddings
uv run oidm-maintain findingmodel build \
    packages/findingmodel/tests/data/defs \
    --output packages/findingmodel/tests/data/test_index.duckdb
```

Note: The build will need mocked embeddings (deterministic hash-based) to avoid API calls. Create a maintenance script or CLI flag for test database generation.

**New file**: `packages/oidm-maintenance/scripts/build_test_fixtures.py`
```python
"""Build test database fixtures with mocked embeddings."""
async def build_test_findingmodel_db():
    """Build findingmodel test database with deterministic embeddings."""
    source_dir = Path("packages/findingmodel/tests/data/defs")
    output_path = Path("packages/findingmodel/tests/data/test_index.duckdb")

    # Use deterministic hash-based embeddings (no API calls)
    with patch_embeddings_deterministic():
        await build_findingmodel_database(source_dir, output_path, generate_embeddings=True)
```

**Test fixture** (simple, just loads the committed database):
```python
@pytest.fixture(scope="session")
def prebuilt_db_path() -> Path:
    """Path to pre-built test database (committed to repo)."""
    return Path(__file__).parent / "data" / "test_index.duckdb"

@pytest.fixture
async def prebuilt_index(prebuilt_db_path: Path) -> AsyncGenerator[DuckDBIndex, None]:
    """Load the pre-built test database."""
    async with DuckDBIndex(prebuilt_db_path) as index:
        yield index
```

**Acceptance Criteria**:
- `test_index.duckdb` exists in `packages/findingmodel/tests/data/`
- Database contains all models from `tests/data/defs/`
- Database has HNSW and FTS indexes
- Build script exists for regenerating fixture when needed

#### 3.8.5b: Migrate Write Operation Tests

**Move to**: `packages/oidm-maintenance/tests/test_findingmodel_build.py`

Tests to migrate:
- `test_add_and_retrieve_model` → verify model added via build
- `test_add_already_existing_model_unchanged` → verify idempotent builds
- `test_add_new_model` → verify incremental builds
- `test_add_updated_model_file` → verify update detection
- `test_remove_not_found_model` → verify removal on rebuild
- `test_update_from_directory` → verify full sync behavior
- `test_update_from_directory_*` variants
- `test_batch_*` tests
- `test_remove_entry_*` tests
- `test_denormalized_*_table` tests (verify table population)

Adapt tests to use `build_findingmodel_database()` instead of direct `DuckDBIndex` write methods.

#### 3.8.5c: Migrate Validation Tests

**Move to**: `packages/oidm-maintenance/tests/test_findingmodel_build.py`

Tests to migrate:
- `test_validate_model_no_duplicates`
- `test_duplicate_oifm_id_fails_validation`
- `test_duplicate_name_fails_validation`
- `test_duplicate_attribute_id_fails_validation`

Note: Validation logic may need to be exposed in build module or tested via build failures.

#### 3.8.5d: Migrate Index Setup Tests

**Move to**: `packages/oidm-maintenance/tests/test_findingmodel_build.py`

Tests to migrate:
- `test_setup_creates_search_indexes`
- `test_write_operations_rebuild_search_indexes`
- `test_batch_update_rebuilds_indexes_once`

Verify that `build_findingmodel_database()` creates proper HNSW and FTS indexes.

#### 3.8.5e: Fix Read-Only Tests in findingmodel

**File**: `packages/findingmodel/tests/test_duckdb_index.py`

These tests should remain and use the **committed pre-built database** from 3.8.5a:

1. Remove the module-level `pytestmark = pytest.mark.skip(...)`
2. Remove/update fixtures that use write operations (`index`, `populated_index`, `session_populated_index`)
3. Use the simple fixture that loads the committed database:

```python
@pytest.fixture(scope="session")
def prebuilt_db_path() -> Path:
    """Path to pre-built test database (committed to repo)."""
    db_path = Path(__file__).parent / "data" / "test_index.duckdb"
    if not db_path.exists():
        pytest.skip("Pre-built test database not found. Run: uv run python -m oidm_maintenance.scripts.build_test_fixtures")
    return db_path

@pytest.fixture
async def prebuilt_index(prebuilt_db_path: Path) -> AsyncGenerator[DuckDBIndex, None]:
    """Load the pre-built test database (read-only)."""
    async with DuckDBIndex(prebuilt_db_path) as index:
        yield index
```

4. Update tests to use `prebuilt_index` instead of `session_populated_index`
5. Remove the complex session-scoped monkeypatch setup (no longer needed)

Tests to keep and fix:
- All `test_search_*` tests
- All `test_count_search_*` tests
- All `test_search_by_slug_*` tests
- `test_populated_index_count`, `test_populated_index_retrieval`
- `test_get_person`, `test_get_organization`, `test_get_people`, `test_get_organizations`
- `test_count_people`, `test_count_organizations`
- All `test_all_*` pagination/sorting tests
- `test_contains_method`, `test_count_method` (read portions)

#### 3.8.5f: Fix ID Generation Tests in findingmodel

**File**: `packages/findingmodel/tests/test_duckdb_index.py`

ID generation tests don't require write access - they use the index to check for collisions but don't write.

Tests to keep:
- All `test_generate_model_id_*` tests
- All `test_generate_attribute_id_*` tests
- All `test_add_ids_to_model_*` tests
- All `test_finalize_placeholder_*` tests

These need the `prebuilt_index` fixture from 3.8.5e.

#### 3.8.5g: Remove Obsolete Tests

Delete tests that are no longer applicable:
- `test_read_only_mode_blocks_writes` - DuckDBIndex is always read-only now
- Any tests for methods that no longer exist

**Acceptance Criteria**:
- No `pytestmark = pytest.mark.skip` in test_duckdb_index.py
- oidm-maintenance has comprehensive build tests (~30 tests)
- findingmodel has working read-only tests (~70 tests)
- `task test` in both packages passes
- No tests skipped with "Write operations moved" message

---

### 3.8.6: Verification

**Goal**: Confirm all fixes work together.

**Commands to run**:
```bash
# Check no read_only parameter leaks
grep -rn "read_only" packages/findingmodel/src/findingmodel/
# Should only show internal setup_duckdb_connection usages in index.py and duckdb_search.py

# Lint and type check
task check

# Run all tests
task test

# Run specific package tests
uv run pytest packages/oidm-maintenance/tests/ -v
uv run pytest packages/findingmodel/tests/test_duckdb_index.py -v

# Verify CLI help
uv run findingmodel index --help
# Should show only: stats

# Verify oidm-maintain CLI
uv run oidm-maintain --help
uv run oidm-maintain findingmodel --help
# Should show: build, publish
```

**Acceptance Criteria**:
- `task check` passes
- `task test` passes with no unexpected skips
- CLI commands work as expected
- No runtime errors when importing findingmodel modules
- Test coverage maintained (not reduced by migration)

---

## Post-Conditions

- findingmodel package is fully functional with read-only DuckDBIndex
- oidm-maintenance package handles all write operations
- No breaking API changes remain
- Test suite is complete with proper coverage:
  - oidm-maintenance: build, validation, index setup tests
  - findingmodel: read operations, search, ID generation tests
- Both build functions are async for consistency

## Files Modified

| File | Change |
|------|--------|
| `packages/findingmodel/src/findingmodel/cli.py` | Remove build/update/validate commands, fix stats |
| `packages/findingmodel/src/findingmodel/tools/finding_enrichment.py` | Remove read_only parameter |
| `packages/findingmodel/src/findingmodel/tools/finding_enrichment_agentic.py` | Remove read_only parameter |
| `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py` | Make async |
| `packages/oidm-maintenance/src/oidm_maintenance/cli.py` | Update for async build |
| `packages/oidm-maintenance/scripts/build_test_fixtures.py` | **NEW** - Script to build test database fixtures |
| `packages/oidm-maintenance/tests/test_findingmodel_build.py` | **NEW** - Migrated build tests |
| `packages/findingmodel/tests/data/test_index.duckdb` | **NEW** - Pre-built test database (committed) |
| `packages/findingmodel/tests/test_duckdb_index.py` | Remove skip, simplify fixtures, keep read-only tests |

## Estimated Test Distribution After Migration

| Package | Test File | Test Count | Coverage |
|---------|-----------|------------|----------|
| oidm-maintenance | test_findingmodel_build.py | ~30 | Build, validation, index setup |
| oidm-maintenance | test_config.py | 3 | Settings |
| oidm-maintenance | test_hashing.py | 3 | File hashing |
| oidm-maintenance | test_s3.py | 7 | S3 operations |
| findingmodel | test_duckdb_index.py | ~70 | Search, read ops, ID generation |
