# Drive to v0.4.0 Release

**Goal**: Ship v0.4.0 with DuckDB-only index (no MongoDB dependency)

**Status**: In Progress
**Created**: 2025-10-14

## Overview

v0.4.0 removes MongoDB dependency for the finding model index, replacing it with DuckDB + remote download via Pooch. This follows the same pattern already working for anatomic location search.

## Phase 1: DuckDB Utilities Extraction

**Goal**: Consolidate duplicated index creation/management code into shared utilities

### Current Duplication
- `anatomic_migration.py::_create_indexes()` - Creates FTS + HNSW indexes for anatomic DB
- `duckdb_index.py::_create_search_indexes()` - Creates FTS + HNSW indexes for finding models
- `duckdb_index.py::_drop_search_indexes()` - Drops both index types

Both are **offline admin build tools** (not end-user runtime), so contexts are identical.

### Implementation Tasks
1. Add to `src/findingmodel/tools/duckdb_utils.py`:
   ```python
   def create_fts_index(
       conn: duckdb.DuckDBPyConnection,
       table: str,
       id_column: str,
       *text_columns: str,
       stemmer: str = "porter",
       stopwords: str = "english",
       overwrite: bool = True,
   ) -> None:
       """Create FTS index on text columns."""

   def create_hnsw_index(
       conn: duckdb.DuckDBPyConnection,
       table: str,
       column: str,
       index_name: str | None = None,
       metric: str = "cosine",
       ef_construction: int = 128,
       ef_search: int = 64,
       m: int = 16,
   ) -> None:
       """Create HNSW vector index."""

   def drop_search_indexes(
       conn: duckdb.DuckDBPyConnection,
       table: str,
       hnsw_index_name: str | None = None,
   ) -> None:
       """Drop HNSW and FTS indexes for table."""
   ```

2. Update `anatomic_migration.py::_create_indexes()` to use utilities
3. Update `duckdb_index.py::_create_search_indexes()` to use utilities
4. Update `duckdb_index.py::_drop_search_indexes()` to use utilities
5. Add tests for new utilities
6. Verify existing tests still pass

**Success Criteria**:
- All tests pass (67 DuckDB index tests + anatomic tests)
- No code duplication between anatomic and index builders
- Same behavior as before extraction

## Phase 2: MongoDB → DuckDB Swap

**Goal**: Make DuckDB the default index, deprecate MongoDB

### File Operations
1. **Backup MongoDB implementation**:
   ```bash
   git mv src/findingmodel/index.py src/findingmodel/mongodb_index.py
   ```

2. **Promote DuckDB implementation**:
   ```bash
   git mv src/findingmodel/duckdb_index.py src/findingmodel/index.py
   ```

3. **Update imports in `src/findingmodel/__init__.py`**:
   ```python
   # OLD:
   from findingmodel.index import Index

   # NEW:
   from findingmodel.index import DuckDBIndex as Index

   # Backward compat (deprecated):
   from findingmodel.mongodb_index import Index as MongoDBIndex
   ```

### Config Updates
4. **Update `config.py`**:
   - Comment out MongoDB settings (keep for reference)
   - Add deprecation notice
   - Ensure DuckDB settings are prominent
   - Set default remote URL/hash for finding models DB (like anatomic locations)

### Dependency Updates
5. **Update `pyproject.toml`**:
   ```toml
   # Move MongoDB from dependencies to optional
   [project]
   dependencies = [
       # ... (remove motor)
   ]

   [project.optional-dependencies]
   mongodb = ["motor>=3.7.1"]  # Deprecated, use DuckDB
   ```

### Testing
6. **Verify integration**:
   - Run full test suite: `task test`
   - Run callout tests: `task test-full`
   - Test CLI commands:
     - `index build`
     - `index update`
     - `index validate`
     - `index stats`
     - `anatomic build`
     - `anatomic stats`
   - Test import patterns in notebooks

**Success Criteria**:
- All tests pass (no regressions)
- CLI commands work with DuckDB
- MongoDB still accessible via `mongodb_index.py` (deprecated)
- Clear migration path documented

## Phase 3: Documentation

**Goal**: Update all docs for v0.4.0

### Updates Needed
1. **README.md**:
   - Update "Installation" section (no MongoDB required)
   - Document Pooch auto-download
   - Add "Database Management" section
   - Note MongoDB deprecation

2. **CHANGELOG.md**:
   ```markdown
   ## [0.4.0] - 2025-01-XX

   ### Breaking Changes
   - MongoDB index replaced with DuckDB (auto-downloads via Pooch)
   - MongoDB dependencies now optional (`pip install findingmodel[mongodb]`)

   ### Added
   - CLI commands for index management
   - Shared DuckDB utilities in `tools/duckdb_utils.py`
   - Remote database download support for finding models index

   ### Deprecated
   - `mongodb_index.py` - use `index.py` (DuckDB) instead
   ```

3. **Migration Guide** (`docs/mongodb-to-duckdb.md`):
   - Why the change
   - What users need to do (hint: nothing!)
   - How to keep using MongoDB if needed
   - Performance comparison

4. **Update `.env.sample`**:
   - Remove MongoDB connection string
   - Add remote DB URL/hash examples (optional overrides)

**Success Criteria**:
- README accurately describes v0.4.0 setup
- CHANGELOG complete
- Migration guide helpful for existing users

## Release Checklist

Before tagging v0.4.0:

- [ ] All Phase 1 tasks complete (utilities extraction)
- [ ] All Phase 2 tasks complete (MongoDB swap)
- [ ] All Phase 3 tasks complete (documentation)
- [ ] All tests passing (`task test`)
- [ ] Callout tests passing (`task test-full`)
- [ ] CLI commands tested
- [ ] Build succeeds (`task build`)
- [ ] Code formatted (`task check`)
- [ ] Sample finding models DB uploaded to hosting
- [ ] Version bumped in `pyproject.toml`
- [ ] Git tag created: `git tag v0.4.0`
- [ ] Release published on GitHub

## Post-Release

- [ ] Update Serena memory: `project_state_january_2025.md`
- [ ] Announce in discussions/issues
- [ ] Update any external documentation
- [ ] Consider blog post about MongoDB → DuckDB migration

## Notes

- **YAGNI principle**: Only implement what's needed for v0.4.0
- **Backward compatibility**: Keep MongoDB accessible but deprecated
- **Test coverage**: Don't break existing functionality
- **Documentation**: Clear migration path for users
