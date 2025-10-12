# Pending Fixes and Technical Debt

## Anatomic Location Search - Hardcoded Dimensions

**Issue**: `anatomic_location_search.py` has hardcoded embedding dimensions (512) instead of using config
- Line: `embeddings = await get_openai_embeddings([query], dimensions=512)`
- Should use: `Config().openai_embedding_dimensions`

**Fix**: Update after DuckDB Index migration complete (see consolidation plan in `index_duckdb_migration_decisions_2025` memory)

## Migration Script - Config Usage

**Issue**: `migrate_anatomic_to_duckdb.py` should use config for embedding dimensions
- Currently hardcoded to 512
- Should match pattern from Index implementation

**Fix**: Update after shared utilities created (see `duckdb-common-patterns.md`)

## Anatomic Location DuckDB Rebuild - Preserve Embeddings

**Issue**: Need to implement rebuild/update strategy for `anatomic_locations.duckdb` that preserves existing embeddings
- Currently: `migrate_anatomic_to_duckdb.py` regenerates all embeddings from scratch
- Problem: Expensive and slow to re-run embeddings for data that hasn't changed
- Needed: Hash-based detection (like Index will use) to only regenerate embeddings for changed anatomic locations

**Context**: 
- Should follow same drop/rebuild HNSW pattern as DuckDB Index migration
- Can reuse hash comparison logic from Index implementation
- Only call OpenAI API for new/changed anatomic location entries

**Fix**: After DuckDB Index migration complete, refactor anatomic location tooling to:
1. Use shared DuckDB utilities (from `duckdb-common-patterns.md`)
2. Implement hash-based change detection
3. Preserve existing embeddings for unchanged entries
4. Only regenerate embeddings when source data changes

---

## DuckDB Migration Script

### Issue: Uses hardcoded embedding configuration
**File**: `notebooks/migrate_anatomic_to_duckdb.py`

**Problem**:
- May have hardcoded model name and dimensions
- Should use config settings for consistency

**Action**: Review and update to use `settings.openai_embedding_model` and `settings.openai_embedding_dimensions`

**Priority**: Low (one-time migration script, already run)

---

## Future Refactoring

### Consolidate DuckDB common code
**Related to**: Index DuckDB migration

**Goal**: Extract shared DuckDB patterns into reusable utilities to avoid code duplication between:
- `src/findingmodel/tools/duckdb_search.py` (anatomic locations)
- `src/findingmodel/duckdb_index.py` (new finding model index)

**See**: `tasks/duckdb-common-patterns.md` for detailed analysis and plan

**Priority**: Medium (do during or after Index implementation)

---

## Remove Cohere references

**Related to**: Various search tools

**Goal**: Cohere isn't worth it, let's make sure we remove all code and the configuration optiosn