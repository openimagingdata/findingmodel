# Pending Fixes and Technical Debt

## Anatomic Location DuckDB Rebuild - Preserve Embeddings

**Issue**: Need to implement rebuild/update strategy for `anatomic_locations.duckdb` that preserves existing embeddings

**Current Behavior**:
- `fm anatomic build` regenerates all embeddings from scratch
- Expensive and slow to re-run embeddings for data that hasn't changed
- Every rebuild makes OpenAI API calls for all anatomic location entries

**Desired Behavior**:
- Hash-based detection (like Index implementation uses) to identify changed entries
- Only regenerate embeddings for new/changed anatomic location entries
- Preserve existing embeddings for unchanged entries
- Follow same drop/rebuild HNSW pattern as DuckDB Index

**Implementation Notes**:
- Can reuse hash comparison logic from Index implementation
- Should follow patterns in `src/findingmodel/index.py` for batch operations
- Uses existing utilities from `src/findingmodel/tools/duckdb_utils.py`

**Related Files**:
- `src/findingmodel/anatomic_migration.py` - Migration logic to update
- `src/findingmodel/cli.py` - CLI commands for anatomic build
- `src/findingmodel/index.py` - Reference for hash-based diffing pattern

**Priority**: Medium - Current rebuild works but is inefficient

---

## Future Considerations

### Search Type Parameter
Add `search_type` parameter to Index search methods to support graceful degradation when OpenAI API key unavailable. See [database_distribution_enhancements.md](database_distribution_enhancements.md) Section 2 for details.

**Priority**: Low - Deferred to v0.5.1+

### Remote DuckDB Access
Add option to open DuckDB files directly via URL without downloading using httpfs extension. See [database_distribution_enhancements.md](database_distribution_enhancements.md) Section 3 for details.

**Priority**: Low - Deferred to v0.5.1+
