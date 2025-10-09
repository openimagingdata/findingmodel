# Index DuckDB Migration - Key Decisions

## Architecture Decisions

### Drop/Rebuild HNSW Strategy (2025-10-08)
**Decision**: Drop HNSW and FTS indexes before batch writes, rebuild after.
**Rationale**: 
- Avoids experimental HNSW persistence flag (safer, simpler)
- No corruption risk from unexpected shutdown
- Worst case: cancelled update, just run again
- Trade-off: Search unavailable for ~5 seconds during batch updates (acceptable - infrequent)

### No Foreign Key Constraints (2025-10-08)
**Decision**: Remove all FK constraints between tables.
**Rationale**:
- Simplifies drop/rebuild strategy
- Integrity enforced by application during rebuild process
- Denormalized tables always refreshed completely before recreating indexes
- Manual cleanup via _delete_denormalized_records() before mutations

### Separate Contributor Tables (2025-10-08)
**Decision**: Separate model_people and model_organizations junction tables (not polymorphic).
**Rationale**:
- Cleaner schema than polymorphic model_contributors
- Better query performance (no type discrimination needed)
- Simpler join logic

### Denormalized Tables (2025-10-08)
**Decision**: 5 denormalized tables (synonyms, tags, attributes, model_people, model_organizations).
**Rationale**:
- Eliminate joins for common queries (synonyms, tags, contributors, attributes)
- Fast exact-match synonym lookups
- Tag filtering support
- Attribute ID uniqueness validation

### Batch Directory Ingestion (2025-10-08)
**Decision**: update_from_directory now uses hash-based diffing with temp table + full outer join.
**Algorithm**:
1. Hash all *.fm.json files
2. Stage filename/hash pairs in temp table (tmp_directory_files)
3. Full outer join with finding_models on filename
4. Classify: added (new in temp), updated (hash differs), removed (not in temp)
5. Drop HNSW/FTS indexes
6. Execute batch deletes/updates/inserts via executemany
7. Rebuild HNSW/FTS indexes

**Rationale**:
- Single transaction for consistency
- Efficient batch operations (executemany)
- Per-file helpers remain but no longer drive ingestion
- SQL-driven diff detection (faster than Python loops)

### Read-Only Mode by Default (2025-10-08)
**Decision**: DuckDB connection is read-only by default, explicit writable mode for updates.
**Rationale**:
- 99% of usage is search (read-only)
- Prevents accidental writes
- Single writer pattern (DuckDB limitation vs multi-writer MongoDB)
- Explicit intent for mutations

### Semantic Search Always Enabled (2025-10-08)
**Decision**: Embeddings NOT NULL, use_semantic parameter removed, OpenAI key REQUIRED.
**Rationale**:
- Semantic search is a core feature, not optional
- Simplifies API (fewer parameters)
- Consistent with anatomic location search pattern
- FLOAT[512] embeddings from text-embedding-3-small (same as anatomic locations)

## Two-Phase Approach (2025-10-09)

### Phase 1: Technology Migration (Current)
**Decision**: Ship monolithic DuckDB implementation first, defer architectural refactoring.
**Rationale**:
1. **Validate technology choice** - Ensure DuckDB hybrid search works for real use cases
2. **Get user feedback** - Learn if semantic search/hybrid weights/tag filtering are valuable
3. **Reduce risk** - Don't combine tech migration + architectural refactoring in one PR
4. **Ship value faster** - Better search capability available sooner
5. **Refactor with confidence** - Once working end-to-end, know what abstractions make sense

**Accepts**: Monolithic code (1,319 lines, 47 methods) - same "god object" pattern as MongoDB Index

### Phase 2: Architectural Refactoring (Future)
**Decision**: Decompose BOTH MongoDB and DuckDB implementations into 5 focused classes.
**Scope**: Repository (protocol-based), Validator, FileManager, SearchEngine, Facade
**Rationale**:
- Applies to both backends (shared abstractions)
- Can learn from Phase 1 experience
- Should not block shipping Phase 1

### Optional: Basic Decomposition in Phase 1
**Decision**: OPTIONAL 2-class split if feels low-risk.
**Options**:
- Option A: Read/Write split (400 lines reader + 900 lines writer)
- Option B: Search/Data split (500 lines search + 800 lines data + 200 lines facade)

**Guidance**: Skip if too risky, defer to Phase 2.

## Schema Decisions

### 8 Tables
1. **finding_models** - main metadata with embeddings (NOT NULL)
2. **people** - normalized person master data
3. **organizations** - normalized organization master data
4. **model_people** - denormalized model→person links (junction)
5. **model_organizations** - denormalized model→organization links (junction)
6. **synonyms** - denormalized synonym storage
7. **attributes** - denormalized attribute storage
8. **tags** - denormalized tag storage

### Indexes
- **FTS**: BM25 on search_text (name + description + synonyms concatenated)
- **HNSW**: Vector similarity on embedding (L2 distance, convert to cosine)
- **Lookup indexes**: synonyms(synonym), tags(tag), model_people(person_id), etc.

## Hybrid Search Strategy

**Algorithm**:
1. **Exact match check** - ID/name/synonym exact match → return immediately (score=1.0)
2. **Tag filtering** (optional) - Get candidate oifm_ids from tags table
3. **FTS search** - BM25 scoring on search_text, filtered by tags
4. **Semantic search** - HNSW approximate NN on embeddings, filtered by tags
5. **Fusion** - Weighted: 0.3 * normalized_bm25 + 0.7 * cosine_similarity

**L2→Cosine conversion**: `cosine_sim = 1 - (l2_distance / 2)` (DuckDB uses L2, not cosine metric)

## Configuration Changes

**Remove**: All MongoDB settings (uri, db, collections, use_atlas_search)
**Add**: 
- duckdb_index_path: Path = Path("data/finding_models.duckdb")
- hybrid_search_bm25_weight: float = 0.3
- hybrid_search_semantic_weight: float = 0.7

**Reuse**: openai_embedding_model, openai_embedding_dimensions (already in config)
