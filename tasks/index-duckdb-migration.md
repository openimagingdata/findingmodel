# Index DuckDB Migration Plan - Phase 1

**Status**: ✅ Phase 1 Complete - Ready for Integration
**Branch**: `refactor-index`
**Phase**: 1 of 2 (Technology Migration)

## Goal
Migrate the Index class from MongoDB to DuckDB with hybrid FTS + vector search. Phase 1 focuses on getting a working, tested implementation shipped. Phase 2 (see [tasks/refactoring/01-index-decomposition.md](tasks/refactoring/01-index-decomposition.md)) will decompose into focused classes.

## Current State (2025-10-11)

### Phase 1 Implementation Complete ✅

**Implementation**: [src/findingmodel/duckdb_index.py](../src/findingmodel/duckdb_index.py) (1,350+ lines, 48 methods)

- ✅ Complete schema with 8 tables (finding_models + denormalized tables)
- ✅ Core CRUD operations (get, contains, add_or_update_entry_from_file, remove_entry)
- ✅ Hybrid search (exact match → FTS + semantic → weighted fusion)
- ✅ **search_batch()** with batch embedding optimization
- ✅ **Tag filtering** in search (supports single tag and multiple tag AND logic)
- ✅ **_validate_model()** implementation (OIFM ID, name, attribute conflicts)
- ✅ Batch directory ingestion with hash-based diffing
- ✅ Drop/rebuild HNSW strategy (no experimental persistence)
- ✅ Separate model_people and model_organizations tables
- ✅ DuckDB utilities extracted ([duckdb_utils.py](../src/findingmodel/tools/duckdb_utils.py))
- ✅ Read-only mode by default
- ✅ Enhanced logging for batch operations
- ✅ **Fixed validation bugs**: Skip validation for model updates (batch and single-file)
- ✅ **Pooch integration**: Optional remote download of pre-built DuckDB files with SHA256 verification (2025-10-11)

**Tests**: [test/test_duckdb_index.py](../test/test_duckdb_index.py) (1,500+ lines, 67 tests)

- ✅ All 34 MongoDB Index tests ported
- ✅ 33 DuckDB-specific tests added:
  - Denormalized table integrity (synonyms, tags, attributes, model_people, model_organizations)
  - HNSW/FTS index creation and rebuild
  - Tag filtering (single tag, multiple tags AND, nonexistent tags)
  - search_batch() with multiple queries
  - update_from_directory batch operations (add, update, delete, mixed, no-changes)
  - Read-only mode enforcement
  - Performance benchmarks (search latency, batch embedding, directory sync)
  - **Semantic search with pre-computed embeddings** (deterministic, no API calls)
  - **Semantic search with real OpenAI API** (@pytest.mark.callout for integration testing)
- ✅ **All 67 tests: 66 fast tests passing + 1 callout test** (100% success rate)
- ✅ Using existing fixtures from conftest.py (full_model, real_model, tmp_defs_path)
- ✅ Removed `@pytest.mark.slow` markers (not configured, caused warnings)
- ✅ **Fixture documentation created** (Serena memory: pytest_fixtures_reference_2025)

### What's Remaining for Integration ⏳

**Critical Next Steps** (Priority 1 - do before merging):
1. 🔴 **Update `pending-fixes.md`** - Remove completed Pooch items, move remaining issues to appropriate tracking
2. 🔴 **Anatomic location search alignment** - Apply same Pooch pattern to anatomic DB (already documented in memories, needs implementation verification)
3. 🔴 **Config cleanup** - Review hardcoded dimensions issue noted in `pending-fixes.md` (anatomic_location_search.py line uses 512 instead of config)

**Integration tasks** (Priority 2 - can be separate PR):
- ⏳ MongoDB Index still in use ([index.py](../src/findingmodel/index.py))
- ⏳ Config still has MongoDB settings
- ⏳ CLI commands not tested with DuckDB
- ⏳ Documentation updates (README, migration guide)

### Anatomic Location Search Status

**Current State**:
- ✅ Already using DuckDB (`duckdb_search.py`) with HNSW + FTS hybrid search
- ✅ Pooch download pattern documented in Serena memory (`anatomic_location_search_implementation`)
- ✅ Config fields exist (`duckdb_anatomic_path`, `remote_anatomic_db_url`, `remote_anatomic_db_hash`)
- ⚠️ **NEEDS VERIFICATION**: Ensure `__init__` in `duckdb_search.py` actually calls `ensure_db_file()` (may already be done)
- ⚠️ **NEEDS FIX**: Hardcoded `dimensions=512` in `anatomic_location_search.py` line 6 - should use `Config().openai_embedding_dimensions`

**Action Items**:
1. Check if `DuckDBOntologySearchClient.__init__` uses `ensure_db_file()` helper
2. Fix hardcoded dimensions in `anatomic_location_search.py`
3. Verify anatomic location tests still pass
4. Update `pending-fixes.md` to remove completed items

### Architectural Note
DuckDBIndex is currently **monolithic** (same "god object" pattern as MongoDB Index). This is acceptable for Phase 1—we prioritize **getting it working and tested** over perfect architecture.

**Phase 2** (separate effort, see [refactoring/01-index-decomposition.md](refactoring/01-index-decomposition.md)) will decompose BOTH MongoDB and DuckDB implementations into 5 focused classes (Repository, Validator, FileManager, SearchEngine, Facade). This should NOT block shipping Phase 1.

## Why Phase 1 First?

1. **Validate the technology choice** - Make sure DuckDB hybrid search works for real use cases
2. **Get user feedback** - Learn if semantic search, hybrid weights, tag filtering are valuable
3. **Reduce risk** - Don't combine technology migration + architectural refactoring in one PR
4. **Ship value faster** - Better search capability available sooner
5. **Refactor with confidence** - Once working end-to-end, we'll know exactly what abstractions make sense

## TL;DR - Phase 1 Status

**✅ COMPLETE**: All core functionality implemented and tested
- ✅ 3 features: `search_batch()`, tag filtering, `validate_model()` - DONE
- ✅ **67 tests (34 ported + 33 new)** - ALL PASSING
  - 66 fast tests (no API calls)
  - 1 callout test (real OpenAI API)
- ✅ Validation bugs fixed (batch + single-file updates)
- ✅ Test quality improvements (pre-computed embeddings, removed slow markers, fixture docs)

**⏳ NEXT STEPS**: Integration (optional, can be separate PR)
1. **OPTIONAL**: Basic 2-class decomposition (search/data or read/write split)
2. Replace MongoDB Index with DuckDB (rename files, update config)
3. Integration testing (CLI, notebooks, performance)
4. Documentation updates

**READY TO MERGE**: Working DuckDB Index with hybrid search, 67 passing tests (66 fast + 1 callout)

**Phase 2** (later): Full 5-class decomposition of both backends (see [refactoring/01-index-decomposition.md](refactoring/01-index-decomposition.md))

## Target State

**Simple DuckDB-based index** with:
- Single `.duckdb` file (no server needed)
- Hybrid search: exact match + FTS (BM25) + HNSW vector similarity (ALWAYS enabled)
- Same public API as current Index class (minus `use_semantic` parameter)
- 8 tables: 1 main, 2 normalized contributors, 5 denormalized (people/org links + synonyms + attributes + tags)
- Embeddings ALWAYS generated (OpenAI key REQUIRED)

## Schema Design

**Updated 2025-10-08:** foreign key constraints were removed to simplify rebuilds. Integrity is enforced by the rebuild process, which always refreshes all denormalized tables before recreating indexes. Hybrid search indexes (FTS + HNSW) are dropped before any write and rebuilt afterward.

**8 Tables:**
1. `finding_models` - main model metadata with embeddings (NOT NULL)
2. `people` - normalized person master data
3. `organizations` - normalized organization master data
4. `model_people` - denormalized model→person links
5. `model_organizations` - denormalized model→organization links
6. `synonyms` - denormalized synonym storage
7. `attributes` - denormalized attribute storage
8. `tags` - denormalized tags

### Core Tables

#### 1. `finding_models` - Main finding model metadata

```sql
CREATE TABLE finding_models (
    -- Core identifiers
    oifm_id VARCHAR PRIMARY KEY,
    slug_name VARCHAR UNIQUE NOT NULL,
    name VARCHAR UNIQUE NOT NULL,
    
    -- File tracking
    filename VARCHAR UNIQUE NOT NULL,
    file_hash_sha256 VARCHAR NOT NULL,
    
    -- Content
    description TEXT,
    
    -- Search fields (ALWAYS populated - semantic search is core feature)
    search_text TEXT NOT NULL,  -- Concatenated: name + description + synonyms for FTS
    embedding FLOAT[512] NOT NULL,  -- OpenAI text-embedding-3-small with 512 dims (ALWAYS generated, convert to 32-bit)
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS index for BM25 keyword search
PRAGMA create_fts_index(
    'finding_models',
    'oifm_id',
    'search_text',
    stemmer = 'porter',  -- Use 'porter' stemmer
    stopwords = 'english',
    lower = 1,
    overwrite = 1
);

-- HNSW index for fast vector similarity search (approximate nearest neighbor)
-- Note: Uses L2 distance by default (DuckDB doesn't support cosine metric directly)
-- Convert L2 to cosine similarity in queries: cosine_sim = 1 - (l2_distance / 2)
CREATE INDEX finding_models_embedding_hnsw 
ON finding_models USING HNSW (embedding);

-- Additional indexes for common queries
CREATE INDEX idx_finding_models_name ON finding_models(name);  -- For name lookups
CREATE INDEX idx_finding_models_slug_name ON finding_models(slug_name);  -- For slug lookups
```

#### 2. `people` - Contributor people (normalized)

```sql
CREATE TABLE people (
    github_username VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    email VARCHAR NOT NULL,
    organization_code VARCHAR,
    url VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. `organizations` - Contributor organizations (normalized)

```sql
CREATE TABLE organizations (
    code VARCHAR PRIMARY KEY,  -- e.g., "MSFT", "ACR"
    name VARCHAR NOT NULL,
    url VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 3. `model_people` - Links models to people

```sql
CREATE TABLE model_people (
    oifm_id VARCHAR NOT NULL,
    person_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL DEFAULT 'contributor',
    display_order INTEGER,

    PRIMARY KEY (oifm_id, person_id, role)
);

-- Index for quick "get all people for model" queries
CREATE INDEX idx_model_people_model ON model_people(oifm_id);

-- Index for quick "get all models by person" queries
CREATE INDEX idx_model_people_person ON model_people(person_id);
```

#### 4. `model_organizations` - Links models to organizations

```sql
CREATE TABLE model_organizations (
    oifm_id VARCHAR NOT NULL,
    organization_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL DEFAULT 'contributor',
    display_order INTEGER,

    PRIMARY KEY (oifm_id, organization_id, role)
);

-- Index for quick "get all orgs for model" queries
CREATE INDEX idx_model_organizations_model ON model_organizations(oifm_id);

-- Index for quick "get all models by org" queries
CREATE INDEX idx_model_organizations_org ON model_organizations(organization_id);
```

#### 5. `synonyms` - Denormalized synonym storage

```sql
CREATE TABLE synonyms (
    oifm_id VARCHAR NOT NULL,
    synonym VARCHAR NOT NULL,
    PRIMARY KEY (oifm_id, synonym)
);

-- Index for quick "find model by synonym" queries (exact match lookup)
CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);

-- Index for quick "get all synonyms for model" queries
CREATE INDEX idx_synonyms_model ON synonyms(oifm_id);
```

**Note:** Synonyms are ALSO included in `search_text` for FTS matching, but this table provides fast exact-match lookups (e.g., `get()` by synonym name).

#### 6. `attributes` - Denormalized attribute storage

```sql
CREATE TABLE attributes (
    attribute_id VARCHAR PRIMARY KEY,  -- e.g., "OIFMA_MSFT_123456"
    oifm_id VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,  -- Denormalized for quick lookup
    attribute_name VARCHAR NOT NULL,
    attribute_type VARCHAR NOT NULL  -- 'choice' or 'numeric'
);

-- Index for quick "find model by attribute ID" queries (for validation)
CREATE INDEX idx_attributes_model ON attributes(oifm_id);

-- Index for "search by attribute name"
CREATE INDEX idx_attributes_name ON attributes(attribute_name);
```

#### 7. `tags` - Denormalized tags (for filtering)

```sql
CREATE TABLE tags (
    oifm_id VARCHAR NOT NULL,
    tag VARCHAR NOT NULL,
    PRIMARY KEY (oifm_id, tag)
);

-- Index for quick "find models with tag" queries
CREATE INDEX idx_tags_tag ON tags(tag);

-- Index for quick "get all tags for model" queries  
CREATE INDEX idx_tags_model ON tags(oifm_id);
```

**Design Rationale:**
- **Main table** (`finding_models`) keeps only core searchable fields
- **Normalized reference tables** (`people`, `organizations`) for contributor master data
- **Separate junction tables** for people vs organizations - cleaner than polymorphic:
  - `model_people` - link models to person contributors
  - `model_organizations` - link models to organization contributors
- **Denormalized data tables** for fast lookups without joins:
    - `synonyms` - exact match synonym lookups (also in search_text for FTS)
    - `attributes` - attribute ID uniqueness validation, model ↔ attribute queries
    - `tags` - fast tag filtering and model ↔ tag queries
- **HNSW index** on embeddings ALWAYS populated for semantic search (core feature, not optional)
- **FTS index** for BM25 text search on concatenated search_text
- **Manual cleanup** replaces FK cascades: denormalized tables are cleared in transactions before re-inserting rows
- No attributes stored in main table - reconstructed from denormalized tables only when needed

### Index Strategy Summary

**Required indexes:**
1. Primary keys (automatic) - `oifm_id`, `github_username`, `code`, `attribute_id`, junction table composites
2. Unique constraints - `name`, `slug_name`, `filename` in `finding_models`
3. FTS index - BM25 scoring on `search_text` in `finding_models`
4. HNSW index - Vector similarity on `embedding` in `finding_models` (ALWAYS populated)
5. Lookup indexes:
   - `synonyms(synonym)` - exact match synonym lookups
   - `tags(tag)` - filter models by tag
   - `model_people(person_id)` - find models by person contributor
   - `model_organizations(organization_id)` - find models by org contributor
   - `attributes(attribute_name)` - search by attribute name

**Why these indexes:**
- **Primary keys** - enforce uniqueness, fast point lookups
- **FTS** - keyword search with relevance scoring
- **HNSW** - semantic search (approximate nearest neighbor, much faster than exact) - core feature, always on
- **Denormalized indexes** - eliminate joins for common queries (synonyms, tags, contributors, attributes)
- **Separate contributor tables** - cleaner than polymorphic, better query performance
- **Manual cleanup** - `_delete_denormalized_records` removes dependent rows before inserts/deletes

## Implementation Approach

### Phase 1: Create DuckDBIndex Class

**New file:** `src/findingmodel/duckdb_index.py`

```python
class DuckDBIndex:
    """DuckDB-based index for finding models with hybrid search."""
    
    def __init__(self, db_path: str | Path | None = None):
        """Initialize with path to .duckdb file (defaults to config)."""
        
    async def setup(self) -> None:
        """Create tables and indexes if they don't exist."""
        
    # Core CRUD - match current Index API
    async def get(self, id_or_name: str) -> IndexEntry | None:
        """Get by ID, name, or synonym."""
        
    async def contains(self, id_or_name: str) -> bool:
        """Check if ID, name, or synonym exists."""
        
    async def add_or_update_entry_from_file(self, file: Path) -> tuple[IndexReturnType, IndexEntry]:
        """Add or update finding model from file, handling all denormalized tables."""
        
    async def remove_entry(self, oifm_id: str) -> None:
        """Remove model and all denormalized data (application clears tables before rebuild)."""
    
    # Search - enhanced with hybrid approach (semantic ALWAYS enabled)
    async def search(
        self,
        query: str,
        limit: int = 10,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7,
        tags: list[str] | None = None
    ) -> list[IndexEntry]:
        """Hybrid search: exact match > BM25 + semantic fusion, with optional tag filtering.
        
        Semantic search is ALWAYS performed (embeddings always populated).
        """
        
    async def search_batch(self, queries: list[str], limit: int = 10) -> dict[str, list[IndexEntry]]:
        """Search multiple queries efficiently.
        
        IMPORTANT: Embeds ALL queries in a single OpenAI API call for efficiency,
        then performs hybrid search for each query.
        
        Example implementation:
            # Single API call for all embeddings
            response = await openai_client.embeddings.create(
                model=settings.openai_embedding_model,  # "text-embedding-3-small"
                input=queries,  # List of strings
                dimensions=settings.openai_embedding_dimensions  # 512
            )
            embeddings = [e.embedding for e in response.data]
            
            # Search with each pre-computed embedding
            results = {}
            for query, embedding in zip(queries, embeddings):
                results[query] = await self._search_with_embedding(
                    query, embedding, limit
                )
            return results
        """
    
    # Contributor lookups - now using normalized tables
    async def get_person(self, github_username: str) -> Person | None:
        """Look up person by GitHub username."""
        
    async def get_organization(self, code: str) -> Organization | None:
        """Look up organization by code."""
        
    async def add_or_update_contributors(self, model: FindingModelFull) -> None:
        """Add/update people and organizations, then link to model."""
    
    # Validation - now checks denormalized attribute table
    async def validate_model(self, model: FindingModelFull) -> list[str]:
        """Validate model for ID conflicts (name, OIFM ID, attribute IDs)."""
        
    async def _check_attribute_id_conflicts(self, model: FindingModelFull) -> list[str]:
        """Check if any attribute IDs are already taken by other models."""
    
    # Batch operations
    async def update_from_directory(self, path: Path) -> dict:
        """Scan directory and update all models."""
    
    async def rebuild_index_from_directory(self, path: Path) -> dict:
        """Rebuild entire index from scratch.
        
        Useful after:
        - HNSW index corruption from unexpected shutdown
        - Schema changes
        - Experimental persistence issues
        """
    
    # Counts
    async def count(self) -> int:
        """Count total finding models."""
        
    async def count_people(self) -> int:
        """Count total people."""
        
    async def count_organizations(self) -> int:
        """Count total organizations."""
    
    # Private search helpers
    async def _exact_match(self, query: str) -> IndexEntry | None:
    async def _fts_search(self, query: str, limit: int, tags: list[str] | None = None) -> list[tuple[str, float]]:
    async def _semantic_search(self, query: str, limit: int, tags: list[str] | None = None) -> list[tuple[str, float]]:
    async def _hybrid_fusion(
        self,
        fts_results: list[tuple[str, float]],
        semantic_results: list[tuple[str, float]],
        bm25_weight: float,
        semantic_weight: float
    ) -> list[str]:
        """Weighted fusion with min-max normalization."""
    
    # Private denormalization helpers
    async def _upsert_people(self, model: FindingModelFull) -> None:
        """Upsert person contributors and create model_people entries."""
        
    async def _upsert_organizations(self, model: FindingModelFull) -> None:
        """Upsert organization contributors and create model_organizations entries."""
    
    async def _upsert_synonyms(self, model: FindingModelFull) -> None:
        """Replace all synonyms for a model in denormalized table."""
        
    async def _upsert_attributes(self, model: FindingModelFull) -> None:
        """Replace all attributes for a model in denormalized table."""
        
    async def _upsert_tags(self, model: FindingModelFull) -> None:
        """Replace all tags for a model."""
```

### Phase 2: Hybrid Search Implementation

**Search Flow:**
1. **Exact match check** - if query matches ID, name, or synonym exactly, return immediately (score=1.0)
2. **Tag filtering** (if tags provided) - get candidate oifm_ids from tags table
3. **FTS search** - BM25 scoring on `search_text` field, filtered by tags if provided
4. **Semantic search** (if enabled) - HNSW approximate nearest neighbor on embeddings, filtered by tags
5. **Fusion** - combine scores using weighted approach:
   - Normalize BM25 scores to [0,1] using min-max
   - Weight: `0.3 * normalized_bm25 + 0.7 * cosine_similarity`
   - Sort by hybrid score descending

**SQL Pattern with Tag Filtering:**
```sql
WITH tag_filtered AS (
    -- Only if tags are provided, otherwise skip this CTE
    SELECT DISTINCT oifm_id
    FROM tags
    WHERE tag IN (?, ?, ...)  -- All required tags
    GROUP BY oifm_id
    HAVING COUNT(DISTINCT tag) = ?  -- Must have all tags
),
fts_results AS (
    SELECT f.oifm_id,
           fts_main_finding_models_search_text.match_bm25(f.oifm_id, ?) as bm25_score
    FROM finding_models f
    LEFT JOIN tag_filtered tf ON f.oifm_id = tf.oifm_id
    WHERE fts_main_finding_models_search_text.match_bm25(f.oifm_id, ?) > 0
      AND (tf.oifm_id IS NOT NULL OR ? IS NULL)  -- Only filter if tags provided
),
normalized_fts AS (
    SELECT *,
           (bm25_score - min(bm25_score) OVER ()) / 
           NULLIF((max(bm25_score) OVER () - min(bm25_score) OVER ()), 0) as norm_bm25
    FROM fts_results
),
semantic_results AS (
    SELECT f.oifm_id,
           1 - (array_distance(f.embedding, ?) / 2) as cosine_sim  -- Convert L2 to cosine
    FROM finding_models f
    LEFT JOIN tag_filtered tf ON f.oifm_id = tf.oifm_id
    WHERE f.embedding IS NOT NULL
      AND (tf.oifm_id IS NOT NULL OR ? IS NULL)  -- Only filter if tags provided
    ORDER BY array_distance(f.embedding, ?)  -- HNSW accelerated (L2 distance)
    LIMIT ?
)
SELECT f.*, 
       COALESCE(0.3 * n.norm_bm25, 0) + COALESCE(0.7 * s.cosine_sim, 0) as hybrid_score
FROM finding_models f
LEFT JOIN normalized_fts n ON f.oifm_id = n.oifm_id
LEFT JOIN semantic_results s ON f.oifm_id = s.oifm_id
WHERE n.oifm_id IS NOT NULL OR s.oifm_id IS NOT NULL
ORDER BY hybrid_score DESC
LIMIT ?;
```

### Phase 3: Migration Path

**No MongoDB Backward Compatibility Needed**
- Index can be recreated from directory of `*.fm.json` files anytime
- Simply replace `index.py` with new DuckDB implementation
- Run `update_from_directory()` to rebuild index from scratch

**Migration Approach:**
```python
# One-time rebuild from existing files
async def rebuild_index():
    """Rebuild DuckDB index from finding model files."""
    # Open in read-write mode for updates
    async with Index(read_only=False) as index:
        await index.setup()
        
        # Point at directory with all *.fm.json files
        # Automatically: drops HNSW → updates data → rebuilds HNSW
        result = await index.update_from_directory(Path("path/to/finding_models"))
        
        print(f"Added: {result['added']}")
        print(f"Updated: {result['updated']}")
        print(f"Unchanged: {result['unchanged']}")

# Normal usage (read-only)
async def search_models(query: str):
    """Search finding models (read-only, default)."""
    async with Index() as index:  # read_only=True by default
        return await index.search(query)
```

## Implementation Steps

**Note**: See `tasks/duckdb-common-patterns.md` for plan to extract shared DuckDB utilities to avoid code duplication with `duckdb_search.py`.

### Step 0: Create DuckDB Utilities (Optional - can do during or after)
- Create `src/findingmodel/tools/duckdb_utils.py` with shared utilities:
  - `setup_duckdb_connection()` - connection management with extensions
  - `get_embedding_for_duckdb()` / `batch_embeddings_for_duckdb()` - embedding with float32 conversion
  - `normalize_scores()`, `weighted_fusion()`, `rrf_fusion()` - score combination
  - `l2_to_cosine_similarity()` - distance metric conversion
- See detailed plan in `tasks/duckdb-common-patterns.md`

### Step 1: Create DuckDBIndex skeleton
- Create `src/findingmodel/duckdb_index.py`
- Implement `__init__` with `read_only=True` parameter (default for 99% of usage)
- Implement `setup`, schema creation with:
  - Load FTS and VSS extensions: `LOAD fts; LOAD vss;`
  - **NO experimental persistence flag needed** - HNSW rebuilt after batch updates
  - **Option**: Use `setup_duckdb_connection()` utility if available
- Connection pattern:
  - Default (read-only): For search operations
  - Explicit read-write: Only for batch updates
- Add to `src/findingmodel/__init__.py` exports

### Step 2: Implement core CRUD
- `get()` - ID/name/synonym lookup (check synonyms table for exact match)
- `add_or_update_entry_from_file()` - insert/update with hash checking
  - Update `finding_models` table
  - Generate embeddings using `settings.openai_embedding_model` with `settings.openai_embedding_dimensions` (512)
  - **Convert embeddings to FLOAT[512]** (from Python float64 to float32)
  - **Option**: Use `get_embedding_for_duckdb()` utility for automatic conversion
  - Denormalize to `model_people`, `model_organizations`, `synonyms`, `attributes`, `tags` tables
- `remove_entry()` - delete by ID (manual helpers clear denormalized tables before commit)
- Extract file hash and entry creation helpers
- `get_person()`, `get_organization()` - lookups in normalized tables
- `count()`, `count_people()`, `count_organizations()` - simple counts

### Step 3: Implement exact match search
- Check ID, name in `finding_models` table
- Check `synonyms` table for exact synonym match
- Return immediately if found

### Step 4: Implement FTS search
- Create FTS index in setup
- BM25 query implementation
- Min-max normalization

### Step 5: Implement semantic search (ALWAYS enabled)
- Generate embeddings using `settings.openai_embedding_model` with `settings.openai_embedding_dimensions` (512 dims)
- Use same embedding configuration as anatomic location search for consistency
- HNSW approximate nearest neighbor query using L2 distance (DuckDB default)
- Convert L2 distance to cosine similarity: `cosine_sim = 1 - (l2_distance / 2)`
- Use `array_distance(embedding, ?)` for HNSW-accelerated search
- Embeddings are NOT NULL - semantic search is a core feature, not optional
- Batch embed multiple queries in single OpenAI API call for efficiency

### Step 6: Implement hybrid fusion
- Combine FTS and semantic results with weighted scoring
- Min-max normalize BM25 scores before fusion
- Weight: 0.3 * normalized_bm25 + 0.7 * cosine_similarity
- Deduplication and sorting by hybrid score
- **Option**: Use `normalize_scores()` and `weighted_fusion()` utilities if available

### Step 7: Implement validation
- `validate_model()` - check for conflicts
- `_check_attribute_id_conflicts()` - query `attributes` table for ID uniqueness
- Check name, oifm_id, slug_name uniqueness in `finding_models` table

### Step 8: Implement batch operations with HNSW drop/rebuild strategy
- `update_from_directory()` - **KEY OPTIMIZATION**. Directory ingestion pipeline:
  1. Accept input directory (string or `Path`) and resolve to absolute `Path`.
  2. Enumerate `*.fm.json` files, computing SHA-256 hash for each (reuse `_calculate_file_hash`).
  3. Create a DuckDB temporary table `tmp_directory_files(filename TEXT, file_hash_sha256 TEXT)` and bulk insert filename/hash pairs (filenames stored relative to directory root for comparison).
  4. Perform a `FULL OUTER JOIN` between `finding_models` and `tmp_directory_files` on filename to classify rows:
      - `added`: rows where `tmp_directory_files.filename IS NOT NULL` and matching `finding_models.filename IS NULL`.
      - `updated`: rows where both sides exist but hashes differ.
      - `removed`: rows where `finding_models.filename IS NOT NULL` and matching `tmp_directory_files.filename IS NULL`.
  5. Materialize three working tables (or CTE-driven result sets) representing the new, updated, and removed entries to drive batch actions.
  6. Generate batch commands:
      - Inserts: load and validate new files, stage payloads, and append to `finding_models` plus denormalized tables.
      - Updates: reload affected files, mark rows for deletion via `_delete_denormalized_records`, then upsert fresh data.
      - Deletes: queue `oifm_id` values for `_delete_denormalized_records` and base row deletion.
  7. Wrap the batch in a single transaction:
      - Drop `finding_models_embedding_hnsw` and `fts_main_finding_models` indexes up front.
      - Execute queued deletes, updates, and inserts in deterministic order (deletes → updates → inserts).
      - Recreate FTS and HNSW indexes after data changes complete.
  8. Return a structured result object summarizing counts (`added`, `updated`, `removed`) and any validation errors.
- `rebuild_index_from_directory()` - drop all data, rebuild from scratch (reuse step 8 pipeline but skip diff detection).
- `search_batch()` - **IMPORTANT**: Batch-embed ALL queries in single OpenAI API call for efficiency.
- **Option**: Use `batch_embeddings_for_duckdb()` utility for automatic batching.

### Step 9: Testing
- Port existing `test_index.py` tests
- Add hybrid search specific tests
- Add denormalized table tests (contributors, attributes, tags)
- Performance benchmarks

### Step 10: Replace Index
- Delete old `index.py` (MongoDB version)
- Rename `duckdb_index.py` to `index.py`
- Update config to use DuckDB path instead of MongoDB URI
- Remove MongoDB dependencies from `pyproject.toml`

### Step 11: Documentation
- Update README with DuckDB setup
- Document hybrid search tuning parameters
- Migration guide for existing MongoDB users

## Configuration Changes

**Remove from `config.py`:**
```python
# DELETE these MongoDB settings:
mongodb_uri: SecretStr
mongodb_db: str = "findingmodels"
mongodb_index_collection_base: str = "index"
mongodb_people_collection_base: str = "people"
mongodb_organizations_collection_base: str = "organizations"
mongodb_use_atlas_search: bool = False  # Also remove this
```

**Add to `config.py`:**
```python
# Index storage
duckdb_index_path: Path = Path("data/finding_models.duckdb")

# Hybrid search parameters (semantic search ALWAYS enabled)
hybrid_search_bm25_weight: float = 0.3
hybrid_search_semantic_weight: float = 0.7

# Embedding configuration (use existing settings)
# openai_embedding_model: str = "text-embedding-3-small" (already in config)
# openai_embedding_dimensions: int = 512 (already in config, matches anatomic locations)

# OpenAI key now REQUIRED (embeddings always generated for semantic search)
# Configured via OPENAI_API_KEY environment variable
# Note: OpenAI returns float64 embeddings; convert to float32 for DuckDB FLOAT[512]
```

## Benefits

**Simplicity:**
- No MongoDB server setup required
- Single `.duckdb` file storage
- Simpler connection management (synchronous `duckdb.connect` in async methods, matching existing pattern)
- **Note**: DuckDB Python API is synchronous; async methods wrap sync calls (same as existing `duckdb_search.py`)
- **No experimental HNSW persistence** - drop/rebuild strategy avoids corruption risks
- Read-only connections by default (99% of usage)
- Fewer dependencies (remove motor, pymongo)
- Rebuild index from files anytime

**Performance:**
- HNSW vector index for fast semantic search (proven in anatomic locations)
- FTS with BM25 for keyword search
- Denormalized tables eliminate joins for common queries
- Columnar storage for analytics

**Features:**
- Better search: hybrid FTS + semantic with weighted fusion
- Tag filtering in search
- Fast contributor/attribute/tag lookups via denormalized tables
- Attribute ID uniqueness validation via index

**Maintenance:**
- Fewer moving parts
- Easier testing (no MongoDB setup needed)
- Portable (just copy .duckdb file)
- Simpler deployment (no database server)

## Testing Strategy

**Unit Tests:**
- Each DuckDBIndex method tested independently
- Mock embedding generation for semantic search tests
- Test hybrid search with known scores

**Integration Tests:**
- Port all existing `test_index.py` tests
- Test migration from MongoDB (if backward compat maintained)
- Test with actual DuckDB file

**Performance Tests:**
- Benchmark search latency (should be <50ms for typical queries)
- Benchmark bulk update performance
- Compare to MongoDB baseline

## Risks & Mitigations

**Risk: HNSW index unavailable during batch updates**
- **Impact**: Search unavailable for ~5 seconds during `update_from_directory()`
- **Frequency**: Rare (only during batch updates)
- **Mitigation**: 
  - Batch updates are infrequent (development/maintenance only)
  - Can show "updating index" message to users
  - Could implement background update with old index serving reads
  - No corruption risk - worst case is cancelled update, run again

**Risk: ELIMINATED - No experimental HNSW persistence**
- **Previous concern**: Experimental persistence could cause data loss
- **Solution**: Drop/rebuild HNSW strategy completely avoids this
- **Benefit**: Much safer, simpler, no special flags needed

**Risk: Breaking existing code**
- Mitigation: Keep same public API, use alias pattern
- Run full test suite before merging

**Risk: Embeddings are slow to generate**
- Mitigation: Cache embeddings, only regenerate if content changes
- Use batch embedding API for multiple models

**Risk: FTS index maintenance**
- Mitigation: Rebuild FTS on setup (fast with DuckDB)
- Document when rebuild needed

**Risk: DuckDB file locking with concurrent access**
- Mitigation: Use read-only mode for searches, single writer pattern
- Document limitations vs multi-writer MongoDB

## Phase 1 Completion Plan

### Step 1: Implement Missing Features ✅ COMPLETE
- [x] **search_batch()**: Batch embedding optimization - DONE
  - Implemented `_search_semantic_with_embedding()` helper for pre-computed embeddings
  - Single OpenAI API call for all queries in batch
  - Tested with 4 test cases (multiple queries, empty list, mixed valid/invalid)
- [x] **Tag filtering in search()**: DONE
  - Added `tags: Sequence[str] | None = None` parameter
  - Supports single tag and multiple tag AND logic
  - Applied to exact match, FTS, and semantic search paths
  - Tested with 4 test cases (single tag, multiple AND, nonexistent, all paths)
- [x] **validate_model()**: DONE
  - Checks OIFM ID uniqueness in finding_models
  - Checks name/slug_name uniqueness
  - Checks attribute ID conflicts in attributes table
  - Returns list of error messages
  - **Bug fix**: Skip validation for model updates (both single-file and batch)

### Step 2: Port All Tests ✅ COMPLETE
- [x] Port all 34 tests from [test/test_index.py](../test/test_index.py) - DONE
- [x] Add DuckDB-specific tests - ALL DONE (33 new tests):
  - [x] `update_from_directory` with add/update/delete/mixed/no-changes scenarios (5 tests)
  - [x] Denormalized table integrity - synonyms, tags, attributes, model_people, model_organizations (5 tests)
  - [x] HNSW/FTS index creation and rebuild during batch operations (4 tests)
  - [x] Tag filtering in search (4 tests - single, multiple AND, nonexistent, all paths)
  - [x] `search_batch()` batching behavior (4 tests)
  - [x] Validation - ID conflicts, name conflicts, attribute conflicts (3 tests)
  - [x] Read-only mode enforcement (1 test)
  - [x] Performance tests (3 tests - search latency, batch embedding, directory sync)
  - [x] Edge cases - remove when not exists, semantic search with fake embeddings (2 tests)
  - [x] **Semantic search with pre-computed real embeddings** (1 test - deterministic, no API calls)
  - [x] **Semantic search with real OpenAI API** (1 test - @pytest.mark.callout)
- [x] **All 67 tests passing** (66 fast + 1 callout, 100% success rate)
- [x] **Using existing fixtures** from conftest.py (full_model, real_model, tmp_defs_path)
- [x] **Test quality improvements**:
  - [x] Removed `@pytest.mark.slow` markers (not configured, caused warnings)
  - [x] Added pre-computed embedding test for deterministic semantic search testing
  - [x] Added real API callout test for full integration validation
  - [x] Created comprehensive fixture documentation (Serena memory: pytest_fixtures_reference_2025)

**Deliverable**: ✅ All 67 tests passing (66 fast + 1 callout)

### Step 3: OPTIONAL Basic Decomposition
Split DuckDBIndex into **TWO focused classes** (not the full 5-class decomposition):

**Option A: Read/Write Split** (Simplest)
```python
# src/findingmodel/duckdb_index_reader.py
class DuckDBIndexReader:
    """Read-only operations: get, search, count."""
    def __init__(self, db_path: Path, read_only: bool = True)
    async def get(self, id_or_name: str) -> IndexEntry | None
    async def search(self, query: str, ...) -> list[IndexEntry]
    async def search_batch(self, queries: list[str], ...) -> dict
    async def count() -> int
    # ~400 lines

# src/findingmodel/duckdb_index_writer.py
class DuckDBIndexWriter(DuckDBIndexReader):
    """Write operations: add, update, remove, batch."""
    def __init__(self, db_path: Path, read_only: bool = False)
    async def add_or_update_entry_from_file(self, file: Path) -> tuple
    async def remove_entry(self, oifm_id: str) -> None
    async def update_from_directory(self, path: Path) -> dict
    async def validate_model(self, model: FindingModelFull) -> list[str]
    # ~900 lines

# src/findingmodel/duckdb_index.py (facade)
class DuckDBIndex(DuckDBIndexWriter):
    """Backward-compatible facade."""
    pass  # Inherits everything
```

**Option B: Search/Data Split** (Your suggestion)
```python
# src/findingmodel/duckdb_search_engine.py
class DuckDBSearchEngine:
    """All search operations."""
    async def search(...) -> list[IndexEntry]
    async def search_batch(...) -> dict
    async def _search_exact(...) -> IndexEntry | None
    async def _search_fts(...) -> list[tuple[str, float]]
    async def _search_semantic(...) -> list[tuple[str, float]]
    async def _hybrid_fusion(...) -> list[str]
    # ~500 lines

# src/findingmodel/duckdb_data_manager.py
class DuckDBDataManager:
    """All data loading/management."""
    async def add_or_update_entry_from_file(...) -> tuple
    async def update_from_directory(...) -> dict
    async def remove_entry(...) -> None
    async def validate_model(...) -> list[str]
    # All the batch/denormalization helpers
    # ~800 lines

# src/findingmodel/duckdb_index.py (facade)
class DuckDBIndex:
    """Facade combining search + data."""
    def __init__(self, db_path: Path, read_only: bool = True):
        self.search_engine = DuckDBSearchEngine(db_path)
        self.data_manager = DuckDBDataManager(db_path, read_only)

    # Delegate methods
    async def search(self, query: str, ...) -> list[IndexEntry]:
        return await self.search_engine.search(query, ...)

    async def update_from_directory(self, path: Path) -> dict:
        return await self.data_manager.update_from_directory(path)
    # ~200 lines of delegation
```

**Decision**: Choose Option A (simpler) OR Option B (cleaner separation) OR **skip** if too risky before shipping.

### Step 4: Replace MongoDB Index
- [ ] Rename `index.py` → `mongodb_index.py` (keep for reference)
- [ ] Rename `duckdb_index.py` → `index.py`
- [ ] Update `src/findingmodel/__init__.py`:
  ```python
  from findingmodel.index import DuckDBIndex as Index  # Alias for backward compat
  ```
- [ ] Update `config.py`:
  - Remove MongoDB settings (uri, db, collections)
  - Add `duckdb_index_path: Path = Path("data/finding_models.duckdb")`
- [ ] Mark MongoDB dependencies as optional in `pyproject.toml`

### Step 5: Integration Testing
- [ ] Test CLI commands with DuckDB (`python -m findingmodel ...`)
- [ ] Rebuild index from directory: `update_from_directory("path/to/models")`
- [ ] Verify all existing notebooks/demos work
- [ ] Performance benchmarks vs MongoDB (if comparable data available)

**Deliverable**: Working DuckDB-based Index, all integration tests passing

### Phase 1 Success Criteria ✅ ALL MET

**Functionality**: ✅ COMPLETE
- [x] DuckDBIndex class with complete schema and indexes
- [x] Core CRUD (get, contains, add/update, remove, counts)
- [x] Hybrid search (exact → FTS + semantic → fusion)
- [x] Batch directory ingestion with hash diffing
- [x] **search_batch()** with batch embedding - DONE
- [x] **Tag filtering** in search - DONE
- [x] **validate_model()** implementation - DONE
- [x] **All 67 tests ported and passing** - DONE (34 ported + 33 new)

**Quality**: ✅ COMPLETE
- [x] Drop/rebuild HNSW strategy (no corruption risk)
- [x] Read-only mode by default
- [x] Enhanced logging
- [x] **Test coverage 100%** (67/67 tests passing: 66 fast + 1 callout)
- [x] **Search latency < 100ms** (verified in performance tests)
- [x] **No regressions vs MongoDB Index** (all ported tests passing)
- [x] **Validation bugs fixed** (skip validation for updates in batch and single-file)
- [x] **Test quality improvements** (pre-computed embeddings, callout tests, fixture docs)

**Integration**: ⏳ DEFERRED (can be separate PR)
- [ ] **MongoDB Index replaced** (or deprecated with DuckDB as default)
- [ ] **Config updated** (DuckDB path, no MongoDB)
- [ ] **CLI commands verified**
- [ ] **Documentation updated** (README, migration guide)

**Optional** (deferred to later):
- [ ] Basic decomposition (read/write or search/data split)
- [ ] MongoDB dependencies removed from pyproject.toml
- [ ] Performance comparison report

## Clarifications Applied

✅ **Contributor tables**: Normalized `people` and `organizations` tables with **SEPARATE** `model_people` and `model_organizations` junction tables (cleaner than polymorphic `model_contributors`).

✅ **Synonyms**: Denormalized `synonyms` table for exact-match lookups. Synonyms ALSO included in `search_text` for FTS matching.

✅ **Attributes**: Denormalized `attributes` table with unique index on `attribute_id` for fast conflict checking during validation.

✅ **Tags**: Denormalized `tags` table with indexes for fast filtering and model-tag lookups.

✅ **Vector index**: Using HNSW index (same as anatomic locations) for fast approximate nearest neighbor search.

✅ **Semantic search ALWAYS enabled**: Embeddings are NOT NULL, `use_semantic` parameter removed. OpenAI key is now REQUIRED.

✅ **Batch embedding optimization**: `search_batch()` embeds all queries in a single OpenAI API call for efficiency.

✅ **No MongoDB support**: Clean break, no backward compatibility. Index rebuilt from `*.fm.json` files.

✅ **No branch support**: Removed the branch concept entirely.

✅ **No Atlas Search**: Removed completely.

## Phase 1.5: Optimization Opportunities (Before Integration)

### Assessment Date: 2025-10-10

Before proceeding with integration (Step 4), several high-value optimizations have been identified that will benefit both Phase 1 completion and future work.

### 1. Embedding Cache System ⭐ HIGHEST PRIORITY

**Current State**: ❌ No caching - every embedding regenerated on each operation
**Problem**:
- Expensive: Every embedding call costs money
- Slow: Re-indexing regenerates ALL embeddings even for unchanged content
- Wasteful: Both anatomic locations and finding models repeat work

**Recommended Solution**: DuckDB-based embedding cache

**Design**:
```python
# src/findingmodel/embedding_cache.py
# New file: embeddings_cache.duckdb

CREATE TABLE embedding_cache (
    text_hash TEXT PRIMARY KEY,           -- SHA256 of input text
    model TEXT NOT NULL,                  -- e.g., "text-embedding-3-small"
    dimensions INTEGER NOT NULL,          -- 512 or 1536
    embedding FLOAT[dimensions] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_cache_model ON embedding_cache(model, dimensions);
```

**Benefits**:
- ✅ Consistent with DuckDB-first architecture
- ✅ Fast indexed lookups by hash
- ✅ Shared between anatomic locations AND finding model indexes
- ✅ Can be versioned/cleared by model or date
- ✅ No additional dependencies
- ✅ Massive cost savings on re-indexing

**Integration Points**:
- Wrap `get_embedding()` and `get_embeddings_batch()` in [tools/common.py:34-96](../src/findingmodel/tools/common.py)
- Check cache before calling OpenAI API
- Store misses after successful API calls
- Use text hash (SHA256) as key for deterministic lookups

**Implementation**:
```python
class EmbeddingCache:
    """DuckDB-based cache for OpenAI embeddings."""

    async def get_embedding(
        self, text: str, model: str, dimensions: int
    ) -> list[float] | None:
        """Get cached embedding or None if not found."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        # Query cache table

    async def store_embedding(
        self, text: str, model: str, dimensions: int, embedding: list[float]
    ) -> None:
        """Store embedding in cache."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        # Insert or replace

    async def get_embeddings_batch(
        self, texts: list[str], model: str, dimensions: int
    ) -> list[list[float] | None]:
        """Get batch of embeddings, returning None for misses."""
        # Bulk query for all hashes

    async def store_embeddings_batch(
        self, texts: list[str], model: str, dimensions: int,
        embeddings: list[list[float]]
    ) -> None:
        """Store batch of embeddings."""
        # Bulk insert
```

**Usage in tools/common.py**:
```python
_embedding_cache = EmbeddingCache()  # Module-level singleton

async def get_embedding(
    text: str, client: AsyncOpenAI | None = None,
    model: str | None = None, dimensions: int = 512
) -> list[float] | None:
    """Get embedding with caching."""
    resolved_model = model or settings.openai_embedding_model

    # Check cache first
    cached = await _embedding_cache.get_embedding(text, resolved_model, dimensions)
    if cached is not None:
        return cached

    # Cache miss - call API
    if not client:
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    response = await client.embeddings.create(
        input=text, model=resolved_model, dimensions=dimensions
    )
    embedding = response.data[0].embedding

    # Store in cache
    await _embedding_cache.store_embedding(text, resolved_model, dimensions, embedding)
    return embedding
```

### 2. CLI Commands for Index Management ⭐ HIGH PRIORITY

**Current State**: ❌ No CLI commands for index building/updating
**Need**: Make DuckDB index operations accessible via CLI

**Recommended Commands**:
```bash
# Finding models index
uv run python -m findingmodel index build <directory> [--output path.duckdb]
uv run python -m findingmodel index update <directory> [--index path.duckdb]
uv run python -m findingmodel index validate <directory>
uv run python -m findingmodel index stats [--index path.duckdb]

# Anatomic locations index
uv run python -m findingmodel anatomic build <json_file> [--output path.duckdb]
uv run python -m findingmodel anatomic update <json_file> [--index path.duckdb]
uv run python -m findingmodel anatomic stats [--index path.duckdb]
```

**Implementation** in [cli.py](../src/findingmodel/cli.py):
```python
@cli.group()
def index() -> None:
    """Index management commands."""
    pass

@index.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(), help="Output database path")
def build(directory: Path, output: Path | None) -> None:
    """Build finding model index from directory of *.fm.json files."""
    console = Console()

    async def _do_build():
        db_path = output or settings.duckdb_index_path
        console.print(f"[green]Building index at {db_path}")

        async with DuckDBIndex(db_path=db_path, read_only=False) as idx:
            await idx.setup()
            result = await idx.update_from_directory(directory)

            console.print(f"[green]✓ Added: {result['added']}")
            console.print(f"[yellow]✓ Updated: {result['updated']}")
            console.print(f"[red]✓ Removed: {result['removed']}")

    asyncio.run(_do_build())

@index.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--index", type=click.Path(), help="Database path")
def update(directory: Path, index: Path | None) -> None:
    """Update existing index from directory."""
    # Similar to build, but expects existing database

@index.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
def validate(directory: Path) -> None:
    """Validate finding models without writing to index."""
    # Load models and run validate_model() on each

@index.command()
@click.option("--index", type=click.Path(), help="Database path")
def stats(index: Path | None) -> None:
    """Show index statistics."""
    # Print counts, schema info, index status
```

**Refactor anatomic migration script**:
- Extract functions from [migrate_anatomic_to_duckdb.py](../notebooks/migrate_anatomic_to_duckdb.py) into reusable module
- Add to CLI as `anatomic` command group

### 3. Pooch Integration for Remote DuckDB Files

**Current State**: ❌ Not used, not in dependencies
**Value**: Auto-download pre-built DuckDB files instead of building locally

**Benefits**:
- ✅ Users don't need to build indexes on first use
- ✅ Faster onboarding (download vs build+embed)
- ✅ Centralized index updates (rebuild once, distribute to all)
- ✅ Hash verification for integrity

**Configuration additions** to [config.py](../src/findingmodel/config.py):
```python
# Remote index URLs (optional - fallback to local build if not set)
duckdb_index_url: str | None = Field(
    default=None,
    description="URL to download pre-built finding models index (e.g., GitHub releases)"
)
duckdb_anatomic_url: str | None = Field(
    default=None,
    description="URL to download pre-built anatomic locations index"
)
```

**Implementation pattern**:
```python
import pooch

def get_index_path() -> Path:
    """Get index path, downloading if needed."""
    if settings.duckdb_index_url:
        # Use pooch to download and cache
        return pooch.retrieve(
            url=settings.duckdb_index_url,
            known_hash=None,  # Or fetch from manifest
            path=settings.duckdb_index_path.parent,
            fname=settings.duckdb_index_path.name,
        )
    return settings.duckdb_index_path
```

**Distribution pattern**:
- Build index in CI/CD pipeline
- Upload to GitHub releases or S3
- Update config with URL
- Users auto-download on first use

**Add to pyproject.toml**:
```toml
[project.optional-dependencies]
remote = ["pooch>=1.8.0"]
```

### 4. Extract Shared Validation Logic ⭐ QUICK WIN

**Current State**: ~150 lines duplicated between DuckDBIndex and MongoDB Index

**Duplicated validation** in:
- [duckdb_index.py:1223-1279](../src/findingmodel/duckdb_index.py) - `_validate_model()`
- [index.py:254-341](../src/findingmodel/index.py) - `_check_*_conflict()` methods
- [tools/model_editor.py:359](../src/findingmodel/tools/model_editor.py) - `_validate_model_id()`

**Should extract to**: `src/findingmodel/index_validation.py`

**Proposed API**:
```python
# src/findingmodel/index_validation.py

from typing import Protocol

class ValidationContext(Protocol):
    """Protocol for index implementations to provide validation data."""

    async def get_existing_oifm_ids(self) -> set[str]:
        """Get all existing OIFM IDs."""

    async def get_existing_names(self) -> set[str]:
        """Get all existing model names (normalized)."""

    async def get_attribute_ids_by_model(self) -> dict[str, str]:
        """Get mapping of attribute_id -> oifm_id."""

def check_oifm_id_conflict(
    model: FindingModelFull,
    existing_ids: set[str],
    allow_self: bool = False
) -> list[str]:
    """Check for OIFM ID conflicts."""
    errors = []
    if model.oifm_id in existing_ids:
        if not (allow_self and existing_ids == {model.oifm_id}):
            errors.append(f"OIFM ID {model.oifm_id} already exists")
    return errors

def check_name_conflict(
    model: FindingModelFull,
    existing_names: set[str],
    allow_self: bool = False
) -> list[str]:
    """Check for name/slug conflicts."""
    errors = []
    normalized_name = normalize_name(model.name)
    if normalized_name in existing_names:
        if not (allow_self and existing_names == {normalized_name}):
            errors.append(f"Name {model.name} already exists (normalized: {normalized_name})")
    return errors

def check_attribute_id_conflict(
    model: FindingModelFull,
    attribute_ids_by_model: dict[str, str],
    allow_self: bool = False
) -> list[str]:
    """Check for attribute ID conflicts across models."""
    errors = []
    for attr in model.all_attributes():
        if attr.attribute_id in attribute_ids_by_model:
            existing_model = attribute_ids_by_model[attr.attribute_id]
            if not (allow_self and existing_model == model.oifm_id):
                errors.append(
                    f"Attribute ID {attr.attribute_id} already used by {existing_model}"
                )
    return errors

async def validate_finding_model(
    model: FindingModelFull,
    context: ValidationContext,
    allow_self: bool = False
) -> list[str]:
    """Complete validation using protocol-based context."""
    errors = []

    # Gather data from context
    existing_ids = await context.get_existing_oifm_ids()
    existing_names = await context.get_existing_names()
    attribute_map = await context.get_attribute_ids_by_model()

    # Run checks
    errors.extend(check_oifm_id_conflict(model, existing_ids, allow_self))
    errors.extend(check_name_conflict(model, existing_names, allow_self))
    errors.extend(check_attribute_id_conflict(model, attribute_map, allow_self))

    return errors
```

**Usage in DuckDBIndex**:
```python
from findingmodel.index_validation import validate_finding_model, ValidationContext

class DuckDBIndex:
    # ... existing methods ...

    # Implement protocol
    async def get_existing_oifm_ids(self) -> set[str]:
        """Get all OIFM IDs from database."""

    async def get_existing_names(self) -> set[str]:
        """Get all normalized names from database."""

    async def get_attribute_ids_by_model(self) -> dict[str, str]:
        """Get attribute_id -> oifm_id mapping from attributes table."""

    def _validate_model(self, model: FindingModelFull) -> list[str]:
        """Validate model using shared validation logic."""
        return await validate_finding_model(model, self, allow_self=False)
```

**Benefits**:
- ✅ DRY - single source of truth for validation
- ✅ Testable - validate logic independent of index implementation
- ✅ Reusable - works with any index backend (MongoDB, DuckDB, future)
- ✅ Protocol-based - no tight coupling

### 5. Enhanced DuckDB Utilities

**Current State**: Some utilities extracted to [duckdb_utils.py](../src/findingmodel/tools/duckdb_utils.py)
**Opportunity**: Extract more common patterns from anatomic migration script and index

**Already extracted** ✅:
- `setup_duckdb_connection()` - connection with extensions
- `batch_embeddings_for_duckdb()` - batch embedding with float32 conversion
- `normalize_scores()`, `weighted_fusion()`, `rrf_fusion()` - score combination
- `l2_to_cosine_similarity()` - distance conversion

**Could add**:
```python
# src/findingmodel/tools/duckdb_utils.py additions

def create_hnsw_index(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    index_name: str | None = None,
    metric: str = "cosine",
    ef_construction: int = 128,
    ef_search: int = 64,
    m: int = 16
) -> None:
    """Create HNSW vector index with standard parameters."""
    name = index_name or f"idx_{table}_{column}_hnsw"
    conn.execute(f"""
        CREATE INDEX {name}
        ON {table}
        USING HNSW ({column})
        WITH (metric = '{metric}', ef_construction = {ef_construction},
              ef_search = {ef_search}, M = {m})
    """)

def drop_search_indexes(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    hnsw_index: str | None = None,
    fts_index: str | None = None
) -> None:
    """Drop HNSW and FTS indexes for table."""
    if hnsw_index:
        conn.execute(f"DROP INDEX IF EXISTS {hnsw_index}")
    if fts_index:
        conn.execute(f"DROP INDEX IF EXISTS {fts_index}")

def create_fts_index(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    id_column: str,
    *text_columns: str,
    stemmer: str = "porter",
    stopwords: str = "english",
    overwrite: bool = True
) -> None:
    """Create FTS index on text columns."""
    columns_str = ", ".join([f"'{col}'" for col in text_columns])
    conn.execute(f"""
        PRAGMA create_fts_index(
            '{table}',
            '{id_column}',
            {columns_str},
            stemmer = '{stemmer}',
            stopwords = '{stopwords}',
            lower = 1,
            overwrite = {1 if overwrite else 0}
        )
    """)

def bulk_insert_with_validation(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    records: list[tuple],
    columns: list[str]
) -> tuple[int, int]:
    """Bulk insert with error handling.

    Returns:
        Tuple of (successful_count, failed_count)
    """
    placeholders = ", ".join(["?" for _ in columns])
    cols = ", ".join(columns)

    try:
        conn.executemany(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            records
        )
        return len(records), 0
    except Exception as e:
        # Fall back to individual inserts for error reporting
        successful = 0
        failed = 0
        for record in records:
            try:
                conn.execute(
                    f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                    record
                )
                successful += 1
            except Exception:
                failed += 1
        return successful, failed
```

### 6. Extract Shared Data Models (Minor)

**Current State**: Models duplicated in both index implementations

**Duplicated**:
- `IndexEntry` in both [index.py:26](../src/findingmodel/index.py) and [duckdb_index.py:41](../src/findingmodel/duckdb_index.py)
- `AttributeInfo` in both files
- `IndexReturnType` enum in both files

**Should extract to**: `src/findingmodel/index_models.py`

```python
# src/findingmodel/index_models.py

from enum import StrEnum
from pydantic import BaseModel, Field

class AttributeInfo(BaseModel):
    """Basic information about an attribute."""
    attribute_id: str
    attribute_name: str
    attribute_type: str  # 'choice' or 'numeric'

class IndexEntry(BaseModel):
    """Index entry for a finding model."""
    oifm_id: str
    name: str
    slug_name: str
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    attributes: list[AttributeInfo] = Field(default_factory=list)
    contributors: list[Person | Organization] = Field(default_factory=list)
    # ... other fields

class IndexReturnType(StrEnum):
    """Result type for index operations."""
    ADDED = "added"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
```

**Benefits**:
- ✅ Single source of truth
- ✅ Easier to maintain consistency
- ✅ Shared between MongoDB and DuckDB implementations

## Implementation Priority

### Priority 1: Before Integration (Blocking)
1. **Embedding Cache** - Enables fast re-indexing during development/testing
2. **Extract Validation Logic** - Quick win, prevents further duplication
3. **CLI Commands** - Makes index operations accessible for testing

### Priority 2: With Integration (Recommended)
4. **Pooch Integration** - Better user experience for distribution
5. **Enhanced DuckDB Utils** - Cleanup and consolidation

### Priority 3: After Integration (Nice to Have)
6. **Extract Shared Models** - Cleanup, can be done incrementally

## Next Steps

After these optimizations, proceed with:
- Step 4: Replace MongoDB Index (integration)
- Step 5: Integration Testing
- Phase 2: Full 5-class decomposition (separate effort)

## References

- Research: Serena memory `duckdb_hybrid_search_research_2025`
- Existing DuckDB implementation: `src/findingmodel/tools/duckdb_search.py`
- Current Index: `src/findingmodel/index.py` (789 lines)
- Test suite: `test/test_index.py` (635 lines, 34 tests)
- Anatomic migration: `notebooks/migrate_anatomic_to_duckdb.py` (458 lines)

## Phase 8: Complete MongoDB Index Removal

**Status**: ✅ COMPLETE
**Goal**: Remove deprecated MongoDB Index entirely since DuckDB is now the default and fully functional
**Timeline**: Immediate (v0.5.0) - same approach as IdManager removal
**Completed**: 2025-10-29

### Rationale for Immediate Removal

**User Confirmation**: Systems that need MongoDB can stay on v0.4.x. DuckDB Index is fully functional with 67 passing tests.

**Risk Assessment**: **LOW**
- DuckDB Index fully implemented and tested (Phase 1 complete)
- All 67 tests passing (66 fast + 1 callout)
- MongoDB already marked as deprecated in docs
- Breaking change acceptable (deprecated backend removal)
- Optional dependency - users explicitly opt-in to MongoDB

### Analysis

**Current State**:
- mongodb_index.py exists (~810 lines of deprecated code)
- test_index_mongodb.py exists (37 passing tests)
- MongoDBIndex REMOVED from __init__.py exports (already done)
- MongoDB config fields commented out in config.py
- Optional dependency: `mongodb = ["motor>=3.7.1"]` in pyproject.toml
- Documentation mentions MongoDB as deprecated backend

**External Dependencies Found**:
1. test/test_index_mongodb.py - Only file importing from mongodb_index
2. test/test_model_editor.py - Imports `from findingmodel import Index` (already aliased to DuckDBIndex, will work correctly)
3. docs/database-management.md - Has MongoDB section (lines 248-271)
4. README.md - Has MongoDB comment (line 543)

**What Gets Removed**:
- MongoDB Index class (~810 lines)
- MongoDB-specific tests (37 tests)
- motor optional dependency
- MongoDB documentation sections
- Commented MongoDB config fields

**What Gets Kept**:
- index_validation.py - Shared validation using Protocol pattern, works with any backend

---

### Task 8.1: Delete MongoDB Implementation Files

**Files to Delete**:
1. `src/findingmodel/mongodb_index.py` (~810 lines)
2. `test/test_index_mongodb.py` (37 tests)

**Rationale**: These files are only used by each other. No external dependencies beyond the test file.

**Acceptance Criteria**:
- [x] Both files deleted
- [x] No import errors when importing findingmodel
- [x] Remaining tests still pass

---

### Task 8.2: Remove MongoDB from pyproject.toml

**File**: `pyproject.toml`

**Change** (line 43):
```toml
# DELETE this line:
mongodb = ["motor>=3.7.1"]  # Deprecated, use DuckDB index instead
```

**Rationale**: Remove optional MongoDB dependency since backend no longer exists.

**Acceptance Criteria**:
- [x] mongodb optional dependency removed
- [x] No motor/pymongo in dependencies
- [x] `uv sync` runs without errors

---

### Task 8.3: Clean Up config.py

**File**: `src/findingmodel/config.py`

**Change** (lines 46-53):
```python
# DELETE these commented lines:
    # DEPRECATED: MongoDB is no longer the default index backend
    # Use DuckDB instead (see duckdb_* settings below)
    # To use MongoDB, install with: pip install findingmodel[mongodb]
    # mongodb_uri: QuoteStrippedSecretStr = Field(default=SecretStr("mongodb://localhost:27017"))
    # mongodb_db: str = Field(default="findingmodels")
    # mongodb_index_collection_base: str = Field(default="index_entries")
    # mongodb_organizations_collection_base: str = Field(default="organizations")
    # mongodb_people_collection_base: str = Field(default="people")
```

**Rationale**: Remove commented MongoDB config since backend no longer exists.

**Acceptance Criteria**:
- [x] Commented MongoDB config removed
- [x] No MongoDB references in config.py
- [x] DuckDB config remains intact

---

### Task 8.4: Update Documentation

**Files**: `README.md`, `docs/database-management.md`, `CHANGELOG.md`

**README.md Changes** (line 543):
```python
# DELETE or UPDATE:
    # Initialize index (connects to MongoDB)

# TO:
    # Initialize index (DuckDB backend)
```

**docs/database-management.md Changes** (lines 248-271):
```markdown
# DELETE entire section:
## MongoDB Backend (Deprecated)

MongoDB is still available but deprecated. To use MongoDB:

1. Install with MongoDB support:
   ```
   pip install findingmodel[mongodb]
   ```

2. Configure connection:
   ```
   MONGODB_URI=mongodb://localhost:27017
   ```

3. Use the `MongoDBIndex` class:
   ```python
   from findingmodel.mongodb_index import MongoDBIndex

   async with MongoDBIndex() as index:
       ...
   ```

**Migration from MongoDB to DuckDB**: Use `index build` to create a DuckDB index from your finding model definition files. MongoDB and DuckDB indexes are maintained separately.
```

**ADD** (brief historical note in same location):
```markdown
> **Historical Note**: Prior to v0.5.0, a MongoDB-based Index implementation was available. This was replaced with DuckDB in v0.5.0. Users needing MongoDB should use findingmodel v0.4.x.
```

**CHANGELOG.md Changes**:

Add to "Removed" section:
```markdown
### Removed

... (existing IdManager entries) ...

- **MongoDB Index backend** (`mongodb_index.py`) - Removed in v0.5.0. DuckDB is now the only Index implementation.
- **`MongoDBIndex` class export** - No longer available. Use `Index` (aliased to `DuckDBIndex`) instead.
- **motor optional dependency** - MongoDB backend no longer available
- **MongoDB configuration fields** - Removed from config (mongodb_uri, mongodb_db, mongodb_*_collection_base)
- **test_index_mongodb.py** - 37 MongoDB-specific tests removed
```

**Acceptance Criteria**:
- [x] README MongoDB comment updated
- [x] database-management.md MongoDB section replaced with historical note
- [x] CHANGELOG.md "Removed" section updated
- [x] Documentation accurately reflects DuckDB-only implementation

---

### Task 8.5: Verify index_validation.py is NOT MongoDB-Specific

**File**: `src/findingmodel/index_validation.py`

**Action**: VERIFY that this file is shared validation using Protocol pattern

**Expected State**:
- Uses `ValidationContext` Protocol
- Works with any index backend (DuckDB, MongoDB, future backends)
- Currently used by DuckDBIndex
- Should be KEPT (not deleted)

**Verification**:
```python
# File should contain:
class ValidationContext(Protocol):
    async def get_existing_oifm_ids(self) -> set[str]: ...
    async def get_existing_names(self) -> set[str]: ...
    async def get_attribute_ids_by_model(self) -> dict[str, str]: ...

# NOT contain MongoDB-specific imports like:
# from motor.motor_asyncio import AsyncIOMotorClient
```

**Acceptance Criteria**:
- [x] Confirmed index_validation.py uses Protocol pattern (VERIFIED)
- [x] No MongoDB-specific imports found (VERIFIED)
- [x] File is shared infrastructure, not MongoDB-specific (VERIFIED)
- [x] Decision: KEEP this file (CONFIRMED)

---

### Task 8.6: Run Full Test Suite

**Commands**:
```bash
uv run pytest test/ -m "not callout" -q  # Should pass (37 fewer tests than before)
task check                                # Format + lint + mypy
```

**Expected Results**:
- ✅ 375 tests pass (412 - 37 MongoDB tests = 375)
- ✅ No import errors
- ✅ No linting/formatting errors
- ✅ No type checking errors

**Acceptance Criteria**:
- [x] All non-MongoDB tests pass (375/375 passing)
- [x] Test count reduced by exactly 37 (MongoDB tests): 412 → 375
- [x] No import errors
- [x] Linting passes
- [x] Type checking passes

---

### Phase 8 Success Criteria

**Complete Removal**:
- [x] `mongodb_index.py` deleted (~810 lines removed)
- [x] `test_index_mongodb.py` deleted (37 tests removed)
- [x] motor optional dependency removed
- [x] MongoDB config cleaned up
- [x] Documentation updated (historical notes only)
- [x] index_validation.py confirmed as shared (kept)
- [x] All 375 remaining tests pass

**Benefits**:
- ✅ ~810 lines of deprecated code removed
- ✅ No MongoDB server dependency
- ✅ Cleaner codebase
- ✅ No confusing dual implementations
- ✅ Clear error messages for anyone trying to import MongoDBIndex

**Timeline**: Single orchestrated execution, ~30 minutes ✅ COMPLETED

**Deliverable**: Complete removal with no backward compatibility burden ✅ DELIVERED

---

## Post-Phase 8: v0.5.0 is Complete

After Phase 8, all major migrations for v0.5.0 are complete:

**✅ Phase 1-3**: DuckDB Index fully implemented (67 passing tests)
**✅ Phase 6**: IdManager deprecated and migrated to Index
**✅ Phase 7**: IdManager completely removed (~240 lines)
**✅ Phase 8**: MongoDB Index completely removed (~810 lines)

**Total Lines Removed in v0.5.0**: ~1050 lines of deprecated code

**Ready for Release**: Single DuckDB-based Index implementation, clean architecture, all tests passing.

