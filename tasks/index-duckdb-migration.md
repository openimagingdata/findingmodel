# Index DuckDB Migration Plan

## Goal
Migrate the Index class from MongoDB to DuckDB with hybrid FTS + vector search, keeping the implementation simple and maintainable.

## Current State

The Index class (`src/findingmodel/index.py`) is 789 lines handling:
- MongoDB storage with 3 collections (index, people, organizations)
- CRUD operations for finding models
- Search by ID, name, or synonym
- File-based batch updates
- Validation and conflict detection
- Basic text search

**Problems:**
- Requires MongoDB server (additional infrastructure)
- No semantic search capability
- Text search is basic pattern matching
- Complex async MongoDB client management
- Over-engineered for simple lookup/search use case

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

## Success Criteria

- [ ] All existing Index tests pass with new DuckDBIndex
- [ ] All denormalized tables properly maintained (contributors, attributes, tags, synonyms)
- [ ] Hybrid search returns relevant results (manual verification with known queries)
- [ ] Tag filtering works correctly in search
- [ ] Attribute ID uniqueness validation works
- [ ] Person and organization lookups work
- [ ] Separate model_people and model_organizations tables work correctly
- [ ] No MongoDB dependency in `pyproject.toml`
- [ ] Search latency < 100ms for typical queries
- [ ] HNSW semantic search faster than exact cosine similarity
- [ ] Index can be rebuilt from directory of `*.fm.json` files
- [ ] `rebuild_index_from_directory()` successfully recovers from corruption
- [ ] Drop/rebuild HNSW strategy works in `update_from_directory()`
- [ ] HNSW index rebuilt successfully after batch updates (< 10 seconds)
- [ ] Read-only mode works by default (no writes possible)
- [ ] Read-write mode works for batch updates
- [ ] Batch embedding optimization works (single OpenAI API call for multiple queries)
- [ ] FLOAT[512] embeddings properly stored and retrieved (matches anatomic location config)
- [ ] Embedding generation uses config settings (model + dimensions)
- [ ] FTS and VSS extensions load correctly on setup
- [ ] No experimental persistence flags needed
- [ ] Documentation updated
- [ ] CLI commands work without changes

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

## References

- Research: Serena memory `duckdb_hybrid_search_research_2025`
- Existing DuckDB implementation: `src/findingmodel/tools/duckdb_search.py`
- Current Index: `src/findingmodel/index.py` (789 lines)
- Test suite: `test/test_index.py` (635 lines, 34 tests)
