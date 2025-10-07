# Index DuckDB Migration - Implementation Notes

## Final Design Decisions (Oct 2025)

### Schema: 8 Tables (Normalized + Denormalized Hybrid)

**Main table:** `finding_models` - core searchable data only
- oifm_id, name, slug_name, filename, file_hash
- description (no synonyms array - separate table)
- search_text (for FTS), embedding **FLOAT[512] NOT NULL** (for semantic)

**Normalized reference tables:**
- `people` - keyed by github_username
- `organizations` - keyed by code

**Denormalized data tables:**
- `model_people` - person contributor links (separate from orgs for cleaner schema)
- `model_organizations` - organization contributor links
- `synonyms` - synonym storage (also in search_text for FTS)
- `attributes` - all attributes with model_id + attribute_id (for uniqueness checking)
- `tags` - all tags with model_id (for filtering)

### Key Indexes

1. **HNSW on embeddings** - Fast approximate nearest neighbor (semantic search)
   - Uses **L2 distance** (default, cosine not supported)
   - Convert to cosine: `cosine_sim = 1 - (l2_distance / 2)`
2. **FTS on search_text** - BM25 keyword search with **'porter' stemmer**
3. **Unique indexes** - name, slug_name, filename, attribute_id
4. **Foreign key indexes** - CASCADE cleanup performance
5. **Lookup indexes** - synonyms(synonym), tags(tag), model_people(person_id), model_organizations(organization_id), attributes(attribute_name)

### Search Implementation

**Hybrid approach (semantic ALWAYS enabled):**
- Exact match priority (ID/name/synonym from tables)
- Tag filtering (if provided)
- FTS with BM25 scoring
- Semantic with HNSW approximate NN (L2 distance)
- Weighted fusion: 0.3 * normalized_bm25 + 0.7 * cosine_similarity
- **Batch embedding**: Single OpenAI API call for multiple queries

**No MongoDB backward compatibility:**
- Clean break from MongoDB
- Index rebuilt from `*.fm.json` files anytime
- Removed: motor, pymongo dependencies
- Removed: branch support, atlas_search config
- Removed: `use_semantic` parameter (always true)

### Migration Approach

Replace `index.py` entirely with new DuckDB implementation.
Run `update_from_directory()` to rebuild from source files.
No gradual migration needed - static data, easily rebuilt.

## Critical Implementation Notes

### ⚠️ HNSW Persistence is EXPERIMENTAL

**Status as of October 2025**: HNSW indexes are experimental for persistent databases.

**Required in setup()**:
```python
self.conn.execute("SET hnsw_enable_experimental_persistence = true;")
```

**Risks**:
- **Data loss or corruption** on unexpected shutdown
- Index memory **does NOT count** toward `memory_limit` setting
- Needs manual capacity planning

**Mitigation**:
- Implement `rebuild_index_from_directory()` for recovery
- Document experimental status
- Keep backups of `.duckdb` file

### Vector Data Type: FLOAT not DOUBLE

**CORRECT**:
```sql
embedding FLOAT[512] NOT NULL  -- ✅ 32-bit single precision, 512 dimensions
```

**Python Conversion Required**:
```python
import numpy as np
openai_embedding = response.data[0].embedding  # float64
duckdb_embedding = np.array(openai_embedding, dtype=np.float32).tolist()
```

### HNSW Index Metric: L2 Only (No Cosine)

**Schema**:
```sql
CREATE INDEX idx USING HNSW (embedding);  -- No metric = L2 default
```

**Query Pattern**:
```sql
SELECT 
    oifm_id,
    1 - (array_distance(embedding, ?) / 2) as cosine_sim,  -- Convert L2 to cosine
    array_distance(embedding, ?) as l2_distance  -- HNSW accelerated
FROM finding_models
ORDER BY array_distance(embedding, ?)
LIMIT 10;
```

### FTS Index Parameters

**Correct syntax**:
```python
PRAGMA create_fts_index(
    'finding_models',
    'oifm_id',
    'search_text',
    stemmer = 'porter',    # ✅ Algorithm name (not 'english')
    stopwords = 'english', # ✅ Language for stopwords
    lower = 1,
    overwrite = 1
);
```

### Extension Loading

**Must load in setup()**:
```python
self.conn.execute("INSTALL fts; LOAD fts;")
self.conn.execute("INSTALL vss; LOAD vss;")
self.conn.execute("SET hnsw_enable_experimental_persistence = true;")
```

### Python Async Pattern

**Reality**: DuckDB Python API is **100% synchronous**.

**Pattern** (matches existing `duckdb_search.py`):
```python
async def search(self, query: str) -> list[IndexEntry]:
    """Async wrapper around sync DuckDB."""
    # DuckDB calls are synchronous but fast (<10ms typically)
    results = self.conn.execute("SELECT ...").fetchall()
    return [self._to_entry(r) for r in results]
```

This is acceptable because:
- DuckDB queries are typically < 10ms
- Matches existing codebase pattern
- No additional dependencies needed

### Batch Embedding Optimization

**Single API call for multiple queries**:
```python
async def search_batch(self, queries: list[str]) -> dict[str, list[IndexEntry]]:
    # ONE API call for all queries
    response = await openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=queries,  # List of strings
        dimensions=settings.openai_embedding_dimensions  # 512
    )
    embeddings = [
        np.array(e.embedding, dtype=np.float32).tolist()
        for e in response.data
    ]
    
    results = {}
    for query, embedding in zip(queries, embeddings):
        results[query] = self._search_with_embedding(query, embedding)
    return results
```

### Embedding Configuration

**Use existing config settings**:
- Model: `settings.openai_embedding_model` (default: "text-embedding-3-small")
- Dimensions: `settings.openai_embedding_dimensions` (default: 512)
- **Matches anatomic location search for consistency**

## Code Consolidation

### DuckDB Common Utilities

**Plan**: Extract shared code to avoid duplication between:
- `src/findingmodel/tools/duckdb_search.py` (anatomic locations)
- `src/findingmodel/duckdb_index.py` (new index)

**Proposed utilities** (`src/findingmodel/tools/duckdb_utils.py`):
1. `setup_duckdb_connection()` - connection with FTS/VSS extensions
2. `get_embedding_for_duckdb()` - embedding with float32 conversion
3. `batch_embeddings_for_duckdb()` - batch embedding API call
4. `normalize_scores()` - min-max normalization
5. `weighted_fusion()` - weighted score combination
6. `rrf_fusion()` - reciprocal rank fusion
7. `l2_to_cosine_similarity()` - distance conversion

**Benefits**:
- Shared, tested utilities
- Consistent embedding handling
- Same config settings usage
- ~200 lines of reusable code

**See**: `tasks/duckdb-common-patterns.md` for detailed plan

### Pending Fixes

**Anatomic location search** should be updated to:
1. Use `settings.openai_embedding_dimensions` (not hardcoded 512)
2. Add float32 conversion (currently missing)
3. Use shared utilities once available

**See**: `tasks/pending-fixes.md` for tracking

## Rationale

**Why denormalize?**
- Separate people/org tables: Cleaner than polymorphic, better queries
- Synonyms: Fast exact-match lookups (also in search_text for FTS)
- Attributes: Fast uniqueness validation during model creation
- Tags: Fast "filter models by tag" in search

**Why HNSW?**
- Proven in anatomic location search
- Much faster than exact cosine similarity for large datasets
- Acceptable accuracy tradeoff for search use case

**Why no MongoDB compat?**
- Index is derived data (can be rebuilt)
- Simpler codebase, fewer dependencies
- No migration complexity needed

**Why semantic always enabled?**
- Core feature, not optional
- Simplifies API (no conditional logic)
- Ensures consistent search quality

**Why 512 dimensions?**
- Matches anatomic location search (consistency)
- Reduces storage (vs 1536 default)
- Sufficient quality for finding model search
- Faster HNSW queries (fewer dimensions)

## Plan Location

`/Users/talkasab/Repos/findingmodel/tasks/index-duckdb-migration.md`

All critical fixes applied and ready for implementation.
