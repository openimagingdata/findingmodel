# Index DuckDB Migration - Final Schema Decisions (2025)

## Summary
Final schema design for migrating FindingModel Index from MongoDB to DuckDB with mandatory hybrid search.

## Key Decisions

### 1. Separate Contributor Junction Tables
**Decision**: Split into `model_people` and `model_organizations` tables instead of polymorphic `model_contributors`.

**Rationale**:
- Cleaner schema - no `contributor_type` discriminator
- Better query performance - no type filtering needed
- Simpler foreign key constraints
- Follows normalized database design principles

**Tables**:
```sql
CREATE TABLE model_people (
    id INTEGER PRIMARY KEY,
    oifm_id VARCHAR NOT NULL REFERENCES finding_models(oifm_id) ON DELETE CASCADE,
    person_id VARCHAR NOT NULL REFERENCES people(id) ON DELETE RESTRICT,
    role VARCHAR NOT NULL,
    display_order INTEGER,
    UNIQUE(oifm_id, person_id, role)
);

CREATE TABLE model_organizations (
    id INTEGER PRIMARY KEY,
    oifm_id VARCHAR NOT NULL REFERENCES finding_models(oifm_id) ON DELETE CASCADE,
    organization_id VARCHAR NOT NULL REFERENCES organizations(id) ON DELETE RESTRICT,
    role VARCHAR NOT NULL,
    display_order INTEGER,
    UNIQUE(oifm_id, organization_id, role)
);
```

### 2. Denormalized Synonyms Table
**Decision**: Create separate `synonyms` table, also include synonyms in `search_text` for FTS.

**Rationale**:
- Fast exact-match lookups for `get(name_or_id)` 
- Synonyms also searchable via FTS in `search_text`
- Consistent with other denormalized tables (attributes, tags)
- No VARCHAR[] array column in main table (cleaner)

**Table**:
```sql
CREATE TABLE synonyms (
    oifm_id VARCHAR NOT NULL,
    synonym VARCHAR NOT NULL,
    PRIMARY KEY (oifm_id, synonym),
    FOREIGN KEY (oifm_id) REFERENCES finding_models(oifm_id) ON DELETE CASCADE
);

CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);
```

**Usage**: 
- Exact match: `SELECT oifm_id FROM synonyms WHERE synonym = ?`
- FTS match: Already in `search_text` field

### 3. Semantic Search Always Enabled
**Decision**: Remove `use_semantic` parameter, embeddings are NOT NULL, OpenAI key REQUIRED.

**Rationale**:
- Semantic search is core feature, not optional
- Simplifies API - no conditional logic
- Ensures consistent search quality
- No need to handle null embeddings

**Changes**:
```python
# OLD API
async def search(
    query: str,
    use_semantic: bool = True,  # ❌ REMOVED
    bm25_weight: float = 0.3,
    semantic_weight: float = 0.7
) -> list[IndexEntry]:

# NEW API
async def search(
    query: str,
    bm25_weight: float = 0.3,
    semantic_weight: float = 0.7
) -> list[IndexEntry]:
    """Semantic search ALWAYS enabled."""
```

**Schema**:
```sql
embedding DOUBLE[1536] NOT NULL  -- text-embedding-3-small, always populated
```

**Config**:
```python
# OpenAI key now REQUIRED (not optional)
# Configured via OPENAI_API_KEY environment variable
```

### 4. Batch Embedding Optimization
**Decision**: `search_batch()` must embed ALL queries in single OpenAI API call.

**Rationale**:
- OpenAI supports batch embedding (more efficient than sequential)
- Reduces API calls from N to 1
- Lower latency for batch operations
- Lower cost (fewer API round-trips)

**Implementation**:
```python
async def search_batch(self, queries: list[str], limit: int = 10) -> dict[str, list[IndexEntry]]:
    """Batch search with optimized embedding.
    
    1. Embed ALL queries in single OpenAI API call
    2. For each query: perform hybrid search with pre-computed embedding
    3. Return dict mapping query → results
    """
    # Single API call for all embeddings
    embeddings = await openai_client.embed_batch(queries)
    
    results = {}
    for query, embedding in zip(queries, embeddings):
        results[query] = await self._search_with_embedding(query, embedding, limit)
    
    return results
```

## Final Schema Summary

**8 Tables Total**:

1. **`finding_models`** - Core metadata
   - Embeddings: `DOUBLE[1536] NOT NULL`
   - Search text: `TEXT NOT NULL` (name + description + synonyms)
   - No synonyms/attributes/tags stored here

2. **`people`** - Normalized person master data
3. **`organizations`** - Normalized organization master data

4. **`model_people`** - Denormalized model→person links
5. **`model_organizations`** - Denormalized model→organization links
6. **`synonyms`** - Denormalized synonyms (also in search_text)
7. **`attributes`** - Denormalized attributes
8. **`tags`** - Denormalized tags

**Indexes**:
- Primary keys (automatic)
- Unique constraints (name, slug_name, filename)
- **FTS index** on search_text (BM25)
- **HNSW index** on embedding (L2, ALWAYS populated)
- Lookup indexes on denormalized tables

**Helper Methods Updated**:
```python
# Split single method into two
async def _upsert_people(self, model: FindingModelFull) -> None
async def _upsert_organizations(self, model: FindingModelFull) -> None

# New method for synonyms
async def _upsert_synonyms(self, model: FindingModelFull) -> None
```

## Migration Impact

**Breaking Changes**:
- `use_semantic` parameter removed from `search()` and `search_batch()`
- OpenAI API key now REQUIRED (was optional)
- Embeddings ALWAYS generated on insert/update

**Benefits**:
- Cleaner API (no conditional semantic search)
- Consistent search quality (always hybrid)
- Better performance (batch embedding, separate contributor tables)
- Simpler implementation (no null embedding handling)

**Implementation Location**: `/Users/talkasab/Repos/findingmodel/tasks/index-duckdb-migration.md`

## Related Memories
- `duckdb_hybrid_search_research_2025` - Research findings
- `index_duckdb_migration_decisions_2025` - Earlier design decisions
