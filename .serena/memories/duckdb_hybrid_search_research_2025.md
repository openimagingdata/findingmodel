# DuckDB Hybrid Search Best Practices (2025)

## Overview
Research for migrating FindingModel Index from MongoDB to DuckDB with hybrid FTS + vector search.

## Key Technical Approaches

### 1. Full-Text Search (FTS)
- Use DuckDB FTS extension with BM25 scoring
- Create FTS index with customizable stemming and stopwords:
  ```sql
  PRAGMA create_fts_index('table_name', id_column, 'text_column',
    stemmer = 'english',
    stopwords = 'english_stopwords',
    lower = 1,
    overwrite = 1
  );
  ```
- BM25 provides relevance scoring for keyword matches
- FTS indexes must be rebuilt on data updates

### 2. Vector Embeddings
- Store embeddings as DOUBLE[] array columns
- Use cosine similarity for semantic search
- Generate embeddings via OpenAI (already in codebase)
- Vector search complements keyword search for semantic understanding

### 3. Hybrid Search Fusion
Two main approaches for combining scores:

**Convex Combination (Weighted Sum):**
```
hybrid_score = λ * normalized_bm25 + (1-λ) * cosine_similarity
```
- Tune λ (0.3 for BM25, 0.7 for semantic is common starting point)
- Requires min-max normalization of BM25 scores to [0,1] range

**Reciprocal Rank Fusion (RRF):**
```
hybrid_rank = 1/rank_fts + 1/rank_embedding
```
- No normalization needed
- More robust to score scale differences

**Best Practice from DuckDB docs:**
- Prioritize exact matches (score = 1.0)
- Normalize BM25 scores using min-max: `(score - min) / (max - min)`
- Apply weighted combination: `0.3 * bm25 + 0.7 * cosine_sim`
- Use SQL window functions for normalization

### 4. Schema Design Best Practices

**Column Types:**
- Use most restrictive types possible (avoid generic strings for metadata)
- JSONB columns supported but keep shallow - extract frequently-queried fields to typed columns
- Store embeddings as DOUBLE[] arrays

**Indexing:**
- DuckDB auto-provides Min-Max indexes
- Manual ART (Adaptive Radix Tree) indexes useful for:
  - Primary keys
  - Highly selective filters
  - Point lookups
- Only create ART index if entire index fits in memory
- All indexes persist to disk

**Constraints:**
- Avoid PRIMARY KEY/UNIQUE constraints unless needed for integrity
- They slow bulk loads without analytical query benefit
- Add after initial data load if needed

**JSONB Handling:**
- Use `ignore_errors=true` for schema-variable JSON imports
- Extract critical searchable fields to dedicated typed columns
- Keep JSON shallow (avoid deep nesting)

## Implementation Pattern from Anatomic Location Search

The existing `DuckDBOntologySearchClient` demonstrates proven patterns:
- Load FTS and VSS extensions on connect
- Separate methods for exact match, FTS, and vector search
- RRF fusion for combining results
- Deduplication by concept_id
- Async context manager pattern

## References
- DuckDB Text Analytics Workflows (2025): https://duckdb.org/2025/06/13/text-analytics.html
- DuckDB FTS Extension docs
- Existing implementation: `src/findingmodel/tools/duckdb_search.py`
