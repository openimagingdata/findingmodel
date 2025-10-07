# Index DuckDB Migration - Implementation Notes

## Final Design Decisions (Oct 2025)

### Schema: Normalized + Denormalized Hybrid

**Main table:** `finding_models` - core searchable data only
- oifm_id, name, slug_name, filename, file_hash
- description, synonyms (array)
- search_text (for FTS), embedding (for semantic)

**Normalized reference tables:**
- `people` - keyed by github_username
- `organizations` - keyed by code

**Denormalized junction tables:**
- `model_contributors` - bidirectional model ↔ contributor links
- `attributes` - all attributes with model_id + attribute_id (for uniqueness checking)
- `tags` - all tags with model_id (for filtering)

### Key Indexes

1. **HNSW on embeddings** - Fast approximate nearest neighbor (semantic search)
2. **FTS on search_text** - BM25 keyword search  
3. **Unique indexes** - name, slug_name, filename, attribute_id
4. **Foreign key indexes** - CASCADE cleanup performance
5. **Lookup indexes** - tags(tag), model_contributors(contributor_id), attributes(attribute_name)

### Search Implementation

**Hybrid approach:**
- Exact match priority (ID/name/synonym)
- Tag filtering (if provided)
- FTS with BM25 scoring
- Semantic with HNSW approximate NN
- Weighted fusion: 0.3 * normalized_bm25 + 0.7 * cosine_similarity

**No MongoDB backward compatibility:**
- Clean break from MongoDB
- Index rebuilt from `*.fm.json` files anytime
- Removed: motor, pymongo dependencies
- Removed: branch support, atlas_search config

### Migration Approach

Replace `index.py` entirely with new DuckDB implementation.
Run `update_from_directory()` to rebuild from source files.
No gradual migration needed - static data, easily rebuilt.

## Rationale

**Why denormalize?**
- Contributors: Fast "get all models by contributor" queries
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
