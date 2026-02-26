# Anatomic Location Search Implementation

## Overview
Successfully implemented a two-agent Pydantic AI tool for finding anatomic locations for medical imaging findings. This tool searches across multiple ontology databases (anatomic_locations, radlex, snomedct) using DuckDB hybrid search.

## CLI Commands

### User CLI (`anatomic-locations`)
```bash
# Search anatomic locations
anatomic-locations search "posterior cruciate ligament"

# Show hierarchy for a location
anatomic-locations hierarchy RID2905

# Show database statistics
anatomic-locations stats
```

### Maintainer CLI (`oidm-maintain`)
Database build/publish lives in `oidm-maintenance` package:
```bash
# Build database from source data
oidm-maintain anatomic build --source /path/to/data.json --output anatomic.duckdb

# Publish to S3
oidm-maintain anatomic publish --db-path anatomic.duckdb
```

## Architecture Decisions

### Two-Agent Pattern
- **Search Agent**: Generates diverse search queries and gathers results from ontology databases (uses smaller model for efficiency)
- **Matching Agent**: Analyzes results and selects best primary and alternate locations based on clinical relevance and specificity (uses larger model for nuanced decisions)

### DuckDB Backend (2025-09)
- Replaced LanceDB with DuckDB for anatomic location search
- Uses HNSW vector indexing for semantic search
- BM25 full-text search on descriptions/synonyms
- Hybrid search with RRF (Reciprocal Rank Fusion)
- Exact match detection with priority (score=1.0)

### Remote Database Downloads (2025-10-11)
**Configuration** (`AnatomicLocationSettings` in `packages/anatomic-locations/src/anatomic_locations/config.py`):
```python
# env_prefix="ANATOMIC_", env_file=".env"
db_path: str | None = None                    # ANATOMIC_DB_PATH
remote_db_url: str | None = None              # ANATOMIC_REMOTE_DB_URL
remote_db_hash: str | None = None            # ANATOMIC_REMOTE_DB_HASH
openai_api_key: SecretStr | None = Field(     # ANATOMIC_OPENAI_API_KEY or OPENAI_API_KEY (AliasChoices)
    default=None,
    validation_alias=AliasChoices("ANATOMIC_OPENAI_API_KEY", "OPENAI_API_KEY"),
)
```

**Key design notes**:
- `env_file=".env"` ensures settings are read from project `.env` (not just exported env vars)
- `AliasChoices` on `openai_api_key` allows fallback to standard `OPENAI_API_KEY`
- Without OpenAI key, search silently degrades to FTS-only (keyword matching)

**Implementation**:
- Helper: `ensure_db_file(filename, url, hash)` downloads if missing and both URL/hash provided
- Files cached in platform-native user data directory via platformdirs
- SHA256 verification via Pooch library
- Explicit paths still honored (for dev/testing)

## AnatomicLocationIndex API (2025-02)

### Base Class
`AnatomicLocationIndex` inherits `ReadOnlyDuckDBIndex` from `oidm_common.duckdb.base`.
- Auto-open: `_ensure_connection()` opens the connection on first use; no explicit `open()` needed.
- Sync + async context managers supported.

### `get()` — Flexible Lookup
```python
index.get(identifier)  # raises KeyError if not found
```
Resolution order (case-insensitive):
1. Direct ID match (`RID2772`)
2. Description match (`"kidney"`)
3. Synonym match (`"renal"`)

### `search_batch()` — Efficient Multi-Query Search
```python
results: dict[str, list[AnatomicLocation]] = await index.search_batch(
    ["knee joint", "liver", "axillary lymph node"],
    limit=5,
)
```
- Batches ALL embedding API calls into one `get_embeddings_batch()` call before running per-query FTS + semantic + RRF pipeline.
- Runs same exact → FTS → semantic → RRF pipeline as `search()` per query.
- Skips blank/whitespace queries; raises `ValueError` if all queries are blank.
- `execute_anatomic_search()` in `findingmodel-ai` uses this for efficiency.

### Reusable Components
- **ReadOnlyDuckDBIndex**: Base class in `oidm_common.duckdb.base` for connection lifecycle
- **OntologySearchResult**: Standardized model for ontology search results with conversion to IndexCode
- **get_openai_model()**: Centralized in common.py for use by all AI tools

## Testing Patterns Established

### Pydantic AI Testing
- Use `models.ALLOW_MODEL_REQUESTS = False` at module level to prevent accidental API calls
- Use `TestModel` for simple deterministic testing
- Use `FunctionModel` for complex controlled behavior testing
- Test actual workflow logic, not library implementation details
- For integration tests, temporarily enable model requests in try/finally block

### Project Conventions
- Demo/proving scripts go in `notebooks/` with `demo_*.py` naming
- Related component tests should be consolidated (e.g., ontology_search tests merged into anatomic_location_search tests)
- Use `@pytest.mark.callout` for tests requiring external API access

### Migration and CLI Testing (2025-10-13)
- 72 comprehensive tests added for migration functions and CLI commands
- Module-scoped fixtures patch embedding generation with deterministic fakes
- Use Click's CliRunner for CLI command testing
- Integration tests use actual 100-record test data file
- Test coverage >90% for new code

## Key Implementation Details

### Error Handling
- Try/finally blocks ensure database connections are always cleaned
- Comprehensive logging at key workflow points for debugging
- Graceful handling of empty search results

### Configuration
- DuckDB database path configurable: `ANATOMIC_DB_PATH`
- Optional remote download: `ANATOMIC_REMOTE_DB_URL`, `ANATOMIC_REMOTE_DB_HASH`
- OpenAI key for semantic search: `OPENAI_API_KEY` or `ANATOMIC_OPENAI_API_KEY` (AliasChoices fallback)
- `env_file=".env"` reads from project `.env` file
- Uses `AnatomicLocationSettings` in `packages/anatomic-locations/src/anatomic_locations/config.py`

### Search Client Refactoring (2025-10-13)
- `duckdb_search.py` now uses `settings.openai_embedding_dimensions` (no hardcoded values)
- All connection/embedding/RRF logic uses common utilities from `duckdb_utils`
- Improved maintainability and consistency

## Lessons Learned

### What Worked Well
- Two-agent architecture provides clear separation of concerns
- Reusable components (DuckDBOntologySearchClient) can be used by other tools
- Dependency injection with SearchContext makes testing easier
- Comprehensive logging helps debug production issues
- DuckDB provides excellent performance with simpler deployment than LanceDB
- CLI commands make database management accessible to non-programmers

### Testing Improvements
- Shifting from mocking everything to using TestModel/FunctionModel made tests more meaningful
- Consolidating related tests improved maintainability
- API call prevention guards prevent expensive mistakes during development
- Deterministic fake embeddings enable reproducible testing

## Integration Points
- Works with existing FindingModel structures via IndexCode conversion
- Follows same patterns as find_similar_models() tool
- Uses centralized get_openai_model() from common.py
