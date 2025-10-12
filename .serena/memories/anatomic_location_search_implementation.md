# Anatomic Location Search Implementation

## Overview
Successfully implemented a two-agent Pydantic AI tool for finding anatomic locations for medical imaging findings. This tool searches across multiple ontology databases (anatomic_locations, radlex, snomedct) using DuckDB hybrid search.

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
**Configuration**:
```python
# In config.py
duckdb_anatomic_path: str = Field(default="anatomic_locations.duckdb")  # filename only
remote_anatomic_db_url: str | None = Field(default=None)
remote_anatomic_db_hash: str | None = Field(default=None)
```

**Implementation**:
- Uses `importlib.resources.files('findingmodel') / 'data'` to locate package data directory
- Helper: `ensure_db_file(filename, url, hash)` downloads if missing and both URL/hash provided
- Files cached in package installation directory
- SHA256 verification via Pooch library
- Explicit paths still honored (for dev/testing)

### Reusable Components
- **DuckDBOntologySearchClient**: Production-ready DuckDB client with proper connection lifecycle management
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

## Key Implementation Details

### Error Handling
- Try/finally blocks ensure database connections are always cleaned
- Comprehensive logging at key workflow points for debugging
- Graceful handling of empty search results

### Configuration
- DuckDB database path configurable: DUCKDB_ANATOMIC_PATH
- Optional remote download: REMOTE_ANATOMIC_DB_URL, REMOTE_ANATOMIC_DB_HASH
- Uses existing settings pattern from config.py
- Falls back to environment variables if not in settings

## Lessons Learned

### What Worked Well
- Two-agent architecture provides clear separation of concerns
- Reusable components (DuckDBOntologySearchClient) can be used by other tools
- Dependency injection with SearchContext makes testing easier
- Comprehensive logging helps debug production issues
- DuckDB provides excellent performance with simpler deployment than LanceDB

### Testing Improvements
- Shifting from mocking everything to using TestModel/FunctionModel made tests more meaningful
- Consolidating related tests improved maintainability
- API call prevention guards prevent expensive mistakes during development

## Integration Points
- Works with existing FindingModel structures via IndexCode conversion
- Follows same patterns as find_similar_models() tool
- Uses centralized get_openai_model() from common.py

## Future Considerations
- DuckDBOntologySearchClient could be extended for other ontology-based searches
- Two-agent pattern could be applied to other complex AI tools
- Consider adding caching for frequently searched terms
