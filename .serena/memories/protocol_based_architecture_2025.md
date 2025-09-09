# Protocol-Based Architecture Pattern (2025-09-09)

## Overview
Implemented Protocol-based architecture for ontology search backends to support multiple data sources (LanceDB vector search and BioOntology REST API) through a clean abstraction layer.

## Key Design Decisions

### Protocol Interface Design
- Used Python's `Protocol` class for structural subtyping (duck typing)
- Defined `OntologySearchProtocol` with standard interface:
  - `async def search()` - main search method
  - `__aenter__` and `__aexit__` - async context manager support
- Allows runtime polymorphism without inheritance

### Implementation Pattern
```python
class OntologySearchProtocol(Protocol):
    async def search(
        self,
        queries: list[str],
        max_results: int = 10,
        filter_anatomical: bool = False,
    ) -> list[OntologySearchResult]:
        ...
    
    async def __aenter__(self) -> Self:
        ...
    
    async def __aexit__(self, *args) -> None:
        ...
```

### Backend Implementations

#### LanceDBOntologySearchClient
- Vector-based semantic search using embeddings
- Searches across local RadLex and SNOMED-CT tables
- Hybrid search combining vector and keyword matching
- Connection lifecycle management with async context manager

#### BioOntologySearchClient  
- REST API client for BioOntology.org
- Access to 800+ medical ontologies
- Semantic type filtering support
- Pagination for comprehensive results
- Connection pooling via httpx AsyncClient

### Auto-Detection and Parallel Execution
- Backends auto-detected based on configuration:
  - LanceDB enabled if `lancedb_uri` configured
  - BioOntology enabled if `bioontology_api_key` configured
- Parallel execution using `asyncio.gather` when multiple backends available
- Results merged and deduplicated across backends

## Benefits

### Flexibility
- Easy to add new search providers
- No changes needed to consuming code
- Mix and match backends based on needs

### Performance
- Parallel backend execution
- Connection pooling and reuse
- ~10 second searches with multiple backends

### Testability
- Clean mocking through Protocol interface
- Backend isolation for unit tests
- Integration tests can target specific backends

### Maintainability
- Clear separation of concerns
- Backend-specific logic isolated
- Consistent interface across providers

## Lessons Learned

### What Worked Well
1. **Protocol over ABC**: Structural subtyping more flexible than inheritance
2. **Auto-detection**: Configuration-based backend selection reduces complexity
3. **Parallel execution**: asyncio.gather maximizes throughput
4. **Context managers**: Clean resource management for connections

### Challenges Overcome
1. **SecretStr handling**: Let backends handle internally, not at call site
2. **Result merging**: Deduplication strategy important for multiple backends
3. **Error handling**: Graceful degradation when backends unavailable

## Future Considerations
- Could add caching layer at Protocol level
- Potential for result ranking across backends
- Backend health checks and circuit breakers
- Metrics collection for backend performance