# Anatomic Location Search Implementation

## Overview
Successfully implemented a two-agent Pydantic AI tool for finding anatomic locations for medical imaging findings. This tool searches across multiple ontology databases (anatomic_locations, radlex, snomedct) using LanceDB hybrid search.

## Architecture Decisions

### Two-Agent Pattern
- **Search Agent**: Generates diverse search queries and gathers results from ontology databases (uses smaller model for efficiency)
- **Matching Agent**: Analyzes results and selects best primary and alternate locations based on clinical relevance and specificity (uses larger model for nuanced decisions)

### Reusable Components
- **OntologySearchClient**: Production-ready LanceDB client with proper connection lifecycle management
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
- Optional LanceDB configuration: LANCEDB_URI, LANCEDB_API_KEY
- Uses existing settings pattern from config.py
- Falls back to environment variables if not in settings

## Lessons Learned

### What Worked Well
- Two-agent architecture provides clear separation of concerns
- Reusable components (OntologySearchClient) can be used by other tools
- Dependency injection with SearchContext makes testing easier
- Comprehensive logging helps debug production issues

### Testing Improvements
- Shifting from mocking everything to using TestModel/FunctionModel made tests more meaningful
- Consolidating related tests improved maintainability
- API call prevention guards prevent expensive mistakes during development

## Integration Points
- Works with existing FindingModel structures via IndexCode conversion
- Follows same patterns as find_similar_models() tool
- Uses centralized get_openai_model() from common.py

## Future Considerations
- OntologySearchClient could be extended for other ontology-based searches
- Two-agent pattern could be applied to other complex AI tools
- Consider adding caching for frequently searched terms