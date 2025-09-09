# BioOntology API Integration (2025-09-09)

## Overview
Integrated BioOntology.org REST API as a search backend for medical ontology concepts, providing access to 800+ medical ontologies including SNOMED-CT, ICD-10, LOINC, RxNorm, and more.

## Implementation Details

### BioOntologySearchClient
- Async/await implementation with httpx AsyncClient
- Connection pooling for efficient API usage
- Supports both single page and paginated searches
- Semantic type filtering for targeted results

### Key Features
1. **Search Methods**:
   - `search_bioontology()`: Single page search with configurable page size
   - `search_all_pages()`: Paginated search up to max_results
   - `search_as_ontology_results()`: Returns standardized OntologySearchResult format
   - `search()`: Protocol-compliant interface method

2. **Result Processing**:
   - Extracts ontology name from concept IDs
   - Preserves synonyms, definitions, and semantic types
   - Provides UI links for BioPortal browsing
   - Converts to standardized IndexCode format

3. **Configuration**:
   - API key stored as SecretStr in settings
   - Configurable ontology filters (e.g., ["SNOMEDCT", "RADLEX"])
   - Semantic type filtering (e.g., T047 for diseases)
   - Page size and max results limits

### API Response Mapping
```python
BioOntologySearchResult:
  concept_id: Full URI from @id
  ontology: Extracted from URI path
  pref_label: From prefLabel field
  synonyms: Array from synonym field
  definition: Optional from definition field
  semantic_types: Array from semanticType field
  ui_link: BioPortal UI link
```

### Integration with Protocol
- Implements OntologySearchProtocol interface
- Works seamlessly with LanceDBOntologySearchClient
- Automatic parallel execution when both configured
- Results merged and deduplicated

## Performance Characteristics
- ~2-5 seconds for single page (50 results)
- ~5-10 seconds for multi-page searches
- Connection pooling reduces latency
- Parallel execution with other backends

## Configuration Requirements
```bash
# .env file
BIOONTOLOGY_API_KEY=your_api_key_here
```

## Testing Approach
- Protocol compliance tests verify interface
- Integration tests marked with @pytest.mark.callout
- Mock API responses for unit tests
- SecretStr handling tests included

## Common Use Cases
1. **Disease/Condition Search**: Finding SNOMED-CT codes
2. **Radiology Terms**: RadLex concept matching
3. **Cross-Ontology Search**: Finding concepts across multiple sources
4. **Semantic Filtering**: Limiting to specific concept types

## Error Handling
- 401 Unauthorized: Invalid or missing API key
- 429 Rate Limited: Automatic retry with backoff
- Network errors: Graceful degradation
- Invalid ontology: Skip and continue

## Advantages Over LanceDB
- Always up-to-date (live API)
- Broader ontology coverage
- No local storage requirements
- Official medical terminology sources

## Limitations
- Requires internet connectivity
- API rate limits apply
- Slightly slower than local search
- Requires API key registration