# Ontology Search Architecture

Consolidated reference for ontology concept search patterns in findingmodel.

## Overview

The ontology search system provides multi-backend search across medical terminologies (SNOMEDCT, RadLex, LOINC, etc.) with AI-powered categorization.

## Architecture

### Protocol-Based Backend Support
- Python Protocol for structural subtyping (duck typing)
- `OntologySearchProtocol` interface with `search()` and async context manager
- Auto-detection based on configuration
- Parallel execution with `asyncio.gather` when multiple backends available

### Current Backends
- **BioOntologySearchClient** – REST API to BioOntology.org (800+ ontologies)
- **DuckDBOntologySearchClient** – Local vector search with hybrid FTS + semantic

### One-Agent Pattern
- Previous: 3 agents with heavy LLM usage (70+ seconds)
- Current: 1 agent for categorization + programmatic query generation (~10 seconds)
- 85% performance improvement by reducing LLM calls

## Search Flow

1. **Generate query terms** – Programmatic generation (not LLM):
   - Add "disease", "syndrome", "condition" suffixes
   - Create partial combinations for multi-word terms
2. **Execute searches** – Parallel across configured backends with normalization
3. **Categorize with LLM** – Single focused categorization agent
4. **Post-process** – Ensure exact matches, respect max_length constraints

## Key Functions

- `normalize_concept()` – Cleans text for deduplication
- `ensure_exact_matches_post_process()` – Guarantees exact matches aren't missed
- `execute_ontology_search()` – Handles search and deduplication
- `create_categorization_agent()` – Simple agent for categorization only

## Configuration

### Default Ontology Set
- Limited to 3 core ontologies: SNOMEDCT, RADLEX, LOINC
- Can override via `ontologies` parameter when needed
- Reduced from 6 (improved performance ~50%)

### SNOMEDCT Prioritization
- AI categorization prompt explicitly favors SNOMEDCT matches
- International standard for clinical terminology

### Cohere Reranking
- Disabled by default (`use_cohere_with_ontology_concept_match=False`)
- Adds 2-3 seconds latency
- Enable when precision more important than speed

## Text Normalization

- Remove TRAILING parenthetical content only
- Preserve important middle parentheses (e.g., "Calcium (2+)")
- Handle RadLex multi-line formatting issues
- Remove content after colons for RadLex results

## Testing Patterns

### Unit Tests
- Mock external services, test actual logic
- Protocol compliance verification
- Default configuration tests

### Integration Tests
- Marked `@pytest.mark.callout`
- Test real BioOntology API integration

## Lessons Learned

1. **Programmatic > LLM** for deterministic tasks
2. **Post-processing** guarantees business rules outside of LLM
3. **Default to performance**, allow optional features
4. **Output validators** should ONLY validate, never transform data
