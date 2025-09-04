# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The `findingmodel` package is a Python library for managing Open Imaging Finding Models - structured data models used to describe medical imaging findings in radiology reports. The library provides tools for creating, converting, and managing these finding models with OpenAI/Perplexity API integration.

## Development Commands

### Core Commands (using Task)
```bash
# Run tests (excluding external API calls)
task test

# Run full test suite (including API integration tests)  
task test-full

# Run specific test
task test -- test/test_findingmodel.py::TestClass::test_method

# Lint and format code
task check

# Build the package
task build

# Run all checks quietly
task quiet
```

### Alternative Commands (using uv directly)
```bash
# Run tests
uv run pytest -rs -m "not callout"

# Format code
uv run ruff format

# Lint with auto-fix
uv run ruff check --fix  

# Type checking
uv run mypy src

# Build package
uv build
```

## Architecture Overview

### Core Data Models
The library is built around a hierarchy of finding model classes:

- **`FindingInfo`** (finding_info.py): Basic finding information with name, description, synonyms, and citations
- **`FindingModelBase`** (finding_model.py): Core finding model with attributes but no IDs
- **`FindingModelFull`** (finding_model.py): Complete model with OIFM IDs, index codes, and contributor metadata
- **Attributes**: Two types - `ChoiceAttribute` (discrete values) and `NumericAttribute` (ranges/measurements)

### Tools Module (`src/findingmodel/tools/`)
AI-powered tools for working with finding models:

#### Core Finding Model Tools
- `create_info_from_name()`: Generate FindingInfo from a finding name using OpenAI
- `add_details_to_info()`: Enhance FindingInfo with detailed descriptions using Perplexity
- `create_model_from_markdown()`: Convert markdown outlines to FindingModel objects
- `create_model_stub_from_info()`: Generate basic model with presence/change attributes
- `add_ids_to_model()`: Add OIFM identifiers to models
- `add_standard_codes_to_model()`: Add RadLex and SNOMED-CT codes
- `find_similar_models()`: Find existing similar models using two-agent search and analysis

#### Anatomic Location Search Tool
- `find_anatomic_locations()`: Two-agent workflow for finding anatomic locations for findings
  - **Search Agent**: Generates diverse search queries and gathers results from ontology databases
  - **Matching Agent**: Selects best primary and alternate locations based on specificity
- `OntologySearchClient`: Reusable LanceDB client for medical terminology hybrid search
- `OntologySearchResult`: Standardized model for ontology search results with conversion to IndexCode

#### Ontology Concept Search Tool
- `search_ontology_concepts()`: High-performance search for relevant medical concepts
  - Generates query terms programmatically for efficiency (~10s vs 70s with LLM)
  - Searches across RadLex and SNOMED-CT ontologies
  - Categorizes results into exact matches, should include, and marginal
  - Post-processing ensures exact matches are never missed
  - Excludes anatomical concepts (use anatomic location search instead)

#### Common Utilities (`common.py`)
- `get_openai_model()`: Centralized OpenAI model instance creation (used by all AI tools)

### Key Configuration
Configuration is managed through `src/findingmodel/config.py`:
- Reads from `.env` file or environment variables
- Required for AI features: `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`
- Optional MongoDB configuration for index functionality
- Optional LanceDB configuration for anatomic location search: `LANCEDB_URI`, `LANCEDB_API_KEY`

### Index System
The `Index` class (index.py) provides MongoDB-based indexing of finding model definitions stored as `.fm.json` files in a `defs/` directory structure. It manages:
- Finding models in a MongoDB collection with proper indexing for fast lookup by ID, name, or synonym
- Separate collections for people and organizations (contributors)
- Full-text search capabilities using MongoDB text indexes
- Batch operations for efficient directory synchronization
- Validation to prevent duplicate IDs, names, and attribute IDs

## Testing Approach

Tests are organized in `test/` directory using **pure pytest** (not unittest):
- Tests marked with `@pytest.mark.callout` require external API access
- Use `task test` to run local tests only
- Use `task test-full` to include API integration tests
- Test data fixtures are in `test/data/`
- Demo/proving scripts go in `notebooks/` with `demo_*.py` naming convention
- Tests should be organized by module (e.g., all ontology_concept_search tests in one file)

### Pydantic AI Testing Patterns
When testing Pydantic AI agents:
- Add `from pydantic_ai import models; models.ALLOW_MODEL_REQUESTS = False` to prevent accidental API calls
- Use `TestModel` for simple deterministic testing of agent behavior
- Use `FunctionModel` for complex controlled behavior testing
- Test actual workflow logic, not library implementation details
- For integration tests, temporarily enable model requests in a try/finally block

## Best Practices and Patterns

### Two-Agent Architecture Pattern
When building complex AI tools, consider using a two-agent pattern:
1. **Search/Gather Agent**: Uses smaller models to collect information efficiently
2. **Analysis/Decision Agent**: Uses larger models to make nuanced decisions

This pattern is used in both `find_similar_models()` and `find_anatomic_locations()`.

### Pydantic AI Patterns (IMPORTANT)
Follow these patterns when using Pydantic AI:
1. **Output Validators are for VALIDATION only**: Never transform data in validators
   - ❌ BAD: Modifying output in validator and returning changed data
   - ✅ GOOD: Using post-processing functions after agent returns
2. **Use structured outputs**: Define clear Pydantic models for agent outputs
3. **Prefer programmatic processing**: When possible, use deterministic code instead of LLMs
4. **Agent simplicity**: Keep agents focused on judgment tasks, not data manipulation

### Reusable Components
- Extract common functionality into shared modules (e.g., `OntologySearchClient`)
- Centralize model creation logic (e.g., `get_openai_model()` in `common.py`)
- Use dependency injection for testability (e.g., `SearchContext` dataclasses)

### Performance Optimization
- Profile before optimizing (use cProfile or similar)
- Reduce LLM prompt sizes by preprocessing data (e.g., normalization)
- Use programmatic query generation instead of LLM-based when possible
- Batch database queries to reduce round trips
- Consider smaller models for simple tasks (gpt-4o-mini vs gpt-4)

### Error Handling
- Always use try/finally blocks for resource cleanup (database connections, etc.)
- Log important workflow steps for debugging production issues
- Provide graceful fallbacks when external services are unavailable

## API Keys and Environment

Create a `.env` file for API keys:
```
# Required for AI features
OPENAI_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here

# Optional for ontology search features
LANCEDB_URI=your_lancedb_uri_here
LANCEDB_API_KEY=your_lancedb_api_key_here

# Optional for MongoDB index
MONGODB_URI=your_mongodb_uri_here
```

The library uses these for AI-powered finding model generation and enhancement features.