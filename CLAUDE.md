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

- `create_info_from_name()`: Generate FindingInfo from a finding name using OpenAI
- `add_details_to_info()`: Enhance FindingInfo with detailed descriptions using Perplexity
- `create_model_from_markdown()`: Convert markdown outlines to FindingModel objects
- `create_model_stub_from_info()`: Generate basic model with presence/change attributes
- `add_ids_to_model()`: Add OIFM identifiers to models
- `add_standard_codes_to_model()`: Add RadLex and SNOMED-CT codes

### Key Configuration
Configuration is managed through `src/findingmodel/config.py`:
- Reads from `.env` file or environment variables
- Required for AI features: `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`
- Optional MongoDB configuration for index functionality

### Index System
The `Index` class (index.py) provides MongoDB-based indexing of finding model definitions stored as `.fm.json` files in a `defs/` directory structure. It manages:
- Finding models in a MongoDB collection with proper indexing for fast lookup by ID, name, or synonym
- Separate collections for people and organizations (contributors)
- Full-text search capabilities using MongoDB text indexes
- Batch operations for efficient directory synchronization
- Validation to prevent duplicate IDs, names, and attribute IDs

## Testing Approach

Tests are organized in `test/` directory:
- Tests marked with `@pytest.mark.callout` require external API access
- Use `task test` to run local tests only
- Use `task test-full` to include API integration tests
- Test data fixtures are in `test/data/`

## API Keys and Environment

Create a `.env` file for API keys:
```
OPENAI_API_KEY=your_key_here
PERPLEXITY_API_KEY=your_key_here
```

The library uses these for AI-powered finding model generation and enhancement features.