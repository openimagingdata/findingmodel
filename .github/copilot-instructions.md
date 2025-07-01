# Copilot Instructions for FindingModel

## Project Overview

The `findingmodel` package defines and provides tools for Open Imaging Finding Models - structured data models 
for medical imaging findings. This is a Python 3.11+ package for radiology/medical imaging domain, focusing on 
standardized finding definitions with attributes and metadata.

## Core Commands

### Build & Package
```bash
uv build                    # Build package
uv publish                  # Publish to PyPI
```

### Testing
```bash
task test                   # Run tests (excludes callout tests)
task test-full              # Run all tests including external API calls
task test -- test_name      # Run specific test
uv run pytest test/test_specific.py::test_function  # Single test
```

### Code Quality
```bash
task check                  # Format, lint, and type check
task quiet                  # Run all checks silently
uv run ruff format          # Format code
uv run ruff check --fix     # Lint and auto-fix
uv run mypy src             # Type checking
```

### CLI Usage
```bash
python -m findingmodel config              # Show configuration
python -m findingmodel make-info           # Generate finding descriptions
python -m findingmodel make-stub-model     # Create basic finding model
python -m findingmodel fm-to-markdown      # Convert JSON to Markdown
python -m findingmodel markdown-to-fm      # Convert Markdown to JSON
```

## Architecture

### Core Components
- **`finding_model.py`**: Main data models (`FindingModelBase`, `FindingModelFull`)
- **`finding_info.py`**: Metadata wrapper (`FindingInfo`)
- **`index.py`**: MongoDB-based finding model index system
- **`cli.py`**: Click-based command-line interface
- **`tools/`**: AI-powered generation tools using OpenAI/Perplexity APIs
- **`config.py`**: Pydantic settings with environment variable support

### Data Models
- **`FindingModelBase`**: Basic finding with name, description, attributes
- **`FindingModelFull`**: Extended model with IDs, contributors, metadata
- **Attributes**: `ChoiceAttribute`, `NumericAttribute` (with ID variants)
- **Index**: MongoDB-based metadata search for finding model repositories

### External Dependencies
- **OpenAI API**: For description generation and model creation (`OPENAI_API_KEY`)
- **Perplexity API**: For detailed finding research (`PERPLEXITY_API_KEY`)
- **MongoDB**: For finding model indexing and metadata storage (`MONGODB_URI`)
- **Pydantic AI**: LLM interactions with structured outputs

## Style & Conventions

### Code Formatting
- **Line length**: 120 characters
- **Formatter**: Ruff with preview features enabled (config in `pyproject.toml`)
- **Import organization**: Also organize imports with Ruff isort (configuration in `pyproject.toml`)
- **Target**: Python 3.11+, but expected to be used in Python 3.12+ environments

### Type Annotations
- **Strict typing**: MyPy strict mode enabled
- **Required**: All function signatures and class attributes
- **Pydantic models**: Use Field() for descriptions and validation
- **Async**: Functions interacting with APIs should be async

### Naming Conventions
- **Variables/functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Leading underscore `_private`
- **OIFM IDs**: Format `OIFM_{SOURCE}_{6_DIGITS}`; SOURCE is 3-4 uppercase letters (e.g., `OIFM_MSFT_134126`)

### Error Handling
- Use custom exception types (e.g., `ConfigurationError`)
- Validate API keys and configuration at startup
- Graceful degradation for optional external services

### Testing
- **Markers**: Use `@pytest.mark.callout` for tests requiring external APIs
- **Async**: Use `pytest-asyncio` for async test functions
- **Fixtures**: Separate test data in `test/data/` directory
- **Coverage**: Focus on core business logic, mock external API calls

### Documentation
- **Docstrings**: Use for public APIs and complex functions
- **README**: Keep examples current with actual API
- **Notebooks**: Demonstrate usage in `notebooks/` directory

## File Organization

```
src/findingmodel/
├── __init__.py           # Main exports
├── finding_model.py      # Core data models
├── finding_info.py       # Metadata models
├── index.py             # Repository indexing
├── cli.py               # Command-line interface
├── config.py            # Settings and configuration
├── tools/               # AI-powered tools
│   ├── tools.py         # Main tool functions
│   ├── add_ids.py       # ID generation
│   └── prompt_templates/ # LLM prompts
test/                    # Test suite
notebooks/               # Usage examples
```

## Domain-Specific Rules

### Medical Imaging Context
- Findings represent observable conditions in medical images
- Attributes describe measurable or categorical properties
- Standard vocabularies: RadLex, SNOMED-CT for coding
- Validation ensures clinical accuracy and completeness

### ID Management
- OIFM IDs are immutable once assigned
- Source prefixes identify contributing organizations
- Duplicate ID validation prevents conflicts
- Hierarchical structure: Model → Attribute → Value

### AI Tool Integration
- OpenAI for natural language generation
- Structured outputs using Pydantic models
- Rate limiting and error handling for API calls
- Fallback behavior when APIs unavailable