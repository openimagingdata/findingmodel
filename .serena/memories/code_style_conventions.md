# Code Style and Conventions

## Python Version
- Python 3.11+ required
- Target version explicitly set in ruff config

## Code Style
- **Line Length**: 120 characters max
- **Formatting**: Handled by ruff format
- **Linting**: ruff with extensive rule sets (bugbear, comprehensions, isort, type annotations, simplify)
- **Type Hints**: Strict type checking with mypy
  - All functions have type annotations
  - Uses `Annotated` types for validation
  - Pydantic models for data structures
  - Custom type aliases (e.g., `NameString`, `AttributeId`)

## Code Patterns
- **Pydantic Models**: BaseModel for all data structures
- **Async/Await**: Used throughout for API calls and database operations
- **Field Validation**: Pydantic Field() with validators
- **Docstrings**: Triple quotes, brief descriptions on classes and key methods
- **Imports**: Organized with isort, type imports separated
- **Constants**: UPPER_SNAKE_CASE (e.g., `ID_LENGTH`, `ATTRIBUTES_FIELD_DESCRIPTION`)

## Logging
- **Framework**: loguru for all logging
- **Singleton**: `from findingmodel import logger`
- **String formatting**: Use f-strings, NOT placeholder syntax
- **Library default**: Logger is disabled in `__init__.py` via `logger.disable("findingmodel")`
- **Application/test activation**: Call `logger.enable("findingmodel")` to see logs
- **Test pattern**: Session-scoped fixture in `conftest.py` enables logging and adds file handler

## Configuration & Secrets
- **Settings class**: `FindingModelAIConfig` in `findingmodel_ai.config` extends Pydantic `BaseSettings`
- **Secrets**: Standard env var names in `.env` (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- **NEVER use os.getenv for config**: All config access through pydantic-settings
- **Logfire**: Token loaded via `settings.configure_logfire()`, NOT from os.environ

## Testing
- pytest with asyncio support
- Test markers: `@pytest.mark.callout` for external API tests
- Test files named `test_*.py`
- Fixtures in conftest.py
- Test data in `test/data/` directory

## Error Handling
- Custom exception classes (e.g., `ConfigurationError`)
- Validation through Pydantic models
- Type checking enforced

## Naming Conventions
- **Classes**: PascalCase (e.g., `FindingModelBase`)
- **Functions/Methods**: snake_case (e.g., `create_info_from_name`)
- **Variables**: snake_case
- **Type Aliases**: PascalCase (e.g., `AttributeId`)
- **Constants**: UPPER_SNAKE_CASE

## Design Principles
- **YAGNI**: Implement only what is required now
- **Process-scoped config**: Configuration is fixed at import time; no runtime singleton replacement

## Package Data Pattern
- **Location**: `src/findingmodel/data/` for package-internal data files
- **Access**: Use `importlib.resources.files('findingmodel') / 'data'` to locate package data directory

## Model Configuration (March 2026)

### Supported Providers

| Provider | Prefix | Config Env Var |
|----------|--------|----------------|
| OpenAI | `openai:` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:` | `ANTHROPIC_API_KEY` |
| Google | `google:` / `google-gla:` / `google-vertex:` | `GOOGLE_API_KEY` |
| Ollama | `ollama:` | `OLLAMA_BASE_URL` |
| Gateway | `gateway/openai:`, `gateway/anthropic:`, `gateway/google:` | `PYDANTIC_AI_GATEWAY_API_KEY` |

### Per-Agent Model Selection (NO TIERS)

There is no tier system. Each agent declares its own model + reasoning in `supported_models.toml` with cross-provider fallback chains.

**Rule: nano for generation, mini for classification, gpt-5.4 for editing.**

| Task Type | Primary Model | Reasoning | Agents |
|-----------|--------------|-----------|--------|
| Generative | `gpt-5.4-nano` | low/none | ontology_search, describe_finding, anatomic_search, similar_plan, describe_details |
| Classification | `gpt-5.4-mini` | none/low | ontology_match, anatomic_select, similar_select, metadata_assign |
| Complex editing | `gpt-5.4` / `claude-opus-4-6` | low/medium | edit_instructions, edit_markdown, import_markdown |

- **Agent model selection**: `settings.get_agent_model("tag")` — no tier parameter
- **Model string for metadata**: `settings.get_effective_model_string("tag")`
- **Environment overrides**: `AGENT_MODEL_OVERRIDES__<tag>=provider:model`
- **Reasoning overrides**: `AGENT_REASONING_OVERRIDES__<tag>=level`
- **Haiku**: Always use `reasoning=none` (extended thinking is catastrophically slow)
- **Gemini Flash**: Fallback, not primary — ~2x slower than GPT-5.4 models
- See `docs/configuration.md` for complete reference