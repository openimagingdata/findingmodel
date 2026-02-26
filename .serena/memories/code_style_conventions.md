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
- **Settings class**: `FindingModelConfig` extends Pydantic `BaseSettings` with `env_prefix="FINDINGMODEL_"`
- **Access**: `from findingmodel.config import get_settings` (lazy singleton, auto-loads `.env`)
- **Secret fields**: Use `SecretStr | None` with `AliasChoices` for shared keys (e.g., `OPENAI_API_KEY`)
- **NEVER use os.getenv**: All config access through `get_settings().*` - ensures validation and type safety
- **Validation**: Call `get_settings().validate_default_model_keys()` at app startup for fail-fast

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
- **YAGNI**: "You Aren't Going To Need It" - implement only what is required now
  - Avoid speculative features, complex versioning systems, or abstractions until they're proven necessary
  - Keep implementations simple and focused on current requirements
  - Example: Pooch integration (2025-10-11) rejected complex versioning in favor of simple config-driven downloads

## Package Data Pattern
- **Location**: `src/findingmodel/data/` for package-internal data files
- **Access**: Use `importlib.resources.files('findingmodel') / 'data'` to locate package data directory
- **Version control**: Add `.gitignore` for large files (e.g., `*.duckdb`)
- Works correctly in pip install, editable install, and venv scenarios

## File Organization
- One main concept per file
- Related utilities grouped in tools/ directory
- Test files mirror source structure

## Model Configuration

### Supported Providers

| Provider | Prefix | Config Env Var |
|----------|--------|----------------|
| OpenAI | `openai:` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:` | `ANTHROPIC_API_KEY` |
| Google (GLA) | `google:` or `google-gla:` | `GOOGLE_API_KEY` |
| Ollama | `ollama:` | `OLLAMA_BASE_URL` |
| Gateway | `gateway/openai:`, `gateway/anthropic:`, `gateway/google:` | `PYDANTIC_AI_GATEWAY_API_KEY` |

### Tier-Based Selection

| Tier | Default | Use Case |
|------|---------|----------|
| `small` | `openai:gpt-5-nano` | Simple classification, query generation |
| `base` | `openai:gpt-5-mini` | Most agent workflows |
| `full` | `openai:gpt-5.2` | Complex reasoning, editing |

Access via: `settings.get_model("base")` or `settings.get_model("small")`

### Coding Rules
- **ModelSpec Type**: Use validated `ModelSpec` type for model strings (see `MODEL_SPEC_PATTERN` in config.py)
- **Test Constants**: Use `TEST_OPENAI_MODEL`, `TEST_ANTHROPIC_MODEL`, `TEST_GOOGLE_MODEL` from conftest.py (cheapest models for testing)
- **Never hard-code**: Avoid hard-coded model strings like `"openai:gpt-4o-mini"` in test/production code
- **Gateway tests**: Construct gateway models from constants: `f"gateway/{TEST_OPENAI_MODEL}"`
- **API Key Validation**: Use `settings.validate_default_model_keys()` at app startup for fail-fast behavior

### Per-Agent Model Configuration
- **AgentTag type**: Use `AgentTag` Literal for valid agent identifiers (14 tags, `{domain}_{verb}` pattern)
- **Agent model selection**: Use `settings.get_agent_model("tag", default_tier="base")` in agent factories
- **Model string for metadata**: Use `settings.get_effective_model_string("tag", "base")` when recording which model was used
- **Environment overrides**: Users configure via `AGENT_MODEL_OVERRIDES__<tag>=provider:model`
- See `docs/configuration.md` for complete tag reference by workflow