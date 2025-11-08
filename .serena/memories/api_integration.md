# API Integration and External Services

## Required API Keys
The project integrates with multiple AI services for enhanced functionality:

### OpenAI API
- **Environment Variable**: `OPENAI_API_KEY`
- **Default Models**: 
  - Main: `gpt-5-mini` (configurable via `OPENAI_DEFAULT_MODEL`)
  - Full: `gpt-5` (configurable via `OPENAI_DEFAULT_MODEL_FULL`)
  - Small: `gpt-5-nano` (configurable via `OPENAI_DEFAULT_MODEL_SMALL`)
- **Used For**:
  - `create_info_from_name()` - Generate finding descriptions
  - `create_model_from_markdown()` - Convert markdown to models
  - Finding similar models
  - All AI-powered tools (when configured as provider)

### Anthropic API
- **Environment Variable**: `ANTHROPIC_API_KEY`
- **Default Models**:
  - Main: `claude-sonnet-4-5` (configurable via `ANTHROPIC_DEFAULT_MODEL`)
  - Full: `claude-opus-4-1` (configurable via `ANTHROPIC_DEFAULT_MODEL_FULL`)
  - Small: `claude-haiku-4-5` (configurable via `ANTHROPIC_DEFAULT_MODEL_SMALL`)
- **Provider Selection**: Set `MODEL_PROVIDER=anthropic` in `.env` or pass `provider="anthropic"` to tool functions
- **Used For**: Alternative to OpenAI for all AI-powered tools
- **Usage Pattern**: `get_model(provider="anthropic")` or configure as default provider

### Tavily API
- **Environment Variable**: `TAVILY_API_KEY`
- **Search Depth**: "basic" or "advanced" (configurable via `TAVILY_SEARCH_DEPTH`)
- **Used For**:
  - `add_details_to_info()` - Enhanced descriptions with citations from trusted radiology sources
  - Research-grade medical information
- **Free Tier**: 1,000 searches/month

## Model Provider Selection Pattern
The project supports multiple LLM providers through a unified interface:
- Configure default provider: `MODEL_PROVIDER=anthropic` or `MODEL_PROVIDER=openai` in `.env`
- Override at runtime: Pass `provider="anthropic"` or `provider="openai"` to tool functions
- Model retrieval: `get_model(provider="anthropic")` returns appropriate model based on configuration
- Fallback: OpenAI is default if no provider specified

## Configuration Management
- Settings loaded from `.env` file or environment variables
- Config accessible via `findingmodel.settings`
- Validation methods:
  - `settings.check_ready_for_openai()`
  - `settings.check_ready_for_anthropic()`
  - `settings.check_ready_for_tavily()`

## Testing with External APIs
- Tests requiring APIs marked with `@pytest.mark.callout`
- Run with `task test-full` to include API tests
- Run with `task test` to exclude API tests (default)

## API Error Handling
- `ConfigurationError` raised if API keys missing when needed
- Graceful fallbacks for optional features
- Detailed error messages for troubleshooting