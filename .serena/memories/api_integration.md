# API Integration and External Services

## Required API Keys
The project integrates with two AI services for enhanced functionality:

### OpenAI API
- **Environment Variable**: `OPENAI_API_KEY`
- **Default Models**: 
  - Main: `gpt-4o` (configurable via `OPENAI_DEFAULT_MODEL`)
  - Small: `gpt-4o-mini` (configurable via `OPENAI_DEFAULT_MODEL_SMALL`)
- **Used For**:
  - `create_info_from_name()` - Generate finding descriptions
  - `create_model_from_markdown()` - Convert markdown to models
  - Finding similar models

### Perplexity API
- **Environment Variable**: `PERPLEXITY_API_KEY`
- **Base URL**: `https://api.perplexity.ai` (configurable)
- **Default Model**: `sonar-pro` (configurable via `PERPLEXITY_DEFAULT_MODEL`)
- **Used For**:
  - `add_details_to_info()` - Enhanced descriptions with citations
  - Research-grade medical information

## MongoDB (Optional)
- **URI**: `mongodb://localhost:27017` (default)
- **Database**: `findingmodels`
- **Collections**:
  - `index_entries` - Finding model index
  - `organizations` - Contributor organizations
  - `people` - Individual contributors
- **Used For**: Advanced indexing functionality

## Configuration Management
- Settings loaded from `.env` file or environment variables
- Config accessible via `findingmodel.settings`
- Validation methods:
  - `settings.check_ready_for_openai()`
  - `settings.check_ready_for_perplexity()`

## Testing with External APIs
- Tests requiring APIs marked with `@pytest.mark.callout`
- Run with `task test-full` to include API tests
- Run with `task test` to exclude API tests (default)

## API Error Handling
- `ConfigurationError` raised if API keys missing when needed
- Graceful fallbacks for optional features
- Detailed error messages for troubleshooting