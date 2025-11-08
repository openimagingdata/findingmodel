# Pydantic AI Usage Notes (Updated Nov 2025)

## Multi-Provider Architecture

### Tier-Based Model Selection
- Use `get_model(model_tier, provider=None)` from `findingmodel.tools.common`
- Three tiers: `"small"` (fast/cheap), `"base"` (default capability), `"full"` (most capable)
- Two providers: `"openai"` (default) or `"anthropic"` - controlled by `settings.model_provider`
- Type-safe with `ModelProvider = Literal["openai", "anthropic"]` and `ModelTier = Literal["base", "small", "full"]`
- **Design choice**: Use "base" not "default" as tier name (more descriptive, less ambiguous)
- **Design choice**: No `model_name` parameter - tier-based selection enforces provider portability

### Provider Configuration
- All tool functions accept optional `provider` parameter to override default
- Provider instances created explicitly with API keys: `OpenAIProvider(api_key=...)`, `AnthropicProvider(api_key=...)`
- Never use provider-specific model names in public APIs - always use tier-based selection
- Default models: OpenAI (gpt-5-mini, gpt-5, gpt-5-nano), Anthropic (claude-sonnet-4-5, claude-opus-4-1, claude-haiku-4-5)

## Agent Pattern
- Prefer `Agent` with explicit `output_type` set to Pydantic models for guaranteed structured responses
- Adjust instructions instead of post-process validation when possible
- Use `Agent[DepsType, OutputType]` for type-safe dependency injection and output handling

## Testing Pattern
- Set `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` at module level in test files
- Use `agent.override(model=TestModel()/FunctionModel())` to simulate LLM output
- `TestModel` auto-satisfies JSON schema for rapid tests
- `FunctionModel` for deterministic tool arguments or outputs
- Use fixtures for reusable test setup
- Integration tests: mark with `@pytest.mark.callout`, set `ALLOW_MODEL_REQUESTS = True` in try/finally blocks

## Validation and Output
- Keep validation that requires async/IO in `@agent.output_validator` rather than duplicating Pydantic model validators
- Only normalize lightweight formatting outside the data model
- Adopt Tool Output (default) for structured returns
- Use `ToolOutput`/`NativeOutput` markers only if model support or behavior demands it

## Configuration
- Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_DEFAULT_MODEL`, `ANTHROPIC_DEFAULT_MODEL_FULL`, `ANTHROPIC_DEFAULT_MODEL_SMALL`
- OpenAI: `OPENAI_API_KEY`, `OPENAI_DEFAULT_MODEL`, `OPENAI_DEFAULT_MODEL_FULL`, `OPENAI_DEFAULT_MODEL_SMALL`
- Provider selection: `MODEL_PROVIDER` (defaults to "openai")