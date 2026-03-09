# Pydantic AI Usage Notes (Updated Nov 2025)

## Multi-Provider Architecture

### Model Selection (findingmodel_ai.config)
- Use `settings.get_model("base")` / `settings.get_model("small")` / `settings.get_model("full")` for tier-based selection
- Use `settings.get_agent_model("agent_tag", default_tier="base")` for per-agent model selection
- Per-agent overrides via env: `AGENT_MODEL_OVERRIDES__<tag>=provider:model`
- Model string for metadata: `settings.get_effective_model_string("tag", "base")`
- Type-safe with `ModelTier = Literal["base", "small", "full"]` and `AgentTag` Literal

### Provider Configuration
- Model spec format: `provider:model-name` (e.g., `openai:gpt-5.4`, `google-gla:gemini-3-flash-preview`)
- API keys loaded from `.env` via pydantic-settings, then passed explicitly to Pydantic AI providers
- Gateway fallback: if direct provider key is missing but `PYDANTIC_AI_GATEWAY_API_KEY` is set, routes through gateway automatically
- Google prefixes (`google:`, `google-gla:`, `google-vertex:`) are interchangeable — routes based on available keys
- Per-tier reasoning levels (`DEFAULT_REASONING_SMALL/BASE/FULL`) normalized per-provider via `supported_models.toml`
- Anthropic Opus 4.6+ uses adaptive thinking; older models use budget_tokens

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
- API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `PYDANTIC_AI_GATEWAY_API_KEY`
- Model tiers: `DEFAULT_MODEL` (base), `DEFAULT_MODEL_FULL`, `DEFAULT_MODEL_SMALL`
- Reasoning: `DEFAULT_REASONING_SMALL`, `DEFAULT_REASONING_BASE`, `DEFAULT_REASONING_FULL`
- All config via `FindingModelAIConfig` in `findingmodel_ai.config` (pydantic-settings, loads `.env`)