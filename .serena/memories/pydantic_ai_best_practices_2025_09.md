# Pydantic AI Usage Notes (Sep 2025)
- Prefer `Agent` with explicit `output_type` set to Pydantic models for guaranteed structured responses; adjust instructions instead of post-process validation when possible.
- Testing pattern: use `pytest`, set `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`, and wrap calls in `agent.override(model=TestModel()/FunctionModel)` to simulate LLM output; use fixtures for reuse.
- `TestModel` auto-satisfies JSON schema for rapid tests; reach for `FunctionModel` when deterministic tool arguments or outputs are needed.
- Keep validation that requires async/IO in `@agent.output_validator` rather than duplicating Pydantic model validators; only normalize lightweight formatting outside the data model.
- Adopt Tool Output (default) for structured returns; use `ToolOutput`/`NativeOutput` markers only if model support or behavior demands it.