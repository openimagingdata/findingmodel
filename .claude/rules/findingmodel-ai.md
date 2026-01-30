---
paths: "packages/findingmodel-ai/**"
---

# findingmodel-ai Package Rules

## Purpose

AI-powered tools for finding model authoring: info generation, model editing, ontology search, anatomic location discovery.

## AI Provider Configuration

See Serena `code_style_conventions` for provider table and model tiers.

- Configure via `DEFAULT_MODEL` env var (e.g., `openai:gpt-4o`)
- Per-agent overrides: `AGENT_MODEL_OVERRIDES__<tag>=provider:model`
- See `docs/configuration.md` for complete reference

## Architecture Patterns

- **Pydantic AI**: All agents use pydantic-ai with multi-provider support
- **Two-agent patterns**: Complex workflows use search + matching agent pairs
- **Structured outputs**: Agents return typed Pydantic models, not raw text
- **Protocol-based backends**: See Serena `protocol_based_architecture_2025`

## Key Tools

- `create_info_from_name()` – generate FindingInfo from a finding name
- `edit_model_natural_language()` / `edit_model_markdown()` – AI-powered model editing
- `find_anatomic_locations()` – two-agent anatomic location discovery
- `match_ontology_concepts()` – multi-backend ontology search

## Testing

### Unit Tests
- Use `TestModel`/`FunctionModel` for deterministic AI agent behavior (see Serena `pydantic_ai_testing_best_practices`)
- Run with `task test`

### Integration Tests
- Marked with `@pytest.mark.callout` for real API calls
- Run with `task test-full`

### Evals
- Assess behavioral quality (0.0-1.0 scores with partial credit)
- `Dataset.evaluate()` with focused evaluators
- Run with `task evals` or `task evals:model_editor`
- See `evals/CLAUDE.md` for eval development guidance

**Key distinction**: Tests verify correctness (pass/fail), evals assess quality.

## Required Environment

```bash
# At least one AI provider key
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...

# Optional: Enhanced search with citations
TAVILY_API_KEY=...

# Optional: Local models
OLLAMA_BASE_URL=...
```

## Serena References

- `pydantic_ai_best_practices_2025_09` – agent implementation patterns
- `pydantic_ai_testing_best_practices` – testing with TestModel/FunctionModel
- `ontology_search_architecture` – multi-backend search architecture
- `anatomic_location_search_implementation` – two-agent discovery workflow
