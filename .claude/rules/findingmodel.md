---
paths: "packages/findingmodel/**"
---

# findingmodel Package Rules

## Purpose
Core models (FindingModelBase, FindingModelFull, FindingInfo), Index API, MCP server, and fm-tool CLI.

## Constraints
- **No AI dependencies**: Core package has no pydantic-ai. AI tools live in findingmodel-ai.
- **Pydantic v2**: All models use Pydantic v2 with strict typing.
- **Async Index API**: All Index methods are async; use `async with Index() as index:`.

## Key Models
- `FindingModelBase` – basic finding model structure
- `FindingModelFull` – with OIFM IDs and index codes
- `FindingInfo` – finding metadata (name, synonyms, description)

## MCP Server
- `findingmodel-mcp` CLI entry point
- Tools: search_finding_models, get_finding_model, list_finding_model_tags, count_finding_models

## Serena References
- `project_overview` – canonical model descriptions
- `index_duckdb_migration_decisions_2025` – Index implementation details
