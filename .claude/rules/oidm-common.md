---
paths: "packages/oidm-common/**"
---

# oidm-common Package Rules

## Purpose
Shared infrastructure (DuckDB auto-download, embedding client, database utilities) used by other OIDM packages.

## Constraints
- **Zero AI dependencies**: No pydantic-ai, no LLM providers. AI logic lives in findingmodel-ai.
- **Minimal surface**: Only expose utilities that multiple packages need.
- **Backward compatibility**: Other packages depend on this; avoid breaking changes.

## Key Patterns
- Database auto-download via `pooch` with checksum verification
- OpenAI embedding client (optional dependency via `[openai]` extra)
- Protocol-based backend pattern for extensibility

## Serena References
- `duckdb_development_patterns` – DuckDB conventions and auto-download logic
- `protocol_based_architecture_2025` – backend interface patterns
