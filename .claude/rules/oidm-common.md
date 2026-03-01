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
- **`_execute_one(conn, sql, params)`** — execute a query and return the single result row as a `dict[str, object] | None`. Uses `cursor.description` to map column names; eliminates positional indexing brittleness. Subclasses use this instead of `.fetchone()` + `row[N]`.
- **`_execute_all(conn, sql, params)`** — execute a query and return all result rows as `list[dict[str, object]]`. Same column-name mapping. Use for bulk hydration (no N+1 per-row re-fetch).

## Serena References
- `duckdb_development_patterns` – DuckDB conventions and auto-download logic
- `protocol_based_architecture_2025` – backend interface patterns
