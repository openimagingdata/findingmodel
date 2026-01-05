# Copilot Instructions · oidm-common

- Use Serena MCP for lookups (project_overview, code_style_conventions, suggested_commands, duckdb_development_patterns, ai_assistant_usage_2025).
- Taskfile is canonical; tasks include required flags (e.g., -m "not callout"). Use raw uv only if no task exists or when intentionally deviating.
- uv-first and commit uv.lock; use `uv sync --frozen` in CI.
- Commands: task test | task test-full | task check | task build | task publish.
- Scope: DuckDB utilities, embeddings (cache/provider), distribution (manifest/download/hash), shared models.
- Update Serena memories after noteworthy changes; keep CLAUDE.md and this file in sync.
