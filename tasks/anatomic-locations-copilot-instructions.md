# Copilot Instructions · anatomic-locations

- Use Serena MCP for lookups (project_overview, code_style_conventions, suggested_commands, anatomic_location_search_implementation, ai_assistant_usage_2025).
- Taskfile is canonical; tasks include required flags (e.g., -m "not callout"). Raw uv only if no task or intentional override.
- uv-first and commit uv.lock; use `uv sync --frozen` in CI.
- Commands: task test | task test-full | task check | task build | task publish.
- Scope: anatomic models/enums, index, migration, config, CLI; depends on oidm-common for DuckDB/distribution/embeddings.
- Update Serena memories after changes; keep CLAUDE.md and this file aligned.
