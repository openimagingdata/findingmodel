# Copilot Instructions · FindingModel

1. **Use Serena MCP for everything.** Before answering questions or editing code, run Serena lookups:

   - `read_memory` → load canonical context (`project_overview`, `code_style_conventions`, `suggested_commands`).
   - `find_symbol` / `get_symbols_overview` / `find_referencing_symbols` → prefer Serena for symbol-aware navigation (definitions + references).
   - `search_for_pattern` → use when you need plain-text pattern matching (strings, config keys, etc.).
   - `write_memory` → record new learnings or workflow changes (see `ai_assistant_usage_2025`).
     Keep Serena open; every substantial change must update or add a memory.

   - **CRITICAL: `.serena/memories/**` is off-limits as a folder.\*\*
     - Do not open, read, edit, or delete memory files via filesystem tools.
     - Only access memories through Serena MCP memory commands: `read_memory`, `write_memory`, `edit_memory`, `delete_memory`.

2. **Project snapshot** (full detail lives in Serena `project_overview`).

   - Purpose: Python library for Open Imaging Finding Models with AI tooling.
   - Stack: Python 3.11+, uv, Taskfile, Pydantic v2, DuckDB for index/search. AI tooling in findingmodel-ai package.
   - Layout: `packages/` (5 packages: findingmodel, findingmodel-ai, anatomic-locations, oidm-common, oidm-maintenance), `docs/`, `evals/`.

3. **Coding standards** (reference Serena `code_style_conventions`).

   - Ruff handles formatting (120 char max) and linting; run `task check` before commits.
   - Strict typing: annotate everything, prefer async for IO, keep validators pure.
   - Naming: snake*case functions/vars, PascalCase classes, OIFM IDs `OIFM*{SOURCE}\_{6_DIGITS}`.

4. **Development workflow** (see Serena `suggested_commands`).

   - Tests: `task test` (local), `task test-full` (includes callouts). Unit tests use `TestModel`/`FunctionModel` with mocks; integration tests marked `@pytest.mark.callout` call real APIs.
   - Quality: `task check` (format/lint/mypy), `uv run ruff format`, `uv run ruff check --fix`, `uv run mypy packages/` as fallbacks.
   - CLI: `python -m findingmodel` lists subcommands for config, info generation, markdown conversions.
   - **Taskfile is canonical**: Prefer Task targets because they bake in required markers/flags (e.g., `-m "not callout"`). Use raw `uv run …` only when no task exists or when deliberately overriding defaults.
   - **uv-first + lockfile**: Use uv for install/test/build/publish; commit and honor `uv.lock` (`uv sync --frozen` in CI).

5. **Documentation + knowledge flow.**
   - Treat this file as the quick-start card; defer to Serena memories for depth.
   - When new architecture, conventions, or commands appear, capture them via `write_memory` (update `instruction_files_plan_2025`).
   - Mirror updates in `CLAUDE.md` so all assistants stay aligned.
   - Path-scoped AI rules live in `.claude/rules/*.md`; update these when package-specific constraints change.
