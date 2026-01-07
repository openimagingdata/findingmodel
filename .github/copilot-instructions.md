# Copilot Instructions · FindingModel

1. **Use Serena MCP for everything.** Before answering questions or editing code, run Serena lookups:

   - `read_memory` → load canonical context (`project_overview`, `code_style_conventions`, `suggested_commands`).
   - `find_symbol` / `search_for_pattern` → inspect code instead of manual greps.
   - `write_memory` → record new learnings or workflow changes (see `ai_assistant_usage_2025`).
     Keep Serena open; every substantial change must update or add a memory.

2. **Project snapshot** (full detail lives in Serena `project_overview`).

   - Purpose: Python library for Open Imaging Finding Models with AI tooling.
   - Stack: Python 3.11+, uv, Taskfile, Pydantic v2, multi-provider AI (OpenAI/Anthropic/Gemini/Ollama/Gateway via `DEFAULT_MODEL`; per-agent overrides via `AGENT_MODEL_OVERRIDES__<tag>`), DuckDB for index/search.
   - Layout: `src/findingmodel/` (core models + tools), `test/` (pytest, fixtures), `notebooks/` (demos).

3. **Coding standards** (reference Serena `code_style_conventions`).

   - Ruff handles formatting (120 char max) and linting; run `task check` before commits.
   - Strict typing: annotate everything, prefer async for IO, keep validators pure.
   - Naming: snake_case functions/vars, PascalCase classes, OIFM IDs `OIFM_{SOURCE}_{6_DIGITS}`.

4. **Development workflow** (see Serena `suggested_commands`).

   - Tests: `task test` (local), `task test-full` (includes callouts). Unit tests use `TestModel`/`FunctionModel` with mocks; integration tests marked `@pytest.mark.callout` call real APIs.
   - Quality: `task check` (format/lint/mypy), `uv run ruff format`, `uv run ruff check --fix`, `uv run mypy src` as fallbacks.
   - CLI: `python -m findingmodel` lists subcommands for config, info generation, markdown conversions.

5. **Documentation + knowledge flow.**
   - Treat this file as the quick-start card; defer to Serena memories for depth.
   - When new architecture, conventions, or commands appear, capture them via `write_memory` (update `instruction_files_plan_2025`).
   - Mirror updates in `CLAUDE.md` so all assistants stay aligned.
