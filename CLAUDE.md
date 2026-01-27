# CLAUDE.md

Claude Code must follow these instructions when working in this repository.

## 0. Operate via Serena MCP at all times

- Keep the Serena MCP server connected; treat it as the source of truth.
- **Look up context** with `read_memory` before inspecting files (start with `project_overview`, `code_style_conventions`, `suggested_commands`, `ai_assistant_usage_2025`).
- **Explore code** using Serena tools (`find_symbol`, `get_symbols_overview`, `find_referencing_symbols`) for symbol-aware navigation; use `search_for_pattern` for plain-text matching.
- **Use LSP tools** when available for precise navigation: `goToDefinition`, `findReferences`, `hover` for type info, `documentSymbol` for file structure.
- **Maintain reference system**: Update existing memories rather than creating new ones; consolidate related information into cohesive references. Keep memories compact and organized.
- Do not create private scratch notes—Serena memories are mandatory so Copilot and Claude stay in sync.
- **CRITICAL: Do not access `.serena/memories/**` via file operations.**
  - Only access/update memories through Serena MCP memory commands: `read_memory`, `write_memory`, `edit_memory`, `delete_memory`.

## 1. Project snapshot (see Serena `project_overview`)

- Purpose: Python 3.11+ monorepo for Open Imaging Finding Models.
- Core stack: uv, Taskfile, Pydantic v2, DuckDB for index/search.
- Layout: `packages/` with 5 packages, `docs/`, `evals/`, `Taskfile.yml`.
- Packages:
  - `findingmodel` – core models, Index API, MCP server, `findingmodel` CLI
  - `findingmodel-ai` – AI-powered tools, `findingmodel-ai` CLI
  - `anatomic-locations` – anatomic location queries, `anatomic-locations` CLI
  - `oidm-common` – shared infrastructure (DuckDB, embeddings)
  - `oidm-maintenance` – database build/publish (maintainers only)

## 2. Coding standards (Serena `code_style_conventions`)

- Formatting/linting via Ruff (120 char lines, preview mode). Run `task check` before committing.
- Strict typing: annotate everything, keep validators side-effect free, prefer `Annotated`/`Field` for constraints.
- Naming: snake_case functions/vars, PascalCase classes, UPPER_SNAKE constants, OIFM IDs `OIFM_{SOURCE}_{6_DIGITS}`.
- Asynchronous operations: use async for IO-bound code; remove async if no awaits (RUF029).
- Error handling: raise project exceptions (e.g., `ConfigurationError`), ensure cleanup with `try/finally`.
- **YAGNI principle**: Implement only what is required now, avoid speculative features or complex abstractions.
- **Package data pattern**: use `importlib.resources.files('<package>') / 'data'` for package-internal data files.

## 3. Testing

**Three-tier structure**: Unit tests (fast, mocked) → Integration tests (`@pytest.mark.callout`) → Evals (quality scores 0.0-1.0).

```bash
task test          # unit tests (no API calls)
task test-full     # integration tests
task evals         # quality evaluations (findingmodel-ai)
task check         # format + lint + mypy
```

- Tests live under `packages/<pkg>/tests/`
- uv fallbacks: `uv run pytest`, `uv run ruff check --fix`, `uv run mypy packages/`

## 4. Workflow expectations

- Follow `suggested_commands` memory for canonical dev commands.
- **Taskfile is canonical**: Prefer `task …` targets because they carry required markers/flags. Use raw `uv run …` only when no task exists.
- **uv-first + lockfile**: Use uv for install/test/build/publish; commit and honor `uv.lock`.
- Use `task_completion_checklist` when wrapping a feature or PR.
- When adding features:
  1. Review relevant Serena memories for conventions.
  2. Implement -> test -> verify.
  3. Update Serena memories summarizing architecture, commands, or conventions.

## 5. Documentation alignment

- Treat `.github/copilot-instructions.md` as the quick card; defer to Serena memories for depth.
- Mirror updates between this file and Copilot instructions to prevent drift (see Serena `instruction_files_plan_2025`).
- Path-scoped AI rules live in `.claude/rules/*.md`; update these when package-specific constraints change.

## 6. Security & secrets

- Keep API keys in `.env`; never print or commit them.
- When testing external integrations, guard credentials and clean up connections.

## 7. Quick Serena reference

**Start here:**
- `project_overview` – architecture and package structure
- `code_style_conventions` – formatting, typing, naming
- `suggested_commands` – development workflow

**By domain:**
- `pydantic_ai_testing_best_practices` – AI agent testing patterns
- `duckdb_architecture` – DuckDB patterns (index, search, rebuild)
- `protocol_based_architecture_2025` – backend interface patterns

**Meta:**
- `ai_assistant_usage_2025` – how assistants should collaborate
- `instruction_files_plan_2025` – documentation structure
- `memory_index` – navigation aid for all memories

---

IMPORTANT: Don't include time estimates in planning.

For dates, use the `date` command—don't assume you know it.

CRITICAL: Always web search to verify LLM model names—training data is outdated.

CRITICAL: Do NOT commit without checking in first, especially with a proposed commit message.
