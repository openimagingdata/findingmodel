# CLAUDE.md

Claude Code must follow these instructions when working in this repository.

## 0. Operate via Serena MCP at all times

- Keep the Serena MCP server connected; treat it as the source of truth.
- **Look up context** with `read_memory` before inspecting files (start with `project_overview`, `code_style_conventions`, `suggested_commands`, `ai_assistant_usage_2025`).
- **Explore code** using Serena tools (`find_symbol`, `search_for_pattern`, `get_symbols_overview`) rather than ad-hoc greps.
- **Maintain reference system**: Update existing memories rather than creating new ones; consolidate related information into cohesive references (e.g., DuckDB download info goes in Index/anatomic location memories, not standalone). Keep memories compact and organized.
- Do not create private scratch notes—Serena memories are mandatory so Copilot and Claude stay in sync.

## 1. Project snapshot (see Serena `project_overview`)

- Purpose: Python 3.11+ library for Open Imaging Finding Models with AI-assisted authoring.
- Core stack: uv, Taskfile, Pydantic v2, OpenAI/Anthropic AI tooling, Tavily search, DuckDB for index/search.
- Layout: `src/findingmodel/` (models, tools, config, CLI), `test/` (pytest + fixtures), `notebooks/` (demos), `Taskfile.yml`, `.env.sample`.
- Key modules to know: `finding_model.py`, `finding_info.py`, `tools/` (LLM workflows), `index.py`, `config.py`.

## 2. Architecture touchpoints

- Protocol-based backend pattern documented in Serena `protocol_based_architecture_2025`; follow that interface when adding search providers.
- Multi-provider AI support: OpenAI, Anthropic, Google Gemini, Ollama, and Gateway backends with tier-based model selection (base/full/small); configure via `DEFAULT_MODEL` env var using Pydantic AI format (e.g., `openai:gpt-5-mini`, `anthropic:claude-sonnet-4-5`, `google:gemini-3-flash`). Per-agent overrides available via `AGENT_MODEL_OVERRIDES__<tag>=provider:model`; see `docs/configuration.md` for tag reference.
- AI workflow conventions rely on two-agent patterns and structured outputs—review Serena `ontology_concept_search_refactoring` and `anatomic_location_search_implementation` before refactoring those areas.
- DuckDB index: drops search indexes before writes, clears denormalized tables manually, rebuilds HNSW/FTS afterward; no foreign keys (see Serena `index_duckdb_migration_decisions_2025`).

## 3. Coding standards (Serena `code_style_conventions`)

- Formatting/linting via Ruff (120 char lines, preview mode). Run `task check` before committing.
- Strict typing: annotate everything, keep validators side-effect free, prefer `Annotated`/`Field` for constraints.
- Naming: snake*case functions/vars, PascalCase classes, UPPER_SNAKE constants, OIFM IDs `OIFM*{SOURCE}\_{6_DIGITS}`.
- Asynchronous operations: use async for IO-bound code; remove async if no awaits (RUF029).
- Error handling: raise project exceptions (e.g., `ConfigurationError`), ensure cleanup with `try/finally`.
- **YAGNI principle**: "You Aren't Going To Need It" - implement only what is required now, avoid speculative features, versioning, or complex abstractions until they're actually needed.
- **Package data pattern**: use `importlib.resources.files('findingmodel') / 'data'` for package-internal data files; store under `src/findingmodel/data/` with `.gitignore` for large files.

## 4. Testing + QA (Serena `pydantic_ai_testing_best_practices`)

### Three-Tier Testing Structure

FindingModel uses a clear separation between tests and evaluations:

1. **Unit Tests** (`test/test_*.py`) - Verify logic correctness with mocked dependencies
   - Fast, no API calls
   - Run with `task test`
   - Default: block external calls by setting `models.ALLOW_MODEL_REQUESTS = False`

2. **Integration Tests** (`test/test_*.py` with `@pytest.mark.callout`) - Verify wiring with real APIs
   - Real API calls, specific scenarios
   - Run with `task test-full`
   - Use `TestModel` / `FunctionModel` for deterministic AI agent tests

3. **Evals** (`evals/*.py`) - Assess behavioral quality comprehensively
   - Dataset.evaluate() with focused evaluators
   - Run with `task evals` or `task evals:model_editor`
   - See `evals/CLAUDE.md` for eval development guidance

**Key distinction**: Tests verify correctness (pass/fail), evals assess quality (0.0-1.0 scores with partial credit).

### Test Data and Fixtures

- Test data lives under `test/data/`; keep fixtures reusable
- Eval suites access test data via `evals/utils.py` helpers

### Tasks

```bash
task test          # unit tests (fast, no API)
task test-full     # integration tests (includes callouts)
task evals         # run all eval suites
task check         # format + lint + mypy
task build         # package build
```

- uv fallbacks: `uv run ruff format`, `uv run ruff check --fix`, `uv run mypy src`, `uv run pytest -rs -m "not callout"`.

## 5. Workflow expectations

- Follow `suggested_commands` memory for canonical dev commands and CLI usage (`uv run python -m findingmodel ...`).
- Use `task_completion_checklist` when wrapping a feature or PR.
- Prefer programmatic solutions before invoking LLMs; agents should perform judgment, not heavy data munging.
- When adding features:
  1. Review relevant Serena memories for conventions.
  2. Implement -> assess/evaluate -> test.
  3. Update or create Serena memories summarizing architecture, commands, or conventions.

## 6. Documentation alignment

- Treat `.github/copilot-instructions.md` as the quick card. Any detail removed from there must exist in Serena memories and, if broader team context is needed, here.
- Mirror updates between this file and Copilot instructions to prevent drift (see Serena `instruction_files_plan_2025`).
- If guidance grows beyond comfortable scan length, create shared references under `docs/` and link via Serena memories.

## 7. Security & secrets

- Keep API keys in `.env`; never print or commit them. Classes should read `SecretStr` directly (Serena `documentation_corrections_2025`).
- Required: At least one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`; optional: `TAVILY_API_KEY` for enhanced search, `OLLAMA_BASE_URL` for local models.
- When testing external integrations, guard credentials and clean up connections.

## 8. Quick Serena reference

- `project_overview` – canonical snapshot.
- `code_style_conventions` – formatting, typing, naming.
- `suggested_commands` – dev workflow.
- `pydantic_ai_testing_best_practices` – AI testing safeguards.
- `ai_assistant_usage_2025` – how Copilot/Claude should collaborate.
- `instruction_files_plan_2025` – update checklist for these instruction files.

IMPORTANT! DON'T include time estimates in your planning—they're not useful.

For dates, you need to use the `date` command--DON'T assume you know it.

- CRITICAL--Do NOT commit without checking in first, especially with a proposed commit message.