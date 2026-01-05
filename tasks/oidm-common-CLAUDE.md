# CLAUDE.md

## Project Snapshot
**oidm-common** provides shared infrastructure for OIDM packages:
- DuckDB utilities (connection, bulk load, search, indexes)
- Embeddings (cache, provider protocol, OpenAI provider)
- Database distribution (manifest fetch, download, hash verification, platformdirs paths)
- Shared models (IndexCode, WebReference)

**Stack:** Python 3.11+, uv, Taskfile, Ruff, mypy, pytest

**Layout:**
- src/oidm_common/{duckdb,embeddings,distribution,models}
- tests/ (unit/integration; no external calls by default)
- Taskfile.yml, pyproject.toml, uv.lock

## Commands
- task test          # pytest -m "not callout"
- task test-full     # pytest (allows callouts)
- task check         # ruff format + ruff check --fix + mypy
- task build         # uv build
- task publish       # uv publish
Taskfile is canonical; raw `uv run …` only when no task fits. uv-first; commit uv.lock (use `uv sync --frozen` in CI).

## Coding Standards
- Ruff (120 char), mypy strict; async for IO (drop async if no awaits).
- Naming: snake_case funcs/vars, PascalCase classes, UPPER_SNAKE constants.
- Validators side-effect free; concise docstrings.
- Package data via importlib.resources if needed.

## DuckDB Patterns
- Bulk load via read_json; quote complex types.
- Convert embeddings to float32.
- Drop search indexes before writes; recreate FTS/HNSW afterward.
- Load fts/vss extensions by default.

## Serena References
- project_overview
- code_style_conventions
- suggested_commands
- ai_assistant_usage_2025
- duckdb_development_patterns
