# CLAUDE.md

## Project Snapshot
**anatomic-locations** provides anatomic location ontology navigation:
- Models/enums for locations, regions, laterality
- Index with hierarchy traversal and hybrid search (FTS + HNSW via oidm-common)
- Migration/build pipeline using oidm-common bulk load
- Config + CLI (click) for queries and DB management

**Stack:** Python 3.11+, uv, Taskfile, DuckDB, Pydantic v2, Click; depends on oidm-common

**Layout:**
- src/anatomic_locations/{models,index,migration,config,cli}
- tests/ with data fixtures and precomputed embeddings
- Taskfile.yml, pyproject.toml, uv.lock

## Commands
- task test          # pytest -m "not callout"
- task test-full     # pytest (allows callouts)
- task check         # ruff format + ruff check --fix + mypy
- task build         # uv build
- task publish       # uv publish
Taskfile is canonical; raw `uv run …` only when no task fits. uv-first; commit uv.lock (use `uv sync --frozen` in CI).

## Coding Standards
- Ruff (120 char), mypy strict; async for IO.
- Naming: snake_case funcs/vars, PascalCase classes, UPPER_SNAKE constants.
- Validators side-effect free; concise docstrings.

## Domain Patterns
- Materialized-path hierarchy for containment.
- Laterality variants (left/right/bilateral/nonlateral/generic).
- Hybrid search: FTS + HNSW (via oidm-common.duckdb), float32 embeddings, index rebuild after writes.
- Database distribution via oidm-common.distribution (manifest + hash verification).

## Serena References
- project_overview
- code_style_conventions
- suggested_commands
- ai_assistant_usage_2025
- anatomic_location_search_implementation
