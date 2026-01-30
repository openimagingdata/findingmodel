# Instruction File Alignment Plan (2025-01)

## Goal
Align Claude and Copilot instruction files with Serena memories so every assistant shares a single source of truth.

## Current Structure (Phase 7 Complete)

### Root-Level Files
- **CLAUDE.md** - Detailed Claude Code instructions (project snapshot, coding standards, testing, workflow)
- **.github/copilot-instructions.md** - Quick-start card for Copilot (mirrors CLAUDE.md essentials)

### Path-Scoped Rules (.claude/rules/)
- `findingmodel.md` - Core package constraints (no AI deps, Pydantic v2, async Index)
- `findingmodel-ai.md` - AI package constraints (Pydantic AI, two-agent patterns, testing)
- `anatomic-locations.md` - Query package constraints (no AI deps, async-first)
- `oidm-common.md` - Infrastructure constraints (zero AI deps, minimal surface)
- `oidm-maintenance.md` - Maintainer tools (AWS credentials, large data ops)
- `index-duckdb.md` - DuckDB index patterns (shared across findingmodel, anatomic-locations, oidm-common)

### Per-Package READMEs
Each package under `packages/` has its own README.md with installation, usage, and API docs.

## Structure Blueprint (Sections in CLAUDE.md)
- **Section 0**: Serena MCP usage (mandatory lookups before inspecting files)
- **Section 1**: Project snapshot (layout, key modules, stack)
- **Section 2**: Coding standards (Ruff, typing, naming, YAGNI, package data)
- **Section 3**: Testing (three-tier: unit → integration → evals)
- **Section 4**: Workflow expectations (Taskfile, uv-first)
- **Section 5**: Documentation alignment (mirrors, path-scoped rules)
- **Section 6**: Security & secrets
- **Section 7**: Quick Serena reference (grouped by domain)

## Memory Consolidation (Phase 7)

Reduced from 27 to 19 memories:
- **DuckDB memories** (5→1): Consolidated into `duckdb_architecture`
- **Ontology memories** (2→1): Consolidated into `ontology_search_architecture`
- **Renamed**: `model_editing_guardrails_2025-09-26` → `model_editing_guardrails_2025`
- **Deleted stale**: `api_integration`, `documentation_corrections_2025`, `test_suite_improvements_2025`
- **Navigation**: Created `memory_index` for discovery

## Duplication Prevention

| Content | Canonical Location | Reference From |
|---------|-------------------|----------------|
| Provider/model config | `code_style_conventions` | findingmodel-ai.md |
| Testing patterns | `pydantic_ai_testing_best_practices` | CLAUDE.md, findingmodel-ai.md |
| DuckDB patterns | `duckdb_architecture` | index-duckdb.md |
| Ontology search | `ontology_search_architecture` | findingmodel-ai.md |

## Update Process
1. Before editing instruction files, review `memory_index` for relevant memories.
2. After updating instructions, amend Serena memories to reflect canonical guidance.
3. When package-specific rules change, update `.claude/rules/<package>.md`.
4. Avoid duplicating content—reference memories instead.
5. Verify reviewers check that Serena memories are referenced in documentation PRs.

## Key References
- `memory_index` - navigation aid for all memories
- `project_overview` - canonical project structure
- `code_style_conventions` - formatting, typing, naming
- `pydantic_ai_testing_best_practices` - AI testing patterns
- `ai_assistant_usage_2025` - how assistants should collaborate
