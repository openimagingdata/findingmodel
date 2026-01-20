# Phase 6: Documentation and AI Setup

**Status:** ✅ COMPLETE

**Goal:** Optimal Claude Code experience

## Tasks

1. ✅ Update root CLAUDE.md (workspace overview)
2. ✅ Create `.claude/rules/` with path-scoped rules:
   - `index-duckdb.md` (unconditional—shared DuckDB patterns)
   - `oidm-common.md` (paths: `packages/oidm-common/**`)
   - `anatomic-locations.md` (paths: `packages/anatomic-locations/**`)
   - `findingmodel.md` (paths: `packages/findingmodel/**`)
   - `findingmodel-ai.md` (paths: `packages/findingmodel-ai/**`)
   - `oidm-maintenance.md` (paths: `packages/oidm-maintenance/**`)
3. ✅ Update Serena memories for workspace structure
4. ✅ Update Taskfile.yml for per-package tasks

## Claude Code Configuration

Claude Code uses a hierarchical system: root `CLAUDE.md` for project-wide instructions, and `.claude/rules/*.md` for path-scoped package rules.

### Structure

```
findingmodel/
├── CLAUDE.md                           # Root: workspace overview, shared standards
├── .claude/
│   └── rules/
│       ├── index-duckdb.md             # Shared DuckDB patterns (unconditional)
│       ├── oidm-common.md              # paths: packages/oidm-common/**
│       ├── anatomic-locations.md       # paths: packages/anatomic-locations/**
│       ├── findingmodel.md             # paths: packages/findingmodel/**
│       ├── findingmodel-ai.md          # paths: packages/findingmodel-ai/**
│       └── oidm-maintenance.md         # paths: packages/oidm-maintenance/**
```

### Approach

| Layer | Use Case |
|-------|----------|
| **Root CLAUDE.md** | Workspace overview, shared coding standards, always loaded |
| **`.claude/rules/*.md`** | Package-specific patterns loaded when working in matching paths |

### Path-Specific Rules Example

```markdown
---
paths: packages/findingmodel-ai/**
---

# FindingModel AI Package Rules

## AI Tool Patterns
- Use pydantic-ai agents with structured outputs
- Follow two-agent pattern for complex workflows
- Test with TestModel/FunctionModel, not real APIs

## Configuration
- API keys via FindingModelAIConfig
- Model selection via DEFAULT_MODEL env var
```

### Loading Priority

When working in `packages/findingmodel-ai/`, Claude loads:
- Root `CLAUDE.md`
- `.claude/rules/findingmodel-ai.md` (path matches)
- `.claude/rules/index-duckdb.md` (unconditional)

### Root CLAUDE.md Template

```markdown
# CLAUDE.md

## Workspace Overview

This is a uv workspace containing four packages:

| Package | Purpose | Key Dependencies |
|---------|---------|------------------|
| **oidm-common** | Shared infrastructure | duckdb, pydantic |
| **anatomic-locations** | Anatomic ontology | oidm-common |
| **findingmodel** | Core models, index, MCP server | oidm-common |
| **findingmodel-ai** | AI authoring tools | findingmodel, openai, pydantic-ai |

## User Installation

\`\`\`bash
pip install findingmodel      # Core: models, index, MCP server
pip install findingmodel-ai   # Full: adds AI authoring tools
\`\`\`

## Quick Reference

\`\`\`bash
task test              # All tests (no callouts)
task test-full         # All tests (with callouts)
task check             # Format + lint + mypy

# Per-package
task test:oidm-common
task test:anatomic
task test:findingmodel
task test:findingmodel-ai
\`\`\`

## Coding Standards

- Ruff formatting (120 char lines, preview mode)
- Strict mypy typing
- pydantic-settings for configuration
- Async for I/O-bound operations

## Package Dependencies

\`\`\`
findingmodel-ai → findingmodel → oidm-common
                ↘ anatomic-locations ↗
\`\`\`

Changes to oidm-common may affect all other packages.
```

## Serena Memory Organization

Shared memories at workspace root:

| Memory | Content |
|--------|---------|
| `project_overview` | Workspace structure, all packages |
| `code_style_conventions` | Shared coding standards |
| `suggested_commands` | Workspace and per-package commands |
| `duckdb_architecture` | DuckDB patterns |
| `pydantic_ai_testing_best_practices` | Testing patterns |

Package-specific patterns go in `.claude/rules/*.md` files with path scoping.
