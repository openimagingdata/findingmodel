# Phase 6: Documentation and AI Setup

**Status:** ⏳ PENDING

**Goal:** Optimal Claude Code experience

## Tasks

1. Update root CLAUDE.md (workspace overview)
2. Create packages/*/CLAUDE.md (package-specific)
3. Create `.claude/rules/` with path-scoped rules:
   - `duckdb-patterns.md` (unconditional—shared DuckDB patterns)
   - `testing.md` (paths: `**/tests/**`)
   - `oidm-common.md` (paths: `packages/oidm-common/**`)
   - `anatomic-locations.md` (paths: `packages/anatomic-locations/**`)
   - `findingmodel-core.md` (paths: `packages/findingmodel/**`)
   - `findingmodel-ai.md` (paths: `packages/findingmodel-ai/**`)
4. Update Serena memories for workspace structure
5. Update Taskfile.yml for per-package tasks
6. Update GitHub workflows

## Claude Code Configuration

Claude Code uses a hierarchical memory system. For monorepos, the recommended approach is a **hybrid** of root CLAUDE.md, `.claude/rules/` with path scoping, and optional per-package CLAUDE.md files.

### Recommended Structure

```
findingmodel/
├── CLAUDE.md                           # Root: workspace overview, shared standards
├── .claude/
│   └── rules/
│       ├── duckdb-patterns.md          # Shared DuckDB patterns (unconditional)
│       ├── testing.md                  # paths: **/tests/**
│       ├── oidm-common.md              # paths: packages/oidm-common/**
│       ├── anatomic-locations.md       # paths: packages/anatomic-locations/**
│       ├── findingmodel-core.md        # paths: packages/findingmodel/**
│       └── findingmodel-ai.md          # paths: packages/findingmodel-ai/**
├── packages/
│   ├── oidm-common/
│   │   └── CLAUDE.md                   # Optional: package-specific context
│   ├── anatomic-locations/
│   │   └── CLAUDE.md
│   ├── findingmodel/
│   │   └── CLAUDE.md
│   └── findingmodel-ai/
│       └── CLAUDE.md
```

### Why Hybrid?

| Approach | Use Case |
|----------|----------|
| **Root CLAUDE.md** | Workspace overview, shared coding standards, always loaded |
| **`.claude/rules/` with paths** | Domain-specific patterns that only apply to certain packages |
| **Per-package CLAUDE.md** | Package context loaded when working in that directory |

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
- API keys via FindingModelAISettings (FINDINGMODEL_AI_ prefix)
- Model selection via model_provider and model_tier settings
```

### Loading Priority (Highest to Lowest)

1. `./CLAUDE.local.md` (personal, gitignored)
2. `./CLAUDE.md` or `./.claude/CLAUDE.md`
3. `./.claude/rules/*.md` (path-matched rules first)
4. Parent directory CLAUDE.md files (recursive up to root)
5. `~/.claude/CLAUDE.md` (user-level)

When working in `packages/findingmodel-ai/`, Claude loads:
- `packages/findingmodel-ai/CLAUDE.md`
- Root `CLAUDE.md`
- `.claude/rules/findingmodel-ai.md` (path matches)
- `.claude/rules/duckdb-patterns.md` (unconditional)

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

### Per-Package CLAUDE.md Template

Keep these short—they're additive to root:

```markdown
# findingmodel-ai

AI authoring tools for Open Imaging Finding Models.

## Key Modules
- `tools/model_editor.py` - Main editing agent
- `tools/ontology_search.py` - Code lookup workflows
- `config.py` - API keys via FINDINGMODEL_AI_ prefix

## Patterns
- Two-agent pattern for complex workflows
- TestModel for deterministic AI tests
- Evals in `evals/` directory

## Commands
\`\`\`bash
task test:findingmodel-ai
task evals
\`\`\`
```

## Serena Memory Organization

Shared memories at workspace root:

| Memory | Content |
|--------|---------|
| `project_overview` | Workspace structure, all three packages |
| `code_style_conventions` | Shared coding standards |
| `suggested_commands` | Workspace and per-package commands |
| `duckdb_development_patterns` | Shared DuckDB patterns |
| `pydantic_ai_testing_best_practices` | Testing patterns |

Package-specific patterns can go in the shared memories with clear sections, or in package CLAUDE.md files.
