# Plan: Monorepo Restructure with uv Workspaces

**Status:** In Progress (Phase 3.6 Complete, Phase 3.7 Planned)
**Supersedes:** [oidm-common-package-plan.md](oidm-common-package-plan.md), [anatomic-locations-package-plan.md](anatomic-locations-package-plan.md)

## Rationale

After analysis, a monorepo with uv workspaces is better than separate repositories because:

1. **AI Context** - Claude Code and Copilot see all packages, relationships, and usage patterns
2. **Development Velocity** - Edit oidm-common → test in anatomic-locations immediately (no publish cycle)
3. **Single Lockfile** - No dependency drift between packages
4. **Atomic Changes** - Cross-package updates in one PR
5. **Shared Configuration** - One ruff/mypy/pytest config
6. **Simpler Setup** - One CLAUDE.md hierarchy, one Serena memory set

The separate repos approach solved organizational problems we don't have while creating AI context problems that hurt us.

## Target Structure

```
findingmodel/                           # Keep existing repo name
├── pyproject.toml                      # Workspace root (no package here)
├── uv.lock                             # Single lockfile for all packages
├── CLAUDE.md                           # Root AI instructions
├── Taskfile.yml                        # Unified task runner
├── .serena/memories/                   # Shared AI context
├── .mcp.json                           # MCP server config
│
├── packages/
│   ├── oidm-common/                    # Shared infrastructure
│   │   ├── pyproject.toml
│   │   ├── CLAUDE.md                   # Package-specific additions
│   │   ├── src/oidm_common/
│   │   │   ├── __init__.py
│   │   │   ├── duckdb/                 # Connection, bulk load, search, indexes
│   │   │   ├── embeddings/             # Cache, provider protocol, OpenAI
│   │   │   ├── distribution/           # Manifest, download, paths
│   │   │   └── models/                 # IndexCode, WebReference
│   │   └── tests/
│   │
│   ├── anatomic-locations/             # Anatomic ontology package (READ-ONLY)
│   │   ├── pyproject.toml
│   │   ├── CLAUDE.md
│   │   ├── src/anatomic_locations/
│   │   │   ├── __init__.py
│   │   │   ├── models/                 # AnatomicLocation, enums
│   │   │   ├── index.py                # AnatomicLocationIndex (search only)
│   │   │   ├── config.py               # Settings, ensure_anatomic_db()
│   │   │   └── cli.py                  # CLI: search, show (no build)
│   │   └── tests/
│   │
│   ├── findingmodel/                   # Core package (models, index, MCP) - READ-ONLY
│   │   ├── pyproject.toml
│   │   ├── CLAUDE.md
│   │   ├── src/findingmodel/
│   │   │   ├── __init__.py
│   │   │   ├── finding_model.py        # FindingModel
│   │   │   ├── finding_info.py         # FindingInfo
│   │   │   ├── abstract_finding_model.py
│   │   │   ├── contributor.py
│   │   │   ├── index.py                # DuckDBIndex (search only, no build)
│   │   │   ├── index_validation.py
│   │   │   ├── config.py               # Uses oidm-common distribution
│   │   │   ├── cli.py                  # CLI: search, show, validate (no build/publish)
│   │   │   └── mcp_server.py           # MCP server for IDE access
│   │   └── tests/
│   │
│   ├── findingmodel-ai/                # AI authoring tools
│   │   ├── pyproject.toml
│   │   ├── CLAUDE.md
│   │   ├── src/findingmodel_ai/
│   │   │   ├── __init__.py
│   │   │   ├── tools/                  # AI agents and workflows
│   │   │   │   ├── model_editor.py
│   │   │   │   ├── similar_finding_models.py
│   │   │   │   ├── ontology_concept_match.py
│   │   │   │   ├── ontology_search.py
│   │   │   │   ├── anatomic_location_search.py
│   │   │   │   ├── create_stub.py
│   │   │   │   ├── finding_description.py
│   │   │   │   ├── markdown_in.py
│   │   │   │   └── common.py           # get_model, tavily client
│   │   │   └── cli.py                  # AI-specific CLI commands
│   │   ├── tests/
│   │   └── evals/                      # Agent evaluation suites
│   │
│   └── oidm-maintenance/               # Database build & publish (maintainers only)
│       ├── pyproject.toml
│       ├── src/oidm_maintenance/
│       │   ├── __init__.py
│       │   ├── config.py               # S3 credentials, endpoints
│       │   ├── s3.py                   # S3 client, upload, manifest ops
│       │   ├── hashing.py              # File hash computation
│       │   ├── anatomic/               # Anatomic build + publish
│       │   ├── findingmodel/           # FindingModel build + publish
│       │   └── cli.py                  # oidm-maintain command
│       └── tests/
│
├── docs/                               # Shared documentation
├── tasks/                              # Planning documents
└── notebooks/                          # Demos and experiments
```

## Package Dependencies

```
                    ┌─────────────────────┐
                    │  oidm-maintenance   │  ← Database build & publish
                    │  (maintainers only) │     (boto3, openai)
                    └─────────┬───────────┘
                              │ depends on all user packages
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  findingmodel-ai │  │  findingmodel   │  │ anatomic-locations  │
│   (AI tools)     │  │ (core package)  │  │  (anatomic ontology)│
└───────┬──────────┘  └────────┬────────┘  └─────────┬───────────┘
        │ depends on           │ depends on          │ depends on
        └───────────┬──────────┴─────────────────────┘
                    ▼
          ┌─────────────────┐
          │   oidm-common   │
          │ (infrastructure)│
          └─────────────────┘
```

**Dependency rules:**
- oidm-common has no internal dependencies
- anatomic-locations depends on oidm-common (no findingmodel dependency)
- findingmodel depends on oidm-common (no anatomic-locations dependency)
- findingmodel-ai depends on both findingmodel and anatomic-locations
- oidm-maintenance depends on all user packages + heavy deps (boto3, openai for builds)

**User installation:**
```bash
pip install findingmodel      # Core: models, index, search, MCP server
pip install findingmodel-ai   # Full: adds AI authoring tools
```

## Phase Status Summary

| Phase | Name | Status | Details |
|-------|------|--------|---------|
| [0](monorepo_plan_phases/phase-0-preparation.md) | Preparation | ✅ COMPLETE | Clean starting point |
| [1](monorepo_plan_phases/phase-1-restructure.md) | Restructure findingmodel | ✅ COMPLETE | Moved code into packages/ |
| [2](monorepo_plan_phases/phase-2-oidm-common.md) | Extract oidm-common | ✅ COMPLETE | Shared infrastructure package |
| [3](monorepo_plan_phases/phase-3-anatomic.md) | Create anatomic-locations | ✅ COMPLETE | Anatomic ontology package |
| [3.5](monorepo_plan_phases/phase-3.5-embeddings.md) | Embedding Utilities Cleanup | ✅ COMPLETE | Consolidated in oidm-common |
| [3.6](monorepo_plan_phases/phase-3.6-distribution.md) | Distribution Code Cleanup | ✅ COMPLETE | anatomic-locations has ZERO findingmodel imports |
| [3.7](monorepo_plan_phases/phase-3.7-maintenance.md) | Create oidm-maintenance | ✅ COMPLETE | Package created; read-only migration moved to 3.8 |
| [3.8](monorepo_plan_phases/phase-3.8-critical-fixes.md) | Complete read-only migration | 📋 PLANNED | Strip DuckDBIndex to read-only, fix tests |
| [4](monorepo_plan_phases/phase-4-findingmodel-ai.md) | Create findingmodel-ai | ⏳ PENDING | Extract AI tools |
| [5](monorepo_plan_phases/phase-5-cleanup.md) | Clean up findingmodel | ⏳ PENDING | Remove extracted code |
| [6](monorepo_plan_phases/phase-6-documentation.md) | Documentation and AI Setup | ⏳ PENDING | CLAUDE.md, rules, Serena |

**Supporting documents:**
- [Configuration Details](monorepo_plan_phases/configuration-details.md) - pyproject.toml configs, env vars, migration notes

## Package Responsibilities Detail

### oidm-common: Shared Infrastructure

The `oidm-common` package provides infrastructure shared by all OIDM packages.

#### distribution/ - Database Distribution

Both findingmodel and anatomic-locations need pre-built DuckDB databases. The distribution module handles:

- **Manifest reading**: Parse `manifest.json` to find database URLs, versions, and checksums
- **Download**: Fetch pre-built DuckDB files from remote storage (S3, HTTP)
- **Version checking**: Compare local version against manifest, re-download when updated
- **Hash verification**: Validate SHA256 checksums after download
- **Caching**: Store databases in platform-specific user data directories
- **Path resolution**: Provide consistent paths across platforms

```python
# Example usage from findingmodel or anatomic-locations
from oidm_common.distribution import ensure_database, get_manifest

manifest = get_manifest()  # Fetches/caches manifest.json
db_path = ensure_database("finding_models.duckdb", manifest)  # Downloads if needed
```

#### duckdb/ - Database Utilities

- **Connection management**: Open/close, read-only mode, extension loading
- **Bulk loading**: Efficient JSON loading with `read_json()` for FLOAT[]/STRUCT[]
- **Search utilities**: Hybrid FTS + vector search, weighted fusion
- **Index management**: HNSW and FTS index creation/rebuilding

#### embeddings/ - Embedding Infrastructure

- **Cache**: DuckDB-based cache for OpenAI embeddings (avoids redundant API calls)
- **Provider protocol**: Abstract interface for embedding providers
- **OpenAI provider**: Default implementation using text-embedding-3-small

**OPENAI_API_KEY requirement:**
- **Not needed** for search queries - pre-built databases include pre-computed embeddings
- **Required** when generating new embeddings:
  - Building/rebuilding anatomic-locations database (`anatomic build`)
  - Adding new entries that need vector search support
  - Any operation that calls the embedding provider

This is why oidm-common has `openai` as an optional dependency:
```toml
[project.optional-dependencies]
openai = ["openai>=1.0"]
```

Users who only search don't need the OpenAI package or API key. Users who build databases install with `pip install oidm-common[openai]` and set `OPENAI_API_KEY`.

#### models/ - Shared Data Models

- **IndexCode**: Standardized code representation (system, code, display)
- **WebReference**: URL + title for citations

### anatomic-locations: Anatomic Ontology

- **Models**: AnatomicLocation with laterality, hierarchy (materialized path)
- **Index**: AnatomicLocationIndex with hybrid search (FTS + vector)
- **Migration**: Database builder from source ontologies
- **CLI**: `anatomic search`, `anatomic show`, `anatomic build`

### findingmodel: Core Library

- **Models**: FindingModel, FindingInfo, AbstractFindingModel, Contributor
- **Index**: FindingModelIndex with search, validation
- **MCP Server**: IDE integration for index queries (no AI)
- **CLI**: `fm-tool search`, `fm-tool show`, `fm-tool validate`

### findingmodel-ai: AI Authoring Tools

- **Tools**: 8 tool modules with 14 AI agents across 5 providers
- **Model management**: Three-tier system (base/small/full) + per-agent overrides
- **Evals**: Agent evaluation suites
- **CLI**: `findingmodel-ai enrich`, `findingmodel-ai edit`, etc.

## Configuration

See [Configuration Details](monorepo_plan_phases/configuration-details.md) for:
- Root pyproject.toml structure
- Per-package pyproject.toml examples
- Configuration hierarchy
- Environment variable strategy
- Migration notes

## Test Topology

### Test Organization

Each package has its own `tests/` directory with package-specific tests:

```
packages/
├── oidm-common/tests/          # Unit tests for infrastructure
├── anatomic-locations/tests/   # Anatomic index, migration tests
├── findingmodel/tests/         # Core model, index, MCP tests
└── findingmodel-ai/
    ├── tests/                  # AI tool unit tests
    └── evals/                  # Agent evaluation suites
```

### Running Tests

```bash
# Workspace-wide (respects per-package pytest.ini_options)
uv run pytest packages/*/tests -m "not callout"

# Per-package
uv run --package oidm-common pytest
uv run --package anatomic-locations pytest
uv run --package findingmodel pytest
uv run --package findingmodel-ai pytest

# Evals (separate from tests)
uv run --package findingmodel-ai python -m evals.model_editor
```

### Test Markers

Markers defined in root pyproject.toml apply to all packages:

```toml
[tool.pytest.ini_options]
markers = [
    "callout: marks tests that make external API calls",
]
```

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

## Taskfile Organization

```yaml
version: "3"

tasks:
  # Workspace-wide
  test:
    desc: "Run all tests"
    cmds:
      - uv run pytest packages/*/tests -m "not callout"

  test-full:
    desc: "Run all tests including callouts"
    cmds:
      - uv run pytest packages/*/tests

  check:
    desc: "Format, lint, type-check"
    cmds:
      - uv run ruff format packages/
      - uv run ruff check --fix packages/
      - uv run mypy packages/

  # Per-package
  test:oidm-common:
    desc: "Test oidm-common"
    cmds:
      - uv run --package oidm-common pytest

  test:anatomic:
    desc: "Test anatomic-locations"
    cmds:
      - uv run --package anatomic-locations pytest

  test:findingmodel:
    desc: "Test findingmodel core"
    cmds:
      - uv run --package findingmodel pytest

  test:findingmodel-ai:
    desc: "Test findingmodel-ai"
    cmds:
      - uv run --package findingmodel-ai pytest

  evals:
    desc: "Run AI agent evaluations"
    cmds:
      - uv run --package findingmodel-ai python -m evals.model_editor
      - uv run --package findingmodel-ai python -m evals.similar_models
      # ... etc

  # Build/publish
  build:all:
    desc: "Build all packages"
    cmds:
      - uv build --package oidm-common
      - uv build --package anatomic-locations
      - uv build --package findingmodel
      - uv build --package findingmodel-ai
```

## Publishing Workflow

Packages must be published in dependency order:

1. **oidm-common** (no internal deps)
2. **anatomic-locations** (depends on oidm-common)
3. **findingmodel** (depends on oidm-common)
4. **findingmodel-ai** (depends on findingmodel, anatomic-locations)

```bash
# Example release
task build:all
uv publish dist/oidm_common-0.1.0-*.whl
uv publish dist/anatomic_locations-0.1.0-*.whl
uv publish dist/findingmodel-0.4.2-*.whl
uv publish dist/findingmodel_ai-0.4.2-*.whl
```

## Open Questions

1. **Notebooks location:** Root or packages/findingmodel/notebooks/?
   - Leaning toward root (demos may use multiple packages)

2. **Version coordination:** Same version for all packages or independent?
   - Leaning toward independent (they're at different maturity levels)

3. **GitHub Actions:** Single workflow or per-package?
   - Start with single, add change detection later if needed

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Import path changes break users | findingmodel keeps same package name, same public API |
| Circular dependencies | Clear dependency direction: common ← anatomic ← findingmodel |
| Complex publishing | Document release order, consider automation later |
| IDE confusion | uv sync creates proper venv, configure Python path |
| Test isolation | Per-package test directories, clear markers |

## Success Criteria

- [ ] All four packages can be built individually
- [ ] All tests pass per-package and workspace-wide
- [ ] `pip install findingmodel` works from built wheel (no AI deps)
- [ ] `pip install findingmodel-ai` works from built wheel (includes AI deps)
- [ ] `pip install anatomic-locations` works from built wheel
- [ ] `pip install oidm-common` works from built wheel
- [ ] MCP server works with just `findingmodel` installed
- [ ] Evals run with `findingmodel-ai` installed
- [ ] Claude Code loads appropriate CLAUDE.md per directory
- [ ] Serena memories reflect workspace structure
