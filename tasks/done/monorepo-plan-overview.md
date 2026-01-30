# Plan: Monorepo Restructure with uv Workspaces

**Status:** ✅ COMPLETE (All phases done; Phase 4.9 cleanup tracked as post-merge issues)
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
│   │   ├── src/findingmodel_ai/
│   │   │   ├── __init__.py
│   │   │   ├── authoring/              # Model creation and editing
│   │   │   │   ├── description.py      # create_info_from_name, add_details_to_info
│   │   │   │   ├── editor.py           # edit_model_natural_language, edit_model_markdown
│   │   │   │   └── markdown_in.py      # create_model_from_markdown
│   │   │   ├── search/                 # Search and matching
│   │   │   │   ├── anatomic.py         # find_anatomic_locations
│   │   │   │   ├── ontology.py         # match_ontology_concepts
│   │   │   │   └── similar.py          # find_similar_models
│   │   │   ├── enrichment/             # Finding enrichment workflows
│   │   │   ├── _internal/              # Shared internals
│   │   │   ├── config.py               # AI settings, model tiers
│   │   │   └── cli.py                  # findingmodel-ai CLI
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
| [3.7](monorepo_plan_phases/phase-3.7-maintenance.md) | Create oidm-maintenance | ✅ COMPLETE | Build/publish ops in dedicated package |
| [3.8](monorepo_plan_phases/phase-3.8-critical-fixes.md) | Complete read-only migration | ✅ COMPLETE | DuckDBIndex read-only, tests use pre-built fixture |
| [3.9](monorepo_plan_phases/phase-3.9-dependency-cleanup.md) | Dependency cleanup | ✅ COMPLETE | Each package declares what it imports directly |
| [4](monorepo_plan_phases/phase-4-findingmodel-ai.md) | Create findingmodel-ai | ✅ COMPLETE | Package created, tools moved |
| [4.5](monorepo_plan_phases/phase-4.5-ai-separation-cleanup.md) | AI Separation Cleanup | ✅ COMPLETE | Clean separation (config, CLI, embeddings via oidm-common) |
| [4.6](monorepo_plan_phases/phase-4.6-duckdb-search-consolidation.md) | DuckDB Search Consolidation | ✅ COMPLETE | Proper async patterns with asyncer, hybrid search in anatomic-locations |
| [4.7](monorepo_plan_phases/phase-4.7-ai-separation-cleanup.md) | AI Separation Cleanup | ✅ COMPLETE | Tests moved to findingmodel-ai, config trimmed |
| [5](monorepo_plan_phases/phase-5-cleanup.md) | Clean up findingmodel | ✅ COMPLETE | AI tools removed, non-AI utils retained |
| [4.9](monorepo_plan_phases/phase-4.9-refactor-cleanup.md) | Refactor Cleanup | 📋 DEFERRED | Tracked as post-merge issues |
| [6](monorepo_plan_phases/phase-6-documentation.md) | Documentation and AI Setup | ✅ COMPLETE | CLAUDE.md, rules, Serena |

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
- **CLI**: `anatomic-locations stats`, `anatomic-locations query ancestors/descendants/laterality/code`

### findingmodel: Core Library

- **Models**: FindingModel, FindingInfo, AbstractFindingModel, Contributor
- **Index**: DuckDBIndex with search (read-only)
- **MCP Server**: IDE integration for index queries (no AI)
- **CLI**: `findingmodel` (search, stats, validate)

### findingmodel-ai: AI Authoring Tools

- **Modules**: `authoring/` (description, editor, markdown_in), `search/` (anatomic, ontology, similar), `enrichment/`
- **Model management**: Three-tier system (base/small/full) + per-agent overrides
- **Evals**: Agent evaluation suites
- **CLI**: `findingmodel-ai` (make-info, make-stub-model, markdown-to-fm)

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

Tests must be run per-package (a single `pytest packages/*/tests` invocation fails due to conftest collisions across packages).

```bash
# All packages (via Taskfile, recommended)
task test                    # Unit tests only (no API calls)
task test-full               # Including integration tests

# Per-package
task test:oidm-common
task test:anatomic
task test:findingmodel
task test:findingmodel-ai
task test:maintenance

# Or directly with uv
uv run pytest packages/oidm-common/tests -rs -m "not callout"

# Evals (separate from tests)
task evals
task evals:model_editor      # Individual eval suite
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

## Taskfile Organization

The actual `Taskfile.yml` runs tests per-package to avoid conftest collisions:

```yaml
version: "3"

tasks:
  # Aggregate (runs per-package sequentially)
  test:
    desc: "Test all packages (no callouts)"
    cmds:
      - task: test:oidm-common
      - task: test:anatomic
      - task: test:findingmodel
      - task: test:findingmodel-ai
      - task: test:maintenance

  # Per-package
  test:oidm-common:
    cmds: [uv run pytest packages/oidm-common/tests -rs -m "not callout"]
  test:anatomic:
    cmds: [uv run pytest packages/anatomic-locations/tests -rs]
  test:findingmodel:
    cmds: [uv run pytest packages/findingmodel/tests -rs -m "not callout"]
  test:findingmodel-ai:
    cmds: [uv run pytest packages/findingmodel-ai/tests -rs -m "not callout"]
  test:maintenance:
    cmds: [uv run pytest packages/oidm-maintenance/tests -rs]

  check:
    cmds:
      - uv run ruff format packages/
      - uv run ruff check --fix packages/
      - uv run mypy

  evals:
    cmds:
      - PYTHONPATH=packages/findingmodel-ai uv run python -m evals.model_editor
      - PYTHONPATH=packages/findingmodel-ai uv run python -m evals.similar_models
      # ... etc

  build:packages:
    cmds:
      - rm -rf dist/
      - uv build --package oidm-common
      - uv build --package findingmodel
      - uv build --package anatomic-locations
      - uv build --package findingmodel-ai

  publish:pypi:
    deps: [build:packages]
    cmds:
      - uv publish dist/oidm_common-*.tar.gz dist/oidm_common-*.whl
      - uv publish dist/findingmodel-[0-9]*.tar.gz dist/findingmodel-[0-9]*.whl
      - uv publish dist/anatomic_locations-*.tar.gz dist/anatomic_locations-*.whl
      - uv publish dist/findingmodel_ai-*.tar.gz dist/findingmodel_ai-*.whl
```

## Publishing Workflow

Packages must be published in dependency order:

1. **oidm-common** 0.2.0 (no internal deps)
2. **findingmodel** 1.0.0 + **anatomic-locations** 0.2.0 (both depend on oidm-common; can publish in parallel)
3. **findingmodel-ai** 0.2.0 (depends on findingmodel + anatomic-locations)

`oidm-maintenance` is internal-only and not published to PyPI.

```bash
task build:packages    # Build all (strips workspace references)
task publish:pypi      # Publish in dependency order
```

See `tasks/pre-merge-release-punchlist.md` for the detailed release plan.

## Resolved Questions

1. **Notebooks location:** Root (`notebooks/`) — demos may use multiple packages.
2. **Version coordination:** Independent versions — packages are at different maturity levels (`findingmodel` at 1.0.0, others at 0.2.0).
3. **GitHub Actions:** Not yet configured. Start with single workflow.
4. **OIDM_MAINTAIN_ env prefix:** Dropped as YAGNI — `MaintenanceSettings` uses unprefixed env vars with `.env` file loading.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Import path changes break users | findingmodel keeps same package name, same public API |
| Circular dependencies | Clear dependency direction: common ← anatomic ← findingmodel |
| Complex publishing | Document release order, consider automation later |
| IDE confusion | uv sync creates proper venv, configure Python path |
| Test isolation | Per-package test directories, clear markers |

## Success Criteria

- [x] All five packages can be built individually (`task build:packages`)
- [x] All tests pass per-package (594 passing)
- [x] `pip install findingmodel` works from built wheel (no AI deps)
- [x] `pip install findingmodel-ai` works from built wheel (includes AI deps)
- [x] `pip install anatomic-locations` works from built wheel
- [x] `pip install oidm-common` works from built wheel
- [x] MCP server works with just `findingmodel` installed
- [x] Evals run with `findingmodel-ai` installed
- [x] Claude Code loads appropriate CLAUDE.md per directory
- [x] Serena memories reflect workspace structure
- [x] `task verify:install` passes (isolated install verification)
