# FindingModel Project Overview

## Purpose
The `findingmodel` monorepo provides Python libraries for managing Open Imaging Finding Models and related medical imaging ontologies. It uses a uv workspace structure with multiple packages.

## Monorepo Structure

```
findingmodel/                           # Workspace root
├── pyproject.toml                      # Workspace config (no package)
├── uv.lock                             # Single lockfile
├── CLAUDE.md                           # AI instructions
├── Taskfile.yml                        # Task runner
├── .serena/memories/                   # Shared AI context
│
├── packages/
│   ├── oidm-common/                    # Shared infrastructure
│   │   └── src/oidm_common/
│   │       ├── duckdb/                 # Connection, search, indexes
│   │       ├── embeddings/             # Cache, OpenAI provider
│   │       ├── distribution/           # Manifest, download, paths
│   │       └── models/                 # IndexCode, WebReference
│   │
│   ├── anatomic-locations/             # Anatomic ontology (READ-ONLY)
│   │   └── src/anatomic_locations/
│   │       ├── models/                 # AnatomicLocation, enums
│   │       ├── index.py                # AnatomicLocationIndex
│   │       ├── config.py               # Settings
│   │       └── cli.py                  # query, stats
│   │
│   ├── findingmodel/                   # Core package (READ-ONLY)
│   │   └── src/findingmodel/
│   │       ├── tools/                  # Non-AI utilities
│   │       ├── finding_model.py        # FindingModel classes
│   │       ├── index.py                # DuckDBIndex (search only)
│   │       ├── cli.py                  # fm-tool
│   │       └── mcp_server.py           # MCP for IDE access
│   │
│   ├── findingmodel-ai/                # AI-powered workflows
│   │   └── src/findingmodel_ai/
│   │       ├── enrichment/             # Finding enrichment pipelines
│   │       │   ├── unified.py          # 3-stage enrichment
│   │       │   └── agentic.py          # Tool-calling approach
│   │       ├── search/                 # Search & matching
│   │       │   ├── ontology.py         # SNOMED/RadLex matching
│   │       │   ├── anatomic.py         # Anatomic location search
│   │       │   ├── similar.py          # Similar model search
│   │       │   └── bioontology.py      # BioOntology API client
│   │       ├── authoring/              # Model creation & editing
│   │       │   ├── description.py      # FindingInfo generation
│   │       │   ├── markdown_in.py      # Markdown → FindingModel
│   │       │   └── editor.py           # NL/markdown editing
│   │       ├── _internal/              # Private utilities
│   │       ├── evaluators.py           # pydantic_evals classes
│   │       ├── config.py               # AI settings
│   │       └── cli.py                  # fm-ai
│   │
│   └── oidm-maintenance/               # Build/publish (maintainers only)
│       └── src/oidm_maintenance/
│           ├── anatomic/               # Anatomic DB build
│           ├── findingmodel/           # FindingModel DB build
│           ├── s3.py                   # S3 upload, manifest
│           └── cli.py                  # oidm-maintain
│
├── docs/                               # Documentation
├── tasks/                              # Planning documents
└── notebooks/                          # Demos
```

## Package Dependencies

```
                    ┌─────────────────────┐
                    │  oidm-maintenance   │  ← Build & publish
                    └─────────┬───────────┘
                              │ depends on all
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  findingmodel-ai │  │  findingmodel   │  │ anatomic-locations  │
│ (AI workflows)   │  │ (core package)  │  │  (anatomic ontology)│
└───────┬──────────┘  └────────┬────────┘  └─────────┬───────────┘
        │                      │                     │
        └──────────────────────┴─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     oidm-common     │
                    │   (infrastructure)  │
                    └─────────────────────┘
```

## Key Design Decisions

1. **Read-only user packages**: findingmodel and anatomic-locations only query pre-built databases
2. **Maintainer-only builds**: oidm-maintenance handles all database creation/publishing
3. **Explicit dependencies**: Each package declares all dependencies it directly imports
4. **Single lockfile**: uv.lock ensures consistent versions across all packages
5. **AI separation**: findingmodel-ai contains all AI/LLM dependencies, core findingmodel has none

## Tech Stack
- **Language**: Python 3.11+
- **Build System**: uv workspaces
- **Task Runner**: go-task
- **Core deps**: pydantic v2, duckdb, click, rich, loguru
- **AI deps** (findingmodel-ai only): pydantic-ai-slim, tavily, httpx

## CLI Commands

```bash
# User commands
fm-tool search "query"              # Search finding models
fm-tool index stats                 # Show index statistics
anatomic query "nasal"              # Query anatomic locations
anatomic stats                      # Show anatomic DB stats

# AI commands (findingmodel-ai)
fm-ai make-info "pneumothorax"      # Generate FindingInfo from name
fm-ai make-stub-model "finding"     # Generate stub model
fm-ai markdown-to-fm file.md        # Convert markdown to model

# Maintainer commands (oidm-maintenance)
oidm-maintain findingmodel build    # Build findingmodel database
oidm-maintain findingmodel publish  # Publish to S3
oidm-maintain anatomic build        # Build anatomic database
oidm-maintain anatomic publish      # Publish to S3
```

## Development

```bash
task test                           # Run all tests (no callout)
task test-full                      # Run all tests including API calls
task check                          # Format + lint + type check
uv sync --all-packages              # Sync workspace
```
