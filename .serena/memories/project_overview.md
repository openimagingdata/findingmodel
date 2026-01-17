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
│   │       ├── tools/                  # AI agents/workflows
│   │       ├── finding_model.py        # FindingModel classes
│   │       ├── index.py                # DuckDBIndex (search only)
│   │       ├── cli.py                  # fm-tool
│   │       └── mcp_server.py           # MCP for IDE access
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
│  (future)        │  │  findingmodel   │  │ anatomic-locations  │
│  findingmodel-ai │  │ (core package)  │  │  (anatomic ontology)│
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

## Tech Stack
- **Language**: Python 3.11+
- **Build System**: uv workspaces
- **Task Runner**: go-task
- **Core deps**: pydantic v2, duckdb, pydantic-ai-slim, click, rich, loguru

## CLI Commands

```bash
# User commands
fm-tool search "query"              # Search finding models
fm-tool index stats                 # Show index statistics
anatomic query "nasal"              # Query anatomic locations
anatomic stats                      # Show anatomic DB stats

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
