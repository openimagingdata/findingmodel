# FindingModel Project Overview

## Purpose
The `findingmodel` monorepo provides Python libraries for managing Open Imaging Finding Models and related medical imaging ontologies. It uses a uv workspace structure with multiple packages.

## Monorepo Structure

```
findingmodel/                           # Workspace root
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
│   │       └── cli.py                  # query, stats
│   │
│   ├── findingmodel/                   # Core package (READ-ONLY)
│   │   └── src/findingmodel/
│   │       ├── tools/                  # Non-AI utilities
│   │       ├── facets.py               # Canonical metadata types (BodyRegion, Subspecialty, etc.)
│   │       ├── finding_model.py        # FindingModel classes (with structured metadata fields)
│   │       ├── index.py                # FindingModelIndex: search(), browse(), related_models()
│   │       ├── cli.py                  # fm-tool
│   │       └── mcp_server.py           # MCP for IDE access
│   │
│   ├── findingmodel-ai/                # AI-powered workflows
│   │   └── src/findingmodel_ai/
│   │       ├── metadata/               # Structured metadata assignment
│   │       │   ├── assignment.py       # assign_metadata() — main pipeline
│   │       │   └── types.py            # MetadataAssignmentResult, review types
│   │       ├── search/                 # Search & matching
│   │       │   ├── ontology.py         # SNOMED/RadLex matching
│   │       │   ├── anatomic.py         # Anatomic location search
│   │       │   ├── similar.py          # 5-phase similar model search
│   │       │   ├── pipeline_helpers.py # Similar search types
│   │       │   └── bioontology.py      # BioOntology API client
│   │       ├── authoring/              # Model creation & editing
│   │       │   ├── description.py      # FindingInfo generation
│   │       │   ├── markdown_in.py      # Markdown → FindingModel
│   │       │   └── editor.py           # NL/markdown editing
│   │       ├── _internal/              # Private utilities
│   │       ├── observability.py        # Logfire setup (single source of truth)
│   │       ├── config.py               # Per-agent model config, FallbackModel chains
│   │       └── cli.py                  # findingmodel-ai CLI
│   │
│   └── oidm-maintenance/               # Build/publish (maintainers only)
│
├── docs/                               # Documentation
│   ├── configuration.md                # Model config reference
│   ├── agent-performance-audit-2026-03.md  # Benchmark data
│   └── plans/                          # Active plans
├── scripts/
│   └── benchmark_models.py             # Subprocess-based benchmarking with Logfire
└── evals/                              # Quality evaluation suites
```

## Key Design Decisions

1. **Read-only user packages**: findingmodel and anatomic-locations only query pre-built databases
2. **AI separation**: findingmodel-ai contains all AI/LLM dependencies
3. **Per-agent model config**: No tiers — each agent has its own model + reasoning in supported_models.toml
4. **FallbackModel chains**: Cross-provider resilience via pydantic-ai FallbackModel
5. **Process-scoped config**: Configuration fixed at import time; benchmarks use subprocess isolation
6. **Logfire observability**: Token through pydantic-settings, httpx instrumentation for external API tracing

## Tech Stack
- **Language**: Python 3.11+
- **Build System**: uv workspaces
- **Task Runner**: go-task
- **Core deps**: pydantic v2, duckdb, click, rich, loguru
- **AI deps** (findingmodel-ai only): pydantic-ai-slim>=1.0.0, tavily, httpx
- **Observability**: logfire[httpx]>=1.0.0