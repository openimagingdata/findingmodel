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

### Root pyproject.toml

```toml
[project]
name = "findingmodel-workspace"
version = "0.0.0"
description = "Workspace root for OIDM packages"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["packages/*"]

# Shared tool configuration
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
markers = [
    "callout: marks tests that make external API calls",
]
```

**Python Version Note:** All packages target `requires-python = ">=3.11"`. uv workspaces enforce a single `requires-python` by taking the intersection of all members' values. This is intentional—we commit to Python 3.11+ across the entire workspace. If a future package needs Python 3.12+ features, we would bump the entire workspace.

### packages/oidm-common/pyproject.toml

```toml
[project]
name = "oidm-common"
version = "0.1.0"
description = "Shared infrastructure for OIDM packages"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.9",
    "httpx>=0.27",
    "platformdirs>=4.0",
    "loguru>=0.7",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]

[build-system]
requires = ["uv_build>=0.9,<0.10"]
build-backend = "uv_build"
```

### packages/anatomic-locations/pyproject.toml

```toml
[project]
name = "anatomic-locations"
version = "0.1.0"
description = "Anatomic location ontology navigation"
requires-python = ">=3.11"
dependencies = [
    "oidm-common",
    "duckdb>=1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.9",
    "click>=8.0",
    "rich>=13.0",
]

[project.optional-dependencies]
build = ["openai>=1.0"]

[project.scripts]
anatomic = "anatomic_locations.cli:main"

[build-system]
requires = ["uv_build>=0.9,<0.10"]
build-backend = "uv_build"

[tool.uv.sources]
oidm-common = { workspace = true }
```

### packages/findingmodel/pyproject.toml

```toml
[project]
name = "findingmodel"
version = "0.6.1"  # Current version
description = "Open Imaging Finding Model library - core models, index, and MCP server"
requires-python = ">=3.11"
dependencies = [
    "oidm-common",
    "duckdb>=1.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.9",
    "click>=8.0",
    "rich>=13.0",
    "mcp>=1.0",
    "httpx>=0.28",
]

[project.scripts]
fm-tool = "findingmodel.cli:cli"
findingmodel-mcp = "findingmodel.mcp_server:main"

[build-system]
requires = ["uv_build>=0.9,<0.10"]
build-backend = "uv_build"

[tool.uv.sources]
oidm-common = { workspace = true }
```

### packages/findingmodel-ai/pyproject.toml

```toml
[project]
name = "findingmodel-ai"
version = "0.6.1"
description = "AI authoring tools for Open Imaging Finding Models"
requires-python = ">=3.11"
dependencies = [
    "findingmodel",
    "anatomic-locations",
    "pydantic-settings>=2.9",
    "openai>=1.0",
    "anthropic>=0.5",
    "pydantic-ai-slim[openai,tavily,anthropic]>=0.3",
    "tavily-python>=0.6",
]

[project.scripts]
findingmodel-ai = "findingmodel_ai.cli:main"

[build-system]
requires = ["uv_build>=0.9,<0.10"]
build-backend = "uv_build"

[tool.uv.sources]
findingmodel = { workspace = true }
anatomic-locations = { workspace = true }
```

## Per-Package Configuration

Each package has its own `config.py` using pydantic-settings for package-specific configuration. This ensures:
- Clear ownership of settings per package
- No circular imports between packages
- Users can configure only what they install

### Configuration Hierarchy

```
oidm_common/config.py      → Base settings (cache dirs, download URLs)
    ↓
anatomic_locations/config.py → Anatomic DB paths, search settings
    ↓
findingmodel/config.py     → Index paths, MCP server settings
    ↓
findingmodel_ai/config.py  → API keys, model selection, Tavily settings
```

### Package Configuration Responsibilities

| Package | Config Responsibility | Key Settings |
|---------|----------------------|--------------|
| **oidm-common** | Infrastructure, embeddings, observability | `OIDM_CACHE_DIR`, `OIDM_MANIFEST_URL`, `OPENAI_API_KEY` (embeddings), `LOGFIRE_TOKEN` |
| **anatomic-locations** | Anatomic database paths | `ANATOMIC_DB_PATH` |
| **findingmodel** | Core index/MCP | `FINDINGMODEL_DB_PATH`, `FINDINGMODEL_MCP_PORT` |
| **findingmodel-ai** | AI model management (5 providers, 14 agents) | `DEFAULT_MODEL*`, `AGENT_MODEL_OVERRIDES__*`, all API keys, `TAVILY_*`, `BIOONTOLOGY_*` |

### Implementation Pattern

Each package follows this pattern:

```python
# packages/oidm-common/src/oidm_common/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from platformdirs import user_cache_dir

class OidmCommonSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OIDM_")

    cache_dir: Path = Path(user_cache_dir("oidm"))
    manifest_url: str = "https://..."
    download_timeout: int = 300

# Singleton instance
_settings: OidmCommonSettings | None = None

def get_settings() -> OidmCommonSettings:
    global _settings
    if _settings is None:
        _settings = OidmCommonSettings()
    return _settings
```

```python
# packages/findingmodel/src/findingmodel/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from oidm_common.config import get_settings as get_common_settings

class FindingModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINDINGMODEL_")

    db_path: Path | None = None  # Falls back to common cache
    mcp_server_port: int = 8080

    @property
    def effective_db_path(self) -> Path:
        if self.db_path:
            return self.db_path
        return get_common_settings().cache_dir / "findingmodel.duckdb"
```

```python
# packages/findingmodel-ai/src/findingmodel_ai/config.py
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_ai import Model

# Type definitions
ModelTier = Literal["base", "small", "full"]
AgentTag = Literal[
    "enrich_classify", "enrich_unified", "enrich_research",  # Enrichment
    "anatomic_search", "anatomic_select",                     # Anatomic
    "similar_search", "similar_assess",                       # Similar models
    "ontology_match", "ontology_search",                      # Ontology
    "edit_instructions", "edit_markdown",                     # Editing
    "describe_finding", "describe_details",                   # Description
    "import_markdown",                                        # Import
]
ModelSpec = str  # e.g., "openai:gpt-5-mini", "anthropic:claude-sonnet-4-5"

class FindingModelAISettings(BaseSettings):
    """AI tool settings - manages models for all 14 agent tags across 5 providers."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    # API keys - standard names (no prefix)
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    google_api_key: SecretStr = Field(default=SecretStr(""))
    pydantic_ai_gateway_api_key: SecretStr = Field(default=SecretStr(""))
    ollama_base_url: str = Field(default="http://localhost:11434/v1")
    tavily_api_key: SecretStr = Field(default=SecretStr(""))
    bioontology_api_key: SecretStr | None = Field(default=None)

    # Model tier defaults - standard names (no prefix)
    default_model: ModelSpec = Field(default="openai:gpt-5-mini")
    default_model_full: ModelSpec = Field(default="openai:gpt-5.2")
    default_model_small: ModelSpec = Field(default="openai:gpt-5-nano")

    # Per-agent overrides via AGENT_MODEL_OVERRIDES__<tag>=provider:model
    agent_model_overrides: dict[AgentTag, ModelSpec] = Field(default_factory=dict)

    def get_model(self, tier: ModelTier = "base") -> Model:
        """Get model instance for tier."""
        model_string = {"base": self.default_model, "full": self.default_model_full,
                        "small": self.default_model_small}[tier]
        return self._create_model_from_string(model_string, tier)

    def get_agent_model(self, tag: AgentTag, *, default_tier: ModelTier = "base") -> Model:
        """Get model for agent, with per-agent override support."""
        if tag in self.agent_model_overrides:
            return self._create_model_from_string(self.agent_model_overrides[tag], default_tier)
        return self.get_model(default_tier)

    def _create_model_from_string(self, model_string: str, tier: ModelTier) -> Model:
        """Factory method dispatching to provider-specific model creation."""
        # Dispatches to _make_openai_model, _make_anthropic_model, etc.
        ...
```

### Environment Variable Strategy

**Principle:** Maintain compatibility with existing env var names. Use prefixes only for new package-specific settings.

#### AI Configuration (findingmodel-ai) - No Prefix

These match the current implementation and require no changes for existing users:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API access |
| `ANTHROPIC_API_KEY` | Anthropic API access |
| `GOOGLE_API_KEY` | Google Gemini API access |
| `PYDANTIC_AI_GATEWAY_API_KEY` | Pydantic AI Gateway |
| `OLLAMA_BASE_URL` | Ollama server URL |
| `DEFAULT_MODEL` | Base tier model (e.g., `openai:gpt-5-mini`) |
| `DEFAULT_MODEL_FULL` | Full tier model (e.g., `openai:gpt-5.2`) |
| `DEFAULT_MODEL_SMALL` | Small tier model (e.g., `openai:gpt-5-nano`) |
| `AGENT_MODEL_OVERRIDES__<tag>` | Per-agent override (e.g., `AGENT_MODEL_OVERRIDES__enrich_classify=anthropic:claude-opus-4-5`) |
| `TAVILY_API_KEY` | Tavily search API |
| `BIOONTOLOGY_API_KEY` | BioOntology.org API |

#### Infrastructure (oidm-common) - No Prefix for Standard Keys

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | For embeddings (shared with findingmodel-ai) |
| `AWS_ACCESS_KEY_ID` | S3 distribution access |
| `AWS_SECRET_ACCESS_KEY` | S3 distribution access |
| `LOGFIRE_TOKEN` | Observability |

#### Package-Specific Settings (With Prefix)

| Package | Env Prefix | Examples |
|---------|-----------|----------|
| oidm-common | `OIDM_` | `OIDM_CACHE_DIR`, `OIDM_MANIFEST_URL` |
| anatomic-locations | `ANATOMIC_` | `ANATOMIC_DB_PATH` |
| findingmodel | `FINDINGMODEL_` | `FINDINGMODEL_DB_PATH`, `FINDINGMODEL_MCP_PORT` |

**Note:** findingmodel-ai does NOT use a prefix for AI settings - this maintains backward compatibility with the current implementation.

#### Implementation Pattern

For oidm-common which needs both prefixed settings and standard API keys:

```python
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class OidmCommonSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OIDM_")

    # Package-specific (uses OIDM_ prefix)
    cache_dir: Path = Path(user_cache_dir("oidm"))
    manifest_url: str = "https://..."

    # Standard API keys (no prefix via validation_alias)
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: SecretStr | None = Field(default=None, validation_alias="AWS_SECRET_ACCESS_KEY")
```

For findingmodel-ai which uses all standard names:

```python
class FindingModelAISettings(BaseSettings):
    # No env_prefix - all fields use their natural names
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    openai_api_key: SecretStr = ...      # reads OPENAI_API_KEY
    default_model: str = ...              # reads DEFAULT_MODEL
    agent_model_overrides: dict = ...     # reads AGENT_MODEL_OVERRIDES__*
```

### Migration Notes

Current `src/findingmodel/config.py` is a monolith that must be carefully split. The table below shows exactly where each field goes:

#### FindingModelConfig Field Migration

| Current Field | Target Package | Target Config | Notes |
|---------------|----------------|---------------|-------|
| **API Keys** ||||
| `openai_api_key` | findingmodel-ai | FindingModelAISettings | Standard env var name |
| `anthropic_api_key` | findingmodel-ai | FindingModelAISettings | Standard env var name |
| `google_api_key` | findingmodel-ai | FindingModelAISettings | Standard env var name |
| `pydantic_ai_gateway_api_key` | findingmodel-ai | FindingModelAISettings | Standard env var name |
| `ollama_base_url` | findingmodel-ai | FindingModelAISettings | Standard env var name |
| `tavily_api_key` | findingmodel-ai | FindingModelAISettings | Standard env var name |
| `bioontology_api_key` | findingmodel-ai | FindingModelAISettings | Standard env var name |
| **Model Configuration** ||||
| `default_model` | findingmodel-ai | FindingModelAISettings | DEFAULT_MODEL env var |
| `default_model_full` | findingmodel-ai | FindingModelAISettings | DEFAULT_MODEL_FULL env var |
| `default_model_small` | findingmodel-ai | FindingModelAISettings | DEFAULT_MODEL_SMALL env var |
| `agent_model_overrides` | findingmodel-ai | FindingModelAISettings | AGENT_MODEL_OVERRIDES__* |
| `AgentTag` type | findingmodel-ai | types.py | 14 agent tag literals |
| `ModelTier` type | findingmodel-ai | types.py | Literal["base", "small", "full"] |
| `get_model()` method | findingmodel-ai | FindingModelAISettings | Model factory by tier |
| `get_agent_model()` method | findingmodel-ai | FindingModelAISettings | Model factory by agent tag |
| `_make_*_model()` methods | findingmodel-ai | FindingModelAISettings | 7 provider factory methods |
| **Database Paths** ||||
| `duckdb_index_path` | findingmodel | FindingModelSettings | FINDINGMODEL_DB_PATH |
| `duckdb_anatomic_path` | anatomic-locations | AnatomicSettings | ANATOMIC_DB_PATH |
| **Distribution** ||||
| `remote_manifest_url` | oidm-common | OidmCommonSettings | OIDM_MANIFEST_URL |
| `remote_index_db_url/hash` | findingmodel | FindingModelSettings | For locked versions |
| `remote_anatomic_db_url/hash` | anatomic-locations | AnatomicSettings | For locked versions |
| **Embeddings** ||||
| `openai_embedding_model` | oidm-common | OidmCommonSettings | Shared by anatomic search |
| `openai_embedding_dimensions` | oidm-common | OidmCommonSettings | Shared by anatomic search |
| **Observability** ||||
| `logfire_token` | oidm-common | OidmCommonSettings | LOGFIRE_TOKEN |
| `disable_send_to_logfire` | oidm-common | OidmCommonSettings | For dev |
| `logfire_verbose` | oidm-common | OidmCommonSettings | For debug |

#### Key Insight: AI Model Management is Entirely in findingmodel-ai

All `get_agent_model()` calls are in `tools/` modules that move to findingmodel-ai. The core findingmodel package does NOT use AI models - it only provides:
- Data models (FindingModel, FindingInfo)
- Index access (read-only search)
- MCP server (index queries, no AI)

This clean separation means:
- `pip install findingmodel` → No AI dependencies, no API keys needed
- `pip install findingmodel-ai` → Full AI tools, all provider support

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

### Per-Package pytest Configuration

Each package can override root pytest settings in its pyproject.toml:

```toml
# packages/findingmodel/pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Assessment Note:** This per-package `testpaths` override may not be necessary—`uv run --package <name> pytest` already scopes to that package's directory. Test this assumption during Phase 1 implementation before adding config to all packages. If `uv run --package findingmodel pytest` correctly discovers `packages/findingmodel/tests/` without explicit config, skip these overrides.

### Cross-Package Tests

Some tests may need to verify integration between packages. Options:

1. **Keep in dependent package**: A test in `findingmodel-ai/tests/` that uses `anatomic-locations` is fine since it has that dependency
2. **Integration test directory** (if needed later): `tests/integration/` at workspace root

### Test Data and Fixture Sharing

**Principle:** Test data lives with the package that owns the domain. Cross-package access uses pytest fixtures that resolve paths at runtime.

| Package | Owns Test Data | Accesses From |
|---------|----------------|---------------|
| oidm-common | `tests/data/` (DuckDB fixtures) | — |
| anatomic-locations | `tests/data/anatomic_*.json` | oidm-common |
| findingmodel | `tests/data/finding_models/`, `tests/data/embeddings/` | oidm-common |
| findingmodel-ai | `tests/data/` (AI-specific mocks) | findingmodel, anatomic-locations |

#### Cross-Package Fixture Access Pattern

When `findingmodel-ai` tests need finding model fixtures from `findingmodel`:

```python
# packages/findingmodel-ai/tests/conftest.py
from pathlib import Path
import pytest

@pytest.fixture
def findingmodel_test_data_dir() -> Path:
    """Resolve path to findingmodel's test data directory."""
    # Navigate from findingmodel-ai/tests/ to findingmodel/tests/data/
    return Path(__file__).parent.parent.parent / "findingmodel" / "tests" / "data"

@pytest.fixture
def sample_finding_model_path(findingmodel_test_data_dir: Path) -> Path:
    """Path to a sample finding model JSON for testing."""
    return findingmodel_test_data_dir / "finding_models" / "sample.fm.json"
```

#### Why This Pattern?

1. **No duplication** - Single source of truth for test data
2. **Clear ownership** - Each package owns its domain's fixtures
3. **Works in CI** - Relative paths from workspace root
4. **Explicit dependencies** - Fixture names document the cross-package relationship
5. **IDE-friendly** - Standard pytest patterns, no magic

#### Alternative Considered: Shared Test Utils Package

Creating a `packages/test-utils/` package was considered but rejected:
- Adds maintenance overhead for a non-publishable package
- YAGNI—conftest.py fixtures are sufficient
- Would need special handling in CI/publishing

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

## Migration Phases

### Phase 0: Preparation ✅ COMPLETE

**Goal:** Clean starting point

1. ✅ Commit/stash current anatomic-locations branch work
2. ✅ Create this plan document
3. ✅ Ensure main branch is stable

### Phase 1: Restructure findingmodel ✅ COMPLETE

**Goal:** Move existing code into packages/ without extraction

**Status:** Complete

**Completed work:**
- Created `packages/findingmodel/` structure
- Moved `src/findingmodel/` to `packages/findingmodel/src/`
- Moved tests to `packages/findingmodel/tests/`
- Created package pyproject.toml
- Updated root pyproject.toml to workspace format
- All tests passing

**Verification:** `uv run --package findingmodel pytest` ✅

### Phase 2: Extract oidm-common ✅ COMPLETE

**Goal:** Create shared infrastructure package

**Status:** Complete

**Completed work:**
- Created `packages/oidm-common/` with subpackages:
  - `duckdb/` - Connection management, bulk loading, search utilities
  - `embeddings/` - EmbeddingCache, provider protocol
  - `distribution/` - Manifest, download, path resolution
  - `models/` - IndexCode, WebReference
- Moved tests to `packages/oidm-common/tests/`
- Updated findingmodel imports to use oidm-common
- All 72 oidm-common tests passing

**Verification:**
- `uv run --package oidm-common pytest` ✅
- `uv run --package findingmodel pytest` ✅

### Phase 3: Create anatomic-locations ✅ COMPLETE

**Goal:** Move anatomic code to standalone package

**Status:** Complete

**Completed work:**
- Created `packages/anatomic-locations/` with structure:
  - `models/` - AnatomicLocation, enums, AnatomicRef
  - `index.py` - AnatomicLocationIndex with hybrid search
  - `migration.py` - Database builder with embedded `_batch_embeddings_for_duckdb`
  - `cli.py` - `anatomic search`, `anatomic show`, `anatomic build`
- Moved tests and test data to `packages/anatomic-locations/tests/`
- **No circular dependencies** - anatomic-locations depends only on oidm-common
- **findingmodel has no dependency on anatomic-locations** (re-exports replaced with ImportError stubs)
- All 80 anatomic-locations tests passing
- All 453 findingmodel tests passing

**Verification:**
- `uv run --package anatomic-locations pytest` ✅
- `uv run --package findingmodel pytest` ✅
- `anatomic --help` works ✅

### Phase 3.5: Embedding Utilities Cleanup ✅ COMPLETE

**Goal:** Consolidate embedding generation code in oidm-common

**Status:** Complete

**Completed work:**
- Created `oidm_common/embeddings/generation.py` with:
  - `generate_embedding(text, client, model, dimensions)` → single embedding
  - `generate_embeddings_batch(texts, client, model, dimensions)` → batch embeddings
  - Both return float32 precision for DuckDB compatibility
- Updated `anatomic_locations/migration.py` to import from oidm-common (removed duplicate)
- Updated `findingmodel/tools/duckdb_utils.py` to use oidm-common functions
- Updated test mocks in anatomic-locations tests

**Design decisions:**
- Standalone async functions (YAGNI - no protocol abstraction needed)
- Functions take explicit `client: AsyncOpenAI` parameter (no config dependency in oidm-common)
- Float32 conversion happens in oidm-common; consumers don't need to handle it
- `duckdb_utils.py` remains as convenience wrappers with config-based defaults

**Verification:**
- `uv run --package oidm-common pytest` ✅ 76 passed
- `uv run --package anatomic-locations pytest` ✅ 80 passed
- `uv run --package findingmodel pytest` ✅ 463 passed
- `anatomic --help` ✅ works
- `ruff check packages/` ✅ All checks passed

### Phase 3.6: Distribution Code Cleanup ✅ COMPLETE

**Goal:** Remove duplicate distribution code from findingmodel, create anatomic-locations config

**Status:** Complete

**Completed work:**
- Created `anatomic_locations/config.py` with:
  - `AnatomicLocationSettings` class (pydantic-settings, ANATOMIC_ prefix)
  - `get_settings()` singleton function
  - `ensure_anatomic_db()` using `oidm_common.distribution.ensure_db_file()`
- Updated anatomic-locations imports:
  - Removed ALL findingmodel imports from anatomic-locations package
  - Direct imports from `anatomic_locations.config` (no fallbacks)
  - Changed logger to use loguru directly
  - Changed OpenAI API key to use `os.getenv("OPENAI_API_KEY")` directly
- Cleaned findingmodel/config.py (~275 lines removed):
  - Removed duplicate distribution functions
  - Removed `ensure_anatomic_db()` (now in anatomic-locations)
  - Removed anatomic-related settings fields
  - Updated `ensure_index_db()` to use oidm-common
- Updated dependent code:
  - `findingmodel/tools/duckdb_search.py` → imports from anatomic_locations.config
  - `findingmodel/evals/anatomic_search.py` → imports from anatomic_locations.config
  - `findingmodel/tests/test_manifest_integration.py` → imports from oidm_common.distribution
- Created tests for `anatomic_locations/config.py` (10 tests)

**Key achievement:** anatomic-locations now has ZERO imports from findingmodel - truly independent.

**Verification:**
- `uv run --package oidm-common pytest` ✅ 76 passed
- `uv run --package anatomic-locations pytest` ✅ 90 passed (80 + 10 new)
- `uv run --package findingmodel pytest` ✅ 460 passed
- `anatomic --help` ✅ works without findingmodel fallback
- `ruff check packages/` ✅ All checks passed

### Phase 3.7: Create oidm-maintenance Package

**Goal:** Create a dedicated maintenance package for database build and publish operations. Make user-facing index classes read-only.

**Status:** Pending

**Architecture decision:**
Database building and publishing are **maintainer operations**, not user features. Rather than scattering this code across packages or using pseudo-package structures, we create a proper `oidm-maintenance` package that:
- Contains ALL build and publish logic in one place
- Has proper package structure (testable, versionable)
- Keeps heavy dependencies (boto3, openai) out of user packages
- Can be published to PyPI later if needed (start as workspace-only)

**Key insight:** Users never build databases—they download pre-built ones via `ensure_db_file()` and query them. All build/publish operations are maintainer-only.

**Strengths of this approach:**
- **Proper Python package**: Real pyproject.toml, tests, versioning—follows community best practices
- **No duplication**: Shared S3/manifest code in one place, used by both anatomic and findingmodel operations
- **Clean dependency isolation**: boto3, openai (for builds) never appear in user package dependencies
- **Simple mental model**: "Users install findingmodel; maintainers also install oidm-maintenance"
- **Testable**: Standard pytest structure, can mock S3 operations
- **Discoverable**: `oidm-maintain --help` shows all available commands

**Potential issues to mitigate:**
- **Extra package to maintain**: Mitigated by starting workspace-only (no PyPI releases until needed)
- **Coupling to user packages**: oidm-maintenance depends on findingmodel and anatomic-locations; schema changes require coordinated updates. Mitigated by keeping build logic close to the data models it serializes.
- **CI complexity**: Build/publish workflows need oidm-maintenance installed. Mitigated by Taskfile targets that handle this transparently.

**Target structure after Phase 3.7:**
```
packages/
  ├── oidm-common/              # Shared infrastructure (download, duckdb, embeddings)
  ├── anatomic-locations/       # READ-ONLY: search, get, query
  ├── findingmodel/             # READ-ONLY: search, get, query, MCP
  ├── findingmodel-ai/          # AI tools (Phase 4)
  └── oidm-maintenance/         # NEW: Build + publish (maintainers only)
      ├── pyproject.toml
      ├── src/oidm_maintenance/
      │   ├── __init__.py
      │   ├── config.py         # Settings (S3 credentials, endpoints)
      │   ├── s3.py             # S3 client, upload, manifest operations
      │   ├── hashing.py        # File hash computation
      │   ├── anatomic/
      │   │   ├── __init__.py
      │   │   ├── build.py      # Build anatomic database
      │   │   └── publish.py    # Publish anatomic database
      │   ├── findingmodel/
      │   │   ├── __init__.py
      │   │   ├── build.py      # Build findingmodel database
      │   │   └── publish.py    # Publish findingmodel database
      │   └── cli.py            # oidm-maintain command
      └── tests/
          ├── test_s3.py
          ├── test_hashing.py
          ├── test_anatomic_build.py
          └── test_findingmodel_build.py
```

**CLI usage after implementation:**
```bash
# Run via uv (workspace-only, not published yet)
uv run --package oidm-maintenance oidm-maintain --help

# Build databases
uv run --package oidm-maintenance oidm-maintain anatomic build --source data/anatomic.csv --output anatomic.duckdb
uv run --package oidm-maintenance oidm-maintain findingmodel build --source models/ --output findingmodel.duckdb

# Publish databases
uv run --package oidm-maintenance oidm-maintain anatomic publish anatomic.duckdb
uv run --package oidm-maintenance oidm-maintain findingmodel publish findingmodel.duckdb

# Or via Taskfile (simpler)
task maintain:anatomic:build
task maintain:anatomic:publish
task maintain:findingmodel:build
task maintain:findingmodel:publish
```

---

## Sub-phase 3.7.1: Create Package Scaffolding

**What:** Create the oidm-maintenance package directory structure and pyproject.toml.

**Steps:**

1. Create directory structure:
   ```bash
   mkdir -p packages/oidm-maintenance/src/oidm_maintenance/anatomic
   mkdir -p packages/oidm-maintenance/src/oidm_maintenance/findingmodel
   mkdir -p packages/oidm-maintenance/tests
   ```

2. Create `packages/oidm-maintenance/pyproject.toml`:
   ```toml
   [project]
   name = "oidm-maintenance"
   version = "0.1.0"
   description = "Maintenance tools for OIDM packages (database build and publish)"
   requires-python = ">=3.11"
   dependencies = [
       "oidm-common",
       "anatomic-locations",
       "findingmodel",
       "boto3>=1.40",
       "openai>=1.0",
       "rich>=13.0",
       "click>=8.0",
   ]

   [project.scripts]
   oidm-maintain = "oidm_maintenance.cli:main"

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src/oidm_maintenance"]

   [tool.uv.sources]
   oidm-common = { workspace = true }
   anatomic-locations = { workspace = true }
   findingmodel = { workspace = true }
   ```

3. Create `packages/oidm-maintenance/src/oidm_maintenance/__init__.py`:
   ```python
   """OIDM Maintenance Tools - Database build and publish utilities."""
   __version__ = "0.1.0"
   ```

4. Create empty `__init__.py` files:
   - `packages/oidm-maintenance/src/oidm_maintenance/anatomic/__init__.py`
   - `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/__init__.py`

**Verify:** `uv sync` completes without errors.

---

## Sub-phase 3.7.2: Create Shared Infrastructure Modules

**What:** Create the config, S3, and hashing modules that are shared by both anatomic and findingmodel operations.

**Steps:**

1. Create `packages/oidm-maintenance/src/oidm_maintenance/config.py`:

   ```python
   """Configuration for maintenance operations."""
   from pydantic import SecretStr
   from pydantic_settings import BaseSettings, SettingsConfigDict


   class MaintenanceSettings(BaseSettings):
       """Settings for database maintenance operations."""

       model_config = SettingsConfigDict(env_prefix="OIDM_MAINTAIN_", extra="ignore")

       # S3/Tigris settings
       s3_endpoint_url: str = "https://fly.storage.tigris.dev"
       s3_bucket: str = "findingmodelsdata"
       aws_access_key_id: SecretStr | None = None
       aws_secret_access_key: SecretStr | None = None

       # Manifest settings
       manifest_key: str = "manifest.json"
       manifest_backup_prefix: str = "manifests/archive/"

       # OpenAI for embeddings during build
       openai_api_key: SecretStr | None = None
       openai_embedding_model: str = "text-embedding-3-small"
       openai_embedding_dimensions: int = 512


   _settings: MaintenanceSettings | None = None


   def get_settings() -> MaintenanceSettings:
       """Get singleton settings instance."""
       global _settings
       if _settings is None:
           _settings = MaintenanceSettings()
       return _settings
   ```

2. Create `packages/oidm-maintenance/src/oidm_maintenance/hashing.py`:

   ```python
   """File hashing utilities."""
   from pathlib import Path

   import pooch


   def compute_file_hash(file_path: Path) -> str:
       """Compute SHA256 hash of a file.

       Args:
           file_path: Path to the file to hash.

       Returns:
           Hash string in format "sha256:abc123..."
       """
       return pooch.file_hash(str(file_path), alg="sha256")
   ```

3. Create `packages/oidm-maintenance/src/oidm_maintenance/s3.py`:

   **Source:** Copy and adapt from `packages/findingmodel/src/findingmodel/db_publish.py`

   The module should contain these functions (read the source file for implementation):
   - `create_s3_client(settings: MaintenanceSettings) -> boto3.client`
   - `upload_file_to_s3(client, bucket: str, key: str, local_path: Path) -> str`
   - `load_manifest_from_s3(client, bucket: str, key: str) -> dict`
   - `backup_manifest(client, bucket: str, manifest: dict, backup_prefix: str) -> str`
   - `save_manifest_to_s3(client, bucket: str, key: str, manifest: dict) -> None`
   - `update_manifest_entry(manifest: dict, db_key: str, entry: dict) -> dict`

   **Key changes from original:**
   - Use `MaintenanceSettings` instead of `FindingModelConfig`
   - Remove findingmodel-specific code (sanity checks, model queries)
   - Make functions pure utilities (no Rich prompts here)

**Verify:** `uv run --package oidm-maintenance python -c "from oidm_maintenance import config, s3, hashing; print('OK')"`

---

## Sub-phase 3.7.3: Create Anatomic Build Module

**What:** Move database build logic from `anatomic_locations/migration.py` to oidm-maintenance.

**Steps:**

1. Read the current implementation:
   - `packages/anatomic-locations/src/anatomic_locations/migration.py`

2. Create `packages/oidm-maintenance/src/oidm_maintenance/anatomic/build.py`:

   ```python
   """Build anatomic-locations database from source data."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.progress import Progress

   from oidm_maintenance.config import get_settings

   console = Console()


   def build_anatomic_database(
       source_csv: Path,
       output_path: Path,
       generate_embeddings: bool = True,
   ) -> Path:
       """Build the anatomic-locations DuckDB database.

       Args:
           source_csv: Path to source CSV with anatomic location data.
           output_path: Path for output DuckDB file.
           generate_embeddings: Whether to generate OpenAI embeddings.

       Returns:
           Path to the created database file.
       """
       # Implementation moved from anatomic_locations/migration.py:
       # 1. Load CSV data
       # 2. Create DuckDB connection
       # 3. Create tables (anatomic_locations with id, description, synonyms, region, sided, vector)
       # 4. Insert data
       # 5. Generate embeddings if requested (using oidm_common.embeddings)
       # 6. Create FTS index
       # 7. Create HNSW vector index
       # 8. Return output path
       ...
   ```

   **Copy the actual implementation from `migration.py`, adapting:**
   - Import `get_settings()` from `oidm_maintenance.config`
   - Use `oidm_common.embeddings.generate_embeddings_batch()` for embeddings
   - Add Rich progress bars for user feedback
   - Remove any CLI decorators (those go in cli.py)

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.4: Create Anatomic Publish Module

**What:** Create publish logic for anatomic-locations database.

**Steps:**

1. Create `packages/oidm-maintenance/src/oidm_maintenance/anatomic/publish.py`:

   ```python
   """Publish anatomic-locations database to S3/Tigris."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.prompt import Confirm
   from rich.table import Table

   from oidm_maintenance.config import get_settings
   from oidm_maintenance.hashing import compute_file_hash
   from oidm_maintenance.s3 import (
       create_s3_client,
       load_manifest_from_s3,
       update_manifest_entry,
       backup_manifest,
       save_manifest_to_s3,
       upload_file_to_s3,
   )

   console = Console()


   def get_anatomic_stats(db_path: Path) -> dict:
       """Get statistics about an anatomic database.

       Returns dict with: record_count, sample_descriptions, regions, etc.
       """
       conn = duckdb.connect(str(db_path), read_only=True)
       try:
           count = conn.execute("SELECT COUNT(*) FROM anatomic_locations").fetchone()[0]
           samples = conn.execute(
               "SELECT description FROM anatomic_locations LIMIT 5"
           ).fetchall()
           regions = conn.execute(
               "SELECT DISTINCT region FROM anatomic_locations"
           ).fetchall()
           return {
               "record_count": count,
               "sample_descriptions": [s[0] for s in samples],
               "regions": [r[0] for r in regions],
           }
       finally:
           conn.close()


   def display_anatomic_stats(stats: dict) -> None:
       """Display database statistics using Rich."""
       table = Table(title="Anatomic Database Statistics")
       table.add_column("Metric", style="cyan")
       table.add_column("Value", style="green")
       table.add_row("Record Count", str(stats["record_count"]))
       table.add_row("Regions", ", ".join(stats["regions"]))
       table.add_row("Sample Descriptions", "\n".join(stats["sample_descriptions"][:3]))
       console.print(table)


   def publish_anatomic_database(
       db_path: Path,
       version: str | None = None,
       dry_run: bool = False,
   ) -> bool:
       """Publish anatomic database to S3.

       Args:
           db_path: Path to the DuckDB file to publish.
           version: Version string (default: YYYY-MM-DD).
           dry_run: If True, show what would happen without uploading.

       Returns:
           True if publish succeeded, False if cancelled.
       """
       settings = get_settings()

       # Step 1: Compute hash and gather stats
       console.print("[bold]Step 1:[/bold] Analyzing database...")
       file_hash = compute_file_hash(db_path)
       stats = get_anatomic_stats(db_path)
       display_anatomic_stats(stats)

       console.print(f"\nFile hash: [cyan]{file_hash}[/cyan]")

       if not Confirm.ask("Proceed with upload?"):
           return False

       if dry_run:
           console.print("[yellow]Dry run - no changes made[/yellow]")
           return True

       # Step 2: Upload to S3
       console.print("\n[bold]Step 2:[/bold] Uploading to S3...")
       client = create_s3_client(settings)
       # ... upload logic ...

       # Step 3: Update manifest
       console.print("\n[bold]Step 3:[/bold] Updating manifest...")
       # ... manifest logic ...

       console.print("\n[bold green]✓ Publish complete![/bold green]")
       return True
   ```

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.5: Create FindingModel Build Module

**What:** Move database build logic from `findingmodel/index.py` write methods to oidm-maintenance.

**Steps:**

1. Read the current implementation:
   - `packages/findingmodel/src/findingmodel/index.py` (look for `setup()`, `ingest()`, `_populate_*` methods)

2. Create `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`:

   ```python
   """Build findingmodel database from source models."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.progress import Progress

   from findingmodel import FindingModel
   from oidm_maintenance.config import get_settings

   console = Console()


   def build_findingmodel_database(
       source_dir: Path,
       output_path: Path,
       generate_embeddings: bool = True,
   ) -> Path:
       """Build the findingmodel DuckDB database.

       Args:
           source_dir: Directory containing .fm.json files.
           output_path: Path for output DuckDB file.
           generate_embeddings: Whether to generate OpenAI embeddings.

       Returns:
           Path to the created database file.
       """
       # Implementation moved from findingmodel/index.py:
       # 1. Discover all .fm.json files in source_dir
       # 2. Load and validate each FindingModel
       # 3. Create DuckDB connection
       # 4. Create tables (finding_models, finding_model_json, etc.)
       # 5. Insert data with embeddings
       # 6. Create FTS index
       # 7. Create HNSW vector index
       # 8. Return output path
       ...
   ```

   **Copy implementation from `index.py`, adapting:**
   - Extract `setup()`, `_create_tables()`, `ingest()`, `_populate_*` methods
   - Use standalone functions instead of class methods
   - Import settings from `oidm_maintenance.config`
   - Add Rich progress bars

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.6: Create FindingModel Publish Module

**What:** Move publish logic from `findingmodel/db_publish.py` to oidm-maintenance.

**Steps:**

1. Read the current implementation:
   - `packages/findingmodel/src/findingmodel/db_publish.py`

2. Create `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/publish.py`:

   ```python
   """Publish findingmodel database to S3/Tigris."""
   from pathlib import Path

   import duckdb
   from rich.console import Console
   from rich.prompt import Confirm
   from rich.table import Table

   from oidm_maintenance.config import get_settings
   from oidm_maintenance.hashing import compute_file_hash
   from oidm_maintenance.s3 import (
       create_s3_client,
       load_manifest_from_s3,
       update_manifest_entry,
       backup_manifest,
       save_manifest_to_s3,
       upload_file_to_s3,
   )

   console = Console()


   def get_findingmodel_stats(db_path: Path) -> dict:
       """Get statistics about a findingmodel database.

       Returns dict with: model_count, sample_oifm_ids, json_roundtrip_ok, etc.
       """
       # Adapted from db_publish.py: get_database_stats(), run_sanity_check()
       ...


   def display_findingmodel_stats(stats: dict) -> None:
       """Display database statistics using Rich."""
       # Adapted from db_publish.py: prompt_user_confirmation()
       ...


   def publish_findingmodel_database(
       db_path: Path,
       version: str | None = None,
       dry_run: bool = False,
   ) -> bool:
       """Publish findingmodel database to S3.

       Args:
           db_path: Path to the DuckDB file to publish.
           version: Version string (default: YYYY-MM-DD).
           dry_run: If True, show what would happen without uploading.

       Returns:
           True if publish succeeded, False if cancelled.
       """
       # Similar structure to anatomic/publish.py
       # Uses findingmodel-specific stats and sanity checks
       ...
   ```

   **Copy implementation from `db_publish.py`, adapting:**
   - Move `DatabaseStats`, `SanityCheckResult` dataclasses here
   - Move `get_database_stats()`, `run_sanity_check()`, `prompt_user_confirmation()`
   - Use shared S3 functions from `oidm_maintenance.s3`
   - Use `MaintenanceSettings` instead of `FindingModelConfig`

**Verify:** Unit test can import the module without errors.

---

## Sub-phase 3.7.7: Create CLI Module

**What:** Create the unified CLI for all maintenance operations.

**Steps:**

1. Create `packages/oidm-maintenance/src/oidm_maintenance/cli.py`:

   ```python
   """CLI for OIDM maintenance operations."""
   from pathlib import Path

   import click
   from rich.console import Console

   console = Console()


   @click.group()
   @click.version_option()
   def main() -> None:
       """OIDM Maintenance Tools - Build and publish databases."""
       pass


   @main.group()
   def anatomic() -> None:
       """Anatomic-locations database operations."""
       pass


   @anatomic.command()
   @click.option("--source", "-s", type=click.Path(exists=True, path_type=Path), required=True,
                 help="Source CSV file with anatomic location data")
   @click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
                 help="Output DuckDB file path")
   @click.option("--no-embeddings", is_flag=True, help="Skip embedding generation")
   def build(source: Path, output: Path, no_embeddings: bool) -> None:
       """Build anatomic-locations database from source CSV."""
       from oidm_maintenance.anatomic.build import build_anatomic_database

       console.print(f"[bold]Building anatomic database...[/bold]")
       console.print(f"  Source: {source}")
       console.print(f"  Output: {output}")

       result = build_anatomic_database(
           source_csv=source,
           output_path=output,
           generate_embeddings=not no_embeddings,
       )
       console.print(f"\n[bold green]✓ Created:[/bold green] {result}")


   @anatomic.command()
   @click.argument("db_path", type=click.Path(exists=True, path_type=Path))
   @click.option("--version", "-v", help="Version string (default: YYYY-MM-DD)")
   @click.option("--dry-run", is_flag=True, help="Show what would happen without uploading")
   def publish(db_path: Path, version: str | None, dry_run: bool) -> None:
       """Publish anatomic-locations database to S3."""
       from oidm_maintenance.anatomic.publish import publish_anatomic_database

       success = publish_anatomic_database(db_path, version=version, dry_run=dry_run)
       if not success:
           raise SystemExit(1)


   @main.group()
   def findingmodel() -> None:
       """FindingModel database operations."""
       pass


   @findingmodel.command()
   @click.option("--source", "-s", type=click.Path(exists=True, path_type=Path), required=True,
                 help="Source directory containing .fm.json files")
   @click.option("--output", "-o", type=click.Path(path_type=Path), required=True,
                 help="Output DuckDB file path")
   @click.option("--no-embeddings", is_flag=True, help="Skip embedding generation")
   def build(source: Path, output: Path, no_embeddings: bool) -> None:
       """Build findingmodel database from source models."""
       from oidm_maintenance.findingmodel.build import build_findingmodel_database

       console.print(f"[bold]Building findingmodel database...[/bold]")
       console.print(f"  Source: {source}")
       console.print(f"  Output: {output}")

       result = build_findingmodel_database(
           source_dir=source,
           output_path=output,
           generate_embeddings=not no_embeddings,
       )
       console.print(f"\n[bold green]✓ Created:[/bold green] {result}")


   @findingmodel.command()
   @click.argument("db_path", type=click.Path(exists=True, path_type=Path))
   @click.option("--version", "-v", help="Version string (default: YYYY-MM-DD)")
   @click.option("--dry-run", is_flag=True, help="Show what would happen without uploading")
   def publish(db_path: Path, version: str | None, dry_run: bool) -> None:
       """Publish findingmodel database to S3."""
       from oidm_maintenance.findingmodel.publish import publish_findingmodel_database

       success = publish_findingmodel_database(db_path, version=version, dry_run=dry_run)
       if not success:
           raise SystemExit(1)


   if __name__ == "__main__":
       main()
   ```

**Verify:**
```bash
uv run --package oidm-maintenance oidm-maintain --help
uv run --package oidm-maintenance oidm-maintain anatomic --help
uv run --package oidm-maintenance oidm-maintain findingmodel --help
```

---

## Sub-phase 3.7.8: Strip Index Classes to Read-Only

**What:** Remove write methods from user-facing index classes.

**Steps:**

1. **Update `packages/anatomic-locations/src/anatomic_locations/index.py`:**

   Remove these methods (they now live in oidm-maintenance):
   - `setup()` or `create_tables()`
   - `ingest()` or `add_entry()`
   - Any method that writes to the database

   Keep only:
   - `__init__()` - opens connection in read-only mode
   - `search()` - hybrid FTS + vector search
   - `get()` or `get_by_id()` - retrieve by ID
   - `close()` or context manager methods

   Update `__init__` to enforce read-only:
   ```python
   def __init__(self, db_path: Path | None = None) -> None:
       if db_path is None:
           from anatomic_locations.config import ensure_anatomic_db
           db_path = ensure_anatomic_db()
       self.conn = duckdb.connect(str(db_path), read_only=True)  # Always read-only
   ```

2. **Update `packages/findingmodel/src/findingmodel/index.py`:**

   Remove these methods:
   - `setup()`
   - `ingest()`
   - `add_model()`
   - `_create_tables()`
   - `_populate_finding_models()`
   - `_populate_finding_model_json()`
   - `_populate_denormalized_tables()`
   - `_rebuild_indexes()`

   Keep only:
   - `__init__()` - opens connection in read-only mode
   - `search()` - search for finding models
   - `get_full()` - get complete model by ID
   - `validate()` - validate a model against the index
   - Context manager methods

   Update `__init__` to enforce read-only:
   ```python
   def __init__(self, db_path: Path | None = None, read_only: bool = True) -> None:
       # ... existing download logic ...
       self.conn = duckdb.connect(str(self.db_path), read_only=True)  # Always read-only
   ```

**Verify:**
```bash
uv run --package anatomic-locations pytest
uv run --package findingmodel pytest
```

---

## Sub-phase 3.7.9: Remove Old Code from Packages

**What:** Delete the build/publish code that has been moved to oidm-maintenance.

**Steps:**

1. **Delete `packages/findingmodel/src/findingmodel/db_publish.py`**

2. **Delete `packages/anatomic-locations/src/anatomic_locations/migration.py`** (if it exists as separate file)

3. **Update `packages/findingmodel/src/findingmodel/__init__.py`:**
   - Remove any exports of `db_publish` functions

4. **Update `packages/findingmodel/src/findingmodel/cli.py`:**
   - Remove any `publish` or `build` commands that referenced db_publish

5. **Update `packages/anatomic-locations/src/anatomic_locations/cli.py`:**
   - Remove `anatomic build` command
   - Keep only: `anatomic search`, `anatomic show`

6. **Update `packages/findingmodel/pyproject.toml`:**
   - Remove `boto3` from dependencies (if present)
   - Remove `boto3-stubs` from dependencies (if present)

7. **Update `packages/anatomic-locations/pyproject.toml`:**
   - Remove `openai` from dependencies (only needed for builds)
   - Remove `[build]` optional dependency group if it exists

**Verify:**
```bash
uv sync
uv run --package anatomic-locations pytest
uv run --package findingmodel pytest
# Ensure boto3 is NOT installed when only user packages are installed
```

---

## Sub-phase 3.7.10: Update Taskfile

**What:** Add convenient task targets for maintenance operations.

**Steps:**

1. Add to `Taskfile.yml`:

   ```yaml
   # Maintenance tasks
   maintain:anatomic:build:
     desc: "Build anatomic-locations database"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain anatomic build {{.CLI_ARGS}}

   maintain:anatomic:publish:
     desc: "Publish anatomic-locations database to S3"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain anatomic publish {{.CLI_ARGS}}

   maintain:findingmodel:build:
     desc: "Build findingmodel database"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain findingmodel build {{.CLI_ARGS}}

   maintain:findingmodel:publish:
     desc: "Publish findingmodel database to S3"
     cmds:
       - uv run --package oidm-maintenance oidm-maintain findingmodel publish {{.CLI_ARGS}}
   ```

**Verify:**
```bash
task maintain:anatomic:build --help
task maintain:findingmodel:build --help
```

---

## Sub-phase 3.7.11: Create Tests for oidm-maintenance

**What:** Add tests for the maintenance package.

**Steps:**

1. Create `packages/oidm-maintenance/tests/conftest.py`:
   ```python
   """Test fixtures for oidm-maintenance."""
   import pytest
   from pathlib import Path
   import tempfile


   @pytest.fixture
   def temp_db_path(tmp_path: Path) -> Path:
       """Temporary path for test databases."""
       return tmp_path / "test.duckdb"
   ```

2. Create `packages/oidm-maintenance/tests/test_hashing.py`:
   ```python
   """Tests for hashing module."""
   from pathlib import Path
   from oidm_maintenance.hashing import compute_file_hash


   def test_compute_file_hash(tmp_path: Path) -> None:
       """Test that file hashing works."""
       test_file = tmp_path / "test.txt"
       test_file.write_text("hello world")

       hash_result = compute_file_hash(test_file)

       assert hash_result.startswith("sha256:")
       assert len(hash_result) == 71  # "sha256:" + 64 hex chars
   ```

3. Create `packages/oidm-maintenance/tests/test_s3.py`:
   ```python
   """Tests for S3 module (mocked)."""
   from unittest.mock import MagicMock, patch

   from oidm_maintenance.s3 import create_s3_client
   from oidm_maintenance.config import MaintenanceSettings


   def test_create_s3_client_with_credentials() -> None:
       """Test S3 client creation with credentials."""
       settings = MaintenanceSettings(
           aws_access_key_id="test-key",
           aws_secret_access_key="test-secret",
       )

       with patch("boto3.client") as mock_client:
           create_s3_client(settings)

           mock_client.assert_called_once()
           call_kwargs = mock_client.call_args.kwargs
           assert call_kwargs["service_name"] == "s3"
           assert call_kwargs["endpoint_url"] == settings.s3_endpoint_url
   ```

4. Create `packages/oidm-maintenance/tests/test_anatomic_build.py`:
   ```python
   """Tests for anatomic build module."""
   import pytest
   from pathlib import Path
   from unittest.mock import patch


   def test_build_anatomic_database_creates_file(tmp_path: Path) -> None:
       """Test that build creates a database file."""
       # Create minimal test CSV
       source_csv = tmp_path / "source.csv"
       source_csv.write_text("id,description,region,sided\n1,Test Location,head,nonlateral\n")

       output_path = tmp_path / "anatomic.duckdb"

       with patch("oidm_maintenance.anatomic.build.generate_embeddings_batch", return_value=[[0.0] * 512]):
           from oidm_maintenance.anatomic.build import build_anatomic_database
           result = build_anatomic_database(source_csv, output_path, generate_embeddings=False)

       assert result.exists()
       assert result == output_path
   ```

**Verify:**
```bash
uv run --package oidm-maintenance pytest
```

---

## Sub-phase 3.7.12: Final Verification

**What:** Ensure everything works together.

**Verification checklist:**

1. **Package installation:**
   ```bash
   uv sync
   # All packages should install without errors
   ```

2. **User packages are read-only:**
   ```bash
   # These should work (read operations)
   uv run --package anatomic-locations python -c "from anatomic_locations import AnatomicLocationIndex; print('OK')"
   uv run --package findingmodel python -c "from findingmodel import DuckDBIndex; print('OK')"

   # Verify no write methods exist
   uv run --package anatomic-locations python -c "
   from anatomic_locations.index import AnatomicLocationIndex
   idx = AnatomicLocationIndex.__new__(AnatomicLocationIndex)
   assert not hasattr(idx, 'setup'), 'setup() should not exist'
   assert not hasattr(idx, 'ingest'), 'ingest() should not exist'
   print('OK - no write methods')
   "
   ```

3. **Maintenance CLI works:**
   ```bash
   uv run --package oidm-maintenance oidm-maintain --help
   uv run --package oidm-maintenance oidm-maintain anatomic --help
   uv run --package oidm-maintenance oidm-maintain anatomic build --help
   uv run --package oidm-maintenance oidm-maintain findingmodel --help
   uv run --package oidm-maintenance oidm-maintain findingmodel build --help
   ```

4. **All tests pass:**
   ```bash
   uv run --package oidm-common pytest
   uv run --package anatomic-locations pytest
   uv run --package findingmodel pytest
   uv run --package oidm-maintenance pytest
   ```

5. **Heavy dependencies isolated:**
   ```bash
   # boto3 should NOT be in user package dependencies
   grep -r "boto3" packages/findingmodel/pyproject.toml && echo "FAIL: boto3 in findingmodel" || echo "OK"
   grep -r "boto3" packages/anatomic-locations/pyproject.toml && echo "FAIL: boto3 in anatomic" || echo "OK"

   # boto3 SHOULD be in oidm-maintenance
   grep "boto3" packages/oidm-maintenance/pyproject.toml && echo "OK" || echo "FAIL: boto3 missing from oidm-maintenance"
   ```

6. **Taskfile works:**
   ```bash
   task maintain:anatomic:build -- --help
   task maintain:findingmodel:build -- --help
   ```

---

## Summary

| Sub-phase | Description | Key Files |
|-----------|-------------|-----------|
| 3.7.1 | Create package scaffolding | `pyproject.toml`, `__init__.py` |
| 3.7.2 | Create shared infrastructure | `config.py`, `s3.py`, `hashing.py` |
| 3.7.3 | Create anatomic build module | `anatomic/build.py` |
| 3.7.4 | Create anatomic publish module | `anatomic/publish.py` |
| 3.7.5 | Create findingmodel build module | `findingmodel/build.py` |
| 3.7.6 | Create findingmodel publish module | `findingmodel/publish.py` |
| 3.7.7 | Create CLI module | `cli.py` |
| 3.7.8 | Strip index classes to read-only | `index.py` in both packages |
| 3.7.9 | Remove old code from packages | Delete `db_publish.py`, `migration.py` |
| 3.7.10 | Update Taskfile | `Taskfile.yml` |
| 3.7.11 | Create tests | `tests/*.py` |
| 3.7.12 | Final verification | Run all checks |

**Total files to create:** ~15 new files in oidm-maintenance
**Total files to modify:** ~6 files in existing packages
**Total files to delete:** ~2 files

### Phase 4: Create findingmodel-ai

**Goal:** Extract AI tools to separate package

```bash
mkdir -p packages/findingmodel-ai/src/findingmodel_ai/tools
mkdir -p packages/findingmodel-ai/tests
mkdir -p packages/findingmodel-ai/evals
```

**Code to move:**

| From findingmodel | To findingmodel-ai |
|-------------------|-------------------|
| `tools/model_editor.py` | `tools/model_editor.py` |
| `tools/similar_finding_models.py` | `tools/similar_finding_models.py` |
| `tools/ontology_concept_match.py` | `tools/ontology_concept_match.py` |
| `tools/ontology_search.py` | `tools/ontology_search.py` |
| `tools/anatomic_location_search.py` | `tools/anatomic_location_search.py` |
| `tools/create_stub.py` | `tools/create_stub.py` |
| `tools/finding_description.py` | `tools/finding_description.py` |
| `tools/markdown_in.py` | `tools/markdown_in.py` |
| `tools/common.py` (AI parts) | `tools/common.py` |
| `tools/evaluators.py` | `tools/evaluators.py` |
| `tools/prompt_template.py` | `tools/prompt_template.py` |
| `evals/*` | `evals/*` |

**Verification:**
- `uv run --package findingmodel-ai pytest`
- All evals still run

### Phase 5: Clean up findingmodel

**Goal:** Remove extracted code, verify clean split

1. Remove tools/ directory (moved to findingmodel-ai)
2. Remove evals/ (moved to findingmodel-ai)
3. Update config.py to use oidm-common distribution
4. Remove AI-related dependencies from pyproject.toml
5. Run full test suite

**Verification:**
- `uv run --package findingmodel pytest`
- `pip install` from built wheel works without AI deps

### Phase 6: Documentation and AI Setup

**Goal:** Optimal Claude Code experience

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

For version pinning in wheels, we can add scripts or use [Una](https://github.com/carderne/una) if needed.

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
