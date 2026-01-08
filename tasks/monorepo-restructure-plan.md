# Plan: Monorepo Restructure with uv Workspaces

**Status:** Planning
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
│   ├── anatomic-locations/             # Anatomic ontology package
│   │   ├── pyproject.toml
│   │   ├── CLAUDE.md
│   │   ├── src/anatomic_locations/
│   │   │   ├── __init__.py
│   │   │   ├── models/                 # AnatomicLocation, enums
│   │   │   ├── index.py                # AnatomicLocationIndex
│   │   │   ├── migration.py            # Database build
│   │   │   └── cli.py                  # CLI commands
│   │   └── tests/
│   │
│   ├── findingmodel/                   # Core package (models, index, MCP)
│   │   ├── pyproject.toml
│   │   ├── CLAUDE.md
│   │   ├── src/findingmodel/
│   │   │   ├── __init__.py
│   │   │   ├── finding_model.py        # FindingModel
│   │   │   ├── finding_info.py         # FindingInfo
│   │   │   ├── abstract_finding_model.py
│   │   │   ├── contributor.py
│   │   │   ├── index.py                # FindingModelIndex
│   │   │   ├── index_validation.py
│   │   │   ├── config.py               # Uses oidm-common distribution
│   │   │   ├── cli.py                  # Basic CLI (search, show, validate)
│   │   │   └── mcp_server.py           # MCP server for IDE access
│   │   └── tests/
│   │
│   └── findingmodel-ai/                # AI authoring tools
│       ├── pyproject.toml
│       ├── CLAUDE.md
│       ├── src/findingmodel_ai/
│       │   ├── __init__.py
│       │   ├── tools/                  # AI agents and workflows
│       │   │   ├── model_editor.py
│       │   │   ├── similar_finding_models.py
│       │   │   ├── ontology_concept_match.py
│       │   │   ├── ontology_search.py
│       │   │   ├── anatomic_location_search.py
│       │   │   ├── create_stub.py
│       │   │   ├── finding_description.py
│       │   │   ├── markdown_in.py
│       │   │   └── common.py           # get_model, tavily client
│       │   └── cli.py                  # AI-specific CLI commands
│       ├── tests/
│       └── evals/                      # Agent evaluation suites
│
├── docs/                               # Shared documentation
├── tasks/                              # Planning documents
└── notebooks/                          # Demos and experiments
```

## Package Dependencies

```
┌──────────────────┐
│  findingmodel-ai │  ← AI authoring, evals
│   (AI tools)     │
└────────┬─────────┘
         │ depends on
         ▼
┌─────────────────┐
│  findingmodel   │  ← Models, index, MCP server
│ (core package)  │
└────────┬────────┘
         │ depends on
┌────────┴────────┐
▼                 ▼
┌─────────────────────┐  ┌─────────────────┐
│ anatomic-locations  │  │   oidm-common   │
│  (anatomic ontology)│  │ (infrastructure)│
└─────────┬───────────┘  └─────────────────┘
          │ depends on           ▲
          └──────────────────────┘
```

**Dependency rules:**
- oidm-common has no internal dependencies
- anatomic-locations depends on oidm-common
- findingmodel depends on oidm-common
- findingmodel-ai depends on findingmodel (and transitively on oidm-common)

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
requires = ["uv_build>=0.7.19,<0.8.0"]
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
requires = ["uv_build>=0.7.19,<0.8.0"]
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
requires = ["uv_build>=0.7.19,<0.8.0"]
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
requires = ["uv_build>=0.7.19,<0.8.0"]
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

### Cross-Package Tests

Some tests may need to verify integration between packages. Options:

1. **Keep in dependent package**: A test in `findingmodel-ai/tests/` that uses `anatomic-locations` is fine since it has that dependency
2. **Integration test directory** (if needed later): `tests/integration/` at workspace root

### Test Data

Test data lives with the package that owns it:

| Package | Test Data |
|---------|-----------|
| anatomic-locations | `tests/data/anatomic_*.json` |
| findingmodel | `tests/data/finding_models/`, `tests/data/embeddings/` |
| findingmodel-ai | Inherits from findingmodel via dependency |

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

### Phase 0: Preparation

**Goal:** Clean starting point

1. Commit/stash current anatomic-locations branch work
2. Create this plan document
3. Ensure main branch is stable

### Phase 1: Restructure findingmodel

**Goal:** Move existing code into packages/ without extraction

```bash
# Create structure
mkdir -p packages/findingmodel/src
mkdir -p packages/findingmodel/tests

# Move source code
git mv src/findingmodel packages/findingmodel/src/

# Move tests (findingmodel-specific ones)
git mv test/test_finding_*.py packages/findingmodel/tests/
git mv test/test_index*.py packages/findingmodel/tests/
git mv test/test_config.py packages/findingmodel/tests/
# ... etc

# Create package pyproject.toml
# Update root pyproject.toml to workspace format
# Update imports in tests
```

**Verification:** `uv sync && uv run --package findingmodel pytest`

### Phase 2: Extract oidm-common

**Goal:** Create shared infrastructure package

```bash
mkdir -p packages/oidm-common/src/oidm_common/{duckdb,embeddings,distribution,models}
mkdir -p packages/oidm-common/tests
```

**Code to extract:**

| From findingmodel | To oidm-common |
|-------------------|----------------|
| `tools/duckdb_utils.py` (most functions) | `duckdb/` subpackage |
| `embedding_cache.py` | `embeddings/cache.py` |
| `config.py` (distribution functions) | `distribution/` subpackage |
| `index_code.py` | `models/index_code.py` |
| `web_reference.py` | `models/web_reference.py` |

**Verification:**
- `uv run --package oidm-common pytest`
- `uv run --package findingmodel pytest` (with updated imports)

### Phase 3: Create anatomic-locations

**Goal:** Move anatomic code to standalone package

```bash
mkdir -p packages/anatomic-locations/src/anatomic_locations/models
mkdir -p packages/anatomic-locations/tests
```

**Code to move:**

| From findingmodel | To anatomic-locations |
|-------------------|----------------------|
| `anatomic_location.py` | `models/location.py` + `models/enums.py` |
| `anatomic_index.py` | `index.py` |
| `anatomic_migration.py` | `migration.py` |
| `cli.py` (anatomic commands) | `cli.py` |

**Tests to move:**

| From | To |
|------|-----|
| `test/test_anatomic_*.py` | `packages/anatomic-locations/tests/` |
| `test/data/anatomic_*.json` | `packages/anatomic-locations/tests/data/` |

**Verification:**
- `uv run --package anatomic-locations pytest`
- `anatomic --help` works

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
3. Update Serena memories for workspace structure
4. Update Taskfile.yml for per-package tasks
5. Update GitHub workflows

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
