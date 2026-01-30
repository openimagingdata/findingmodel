# Configuration Details

This document contains detailed configuration for the monorepo restructure.

## Root pyproject.toml

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

## packages/oidm-common/pyproject.toml

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

## packages/anatomic-locations/pyproject.toml

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

## packages/findingmodel/pyproject.toml

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

## packages/findingmodel-ai/pyproject.toml

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

## Environment Variable Strategy

**Principle:** Maintain compatibility with existing env var names. Use prefixes only for new package-specific settings.

### AI Configuration (findingmodel-ai) - No Prefix

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

### Infrastructure (oidm-common) - No Prefix for Standard Keys

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | For embeddings (shared with findingmodel-ai) |
| `AWS_ACCESS_KEY_ID` | S3 distribution access |
| `AWS_SECRET_ACCESS_KEY` | S3 distribution access |
| `LOGFIRE_TOKEN` | Observability |

### Package-Specific Settings (With Prefix)

| Package | Env Prefix | Examples |
|---------|-----------|----------|
| oidm-common | `OIDM_` | `OIDM_CACHE_DIR`, `OIDM_MANIFEST_URL` |
| anatomic-locations | `ANATOMIC_` | `ANATOMIC_DB_PATH` |
| findingmodel | `FINDINGMODEL_` | `FINDINGMODEL_DB_PATH`, `FINDINGMODEL_MCP_PORT` |

**Note:** findingmodel-ai does NOT use a prefix for AI settings - this maintains backward compatibility with the current implementation.

### Implementation Pattern

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

## Migration Notes

Current `src/findingmodel/config.py` is a monolith that must be carefully split. The table below shows exactly where each field goes:

### FindingModelConfig Field Migration

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

### Key Insight: AI Model Management is Entirely in findingmodel-ai

All `get_agent_model()` calls are in `tools/` modules that move to findingmodel-ai. The core findingmodel package does NOT use AI models - it only provides:
- Data models (FindingModel, FindingInfo)
- Index access (read-only search)
- MCP server (index queries, no AI)

This clean separation means:
- `pip install findingmodel` → No AI dependencies, no API keys needed
- `pip install findingmodel-ai` → Full AI tools, all provider support

## Test Topology

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
