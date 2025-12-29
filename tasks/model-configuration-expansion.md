# Model Configuration Architecture Expansion Plan

## Overview

Expand findingmodel's model configuration to:
1. Fix hard-coded models in tests
2. Support additional providers (Gemini, Ollama)
3. Enable per-agent model selection across all 15 agents

## Phase Summary

| Phase | Description | Dependency |
|-------|-------------|------------|
| 1 | Fix hard-coded models | None (foundation) |
| 2 | Add Gemini support | Phase 1 |
| 3 | Add Ollama support | Phase 1 |
| 4 | Per-agent configuration | Phases 2 & 3 |

---

## Phase 1: Fix Hard-Coded Models

**Goal**: Centralize test model strings, add type-safe ModelSpec, and update outdated defaults.

### 1.0 Add ModelSpec Type (`src/findingmodel/config.py`)

```python
from typing import Annotated
from pydantic import Field

# Pattern validates: provider:model or gateway/provider:model
# Update this pattern when adding new providers
MODEL_SPEC_PATTERN = r"^(openai|anthropic|gemini|ollama|gateway/(openai|anthropic|gemini)):[\w.:-]+$"

ModelSpec = Annotated[
    str,
    Field(
        pattern=MODEL_SPEC_PATTERN,
        description="Model spec: 'provider:model' (e.g., 'openai:gpt-5-mini', 'ollama:llama3.2:70b')",
    ),
]
```

Use `ModelSpec` for: `default_model`, `default_model_full`, `default_model_small`
Use `ModelSpec | None` for: `AgentModels` fields

### 1.1 Add Test Constants (`test/conftest.py`)

```python
# Cheapest models for test fixtures - minimize cost during testing
TEST_OPENAI_MODEL = "openai:gpt-5-nano"
TEST_ANTHROPIC_MODEL = "anthropic:claude-haiku-4-5"
```

### 1.2 Update Test Files

| File | Changes |
|------|---------|
| `test/test_finding_enrichment.py` | Replace 13+ `"openai:gpt-4-turbo"` with `TEST_OPENAI_MODEL` |
| `test/test_model_editor.py:140` | Delete hard-coded override, use default |
| `test/test_tools.py:905` | Use `TEST_OPENAI_MODEL` constant |

### 1.3 Update Defaults (`src/findingmodel/config.py`)

```python
default_model_full: str = Field(default="openai:gpt-5.2")  # Was gpt-5
```

### 1.4 Documentation

- Add model configuration guidance to CLAUDE.md section 3
- Update Serena `code_style_conventions` memory
- Create memory `model_configuration_architecture_2025`

### Files Modified
- `test/conftest.py`
- `test/test_finding_enrichment.py`
- `test/test_model_editor.py`
- `test/test_tools.py`
- `src/findingmodel/config.py`
- `CLAUDE.md`
- `.serena/memories/code_style_conventions.md`

---

## Phase 2: Add Gemini Support

**Goal**: Enable Google Gemini models (gemini-3-flash, gemini-3-pro).

**Why Gemini 3 Flash**: $0.50/$3 per 1M tokens, 78% SWE-bench (beats GPT-5's 74.9%), 3x faster.

### 2.1 Config Field (`src/findingmodel/config.py`)

```python
google_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
```

### 2.2 Factory Method

```python
def _make_gemini_model(self, model_name: str) -> Model:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    api_key = self.google_api_key.get_secret_value()
    if not api_key:
        raise ConfigurationError("GOOGLE_API_KEY not configured")
    return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key))
```

### 2.3 Routing in `get_model()`

```python
elif parts == ["gemini"]:
    return self._make_gemini_model(model_name)
```

### 2.4 Dependency (`pyproject.toml`)

```toml
"pydantic-ai-slim[openai,tavily,anthropic,google]>=0.3.2",
```

### 2.5 Usage

```bash
DEFAULT_MODEL=gemini:gemini-3-flash
DEFAULT_MODEL_FULL=gemini:gemini-3-pro
```

### Files Modified
- `src/findingmodel/config.py`
- `pyproject.toml`
- `.env.sample`

---

## Phase 3: Add Ollama Support

**Goal**: Enable local model execution for development and air-gapped production.

### 3.1 Config Field (`src/findingmodel/config.py`)

```python
ollama_base_url: str = Field(
    default="http://localhost:11434/v1",
    description="Base URL for Ollama API",
)
```

### 3.2 Factory Method

```python
def _make_ollama_model(self, model_name: str) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    return OpenAIChatModel(
        model_name,
        provider=OllamaProvider(base_url=self.ollama_base_url)
    )
```

### 3.3 Routing

```python
elif parts == ["ollama"]:
    return self._make_ollama_model(model_name)
```

### 3.4 Usage

```bash
DEFAULT_MODEL=ollama:llama3.2
OLLAMA_BASE_URL=http://gpu-server:11434/v1  # Remote Ollama
```

### Files Modified
- `src/findingmodel/config.py`
- `pyproject.toml` (if ollama extra needed)
- `.env.sample`

---

## Phase 4: Per-Agent Model Configuration

**Goal**: Fine-grained control over which model each agent uses.

### 4.1 Agent Inventory (15 agents)

| Config Field | File | Factory | Default Tier |
|--------------|------|---------|--------------|
| `enrichment_agent_model` | finding_enrichment.py | create_enrichment_agent | base |
| `enrichment_unified_model` | finding_enrichment.py | enrich_finding_unified | base |
| `agentic_enrichment_model` | finding_enrichment_agentic.py | create_agentic_enrichment_agent | base |
| `anatomic_query_model` | anatomic_location_search.py | generate_anatomic_query_terms | small |
| `location_selection_model` | anatomic_location_search.py | create_location_selection_agent | small |
| `search_agent_model` | similar_finding_models.py | create_search_agent | base |
| `term_generation_model` | similar_finding_models.py | create_term_generation_agent | small |
| `analysis_agent_model` | similar_finding_models.py | create_analysis_agent | base |
| `categorization_model` | ontology_concept_match.py | create_categorization_agent | base |
| `query_generator_model` | ontology_concept_match.py | create_query_generator_agent | small |
| `edit_agent_model` | model_editor.py | create_edit_agent | base |
| `markdown_edit_model` | model_editor.py | create_markdown_edit_agent | base |
| `finding_info_model` | finding_description.py | _create_finding_info_agent | varies |
| `finding_details_model` | finding_description.py | add_details_to_info | small |
| `markdown_in_model` | markdown_in.py | create_model_from_markdown | base |

### 4.2 Nested Config Structure (`src/findingmodel/config.py`)

Use Pydantic Settings nested model pattern with `env_nested_delimiter='__'`:

```python
from pydantic import BaseModel

class AgentModels(BaseModel):
    """Per-agent model overrides. None = use tier default."""
    enrichment: ModelSpec | None = None
    enrichment_unified: ModelSpec | None = None
    agentic_enrichment: ModelSpec | None = None
    anatomic_query: ModelSpec | None = None
    location_selection: ModelSpec | None = None
    search: ModelSpec | None = None
    term_generation: ModelSpec | None = None
    analysis: ModelSpec | None = None
    categorization: ModelSpec | None = None
    query_generator: ModelSpec | None = None
    edit: ModelSpec | None = None
    markdown_edit: ModelSpec | None = None
    finding_info: ModelSpec | None = None
    finding_details: ModelSpec | None = None
    markdown_in: ModelSpec | None = None

class FindingModelConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",  # Enables AGENT_MODELS__ENRICHMENT
    )

    # ... existing fields ...

    agent_models: AgentModels = Field(default_factory=AgentModels)
```

**Environment variable usage:**
```bash
AGENT_MODELS__ENRICHMENT=openai:gpt-5
AGENT_MODELS__EDIT=anthropic:claude-sonnet-4-5
AGENT_MODELS__ANATOMIC_QUERY=gemini:gemini-3-flash
```

### 4.3 New Method

```python
def get_model_for_agent(
    self,
    agent_name: str,
    default_tier: ModelTier = "base"
) -> Model:
    """Get model for a specific agent with fallback chain.

    Resolution order:
    1. Agent-specific config (e.g., AGENT_MODELS__ENRICHMENT env var)
    2. Tier-based default (base/small/full)

    Args:
        agent_name: Agent field name in AgentModels (e.g., "enrichment", "edit")
        default_tier: Tier to use if no agent-specific config
    """
    agent_model = getattr(self.agent_models, agent_name, None)
    if agent_model:
        return self._resolve_model_string(agent_model)
    return self.get_model(default_tier)
```

### 4.4 Update Agent Factories

Pattern for each factory:
```python
def create_enrichment_agent(
    model_tier: ModelTier = "base",
    model: str | None = None,
) -> Agent[...]:
    resolved_model = (
        model if model
        else settings.get_model_for_agent("enrichment_agent_model", model_tier)
    )
    return Agent(model=resolved_model, ...)
```

### Files Modified
- `src/findingmodel/config.py` - Add 15 fields + get_model_for_agent()
- `src/findingmodel/tools/finding_enrichment.py` - Update 2 factories
- `src/findingmodel/tools/finding_enrichment_agentic.py` - Update 1 factory
- `src/findingmodel/tools/anatomic_location_search.py` - Update 2 factories
- `src/findingmodel/tools/similar_finding_models.py` - Update 3 factories
- `src/findingmodel/tools/ontology_concept_match.py` - Update 2 factories
- `src/findingmodel/tools/model_editor.py` - Update 2 factories
- `src/findingmodel/tools/finding_description.py` - Update 2 factories
- `src/findingmodel/tools/markdown_in.py` - Update 1 factory
- `.env.sample` - Document per-agent vars

---

## Recommended Default Models (Post-Implementation)

After Phases 2-3, evaluate for production defaults:

| Tier | Current | Candidate | Notes |
|------|---------|-----------|-------|
| small | openai:gpt-5-nano ($0.05/$0.40) | Keep | Cheapest option |
| base | openai:gpt-5-mini ($0.25/$2) | gemini:gemini-3-flash ($0.50/$3) | 78% SWE-bench |
| full | openai:gpt-5.2 ($1.75/$14) | Keep or gemini:gemini-3-pro | Best reasoning |

---

## Testing Strategy

| Phase | Tests |
|-------|-------|
| 1 | Run full test suite after constant replacement |
| 2 | Unit test + `@pytest.mark.callout` for Gemini API |
| 3 | Unit test + local Ollama integration |
| 4 | Test fallback chain, test each agent factory |

---

## Execution Order

```
Phase 1 (Foundation)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              │
Phase 2        Phase 3           │
(Gemini)       (Ollama)          │
    │              │              │
    └──────────────┴──────────────┘
                   │
                   ▼
              Phase 4
         (Per-Agent Config)
```

Phases 2 and 3 can be done in parallel after Phase 1 completes.

---

## Sources

- [OpenAI Models](https://platform.openai.com/docs/models/gpt-5)
- [Pydantic AI Gemini Support](https://ai.pydantic.dev/models/google/)
- [Gemini 3 Flash](https://blog.google/products/gemini/gemini-3-flash/)
- [Pydantic AI Ollama Support](https://ai.pydantic.dev/models/openai/)
