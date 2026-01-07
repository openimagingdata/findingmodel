# Model Configuration Architecture Expansion Plan

## Overview

Expand findingmodel's model configuration to:
1. Fix hard-coded models in tests
2. Support additional providers (Gemini, Ollama)
3. Enable per-agent model selection across all 15 agents

## Phase Summary

| Phase | Description | Dependency | Status |
|-------|-------------|------------|--------|
| 1 | Fix hard-coded models | None (foundation) | ✅ Complete |
| 2 | Add Gemini support | Phase 1 | ✅ Complete |
| 3 | Add Ollama support | Phase 1 | ✅ Complete |
| 4 | Per-agent configuration (dict-based) | Phases 2 & 3 | ❌ Lost (reverted) |
| 4R | Phase 4 Remediation: Explicit tags | Phase 4 | ⏸️ Superseded |
| 4R2 | Full Re-Implementation | None | ✅ Complete |

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

**Goal**: Fine-grained control over which model each agent uses, without configuration explosion.

### Design Principles

Based on research into configuration best practices:

1. **Convention over Configuration**: Agents default to their tier; override only when needed
2. **No Configuration Explosion**: Single dict field instead of 15+ explicit fields
3. **Zero Registration**: Agent names derived from function names via introspection
4. **Type Safety**: Dict values validated against `ModelSpec` pattern

Sources:
- [Google SRE Configuration Design](https://sre.google/workbook/configuration-design/)
- [Pydantic Settings Nested Config](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

### 4.1 Config Changes (`src/findingmodel/config.py`)

**Enable nested delimiter** in SettingsConfigDict:

```python
model_config = SettingsConfigDict(
    env_file=".env",
    extra="ignore",
    env_nested_delimiter="__",  # Enables AGENT_MODEL_OVERRIDES__function_name
)
```

**Add single dict field** for all agent overrides:

```python
# Per-agent model overrides (empty by default = no config explosion)
# Keys are function names, values validated against ModelSpec pattern
agent_model_overrides: dict[str, ModelSpec] = Field(default_factory=dict)
```

### 4.2 New Method with Function Name Introspection

```python
import inspect

def get_agent_model(
    self,
    agent_name: str | None = None,
    *,
    default_tier: ModelTier = "base"
) -> Model:
    """Get model for an agent, with optional per-agent override.

    Resolution order:
    1. Check agent_model_overrides dict for agent_name
    2. Fall back to tier-based default (base/small/full)

    Args:
        agent_name: Explicit name, or None to auto-detect from caller's function name
        default_tier: Tier to use if no override configured

    Returns:
        Configured Model instance

    Example:
        # Auto-detect from calling function:
        def create_enrichment_agent(...):
            model = settings.get_agent_model(default_tier="base")
            # Looks up "create_enrichment_agent" in overrides

        # User overrides via environment:
        # AGENT_MODEL_OVERRIDES__create_enrichment_agent=anthropic:claude-opus-4-5
    """
    if agent_name is None:
        # Introspect caller's function name
        frame = inspect.currentframe()
        if frame and frame.f_back:
            agent_name = frame.f_back.f_code.co_name
        else:
            agent_name = "unknown"

    if agent_name in self.agent_model_overrides:
        model_string = self.agent_model_overrides[agent_name]
        # Reuse existing model creation logic
        return self._create_model_from_string(model_string)
    return self.get_model(default_tier)


def _create_model_from_string(self, model_string: str) -> Model:
    """Create a Model instance from a model spec string.

    Extracts the model creation logic from get_model() for reuse.
    """
    # Implementation: factor out the provider routing from get_model()
    ...
```

### 4.3 Update Agent Factories

**Pattern** - minimal change, auto-detects function name:

```python
def create_enrichment_agent(
    model_tier: ModelTier = "base",
    model: str | None = None,
) -> Agent[...]:
    # get_agent_model() auto-detects "create_enrichment_agent" as the key
    resolved_model = model if model else settings.get_agent_model(default_tier=model_tier)
    return Agent(model=resolved_model, ...)
```

### 4.4 Environment Variable Usage

```bash
# Override specific agents (function names as keys)
AGENT_MODEL_OVERRIDES__create_enrichment_agent=anthropic:claude-opus-4-5
AGENT_MODEL_OVERRIDES__create_location_selection_agent=google:gemini-3-flash-preview
AGENT_MODEL_OVERRIDES__create_edit_agent=ollama:llama3

# Everything else uses tier defaults (no config needed)
```

### 4.5 Agent Inventory (for reference)

| Function Name | File | Default Tier |
|---------------|------|--------------|
| `create_enrichment_agent` | finding_enrichment.py | base |
| `enrich_finding_unified` | finding_enrichment.py | base |
| `create_agentic_enrichment_agent` | finding_enrichment_agentic.py | base |
| `generate_anatomic_query_terms` | anatomic_location_search.py | small |
| `create_location_selection_agent` | anatomic_location_search.py | small |
| `create_search_agent` | similar_finding_models.py | base |
| `create_term_generation_agent` | similar_finding_models.py | small |
| `create_analysis_agent` | similar_finding_models.py | base |
| `create_categorization_agent` | ontology_concept_match.py | base |
| `create_query_generator_agent` | ontology_concept_match.py | small |
| `create_edit_agent` | model_editor.py | base |
| `create_markdown_edit_agent` | model_editor.py | base |
| `_create_finding_info_agent` | finding_description.py | varies |
| `add_details_to_info` | finding_description.py | small |
| `create_model_from_markdown` | markdown_in.py | base |

### 4.6 Testing Strategy

1. **Unit tests** for `get_agent_model()`:
   - Returns tier default when no override
   - Returns override when configured
   - Correctly introspects caller function name
   - Validates ModelSpec pattern on override values

2. **Integration tests**:
   - Set env var, verify agent uses overridden model
   - Verify fallback chain works correctly

### Files Modified
- `src/findingmodel/config.py`:
  - Add `env_nested_delimiter="__"` to model_config
  - Add `agent_model_overrides: dict[str, ModelSpec]` field
  - Add `get_agent_model()` method with introspection
  - Factor out `_create_model_from_string()` helper
- `test/test_tools.py` - Add TestAgentModelOverrides test class
- `.env.sample` - Document per-agent override pattern
- `docs/configuration.md` - Document per-agent configuration

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
| 4 | Unit tests for `get_agent_model()`: introspection, override lookup, tier fallback, ModelSpec validation |

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

## Phase 4 Remediation: Explicit Agent Tags

**Status**: Phase 4 infrastructure was implemented but has issues requiring remediation.

### Problem Statement

The initial Phase 4 implementation:
1. ✅ Added `agent_model_overrides: dict[str, ModelSpec]` field
2. ✅ Added `get_agent_model()` method
3. ❌ Used "magic" introspection to detect caller function names
4. ❌ Did NOT wire up any agents to actually use `get_agent_model()`
5. ❌ Documentation claims features work that don't

### Remediation Goals

1. Replace introspection magic with explicit, documented agent tags
2. Wire up ALL agent creation points to use configurable model selection
3. Fix misleading documentation
4. Update tests to match new approach

---

### 4R.1 Define Agent Tags as Constants (`src/findingmodel/config.py`)

Create a `Literal` type for valid agent tags. Tags should be user-friendly names, not function names:

```python
from typing import Literal

# Agent tags for per-agent model configuration
# These are user-facing identifiers used in environment variables
AgentTag = Literal[
    # Finding enrichment
    "enrichment",
    "enrichment_unified",
    "agentic_enrichment",
    # Anatomic location
    "anatomic_query",
    "location_selection",
    # Similar finding models
    "search",
    "term_generation",
    "analysis",
    # Ontology matching
    "categorization",
    "query_generator",
    # Model editing
    "edit",
    "markdown_edit",
    # Finding description
    "finding_info",
    "finding_details",
    # Markdown import
    "markdown_import",
]
```

**Environment variable format**: `AGENT_MODEL_OVERRIDES__<tag>=provider:model`

Example:
```bash
AGENT_MODEL_OVERRIDES__enrichment=anthropic:claude-opus-4-5
AGENT_MODEL_OVERRIDES__edit=ollama:llama3
```

### 4R.2 Refactor `get_agent_model()` (`src/findingmodel/config.py`)

Remove introspection, require explicit tag:

```python
def get_agent_model(self, agent_tag: AgentTag, *, default_tier: ModelTier = "base") -> Model:
    """Get model for a named agent, with optional per-agent override.

    Resolution order:
    1. Check agent_model_overrides dict for agent_tag
    2. Fall back to tier-based default (base/small/full)

    Args:
        agent_tag: Agent identifier from AgentTag (e.g., "enrichment", "edit")
        default_tier: Tier to use if no override configured

    Returns:
        Configured Model instance

    Example:
        model = settings.get_agent_model("enrichment", default_tier="base")

        # User overrides via environment:
        # AGENT_MODEL_OVERRIDES__enrichment=anthropic:claude-opus-4-5
    """
    if agent_tag in self.agent_model_overrides:
        model_string = self.agent_model_overrides[agent_tag]
        return self._create_model_from_string(model_string, default_tier)
    return self.get_model(default_tier)
```

**Changes from current implementation**:
- Remove `import inspect`
- Remove `agent_name: str | None = None` parameter
- Make `agent_tag: AgentTag` required (first positional parameter)
- Remove all introspection logic

### 4R.3 Wire Up All Agent Factories

Update each agent creation point to use `get_agent_model()`:

| File | Function | Tag | Default Tier |
|------|----------|-----|--------------|
| `finding_enrichment.py:581` | `create_enrichment_agent` | `"enrichment"` | base |
| `finding_enrichment.py:798` | `enrich_finding_unified` | `"enrichment_unified"` | base |
| `finding_enrichment_agentic.py:133` | `create_agentic_enrichment_agent` | `"agentic_enrichment"` | base |
| `anatomic_location_search.py:57` | `generate_anatomic_query_terms` | `"anatomic_query"` | small |
| `anatomic_location_search.py:168` | `create_location_selection_agent` | `"location_selection"` | small |
| `similar_finding_models.py:95` | `create_search_agent` | `"search"` | base |
| `similar_finding_models.py:131` | `create_term_generation_agent` | `"term_generation"` | small |
| `similar_finding_models.py:168` | `create_analysis_agent` | `"analysis"` | base |
| `ontology_concept_match.py:217` | `create_categorization_agent` | `"categorization"` | base |
| `ontology_concept_match.py:342` | `create_query_generator_agent` | `"query_generator"` | small |
| `model_editor.py:103` | `create_edit_agent` | `"edit"` | base |
| `model_editor.py:144` | `create_markdown_edit_agent` | `"markdown_edit"` | base |
| `finding_description.py:82` | `_create_finding_info_agent` | `"finding_info"` | varies |
| `finding_description.py:160` | `add_details_to_info` | `"finding_details"` | small |
| `markdown_in.py:43` | `create_model_from_markdown` | `"markdown_import"` | base |

**Pattern for each update**:

```python
# Before:
agent = Agent(...
    model=settings.get_model(model_tier),
    ...
)

# After:
agent = Agent(...
    model=model if model else settings.get_agent_model("enrichment", default_tier=model_tier),
    ...
)
```

For functions that already have a `model: str | None` parameter, preserve that override capability.

### 4R.4 Update Tests (`test/test_tools.py`)

**Remove** introspection-based tests:
- `test_introspects_caller_function_name` - No longer applicable

**Update** remaining tests:
- `test_returns_tier_default_when_no_override` - Use explicit tag
- `test_returns_override_when_configured` - Use explicit tag

**Add** new tests:
- `test_invalid_agent_tag_rejected` - Verify type checking catches invalid tags
- `test_all_agent_tags_resolve` - Verify each defined tag works

```python
def test_invalid_agent_tag_rejected(self) -> None:
    """Invalid agent tag should be caught by type checker (runtime behavior TBD)."""
    # This is primarily a type-checking test; mypy should catch invalid tags
    pass

def test_explicit_tag_required(self) -> None:
    """get_agent_model() requires an explicit tag parameter."""
    from findingmodel.config import FindingModelConfig
    config = FindingModelConfig(...)

    # Should work with explicit tag
    model = config.get_agent_model("enrichment", default_tier="base")
    assert model is not None

    # TypeError if tag omitted (enforced by signature)
```

### 4R.5 Fix Documentation

**`.env.sample`**: Fix incorrect function names:
```bash
# Before (wrong):
# AGENT_MODEL_OVERRIDES__create_finding_model=openai:gpt-5.2
# AGENT_MODEL_OVERRIDES__query_anatomic_location=anthropic:claude-sonnet-4-5

# After (correct):
# AGENT_MODEL_OVERRIDES__enrichment=openai:gpt-5.2
# AGENT_MODEL_OVERRIDES__edit=anthropic:claude-sonnet-4-5
```

**`docs/configuration.md`**: Update the Per-Agent Model Overrides section:
- Remove misleading "Available Agent Functions" table
- Replace with "Agent Tags" table showing tag → purpose mapping
- Update environment variable examples to use tags
- Remove programmatic example showing introspection

**Updated table for docs**:
```markdown
| Tag | Purpose | Default Tier |
|-----|---------|--------------|
| `enrichment` | Finding enrichment classification | base |
| `enrichment_unified` | Unified enrichment workflow | base |
| `agentic_enrichment` | Tool-using enrichment agent | base |
| `anatomic_query` | Anatomic location query generation | small |
| `location_selection` | Location search result selection | small |
| `search` | Similar model search strategy | base |
| `term_generation` | Search term generation | small |
| `analysis` | Similar model analysis | base |
| `categorization` | Ontology concept categorization | base |
| `query_generator` | Ontology search queries | small |
| `edit` | Model editing | base |
| `markdown_edit` | Markdown-based editing | base |
| `finding_info` | Finding info generation | varies |
| `finding_details` | Detail enrichment | small |
| `markdown_import` | Markdown to model conversion | base |
```

### 4R.6 Update Serena Memory

Update `code_style_conventions` memory to document:
- `AgentTag` type and usage
- Pattern for using `get_agent_model()` in agent factories

---

### Files Modified (Summary)

| File | Changes |
|------|---------|
| `src/findingmodel/config.py` | Add `AgentTag` type, refactor `get_agent_model()`, remove `inspect` |
| `src/findingmodel/tools/finding_enrichment.py` | Wire up 2 agents |
| `src/findingmodel/tools/finding_enrichment_agentic.py` | Wire up 1 agent |
| `src/findingmodel/tools/anatomic_location_search.py` | Wire up 2 agents |
| `src/findingmodel/tools/similar_finding_models.py` | Wire up 3 agents |
| `src/findingmodel/tools/ontology_concept_match.py` | Wire up 2 agents |
| `src/findingmodel/tools/model_editor.py` | Wire up 2 agents |
| `src/findingmodel/tools/finding_description.py` | Wire up 2 agents |
| `src/findingmodel/tools/markdown_in.py` | Wire up 1 agent |
| `test/test_tools.py` | Update/add tests for explicit tags |
| `.env.sample` | Fix example tag names |
| `docs/configuration.md` | Update with correct tag-based documentation |

---

### Execution Order

```
4R.1 Define AgentTag type
    │
    ▼
4R.2 Refactor get_agent_model()
    │
    ▼
4R.3 Wire up all agent factories (can parallelize by file)
    │
    ▼
4R.4 Update tests
    │
    ▼
4R.5 Fix documentation
    │
    ▼
4R.6 Update Serena memory
```

---

---

## Phase 4 Remediation v2: Full Re-Implementation

**Status**: config.py changes were lost during planning session. Need full re-implementation.

### What Was Lost

The following were implemented in config.py but reverted:
- `agent_model_overrides: dict[str, ModelSpec]` field
- `env_nested_delimiter="__"` in SettingsConfigDict
- `get_agent_model()` method with introspection
- `_create_model_from_string()` helper method

### What Still Exists

- Tests in `test/test_tools.py` (7 tests in `TestAgentModelOverrides` class)
- Documentation updates in `docs/configuration.md` and `.env.sample`

Note: The tests include `test_introspects_caller_function_name` which needs to be removed since we're dropping introspection.

---

### Revised Design Decisions

#### 1. Dict Type: `dict[AgentTag, ModelSpec]`

Use typed dict for both keys and values:
```python
agent_model_overrides: dict[AgentTag, ModelSpec] = Field(default_factory=dict)
```

This provides:
- Key validation: Only valid agent tags accepted
- Value validation: Only valid model specs (provider:model format)
- IDE autocomplete on both keys and values

#### 2. Informative Tag Names

Tags use `{domain}_{verb}` pattern for clarity. Each tag describes what the agent DOES.

| Agent Function | Tag | What It Does |
|----------------|-----|--------------|
| `create_enrichment_agent` | `enrich_classify` | Classifies finding attributes |
| `enrich_finding_unified` | `enrich_unified` | Unified enrichment workflow |
| `create_agentic_enrichment_agent` | `enrich_research` | Researches with tools before classifying |
| `generate_anatomic_query_terms` | `anatomic_search` | Searches for anatomic locations |
| `create_location_selection_agent` | `anatomic_select` | Selects from location candidates |
| `create_search_agent` | `similar_search` | Searches for similar findings |
| `create_term_generation_agent` | `similar_search` | (shares tag - part of search workflow) |
| `create_analysis_agent` | `similar_assess` | Assesses similarity results |
| `create_categorization_agent` | `ontology_match` | Matches to ontology concepts |
| `create_query_generator_agent` | `ontology_search` | Searches ontologies |
| `create_edit_agent` | `edit_instructions` | Edits from natural language instructions |
| `create_markdown_edit_agent` | `edit_markdown` | Edits from markdown input |
| `_create_finding_info_agent` | `describe_finding` | Describes a finding from its name |
| `add_details_to_info` | `describe_details` | Adds details/citations to description |
| `create_model_from_markdown` | `import_markdown` | Imports finding model from markdown |

**Domain prefixes**: `enrich_`, `anatomic_`, `similar_`, `ontology_`, `edit_`, `describe_`, `import_`

**14 tags for 15 agents** (similar_search covers two agents in the search workflow)

---

### 4R2.1 Define AgentTag and Add Field (`src/findingmodel/config.py`)

```python
# Agent tags for per-agent model configuration
# Pattern: {domain}_{verb} - describes what the agent DOES
AgentTag = Literal[
    # Enrichment domain
    "enrich_classify",      # Classifies finding attributes
    "enrich_unified",       # Unified enrichment workflow
    "enrich_research",      # Researches with tools before classifying
    # Anatomic location domain
    "anatomic_search",      # Searches for anatomic locations
    "anatomic_select",      # Selects from location candidates
    # Similar models domain
    "similar_search",       # Searches for similar findings (covers 2 agents)
    "similar_assess",       # Assesses similarity results
    # Ontology domain
    "ontology_match",       # Matches to ontology concepts
    "ontology_search",      # Searches ontologies
    # Editing domain
    "edit_instructions",    # Edits from natural language instructions
    "edit_markdown",        # Edits from markdown input
    # Description domain
    "describe_finding",     # Describes a finding from its name
    "describe_details",     # Adds details/citations to description
    # Import domain
    "import_markdown",      # Imports finding model from markdown
]

class FindingModelConfig(BaseSettings):
    # ... existing fields ...

    # Per-agent model overrides
    agent_model_overrides: dict[AgentTag, ModelSpec] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",  # Enables AGENT_MODEL_OVERRIDES__tag=value
    )
```

### 4R2.2 Re-implement Helper Methods (`src/findingmodel/config.py`)

Need to re-implement `_create_model_from_string()` (if lost) and refactor `get_agent_model()`:

```python
def get_agent_model(self, agent_tag: AgentTag, *, default_tier: ModelTier = "base") -> Model:
    """Get model for a named agent, with optional per-agent override.

    Args:
        agent_tag: Agent identifier (e.g., "enrich_classify", "edit_instructions")
        default_tier: Tier to use if no override configured

    Returns:
        Configured Model instance

    Example:
        model = settings.get_agent_model("enrich_classify", default_tier="base")

        # User overrides via environment:
        # AGENT_MODEL_OVERRIDES__edit_instructions=anthropic:claude-opus-4-5
    """
    if agent_tag in self.agent_model_overrides:
        model_string = self.agent_model_overrides[agent_tag]
        return self._create_model_from_string(model_string, default_tier)
    return self.get_model(default_tier)
```

### 4R2.3 Wire Up Agent Factories (with new tags)

| File | Function | Tag | Default Tier |
|------|----------|-----|--------------|
| `finding_enrichment.py:581` | `create_enrichment_agent` | `"enrich_classify"` | base |
| `finding_enrichment.py:798` | `enrich_finding_unified` | `"enrich_unified"` | base |
| `finding_enrichment_agentic.py:133` | `create_agentic_enrichment_agent` | `"enrich_research"` | base |
| `anatomic_location_search.py:57` | `generate_anatomic_query_terms` | `"anatomic_search"` | small |
| `anatomic_location_search.py:168` | `create_location_selection_agent` | `"anatomic_select"` | small |
| `similar_finding_models.py:95` | `create_search_agent` | `"similar_search"` | base |
| `similar_finding_models.py:131` | `create_term_generation_agent` | `"similar_search"` | small |
| `similar_finding_models.py:168` | `create_analysis_agent` | `"similar_assess"` | base |
| `ontology_concept_match.py:217` | `create_categorization_agent` | `"ontology_match"` | base |
| `ontology_concept_match.py:342` | `create_query_generator_agent` | `"ontology_search"` | small |
| `model_editor.py:103` | `create_edit_agent` | `"edit_instructions"` | base |
| `model_editor.py:144` | `create_markdown_edit_agent` | `"edit_markdown"` | base |
| `finding_description.py:82` | `_create_finding_info_agent` | `"describe_finding"` | varies |
| `finding_description.py:160` | `add_details_to_info` | `"describe_details"` | small |
| `markdown_in.py:43` | `create_model_from_markdown` | `"import_markdown"` | base |

### 4R2.4 Update Tests (`test/test_tools.py`)

**Remove**:
- `test_introspects_caller_function_name` (introspection removed)

**Update all tests to use new tag names**:
- Change `"test_agent"` → use actual tags from `AgentTag`
- Change `"nonexistent_agent"` → test with actual invalid tag handling

**Add**:
- Test that `dict[AgentTag, ModelSpec]` validation rejects invalid tags
- Test that env var format works: `AGENT_MODEL_OVERRIDES__enrich_classify=...`

### 4R2.5 Update Documentation

**`.env.sample`**:
```bash
# Per-agent model overrides (use domain_verb tag names)
# AGENT_MODEL_OVERRIDES__enrich_classify=anthropic:claude-opus-4-5
# AGENT_MODEL_OVERRIDES__edit_instructions=openai:gpt-5.2
# AGENT_MODEL_OVERRIDES__anatomic_select=ollama:llama3
```

**`docs/configuration.md`** - Replace tag table with new names:

| Tag | Purpose | Default Tier |
|-----|---------|--------------|
| `enrich_classify` | Classifies finding attributes | base |
| `enrich_unified` | Unified enrichment workflow | base |
| `enrich_research` | Researches with tools before classifying | base |
| `anatomic_search` | Searches for anatomic locations | small |
| `anatomic_select` | Selects from location candidates | small |
| `similar_search` | Searches for similar findings | base/small |
| `similar_assess` | Assesses similarity results | base |
| `ontology_match` | Matches to ontology concepts | base |
| `ontology_search` | Searches ontologies | small |
| `edit_instructions` | Edits from natural language instructions | base |
| `edit_markdown` | Edits from markdown input | base |
| `describe_finding` | Describes a finding from its name | varies |
| `describe_details` | Adds details/citations to description | small |
| `import_markdown` | Imports finding model from markdown | base |

---

## Sources

- [OpenAI Models](https://platform.openai.com/docs/models/gpt-5)
- [Pydantic AI Gemini Support](https://ai.pydantic.dev/models/google/)
- [Gemini 3 Flash](https://blog.google/products/gemini/gemini-3-flash/)
- [Pydantic AI Ollama Support](https://ai.pydantic.dev/models/openai/)
- [Google SRE Configuration Design](https://sre.google/workbook/configuration-design/)
- [Pydantic Settings Nested Config](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
