# Plan: Per-Agent Model Configuration

## Context

The March 2026 agent performance audit (`docs/agent-performance-audit-2026-03.md`) revealed that the tier system (small/base/full) is the wrong abstraction for model selection. Different agents need fundamentally different model × reasoning pairings — budget models are fastest for simple tasks but slowest for complex medical judgment. The audit measured specific optimal configs per agent across 439 runs.

**Goal**: Each agent declares its preferred model + reasoning in code, with graceful fallback when a provider isn't available. A single resolution path determines the final `(model_string, reasoning_level)` pair for both runtime and metadata.

**Scope**: Per-agent defaults, per-agent reasoning overrides, and configurable fallback chains using pydantic-ai's `FallbackModel`. Each agent can declare an ordered list of model+reasoning pairs; at runtime these are wrapped in a `FallbackModel` that tries each in order on API errors, with automatic provider-availability filtering.

**Plan file**: This plan will be copied to `docs/plans/per-agent-model-config.md` at implementation start.

---

## Key Design Decision: Single Resolver

The review identified that multiple code paths (runtime, metadata, validation) must agree on which model+reasoning an agent will use. The fix is a **single resolver** that returns the fully-resolved `(model_string, reasoning_level)` tuple after all fallbacks:

```python
def resolve_agent_config(self, agent_tag: AgentTag, default_tier: ModelTier = "base") -> tuple[str, ReasoningLevel]:
    """Single source of truth for what model+reasoning an agent will use.

    Resolution order:
    1. Env overrides (AGENT_MODEL_OVERRIDES / AGENT_REASONING_OVERRIDES)
    2. Per-agent defaults from [agents] in supported_models.toml (if provider available)
    3. Tier-based defaults

    Returns (model_string, reasoning_level) — both normalized for the resolved model.
    """
```

All existing methods (`get_agent_model`, `get_effective_model_string`, `get_effective_reasoning_level`, and startup validation) delegate to this resolver. No separate code paths.

---

## Changes

### 1. Bump pydantic-ai dependency floor

**File**: `packages/findingmodel-ai/pyproject.toml`

The current floor `pydantic-ai-slim>=0.3.2` is dangerously loose — this plan depends on `FallbackModel`, per-model `settings=`, and `OpenAIResponsesModel`, all of which are post-1.0 features. Bump to:

```toml
"pydantic-ai-slim>=1.0.0",
```

Run `uv lock` after to update the lockfile.

### 2. Add `agent_reasoning_overrides` field to config

**File**: `config.py`

Replace the TODO comment at line 107 with the actual field:

```python
agent_reasoning_overrides: dict[AgentTag, ReasoningLevel] = Field(default_factory=dict)
```

Env: `AGENT_REASONING_OVERRIDES__anatomic_select=medium`

### 3. Add `[agents.*]` section to `supported_models.toml`

**File**: `data/supported_models.toml`

Each agent entry declares an ordered **fallback chain** of `(model, reasoning)` pairs. At runtime, unavailable providers are filtered out, and the remaining models are wrapped in pydantic-ai's `FallbackModel`. If only one model remains after filtering, it's used directly (no FallbackModel wrapper overhead).

Format:
- `models`: ordered list of `{model, reasoning}` dicts — first available is primary, rest are fallbacks
- `tier_fallback`: which tier to use if ALL models in the chain are unavailable (last resort)

```toml
# =============================================================================
# Per-agent defaults (from March 2026 audit)
# Ordered by: fastest/best for this task → next best → etc.
# =============================================================================

# --- Simple generative: gpt-5-nano fastest + most consistent ---

[agents.ontology_search]
tier_fallback = "small"
models = [
    { model = "openai:gpt-5-nano", reasoning = "low" },
    { model = "google-gla:gemini-3-flash-preview", reasoning = "low" },
    { model = "anthropic:claude-haiku-4-5", reasoning = "low" },
]

[agents.describe_finding]
tier_fallback = "small"
models = [
    { model = "openai:gpt-5-nano", reasoning = "low" },
    { model = "google-gla:gemini-3-flash-preview", reasoning = "low" },
    { model = "anthropic:claude-haiku-4-5", reasoning = "low" },
]

# --- Structured generation: gemini-flash/minimal fastest ---

[agents.anatomic_search]
tier_fallback = "small"
models = [
    { model = "google-gla:gemini-3-flash-preview", reasoning = "minimal" },
    { model = "openai:gpt-5-nano", reasoning = "low" },
    { model = "anthropic:claude-haiku-4-5", reasoning = "low" },
]

[agents.similar_term_gen]    # NEW tag (split from similar_search)
tier_fallback = "small"
models = [
    { model = "google-gla:gemini-3-flash-preview", reasoning = "minimal" },
    { model = "openai:gpt-5-nano", reasoning = "low" },
    { model = "anthropic:claude-haiku-4-5", reasoning = "low" },
]

# --- Medical classification: gemini-3.1-pro eliminates tail latency ---

[agents.ontology_match]
tier_fallback = "base"
models = [
    { model = "google-gla:gemini-3.1-pro-preview", reasoning = "low" },
    { model = "openai:gpt-5-mini", reasoning = "medium" },
    { model = "anthropic:claude-sonnet-4-6", reasoning = "low" },
]

[agents.anatomic_select]
tier_fallback = "base"
models = [
    { model = "google-gla:gemini-3.1-pro-preview", reasoning = "medium" },
    { model = "openai:gpt-5-mini", reasoning = "medium" },
    { model = "anthropic:claude-sonnet-4-6", reasoning = "low" },
]

# --- Simple classification: budget model fastest ---

[agents.similar_assess]
tier_fallback = "base"
models = [
    { model = "google-gla:gemini-3.1-flash-lite-preview", reasoning = "medium" },
    { model = "openai:gpt-5-mini", reasoning = "medium" },
    { model = "anthropic:claude-haiku-4-5", reasoning = "low" },
]

# --- Tool-using search: needs strategy planning ---

[agents.similar_search]
tier_fallback = "base"
models = [
    { model = "openai:gpt-5.4", reasoning = "low" },
    { model = "anthropic:claude-sonnet-4-6", reasoning = "low" },
    { model = "google-gla:gemini-3.1-pro-preview", reasoning = "low" },
]

# --- Complex structured output: quality over speed ---

[agents.edit_instructions]
tier_fallback = "base"
models = [
    { model = "openai:gpt-5.4", reasoning = "low" },
    { model = "anthropic:claude-opus-4-6", reasoning = "medium" },
    { model = "google-gla:gemini-3.1-pro-preview", reasoning = "low" },
]

[agents.edit_markdown]
tier_fallback = "base"
models = [
    { model = "openai:gpt-5.4", reasoning = "low" },
    { model = "anthropic:claude-opus-4-6", reasoning = "medium" },
    { model = "google-gla:gemini-3.1-pro-preview", reasoning = "low" },
]

[agents.import_markdown]
tier_fallback = "base"
models = [
    { model = "anthropic:claude-opus-4-6", reasoning = "medium" },
    { model = "openai:gpt-5.4", reasoning = "low" },
    { model = "google-gla:gemini-3.1-pro-preview", reasoning = "medium" },
]

# --- Tool-using detail search ---

[agents.describe_details]
tier_fallback = "small"
models = [
    { model = "google-gla:gemini-3-flash-preview", reasoning = "low" },
    { model = "openai:gpt-5-mini", reasoning = "low" },
    { model = "anthropic:claude-haiku-4-5", reasoning = "low" },
]

# --- Enrichment: not yet tuned (being overhauled separately) ---

[agents.enrich_classify]
tier_fallback = "base"
[agents.enrich_unified]
tier_fallback = "base"
[agents.enrich_research]
tier_fallback = "base"
```

### 4. Implement `resolve_agent_config()` — the single resolver

**File**: `config.py`

Returns a list of `(model_string, reasoning_level)` pairs — the available models in fallback order. Callers that need metadata use `[0]` (the primary). Callers that build models wrap the list in `FallbackModel`.

```python
@dataclass
class ResolvedModelConfig:
    """Resolved model + reasoning for a single model in a fallback chain."""
    model_string: str
    reasoning: ReasoningLevel

def resolve_agent_config(
    self, agent_tag: AgentTag, default_tier: ModelTier = "base"
) -> list[ResolvedModelConfig]:
    """Resolve the ordered list of (model, reasoning) for an agent.

    Single source of truth used by get_agent_model(), get_effective_model_string(),
    get_effective_reasoning_level(), and validate_agent_keys().

    Resolution:
    1. Env overrides → single model, no fallback chain
    2. Per-agent TOML models list → filter to available providers, keep order
    3. Tier default → single model fallback

    Returns at least one entry. First entry is the primary model.
    """
    agent_entry = _AGENT_REGISTRY.get(agent_tag, {})
    tier = agent_entry.get("tier_fallback", default_tier)

    # 1. Env override (highest priority — single model, no chain)
    if agent_tag in self.agent_model_overrides:
        model_string = self.agent_model_overrides[agent_tag]
        reasoning = self.agent_reasoning_overrides.get(agent_tag) or self._tier_reasoning(tier)
        reasoning = self._normalize_reasoning(model_string, reasoning)
        return [ResolvedModelConfig(model_string, reasoning)]

    # 2. Per-agent TOML chain — filter to available providers
    models_list = agent_entry.get("models", [])
    available: list[ResolvedModelConfig] = []
    for entry in models_list:
        model_string = entry["model"]
        if self._can_use_model(model_string):
            # Env reasoning override applies to ALL models in chain
            reasoning = self.agent_reasoning_overrides.get(agent_tag) or entry.get("reasoning") or self._tier_reasoning(tier)
            reasoning = self._normalize_reasoning(model_string, reasoning)
            available.append(ResolvedModelConfig(model_string, reasoning))

    if available:
        return available

    # 3. Tier fallback (all TOML models unavailable or no models list)
    if models_list:
        logfire.info("Agent {tag}: no preferred models available, using tier {tier} fallback",
                    tag=agent_tag, tier=tier)
    model_string = self._tier_model_string(tier)
    reasoning = self.agent_reasoning_overrides.get(agent_tag) or self._tier_reasoning(tier)
    reasoning = self._normalize_reasoning(model_string, reasoning)
    return [ResolvedModelConfig(model_string, reasoning)]
```

### 5. Rewrite `get_agent_model()` to build FallbackModel from chain

```python
def get_agent_model(self, agent_tag: AgentTag, *, default_tier: ModelTier = "base") -> Model:
    chain = self.resolve_agent_config(agent_tag, default_tier)

    # Build Model instances for each entry in the chain
    models = [self._create_model_from_resolved(c.model_string, c.reasoning) for c in chain]

    if len(models) == 1:
        return models[0]  # No FallbackModel overhead for single model

    from pydantic_ai.models.fallback import FallbackModel
    return FallbackModel(models[0], *models[1:])
```

Where `_create_model_from_resolved()` is a slimmed-down version of `_create_model_from_string()` that takes an already-resolved reasoning level instead of a tier. Each model in the chain carries its own provider-specific reasoning settings.

**SDK retry control for fallback chains**: When using FallbackModel, the non-last models must have SDK-level retries disabled to avoid 60s delays before fallback activates. This requires changes to the existing `_make_*_model()` factory methods:

Each factory (`_make_openai_model`, `_make_anthropic_model`, `_make_google_gla_model`, and their gateway variants) currently constructs a provider with default SDK settings. Add an optional `disable_retries: bool = False` parameter to each:

- **OpenAI** (`_make_openai_model`): Pass `max_retries=0` to `OpenAIProvider(api_key=..., openai_client=AsyncOpenAI(max_retries=0))` when `disable_retries=True`. The current code at config.py:469 constructs `OpenAIProvider(api_key=api_key)` — we need to optionally pass a custom client.
- **Anthropic** (`_make_anthropic_model`): Pass `max_retries=0` to `AnthropicProvider(api_key=..., anthropic_client=AsyncAnthropic(max_retries=0))`. Current code at config.py:480.
- **Google** (`_make_google_gla_model`): Google's SDK does not expose `max_retries` the same way. For v1, accept this as a known limitation — Google models in non-final chain positions may retry internally. Document this.
- **Gateway variants**: Same pattern — pass through to the underlying provider client.

Then in `get_agent_model()`, when building the chain:

```python
models = []
for i, c in enumerate(chain):
    is_last = (i == len(chain) - 1)
    models.append(self._create_model_from_resolved(
        c.model_string, c.reasoning, disable_retries=not is_last
    ))
```

### 6. Rewrite `get_effective_model_string()` to use the resolver

Returns the **primary** model (first in chain) — this is what metadata/logs report as the *configured* model.

**Known observability gap**: With FallbackModel active, the model that actually handles a request may differ from the primary if fallback activates at runtime. In v1, `get_effective_model_string()` reports the configured primary, not the actual model used. Pydantic AI's `ModelResponse.model_name` captures the actual model, but we don't currently surface that in our metadata paths (enrichment metadata in `unified.py:1038`, `agentic.py:289`). This means logs may show the intended model, not the one that ran.

This is acceptable for v1 because: (a) fallback only activates on API errors, which are rare in normal operation, and (b) the primary model reports correctly when no fallback occurs. But it should be addressed — see Deferred section.

```python
def get_effective_model_string(self, agent_tag: AgentTag, default_tier: ModelTier = "base") -> str:
    chain = self.resolve_agent_config(agent_tag, default_tier)
    return chain[0].model_string
```

### 7. Add `get_effective_reasoning_level()`

```python
def get_effective_reasoning_level(self, agent_tag: AgentTag, default_tier: ModelTier = "base") -> ReasoningLevel:
    chain = self.resolve_agent_config(agent_tag, default_tier)
    return chain[0].reasoning
```

### 8. Fix startup validation

**File**: `config.py` — `validate_default_model_keys()`

The current validation checks only the three tier defaults. With per-agent defaults, a command could work even if a tier default's provider is missing (because the agent has its own model configured to a different provider).

Change to **lazy validation at resolution time** rather than eager startup validation. The resolver already handles this via `_can_use_model()` — if the preferred model is unavailable, it falls back. The only startup validation needed is: "can we resolve at least one working model for each tier?" (so the fallback path works).

Update `validate_default_model_keys()`:
- Keep the tier-default validation as a warning, not an error
- Add: for each agent with a TOML default, check if either the preferred model OR the tier fallback is available
- Raise `ConfigurationError` only if an agent has NO viable path (neither preferred model nor tier fallback has a key)

### 9. Add `_can_use_model()` helper

```python
def _can_use_model(self, model_string: str) -> bool:
    """Check if we can use this model (API key or gateway available)."""
    key_field = self._get_required_key_field(model_string)
    if key_field is None:
        return True
    if self._has_key(key_field):
        return True
    provider_part = model_string.split(":")[0]
    gateway_eligible = {"openai", "anthropic", "google", "google-gla", "google-vertex"}
    return self._has_key("pydantic_ai_gateway_api_key") and provider_part in gateway_eligible
```

### 10. Split `similar_search` tag → add `similar_term_gen`

**File**: `config.py` — add `"similar_term_gen"` to `AgentTag` Literal

**File**: `search/similar.py` — update `create_term_generation_agent()` to use `"similar_term_gen"` tag

**File**: `supported_models.toml` — already included above

### 11. Helper refactors

Extract these small helpers from `_create_model_from_string()` to support the resolver pattern:

```python
def _tier_model_string(self, tier: ModelTier) -> str:
    return {"base": self.default_model, "full": self.default_model_full, "small": self.default_model_small}[tier]

def _tier_reasoning(self, tier: ModelTier) -> ReasoningLevel:
    return {"small": self.default_reasoning_small, "base": self.default_reasoning_base,
            "full": self.default_reasoning_full}[tier]

def _create_model_from_resolved(self, model_string: str, reasoning: ReasoningLevel) -> Model:
    """Create model from already-resolved model string + reasoning level."""
    # Same as _create_model_from_string but takes reasoning directly
    provider_part, model_name = model_string.split(":", 1)
    model_settings = self._build_reasoning_settings(provider_part, model_name, reasoning)
    # ... provider routing (same as current _create_model_from_string)
```

### 12. Tests

**File**: `tests/test_config.py`

New tests:
- `test_resolve_agent_config_returns_toml_chain()` — verify full chain from TOML with all providers
- `test_resolve_agent_config_filters_unavailable_providers()` — remove Google key → chain omits Google models
- `test_resolve_agent_config_all_unavailable_falls_to_tier()` — no provider keys → tier fallback
- `test_resolve_agent_config_env_overrides_chain()` — env override produces single-model result, no chain
- `test_resolve_agent_config_reasoning_env_override_applies_to_chain()` — `AGENT_REASONING_OVERRIDES__<tag>` overrides reasoning for ALL models in chain
- `test_resolve_agent_config_reasoning_normalized_per_model()` — each model in chain gets its own normalization
- `test_get_agent_model_builds_fallback_model()` — 2+ available models → FallbackModel instance
- `test_get_agent_model_single_model_no_wrapper()` — 1 available model → direct model, no FallbackModel
- `test_get_effective_model_string_returns_primary()` — metadata reports first model in chain
- `test_get_effective_reasoning_level_returns_primary()` — reasoning metadata matches primary
- `test_validate_keys_with_per_agent_defaults()` — validation accounts for TOML chains
- `test_similar_term_gen_tag_exists()` — new tag is usable
- `test_enrichment_agents_use_tier_fallback()` — agents without `models` list fall back to tier

### 13. Documentation

- Copy this plan to `docs/plans/per-agent-model-config.md`
- Update `docs/configuration.md` — document per-agent config system, env vars, TOML format
- Update `docs/agent-performance-audit-2026-03.md` — note implementation status
- Update `CHANGELOG.md`

---

## Files to Modify

| File | Change |
|------|--------|
| `packages/findingmodel-ai/pyproject.toml` | Bump `pydantic-ai-slim>=0.3.2` → `>=1.0.0` |
| `config.py` | `agent_reasoning_overrides` field, `ResolvedModelConfig` dataclass, `resolve_agent_config()`, rewrite `get_agent_model()` (with FallbackModel), `get_effective_model_string()`, `get_effective_reasoning_level()`, `_can_use_model()`, `_tier_model_string()`, `_tier_reasoning()`, `_create_model_from_resolved()` (with `disable_retries`), update `validate_default_model_keys()`, update `_make_openai_model()` / `_make_anthropic_model()` to accept `disable_retries` param |
| `data/supported_models.toml` | Add `[agents.*]` section with per-agent fallback chains |
| `search/similar.py` | Use `"similar_term_gen"` tag in `create_term_generation_agent()` |
| `tests/test_config.py` | Add ~13 new tests for resolver, chains, FallbackModel construction, provider filtering, metadata consistency |
| `docs/configuration.md` | Document per-agent config, fallback chains, env overrides |
| `docs/plans/per-agent-model-config.md` | Copy of this plan |
| `CHANGELOG.md` | Document the change |

## Deferred to v2

- **Per-agent model overrides with inline fallback chains via env vars** — currently env override is a single model. Could extend to `AGENT_MODEL_OVERRIDES__anatomic_select=google-gla:gemini-3.1-pro-preview,openai:gpt-5-mini` but this adds env var parsing complexity for limited benefit.
- **Runtime model reporting** — when FallbackModel activates a non-primary model, surface which model actually handled the request. Pydantic AI's `ModelResponse.model_name` captures the actual model name. To close this gap, the enrichment metadata paths (`unified.py:1038`, `agentic.py:289`) would need to read `result.model_name` from the agent run result instead of `get_effective_model_string()`. This is a real observability hole introduced by fallback chains but acceptable for v1 since fallback only fires on API errors.
- **Google SDK retry control** — Google's GenAI SDK doesn't expose `max_retries` like OpenAI/Anthropic. Google models in non-final chain positions may retry internally before fallback. Acceptable for v1 since Google models are typically first in their chains (they're the preferred model for classification), so this only matters if Google is a fallback for another provider.

## Verification

1. `task check` — lint + type check passes
2. `task test` — existing tests pass (backward compat)
3. New resolver tests pass:
   - Per-agent TOML defaults resolve correctly
   - Env overrides take priority over TOML
   - Provider unavailability filters chain correctly (remove Google → still get OpenAI + Anthropic)
   - All providers unavailable → tier fallback
   - `get_effective_model_string()` / `get_effective_reasoning_level()` match primary model in resolved chain
   - FallbackModel is constructed when chain has 2+ available models
   - Single available model → no FallbackModel wrapper
   - `AGENT_REASONING_OVERRIDES` applies to all models in chain
4. Manual: remove `GOOGLE_API_KEY`, verify agents with Google-primary chains still work via OpenAI/Anthropic fallbacks
5. Manual: set `AGENT_REASONING_OVERRIDES__anatomic_select=high`, verify it takes effect
6. Run a subset of the audit script to verify latency with new defaults
