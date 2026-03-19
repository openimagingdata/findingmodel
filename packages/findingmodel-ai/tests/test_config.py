"""Tests for FindingModelAIConfig reasoning settings and model defaults."""

import pytest
from findingmodel.config import ConfigurationError
from findingmodel_ai.config import FindingModelAIConfig, settings

# ---------------------------------------------------------------------------
# _build_reasoning_settings — OpenAI
# (Provider settings classes are TypedDicts → plain dicts at runtime)
# ---------------------------------------------------------------------------


def test_openai_base_tier_gets_low_reasoning() -> None:
    """Base tier with gpt-5.2 produces low reasoning effort."""
    result = settings._build_reasoning_settings("openai", "gpt-5.2", "low")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "low"


def test_openai_medium_reasoning() -> None:
    """OpenAI with medium reasoning produces medium effort."""
    result = settings._build_reasoning_settings("openai", "gpt-5.4", "medium")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "medium"


def test_openai_small_tier_gpt5mini_gets_low_reasoning() -> None:
    """gpt-5-mini with low reasoning produces low effort."""
    result = settings._build_reasoning_settings("openai", "gpt-5-mini", "low")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "low"


def test_openai_unknown_model_passes_minimal_through() -> None:
    """Unknown OpenAI models pass 'minimal' through unchanged (no registry entry)."""
    result = settings._build_reasoning_settings("openai", "gpt-future", "minimal")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "minimal"


def test_openai_gpt54_minimal_normalized_to_low() -> None:
    """gpt-5.4 also rejects 'minimal'; it normalizes to 'low'."""
    result = settings._build_reasoning_settings("openai", "gpt-5.4", "minimal")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "low"


def test_openai_gpt54_xhigh_reasoning() -> None:
    """gpt-5.4 supports 'xhigh' reasoning effort."""
    result = settings._build_reasoning_settings("openai", "gpt-5.4", "xhigh")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "xhigh"


def test_openai_gpt5_mini_minimal_normalized_to_low() -> None:
    """gpt-5-mini also normalizes 'minimal' to 'low' (per supported_models.toml)."""
    result = settings._build_reasoning_settings("openai", "gpt-5-mini", "minimal")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "low"


def test_openai_none_reasoning_returns_no_settings() -> None:
    """'none' reasoning for OpenAI returns None (API default, no reasoning)."""
    result = settings._build_reasoning_settings("openai", "gpt-5.4", "none")
    assert result is None


# ---------------------------------------------------------------------------
# _build_reasoning_settings — Google
# ---------------------------------------------------------------------------


def test_google_flash_gets_correct_thinking_level() -> None:
    """gemini-3-flash-preview with low reasoning → thinking_level LOW."""
    result = settings._build_reasoning_settings("google-gla", "gemini-3-flash-preview", "low")
    assert isinstance(result, dict)
    config = result.get("google_thinking_config")
    assert config is not None
    assert config["thinking_level"] == "LOW"


def test_google_flash_none_normalized_to_minimal() -> None:
    """Flash with 'none' reasoning normalizes to MINIMAL (Flash requires thinking)."""
    result = settings._build_reasoning_settings("google-gla", "gemini-3-flash-preview", "none")
    assert isinstance(result, dict)
    config = result.get("google_thinking_config")
    assert config is not None
    assert config["thinking_level"] == "MINIMAL"


def test_google_flash_minimal_normalized_to_minimal() -> None:
    """Flash with 'minimal' reasoning → MINIMAL."""
    result = settings._build_reasoning_settings("google-gla", "gemini-3-flash-preview", "minimal")
    assert isinstance(result, dict)
    config = result.get("google_thinking_config")
    assert config is not None
    assert config["thinking_level"] == "MINIMAL"


def test_google_non_flash_none_maps_to_low() -> None:
    """Non-Flash Google models can't disable thinking; 'none' maps to LOW (minimum valid level)."""
    result = settings._build_reasoning_settings("google-gla", "gemini-3.1-pro-preview", "none")
    assert isinstance(result, dict)
    config = result.get("google_thinking_config")
    assert config is not None
    assert config["thinking_level"] == "LOW"


def test_google_non_flash_minimal_maps_to_low() -> None:
    """Non-Flash Google models don't support MINIMAL; 'minimal' maps to LOW."""
    result = settings._build_reasoning_settings("google-gla", "gemini-3.1-pro-preview", "minimal")
    assert isinstance(result, dict)
    config = result.get("google_thinking_config")
    assert config is not None
    assert config["thinking_level"] == "LOW"


def test_google_xhigh_maps_to_high() -> None:
    """'xhigh' has no Google equivalent; maps to HIGH."""
    result = settings._build_reasoning_settings("google-gla", "gemini-3-flash-preview", "xhigh")
    assert isinstance(result, dict)
    config = result.get("google_thinking_config")
    assert config is not None
    assert config["thinking_level"] == "HIGH"


def test_google_gateway_provider_handled() -> None:
    """gateway/google prefix is normalized to google for settings lookup."""
    result = settings._build_reasoning_settings("gateway/google", "gemini-3-flash-preview", "low")
    assert isinstance(result, dict)
    assert "google_thinking_config" in result


def test_google_vertex_gateway_provider_handled() -> None:
    """gateway/google-vertex prefix is normalized to google for settings lookup."""
    result = settings._build_reasoning_settings("gateway/google-vertex", "gemini-3-flash-preview", "low")
    assert isinstance(result, dict)
    assert "google_thinking_config" in result


def test_gateway_openai_gets_same_normalization() -> None:
    """gateway/openai:gpt-5.4 applies the same normalization as openai:gpt-5.4."""
    # gpt-5.4 normalizes minimal → low (per TOML)
    result = settings._build_reasoning_settings("gateway/openai", "gpt-5.4", "minimal")
    assert isinstance(result, dict)
    assert result.get("openai_reasoning_effort") == "low"


def test_gateway_google_gets_same_normalization() -> None:
    """gateway/google:gemini-3-flash-preview applies same normalization as google-gla variant."""
    # gemini-3-flash-preview normalizes none → minimal (per TOML)
    result = settings._build_reasoning_settings("gateway/google", "gemini-3-flash-preview", "none")
    assert isinstance(result, dict)
    config = result.get("google_thinking_config")
    assert config is not None
    assert config["thinking_level"] == "MINIMAL"


# ---------------------------------------------------------------------------
# _build_reasoning_settings — Anthropic (extended thinking, older models)
# ---------------------------------------------------------------------------


def test_anthropic_sonnet_reasoning_low_builds_thinking() -> None:
    """Sonnet + 'low' → enabled thinking with 1024 budget tokens."""
    result = settings._build_reasoning_settings("anthropic", "claude-sonnet-4-6", "low")
    assert isinstance(result, dict)
    thinking = result.get("anthropic_thinking")
    assert thinking is not None
    assert thinking["type"] == "enabled"
    assert thinking["budget_tokens"] == 1024
    assert result.get("max_tokens") == 8192


def test_anthropic_sonnet_reasoning_medium_builds_larger_budget() -> None:
    """Sonnet + 'medium' → 4096 budget tokens."""
    result = settings._build_reasoning_settings("anthropic", "claude-sonnet-4-6", "medium")
    assert isinstance(result, dict)
    thinking = result.get("anthropic_thinking")
    assert thinking is not None
    assert thinking["budget_tokens"] == 4096


def test_anthropic_reasoning_none_disables_thinking() -> None:
    """Anthropic + 'none' → no settings (natural model behavior, no thinking overhead)."""
    result = settings._build_reasoning_settings("anthropic", "claude-sonnet-4-6", "none")
    assert result is None


# ---------------------------------------------------------------------------
# _build_reasoning_settings — Anthropic (adaptive thinking, opus-4-6+)
# ---------------------------------------------------------------------------


def test_anthropic_opus46_uses_adaptive_thinking() -> None:
    """Opus 4.6 uses adaptive thinking (budget_tokens is deprecated)."""
    result = settings._build_reasoning_settings("anthropic", "claude-opus-4-6", "high")
    assert isinstance(result, dict)
    thinking = result.get("anthropic_thinking")
    assert thinking is not None
    assert thinking["type"] == "adaptive"
    assert result.get("anthropic_effort") == "high"


def test_anthropic_opus46_low_maps_to_effort_low() -> None:
    """Opus 4.6 with 'low' reasoning → adaptive thinking with effort 'low'."""
    result = settings._build_reasoning_settings("anthropic", "claude-opus-4-6", "low")
    assert isinstance(result, dict)
    assert result.get("anthropic_thinking") == {"type": "adaptive"}
    assert result.get("anthropic_effort") == "low"


def test_anthropic_opus46_xhigh_maps_to_effort_high() -> None:
    """Opus 4.6 with 'xhigh' reasoning → adaptive thinking with effort 'high'."""
    result = settings._build_reasoning_settings("anthropic", "claude-opus-4-6", "xhigh")
    assert isinstance(result, dict)
    assert result.get("anthropic_thinking") == {"type": "adaptive"}
    assert result.get("anthropic_effort") == "high"


def test_anthropic_opus46_none_disables_thinking() -> None:
    """Opus 4.6 with 'none' → no settings (natural model behavior)."""
    result = settings._build_reasoning_settings("anthropic", "claude-opus-4-6", "none")
    assert result is None


def test_anthropic_haiku_still_uses_budget_tokens() -> None:
    """Haiku (older model) still uses budget_tokens, not adaptive."""
    result = settings._build_reasoning_settings("anthropic", "claude-haiku-4-5", "medium")
    assert isinstance(result, dict)
    thinking = result.get("anthropic_thinking")
    assert thinking is not None
    assert thinking["type"] == "enabled"
    assert "budget_tokens" in thinking


# ---------------------------------------------------------------------------
# Ollama — no settings
# ---------------------------------------------------------------------------


def test_ollama_returns_no_settings() -> None:
    """Ollama models always return None (no reasoning support)."""
    result = settings._build_reasoning_settings("ollama", "llama3", "low")
    assert result is None


# ---------------------------------------------------------------------------
# Default model values
# ---------------------------------------------------------------------------


def test_default_model_small_is_gemini_flash() -> None:
    config = FindingModelAIConfig()
    assert config.default_model_small == "google-gla:gemini-3-flash-preview"


def test_default_model_base_is_gpt54_mini() -> None:
    config = FindingModelAIConfig()
    assert config.default_model == "openai:gpt-5.4-mini"


def test_default_model_full_is_gpt54() -> None:
    config = FindingModelAIConfig()
    assert config.default_model_full == "openai:gpt-5.4"


def test_default_reasoning_levels() -> None:
    config = FindingModelAIConfig()
    assert config.default_reasoning_small == "low"
    assert config.default_reasoning_base == "none"
    assert config.default_reasoning_full == "high"


# ---------------------------------------------------------------------------
# Per-tier reasoning env override
# ---------------------------------------------------------------------------


def test_per_tier_reasoning_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEFAULT_REASONING_BASE env var overrides the base reasoning level."""
    monkeypatch.setenv("DEFAULT_REASONING_BASE", "medium")
    config = FindingModelAIConfig()
    assert config.default_reasoning_base == "medium"


def test_per_tier_small_reasoning_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEFAULT_REASONING_SMALL=none disables small-tier reasoning."""
    monkeypatch.setenv("DEFAULT_REASONING_SMALL", "none")
    config = FindingModelAIConfig()
    assert config.default_reasoning_small == "none"


# ---------------------------------------------------------------------------
# Agent model override
# ---------------------------------------------------------------------------


def test_agent_model_override_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT_MODEL_OVERRIDES__anatomic_search sets the model string for that agent."""
    monkeypatch.setenv("AGENT_MODEL_OVERRIDES__anatomic_search", "openai:gpt-5-mini")
    config = FindingModelAIConfig()
    assert config.get_effective_model_string("anatomic_search") == "openai:gpt-5-mini"


# ---------------------------------------------------------------------------
# Missing API key errors — actionable messages
# ---------------------------------------------------------------------------


def test_google_missing_key_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing GOOGLE_API_KEY raises ConfigurationError with override hint."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    config = FindingModelAIConfig()
    with pytest.raises(ConfigurationError, match="DEFAULT_MODEL_SMALL"):
        config._make_google_gla_model("gemini-3-flash-preview")


def test_anthropic_missing_key_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ANTHROPIC_API_KEY raises ConfigurationError with override hint."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    config = FindingModelAIConfig()
    with pytest.raises(ConfigurationError, match="DEFAULT_MODEL"):
        config._make_anthropic_model("claude-sonnet-4-6")


def test_validate_default_model_keys_message_includes_remediation(monkeypatch: pytest.MonkeyPatch) -> None:
    """validate_default_model_keys() error includes override hint when key is missing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig()
    with pytest.raises(ConfigurationError, match="DEFAULT_MODEL_SMALL"):
        config.validate_default_model_keys()


# ---------------------------------------------------------------------------
# Gateway fallback — basic
# ---------------------------------------------------------------------------


def test_gateway_fallback_openai_when_no_direct_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI model falls back to gateway when OPENAI_API_KEY missing but gateway key set."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "test-gateway-key")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("openai:gpt-5.4")
    assert model is not None


def test_gateway_fallback_anthropic_when_no_direct_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic model falls back to gateway when ANTHROPIC_API_KEY missing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "test-gateway-key")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("anthropic:claude-sonnet-4-6")
    assert model is not None


def test_gateway_fallback_google_when_no_direct_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Google model falls back to gateway/vertex when GOOGLE_API_KEY missing."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "test-gateway-key")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("google-gla:gemini-3-flash-preview")
    assert model is not None


def test_no_fallback_when_both_keys_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ConfigurationError when both direct key and gateway key are missing."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig()
    with pytest.raises(ConfigurationError, match="PYDANTIC_AI_GATEWAY_API_KEY"):
        config._create_model_from_string("openai:gpt-5.4")


def test_direct_key_preferred_over_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both direct and gateway keys are set, direct key is used."""
    monkeypatch.setenv("OPENAI_API_KEY", "direct-key")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "gateway-key")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("openai:gpt-5.4")
    # OpenAIResponsesModel uses the direct provider, not gateway
    from pydantic_ai.providers.openai import OpenAIProvider

    assert isinstance(model._provider, OpenAIProvider)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Gateway fallback — Google prefix unification
# ---------------------------------------------------------------------------


def test_google_gla_prefix_with_only_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """google-gla: prefix works via gateway when only gateway key is set."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "test-gateway-key")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("google-gla:gemini-3-flash-preview")
    assert model is not None


def test_google_vertex_prefix_with_only_google_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """google-vertex: prefix routes to GLA when only GOOGLE_API_KEY is set."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("google-vertex:gemini-3-flash-preview")
    assert model is not None


def test_bare_google_prefix_with_google_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """google: prefix routes to GLA when GOOGLE_API_KEY is set."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("google:gemini-3-flash-preview")
    assert model is not None


def test_bare_google_prefix_with_only_gateway_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """google: prefix routes to gateway/vertex when only gateway key is set."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "test-gateway-key")
    config = FindingModelAIConfig()
    model = config._create_model_from_string("google:gemini-3-flash-preview")
    assert model is not None


def test_google_vertex_prefix_no_keys_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """google-vertex: with no keys raises ConfigurationError."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig()
    with pytest.raises(ConfigurationError, match="PYDANTIC_AI_GATEWAY_API_KEY"):
        config._create_model_from_string("google-vertex:gemini-3-flash-preview")


# ---------------------------------------------------------------------------
# Gateway fallback — validate_default_model_keys
# ---------------------------------------------------------------------------


def test_validate_default_model_keys_passes_with_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """validate_default_model_keys() passes when gateway key covers missing direct keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "test-gateway-key")
    config = FindingModelAIConfig()
    config.validate_default_model_keys()  # Should not raise


def test_validate_default_model_keys_fails_without_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """validate_default_model_keys() fails when no direct key AND no gateway key."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig()
    with pytest.raises(ConfigurationError, match="PYDANTIC_AI_GATEWAY_API_KEY"):
        config.validate_default_model_keys()


# ---------------------------------------------------------------------------
# resolve_agent_config — TOML chain resolution
# ---------------------------------------------------------------------------


def test_resolve_agent_config_returns_toml_chain() -> None:
    """With all 3 provider keys set, ontology_search returns 3 entries matching TOML order."""
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    chain = config.resolve_agent_config("ontology_search")
    assert len(chain) == 3
    assert chain[0].model_string == "openai:gpt-5.4-nano"
    assert chain[1].model_string == "google-gla:gemini-3-flash-preview"
    assert chain[2].model_string == "anthropic:claude-haiku-4-5"


def test_resolve_agent_config_filters_unavailable_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without Google key, ontology_match chain contains only OpenAI and Anthropic entries."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
    )
    chain = config.resolve_agent_config("ontology_match")
    model_strings = [c.model_string for c in chain]
    assert "google-gla:gemini-3.1-pro-preview" not in model_strings
    assert "openai:gpt-5.4-mini" in model_strings
    assert "anthropic:claude-sonnet-4-6" in model_strings
    assert len(chain) == 2


def test_resolve_agent_config_all_unavailable_falls_to_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no provider or gateway keys, resolve falls back to the tier default model."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig()
    # ontology_search has tier_fallback = "small" → default_model_small
    chain = config.resolve_agent_config("ontology_search")
    assert len(chain) == 1
    assert chain[0].model_string == config.default_model_small


def test_resolve_agent_config_env_override_single_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT_MODEL_OVERRIDES__anatomic_select returns a single-entry chain with the override model."""
    monkeypatch.setenv("AGENT_MODEL_OVERRIDES__anatomic_select", "openai:gpt-5.4")
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    chain = config.resolve_agent_config("anatomic_select")
    assert len(chain) == 1
    assert chain[0].model_string == "openai:gpt-5.4"


def test_resolve_agent_config_reasoning_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """AGENT_REASONING_OVERRIDES__ontology_search=high applies high reasoning to all chain entries."""
    monkeypatch.setenv("AGENT_REASONING_OVERRIDES__ontology_search", "high")
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    chain = config.resolve_agent_config("ontology_search")
    assert len(chain) == 3
    # Each entry should have "high" reasoning (after per-model normalization — "high" is valid for all three)
    for entry in chain:
        assert entry.reasoning == "high"


def test_resolve_agent_config_reasoning_per_model_normalization() -> None:
    """Reasoning levels are normalized per model: gpt-5.4-nano differs from gemini flash."""
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    # similar_plan uses reasoning = "minimal" for gemini-3-flash and "low" for gpt-5.4-nano
    # gpt-5.4-nano: minimal → low (per TOML normalize)
    # gemini-3-flash-preview: minimal → minimal (valid for flash, no normalization needed)
    chain = config.resolve_agent_config("similar_plan")
    assert len(chain) == 3
    nano_entry = next(c for c in chain if c.model_string == "openai:gpt-5.4-nano")
    flash_entry = next(c for c in chain if c.model_string == "google-gla:gemini-3-flash-preview")
    assert nano_entry.reasoning == "low"  # TOML says "low" for gpt-5.4-nano
    assert flash_entry.reasoning == "minimal"  # minimal stays minimal for gemini-flash


# ---------------------------------------------------------------------------
# get_agent_model — FallbackModel vs single model
# ---------------------------------------------------------------------------


def test_get_agent_model_builds_fallback_model() -> None:
    """With all 3 provider keys, get_agent_model returns a FallbackModel instance."""
    from pydantic_ai.models.fallback import FallbackModel

    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    model = config.get_agent_model("ontology_search")
    assert isinstance(model, FallbackModel)


def test_get_agent_model_single_available_no_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """With only OpenAI key, get_agent_model returns an OpenAIResponsesModel directly."""
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.openai import OpenAIResponsesModel

    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "")
    config = FindingModelAIConfig(openai_api_key="test-openai-key")
    model = config.get_agent_model("ontology_search")
    assert isinstance(model, OpenAIResponsesModel)
    assert not isinstance(model, FallbackModel)


# ---------------------------------------------------------------------------
# get_effective_model_string / get_effective_reasoning_level
# ---------------------------------------------------------------------------


def test_get_effective_model_string_returns_primary() -> None:
    """get_effective_model_string returns the first model in the chain."""
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    # ontology_search: first model is openai:gpt-5.4-nano
    result = config.get_effective_model_string("ontology_search")
    assert result == "openai:gpt-5.4-nano"


def test_get_effective_reasoning_level_returns_primary() -> None:
    """get_effective_reasoning_level returns the reasoning of the first model in the chain."""
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    # ontology_search: first model is openai:gpt-5.4-nano with reasoning = "low"
    result = config.get_effective_reasoning_level("ontology_search")
    assert result == "low"


# ---------------------------------------------------------------------------
# AgentTag validity — similar_plan
# ---------------------------------------------------------------------------


def test_similar_plan_tag_exists() -> None:
    """'similar_plan' is a valid AgentTag and can be resolved without error."""
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    chain = config.resolve_agent_config("similar_plan")
    assert len(chain) >= 1
    assert all(c.model_string for c in chain)


# ---------------------------------------------------------------------------
# Metadata assignment agent — has TOML models list
# ---------------------------------------------------------------------------


def test_metadata_assign_tag_resolves() -> None:
    """metadata_assign has a TOML models list and resolves to a chain."""
    config = FindingModelAIConfig(
        openai_api_key="test-openai-key",
        google_api_key="test-google-key",
        anthropic_api_key="test-anthropic-key",
    )
    chain = config.resolve_agent_config("metadata_assign")
    assert len(chain) >= 1
    assert all(c.model_string for c in chain)


# ---------------------------------------------------------------------------
# _can_use_model — gateway key counts as available
# ---------------------------------------------------------------------------


def test_can_use_model_with_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """_can_use_model returns True for an OpenAI model when only gateway key is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    config = FindingModelAIConfig(pydantic_ai_gateway_api_key="test-gateway-key")
    assert config._can_use_model("openai:gpt-5.4") is True
