import importlib.resources
import tomllib
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import httpx
import logfire
from findingmodel.config import ConfigurationError
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_supported_models() -> dict[str, Any]:
    """Load the supported models reference from package data."""
    ref = importlib.resources.files("findingmodel_ai") / "data" / "supported_models.toml"
    return tomllib.loads(ref.read_text(encoding="utf-8"))


# Module-level cache — loaded once at import time
_MODEL_REGISTRY: dict[str, Any] = _load_supported_models()
SUPPORTED_MODELS: dict[str, Any] = _MODEL_REGISTRY.get("models", {})
PROVIDER_CONFIGS: dict[str, Any] = _MODEL_REGISTRY.get("providers", {})
_AGENT_REGISTRY: dict[str, Any] = _MODEL_REGISTRY.get("agents", {})

# Type definitions for model configuration
ModelTier = Literal["base", "small", "full"]
ReasoningLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

# Agent tags for per-agent model configuration
# Pattern: {domain}_{verb} - describes what the agent DOES
AgentTag = Literal[
    # Enrichment domain
    "enrich_classify",  # Classifies finding attributes
    "enrich_unified",  # Unified enrichment workflow
    "enrich_research",  # Researches with tools before classifying
    # Anatomic location domain
    "anatomic_search",  # Searches for anatomic locations
    "anatomic_select",  # Selects from location candidates
    # Similar models domain
    "similar_term_gen",  # Generates search terms for similar finding discovery
    "similar_search",  # Searches for similar findings (tool-using search agent)
    "similar_assess",  # Assesses similarity results
    # Ontology domain
    "ontology_match",  # Matches to ontology concepts
    "ontology_search",  # Searches ontologies
    # Editing domain
    "edit_instructions",  # Edits from natural language instructions
    "edit_markdown",  # Edits from markdown input
    # Description domain
    "describe_finding",  # Describes a finding from its name
    "describe_details",  # Adds details/citations to description
    # Import domain
    "import_markdown",  # Imports finding model from markdown
]

# Pattern validates: provider:model or gateway/provider:model
# Direct providers:
#   - openai, anthropic, ollama: standard providers
#   - google, google-gla, google-vertex: Google Gemini models (routed to GLA or Vertex based on available keys)
# Gateway providers (via Pydantic AI Gateway):
#   - gateway/openai, gateway/anthropic: standard gateway routing
#   - gateway/google, gateway/google-vertex: Google Vertex AI via gateway
# Model names: alphanumeric, hyphens, dots, colons (for versions), underscores
MODEL_SPEC_PATTERN = (
    r"^(openai|anthropic|google|google-gla|google-vertex|ollama"
    r"|gateway/(openai|anthropic|google|google-vertex)):[\w.:-]+$"
)

ModelSpec = Annotated[
    str,
    Field(
        pattern=MODEL_SPEC_PATTERN,
        description="Model spec: 'provider:model' (e.g., 'openai:gpt-5.4', 'google-gla:gemini-3-flash-preview')",
    ),
]


@dataclass
class ResolvedModelConfig:
    """A resolved model + reasoning pair in a fallback chain."""

    model_string: str
    reasoning: ReasoningLevel


class FindingModelAIConfig(BaseSettings):
    # API Keys
    openai_api_key: SecretStr = Field(default=SecretStr(""))
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    google_api_key: SecretStr = Field(default=SecretStr(""))
    pydantic_ai_gateway_api_key: SecretStr = Field(default=SecretStr(""))
    pydantic_ai_gateway_base_url: str = Field(
        default="https://gateway.pydantic.dev/proxy/",
        description="Base URL for Pydantic AI Gateway (default: hosted gateway)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for Ollama API",
    )

    # Model configuration (Pydantic AI string format: "provider:model")
    # See MODEL_SPEC_PATTERN for supported providers
    default_model: ModelSpec = Field(default="openai:gpt-5.4")
    default_model_full: ModelSpec = Field(default="openai:gpt-5.4")
    default_model_small: ModelSpec = Field(default="google-gla:gemini-3-flash-preview")

    # Per-tier reasoning levels — overridable via env (e.g., DEFAULT_REASONING_SMALL=none)
    default_reasoning_small: ReasoningLevel = Field(default="low")
    default_reasoning_base: ReasoningLevel = Field(default="none")
    default_reasoning_full: ReasoningLevel = Field(default="high")

    # Per-agent model overrides
    agent_model_overrides: dict[AgentTag, ModelSpec] = Field(default_factory=dict)
    agent_reasoning_overrides: dict[AgentTag, ReasoningLevel] = Field(default_factory=dict)

    # Tavily API
    tavily_api_key: SecretStr = Field(default=SecretStr(""))
    tavily_search_depth: Literal["basic", "advanced"] = Field(
        default="advanced",
        description="Tavily search depth: 'basic' or 'advanced'",
    )

    # BioOntology API
    bioontology_api_key: SecretStr | None = Field(default=None, description="BioOntology.org API key")

    # Logfire observability
    logfire_token: SecretStr = Field(default=SecretStr(""), description="Logfire write token for cloud tracing")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    # Private instance cache for Ollama available models
    _ollama_models_cache: set[str] | None = PrivateAttr(default=None)

    def configure_logfire(self) -> None:
        """Configure Logfire observability using the token from project config.

        Idempotent — safe to call multiple times. Instruments pydantic-ai automatically.
        This is the ONLY correct way to set up Logfire in this project. Do not call
        logfire.configure() directly or rely on os.environ for the token.
        """
        token = self.logfire_token.get_secret_value() if self.logfire_token else None
        logfire.configure(
            send_to_logfire=bool(token),
            token=token or None,
            console=False,
            inspect_arguments=False,
        )
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx()

    def check_ready_for_tavily(self) -> Literal[True]:
        if not self.tavily_api_key.get_secret_value():
            raise ConfigurationError("Tavily API key is not set")
        return True

    def _has_key(self, field_name: str) -> bool:
        """Check if an API key field has a non-empty value."""
        value: SecretStr | None = getattr(self, field_name, None)
        return bool(value and value.get_secret_value())

    def _get_required_key_field(self, model_string: str) -> str | None:
        """Return the API key field name required for a model string.

        Args:
            model_string: Model specification (e.g., "openai:gpt-5-mini")

        Returns:
            Field name (e.g., "openai_api_key") or None if no key needed (Ollama)
        """
        if ":" not in model_string:
            return None

        provider_part = model_string.split(":")[0]

        # Gateway providers all use gateway key
        if provider_part.startswith("gateway/"):
            return "pydantic_ai_gateway_api_key"

        # Map direct providers to their API key fields
        return {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "google_api_key",
            "google-gla": "google_api_key",
            "google-vertex": "google_api_key",
            "ollama": None,  # No API key required
        }.get(provider_part)

    def _tier_model_string(self, tier: ModelTier) -> str:
        """Return the model string for a given tier."""
        return {"base": self.default_model, "full": self.default_model_full, "small": self.default_model_small}[tier]

    def _tier_reasoning(self, tier: ModelTier) -> ReasoningLevel:
        """Return the reasoning level for a given tier."""
        return {
            "small": self.default_reasoning_small,
            "base": self.default_reasoning_base,
            "full": self.default_reasoning_full,
        }[tier]

    def _can_use_model(self, model_string: str) -> bool:
        """Check if we can use this model (API key or gateway available)."""
        key_field = self._get_required_key_field(model_string)
        if key_field is None:
            return True  # Ollama, no key needed
        if self._has_key(key_field):
            return True
        provider_part = model_string.split(":")[0]
        gateway_eligible = {"openai", "anthropic", "google", "google-gla", "google-vertex"}
        return self._has_key("pydantic_ai_gateway_api_key") and provider_part in gateway_eligible

    def resolve_agent_config(self, agent_tag: AgentTag, default_tier: ModelTier = "base") -> list[ResolvedModelConfig]:
        """Resolve the ordered list of (model, reasoning) for an agent.

        Single source of truth used by get_agent_model(), get_effective_model_string(),
        and get_effective_reasoning_level().

        Resolution order:
        1. Env overrides (AGENT_MODEL_OVERRIDES / AGENT_REASONING_OVERRIDES) → single model, no chain
        2. Per-agent TOML models list → filter to available providers, keep order
        3. Tier default → single model fallback

        Returns at least one entry. First entry is the primary model.
        """
        agent_entry = _AGENT_REGISTRY.get(agent_tag, {})
        tier: ModelTier = agent_entry.get("tier_fallback", default_tier)

        # 1. Env override (highest priority — single model, no chain)
        if agent_tag in self.agent_model_overrides:
            model_string = self.agent_model_overrides[agent_tag]
            reasoning = self.agent_reasoning_overrides.get(agent_tag) or self._tier_reasoning(tier)
            reasoning = self._normalize_reasoning(model_string, reasoning)
            return [ResolvedModelConfig(model_string, reasoning)]

        # 2. Per-agent TOML chain — filter to available providers
        models_list: list[dict[str, str]] = agent_entry.get("models", [])
        available: list[ResolvedModelConfig] = []
        for entry in models_list:
            model_string = entry["model"]
            if self._can_use_model(model_string):
                # Env reasoning override applies to ALL models in chain
                toml_reasoning: ReasoningLevel | None = entry.get("reasoning")  # type: ignore[assignment]
                reasoning = (
                    self.agent_reasoning_overrides.get(agent_tag) or toml_reasoning or self._tier_reasoning(tier)
                )
                reasoning = self._normalize_reasoning(model_string, reasoning)
                available.append(ResolvedModelConfig(model_string, reasoning))

        if available:
            return available

        # 3. Tier fallback (all TOML models unavailable or no models list)
        if models_list:
            logfire.info(
                "Agent {tag}: no preferred models available, using tier {tier} fallback",
                tag=agent_tag,
                tier=tier,
            )
        model_string = self._tier_model_string(tier)
        reasoning = self.agent_reasoning_overrides.get(agent_tag) or self._tier_reasoning(tier)
        reasoning = self._normalize_reasoning(model_string, reasoning)
        return [ResolvedModelConfig(model_string, reasoning)]

    def validate_default_model_keys(self) -> None:
        """Validate that API keys are configured for all default models.

        Call this at application startup to fail fast if required API keys
        are missing for the configured default models.

        Raises:
            ConfigurationError: If any required API key is missing, with a
                message listing which keys need to be set.
        """
        models_to_check = [
            ("default_model", self.default_model),
            ("default_model_full", self.default_model_full),
            ("default_model_small", self.default_model_small),
        ]

        has_gateway = self._has_key("pydantic_ai_gateway_api_key")
        gateway_eligible_providers = {"openai", "anthropic", "google", "google-gla", "google-vertex"}

        missing: list[tuple[str, str, str]] = []
        for tier_name, model_string in models_to_check:
            key_field = self._get_required_key_field(model_string)
            if key_field and key_field != "pydantic_ai_gateway_api_key" and not self._has_key(key_field):
                # Check if gateway fallback covers this provider
                provider_part = model_string.split(":")[0]
                if has_gateway and provider_part in gateway_eligible_providers:
                    continue  # Gateway will handle this
                missing.append((tier_name, model_string, key_field))

        if missing:
            details = "; ".join(f"{tier} ({model}) requires {key.upper()}" for tier, model, key in missing)
            raise ConfigurationError(
                f"Missing API keys for default models: {details}. "
                "Set the missing key(s) in .env, set PYDANTIC_AI_GATEWAY_API_KEY for gateway fallback, "
                "or override the model tier (e.g., DEFAULT_MODEL_SMALL=openai:gpt-5-mini)."
            )

    @staticmethod
    def _canonical_model_spec(model_spec: str) -> str:
        """Map a model spec to its canonical form for TOML lookup.

        Gateway models use the same reasoning config as their direct counterparts:
          gateway/openai:gpt-5.4      → openai:gpt-5.4
          gateway/anthropic:claude-... → anthropic:claude-...
          gateway/google:gemini-...    → google-gla:gemini-...
          gateway/google-vertex:...    → google-gla:...
        """
        if ":" not in model_spec:
            return model_spec
        provider_part, model_name = model_spec.split(":", 1)
        if provider_part.startswith("gateway/"):
            backend = provider_part.split("/", 1)[1]
            # Gateway Google routes through Vertex, but same models as google-gla
            if backend in ("google", "google-vertex"):
                return f"google-gla:{model_name}"
            return f"{backend}:{model_name}"
        # google: and google-vertex: are aliases for google-gla: (canonical TOML form)
        if provider_part in ("google", "google-vertex"):
            return f"google-gla:{model_name}"
        return model_spec

    @staticmethod
    def _normalize_reasoning(model_spec: str, reasoning_level: ReasoningLevel) -> ReasoningLevel:
        """Normalize a reasoning level for a known model using the supported models registry.

        If the model has a normalization rule for this level (e.g., gpt-5.4 maps
        "minimal" → "low"), apply it. Gateway models resolve to their canonical
        direct-provider form for lookup. Unknown models pass through unchanged.
        """
        canonical = FindingModelAIConfig._canonical_model_spec(model_spec)
        model_entry = SUPPORTED_MODELS.get(canonical)
        if model_entry:
            normalize_map = model_entry.get("normalize", {})
            if reasoning_level in normalize_map:
                return normalize_map[reasoning_level]  # type: ignore[no-any-return]
        return reasoning_level

    def _build_reasoning_settings(
        self, provider_part: str, model_name: str, reasoning_level: ReasoningLevel
    ) -> ModelSettings | None:
        """Build provider-appropriate ModelSettings for the requested reasoning level.

        Consults supported_models.toml for normalization rules. For known models,
        reasoning levels are mapped to valid provider values. Unknown models get
        best-effort provider defaults.
        """
        # Normalize using the registry (e.g., "minimal" → "low" for gpt-5.4)
        model_spec = f"{provider_part}:{model_name}"
        level = self._normalize_reasoning(model_spec, reasoning_level)

        # Strip gateway/ prefix and normalize google aliases to canonical names
        provider = provider_part.split("/")[-1]
        if provider in ("google-gla", "google-vertex"):
            provider = "google"

        if provider == "openai":
            if level == "none":
                return None  # no settings = API default (no reasoning)
            from pydantic_ai.models.openai import OpenAIResponsesModelSettings

            return OpenAIResponsesModelSettings(openai_reasoning_effort=level)

        if provider == "google":
            from google.genai.types import ThinkingLevel
            from pydantic_ai.models.google import GoogleModelSettings

            thinking_level = ThinkingLevel(level.upper())
            return GoogleModelSettings(google_thinking_config={"thinking_level": thinking_level})

        if provider == "anthropic":
            from pydantic_ai.models.anthropic import AnthropicModelSettings

            if level == "none":
                return None  # No settings = natural model behavior (no thinking overhead)

            # Opus 4.6+ uses adaptive thinking (budget_tokens is deprecated)
            if model_name.startswith("claude-opus-4-6"):
                effort_map: dict[str, Literal["low", "medium", "high", "max"]] = {
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "high",
                }
                return AnthropicModelSettings(
                    anthropic_thinking={"type": "adaptive"},
                    anthropic_effort=effort_map.get(level, "medium"),
                )

            # Older models (sonnet-4-6, haiku-4-5, etc.): extended thinking with budget_tokens
            anthropic_config = PROVIDER_CONFIGS.get("anthropic", {})
            budget_map = anthropic_config.get("budget_map", {})
            if level in budget_map:
                entry = budget_map[level]
                budget_tokens = entry["budget_tokens"]
                max_tokens = entry["max_tokens"]
            else:
                budget_tokens, max_tokens = 4096, 16384
            return AnthropicModelSettings(
                anthropic_thinking={"type": "enabled", "budget_tokens": budget_tokens},
                max_tokens=max_tokens,
            )

        return None  # ollama and unknown providers: no settings

    def _resolve_with_fallback(
        self,
        model_name: str,
        model_settings: ModelSettings | None,
        *,
        direct_key: str,
        key_env_name: str,
        make_direct: str,
        make_gateway: str,
        disable_retries: bool = False,
    ) -> Model:
        """Resolve a model using direct key or gateway fallback.

        Args:
            model_name: Model name (e.g., "gpt-5.4")
            model_settings: Reasoning settings to pass through
            direct_key: Config field name for the direct API key (e.g., "openai_api_key")
            key_env_name: Human-readable env var name for error messages (e.g., "OPENAI_API_KEY")
            make_direct: Name of the _make_*_model method for direct access
            make_gateway: Name of the _make_gateway_*_model method for fallback
            disable_retries: If True, disable SDK-level retries (use for non-last models in FallbackModel chains)
        """
        if self._has_key(direct_key):
            result: Model = getattr(self, make_direct)(model_name, model_settings, disable_retries=disable_retries)
            return result
        if self._has_key("pydantic_ai_gateway_api_key"):
            logfire.info("Using gateway fallback for {model_name} (no {key})", model_name=model_name, key=key_env_name)
            result = getattr(self, make_gateway)(model_name, model_settings, disable_retries=disable_retries)
            return result
        raise ConfigurationError(
            f"{key_env_name} is not configured and no PYDANTIC_AI_GATEWAY_API_KEY for fallback. "
            "Set one of these in .env or your environment."
        )

    def _create_model_from_string(self, model_string: str, default_tier: ModelTier = "base") -> Model:
        """Create a Pydantic AI model instance from a model string.

        Routes to the best available provider based on configured API keys.
        For direct providers (openai, anthropic, google*), falls back to the
        Pydantic AI Gateway if the provider-specific key is missing but
        PYDANTIC_AI_GATEWAY_API_KEY is set.

        Args:
            model_string: Model specification (e.g., "openai:gpt-5.4", "anthropic:claude-sonnet-4-6")
            default_tier: Tier to use for per-tier reasoning settings resolution

        Returns:
            Configured Model instance

        Raises:
            ConfigurationError: If model string is invalid, API key missing, or provider unknown
        """
        if ":" not in model_string:
            raise ConfigurationError(f"Invalid model format '{model_string}'. Expected 'provider:model_name'")

        provider_part, model_name = model_string.split(":", 1)
        parts = provider_part.split("/")

        # Resolve reasoning level for this tier and build provider-appropriate settings
        reasoning_level: ReasoningLevel = {
            "small": self.default_reasoning_small,
            "base": self.default_reasoning_base,
            "full": self.default_reasoning_full,
        }[default_tier]
        model_settings = self._build_reasoning_settings(provider_part, model_name, reasoning_level)

        if parts == ["openai"]:
            return self._resolve_with_fallback(
                model_name,
                model_settings,
                direct_key="openai_api_key",
                key_env_name="OPENAI_API_KEY",
                make_direct="_make_openai_model",
                make_gateway="_make_gateway_openai_model",
            )
        elif parts == ["anthropic"]:
            return self._resolve_with_fallback(
                model_name,
                model_settings,
                direct_key="anthropic_api_key",
                key_env_name="ANTHROPIC_API_KEY",
                make_direct="_make_anthropic_model",
                make_gateway="_make_gateway_anthropic_model",
            )
        elif parts[0] in ("google", "google-gla", "google-vertex"):
            return self._resolve_with_fallback(
                model_name,
                model_settings,
                direct_key="google_api_key",
                key_env_name="GOOGLE_API_KEY",
                make_direct="_make_google_gla_model",
                make_gateway="_make_gateway_google_vertex_model",
            )
        elif parts == ["ollama"]:
            return self._make_ollama_model(model_name)
        elif parts == ["gateway", "openai"]:
            return self._make_gateway_openai_model(model_name, model_settings)
        elif parts == ["gateway", "anthropic"]:
            return self._make_gateway_anthropic_model(model_name, model_settings)
        elif parts in [["gateway", "google"], ["gateway", "google-vertex"]]:
            return self._make_gateway_google_vertex_model(model_name, model_settings)
        else:
            raise ConfigurationError(f"Unknown provider '{provider_part}'")

    def get_effective_model_string(self, agent_tag: AgentTag, default_tier: ModelTier = "base") -> str:
        """Return the primary model string for an agent (first in fallback chain).

        Useful for metadata/logging when you need the string, not the Model object.

        Note: With FallbackModel, the actual model used at runtime may differ if
        the primary fails. See ModelResponse.model_name for the actual model.

        Args:
            agent_tag: Agent identifier
            default_tier: Tier to use if no override configured

        Returns:
            Model specification string (e.g., "openai:gpt-5-mini")
        """
        chain = self.resolve_agent_config(agent_tag, default_tier)
        return chain[0].model_string

    def get_effective_reasoning_level(self, agent_tag: AgentTag, default_tier: ModelTier = "base") -> ReasoningLevel:
        """Return the reasoning level for an agent's primary model.

        Args:
            agent_tag: Agent identifier
            default_tier: Tier to use if no override configured

        Returns:
            Resolved reasoning level for the primary model in the chain
        """
        chain = self.resolve_agent_config(agent_tag, default_tier)
        return chain[0].reasoning

    def get_agent_model(self, agent_tag: AgentTag, *, default_tier: ModelTier = "base") -> Model:
        """Get model for a named agent, with fallback chain from TOML config.

        Resolution order:
        1. Env override (AGENT_MODEL_OVERRIDES) → single model
        2. Per-agent TOML defaults → FallbackModel with available providers
        3. Tier default → single model

        Returns a FallbackModel when multiple models are available in the chain,
        or a direct model when only one is available.

        Args:
            agent_tag: Agent identifier (e.g., "enrich_classify", "edit_instructions")
            default_tier: Tier to use if no TOML agent config is found

        Returns:
            Configured Model instance (direct or FallbackModel)

        Example:
            model = settings.get_agent_model("enrich_classify", default_tier="base")

            # User overrides via environment:
            # AGENT_MODEL_OVERRIDES__edit_instructions=anthropic:claude-opus-4-6
        """
        chain = self.resolve_agent_config(agent_tag, default_tier)

        # Build Model instances — disable SDK retries on non-last models for fast fallback
        models: list[Model] = []
        for i, c in enumerate(chain):
            is_last = i == len(chain) - 1
            models.append(self._create_model_from_resolved(c.model_string, c.reasoning, disable_retries=not is_last))

        if len(models) == 1:
            return models[0]

        from pydantic_ai.models.fallback import FallbackModel

        return FallbackModel(models[0], *models[1:])

    def _create_model_from_resolved(
        self, model_string: str, reasoning: ReasoningLevel, *, disable_retries: bool = False
    ) -> Model:
        """Create a Model from an already-resolved model string + reasoning level.

        Unlike _create_model_from_string(), this takes an already-resolved reasoning
        level directly (not a tier), and supports disable_retries for FallbackModel chains.

        Args:
            model_string: Model specification (e.g., "openai:gpt-5.4")
            reasoning: Already-resolved and normalized reasoning level
            disable_retries: If True, disable SDK-level retries (for non-last models in FallbackModel)

        Returns:
            Configured Model instance

        Raises:
            ConfigurationError: If model string is invalid or provider unknown
        """
        if ":" not in model_string:
            raise ConfigurationError(f"Invalid model format '{model_string}'. Expected 'provider:model_name'")

        provider_part, model_name = model_string.split(":", 1)
        parts = provider_part.split("/")
        model_settings = self._build_reasoning_settings(provider_part, model_name, reasoning)

        if parts == ["openai"]:
            return self._resolve_with_fallback(
                model_name,
                model_settings,
                direct_key="openai_api_key",
                key_env_name="OPENAI_API_KEY",
                make_direct="_make_openai_model",
                make_gateway="_make_gateway_openai_model",
                disable_retries=disable_retries,
            )
        elif parts == ["anthropic"]:
            return self._resolve_with_fallback(
                model_name,
                model_settings,
                direct_key="anthropic_api_key",
                key_env_name="ANTHROPIC_API_KEY",
                make_direct="_make_anthropic_model",
                make_gateway="_make_gateway_anthropic_model",
                disable_retries=disable_retries,
            )
        elif parts[0] in ("google", "google-gla", "google-vertex"):
            return self._resolve_with_fallback(
                model_name,
                model_settings,
                direct_key="google_api_key",
                key_env_name="GOOGLE_API_KEY",
                make_direct="_make_google_gla_model",
                make_gateway="_make_gateway_google_vertex_model",
                disable_retries=disable_retries,
            )
        elif parts == ["ollama"]:
            return self._make_ollama_model(model_name)
        elif parts == ["gateway", "openai"]:
            return self._make_gateway_openai_model(model_name, model_settings, disable_retries=disable_retries)
        elif parts == ["gateway", "anthropic"]:
            return self._make_gateway_anthropic_model(model_name, model_settings, disable_retries=disable_retries)
        elif parts in [["gateway", "google"], ["gateway", "google-vertex"]]:
            return self._make_gateway_google_vertex_model(model_name, model_settings)
        else:
            raise ConfigurationError(f"Unknown provider '{provider_part}'")

    def get_model(self, model_tier: ModelTier = "base") -> Model:
        """Get a Pydantic AI model instance for the requested tier.

        Args:
            model_tier: "base", "small", or "full"

        Returns:
            Configured Model instance

        Raises:
            ConfigurationError: If API key missing or provider unknown
        """
        model_string = {
            "base": self.default_model,
            "full": self.default_model_full,
            "small": self.default_model_small,
        }[model_tier]

        return self._create_model_from_string(model_string, model_tier)

    def _make_openai_model(
        self, model_name: str, settings: ModelSettings | None, *, disable_retries: bool = False
    ) -> OpenAIResponsesModel:
        """Create an OpenAI model with configured API key.

        Args:
            disable_retries: If True, create client with max_retries=0 (use for non-last models in FallbackModel chains)
        """
        from openai import AsyncOpenAI

        api_key = self.openai_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is not configured. "
                "Set it in .env, set PYDANTIC_AI_GATEWAY_API_KEY for gateway fallback, "
                "or override the model tier (e.g., DEFAULT_MODEL=google-gla:gemini-3-flash-preview)."
            )
        if disable_retries:
            openai_client = AsyncOpenAI(api_key=api_key, max_retries=0)
            return OpenAIResponsesModel(
                model_name, provider=OpenAIProvider(openai_client=openai_client), settings=settings
            )
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_key), settings=settings)

    def _make_anthropic_model(
        self, model_name: str, settings: ModelSettings | None = None, *, disable_retries: bool = False
    ) -> Model:
        """Create an Anthropic model with configured API key.

        Args:
            disable_retries: If True, create client with max_retries=0 (use for non-last models in FallbackModel chains)
        """
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        api_key = self.anthropic_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not configured. "
                "Set it in .env, set PYDANTIC_AI_GATEWAY_API_KEY for gateway fallback, "
                "or override the model tier (e.g., DEFAULT_MODEL=openai:gpt-5.4)."
            )
        if disable_retries:
            anthropic_client = AsyncAnthropic(api_key=api_key, max_retries=0)
            return AnthropicModel(
                model_name, provider=AnthropicProvider(anthropic_client=anthropic_client), settings=settings
            )
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_key), settings=settings)

    def _make_google_gla_model(
        self, model_name: str, settings: ModelSettings | None = None, *, disable_retries: bool = False
    ) -> Model:
        """Create a Google Gemini model via Generative Language API.

        Used for direct Google API access (google: or google-gla: prefixes).
        Requires GOOGLE_API_KEY from aistudio.google.com.

        Note: disable_retries is accepted for API compatibility but not applied —
        Google's GenAI SDK does not expose max_retries in the same way as OpenAI/Anthropic.
        Google models in non-final fallback chain positions may retry internally.
        """
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        api_key = self.google_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY is not configured. "
                "Set it in .env, set PYDANTIC_AI_GATEWAY_API_KEY for gateway fallback, "
                "or override the model tier (e.g., DEFAULT_MODEL_SMALL=openai:gpt-5-mini)."
            )
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key), settings=settings)

    def _make_gateway_openai_model(
        self, model_name: str, settings: ModelSettings | None, *, disable_retries: bool = False
    ) -> OpenAIResponsesModel:
        """Create an OpenAI model via Pydantic AI Gateway.

        Args:
            disable_retries: If True, create client with max_retries=0 (use for non-last models in FallbackModel chains)
        """
        from openai import AsyncOpenAI
        from pydantic_ai.providers.gateway import gateway_provider

        api_key = self.pydantic_ai_gateway_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("PYDANTIC_AI_GATEWAY_API_KEY not configured")
        provider = gateway_provider("openai-responses", api_key=api_key, base_url=self.pydantic_ai_gateway_base_url)
        if disable_retries:
            # gateway_provider returns an OpenAIProvider-compatible object; wrap with a no-retry client
            gateway_base_url = self.pydantic_ai_gateway_base_url
            openai_client = AsyncOpenAI(api_key=api_key, base_url=gateway_base_url, max_retries=0)
            direct_provider = OpenAIProvider(openai_client=openai_client)
            return OpenAIResponsesModel(model_name, provider=direct_provider, settings=settings)
        return OpenAIResponsesModel(model_name, provider=provider, settings=settings)

    def _make_gateway_anthropic_model(
        self, model_name: str, settings: ModelSettings | None = None, *, disable_retries: bool = False
    ) -> Model:
        """Create an Anthropic model via Pydantic AI Gateway.

        Args:
            disable_retries: If True, create client with max_retries=0 (use for non-last models in FallbackModel chains)
        """
        from anthropic import AsyncAnthropic
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        from pydantic_ai.providers.gateway import gateway_provider

        api_key = self.pydantic_ai_gateway_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("PYDANTIC_AI_GATEWAY_API_KEY not configured")
        if disable_retries:
            gateway_base_url = self.pydantic_ai_gateway_base_url
            anthropic_client = AsyncAnthropic(api_key=api_key, base_url=gateway_base_url, max_retries=0)
            direct_provider = AnthropicProvider(anthropic_client=anthropic_client)
            return AnthropicModel(model_name, provider=direct_provider, settings=settings)
        provider = gateway_provider("anthropic", api_key=api_key, base_url=self.pydantic_ai_gateway_base_url)
        return AnthropicModel(model_name, provider=provider, settings=settings)

    def _make_gateway_google_vertex_model(
        self, model_name: str, settings: ModelSettings | None = None, *, disable_retries: bool = False
    ) -> Model:
        """Create a Google Gemini model via Pydantic AI Gateway (Vertex AI).

        Used for gateway/google: or gateway/google-vertex: prefixes.
        Gateway only supports Google via Vertex AI backend.
        Requires PYDANTIC_AI_GATEWAY_API_KEY.

        Note: disable_retries is accepted for API compatibility but not applied —
        Google's GenAI SDK does not expose max_retries in the same way as OpenAI/Anthropic.
        """
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.gateway import gateway_provider

        api_key = self.pydantic_ai_gateway_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("PYDANTIC_AI_GATEWAY_API_KEY not configured")
        provider = gateway_provider("google-vertex", api_key=api_key, base_url=self.pydantic_ai_gateway_base_url)
        return GoogleModel(model_name, provider=provider, settings=settings)

    # -------------------------------------------------------------------------
    # Ollama model validation and creation
    # -------------------------------------------------------------------------

    def _get_ollama_native_base_url(self) -> str:
        """Get Ollama native API base URL (strips /v1 suffix if present).

        The ollama_base_url setting points to the OpenAI-compatible endpoint (/v1),
        but the native Ollama API (e.g., /api/tags) is at the root.
        """
        base = self.ollama_base_url
        base = base.removesuffix("/v1")
        return base.rstrip("/")

    def _fetch_ollama_models(self) -> set[str]:
        """Fetch available model names from Ollama server.

        Returns:
            Set of model names (e.g., {"gpt-oss:20b", "llama3:latest"})

        Raises:
            httpx.ConnectError: If server is unreachable
            httpx.HTTPStatusError: If server returns error status
        """
        url = f"{self._get_ollama_native_base_url()}/api/tags"
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        return {model["name"] for model in data.get("models", [])}

    def _get_ollama_available_models(self) -> set[str]:
        """Get available Ollama models with caching.

        Returns cached result if available, otherwise fetches from server.
        Use clear_ollama_models_cache() to invalidate.
        """
        if self._ollama_models_cache is None:
            self._ollama_models_cache = self._fetch_ollama_models()
        return self._ollama_models_cache

    def clear_ollama_models_cache(self) -> None:
        """Clear the cached Ollama models list.

        Call this after pulling new models or changing ollama_base_url.
        """
        self._ollama_models_cache = None

    def _validate_ollama_model(self, model_name: str) -> None:
        """Validate that model is available on Ollama server.

        Args:
            model_name: Model name to validate (e.g., "llama3" or "llama3:70b")

        Raises:
            ConfigurationError: If server is unreachable, returns an error,
                or the model is not available.
        """
        try:
            available = self._get_ollama_available_models()
        except httpx.ConnectError as e:
            raise ConfigurationError(
                f"Ollama server not reachable at {self._get_ollama_native_base_url()}. "
                f"Ensure Ollama is running: ollama serve"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ConfigurationError(
                f"Ollama server returned status {e.response.status_code} at {self._get_ollama_native_base_url()}"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Cannot connect to Ollama server: {e}") from e

        # Check for exact match or implicit :latest tag
        if model_name in available:
            return
        if f"{model_name}:latest" in available:
            return

        # Model not found - raise helpful error
        available_list = ", ".join(sorted(available)[:10])
        if len(available) > 10:
            available_list += f", ... ({len(available)} total)"

        raise ConfigurationError(
            f"Ollama model '{model_name}' not found. Available: {available_list}. Pull with: ollama pull {model_name}"
        )

    def _make_ollama_model(self, model_name: str) -> Model:
        """Create an Ollama model, validating availability first."""
        self._validate_ollama_model(model_name)

        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider

        return OpenAIChatModel(model_name, provider=OllamaProvider(base_url=self.ollama_base_url))


settings = FindingModelAIConfig()


__all__ = [
    "AgentTag",
    "FindingModelAIConfig",
    "ModelTier",
    "ReasoningLevel",
    "ResolvedModelConfig",
    "settings",
]
