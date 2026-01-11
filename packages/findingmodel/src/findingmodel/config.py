from pathlib import Path
from typing import Annotated, Literal, cast

import httpx
import openai
from oidm_common.distribution import ensure_db_file as oidm_ensure_db_file
from pydantic import BeforeValidator, Field, PrivateAttr, SecretStr, model_validator
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class ConfigurationError(RuntimeError):
    pass


# Type definitions for model configuration
ModelTier = Literal["base", "small", "full"]

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
    "similar_search",  # Searches for similar findings (covers 2 agents)
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
#   - google, google-gla: Google Generative Language API (requires GOOGLE_API_KEY)
# Gateway providers (via Pydantic AI Gateway):
#   - gateway/openai, gateway/anthropic: standard gateway routing
#   - gateway/google, gateway/google-vertex: Google Vertex AI (gateway only supports Vertex)
# Model names: alphanumeric, hyphens, dots, colons (for versions), underscores
MODEL_SPEC_PATTERN = (
    r"^(openai|anthropic|google|google-gla|ollama|gateway/(openai|anthropic|google|google-vertex)):[\w.:-]+$"
)

ModelSpec = Annotated[
    str,
    Field(
        pattern=MODEL_SPEC_PATTERN,
        description="Model spec: 'provider:model' (e.g., 'openai:gpt-5-mini', 'google:gemini-3-flash-preview')",
    ),
]


def strip_quotes(value: str) -> str:
    return value.strip("\"'")


def strip_quotes_secret(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    return strip_quotes(value)


QuoteStrippedStr = Annotated[str, BeforeValidator(strip_quotes)]


QuoteStrippedSecretStr = Annotated[SecretStr, BeforeValidator(strip_quotes_secret)]


class FindingModelConfig(BaseSettings):
    # API Keys
    openai_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    anthropic_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    google_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    pydantic_ai_gateway_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
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
    default_model: ModelSpec = Field(default="openai:gpt-5-mini")
    default_model_full: ModelSpec = Field(default="openai:gpt-5.2")
    default_model_small: ModelSpec = Field(default="openai:gpt-5-nano")

    # Per-agent model overrides
    agent_model_overrides: dict[AgentTag, ModelSpec] = Field(default_factory=dict)

    # Tavily API
    tavily_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
    tavily_search_depth: Literal["basic", "advanced"] = Field(
        default="advanced",
        description="Tavily search depth: 'basic' or 'advanced'",
    )

    # BioOntology API
    bioontology_api_key: QuoteStrippedSecretStr | None = Field(default=None, description="BioOntology.org API key")

    # Logfire configuration (observability platform)
    logfire_token: QuoteStrippedSecretStr | None = Field(
        default=None,
        description="Logfire.dev write token for cloud tracing (optional)",
    )
    disable_send_to_logfire: bool = Field(
        default=False,
        description="Disable sending data to Logfire platform (local-only mode)",
    )
    logfire_verbose: bool = Field(
        default=False,
        description="Enable verbose Logfire console logging",
    )

    # DuckDB configuration
    duckdb_index_path: str | None = Field(
        default=None,
        description="Path to finding models index database (absolute, relative to user data dir, or None for default)",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI model for generating embeddings"
    )
    openai_embedding_dimensions: int = Field(
        default=512, description="Embedding dimensions (512 for text-embedding-3-small reduced, 1536 for full)"
    )

    # Optional remote DuckDB download URLs
    remote_index_db_url: str | None = Field(
        default=None,
        description="URL to download finding models index database",
    )
    remote_index_db_hash: str | None = Field(
        default=None,
        description="SHA256 hash for index DB (e.g. 'sha256:def...')",
    )
    remote_manifest_url: str | None = Field(
        default="https://findingmodelsdata.t3.storage.dev/manifest.json",
        description="URL to JSON manifest for database versions",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    # Private instance cache for Ollama available models
    _ollama_models_cache: set[str] | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_remote_db_config(self) -> Self:
        """Validate that remote URL and hash are provided together (or neither)."""
        # Check index database config
        if (self.remote_index_db_url is None) != (self.remote_index_db_hash is None):
            raise ValueError(
                "Must provide both REMOTE_INDEX_DB_URL and REMOTE_INDEX_DB_HASH, or neither. "
                f"Got URL={'set' if self.remote_index_db_url else 'unset'}, "
                f"hash={'set' if self.remote_index_db_hash else 'unset'}"
            )

        return self

    def check_ready_for_tavily(self) -> Literal[True]:
        if not self.tavily_api_key.get_secret_value():
            raise ConfigurationError("Tavily API key is not set")
        return True

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
            "ollama": None,  # No API key required
        }.get(provider_part)

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

        missing: list[tuple[str, str, str]] = []
        for tier_name, model_string in models_to_check:
            key_field = self._get_required_key_field(model_string)
            if key_field:
                key_value = getattr(self, key_field).get_secret_value()
                if not key_value:
                    missing.append((tier_name, model_string, key_field))

        if missing:
            # Format: "default_model (openai:gpt-5-mini): OPENAI_API_KEY"
            details = "; ".join(f"{tier} ({model}): {key.upper()}" for tier, model, key in missing)
            raise ConfigurationError(f"Missing API keys for default models: {details}")

    def _create_model_from_string(self, model_string: str, default_tier: ModelTier = "base") -> Model:
        """Create a Pydantic AI model instance from a model string.

        Args:
            model_string: Model specification (e.g., "openai:gpt-5-mini", "anthropic:claude-sonnet-4-5")
            default_tier: Tier to use for tier-specific settings (e.g., "small" enables minimal reasoning)

        Returns:
            Configured Model instance

        Raises:
            ConfigurationError: If model string is invalid, API key missing, or provider unknown
        """
        if ":" not in model_string:
            raise ConfigurationError(f"Invalid model format '{model_string}'. Expected 'provider:model_name'")

        provider_part, model_name = model_string.split(":", 1)
        parts = provider_part.split("/")

        # Small tier uses minimal reasoning for faster/cheaper OpenAI responses
        # Only gpt-5+ and o1+ models support the reasoning parameter
        supports_reasoning = model_name.startswith(("gpt-5", "o1"))
        openai_settings = (
            ModelSettings(extra_body={"reasoning": {"effort": "minimal"}})
            if default_tier == "small" and supports_reasoning
            else None
        )

        if parts == ["openai"]:
            return self._make_openai_model(model_name, openai_settings)
        elif parts == ["anthropic"]:
            return self._make_anthropic_model(model_name)
        elif parts in [["google"], ["google-gla"]]:
            # Both 'google:' and 'google-gla:' map to Generative Language API
            return self._make_google_gla_model(model_name)
        elif parts == ["ollama"]:
            return self._make_ollama_model(model_name)
        elif parts == ["gateway", "openai"]:
            return self._make_gateway_openai_model(model_name, openai_settings)
        elif parts == ["gateway", "anthropic"]:
            return self._make_gateway_anthropic_model(model_name)
        elif parts in [["gateway", "google"], ["gateway", "google-vertex"]]:
            # Both 'gateway/google:' and 'gateway/google-vertex:' map to Vertex AI
            return self._make_gateway_google_vertex_model(model_name)
        else:
            raise ConfigurationError(f"Unknown provider '{provider_part}'")

    def get_effective_model_string(self, agent_tag: AgentTag, default_tier: ModelTier = "base") -> str:
        """Return the model string that would be used for an agent.

        Useful for metadata/logging when you need the string, not the Model object.

        Args:
            agent_tag: Agent identifier
            default_tier: Tier to use if no override configured

        Returns:
            Model specification string (e.g., "openai:gpt-5-mini")
        """
        if agent_tag in self.agent_model_overrides:
            return self.agent_model_overrides[agent_tag]
        return {"base": self.default_model, "full": self.default_model_full, "small": self.default_model_small}[
            default_tier
        ]

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

    def _make_openai_model(self, model_name: str, settings: ModelSettings | None) -> OpenAIResponsesModel:
        """Create an OpenAI model with configured API key."""
        api_key = self.openai_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY not configured")
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_key), settings=settings)

    def _make_anthropic_model(self, model_name: str) -> Model:
        """Create an Anthropic model with configured API key."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        api_key = self.anthropic_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY not configured")
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_key))

    def _make_google_gla_model(self, model_name: str) -> Model:
        """Create a Google Gemini model via Generative Language API.

        Used for direct Google API access (google: or google-gla: prefixes).
        Requires GOOGLE_API_KEY from aistudio.google.com.
        """
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        api_key = self.google_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("GOOGLE_API_KEY not configured")
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key))

    def _make_gateway_openai_model(self, model_name: str, settings: ModelSettings | None) -> OpenAIResponsesModel:
        """Create an OpenAI model via Pydantic AI Gateway."""
        from pydantic_ai.providers.gateway import gateway_provider

        api_key = self.pydantic_ai_gateway_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("PYDANTIC_AI_GATEWAY_API_KEY not configured")
        provider = gateway_provider("openai-responses", api_key=api_key, base_url=self.pydantic_ai_gateway_base_url)
        return OpenAIResponsesModel(model_name, provider=provider, settings=settings)

    def _make_gateway_anthropic_model(self, model_name: str) -> Model:
        """Create an Anthropic model via Pydantic AI Gateway."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.gateway import gateway_provider

        api_key = self.pydantic_ai_gateway_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("PYDANTIC_AI_GATEWAY_API_KEY not configured")
        provider = gateway_provider("anthropic", api_key=api_key, base_url=self.pydantic_ai_gateway_base_url)
        return AnthropicModel(model_name, provider=provider)

    def _make_gateway_google_vertex_model(self, model_name: str) -> Model:
        """Create a Google Gemini model via Pydantic AI Gateway (Vertex AI).

        Used for gateway/google: or gateway/google-vertex: prefixes.
        Gateway only supports Google via Vertex AI backend.
        Requires PYDANTIC_AI_GATEWAY_API_KEY.
        """
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.gateway import gateway_provider

        api_key = self.pydantic_ai_gateway_api_key.get_secret_value()
        if not api_key:
            raise ConfigurationError("PYDANTIC_AI_GATEWAY_API_KEY not configured")
        provider = gateway_provider("google-vertex", api_key=api_key, base_url=self.pydantic_ai_gateway_base_url)
        return GoogleModel(model_name, provider=provider)

    # -------------------------------------------------------------------------
    # Ollama model validation and creation
    # -------------------------------------------------------------------------

    def _get_ollama_native_base_url(self) -> str:
        """Get Ollama native API base URL (strips /v1 suffix if present).

        The ollama_base_url setting points to the OpenAI-compatible endpoint (/v1),
        but the native Ollama API (e.g., /api/tags) is at the root.
        """
        base = self.ollama_base_url
        if base.endswith("/v1"):
            base = base[:-3]
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


settings = FindingModelConfig()
openai.api_key = settings.openai_api_key.get_secret_value()


def ensure_index_db() -> Path:
    """Ensure finding models index database is available.

    Uses settings to determine path, URL, and hash configuration.
    Automatically downloads from manifest if no local file exists.

    Returns:
        Path to the finding models index database
    """
    return cast(
        Path,
        oidm_ensure_db_file(
            file_path=settings.duckdb_index_path,
            remote_url=settings.remote_index_db_url,
            remote_hash=settings.remote_index_db_hash,
            manifest_key="finding_models",
            manifest_url=settings.remote_manifest_url,
            app_name="findingmodel",
        ),
    )


__all__ = [
    "AgentTag",
    "ConfigurationError",
    "FindingModelConfig",
    "ModelTier",
    "ensure_index_db",
    "settings",
]
