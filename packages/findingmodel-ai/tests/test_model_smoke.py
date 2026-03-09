"""Smoke tests: verify every supported model is reachable and accepts its reasoning configs.

Tests are driven from supported_models.toml — adding a model there automatically
adds connectivity and reasoning tests here.

Run with: task test-full
"""

import httpx
import pytest
from pydantic_ai import Agent, models
from pydantic_ai.models import Model

from findingmodel_ai.config import SUPPORTED_MODELS, settings as ai_settings

# Prevent accidental model requests in unit tests
models.ALLOW_MODEL_REQUESTS = False


# ---------------------------------------------------------------------------
# Key / server availability
# ---------------------------------------------------------------------------
HAS_OPENAI_KEY = bool(ai_settings.openai_api_key.get_secret_value())
HAS_ANTHROPIC_KEY = bool(ai_settings.anthropic_api_key.get_secret_value())
HAS_GOOGLE_KEY = bool(ai_settings.google_api_key.get_secret_value())

_KEY_FLAGS = {
    "openai": HAS_OPENAI_KEY,
    "anthropic": HAS_ANTHROPIC_KEY,
    "google": HAS_GOOGLE_KEY,
}


def _ollama_available_models() -> set[str]:
    """Return set of models available on local Ollama, or empty set if server is down."""
    try:
        return ai_settings._fetch_ollama_models()
    except (httpx.ConnectError, httpx.HTTPStatusError, Exception):
        return set()


_OLLAMA_MODELS = _ollama_available_models()


def _skip_reason(spec: str, provider: str) -> str | None:
    """Return a skip reason if the model can't be tested, or None if it can."""
    if provider == "ollama":
        if not _OLLAMA_MODELS:
            return "Ollama server not reachable"
        # Extract model name after "ollama:" prefix
        model_name = spec.split(":", 1)[1]
        if model_name not in _OLLAMA_MODELS and f"{model_name}:latest" not in _OLLAMA_MODELS:
            return f"Ollama model '{model_name}' not pulled"
        return None
    if provider in _KEY_FLAGS and not _KEY_FLAGS[provider]:
        key_name = provider.upper() + "_API_KEY"
        return f"{key_name} not configured"
    return None


# ---------------------------------------------------------------------------
# Parametrize from TOML: connectivity tests (one per model)
# ---------------------------------------------------------------------------
def _connectivity_params() -> list[pytest.param]:
    """Generate one pytest.param per model in supported_models.toml."""
    params = []
    for spec, entry in SUPPORTED_MODELS.items():
        provider = entry.get("provider", "unknown")
        reason = _skip_reason(spec, provider)
        marks = [pytest.mark.callout]
        if reason:
            marks.append(pytest.mark.skipif(True, reason=reason))
        params.append(pytest.param(spec, marks=marks, id=spec))
    return params


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec", _connectivity_params())
async def test_model_reachable(model_spec: str) -> None:
    """Verify the model responds to a trivial prompt."""
    model: Model = ai_settings._create_model_from_string(model_spec)
    agent: Agent[None, str] = Agent(model=model, output_type=str, system_prompt="Reply with exactly: OK")
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        result = await agent.run("Say OK")
        assert isinstance(result.output, str)
        assert len(result.output.strip()) > 0
    finally:
        models.ALLOW_MODEL_REQUESTS = original


# ---------------------------------------------------------------------------
# Parametrize from TOML: reasoning tests (one per model × valid level)
# ---------------------------------------------------------------------------
def _reasoning_params() -> list[pytest.param]:
    """Generate one pytest.param per model × valid reasoning level."""
    params = []
    for spec, entry in SUPPORTED_MODELS.items():
        provider = entry.get("provider", "unknown")
        valid_levels = entry.get("valid_reasoning", [])
        if not valid_levels:
            continue  # Ollama — no reasoning support
        reason = _skip_reason(spec, provider)
        for level in valid_levels:
            marks = [pytest.mark.callout]
            if reason:
                marks.append(pytest.mark.skipif(True, reason=reason))
            params.append(pytest.param(spec, level, marks=marks, id=f"{spec}[{level}]"))
    return params


@pytest.mark.asyncio
@pytest.mark.parametrize("model_spec,reasoning_level", _reasoning_params())
async def test_reasoning_accepted(model_spec: str, reasoning_level: str) -> None:
    """Verify the model accepts the given reasoning configuration."""
    provider_part, model_name = model_spec.split(":", 1)
    model_settings = ai_settings._build_reasoning_settings(provider_part, model_name, reasoning_level)  # type: ignore[arg-type]
    model: Model = ai_settings._create_model_from_string(model_spec)
    agent: Agent[None, str] = Agent(model=model, output_type=str, system_prompt="Reply with exactly: OK")
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        result = await agent.run("Say OK", model_settings=model_settings)
        assert isinstance(result.output, str)
        assert len(result.output.strip()) > 0
    finally:
        models.ALLOW_MODEL_REQUESTS = original


# ---------------------------------------------------------------------------
# Default tiers — verify the configured defaults work end-to-end
# ---------------------------------------------------------------------------
def _tier_params() -> list[pytest.param]:
    """Generate one pytest.param per configured tier."""
    tiers = [
        ("base", ai_settings.default_model),
        ("small", ai_settings.default_model_small),
        ("full", ai_settings.default_model_full),
    ]
    params = []
    for tier_name, model_string in tiers:
        key_field = ai_settings._get_required_key_field(model_string)
        marks = [pytest.mark.callout]
        if key_field:
            key_value = getattr(ai_settings, key_field).get_secret_value()
            if not key_value:
                marks.append(
                    pytest.mark.skipif(True, reason=f"{key_field.upper()} not configured for {tier_name} tier")
                )
        params.append(pytest.param(tier_name, marks=marks, id=f"tier-{tier_name}"))
    return params


@pytest.mark.asyncio
@pytest.mark.parametrize("tier_name", _tier_params())
async def test_tier_with_reasoning(tier_name: str) -> None:
    """Verify get_model() with configured reasoning works for each tier."""
    model = ai_settings.get_model(tier_name)  # type: ignore[arg-type]
    agent: Agent[None, str] = Agent(model=model, output_type=str, system_prompt="Reply with exactly: OK")
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = True
    try:
        result = await agent.run("Say OK")
        assert len(result.output.strip()) > 0
    finally:
        models.ALLOW_MODEL_REQUESTS = original
