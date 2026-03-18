"""Evaluation suites for findingmodel-ai agents.

This module provides Logfire instrumentation configuration for eval suites.
Individual eval modules require NO additional Logfire code - instrumentation is provided by:
- Pydantic Evals: Dataset.evaluate() creates root + per-case spans
- Pydantic AI: logfire.instrument_pydantic_ai() traces agent/model/tool calls

Configuration: LOGFIRE_TOKEN in .env is loaded via FindingModelAIConfig (pydantic-settings).
Do NOT rely on os.environ or call logfire.configure() directly.

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

# Track instrumentation state to make ensure_instrumented() idempotent
_instrumented = False


def ensure_instrumented() -> None:
    """Ensure Logfire is configured and Pydantic AI is instrumented.

    This function is idempotent - safe to call multiple times.
    Call this explicitly in eval suite __main__ blocks before running evals.

    Uses FindingModelAIConfig.configure_logfire() which reads LOGFIRE_TOKEN
    from .env via pydantic-settings — the single config path for this project.
    """
    global _instrumented

    if _instrumented:
        return

    from findingmodel_ai.config import settings

    settings.configure_logfire()

    _instrumented = True


__all__ = ["ensure_instrumented"]
