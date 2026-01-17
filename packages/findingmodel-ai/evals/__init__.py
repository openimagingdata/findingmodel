"""Evaluation suites for findingmodel-ai agents.

This module provides Logfire instrumentation configuration for eval suites.
Individual eval modules require NO additional Logfire code - instrumentation is provided by:
- Pydantic Evals: Dataset.evaluate() creates root + per-case spans
- Pydantic AI: logfire.instrument_pydantic_ai() traces agent/model/tool calls

Configuration is read from .env via findingmodel.config.settings.

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import logfire
from findingmodel.config import settings
from logfire import ConsoleOptions

# Track instrumentation state to make ensure_instrumented() idempotent
_instrumented = False


def ensure_instrumented() -> None:
    """Ensure Logfire is configured and Pydantic AI is instrumented.

    This function is idempotent - safe to call multiple times.
    Call this explicitly in eval suite __main__ blocks before running evals.
    """
    global _instrumented

    if _instrumented:
        return

    # Configure Logfire once for entire evals package
    logfire.configure(
        token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
        send_to_logfire=False if settings.disable_send_to_logfire else "if-token-present",
        console=ConsoleOptions(
            colors="auto",
            min_log_level="debug",
        )
        if settings.logfire_verbose
        else False,
    )

    # Instrument Pydantic AI once for automatic agent/model/tool tracing
    logfire.instrument_pydantic_ai()

    _instrumented = True


__all__ = ["ensure_instrumented"]
