"""Evaluation suites for findingmodel agents.

This module configures Logfire observability once for the entire package.
Individual eval modules require NO Logfire code - automatic instrumentation
is provided by:
- Pydantic Evals: Dataset.evaluate() creates root + per-case spans
- Pydantic AI: logfire.instrument_pydantic_ai() traces agent/model/tool calls

Configuration is read from .env via findingmodel.config.settings.

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import logfire
from logfire import ConsoleOptions

from findingmodel.config import settings

# Configure Logfire once for entire evals package
# Follows Python logging best practice: configure at package level
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

__all__ = []  # No public exports - configuration only
