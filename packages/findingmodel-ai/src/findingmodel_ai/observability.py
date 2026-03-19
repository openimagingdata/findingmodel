"""Shared Logfire runtime configuration for findingmodel-ai."""

from typing import Literal

import logfire

from findingmodel_ai.config import settings

_instrumented = False


def ensure_logfire_configured(*, console: bool = False, send_to_logfire: bool | None = None) -> None:
    """Configure Logfire once and enable Pydantic AI + HTTPX instrumentation.

    Reads LOGFIRE_TOKEN from .env via FindingModelAIConfig (pydantic-settings).
    Do not rely on ambient env vars — this function is the single runtime path.

    Args:
        console: Whether to enable console output (default False)
        send_to_logfire: Override cloud sending. None = auto (send if token present).
    """
    global _instrumented

    if _instrumented:
        return

    token = settings.logfire_token.get_secret_value() if settings.logfire_token else None

    if settings.disable_send_to_logfire:
        effective_send = False
    elif send_to_logfire is not None:
        effective_send = send_to_logfire
    else:
        effective_send = bool(token)

    console_option: logfire.ConsoleOptions | Literal[False] = logfire.ConsoleOptions() if console else False
    logfire.configure(
        send_to_logfire=effective_send,
        token=token or None,
        console=console_option,
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()

    _instrumented = True


__all__ = ["ensure_logfire_configured"]
