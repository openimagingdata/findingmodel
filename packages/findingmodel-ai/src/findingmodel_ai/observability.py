"""Shared Logfire runtime configuration for findingmodel-ai."""

from typing import Literal

import logfire

from findingmodel_ai.config import settings

_instrumented = False


def ensure_logfire_configured(
    *, console: bool = False, send_to_logfire: bool | Literal["if-token-present"] | None = None
) -> None:
    """Configure Logfire once and enable Pydantic AI + HTTPX instrumentation.

    This is the runtime entrypoint for processes that want observability beyond evals,
    including non-LLM HTTP calls such as BioOntology lookups.
    """
    global _instrumented

    if _instrumented:
        return

    effective_send_to_logfire: bool | Literal["if-token-present"]
    if settings.disable_send_to_logfire:
        effective_send_to_logfire = False
    elif send_to_logfire is None:
        effective_send_to_logfire = "if-token-present"
    else:
        effective_send_to_logfire = send_to_logfire

    console_option: logfire.ConsoleOptions | Literal[False] = logfire.ConsoleOptions() if console else False
    logfire.configure(send_to_logfire=effective_send_to_logfire, console=console_option)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()

    _instrumented = True


__all__ = ["ensure_logfire_configured"]
