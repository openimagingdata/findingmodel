"""Evaluation suites for findingmodel-ai agents.

This module provides Logfire instrumentation configuration for eval suites.
Individual eval modules require NO additional Logfire code - instrumentation is provided by:
- Pydantic Evals: Dataset.evaluate() creates root + per-case spans
- Pydantic AI: traces agent/model/tool calls
- HTTPX: traces outbound API calls such as BioOntology requests

Configuration via environment:
- LOGFIRE_TOKEN: Set to send to Logfire cloud (optional)

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

from findingmodel_ai.observability import ensure_logfire_configured


def ensure_instrumented() -> None:
    """Ensure Logfire is configured and Pydantic AI is instrumented.

    This function is idempotent - safe to call multiple times.
    Call this explicitly in eval suite __main__ blocks before running evals.
    """
    ensure_logfire_configured(console=False)


__all__ = ["ensure_instrumented"]
