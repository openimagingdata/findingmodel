"""Tests for shared Logfire runtime observability setup."""

from findingmodel_ai import observability
from findingmodel_ai.config import FindingModelAIConfig
from pytest import MonkeyPatch


def test_ensure_logfire_configured_instruments_pydantic_ai_and_httpx(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(observability, "_instrumented", False)
    monkeypatch.setattr(observability, "settings", FindingModelAIConfig())
    monkeypatch.setattr(
        observability.logfire,
        "configure",
        lambda **kwargs: calls.append(("configure", kwargs)),
    )
    monkeypatch.setattr(
        observability.logfire,
        "instrument_pydantic_ai",
        lambda: calls.append(("instrument_pydantic_ai", None)),
    )
    monkeypatch.setattr(
        observability.logfire,
        "instrument_httpx",
        lambda: calls.append(("instrument_httpx", None)),
    )

    observability.ensure_logfire_configured()

    assert calls == [
        ("configure", {"send_to_logfire": "if-token-present", "console": False}),
        ("instrument_pydantic_ai", None),
        ("instrument_httpx", None),
    ]


def test_ensure_logfire_configured_respects_disable_flag(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(observability, "_instrumented", False)
    monkeypatch.setattr(observability, "settings", FindingModelAIConfig(disable_send_to_logfire=True))
    monkeypatch.setattr(
        observability.logfire,
        "configure",
        lambda **kwargs: calls.append(("configure", kwargs)),
    )
    monkeypatch.setattr(observability.logfire, "instrument_pydantic_ai", lambda: None)
    monkeypatch.setattr(observability.logfire, "instrument_httpx", lambda: None)

    observability.ensure_logfire_configured(console=True)

    assert len(calls) == 1
    assert calls[0][0] == "configure"
    assert calls[0][1]["send_to_logfire"] is False
    # console=True is converted to ConsoleOptions()
    assert isinstance(calls[0][1]["console"], observability.logfire.ConsoleOptions)


def test_ensure_logfire_configured_is_idempotent(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(observability, "_instrumented", False)
    monkeypatch.setattr(observability, "settings", FindingModelAIConfig())
    monkeypatch.setattr(observability.logfire, "configure", lambda **kwargs: calls.append("configure"))
    monkeypatch.setattr(observability.logfire, "instrument_pydantic_ai", lambda: calls.append("pydantic_ai"))
    monkeypatch.setattr(observability.logfire, "instrument_httpx", lambda: calls.append("httpx"))

    observability.ensure_logfire_configured()
    observability.ensure_logfire_configured(console=True, send_to_logfire=False)

    assert calls == ["configure", "pydantic_ai", "httpx"]
