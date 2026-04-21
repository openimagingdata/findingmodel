"""Regression: agents package must not depend on external ``findingmodels``."""

from __future__ import annotations

import importlib
import sys

import pytest


def test_single_agent_module_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """``single_agent`` builds the OpenAI model at import time; provide a dummy API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-import-only-not-used")
    sys.modules.pop("findingmodel_ai.agents.single_agent", None)
    sa = importlib.import_module("findingmodel_ai.agents.single_agent")

    assert sa.HoodJsonAdapter is not None
    assert sa.create_single_agent() is sa.single_agent
