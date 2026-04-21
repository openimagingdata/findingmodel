"""Compatibility shims between ``findingmodel`` and ``findingmodel-ai``.

Historically, some projects imported a separate ``compat`` module to monkeypatch
``findingmodel.index`` before using AI tooling. In this monorepo, ``findingmodel``
exposes :class:`findingmodel.index.FindingModelIndex` (re-exported as ``Index``) and
``findingmodel.tools.add_ids_to_model`` already uses that index API.

This module is imported from :mod:`findingmodel_ai` so a single, documented place
exists for any future runtime patches. Today it applies no patches.
"""


def apply_compat() -> None:
    """Apply compatibility patches if needed (currently a no-op)."""


apply_compat()
