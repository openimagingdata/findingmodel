"""Prompt resource loading for metadata enrichment agents."""

from __future__ import annotations

from importlib.resources import files

PROMPT_PACKAGE = "findingmodel_ai.metadata.prompts"


def load_metadata_prompt(prompt_id: str) -> str:
    """Load a Markdown prompt resource by stable prompt id."""

    if "/" in prompt_id or "\\" in prompt_id or prompt_id.startswith("."):
        raise ValueError(f"Invalid metadata prompt id: {prompt_id!r}")
    prompt_path = files(PROMPT_PACKAGE).joinpath(f"{prompt_id}.md")
    return prompt_path.read_text(encoding="utf-8").strip()
