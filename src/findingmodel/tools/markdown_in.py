"""Finding model creation from markdown tools."""

from pathlib import Path

from pydantic_ai import Agent

from findingmodel.config import ModelTier, settings
from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import FindingModelBase

from .common import get_markdown_text_from_path_or_text, get_model
from .prompt_template import load_prompt_template, render_agent_prompt


async def create_model_from_markdown(
    finding_info: FindingInfo,
    /,
    markdown_path: str | Path | None = None,
    markdown_text: str | None = None,
    model_tier: ModelTier = "base",
) -> FindingModelBase:
    """
    Create a finding model from a markdown file or text using the OpenAI API.
    :param finding_info: The finding information or name to use for the model.
    :param markdown_path: The path to the markdown file containing the outline.
    :param markdown_text: The markdown text containing the outline.
    :param model_tier: The model tier to use ("small", "base", or "full").
    :return: A FindingModelBase object containing the finding model.
    """

    assert isinstance(finding_info, FindingInfo), "Finding info must be a FindingInfo object"
    markdown_text = get_markdown_text_from_path_or_text(
        markdown_text=markdown_text,
        markdown_path=markdown_path,
    )
    prompt_template = load_prompt_template("get_finding_model_from_outline")
    instructions, user_prompt = render_agent_prompt(
        prompt_template,
        finding_info=finding_info,
        outline=markdown_text,
    )
    agent = Agent[None, FindingModelBase](
        model=get_model(model_tier),
        output_type=FindingModelBase,
        instructions=instructions,
    )
    result = await agent.run(user_prompt)
    if not isinstance(result.output, FindingModelBase):
        raise ValueError("Finding model not returned.")
    return result.output


# Deprecated alias for backward compatibility
async def create_finding_model_from_markdown(
    finding_info: FindingInfo,
    /,
    markdown_path: str | Path | None = None,
    markdown_text: str | None = None,
    openai_model: str = settings.openai_default_model,
) -> FindingModelBase:
    """
    DEPRECATED: Use create_model_from_markdown instead.
    Create a finding model from a markdown file or text using the OpenAI API.
    """
    import warnings

    warnings.warn(
        "create_finding_model_from_markdown is deprecated, use create_model_from_markdown instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # Map old model_name parameter to new model_tier - use "base" as default since that matches the old default
    return await create_model_from_markdown(
        finding_info, markdown_path=markdown_path, markdown_text=markdown_text, model_tier="base"
    )
