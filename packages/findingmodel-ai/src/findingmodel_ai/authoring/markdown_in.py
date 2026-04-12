"""Finding model creation from markdown tools."""

from pathlib import Path

from findingmodel import FindingInfo, FindingModelBase
from pydantic_ai import Agent

from findingmodel_ai._internal.common import get_markdown_text_from_path_or_text
from findingmodel_ai._internal.prompts import load_prompt_template, render_agent_prompt
from findingmodel_ai.config import settings


async def create_model_from_markdown(
    finding_info: FindingInfo,
    /,
    markdown_path: str | Path | None = None,
    markdown_text: str | None = None,
) -> FindingModelBase:
    """
    Create a finding model from a markdown outline or free-form text using the OpenAI API.

    This is an AI-assisted outline importer for authoring convenience. It is not intended
    to guarantee faithful reconstruction of a previously exported finding model Markdown view.

    :param finding_info: The finding information or name to use for the model.
    :param markdown_path: The path to the markdown file containing the outline.
    :param markdown_text: The markdown text containing the outline.
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
        model=settings.get_agent_model("import_markdown"),
        output_type=FindingModelBase,
        instructions=instructions,
    )
    result = await agent.run(user_prompt)
    if not isinstance(result.output, FindingModelBase):
        raise ValueError("Finding model not returned.")
    return result.output
