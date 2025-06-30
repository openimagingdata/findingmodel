"""Finding model creation from markdown tools."""

from pathlib import Path

from findingmodel.config import settings
from findingmodel.finding_info import FindingInfo
from findingmodel.finding_model import FindingModelBase

from .common import get_async_instructor_client, get_markdown_text_from_path_or_text
from .prompt_template import create_prompt_messages, load_prompt_template


async def create_model_from_markdown(
    finding_info: FindingInfo,
    /,
    markdown_path: str | Path | None = None,
    markdown_text: str | None = None,
    openai_model: str = settings.openai_default_model,
) -> FindingModelBase:
    """
    Create a finding model from a markdown file or text using the OpenAI API.
    :param finding_info: The finding information or name to use for the model.
    :param markdown_path: The path to the markdown file containing the outline.
    :param markdown_text: The markdown text containing the outline.
    :param openai_model: The OpenAI model to use for the finding model.
    :return: A FindingModelBase object containing the finding model.
    """

    assert isinstance(finding_info, FindingInfo), "Finding info must be a FindingInfo object"
    markdown_text = get_markdown_text_from_path_or_text(
        markdown_text=markdown_text,
        markdown_path=markdown_path,
    )
    prompt_template = load_prompt_template("get_finding_model_from_outline")
    messages = create_prompt_messages(
        prompt_template,
        finding_info=finding_info,
        outline=markdown_text,
    )
    client = get_async_instructor_client()
    result = await client.chat.completions.create(
        messages=messages,
        response_model=FindingModelBase,
        model=openai_model,
    )
    if not isinstance(result, FindingModelBase):
        raise ValueError("Finding model not returned.")
    return result


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
    return await create_model_from_markdown(
        finding_info, markdown_path=markdown_path, markdown_text=markdown_text, openai_model=openai_model
    )
