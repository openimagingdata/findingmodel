"""Common utility functions for tools."""

from pathlib import Path

from instructor import AsyncInstructor, from_openai
from openai import AsyncOpenAI

from findingmodel.config import settings


def get_async_instructor_client() -> AsyncInstructor:
    settings.check_ready_for_openai()
    return from_openai(AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()))


def get_async_perplexity_client() -> AsyncOpenAI:
    settings.check_ready_for_perplexity()
    return AsyncOpenAI(
        api_key=str(settings.perplexity_api_key.get_secret_value()), base_url=str(settings.perplexity_base_url)
    )


def get_markdown_text_from_path_or_text(
    *, markdown_text: str | None = None, markdown_path: str | Path | None = None
) -> str:
    """
    Get the markdown text from either a string or a file path.
    Exactly one of markdown_text or markdown_path must be provided.

    :param markdown_text: The markdown text as a string.
    :param markdown_path: The path to the markdown file.
    :return: The markdown text.
    """
    if markdown_text is not None and markdown_path is not None:
        raise ValueError("Only one of markdown_text or markdown_path should be provided")
    if markdown_text is None and markdown_path is None:
        raise ValueError("Either markdown_text or markdown_path must be provided")

    if markdown_text is not None:
        return markdown_text

    # If markdown_path is provided
    if isinstance(markdown_path, str):
        markdown_path = Path(markdown_path)
    if not markdown_path or not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
    return markdown_path.read_text()
