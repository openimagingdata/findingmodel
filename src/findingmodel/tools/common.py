"""Common utility functions for tools."""

from pathlib import Path

from instructor import AsyncInstructor, from_openai
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from findingmodel import logger
from findingmodel.config import settings


def get_async_instructor_client() -> AsyncInstructor:
    settings.check_ready_for_openai()
    return from_openai(AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()))


def get_async_perplexity_client() -> AsyncOpenAI:
    settings.check_ready_for_perplexity()
    return AsyncOpenAI(
        api_key=str(settings.perplexity_api_key.get_secret_value()), base_url=str(settings.perplexity_base_url)
    )


def get_openai_model(model_name: str) -> OpenAIModel:
    """Helper function to get OpenAI model instance - moved from similar_finding_models.py"""
    return OpenAIModel(
        model_name=model_name,
        provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value()),
    )


async def get_embedding(
    text: str, client: AsyncOpenAI | None = None, model: str | None = None, dimensions: int = 512
) -> list[float] | None:
    """Get embedding for a single text using OpenAI embeddings API.

    Args:
        text: Text to embed
        client: Optional OpenAI client (creates one if not provided)
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embedding (default: 512)

    Returns:
        Embedding vector or None if failed
    """
    if not client:
        if not settings.openai_api_key:
            return None
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    # Use config setting if model not explicitly provided
    if model is None:
        model = settings.openai_embedding_model

    try:
        response = await client.embeddings.create(input=text, model=model, dimensions=dimensions)
        return response.data[0].embedding
    except Exception as e:
        logger.warning(f"Failed to get embedding: {e}")
        return None


async def get_embeddings_batch(
    texts: list[str], client: AsyncOpenAI | None = None, model: str | None = None, dimensions: int = 512
) -> list[list[float] | None]:
    """Get embeddings for a batch of texts using OpenAI embeddings API.

    Args:
        texts: List of texts to embed
        client: Optional OpenAI client (creates one if not provided)
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embeddings (default: 512)

    Returns:
        List of embedding vectors (or None for failed embeddings)
    """
    if not texts:
        return []

    if not client:
        if not settings.openai_api_key:
            return [None] * len(texts)
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    # Use config setting if model not explicitly provided
    if model is None:
        model = settings.openai_embedding_model

    try:
        response = await client.embeddings.create(input=texts, model=model, dimensions=dimensions)
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Failed to get embeddings batch: {e}")
        return [None] * len(texts)


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
