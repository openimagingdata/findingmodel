"""Common utility functions for tools."""

from pathlib import Path

from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from tavily import AsyncTavilyClient

from findingmodel import logger
from findingmodel.config import ConfigurationError, ModelProvider, ModelTier, settings
from findingmodel.embedding_cache import EmbeddingCache

# Module-level singleton for embedding cache
_embedding_cache: EmbeddingCache | None = None


async def _get_embedding_cache() -> EmbeddingCache:
    """Get or initialize the embedding cache singleton.

    Returns:
        Initialized EmbeddingCache instance
    """
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
        await _embedding_cache.setup()
    return _embedding_cache


def get_async_tavily_client() -> AsyncTavilyClient:
    """Get configured async Tavily search client."""
    settings.check_ready_for_tavily()
    return AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())


def get_model(
    model_tier: ModelTier = "base",
    *,
    provider: ModelProvider | None = None,
) -> Model:
    """Get AI model from configured provider.

    Args:
        model_tier: Model capability tier - "small" (fast/cheap), "base" (default),
                   or "full" (most capable)
        provider: "openai" or "anthropic" (defaults to settings.model_provider)

    Returns:
        Configured Model instance

    Raises:
        ConfigurationError: If provider unsupported or API key missing

    Example:
        # Use default provider's base model
        model = get_model()

        # Use small model for quick operations
        model = get_model("small")

        # Override provider
        model = get_model("base", provider="anthropic")
    """
    provider = provider or settings.model_provider

    if provider == "anthropic":
        settings.check_ready_for_anthropic()
        tier_map = {
            "base": settings.anthropic_default_model,
            "full": settings.anthropic_default_model_full,
            "small": settings.anthropic_default_model_small,
        }
        model_name = tier_map[model_tier]
        anthropic_provider = AnthropicProvider(api_key=settings.anthropic_api_key.get_secret_value())
        return AnthropicModel(model_name, provider=anthropic_provider)

    elif provider == "openai":
        settings.check_ready_for_openai()
        tier_map = {
            "base": settings.openai_default_model,
            "full": settings.openai_default_model_full,
            "small": settings.openai_default_model_small,
        }
        model_name = tier_map[model_tier]
        openai_provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
        return OpenAIModel(model_name, provider=openai_provider)

    else:
        raise ConfigurationError(f"Unsupported model provider: {provider}")


def get_openai_model(model_name: str | None = None) -> Model:
    """Get OpenAI model by specific name (deprecated - use get_model() instead).

    Args:
        model_name: Specific OpenAI model name (e.g., "gpt-5-mini", "gpt-5-nano")
                   If None, uses openai_default_model

    Returns:
        OpenAI model instance

    Deprecated: Use get_model("base"|"small"|"full") for tier-based selection
    """
    import warnings

    warnings.warn(
        "get_openai_model() is deprecated, use get_model() with tier parameter instead",
        DeprecationWarning,
        stacklevel=2,
    )
    settings.check_ready_for_openai()
    if model_name is None:
        model_name = settings.openai_default_model
    openai_provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
    return OpenAIModel(model_name, provider=openai_provider)


async def get_embedding(
    text: str,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dimensions: int = 512,
    use_cache: bool = True,
) -> list[float] | None:
    """Get embedding for a single text using OpenAI embeddings API.

    Args:
        text: Text to embed
        client: Optional OpenAI client (creates one if not provided)
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embedding (default: 512)
        use_cache: Whether to use embedding cache (default: True)

    Returns:
        Embedding vector or None if failed
    """
    # Use config setting if model not explicitly provided
    if model is None:
        model = settings.openai_embedding_model

    # Try cache lookup first if enabled
    if use_cache:
        try:
            cache = await _get_embedding_cache()
            cached_embedding = await cache.get_embedding(text, model, dimensions)
            if cached_embedding is not None:
                return cached_embedding
        except Exception as e:
            logger.warning(f"Cache lookup failed (continuing without cache): {e}")

    # Cache miss or cache disabled - call API
    if not client:
        if not settings.openai_api_key:
            return None
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

    try:
        response = await client.embeddings.create(input=text, model=model, dimensions=dimensions)
        embedding = response.data[0].embedding

        # Store in cache if enabled
        if use_cache:
            try:
                cache = await _get_embedding_cache()
                await cache.store_embedding(text, model, dimensions, embedding)
            except Exception as e:
                logger.warning(f"Cache store failed (non-fatal): {e}")

        return embedding
    except Exception as e:
        logger.warning(f"Failed to get embedding: {e}")
        return None


async def _lookup_cached_embeddings(
    texts: list[str], model: str, dimensions: int
) -> tuple[list[list[float] | None], list[str], list[int]]:
    """Look up cached embeddings and identify texts that need to be fetched.

    Args:
        texts: List of texts to look up
        model: Embedding model name
        dimensions: Embedding dimension count

    Returns:
        Tuple of (embeddings list with cached values, texts to fetch, indices to fetch)
    """
    embeddings: list[list[float] | None] = [None] * len(texts)
    texts_to_fetch: list[str] = []
    fetch_indices: list[int] = []

    try:
        cache = await _get_embedding_cache()
        cached_embeddings = await cache.get_embeddings_batch(texts, model, dimensions)

        # Separate cached from missing
        for i, cached in enumerate(cached_embeddings):
            if cached is not None:
                embeddings[i] = cached
            else:
                texts_to_fetch.append(texts[i])
                fetch_indices.append(i)

    except Exception as e:
        logger.warning(f"Batch cache lookup failed (fetching all from API): {e}")
        # On cache error, fetch everything from API
        texts_to_fetch = texts
        fetch_indices = list(range(len(texts)))

    return embeddings, texts_to_fetch, fetch_indices


async def _fetch_and_store_embeddings(
    texts_to_fetch: list[str],
    fetch_indices: list[int],
    embeddings: list[list[float] | None],
    client: AsyncOpenAI,
    model: str,
    dimensions: int,
    use_cache: bool,
) -> None:
    """Fetch embeddings from API and optionally store in cache.

    Args:
        texts_to_fetch: Texts that need embeddings
        fetch_indices: Indices in result list for fetched embeddings
        embeddings: Result list to update (modified in place)
        client: OpenAI client
        model: Embedding model name
        dimensions: Embedding dimension count
        use_cache: Whether to store results in cache
    """
    try:
        response = await client.embeddings.create(input=texts_to_fetch, model=model, dimensions=dimensions)
        fetched_embeddings = [data.embedding for data in response.data]

        # Store fetched embeddings in result list
        for idx, embedding in zip(fetch_indices, fetched_embeddings, strict=True):
            embeddings[idx] = embedding

        # Store newly fetched embeddings in cache if enabled
        if use_cache and fetched_embeddings:
            try:
                cache = await _get_embedding_cache()
                await cache.store_embeddings_batch(texts_to_fetch, model, dimensions, fetched_embeddings)
            except Exception as e:
                logger.warning(f"Batch cache store failed (non-fatal): {e}")

    except Exception as e:
        logger.error(f"Failed to get embeddings batch: {e}")
        # Leave None for failed indices


async def get_embeddings_batch(
    texts: list[str],
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    dimensions: int = 512,
    use_cache: bool = True,
) -> list[list[float] | None]:
    """Get embeddings for a batch of texts using OpenAI embeddings API.

    Args:
        texts: List of texts to embed
        client: Optional OpenAI client (creates one if not provided)
        model: Embedding model to use (default: from config settings)
        dimensions: Number of dimensions for the embeddings (default: 512)
        use_cache: Whether to use embedding cache (default: True)

    Returns:
        List of embedding vectors (or None for failed embeddings)
    """
    if not texts:
        return []

    # Use config setting if model not explicitly provided
    if model is None:
        model = settings.openai_embedding_model

    # Try cache lookup first if enabled
    if use_cache:
        embeddings, texts_to_fetch, fetch_indices = await _lookup_cached_embeddings(texts, model, dimensions)
    else:
        # Cache disabled - fetch everything
        embeddings = [None] * len(texts)
        texts_to_fetch = texts
        fetch_indices = list(range(len(texts)))

    # Fetch missing embeddings from API if needed
    if texts_to_fetch:
        if not client:
            if not settings.openai_api_key:
                return embeddings  # Return partial results with None for missing
            client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())

        await _fetch_and_store_embeddings(
            texts_to_fetch, fetch_indices, embeddings, client, model, dimensions, use_cache
        )

    return embeddings


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
