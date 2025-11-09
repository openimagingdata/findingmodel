"""Finding description and detail generation tools."""

import warnings
from typing import Literal

from pydantic_ai import Agent

from findingmodel import logger
from findingmodel.config import ModelProvider, ModelTier, settings
from findingmodel.finding_info import FindingInfo

from .common import get_async_tavily_client, get_model
from .prompt_template import load_prompt_template, render_agent_prompt

PROMPT_TEMPLATE_NAME = "get_finding_description"


async def create_info_from_name(
    finding_name: str,
    model_tier: ModelTier = "small",
    provider: ModelProvider | None = None,
) -> FindingInfo:
    """
    Create a FindingInfo object from a finding name using the AI API.
    :param finding_name: The name of the finding to describe.
    :param model_tier: The model tier to use ("small", "base", or "full").
    :param provider: The model provider to use ("openai" or "anthropic"). If None, uses default from settings.
    :return: A FindingInfo object containing the finding name, synonyms, and description.
    """
    template = load_prompt_template(PROMPT_TEMPLATE_NAME)
    instructions, user_prompt = render_agent_prompt(template, finding_name=finding_name)

    agent = _create_finding_info_agent(model_tier, instructions, provider=provider)

    result = await agent.run(user_prompt)
    finding_info = _normalize_finding_info(result.output, original_input=finding_name)

    if finding_info.name != finding_name:
        logger.info(f"Normalized finding name from '{finding_name}' to '{finding_info.name}'")

    return finding_info


def _normalize_finding_info(finding_info: FindingInfo, *, original_input: str) -> FindingInfo:
    """Trim whitespace, deduplicate synonyms, and ensure the original term is preserved when renamed."""

    cleaned_name = finding_info.name.strip()
    name_key = cleaned_name.casefold()

    synonyms = finding_info.synonyms or []
    seen: set[str] = set()
    normalized_synonyms: list[str] = []
    for synonym in synonyms:
        cleaned_synonym = synonym.strip()
        if not cleaned_synonym:
            continue
        key = cleaned_synonym.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized_synonyms.append(cleaned_synonym)

    original_term = original_input.strip()
    if original_term:
        original_key = original_term.casefold()
        if original_key != name_key and original_key not in seen:
            normalized_synonyms.append(original_term)

    updated_synonyms = normalized_synonyms or None

    if cleaned_name == finding_info.name and updated_synonyms == finding_info.synonyms:
        return finding_info

    return finding_info.model_copy(update={"name": cleaned_name, "synonyms": updated_synonyms})


def _create_finding_info_agent(
    model_tier: ModelTier,
    instructions: str,
    provider: ModelProvider | None = None,
) -> Agent[None, FindingInfo]:
    """Factory to build the finding info agent, extracted for easier testing overrides."""

    return Agent[None, FindingInfo](
        model=get_model(model_tier, provider=provider),
        output_type=FindingInfo,
        instructions=instructions,
    )


# Trusted radiology and medical imaging domains for Tavily searches
RADIOLOGY_DOMAINS = [
    "radiopaedia.org",
    "radiologyassistant.nl",
    "learningradiology.com",
    "ctisus.com",
    "wikipedia.org",
    "radiologycafe.com",
]


async def add_details_to_info(
    finding: FindingInfo, search_depth: Literal["basic", "advanced"] = settings.tavily_search_depth
) -> FindingInfo | None:
    """
    Add detailed description and citations to a FindingInfo object using a Pydantic AI agent with Tavily search.

    The agent searches trusted radiology domains and synthesizes information focusing on:
    - Characteristics used to identify and describe the finding
    - Anatomic locations where the finding is typically seen
    - Medical terminology and imaging features

    :param finding: The finding to add details to.
    :param search_depth: The Tavily search depth to use ('basic' or 'advanced').
    :return: A FindingInfo object with added detail and citations, or None if search fails.
    """
    from pydantic_ai import Agent, RunContext

    # Define the tool for Tavily search with domain filtering
    async def search_radiology_sources(ctx: RunContext[FindingInfo], query: str) -> str:
        """Search trusted radiology and medical imaging sources for information about a finding.

        Args:
            query: The search query to execute

        Returns:
            Combined search results with source URLs
        """
        client = get_async_tavily_client()
        response = await client.search(
            query=query,
            search_depth=search_depth,
            max_results=10,
            include_domains=RADIOLOGY_DOMAINS,
        )

        if not response or not response.get("results"):
            return "No results found"

        # Format results with content and sources
        results = []
        for result in response["results"]:
            content = result.get("content", "")
            url = result.get("url", "")
            if content:
                results.append(f"{content}\n\nSource: {url}")

        return "\n\n---\n\n".join(results) if results else "No relevant information found"

    # Load the finding detail prompt template
    template = load_prompt_template("get_finding_detail")
    instructions, user_prompt = render_agent_prompt(
        template,
        finding={
            "finding_name": finding.name,
            "description": finding.description or "",
            "synonyms": ", ".join(finding.synonyms) if finding.synonyms else "none",
        },
    )

    # Create agent with search tool
    agent = Agent[FindingInfo, str](
        get_model("small"),
        deps_type=FindingInfo,
        output_type=str,
        tools=[search_radiology_sources],
        instructions=instructions,
    )

    # Run the agent
    result = await agent.run(user_prompt, deps=finding)

    # Extract citations from tool call messages
    import re

    citations: list[str] = []
    for message in result.all_messages():
        # Look for Source: URLs in any message content
        content_str = str(message)
        # Match URLs - stop at actual whitespace, newlines (including \n literals), or end markers
        urls = re.findall(r"Source: (https?://[^\s\\]+)", content_str)
        citations.extend(urls)

    # Remove duplicates while preserving order
    unique_citations: list[str] = []
    seen: set[str] = set()
    for url in citations:
        if url not in seen:
            seen.add(url)
            unique_citations.append(url)

    if not result.output:
        return None

    return FindingInfo(
        name=finding.name,
        synonyms=finding.synonyms,
        description=finding.description,
        detail=result.output,
        citations=unique_citations if unique_citations else None,
    )


# Deprecated aliases for backward compatibility
async def describe_finding_name(finding_name: str, model_name: str = settings.openai_default_model) -> FindingInfo:
    """
    DEPRECATED: Use create_info_from_name instead.
    Get a description of a finding name using the OpenAI API.
    """
    warnings.warn(
        "describe_finding_name is deprecated, use create_info_from_name instead", DeprecationWarning, stacklevel=2
    )
    # Map model_name to tier for backward compatibility
    if model_name == settings.openai_default_model_small:
        tier: ModelTier = "small"
    elif model_name == settings.openai_default_model_full:
        tier = "full"
    else:
        tier = "base"
    return await create_info_from_name(finding_name, tier)


async def get_detail_on_finding(
    finding: FindingInfo, search_depth: Literal["basic", "advanced"] = settings.tavily_search_depth
) -> FindingInfo | None:
    """
    DEPRECATED: Use add_details_to_info instead.
    Get a detailed description of a finding using the Tavily search API.
    """
    warnings.warn(
        "get_detail_on_finding is deprecated, use add_details_to_info instead", DeprecationWarning, stacklevel=2
    )
    return await add_details_to_info(finding, search_depth)


# Additional deprecated aliases for the intermediate names
async def create_finding_info_from_name(
    finding_name: str, model_name: str = settings.openai_default_model
) -> FindingInfo:
    """
    DEPRECATED: Use create_info_from_name instead.
    Create a FindingInfo object from a finding name using the OpenAI API.
    """
    warnings.warn(
        "create_finding_info_from_name is deprecated, use create_info_from_name instead",
        DeprecationWarning,
        stacklevel=2,
    )
    # Map model_name to tier for backward compatibility
    if model_name == settings.openai_default_model_small:
        tier: ModelTier = "small"
    elif model_name == settings.openai_default_model_full:
        tier = "full"
    else:
        tier = "base"
    return await create_info_from_name(finding_name, tier)


async def add_details_to_finding_info(
    finding: FindingInfo, search_depth: Literal["basic", "advanced"] = settings.tavily_search_depth
) -> FindingInfo | None:
    """
    DEPRECATED: Use add_details_to_info instead.
    Add detailed description and citations to a FindingInfo object using the Tavily search API.
    """
    warnings.warn(
        "add_details_to_finding_info is deprecated, use add_details_to_info instead", DeprecationWarning, stacklevel=2
    )
    return await add_details_to_info(finding, search_depth)
