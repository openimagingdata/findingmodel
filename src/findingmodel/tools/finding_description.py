"""Finding description and detail generation tools."""

import warnings
from typing import cast

from pydantic_ai import Agent

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.finding_info import FindingInfo

from .common import get_async_perplexity_client, get_openai_model
from .prompt_template import create_prompt_messages, load_prompt_template

PROMPT_TEMPLATE_NAME = "get_finding_description"


def _render_finding_description_prompt(finding_name: str) -> tuple[str, str]:
    """Render the system instructions and user prompt for the finding description agent."""

    template = load_prompt_template(PROMPT_TEMPLATE_NAME)
    messages = create_prompt_messages(template, finding_name=finding_name)

    system_sections = [
        cast(str, msg["content"]) for msg in messages if msg.get("role") == "system" and "content" in msg
    ]
    user_sections = [cast(str, msg["content"]) for msg in messages if msg.get("role") == "user" and "content" in msg]

    if not user_sections:
        raise ValueError("Prompt template must include a user section")

    instructions = "\n\n".join(system_sections) if system_sections else ""
    user_prompt = "\n\n".join(user_sections)
    return instructions, user_prompt


async def create_info_from_name(finding_name: str, model_name: str = settings.openai_default_model) -> FindingInfo:
    """
    Create a FindingInfo object from a finding name using the OpenAI API.
    :param finding_name: The name of the finding to describe.
    :param model_name: The OpenAI model to use for the description.
    :return: A FindingInfo object containing the finding name, synonyms, and description.
    """
    settings.check_ready_for_openai()
    instructions, user_prompt = _render_finding_description_prompt(finding_name)

    agent = _create_finding_info_agent(model_name, instructions)

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


def _create_finding_info_agent(model_name: str, instructions: str) -> Agent[None, FindingInfo]:
    """Factory to build the finding info agent, extracted for easier testing overrides."""

    return Agent[None, FindingInfo](
        model=get_openai_model(model_name),
        output_type=FindingInfo,
        instructions=instructions,
    )


async def add_details_to_info(
    finding: FindingInfo, model_name: str = settings.perplexity_default_model
) -> FindingInfo | None:
    """
    Add detailed description and citations to a FindingInfo object using the Perplexity API.
    :param finding: The finding to add details to.
    :param model_name: The Perplexity model to use for the description.
    :return: A FindingInfo object containing the finding name, synonyms, description, detail, and citations.
    """
    client = get_async_perplexity_client()
    prompt_template = load_prompt_template("get_finding_detail")
    prompt_messages = create_prompt_messages(prompt_template, finding=finding)
    response = await client.chat.completions.create(
        messages=prompt_messages,
        model=model_name,
    )
    if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
        return None

    out = FindingInfo(
        name=finding.name,
        synonyms=finding.synonyms,
        description=finding.description,
        detail=response.choices[0].message.content,
    )
    if response.citations:  # type: ignore
        out.citations = response.citations  # type: ignore

    # If the detail contains any URLs, we should add them to the citations
    if out.detail and "http" in out.detail:
        if not out.citations:
            out.citations = []
        out.citations.extend([url for url in out.detail.split() if "http" in url])

    return out


# Deprecated aliases for backward compatibility
async def describe_finding_name(finding_name: str, model_name: str = settings.openai_default_model) -> FindingInfo:
    """
    DEPRECATED: Use create_info_from_name instead.
    Get a description of a finding name using the OpenAI API.
    """
    warnings.warn(
        "describe_finding_name is deprecated, use create_info_from_name instead", DeprecationWarning, stacklevel=2
    )
    return await create_info_from_name(finding_name, model_name)


async def get_detail_on_finding(
    finding: FindingInfo, model_name: str = settings.perplexity_default_model
) -> FindingInfo | None:
    """
    DEPRECATED: Use add_details_to_info instead.
    Get a detailed description of a finding using the Perplexity API.
    """
    warnings.warn(
        "get_detail_on_finding is deprecated, use add_details_to_info instead", DeprecationWarning, stacklevel=2
    )
    return await add_details_to_info(finding, model_name)


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
    return await create_info_from_name(finding_name, model_name)


async def add_details_to_finding_info(
    finding: FindingInfo, model_name: str = settings.perplexity_default_model
) -> FindingInfo | None:
    """
    DEPRECATED: Use add_details_to_info instead.
    Add detailed description and citations to a FindingInfo object using the Perplexity API.
    """
    warnings.warn(
        "add_details_to_finding_info is deprecated, use add_details_to_info instead", DeprecationWarning, stacklevel=2
    )
    return await add_details_to_info(finding, model_name)
