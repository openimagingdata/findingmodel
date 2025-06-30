"""Finding description and detail generation tools."""

import warnings

from findingmodel.config import settings
from findingmodel.finding_info import FindingInfo

from .common import get_async_instructor_client, get_async_perplexity_client
from .prompt_template import create_prompt_messages, load_prompt_template


async def create_info_from_name(finding_name: str, model_name: str = settings.openai_default_model) -> FindingInfo:
    """
    Create a FindingInfo object from a finding name using the OpenAI API.
    :param finding_name: The name of the finding to describe.
    :param model_name: The OpenAI model to use for the description.
    :return: A FindingInfo object containing the finding name, synonyms, and description.
    """
    client = get_async_instructor_client()
    prompt_template = load_prompt_template("get_finding_description")
    messages = create_prompt_messages(prompt_template, finding_name=finding_name)
    result = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        response_model=FindingInfo,
    )
    assert isinstance(result, FindingInfo), "Finding description not returned."
    return result


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
