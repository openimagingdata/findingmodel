from pathlib import Path
from typing import Protocol, Sequence

from instructor import AsyncInstructor, from_openai
from openai import AsyncOpenAI

from findingmodel.index_code import STANDARD_CODES

from .config import settings
from .finding_info import FindingInfo
from .finding_model import (
    ChoiceAttribute,
    ChoiceAttributeIded,
    ChoiceValue,
    FindingModelBase,
    FindingModelFull,
    IndexCodeList,
    generate_oifm_id,
    generate_oifma_id,
)
from .prompt_template import create_prompt_messages, load_prompt_template


def get_async_instructor_client() -> AsyncInstructor:
    settings.check_ready_for_openai()
    return from_openai(AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()))


def get_async_perplexity_client() -> AsyncOpenAI:
    settings.check_ready_for_perplexity()
    return AsyncOpenAI(
        api_key=str(settings.perplexity_api_key.get_secret_value()), base_url=str(settings.perplexity_base_url)
    )


async def describe_finding_name(finding_name: str, model_name: str = settings.openai_default_model) -> FindingInfo:
    """
    Get a description of a finding name using the OpenAI API.
    :param finding_name: The name of the finding to describe.
    :param model_name: The OpenAI model to use for the description.
    :return: A BaseFindingInfo object containing the finding name, synonyms, and description.
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


async def get_detail_on_finding(
    finding: FindingInfo, model_name: str = settings.perplexity_default_model
) -> FindingInfo | None:
    """
    Get a detailed description of a finding using the Perplexity API.
    :param finding: The finding to describe.
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


async def create_finding_model_from_markdown(
    finding_info: FindingInfo,
    /,
    markdown_path: str | Path | None = None,
    markdown_text: str | None = None,
    openai_model: str = settings.openai_default_model,
) -> FindingModelBase:
    """
    Create a finding model from a markdown file or text using the OpenAI API.
    :param finding: The finding information or name to use for the model.
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


def create_finding_model_stub_from_finding_info(
    finding_info: FindingInfo, tags: list[str] | None = None
) -> FindingModelBase:
    """
    Create a finding model stub from a FindingInfo object.
    :param finding_info: The FindingInfo object to use for the model.
    :param tags: Optional tags to add to the finding model.
    :return: A FindingModelBase object containing the finding model stub.
    """
    finding_name = finding_info.name.lower()

    def create_presence_element(finding_name: str) -> ChoiceAttribute:
        return ChoiceAttribute(
            name="presence",
            description=f"Presence or absence of {finding_name}",
            values=[
                ChoiceValue(name="absent", description=f"{finding_name.capitalize()} is absent"),
                ChoiceValue(name="present", description=f"{finding_name.capitalize()} is present"),
                ChoiceValue(name="indeterminate", description=f"Presence of {finding_name} cannot be determined"),
                ChoiceValue(name="unknown", description=f"Presence of {finding_name} is unknown"),
            ],
        )

    def create_change_element(finding_name: str) -> ChoiceAttribute:
        return ChoiceAttribute(
            name="change from prior",
            description=f"Whether and how a {finding_name} has changed over time",
            values=[
                ChoiceValue(name="unchanged", description=f"{finding_name.capitalize()} is unchanged"),
                ChoiceValue(name="stable", description=f"{finding_name.capitalize()} is stable"),
                ChoiceValue(name="new", description=f"{finding_name.capitalize()} is new"),
                ChoiceValue(
                    name="resolved", description=f"{finding_name.capitalize()} seen on a prior exam has resolved"
                ),
                ChoiceValue(name="increased", description=f"{finding_name.capitalize()} has increased"),
                ChoiceValue(name="decreased", description=f"{finding_name.capitalize()} has decreased"),
                ChoiceValue(name="larger", description=f"{finding_name.capitalize()} is larger"),
                ChoiceValue(name="smaller", description=f"{finding_name.capitalize()} is smaller"),
            ],
        )

    stub = FindingModelBase(
        name=finding_name,
        description=finding_info.description,
        synonyms=finding_info.synonyms,
        attributes=[
            create_presence_element(finding_name),
            create_change_element(finding_name),
        ],
    )
    if tags:
        stub.tags = tags
    return stub


def add_ids_to_finding_model(
    finding_model: FindingModelBase,
    source: str,
) -> FindingModelFull:
    """
    Generate and add OIFM IDs to the ID-less finding models with a source code.
    :param finding_model: The finding model to add IDs to.
    :param source: 3-4 letter code for the originating organization.
    :return: The finding model with IDs added.
    """

    finding_model_dict = finding_model.model_dump()
    if "oifm_id" not in finding_model_dict:
        finding_model_dict["oifm_id"] = generate_oifm_id(source)
    for attribute in finding_model_dict["attributes"]:
        if "oifma_id" not in attribute:
            attribute["oifma_id"] = generate_oifma_id(source)

    return FindingModelFull.model_validate(finding_model_dict)


class Codeable(Protocol):
    index_codes: IndexCodeList | None


def _add_index_codes(target: Codeable, name: str) -> None:
    found_codes = STANDARD_CODES.get(name, None)
    if not found_codes or not isinstance(found_codes, Sequence) or len(found_codes) == 0:
        return
    if target.index_codes is None:
        target.index_codes = []
    assert isinstance(target.index_codes, list)
    have_codes = {f"{code.system}:{code.code}" for code in target.index_codes}
    for code in found_codes:
        if f"{code.system}:{code.code}" in have_codes:
            continue
        target.index_codes.append(code)
        have_codes.add(f"{code.system}:{code.code}")


def add_standard_codes_to_finding_model(finding_model: FindingModelFull) -> None:
    """
    Add standard codes to the finding model.
    :param finding_model: The finding model to add standard codes to.
    :return: None
    """
    for attribute in finding_model.attributes:
        _add_index_codes(attribute, attribute.name.lower())
        if isinstance(attribute, ChoiceAttributeIded):
            for value in attribute.values:
                _add_index_codes(value, value.name.lower())
