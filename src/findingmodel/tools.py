import random
from pathlib import Path

from instructor import AsyncInstructor, from_openai
from openai import AsyncOpenAI

from .config import settings
from .finding_info import FindingInfo
from .finding_model import (
    ID_LENGTH,
    ChoiceAttribute,
    ChoiceValue,
    FindingModelBase,
    FindingModelFull,
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


async def create_finding_model_from_markdown(
    finding: FindingInfo | str,
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
    if isinstance(finding, str):
        finding_info = await describe_finding_name(finding)
    elif isinstance(finding, FindingInfo):
        finding_info = finding
    else:
        raise TypeError("Must get either a finding name or a FindingInfo object")
    if not markdown_path and not markdown_text:
        raise ValueError("Either markdown_path or markdown_text must be provided")
    if markdown_path and markdown_text:
        raise ValueError("Only one of markdown_path or markdown_text should be provided")
    if markdown_path:
        markdown_path = Path(markdown_path)
        if not markdown_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
        markdown_text = markdown_path.read_text()
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
                ChoiceValue(name="increased", description=f"{finding_name.capitalize()} has increased"),
                ChoiceValue(name="decreased", description=f"{finding_name.capitalize()} has decreased"),
                ChoiceValue(name="new", description=f"{finding_name.capitalize()} is new"),
                ChoiceValue(
                    name="resolved", description=f"{finding_name.capitalize()} seen on a prior exam has resolved"
                ),
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


STANDARD_CODES = [
    {"name": "presence", "snomed_code": "705057003", "snomed_display": "Presence (property) (qualifier value)"},
    {
        "name": "absent",
        "radlex_code": "RID28473",
        "radlex_display": "absent",
        "snomed_code": "2667000",
        "snomed_display": "Absent (qualifier value)",
    },
    {
        "name": "present",
        "radlex_code": "RID28472",
        "radlex_display": "present",
        "snomed_code": "52101004",
        "snomed_display": "Present (qualifier value)",
    },
    {
        "name": "indeterminate",
        "radlex_code": "RID39110",
        "radlex_display": "indeterminate",
        "snomed_code": "82334004",
        "snomed_display": "Indeterminate (qualifier value)",
    },
    {
        "name": "unknown",
        "radlex_code": "RID5655",
        "radlex_display": "unknown",
        "snomed_code": "261665006",
        "snomed_display": "Unknown (qualifier value)",
    },
    {
        "name": "location",
        "radlex_code": "RID39038",
        "radlex_display": "location",
        "snomed_code": "758637006",
        "snomed_display": "Anatomic location (property) (qualifier value)",
    },
    {"name": "size", "snomed_code": "246115007", "snomed_display": "Size (attribute)"},
    {"name": "changed", "snomed_code": "263703002", "snomed_display": "Changed status (qualifier value)"},
    {
        "name": "stable",
        "radlex_code": "RID5734",
        "radlex_display": "stable",
        "snomed_code": "58158008",
        "snomed_display": "Stable (qualifier value)",
    },
    {
        "name": "unchanged",
        "radlex_code": "RID39268",
        "radlex_display": "unchanged",
        "snomed_code": "260388006",
        "snomed_display": "No status change (qualifier value)",
    },
    {
        "name": "increased",
        "radlex_code": "RID36043",
        "radlex_display": "increased",
        "snomed_code": "35105006",
        "snomed_display": "Increased (qualifier value)",
    },
    {
        "name": "decreased",
        "radlex_code": "RID36044",
        "radlex_display": "decreased",
        "snomed_code": "1250004",
        "snomed_display": "Decreased (qualifier value)",
    },
    {
        "name": "new",
        "radlex_code": "RID5720",
        "radlex_display": "new",
        "snomed_code": "7147002",
        "snomed_display": "New (qualifier value)",
    },
    {
        "name": "quantity",
        "radlex_code": "RID5761",
        "radlex_display": "quantity descriptor",
        "snomed_code": "246205007",
        "snomed_display": "Quantity (attribute)",
    },
    {
        "name": "multiple",
        "radlex_code": "RID5765",
        "radlex_display": "multiple",
        "snomed_code": "255204007",
        "snomed_display": "Multiple (qualifier value)",
    },
    {
        "name": "single",
        "radlex_code": "RID5762",
        "radlex_display": "single",
        "snomed_code": "50607009",
        "snomed_display": "Singular (qualifier value)",
    },
    {"name": "severity", "snomed_code": "246112005", "snomed_display": "Severity (attribute)"},
    {
        "name": "mild",
        "radlex_code": "RID5671",
        "radlex_display": "mild",
        "snomed_code": "255604002",
        "snomed_display": "Mild (qualifier value)",
    },
    {
        "name": "moderate",
        "radlex_code": "RID5672",
        "radlex_display": "moderate",
        "snomed_code": "1255665007",
        "snomed_display": "Moderate (qualifier value)",
    },
    {
        "name": "severe",
        "radlex_code": "RID5673",
        "radlex_display": "severe",
        "snomed_code": "24484000",
        "snomed_display": "Severe (severity modifier) (qualifier value)",
    },
    {
        "name": "change",
        "radlex_code": "RID49896",
        "radlex_display": "change",
        "snomed_code": "243326001",
        "snomed_display": "Changing (qualifier value)",
    },
    {
        "name": "location",
        "radlex_code": "RID39038",
        "radlex_display": "location",
        "snomed_code": "758637006",
        "snomed_display": "Anatomic location (property) (qualifier value)",
    },
]


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

    def random_digits(length: int) -> str:
        return "".join([str(random.randint(0, 9)) for _ in range(length)])

    def generate_oifm_id(source: str) -> str:
        return f"OIFM_{source.upper()}_{random_digits(ID_LENGTH)}"

    def generate_oifma_id(source: str) -> str:
        return f"OIFMA_{source.upper()}_{random_digits(ID_LENGTH)}"

    finding_model_dict = finding_model.model_dump()
    if "oifm_id" not in finding_model_dict:
        finding_model_dict["oifm_id"] = generate_oifm_id(source)
    for attribute in finding_model_dict["attributes"]:
        if "oifma_id" not in attribute:
            attribute["oifma_id"] = generate_oifma_id(source)

    return FindingModelFull.model_validate(finding_model_dict)
