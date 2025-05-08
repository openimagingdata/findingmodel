import random
import re
from enum import Enum
from typing import Annotated, Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field, model_validator

from findingmodel.contributor import Organization, Person

from .fm_md_template import UNIFIED_MARKDOWN_TEMPLATE
from .index_code import IndexCode

# TODO: Pull in the tools in common.py
# TODO: Move random digits and ID generation to common.py
# TODO: Add `slug_name` properties to the models, attributes, and choice values
# TODO: Add `index_codes` field to the base model class as well
# TODO: Move all the Markdown code to the markdown module


class AttributeType(str, Enum):
    CHOICE = "choice"
    NUMERIC = "numeric"


# Define a ID length constant
ID_LENGTH = 6

AttributeId = Annotated[
    str,
    Field(
        description="The ID of the attribute in the Open Imaging Data Model Finding Model registry",
        pattern=r"^OIFMA_[A-Z]{3,4}_[0-9]{" + str(ID_LENGTH) + r"}$",
    ),
]


def _random_digits(length: int) -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def generate_oifma_id(source: str) -> str:
    return f"OIFMA_{source.upper()}_{_random_digits(ID_LENGTH)}"


class ChoiceValue(BaseModel):
    """A value that a radiologist might choose for a choice attribute. For example, the severity of a finding might be
    severe, or the shape of a finding might be oval."""

    name: str
    description: str | None = None


AttributeValueCode = Annotated[
    str,
    Field(
        description="The code for the value in the Open Imaging Data Model Finding Model registry",
        pattern=r"^OIFMA_[A-Z]{3,4}_[0-9]{" + str(ID_LENGTH) + r"}\.\d+$",
    ),
]

IndexCodeList = Annotated[
    list[IndexCode],
    Field(
        default=None,
        description="References to concepts in standad ontologies to facilitate interoperability between systems.",
        min_length=1,
    ),
]


def _index_codes_str(index_codes: Iterable[IndexCode] | None) -> str | None:
    return "; ".join([str(code) for code in index_codes]) if index_codes else None


class ChoiceValueIded(BaseModel):
    """A value that a radiologist might choose for a choice attribute. For example, the severity of a finding might be
    severe, or the shape of a finding might be oval."""

    value_code: AttributeValueCode
    name: str
    description: str | None = None
    index_codes: IndexCodeList | None = None

    # @computed_field  # type: ignore[prop-decorator]
    @property
    def index_codes_str(self) -> str | None:
        return _index_codes_str(self.index_codes)


AttributeNameStr = Annotated[
    str,
    Field(
        description="Short, descriptive name of the attribute as an case-sensitive English phrase (preserving capitalization of acronyms).",
        min_length=3,
        max_length=100,
    ),
]

AttributeDescriptionStr = Annotated[
    str | None,
    Field(
        default=None,
        description="A one-to-two sentence description of the attribute that might be included in a medical textbook",
        min_length=5,
        max_length=500,
    ),
]

RequiredBool = Annotated[
    bool,
    Field(
        default=False,
        description="Whether the attribute is used every time a radiologist describes the finding",
    ),
]

MaxSelectedInt = Annotated[
    int,
    Field(
        default=1,
        description="The maximum number of values that can be selected for a choice attribute (defaults to 1).",
        ge=1,
    ),
]


def fix_max_selected_validator(cls, data: dict[str, Any]) -> dict[str, Any]:  # type: ignore # noqa: ANN001
    """Fix the max_selected value to be an integer if it is set to "all"
    or greater than the number of actual choices
    """
    if not data.get("max_selected"):
        data["max_selected"] = 1
    if data.get("max_selected") == "all" or data.get("max_selected") > len(data["values"]):  # type: ignore
        data["max_selected"] = len(data["values"])
    return data


class ChoiceAttribute(BaseModel):
    """An attribute of a radiology finding where the radiologist would choose from a list of options. For example,
    the severity of a finding (mild, moderate, or severe), or the shape of a finding (round, oval, or
    irregular). For attributes which can have multiple values from a series of choices, max_selected can be set to"
    a value greater than 1 or "all"."""

    name: AttributeNameStr
    description: AttributeDescriptionStr = None
    type: Literal[AttributeType.CHOICE] = AttributeType.CHOICE
    values: Annotated[list[ChoiceValue], Field(..., min_length=2)]
    required: RequiredBool = False
    max_selected: MaxSelectedInt = 1

    @model_validator(mode="before")
    @classmethod
    def fix_max_selected(cls, data):  # type: ignore  # noqa: ANN001, ANN206
        return fix_max_selected_validator(cls, data)


class ChoiceAttributeIded(BaseModel):
    oifma_id: AttributeId
    name: AttributeNameStr
    description: AttributeDescriptionStr = None
    type: Literal[AttributeType.CHOICE] = AttributeType.CHOICE
    values: Annotated[list[ChoiceValueIded], Field(..., min_length=2)]
    required: RequiredBool = False
    max_selected: MaxSelectedInt = 1
    index_codes: IndexCodeList | None = None

    @model_validator(mode="before")
    @classmethod
    def fix_max_selected(cls, data):  # type: ignore  # noqa: ANN001, ANN206
        return fix_max_selected_validator(cls, data)

    @model_validator(mode="before")
    @classmethod
    def add_value_codes(cls, data):  # type: ignore # noqa: ANN001, ANN206
        """Add the value codes to the choice values if they are not already present."""
        oifma_id = data.get("oifma_id")
        if oifma_id is None:
            raise ValueError("Cannot generate value codes without an OIFMA ID")
        new_values = []
        for i, value_def in enumerate(data["values"]):
            value_code = {"value_code": f"{oifma_id}.{i}"}
            if isinstance(value_def, (ChoiceValue, ChoiceValueIded)):
                new_value = value_def.model_dump() | value_code
            elif isinstance(value_def, dict):
                new_value = value_def | value_code
            else:
                raise ValueError("Invalid value definition")
            new_values.append(new_value)
        data["values"] = new_values
        return data

    # @computed_field  # type: ignore[prop-decorator]
    @property
    def index_codes_str(self) -> str | None:
        return _index_codes_str(self.index_codes)


ChoiceAttributeIded.__doc__ = ChoiceAttribute.__doc__

MinimumNumeric = Annotated[
    int | float | None,
    Field(
        default=None,
        description="The minimum value for the attribute.",
    ),
]

MaximumNumeric = Annotated[
    int | float | None,
    Field(
        default=None,
        description="The maximum value for the attribute.",
    ),
]

UnitStr = Annotated[
    str | None,
    Field(
        default=None,
        description="The unit of measure for the attribute",
    ),
]


class NumericAttribute(BaseModel):
    """An attribute of a radiology finding where the radiologist would choose a number from a range. For example, the
    size of a finding might be up to 10 cm or the number of findings might be between 1 and 10."""

    name: AttributeNameStr
    description: AttributeDescriptionStr = None
    type: Literal[AttributeType.NUMERIC] = AttributeType.NUMERIC
    minimum: MinimumNumeric = None
    maximum: MaximumNumeric = None
    unit: UnitStr = None
    required: RequiredBool = False


class NumericAttributeIded(BaseModel):
    oifma_id: AttributeId
    name: AttributeNameStr
    description: AttributeDescriptionStr = None
    type: Literal[AttributeType.NUMERIC] = AttributeType.NUMERIC
    minimum: MinimumNumeric = None
    maximum: MaximumNumeric = None
    unit: UnitStr = None
    required: RequiredBool = False
    index_codes: IndexCodeList | None = None

    # @computed_field  # type: ignore[prop-decorator]
    @property
    def index_codes_str(self) -> str | None:
        return _index_codes_str(self.index_codes)


NumericAttributeIded.__doc__ = NumericAttribute.__doc__

ATTRIBUTE_FIELD_DESCRIPTION = (
    "An attribute that a radiologist would use to characterize a particular finding in a radiology report"
)
Attribute = Annotated[
    ChoiceAttribute | NumericAttribute,
    Field(
        discriminator="type",
        description=ATTRIBUTE_FIELD_DESCRIPTION,
    ),
]

AttributeIded = Annotated[
    ChoiceAttributeIded | NumericAttributeIded,
    Field(discriminator="type", description=ATTRIBUTE_FIELD_DESCRIPTION),
]

NameString = Annotated[
    str,
    Field(
        description="The name of the finding model. This should be a short, descriptive name that is easy to remember",
        min_length=5,
    ),
]

DescriptionString = Annotated[
    str,
    Field(
        description="A one-to-two sentence description of the finding that might be included in a textbook",
        min_length=5,
    ),
]

SynonymSequence = Annotated[
    Sequence[str] | None,
    Field(
        default=None,
        description="Other terms that might be used to describe the finding in a radiology report",
        min_length=1,
    ),
]

TagSequence = Annotated[
    Sequence[str] | None,
    Field(
        default=None,
        description="Tags that might be used to categorize the finding among other findings",
        min_length=1,
    ),
]

ATTRIBUTES_FIELD_DESCRIPTION = (
    "The attributes a radiologist would use to characterize a particular finding in a radiology report"
)


class FindingModelBase(BaseModel):
    """The definition of a radiology finding what the finding is such as might be included in a textbook
    along with definitions of the relevant attributes that a radiologist might use to characterize the finding in a
    radiology report."""

    name: NameString
    description: DescriptionString
    synonyms: SynonymSequence = None
    tags: TagSequence = None
    attributes: Annotated[
        Sequence[Attribute],
        Field(min_length=1, description=ATTRIBUTES_FIELD_DESCRIPTION),
    ]

    def as_markdown(self) -> str:
        return UNIFIED_MARKDOWN_TEMPLATE.render(
            oifm_id=None,
            show_ids=False,
            name=self.name,
            synonyms=self.synonyms,
            tags=self.tags,
            description=self.description,
            attributes=self.attributes,
            index_code_str=None,
        )


OifmIdStr = Annotated[
    str,
    Field(
        description="The ID of the finding model in the Open Imaging Data Model Finding Model registry",
        pattern=r"^OIFM_[A-Z]{3,4}_[0-9]{" + str(ID_LENGTH) + r"}$",
    ),
]


def generate_oifm_id(source: str) -> str:
    return f"OIFM_{source.upper()}_{_random_digits(ID_LENGTH)}"


Contributor = Person | Organization


class FindingModelFull(BaseModel):
    oifm_id: OifmIdStr
    name: NameString
    description: DescriptionString
    synonyms: SynonymSequence = None
    tags: TagSequence = None
    contributors: list[Contributor] | None = Field(
        default=None, description="The contributing users and organizations to the finding model"
    )
    attributes: Annotated[
        Sequence[AttributeIded],
        Field(min_length=1, description=ATTRIBUTES_FIELD_DESCRIPTION),
    ]
    index_codes: IndexCodeList | None = None

    def as_markdown(self, hide_ids: bool = False) -> str:
        footer: str | None = None
        if self.contributors:
            footer_lines = []
            for contributor in self.contributors:
                match contributor:
                    case Organization():
                        line = (
                            f"- [{contributor.name}]({contributor.url}) ({contributor.code})"
                            if contributor.url
                            else f"- {contributor.name} ({contributor.code})"
                        )
                        footer_lines.append(line)
                    case Person():
                        line = f"- {contributor.name} ({contributor.organization_code}) — [Email](mailto:{contributor.email})"
                        if contributor.url:
                            line += f" — [Link]({contributor.url})"
                        footer_lines.append(line)
                    case _:
                        raise ValueError("Invalid contributor type")
            footer = "\n\n---\n\n**Contributors**\n\n" + "\n".join(footer_lines)

        result = UNIFIED_MARKDOWN_TEMPLATE.render(
            oifm_id=self.oifm_id,
            show_ids=not hide_ids,
            name=self.name,
            synonyms=self.synonyms,
            tags=self.tags,
            description=self.description,
            attributes=self.attributes,
            index_codes_str=self.index_codes_str,
            footer=footer,
        ).strip()
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    @property
    def index_codes_str(self) -> str | None:
        return _index_codes_str(self.index_codes)


FindingModelFull.__doc__ = FindingModelBase.__doc__
