from typing import Literal, Protocol, Sequence

from findingmodel.finding_model import AttributeType


class AbstractNumericAttribute(Protocol):
    """A protocol for the common features of the NumericAttribute and NumericAttributeIded classes."""

    name: str
    description: str | None
    type: Literal[AttributeType.NUMERIC]
    minimum: int | float | None
    maximum: int | float | None
    unit: str | None
    required: bool


class AbstractChoiceValue(Protocol):
    """A protocol for the common features of the ChoiceValue and ChoiceValueIded classes."""

    name: str
    description: str | None


class AbstractChoiceAttribute(Protocol):
    """A protocol for the common features of the ChoiceAttribute and ChoiceAttributeIded classes."""

    name: str
    description: str | None
    type: Literal[AttributeType.CHOICE]
    values: Sequence[AbstractChoiceValue]
    required: bool
    max_selected: int


class AbstractFindingModel(Protocol):
    """A protocol for the common features of the FindingModelBase and FindingModelFull classes."""

    name: str
    description: str
    synonyms: Sequence[str] | None
    tags: Sequence[str] | None
    attributes: Sequence[AbstractChoiceAttribute | AbstractNumericAttribute]

    def as_markdown(self) -> str:
        """Render the finding model as a markdown string."""
        ...
