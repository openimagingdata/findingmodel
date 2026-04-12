"""Finding model types: FindingModelBase and FindingModelFull.

Defines the core finding model structures used throughout the system.
"""

import re
from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, Field

from findingmodel._id_gen import ID_LENGTH
from findingmodel.contributor import Organization, Person
from findingmodel.fm_md_template import UNIFIED_MARKDOWN_TEMPLATE

from .attributes import Attribute, AttributeIded, IndexCodeList, _index_codes_str
from .metadata import (
    EntityType,
    ExpectedTimeCourse,
    NormalizedAgeProfile,
    NormalizedBodyRegionList,
    NormalizedEtiologyList,
    NormalizedModalityList,
    SexSpecificity,
    Subspecialty,
    format_age_profile,
    format_time_course,
)

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
        description="Other terms that might be used to describe the finding in a radiology report",
        min_length=1,
    ),
]

TagSequence = Annotated[
    Sequence[str] | None,
    Field(
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
    body_regions: NormalizedBodyRegionList | None = None
    subspecialties: list[Subspecialty] | None = None
    etiologies: NormalizedEtiologyList | None = None
    entity_type: EntityType | None = None
    applicable_modalities: NormalizedModalityList | None = None
    expected_time_course: ExpectedTimeCourse | None = None
    age_profile: NormalizedAgeProfile | None = None
    sex_specificity: SexSpecificity | None = None
    attributes: Annotated[
        Sequence[Attribute],
        Field(min_length=1, description=ATTRIBUTES_FIELD_DESCRIPTION),
    ]

    def as_markdown(self) -> str:
        result = UNIFIED_MARKDOWN_TEMPLATE.render(
            oifm_id=None,
            show_ids=False,
            name=self.name,
            synonyms=self.synonyms,
            tags=self.tags,
            description=self.description,
            attributes=self.attributes,
            index_codes_str=None,
            entity_type=self.entity_type.value if self.entity_type else None,
            body_regions=[r.value for r in self.body_regions] if self.body_regions else None,
            applicable_modalities=[m.value for m in self.applicable_modalities] if self.applicable_modalities else None,
            subspecialties=[s.value for s in self.subspecialties] if self.subspecialties else None,
            etiologies=[e.value for e in self.etiologies] if self.etiologies else None,
            time_course_str=format_time_course(self.expected_time_course) if self.expected_time_course else None,
            age_profile_str=format_age_profile(self.age_profile) if self.age_profile else None,
            sex_specificity=self.sex_specificity.value if self.sex_specificity else None,
        ).strip()
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result


OifmIdStr = Annotated[
    str,
    Field(
        description="The ID of the finding model in the Open Imaging Data Model Finding Model registry",
        pattern=r"^OIFM_[A-Z]{3,4}_[0-9]{" + str(ID_LENGTH) + r"}$",
    ),
]

Contributor = Person | Organization


class FindingModelFull(BaseModel):
    oifm_id: OifmIdStr
    name: NameString
    description: DescriptionString
    synonyms: SynonymSequence = None
    tags: TagSequence = None
    body_regions: NormalizedBodyRegionList | None = None
    subspecialties: list[Subspecialty] | None = None
    etiologies: NormalizedEtiologyList | None = None
    entity_type: EntityType | None = None
    applicable_modalities: NormalizedModalityList | None = None
    expected_time_course: ExpectedTimeCourse | None = None
    age_profile: NormalizedAgeProfile | None = None
    sex_specificity: SexSpecificity | None = None
    anatomic_locations: IndexCodeList | None = None
    contributors: list[Contributor] | None = Field(
        default=None, description="The contributing users and organizations to the finding model"
    )
    attributes: Annotated[
        Sequence[AttributeIded],
        Field(min_length=1, description=ATTRIBUTES_FIELD_DESCRIPTION),
    ]
    # Canonical index_codes: exact matches or clinically substitutable near-equivalents only.
    # Non-exact ontology candidates belong in the enrichment review artifact.
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
            entity_type=self.entity_type.value if self.entity_type else None,
            body_regions=[r.value for r in self.body_regions] if self.body_regions else None,
            applicable_modalities=[m.value for m in self.applicable_modalities] if self.applicable_modalities else None,
            subspecialties=[s.value for s in self.subspecialties] if self.subspecialties else None,
            etiologies=[e.value for e in self.etiologies] if self.etiologies else None,
            time_course_str=format_time_course(self.expected_time_course) if self.expected_time_course else None,
            age_profile_str=format_age_profile(self.age_profile) if self.age_profile else None,
            sex_specificity=self.sex_specificity.value if self.sex_specificity else None,
            footer=footer,
        ).strip()
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    @property
    def index_codes_str(self) -> str | None:
        return _index_codes_str(self.index_codes)


FindingModelFull.__doc__ = FindingModelBase.__doc__
