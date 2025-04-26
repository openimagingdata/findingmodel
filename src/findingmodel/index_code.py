from pydantic import BaseModel, Field


class IndexCode(BaseModel):
    """Code representing an entry in a standard ontology, e.g., SNOMED or RadLex, which can be applied
    to a finding or attribute. This is used to standardize the representation of findings and attributes
    across different systems and to facilitate interoperability between different systems.
    """

    system: str = Field(description="The system that the code is from, e.g., SNOMED or RadLex.", min_length=3)
    code: str = Field(description="The code representing the entry in the standard ontology.", min_length=3)
    display: str | None = Field(
        default=None,
        description="The display name of the code in the standard ontology.",
        min_length=3,
    )

    def __str__(self) -> str:
        out = f"{self.system} {self.code}"
        if self.display:
            out += f" {self.display}"
        return out


STANDARD_CODES = {
    "presence": [IndexCode(system="SNOMED", code="705057003", display="Presence (property) (qualifier value)")],
    "absent": [
        IndexCode(system="RADLEX", code="RID28473", display="absent"),
        IndexCode(system="SNOMED", code="2667000", display="Absent (qualifier value)"),
    ],
    "present": [
        IndexCode(system="RADLEX", code="RID28472", display="present"),
        IndexCode(system="SNOMED", code="52101004", display="Present (qualifier value)"),
    ],
    "indeterminate": [
        IndexCode(system="RADLEX", code="RID39110", display="indeterminate"),
        IndexCode(system="SNOMED", code="82334004", display="Indeterminate (qualifier value)"),
    ],
    "unknown": [
        IndexCode(system="RADLEX", code="RID5655", display="unknown"),
        IndexCode(system="SNOMED", code="261665006", display="Unknown (qualifier value)"),
    ],
    "change from prior": [
        IndexCode(system="RADLEX", code="RID49896", display="change"),
        IndexCode(system="SNOMED", code="263703002", display="Changed status (qualifier value)"),
    ],
    "stable": [
        IndexCode(system="RADLEX", code="RID5734", display="stable"),
        IndexCode(system="SNOMED", code="58158008", display="Stable (qualifier value)"),
    ],
    "unchanged": [
        IndexCode(system="RADLEX", code="RID39268", display="unchanged"),
        IndexCode(system="SNOMED", code="260388006", display="No status change (qualifier value)"),
    ],
    "increased": [
        IndexCode(system="RADLEX", code="RID36043", display="increased"),
        IndexCode(system="SNOMED", code="35105006", display="Increased (qualifier value)"),
    ],
    "decreased": [
        IndexCode(system="RADLEX", code="RID36044", display="decreased"),
        IndexCode(system="SNOMED", code="1250004", display="Decreased (qualifier value)"),
    ],
    "new": [
        IndexCode(system="RADLEX", code="RID5720", display="new"),
        IndexCode(system="SNOMED", code="7147002", display="New (qualifier value)"),
    ],
    "quantity": [
        IndexCode(system="RADLEX", code="RID5761", display="quantity descriptor"),
        IndexCode(system="SNOMED", code="246205007", display="Quantity (attribute)"),
    ],
    "multiple": [
        IndexCode(system="RADLEX", code="RID5765", display="multiple"),
        IndexCode(system="SNOMED", code="255204007", display="Multiple (qualifier value)"),
    ],
    "single": [
        IndexCode(system="RADLEX", code="RID5762", display="single"),
        IndexCode(system="SNOMED", code="50607009", display="Singular (qualifier value)"),
    ],
    "severity": [IndexCode(system="SNOMED", code="246112005", display="Severity (attribute)")],
    "mild": [
        IndexCode(system="RADLEX", code="RID5671", display="mild"),
        IndexCode(system="SNOMED", code="255604002", display="Mild (qualifier value)"),
    ],
    "moderate": [
        IndexCode(system="RADLEX", code="RID5672", display="moderate"),
        IndexCode(system="SNOMED", code="1255665007", display="Moderate (qualifier value)"),
    ],
    "severe": [
        IndexCode(system="RADLEX", code="RID5673", display="severe"),
        IndexCode(system="SNOMED", code="24484000", display="Severe (severity modifier) (qualifier value)"),
    ],
    "location": [
        IndexCode(system="RADLEX", code="RID39038", display="location"),
        IndexCode(system="SNOMED", code="758637006", display="Anatomic location (property) (qualifier value)"),
    ],
    "size": [
        IndexCode(system="SNOMED", code="246115007", display="Size (attribute)"),
        IndexCode(system="RADLEX", code="RID5772", display="size descriptor"),
    ],
    "larger": [
        IndexCode(system="RADLEX", code="RID5791", display="enlarged"),
        IndexCode(system="SNOMED", code="263768009", display="Greater (qualifier value)"),
    ],
    "smaller": [
        IndexCode(system="RADLEX", code="RID38669", display="diminished"),
        IndexCode(system="SNOMED", code="263796003", display="Lesser (qualifier value)"),
    ],
}
