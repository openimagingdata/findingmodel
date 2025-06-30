from pydantic import BaseModel, Field


class IndexCode(BaseModel):
    """Code representing an entry in a standard ontology, e.g., SNOMED or RadLex, which can be applied
    to a finding or attribute. This is used to standardize the representation of findings and attributes
    across different systems and to facilitate interoperability between different systems.
    """

    system: str = Field(description="The system that the code is from, e.g., SNOMED or RadLex.", min_length=3)
    code: str = Field(description="The code representing the entry in the standard ontology.", min_length=2)
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
