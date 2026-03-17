from pydantic import BaseModel, Field


class IndexCode(BaseModel):
    """Simple ontology identifier: system, code, and optional display name.

    IndexCode is a plain identifier object — it carries no relationship semantics (e.g., broader,
    narrower, related). When used on canonical FindingModel.index_codes, each code must be an exact
    match or clinically substitutable near-equivalent for the full model concept. Merely related,
    broader, narrower, or temporally qualified codes belong in the enrichment review artifact, not
    on the canonical model. If relationship-bearing ontology links are needed in the future, use a
    separate typed wrapper rather than overloading IndexCode.
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
