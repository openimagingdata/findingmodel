"""Protocol definitions for findingmodel interfaces.

This module defines common protocols and interfaces used across findingmodel packages.
Protocol definitions are centralized here to avoid circular dependencies.
"""

from typing import Any, Protocol

from oidm_common.models import IndexCode
from pydantic import BaseModel

# Table name to index code system mapping
TABLE_TO_INDEX_CODE_SYSTEM = {
    "anatomic_locations": "ANATOMICLOCATIONS",
    "radlex": "RADLEX",
    "snomedct": "SNOMEDCT",
    "loinc": "LOINC",
    "icd10cm": "ICD10CM",
    "gamuts": "GAMUTS",
    "cpt": "CPT",
}


def normalize_concept(text: str) -> str:
    """
    Normalize concept text for deduplication by removing semantic tags and trailing parenthetical content.

    Args:
        text: Original concept text

    Returns:
        Normalized text for comparison
    """
    # Take only the first line if multi-line
    normalized = text.split("\n")[0]

    # Remove everything after colon (common in RadLex results like "berry aneurysm: description...")
    if ":" in normalized:
        normalized = normalized.split(":")[0]

    # Remove TRAILING parenthetical content only (e.g., "Liver (organ)" -> "Liver")
    # But preserve middle parenthetical content (e.g., "Calcium (2+) level" stays as is)
    normalized = normalized.strip()

    # Check if string ends with parentheses
    if normalized.endswith(")"):
        # Find the matching opening parenthesis for the trailing group
        paren_count = 0
        start_pos = -1

        # Work backwards from the end
        for i in range(len(normalized) - 1, -1, -1):
            if normalized[i] == ")":
                paren_count += 1
            elif normalized[i] == "(":
                paren_count -= 1
                if paren_count == 0:
                    start_pos = i
                    break

        # If we found a matching opening parenthesis, check if it's trailing
        # (i.e., only whitespace between the opening paren and what comes before)
        if start_pos > 0:
            # Get text before the parenthesis
            before_paren = normalized[:start_pos].rstrip()
            # If there's text before and it doesn't end with another closing paren,
            # this is a trailing parenthetical expression
            if before_paren and not before_paren.endswith(")"):
                normalized = before_paren

    # Normalize whitespace (but preserve case)
    normalized = " ".join(normalized.split())

    return normalized


class OntologySearchResult(BaseModel):
    """Standard ontology search result model.

    This is the common format used across all search backends to represent
    ontology concept search results.
    """

    concept_id: str
    concept_text: str
    score: float
    table_name: str

    def as_index_code(self) -> IndexCode:
        """Convert to IndexCode format"""
        return IndexCode(
            system=TABLE_TO_INDEX_CODE_SYSTEM.get(self.table_name, self.table_name),
            code=self.concept_id,
            display=normalize_concept(self.concept_text),
        )


class OntologySearchProtocol(Protocol):
    """Protocol defining the interface for ontology search clients.

    This protocol establishes a common interface that all ontology search
    implementations must follow, enabling polymorphic usage of different
    search backends (BioOntology, DuckDB, etc.).
    """

    async def search(
        self,
        queries: list[str],
        max_results: int = 30,
        filter_anatomical: bool = True,
    ) -> list[OntologySearchResult]:
        """Execute ontology search with given queries.

        Args:
            queries: List of search terms to query
            max_results: Maximum number of results to return
            filter_anatomical: Whether to filter out anatomical concepts

        Returns:
            List of OntologySearchResult objects
        """
        ...

    async def __aenter__(self) -> "OntologySearchProtocol":
        """Async context manager entry."""
        ...

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,  # noqa: ANN401
    ) -> None:
        """Async context manager exit."""
        ...

    async def search_parallel(
        self,
        queries: list[str],
        tables: list[str] | None = None,
        limit_per_query: int = 30,
        filter_anatomical: bool = False,
    ) -> list[OntologySearchResult]:
        """Search multiple queries in parallel.

        Args:
            queries: List of search queries
            tables: List of tables to search (optional)
            limit_per_query: Maximum results per query
            filter_anatomical: Whether to filter anatomical concepts

        Returns:
            Combined list of OntologySearchResult objects
        """
        ...
