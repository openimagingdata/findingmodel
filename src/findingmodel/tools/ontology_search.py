# src/findingmodel/tools/ontology_search.py
"""
Ontology search tools with Protocol-based architecture for multiple backends.

This module provides:
- OntologySearchProtocol: Common interface for all search backends
- BioOntologySearchClient: REST API search implementation for BioOntology.org
- Utility functions for concept normalization and reranking
"""

import asyncio
from typing import Any, ClassVar, Optional, Protocol, cast

import httpx
from pydantic import BaseModel, Field

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.index_code import IndexCode

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
    ) -> list["OntologySearchResult"]:
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
    ) -> list["OntologySearchResult"]:
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


# BioOntology API Integration
class BioOntologySearchResult(BaseModel):
    """A single search result from BioOntology API."""

    concept_id: str = Field(description="The full URI/ID of the concept")
    ontology: str = Field(description="Source ontology (e.g., SNOMEDCT, RADLEX)")
    pref_label: str = Field(description="Preferred label for the concept")
    synonyms: list[str] = Field(default_factory=list, description="Alternative labels/synonyms")
    definition: Optional[str] = Field(default=None, description="Definition of the concept")
    semantic_types: list[str] = Field(default_factory=list, description="UMLS semantic type codes")
    ui_link: str = Field(description="BioPortal UI link for the concept")

    @classmethod
    def from_api_response(cls, item: dict[str, Any]) -> "BioOntologySearchResult":
        """Create from BioOntology API response item."""
        # Extract ontology from the links
        ontology = ""
        links = item.get("links", {})
        if isinstance(links, dict) and "ontology" in links:
            ontology_url = str(links["ontology"])
            # Extract ontology name from URL like: https://data.bioontology.org/ontologies/LOINC
            ontology = ontology_url.split("/")[-1] if "/" in ontology_url else ""

        # Handle definition which can sometimes be a list
        definition_raw = item.get("definition")
        definition: Optional[str] = None
        if isinstance(definition_raw, list):
            definition = str(definition_raw[0]) if definition_raw else None
        elif isinstance(definition_raw, str):
            definition = definition_raw

        # Handle synonyms
        synonyms_raw = item.get("synonym", [])
        synonyms = synonyms_raw if isinstance(synonyms_raw, list) else []

        # Handle semantic types
        semantic_raw = item.get("semanticType", [])
        semantic_types = semantic_raw if isinstance(semantic_raw, list) else []

        # Get UI link
        ui_link = ""
        if isinstance(links, dict):
            ui_link = str(links.get("ui", ""))

        return cls(
            concept_id=str(item.get("@id", "")),
            ontology=ontology,
            pref_label=str(item.get("prefLabel", "")),
            synonyms=[str(s) for s in synonyms],
            definition=definition,
            semantic_types=[str(s) for s in semantic_types],
            ui_link=ui_link,
        )

    def to_ontology_search_result(self) -> OntologySearchResult:
        """Convert to standard OntologySearchResult format."""
        # Map BioOntology source to our standard table names
        table_mapping = {
            "SNOMEDCT": "snomedct",
            "RADLEX": "radlex",
            "LOINC": "loinc",
            "ICD10CM": "icd10cm",
            "GAMUTS": "gamuts",
            "CPT": "cpt",
        }
        table_name = table_mapping.get(self.ontology, self.ontology.lower())

        id = self.concept_id.split("/")[-1] if "/" in self.concept_id else self.concept_id

        return OntologySearchResult(
            concept_id=id,
            concept_text=self.pref_label,
            table_name=table_name,
            score=1.0,  # BioOntology doesn't provide scores, use 1.0 as default
        )


class BioOntologySearchResults(BaseModel):
    """Results from a BioOntology search."""

    query: str = Field(description="The search query used")
    total_count: int = Field(description="Total number of results available")
    page_count: int = Field(description="Number of pages available")
    current_page: int = Field(description="Current page number")
    results: list[BioOntologySearchResult] = Field(description="Search results for current page")


class BioOntologySearchClient:
    """Async client for searching medical concepts via BioOntology REST API."""

    DEFAULT_ONTOLOGIES: ClassVar[list[str]] = ["SNOMEDCT", "RADLEX", "LOINC"]
    DEFAULT_INCLUDE_FIELDS: ClassVar[str] = "prefLabel,synonym,definition,semanticType"
    API_BASE_URL: ClassVar[str] = "https://data.bioontology.org"

    def __init__(self, api_key: Optional[str] = None, client: Optional[httpx.AsyncClient] = None) -> None:
        """
        Initialize the BioOntology search client.

        Args:
            api_key: BioOntology API key. If not provided, will try to get from settings.
            client: Optional pre-configured httpx.AsyncClient for connection pooling.
        """
        self.api_key: Optional[str] = None
        self._client = client
        self._owns_client = client is None  # Track if we created the client

        if api_key:
            self.api_key = api_key
        else:
            bio_key = getattr(settings, "bioontology_api_key", None)
            if bio_key:
                # Handle SecretStr from pydantic
                self.api_key = bio_key.get_secret_value() if hasattr(bio_key, "get_secret_value") else str(bio_key)

        if not self.api_key:
            raise ValueError("BioOntology API key is required. Set BIOONTOLOGY_API_KEY in .env file.")

    async def __aenter__(self) -> "BioOntologySearchClient":
        """Async context manager entry."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,  # noqa: ANN401
    ) -> None:
        """Async context manager exit."""
        # Parameters are required by protocol but not used
        _ = (exc_type, exc_val, exc_tb)
        if self._owns_client and self._client:
            await self._client.aclose()
            self._client = None

    async def search_bioontology(
        self,
        query: str,
        ontologies: Optional[list[str]] = None,
        page_size: int = 50,
        page: int = 1,
        include_fields: Optional[str] = None,
        require_exact_match: bool = False,
        semantic_types: Optional[list[str]] = None,
    ) -> BioOntologySearchResults:
        """
        Search for concepts across specified ontologies.

        Args:
            query: Search query string
            ontologies: List of ontology acronyms to search (e.g., ["SNOMEDCT", "RADLEX"])
                       If None, uses default ontologies
            page_size: Number of results per page (max 100)
            page: Page number to retrieve (1-based)
            include_fields: Comma-separated fields to include in response
            require_exact_match: If True, only return exact matches
            semantic_types: Filter by UMLS semantic types (e.g., ["T047"] for disease/syndrome)

        Returns:
            BioOntologySearchResults containing the search results

        Raises:
            httpx.HTTPError: If the API request fails
            ValueError: If API key is not set or client is not initialized
        """
        if not self.api_key:
            raise ValueError("BioOntology API key not configured")

        if self._client is None:
            raise ValueError("Client not initialized. Use async context manager or provide client.")

        # Build request parameters
        params: dict[str, str | int] = {
            "q": query,
            "pagesize": min(page_size, 100),  # API max is 100
            "page": page,
        }

        # Add ontologies filter
        if ontologies:
            params["ontologies"] = ",".join(ontologies)
        else:
            params["ontologies"] = ",".join(self.DEFAULT_ONTOLOGIES)

        # Add fields to include
        params["include"] = include_fields or self.DEFAULT_INCLUDE_FIELDS

        # Add exact match requirement
        if require_exact_match:
            params["require_exact_match"] = "true"

        # Add semantic type filter
        if semantic_types:
            params["semantic_types"] = ",".join(semantic_types)

        # Make API request
        headers = {"Authorization": f"apikey token={self.api_key}"}

        response = await self._client.get(f"{self.API_BASE_URL}/search", params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        # Parse response
        results = []
        for item in data.get("collection", []):
            try:
                results.append(BioOntologySearchResult.from_api_response(item))
            except Exception as e:
                # Skip items that fail to parse
                logger.warning(f"Failed to parse BioOntology result item: {e}")
                continue

        return BioOntologySearchResults(
            query=query,
            total_count=data.get("totalCount", 0),
            page_count=data.get("pageCount", 1),
            current_page=data.get("page", 1),
            results=results,
        )

    async def search_all_pages(
        self,
        query: str,
        ontologies: Optional[list[str]] = None,
        max_results: int = 100,
        include_fields: Optional[str] = None,
        require_exact_match: bool = False,
        semantic_types: Optional[list[str]] = None,
    ) -> list[BioOntologySearchResult]:
        """
        Search and retrieve all results up to max_results, handling pagination automatically.

        Args:
            query: Search query string
            ontologies: List of ontology acronyms to search
            max_results: Maximum number of results to retrieve
            include_fields: Comma-separated fields to include in response
            require_exact_match: If True, only return exact matches
            semantic_types: Filter by UMLS semantic types

        Returns:
            List of all search results up to max_results
        """
        all_results: list[BioOntologySearchResult] = []
        page = 1
        page_size = min(50, max_results)

        while len(all_results) < max_results:
            search_results = await self.search_bioontology(
                query=query,
                ontologies=ontologies,
                page_size=page_size,
                page=page,
                include_fields=include_fields,
                require_exact_match=require_exact_match,
                semantic_types=semantic_types,
            )

            all_results.extend(search_results.results)

            # Check if we have all available results or reached max
            if len(all_results) >= search_results.total_count or len(all_results) >= max_results:
                break

            # Check if there are more pages
            if page >= search_results.page_count:
                break

            page += 1

        return all_results[:max_results]

    async def search_as_ontology_results(
        self,
        query: str,
        ontologies: Optional[list[str]] = None,
        max_results: int = 50,
        **kwargs: Any,  # noqa: ANN401
    ) -> list[OntologySearchResult]:
        """
        Search BioOntology and return results in standard OntologySearchResult format.

        This method provides compatibility with the existing LanceDBOntologySearchClient interface.

        Args:
            query: Search query string
            ontologies: List of ontology acronyms to search
            max_results: Maximum number of results to retrieve
            **kwargs: Additional search parameters

        Returns:
            List of OntologySearchResult objects
        """
        bio_results = await self.search_all_pages(
            query=query,
            ontologies=ontologies,
            max_results=max_results,
            **kwargs,
        )

        return [result.to_ontology_search_result() for result in bio_results]

    async def search(
        self,
        queries: list[str],
        max_results: int = 30,
        filter_anatomical: bool = True,
    ) -> list[OntologySearchResult]:
        """Execute ontology search implementing the OntologySearchProtocol interface.

        This method provides a Protocol-compliant interface by wrapping the
        search_as_ontology_results() method. When multiple queries are provided,
        they are combined with OR logic in the BioOntology search.

        Args:
            queries: List of search terms to query (combined with OR)
            max_results: Maximum number of results to return
            filter_anatomical: Whether to filter out anatomical concepts (uses semantic type filtering)

        Returns:
            List of OntologySearchResult objects from BioOntology
        """
        if not queries:
            return []

        # Combine multiple queries with OR for BioOntology search
        # BioOntology handles this well with space-separated terms
        combined_query = " OR ".join(queries)

        # Map filter_anatomical to semantic types if needed
        # T023 = Body Part, Organ, or Organ Component
        # T029 = Body Location or Region
        # T030 = Body Space or Junction
        # Since BioOntology doesn't have exclude, we handle this post-search

        # Use the existing search_as_ontology_results method
        results = await self.search_as_ontology_results(
            query=combined_query,
            ontologies=None,  # Use default ontologies
            max_results=max_results,
        )

        # Filter anatomical concepts if requested
        if filter_anatomical:
            # Filter out anatomical semantic types (T023, T029, T030)
            filtered_results = []
            for result in results:
                # Check if result has anatomical semantic types
                # The semantic types are stored in the original BioOntologySearchResult
                # We need to check the concept_text for anatomical indicators
                # For now, do simple keyword filtering
                lower_text = result.concept_text.lower()
                is_anatomical = any(
                    term in lower_text
                    for term in [
                        "anatomy",
                        "anatomical",
                        "body part",
                        "organ",
                        "tissue",
                        "structure",
                        "region",
                        "location",
                        "site",
                    ]
                )
                if not is_anatomical:
                    filtered_results.append(result)
            return filtered_results

        return results

    async def search_parallel(
        self,
        queries: list[str],
        tables: list[str] | None = None,
        limit_per_query: int = 30,
        filter_anatomical: bool = False,
    ) -> list[OntologySearchResult]:
        """Execute multiple search queries in parallel.

        This method implements the OntologySearchProtocol.search_parallel interface
        for BioOntology. Since BioOntology doesn't have the concept of tables,
        the tables parameter is ignored.

        Args:
            queries: List of search queries to execute
            tables: Ignored - BioOntology doesn't use table separation
            limit_per_query: Maximum results per individual query
            filter_anatomical: Whether to filter out anatomical concepts

        Returns:
            Combined list of OntologySearchResult objects from all queries
        """
        # Tables parameter is required by protocol but not used by BioOntology
        _ = tables
        if not queries:
            return []

        # Execute each query in parallel using asyncio.gather
        tasks = [
            self.search(queries=[query], max_results=limit_per_query, filter_anatomical=filter_anatomical)
            for query in queries
        ]

        # Execute all queries in parallel
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results, handling any exceptions
        all_results = []
        seen_ids = set()

        for i, results_or_error in enumerate(results_lists):
            if isinstance(results_or_error, Exception):
                logger.warning(f"BioOntology search failed for query '{queries[i]}': {results_or_error}")
                continue

            # Type narrowing: after isinstance check, this must be list[OntologySearchResult]
            results = cast(list[OntologySearchResult], results_or_error)
            for result in results:
                # Deduplicate by concept_id
                if result.concept_id not in seen_ids:
                    seen_ids.add(result.concept_id)
                    all_results.append(result)

        # Sort by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results
