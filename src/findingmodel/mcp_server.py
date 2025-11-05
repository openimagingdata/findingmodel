"""MCP server that exposes the Finding Model Index search functionality."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from findingmodel.index import DuckDBIndex


class AttributeResult(BaseModel):
    """Represents an attribute in the search results."""

    attribute_id: str
    name: str
    type: str


class SearchResult(BaseModel):
    """Represents a single finding model search result."""

    oifm_id: str
    name: str
    slug_name: str
    filename: str
    description: str | None = None
    synonyms: list[str] | None = None
    tags: list[str] | None = None
    contributors: list[str] | None = None
    attributes: list[AttributeResult] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from the search tool."""

    query: str
    limit: int
    tags: list[str] | None = None
    result_count: int
    results: list[SearchResult]


# Create the MCP server
mcp = FastMCP("finding-model-search")


@mcp.tool()
async def search_finding_models(
    query: str,
    limit: int = 10,
    tags: list[str] | None = None,
) -> SearchResponse:
    """Search for finding models using hybrid search (FTS + semantic).

    This tool searches the Finding Model Index using a combination of
    full-text search (FTS) and semantic search with Reciprocal Rank Fusion.
    It returns exact matches immediately if found, otherwise performs
    hybrid search to find the most relevant models.

    Args:
        query: Search query string (e.g., "pneumothorax", "lung nodule")
        limit: Maximum number of results to return (default: 10, max: 100)
        tags: Optional list of tags - models must have ALL specified tags

    Returns:
        SearchResponse with matched finding models including:
            - oifm_id: Unique identifier for the finding model
            - name: Display name of the finding model
            - description: Detailed description (if available)
            - synonyms: Alternative names for the finding
            - tags: Associated tags for categorization
            - attributes: List of attributes defined in the model

    Examples:
        Search for pneumothorax:
            query: "pneumothorax"
            limit: 5

        Search for liver findings with specific tag:
            query: "liver lesion"
            limit: 10
            tags: ["abdominal"]
    """
    # Validate limit
    if limit < 1:
        limit = 1
    elif limit > 100:
        limit = 100

    # Normalize tags to None if empty list
    normalized_tags = tags if tags else None

    # Perform search using the Index
    async with DuckDBIndex() as index:
        entries = await index.search(query, limit=limit, tags=normalized_tags)

        # Convert entries to response format
        results = [
            SearchResult(
                oifm_id=entry.oifm_id,
                name=entry.name,
                slug_name=entry.slug_name,
                filename=entry.filename,
                description=entry.description,
                synonyms=entry.synonyms,
                tags=entry.tags,
                contributors=entry.contributors,
                attributes=[
                    AttributeResult(
                        attribute_id=attr.attribute_id,
                        name=attr.name,
                        type=attr.type,
                    )
                    for attr in (entry.attributes or [])
                ],
            )
            for entry in entries
        ]

        return SearchResponse(
            query=query,
            limit=limit,
            tags=normalized_tags,
            result_count=len(results),
            results=results,
        )


@mcp.tool()
async def get_finding_model(identifier: str) -> SearchResult | None:
    """Retrieve a specific finding model by its ID, name, or synonym.

    This tool looks up a finding model using exact matching on:
    - OIFM ID (e.g., "OIFM_RSNA_000001")
    - Name (case-insensitive)
    - Slug name (normalized version of the name)
    - Synonyms (case-insensitive)

    Args:
        identifier: The OIFM ID, name, slug name, or synonym to look up

    Returns:
        SearchResult with the finding model details, or None if not found

    Examples:
        Get by OIFM ID:
            identifier: "OIFM_RSNA_000001"

        Get by name:
            identifier: "Pneumothorax"

        Get by synonym:
            identifier: "collapsed lung"
    """
    async with DuckDBIndex() as index:
        entry = await index.get(identifier)

        if entry is None:
            return None

        return SearchResult(
            oifm_id=entry.oifm_id,
            name=entry.name,
            slug_name=entry.slug_name,
            filename=entry.filename,
            description=entry.description,
            synonyms=entry.synonyms,
            tags=entry.tags,
            contributors=entry.contributors,
            attributes=[
                AttributeResult(
                    attribute_id=attr.attribute_id,
                    name=attr.name,
                    type=attr.type,
                )
                for attr in (entry.attributes or [])
            ],
        )


@mcp.tool()
async def count_finding_models() -> dict[str, int]:
    """Get statistics about the finding model index.

    Returns counts of:
    - Total finding models
    - Total contributors (people)
    - Total organizations

    Returns:
        Dictionary with count statistics

    Example:
        Returns: {
            "finding_models": 150,
            "people": 45,
            "organizations": 12
        }
    """
    async with DuckDBIndex() as index:
        models_count = await index.count()
        people_count = await index.count_people()
        orgs_count = await index.count_organizations()

        return {
            "finding_models": models_count,
            "people": people_count,
            "organizations": orgs_count,
        }


def main() -> None:
    """Run the MCP server using stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
