"""
Anatomic Location Search Tool

Uses two specialized Pydantic AI agents to find relevant anatomic locations for imaging findings
by searching across multiple ontology terminology databases in LanceDB.

Agent 1: Search Strategy - Generates search queries and gathers comprehensive results
Agent 2: Matching - Analyzes results and selects best primary/alternate locations
"""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.tools.common import get_openai_model
from findingmodel.tools.ontology_search import OntologySearchClient, OntologySearchResult


class RawSearchResults(BaseModel):
    """Output from search agent."""

    search_terms_used: list[str] = Field(description="List of search terms that were actually used")
    search_results: list[OntologySearchResult] = Field(description="All unique search results found")


class LocationSearchResponse(BaseModel):
    """Output from matching agent."""

    primary_location: OntologySearchResult = Field(description="Best primary anatomic location")
    alternate_locations: list[OntologySearchResult] = Field(description="2-3 good alternate locations", max_length=3)
    reasoning: str = Field(description="Clear reasoning for selections made")


@dataclass
class SearchContext:
    """Context class for dependency injection."""

    search_client: OntologySearchClient


async def ontology_search_tool(ctx: RunContext[SearchContext], query: str, limit: int = 10) -> dict[str, Any]:
    """
    Tool for searching medical terminology databases.

    Args:
        query: Search terms (anatomical terms, body regions, organ systems, etc.)
        limit: Maximum number of results per table (default 10)

    Returns:
        Dictionary mapping table names to lists of search results
    """
    try:
        results = await ctx.deps.search_client.search_tables(
            query, tables=["anatomic_locations"], limit_per_table=limit
        )

        # Return dict for agent to process (not JSON string)
        return {table: [r.model_dump() for r in results_list] for table, results_list in results.items()}

    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return {"error": f"Search failed: {e!s}"}


def create_search_agent(model_name: str) -> Agent[SearchContext, RawSearchResults]:
    """Create the search agent for generating queries and gathering results."""
    return Agent[SearchContext, RawSearchResults](
        model=get_openai_model(model_name),
        output_type=RawSearchResults,
        deps_type=SearchContext,
        tools=[ontology_search_tool],
        retries=3,
        system_prompt="""You are a medical terminology search specialist. Your role is to generate 
effective search queries for finding anatomic locations in medical ontology databases.

Given a finding name and optional description, you should:
1. Generate 2-4 diverse search queries that might find relevant anatomic locations
2. Consider anatomical terms, body regions, organ systems
3. Try both specific and general terms
4. Use the search tool to gather results from the ontology databases

Return all unique results found across your searches.""",
    )


def create_matching_agent(model_name: str) -> Agent[None, LocationSearchResponse]:
    """Create the matching agent for picking best locations from search results."""
    return Agent[None, LocationSearchResponse](
        model=get_openai_model(model_name),
        output_type=LocationSearchResponse,
        system_prompt="""You are a medical imaging specialist who selects appropriate anatomic 
locations for imaging findings. Given search results from medical ontology databases, you must 
select the best primary anatomic location and 2-3 good alternates.

Selection criteria:
- Find the "sweet spot" of specificity - specific enough to be accurate but general enough 
  to encompass all locations where the finding can occur
- Prefer established ontology terms over very granular subdivisions
- Consider clinical relevance and common usage
- Provide clear reasoning for your selections

Always explain why you selected each location.""",
    )


async def find_anatomic_locations(
    finding_name: str,
    description: str | None = None,
    search_model: str | None = None,
    matching_model: str | None = None,
) -> LocationSearchResponse:
    """
    Find relevant anatomic locations for a finding using two-agent workflow.

    Args:
        finding_name: Name of the finding (e.g., "PCL tear")
        description: Optional detailed description
        search_model: Model for search agent (defaults to small model)
        matching_model: Model for matching agent (defaults to main model)

    Returns:
        LocationSearchResponse with selected locations and reasoning
    """
    # Set default models
    if search_model is None:
        search_model = settings.openai_default_model_small
    if matching_model is None:
        matching_model = settings.openai_default_model

    # Create search client
    search_client = OntologySearchClient()
    await search_client.connect()

    try:
        # Step 1: Search agent generates queries and gathers results
        search_agent = create_search_agent(search_model)
        search_context = SearchContext(search_client=search_client)

        prompt = f"Finding: {finding_name}"
        if description:
            prompt += f"\nDescription: {description}"

        logger.info(f"Starting search for anatomic locations: '{finding_name}'")
        search_result = await search_agent.run(prompt, deps=search_context)
        search_output = search_result.output

        logger.info(
            f"Search completed with {len(search_output.search_results)} total results "
            f"using terms: {search_output.search_terms_used}"
        )

        if not search_output.search_results:
            # Return a default response if no results found
            logger.warning(f"No search results found for '{finding_name}'")
            # This will likely cause the matching agent to fail, but let's try anyway

        # Step 2: Matching agent picks best locations from results
        matching_agent = create_matching_agent(matching_model)

        matching_prompt = f"""
Finding: {finding_name}
Description: {description or ""}

Search Results Found ({len(search_output.search_results)} total):
{json.dumps([r.model_dump() for r in search_output.search_results], indent=2)}

Select the best primary anatomic location and 2-3 good alternates. 

The goal is to find the "sweet spot" where it's as specific as possible, 
but still encompassing the locations where the finding can occur. For example,
for the finding of "adrenal nodule", you would prefer "adrenal gland" to 
"abdomen" (too broad) and "anterior limb of adrenal" (too specific).
"""

        logger.info(f"Starting location matching analysis using {matching_model}")
        matching_result = await matching_agent.run(matching_prompt)
        final_response = matching_result.output

        logger.info(
            f"Location matching complete for '{finding_name}': "
            f"primary='{final_response.primary_location.concept_text}', "
            f"alternates={len(final_response.alternate_locations)}"
        )

        return final_response

    finally:
        if search_client.connected:
            search_client.disconnect()  # Note: disconnect is synchronous
