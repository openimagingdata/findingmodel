"""
Anatomic Location Search Tool

Uses two specialized Pydantic AI agents to find relevant anatomic locations for imaging findings
by searching across multiple ontology terminology databases in LanceDB.

Agent 1: Search Strategy - Generates search queries and gathers comprehensive results
Agent 2: Matching - Analyzes results and selects best primary/alternate locations
"""

import json

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing_extensions import Literal

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.tools.common import get_openai_model
from findingmodel.tools.duckdb_search import DuckDBOntologySearchClient
from findingmodel.tools.ontology_search import (
    OntologySearchProtocol,
    OntologySearchResult,
    normalize_concept,
)


class AnatomicQueryTerms(BaseModel):
    """Output from query generation agent."""

    region: (
        Literal["Abdomen", "Neck", "Lower Extremity", "Breast", "Body", "Thorax", "Upper Extremity", "Head", "Pelvis"]
        | None
    ) = Field(
        default=None,
        description="Primary anatomic region; one of the predefined regions: Abdomen, Neck, Lower Extremity, Breast, Body, Thorax, Upper Extremity, Head, Pelvis",
    )
    terms: list[str] = Field(description="List of anatomic location search terms", default_factory=list)


async def generate_anatomic_query_terms(
    finding_name: str, finding_description: str | None = None, model: str | None = None
) -> AnatomicQueryTerms:
    """Generate anatomic location search terms for a finding.

    First identifies the most appropriate anatomic location,
    then generates ontology term variations.

    Args:
        finding_name: Name of the finding
        finding_description: Optional detailed description
        model: OpenAI model to use (defaults to small model from settings)

    Returns:
        List of anatomic location search terms
    """
    if model is None:
        model = settings.openai_default_model_small

    agent = Agent[None, AnatomicQueryTerms](
        model=get_openai_model(model),
        output_type=AnatomicQueryTerms,
        system_prompt="""You are an anatomic location specialist for medical imaging findings.
        
Given a medical finding, you must:
1. First identify the REGION and PRIMARY anatomic location where this finding occurs
2. Then generate 3-5 ontology term variations for that location

- Focus on formal medical terminology used in ontologies.
- Do NOT include acronyms or layman terms. 
- Do NOT include bare adjectives (e.g., "abdominal", "cervical")--we're looking for nouns/noun phrases
- Do NOT separately search for left and right; only search for general terms.

THINK about what location is most specific to this finding but still general enough to cover
all locations where the finding can occur.

Example:
Finding: "meniscal tear"
Primary location: knee meniscus
Region: "Lower Extremity"
Terms: ["meniscus", "middle meniscus", "tibial meniscus"]

Example:
Finding: "pneumonia"
Primary location: lung
Region: "Thorax"
Terms: ["lung", "lung parenchyma", "lower respiratory tract"]

Return ONLY the region and list of terms, nothing else.""",
    )

    prompt = f"Finding: {finding_name}"
    if finding_description:
        prompt += f"\nDescription: {finding_description}"

    try:
        result = await agent.run(prompt)
        terms = result.output.terms

        # Ensure finding name is included if not already
        if finding_name.lower() not in [t.lower() for t in terms]:
            terms.append(finding_name)

        logger.info(f"Generated {len(terms)} anatomic query terms for '{finding_name}'")
        return result.output
    except Exception as e:
        logger.warning(f"Failed to generate anatomic query terms: {e}, using fallback")
        # Fallback to just the finding name
        return AnatomicQueryTerms(region=None, terms=[finding_name])


async def execute_anatomic_search(
    query_info: AnatomicQueryTerms, client: OntologySearchProtocol, limit: int = 30
) -> list[OntologySearchResult]:
    """Execute search on anatomic_locations table with region and sided filtering.

    Returns results from hybrid search filtered by region and sided.

    Args:
        query_info: AnatomicQueryTerms containing terms and optional region
        client: OntologySearchProtocol instance (DuckDB or LanceDB)
        limit: Maximum results per query term (default 30)

    Returns:
        List of OntologySearchResult objects with normalized concept text
    """
    # For DuckDB client, we need to pass region and sided filters
    results: list[OntologySearchResult]
    if hasattr(client, "search_with_filters"):
        # Use the filtered search if available
        results = await client.search_with_filters(
            queries=query_info.terms,
            region=query_info.region,
            sided_filter=["generic", "nonlateral"],  # Only generic or nonlateral sided
            limit_per_query=limit,
        )
    else:
        # Fallback to standard search for compatibility
        results = await client.search_parallel(
            queries=query_info.terms,
            tables=["anatomic_locations"],  # Explicit table specification
            limit_per_query=limit,
            filter_anatomical=False,  # Not needed, we're only searching anatomic table
        )

    # Normalize concept text for all results
    for result in results:
        result.concept_text = normalize_concept(result.concept_text)

    logger.info(f"Found {len(results)} anatomic location results")
    return results


class LocationSearchResponse(BaseModel):
    """Output from matching agent."""

    primary_location: OntologySearchResult = Field(description="Best primary anatomic location")
    alternate_locations: list[OntologySearchResult] = Field(description="2-3 good alternate locations", max_length=3)
    reasoning: str = Field(description="Clear reasoning for selections made")


def create_location_selection_agent(model: str | None = None) -> Agent[None, LocationSearchResponse]:
    """Create agent for selecting best anatomic locations from search results.

    Args:
        model: OpenAI model to use (defaults to main model from settings)

    Returns:
        Agent configured for location selection
    """
    if model is None:
        model = settings.openai_default_model_small

    return Agent[None, LocationSearchResponse](
        model=get_openai_model(model),
        output_type=LocationSearchResponse,
        system_prompt="""You are a medical imaging specialist who selects appropriate anatomic 
locations for imaging findings. Given search results from medical ontology databases, you must 
select the best primary anatomic location and 2-3 possible alternates.

Selection criteria:
- Find the "sweet spot" of specificity - specific enough to be accurate but general enough 
  to encompass all locations where the finding can occur
- Consider clinical relevance and common usage, but do NOT select overly broad locations
  or overly narrow/specific ones
- Provide concise reasoning for your selections
- Note: If results appear pre-ranked, top results are likely most relevant

Examples of good primary locations:
"abdominal abscess" -> "RID56: abdomen"
"medial meniscal tear" -> "RID2772: medial meniscus"
"pneumonia" -> "RID1301: lung"
"mediastinal lymphadenopathy" -> "RID28852: set of mediastinal lymph nodes"
"coronary artery calcification" -> "RID1385: heart"
""",
    )


async def find_anatomic_locations(
    finding_name: str,
    description: str | None = None,
    use_duckdb: bool = True,
    model: str | None = None,
) -> LocationSearchResponse:
    """Find relevant anatomic locations for a finding using 3-stage pipeline.

    Pipeline stages:
    1. Generate query terms using AI
    2. Execute direct search on anatomic_locations table
    3. Select best locations using AI agent

    Args:
        finding_name: Name of the finding (e.g., "PCL tear")
        description: Optional detailed description
        use_duckdb: Use DuckDB client if True, LanceDB if False (default True)
        model: OpenAI model to use (defaults to small model from settings)

    Returns:
        LocationSearchResponse with selected locations and reasoning
    """
    logger.info(f"Starting anatomic location search for: {finding_name}")

    # Stage 1: Generate query terms
    query_info = await generate_anatomic_query_terms(finding_name, description, model=model)
    logger.info(f"Generated query terms: {query_info.terms}, region: {query_info.region}")

    # Stage 2: Execute search with DuckDB client
    if use_duckdb:
        async with DuckDBOntologySearchClient() as client:
            search_results = await execute_anatomic_search(query_info, client)
    else:
        logger.error("DuckDB is the only supported backend for anatomic location search")
        raise ValueError("DuckDB is required for anatomic location search")

    if not search_results:
        logger.warning(f"No search results found for '{finding_name}'")
        # Return a default response
        default_location = OntologySearchResult(
            concept_id="NO_RESULTS",
            concept_text="unspecified anatomic location",
            score=0.0,
            table_name="anatomic_locations",
        )
        return LocationSearchResponse(
            primary_location=default_location,
            alternate_locations=[],
            reasoning=f"No anatomic locations found for '{finding_name}'. Using default.",
        )

    # Stage 3: Selection using AI agent
    selection_agent = create_location_selection_agent(model=model)

    # Build structured prompt for the agent
    prompt = f"""
Finding: {finding_name}
Description: {description or "Not provided"}

Search Results ({len(search_results)} locations found):
{json.dumps([r.model_dump() for r in search_results], indent=2)}

Select the best primary anatomic location and 2-3 good alternates.
The goal is to find the "sweet spot" where it's as specific as possible,
but still encompassing the locations where the finding can occur.
"""

    logger.info(f"Starting location selection analysis for {finding_name}")
    result = await selection_agent.run(prompt)
    final_response = result.output

    logger.info(
        f"Location selection complete for '{finding_name}': "
        f"primary='{final_response.primary_location.concept_text}', "
        f"alternates={len(final_response.alternate_locations)}"
    )

    return final_response
