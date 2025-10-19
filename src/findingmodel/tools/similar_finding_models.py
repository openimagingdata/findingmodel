"""
Similar Finding Models Tool

Uses two specialized Pydantic AI agents to find existing finding models that are
similar enough to a proposed model that editing them might be better than creating new ones.

Agent 1: Search Strategy - Determines search terms and gathers comprehensive results
Agent 2: Analysis - Analyzes results and makes similarity recommendations
"""

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing_extensions import NotRequired, TypedDict

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.index import DuckDBIndex as Index
from findingmodel.tools.common import get_openai_model


class SearchResult(TypedDict):
    """TypedDict for individual search results."""

    oifm_id: str
    name: str
    description: NotRequired[str]
    synonyms: NotRequired[list[str]]


class SearchTerms(BaseModel):
    """Search terms generated for finding similar models."""

    search_terms: list[str] = Field(description="3-5 search terms for finding similar medical imaging definitions")


class SearchStrategy(BaseModel):
    """Search strategy and results from the first agent."""

    search_terms_used: list[str] = Field(description="List of search terms that were actually used")
    total_results_found: int = Field(description="Total number of unique models found across all searches")
    search_results: list[SearchResult] = Field(description="All unique search results with model details")
    search_summary: str = Field(description="Summary of the search strategy and what was found")


@dataclass
class SearchContext:
    """Context class to pass the index to search tools."""

    index: Index


async def search_models_tool(ctx: RunContext[SearchContext], query: str, limit: int = 5) -> str:
    """
    Search for existing finding models in the index using a text query.

    Args:
        query: Search terms (can be finding names, synonyms, anatomical terms, etc.)
        limit: Maximum number of results to return (default 5)

    Returns:
        JSON string containing search results with model details
    """
    try:
        results = await ctx.deps.index.search(query, limit=limit)

        if not results:
            return f"No models found for query: '{query}'"

        # Format results for the agent
        formatted_results: list[SearchResult] = []
        for result in results:
            search_result = SearchResult(oifm_id=result.oifm_id, name=result.name)
            if result.description:
                search_result["description"] = result.description
            if result.synonyms:
                search_result["synonyms"] = result.synonyms
            formatted_results.append(search_result)

        return json.dumps({"query": query, "count": len(results), "results": formatted_results}, indent=2)

    except Exception as e:
        return f"Search failed for '{query}': {e!s}"


def create_search_agent(openai_model: str) -> Agent[SearchContext, SearchStrategy]:
    """Create the search agent for gathering comprehensive results."""
    return Agent[SearchContext, SearchStrategy](
        model=get_openai_model(openai_model),
        output_type=SearchStrategy,
        deps_type=SearchContext,
        tools=[search_models_tool],
        retries=3,
        system_prompt="""You are a search specialist for identifying standard codes for medical imaging findings. 
Your job is to systematically search for existing definitions that might be referred to using natural language; we 
want to find the best match in existing definitions, if there is one.

Your goal is to be broad in your search strategy. Use a couple of approaches, searching for related
terms and concepts to find likely relevant existing definitions.

1. **Direct searches**: Exact finding name and each synonym
2. **Anatomical searches**: Body parts, organs, regions mentioned
3. **Pathology searches**: Conditions, abnormalities, diseases
6. **Combination searches**: Multiple terms together

Strategy:
- Start with obvious searches (name, synonyms)
- Extract key medical terms from description and search those
- Search for anatomical locations

Generate about 5 likely search terms, perform the searches and keep track of unique results.
Pick about 10 broadly likely results from all searches; you don't need to return all the reults,
but don't be too restrictive--the next agent will analyze the results.
""",
    )


def create_term_generation_agent(openai_model: str) -> Agent[None, SearchTerms]:
    """Create a lightweight agent for generating search terms."""
    return Agent[None, SearchTerms](
        model=get_openai_model(openai_model),
        output_type=SearchTerms,
        system_prompt="""You are a medical terminology specialist. Your job is to generate 3-5 effective search terms 
for finding existing medical imaging finding definitions that might be similar to a proposed new finding.

Generate diverse search terms using these strategies:
1. **Direct terms**: The exact finding name and key synonyms
2. **Anatomical terms**: Body parts, organs, regions mentioned  
3. **Pathological terms**: Conditions, abnormalities, disease processes
4. **Alternative phrasings**: Different ways to describe the same concept

Keep terms concise and focused. Aim for terms that would appear in medical definitions.
Return exactly 3-5 terms, prioritizing the most likely to find relevant matches.""",
    )


class SimilarModelAnalysis(BaseModel):
    """Final analysis of similar models from the second agent."""

    similar_models: list[SearchResult] = Field(
        description="List of 1-3 existing models that are most similar and might be better to edit instead",
        min_length=0,
        max_length=3,
    )
    recommendation: Literal["edit_existing", "create_new"] = Field(
        description="Whether to edit existing models or create the new one"
    )
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 for the recommendation", ge=0.0, le=1.0)


def create_analysis_agent(openai_model: str) -> Agent[None, SimilarModelAnalysis]:
    """Create the analysis agent for evaluating similarity and making recommendations."""
    return Agent[None, SimilarModelAnalysis](
        model=get_openai_model(openai_model),
        output_type=SimilarModelAnalysis,
        retries=3,
        system_prompt="""You are an expert medical imaging informatics analyst specializing in mapping natural language
to standard codes and language for findings in medical imaging. Given a basic description of an imaging finding, your job 
is to analyze search results from a dictionary of standard codes and language and determine if any existing definitions
are enough to the to be used for the proposed finding language, or if the described finding is distinct enough that it 
should be a new definition.

Evaluate similarity based on:

**HIGH SIMILARITY (70%+ - recommend editing existing):**
- Same anatomical focus AND same finding type
- New term could be included as a synonym or alternative name for the existing definition
- Same clinical use case or scenario
- Would serve same diagnostic purpose

**MEDIUM SIMILARITY (40-70% - consider editing):**
- Same anatomical focus OR same finding type
- New term could be a useful synonym, but might required updating existing definition
- Related clinical use cases
- Could potentially be extended to cover new use case

**LOW SIMILARITY (<40% - create new):**
- Different anatomical focus AND different finding type
- New term is sufficiently distinct from existing definitions that it would not fit well
- Different clinical purposes
- Would require major restructuring to accommodate

Guidelines:
- Only recommend editing if there's genuine overlap in clinical meaning and purpose
- Consider if extending an existing definition would make it too broad/complex
- Prefer creating new definitions when the clinical context is clearly different
- Be conservative - when in doubt, recommend creating new

Return 1-3 most similar models ONLY if they meet the criteria for editing.""",
    )


async def find_similar_models(  # noqa: C901
    finding_name: str,
    description: str | None = None,
    synonyms: list[str] | None = None,
    index: Index | None = None,
    search_model: str | None = None,
    analysis_model: str | None = None,
) -> SimilarModelAnalysis:
    """
    Find existing finding models that are similar enough to the proposed model
    that it might be better to edit those instead of creating a new one.

    Optimized approach:
    1. Generate search terms using small model (with fallback to default if needed)
    2. Batch search all terms at once
    3. Analyze results using default model

    :param finding_name: Name of the proposed finding model
    :param description: Description of the proposed finding model
    :param synonyms: List of synonyms for the proposed finding model
    :param index: Index object to search. If None, creates a new one
    :param search_model: Model for generating search terms (defaults to small model)
    :param analysis_model: Model for analyzing results (defaults to default model)
    :return: Analysis with similar models and recommendation
    """

    # Set default models
    if search_model is None:
        search_model = settings.openai_default_model_small
    if analysis_model is None:
        analysis_model = settings.openai_default_model

    # Create index if not provided
    if index is None:
        index = Index()

    # Check if the index already has an exact match
    existing_model = await index.get(finding_name)
    if existing_model:
        logger.info(
            f"Exact match found in index for '{finding_name}': {existing_model.oifm_id} ({existing_model.name})"
        )
        confidence = 1.0 if existing_model.name == finding_name else 0.9
        return SimilarModelAnalysis(
            similar_models=[SearchResult(oifm_id=existing_model.oifm_id, name=existing_model.name)],
            recommendation="edit_existing",
            confidence=confidence,
        )

    # Check for an exact match in synonyms
    if synonyms:
        for synonym in synonyms:
            existing_model = await index.get(synonym)
            if existing_model:
                confidence = 0.9 if existing_model.name == synonym else 0.8
                logger.info(
                    f"Synonym match found in index for synonym '{synonym}': {existing_model.oifm_id} ({existing_model.name})"
                )
                return SimilarModelAnalysis(
                    similar_models=[SearchResult(oifm_id=existing_model.oifm_id, name=existing_model.name)],
                    recommendation="edit_existing",
                    confidence=confidence,
                )

    model_info = f"Name: {finding_name}\n"
    if description:
        model_info += f"Description: {description}\n"
    if synonyms:
        model_info += f"Synonyms: {', '.join(synonyms)}\n"

    term_prompt = f"Generate 3-5 search terms for finding existing medical imaging definitions similar to this finding:\n\n{model_info}"

    try:
        # Step 1: Generate search terms with smart model selection
        search_terms = await _generate_search_terms_with_fallback(term_prompt, search_model, finding_name)

        # Step 2: Batch search all terms at once
        logger.info("Performing batch search for all terms")
        batch_results = await index.search_batch(search_terms, limit=5)

        # Combine and deduplicate results
        all_found_models = {}
        for _query, results in batch_results.items():
            for result in results:
                all_found_models[result.oifm_id] = result

        total_unique = len(all_found_models)
        logger.info(f"Batch search found {total_unique} unique models across {len(search_terms)} terms")

        if not all_found_models:
            logger.info("No similar models found")
            return SimilarModelAnalysis(
                similar_models=[],
                recommendation="create_new",
                confidence=1.0,
            )

        # Step 3: Analyze results using default model
        search_results_data: list[dict[str, str | list[str]]] = []
        for result in all_found_models.values():
            result_data: dict[str, str | list[str]] = {"oifm_id": result.oifm_id, "name": result.name}
            if result.description:
                result_data["description"] = result.description
            if result.synonyms:
                result_data["synonyms"] = result.synonyms
            search_results_data.append(result_data)

        analysis_prompt = f"""
Based on the search results, analyze the similarity between the proposed model and existing models.

Finding Information:
{model_info}

SEARCH RESULTS:
Search Terms Used: {search_terms}
Total Models Found: {total_unique}

Existing definitions Found:
```json
{json.dumps(search_results_data, indent=2)}
```

Your task: Analyze these results and determine if any existing definitions are similar enough that editing them would be better 
than creating a new definition based on this finding. Apply the similarity criteria strictly and be conservative in your
recommendations.
"""

        # Create analysis agent (standard model)
        analysis_agent = create_analysis_agent(analysis_model)
        logger.info(f"Starting similarity analysis using {analysis_model}")
        analysis_result = await analysis_agent.run(analysis_prompt)
        final_analysis = analysis_result.output

        logger.info(
            f"Analysis complete for '{finding_name}': {final_analysis.recommendation} "
            f"(confidence: {final_analysis.confidence:.2f})"
        )

        if final_analysis.similar_models:
            logger.info(f"Similar models found: {final_analysis.similar_models}")

        return final_analysis

    except Exception as e:
        logger.error(f"Optimized analysis failed: {e}")
        # Return a fallback response
        return SimilarModelAnalysis(
            similar_models=[],
            recommendation="create_new",
            confidence=0.0,
        )


async def _generate_search_terms_with_fallback(term_prompt: str, search_model: str, finding_name: str) -> list[str]:
    """
    Generate search terms with fallback to default model if small model performs poorly.
    """
    import time

    # Try the specified model first
    logger.info(f"Generating search terms for '{finding_name}' using {search_model}")

    term_agent = create_term_generation_agent(search_model)
    start_time = time.time()
    term_result = await term_agent.run(term_prompt)
    duration = time.time() - start_time
    search_terms = term_result.output.search_terms

    logger.info(f"Generated {len(search_terms)} search terms in {duration:.2f}s: {search_terms}")

    # Check if we got reasonable results (at least 2 terms, reasonable performance)
    if len(search_terms) < 2 or duration > 3.0:
        fallback_model = settings.openai_default_model
        logger.info(f"Small model underperformed ({len(search_terms)} terms, {duration:.2f}s), trying {fallback_model}")

        fallback_agent = create_term_generation_agent(fallback_model)
        start_time = time.time()
        fallback_result = await fallback_agent.run(term_prompt)
        fallback_duration = time.time() - start_time
        fallback_terms = fallback_result.output.search_terms

        logger.info(
            f"Fallback generated {len(fallback_terms)} search terms in {fallback_duration:.2f}s: {fallback_terms}"
        )

        # Use fallback if it's significantly better
        if len(fallback_terms) > len(search_terms):
            return fallback_terms

    return search_terms
