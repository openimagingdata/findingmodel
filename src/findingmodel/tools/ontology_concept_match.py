"""
Ontology Concept Search Tool

Uses two specialized Pydantic AI agents to find and categorize relevant medical concepts
from BioOntology API, excluding anatomical structures.

Agent 1: Search Strategy - Generates diverse search queries and gathers comprehensive results
Agent 2: Categorization - Analyzes results and categorizes into relevance tiers
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from findingmodel import logger
from findingmodel.config import settings
from findingmodel.tools.common import get_openai_model
from findingmodel.tools.ontology_search import (
    BioOntologySearchClient,
    OntologySearchResult,
    normalize_concept,
)


class CategorizedConcepts(BaseModel):
    """Categorized concept IDs with validation constraints."""

    exact_matches: list[str] = Field(description="Concept IDs that exactly match the finding", max_length=5)
    should_include: list[str] = Field(description="Concept IDs that should be included", max_length=10)
    marginal: list[str] = Field(description="Concept IDs that are marginally relevant", max_length=10)
    rationale: str = Field(description="Brief categorization rationale")


class CategorizedOntologyConcepts(BaseModel):
    """Categorized ontology concepts for a finding model."""

    exact_matches: list[OntologySearchResult] = Field(
        description="Concepts that directly represent what the finding model describes"
    )

    should_include: list[OntologySearchResult] = Field(
        description="Related concepts that should be included as relevant"
    )

    marginal_concepts: list[OntologySearchResult] = Field(description="Peripherally related concepts to consider")

    search_summary: str = Field(description="Summary of search strategy and categorization rationale")

    excluded_anatomical: list[str] = Field(
        default_factory=list, description="List of anatomical concepts that were filtered out"
    )


@dataclass
class CategorizationContext:
    """Context for categorization agent with dependencies."""

    finding_name: str
    search_results: list[OntologySearchResult]
    query_terms: list[str]


def _filter_anatomical_concepts(search_results: list[OntologySearchResult]) -> list[OntologySearchResult]:
    """Filter out anatomical concepts from search results."""
    anatomical_terms = [
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

    filtered_results = []
    for result in search_results:
        lower_text = result.concept_text.lower()
        is_anatomical = any(term in lower_text for term in anatomical_terms)
        if not is_anatomical:
            filtered_results.append(result)

    return filtered_results


def _add_exact_matches(
    sorted_results: list[OntologySearchResult], query_terms: list[str], max_results: int, selected_ids: set[str]
) -> tuple[list[OntologySearchResult], int]:
    """Add exact matches that were ranked lower to the top results."""
    query_terms_lower = [term.lower() for term in query_terms]
    exact_matches_added = 0
    top_results = sorted_results[:max_results].copy()

    for result in sorted_results[max_results:]:
        # Normalize the concept text for comparison
        normalized_text = normalize_concept(result.concept_text).lower()

        # Check if this is an exact match for any query term
        if normalized_text in query_terms_lower and result.concept_id not in selected_ids:
            top_results.append(result)
            selected_ids.add(result.concept_id)
            exact_matches_added += 1
            matched_term = query_terms[query_terms_lower.index(normalized_text)]
            logger.info(
                f"Added exact match for term '{matched_term}': "
                f"{result.concept_id} ({result.concept_text}) [score: {result.score:.3f}]"
            )

    return top_results, exact_matches_added


async def execute_ontology_search(
    query_terms: list[str],
    exclude_anatomical: bool = True,
    base_limit: int = 30,
    max_results: int = 12,
    ontologies: list[str] | None = None,
) -> list[OntologySearchResult]:
    """Execute ontology search using BioOntology API.

    This function executes a search against the BioOntology API and applies
    a smart selection strategy to ensure both high-scoring results and exact matches
    are included in the final result set.

    Strategy:
    1. Execute search with all query terms using BioOntology API
    2. Sort all results by score in descending order
    3. Take the top N results by score
    4. Check remaining results for exact matches of any query term
    5. Add any missing exact matches (important for cases like RID5350 for "pneumonia")
    6. Apply concept text normalization to all results

    Args:
        query_terms: List of search terms to query the ontology databases
        exclude_anatomical: Whether to filter out anatomical structure concepts (default: True)
        base_limit: Initial limit per query for casting a wider net (default: 30)
        max_results: Maximum number of results to return after selection (default: 12)
        ontologies: Optional list of ontology acronyms to search (default: SNOMEDCT, RADLEX, LOINC)

    Returns:
        List of OntologySearchResult objects with normalized concept text,
        sorted by relevance score with exact matches guaranteed to be included.

    Raises:
        ValueError: If BioOntology API key is not configured
        Exception: If the search operation fails
    """
    if not settings.bioontology_api_key:
        raise ValueError(
            "BioOntology API key is required for ontology concept matching. "
            "Please set BIOONTOLOGY_API_KEY in your environment."
        )

    try:
        logger.info(f"Executing ontology search with {len(query_terms)} terms")

        # Execute search using BioOntology API
        async with BioOntologySearchClient() as client:
            combined_query = " OR ".join(query_terms)
            search_results = await client.search_as_ontology_results(
                query=combined_query,
                ontologies=ontologies,
                max_results=base_limit * len(query_terms) if query_terms else base_limit,
            )

            # Apply anatomical filtering if requested
            if exclude_anatomical:
                search_results = _filter_anatomical_concepts(search_results)

        logger.info(f"Search returned {len(search_results)} total results")

        if not search_results:
            logger.warning("No search results found")
            return []

        # Sort all results by score (descending)
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)

        # Take the top N results by score and add exact matches
        selected_ids = {r.concept_id for r in sorted_results[:max_results]}
        top_results, exact_matches_added = _add_exact_matches(sorted_results, query_terms, max_results, selected_ids)

        if exact_matches_added > 0:
            logger.info(f"Added {exact_matches_added} exact match(es) that were ranked lower")

        # Apply normalization to all selected results
        for result in top_results:
            result.concept_text = normalize_concept(result.concept_text)

        logger.info(
            f"Selected {len(top_results)} results for categorization "
            f"({max_results} top-scored + {exact_matches_added} exact matches)"
        )

        return top_results

    except Exception as e:
        logger.error(f"Error executing ontology search: {e}")
        raise


def create_categorization_agent() -> Agent[CategorizationContext, CategorizedConcepts]:
    """Create categorization agent following proper Pydantic AI patterns.

    This agent categorizes ontology search results into relevance tiers.
    Post-processing is handled separately to ensure exact matches are properly identified.

    Returns:
        Agent that takes CategorizationContext and produces CategorizedConcepts
    """
    model = get_openai_model(settings.openai_default_model)

    return Agent[CategorizationContext, CategorizedConcepts](
        model=model,
        output_type=CategorizedConcepts,
        deps_type=CategorizationContext,
        system_prompt="""You are a medical ontology expert.

A series of search results against a medical ontology has returned a batch of possibly related
concepts. Your task is to analyze these concepts and categorize them by relevance to the main finding.

The goal is to find the most relevant concepts that accurately represent the finding, even if they 
use slightly different wording. Obviously, prioritize exact matches above all else.

Categories:
1. **exact_matches** (max 3): Concepts that exactly match the name--that is, they refer to the
     exact same idea (or very nearly so), even if they use slightly different wording.
   - CRITICAL: Any concept with text that exactly equals the finding name MUST go here
   - Include concepts whose normalized text exactly matches the finding name or its synonyms
   - **PRIORITIZE SNOMEDCT**: If multiple exact matches exist, include the SNOMEDCT version
   - These are the most important - never miss an exact match!
   
2. **should_include** (max 10): Highly relevant related concepts
   - Closely related medical conditions--for example, subtypes or variants of the finding
   - Concepts that medical professionals would strongly associate or expect to see together
   
3. **marginal** (max 10): Peripherally related concepts
   - Broader categories or parent conditions
   - Related but distinct conditions
   - Concepts with weaker associations

Do NOT include terms that are unrelated or do not refer to things that could be imaging findings
or diagnoses, like drug names, procedures, or anatomical structures.

IMPORTANT: 
- Return ONLY relevant concepts--concepts that might have the same words but which refer to
  unrelated ideas should NOT be included in ANY category.
- Return only the concept IDs in each category
- A concept can only appear in one category
- Prioritize exact name matches above all else
- Make your rationale CONCISE (1-2 sentences max)""",
    )


def ensure_exact_matches_post_process(
    output: CategorizedConcepts,
    search_results: list[OntologySearchResult],
    query_terms: list[str],
) -> CategorizedConcepts:
    """Post-process categorization output to ensure exact matches are properly identified.

    This function ensures that any concept whose normalized text exactly
    matches any of the query terms is included in exact_matches.
    If any are missing, it automatically corrects the categorization.

    Args:
        output: The categorization output from the agent
        search_results: List of search results to check for exact matches
        query_terms: List of search terms used for querying

    Returns:
        CategorizedConcepts with exact matches properly identified and corrected
    """
    # Get lowercase versions of query terms for comparison
    query_terms_lower = [term.lower() for term in query_terms]

    # Find concepts that should be exact matches
    missing_exact_matches = []

    for result in search_results:
        # Normalize the concept text for comparison (already normalized in search)
        normalized_text = result.concept_text.lower()

        # Check if this is an exact match for any query term
        if normalized_text in query_terms_lower and result.concept_id not in output.exact_matches:
            matched_term = query_terms[query_terms_lower.index(normalized_text)]
            missing_exact_matches.append((result.concept_id, result.concept_text, matched_term))

    # If we found missing exact matches, auto-correct by adding them
    if missing_exact_matches:
        # Create corrected lists
        corrected_exact_matches = list(output.exact_matches)
        corrected_should_include = list(output.should_include)
        corrected_marginal = list(output.marginal)

        # Add missing exact matches and remove from other categories
        # Respect the max_length constraint of 5
        for concept_id, concept_text, matched_term in missing_exact_matches:
            # Check if we've hit the limit
            if len(corrected_exact_matches) >= 5:
                logger.warning(f"Cannot add {concept_id} to exact_matches - already at limit of 5")
                break

            corrected_exact_matches.append(concept_id)

            # Remove from other categories if present
            if concept_id in corrected_should_include:
                corrected_should_include.remove(concept_id)
            if concept_id in corrected_marginal:
                corrected_marginal.remove(concept_id)

            logger.info(
                f"Auto-corrected: Added {concept_id} ('{concept_text}') to exact_matches (matched '{matched_term}')"
            )

        # Return corrected categorization
        return CategorizedConcepts(
            exact_matches=corrected_exact_matches,
            should_include=corrected_should_include,
            marginal=corrected_marginal,
            rationale=f"{output.rationale} [Auto-corrected: Added {min(len(missing_exact_matches), 5 - len(output.exact_matches))} missing exact matches]",
        )

    # No corrections needed - return original output
    return output


def create_query_generator_agent() -> Agent[None, list[str]]:
    """Create agent for generating alternative medical terms for ontology matching.

    Returns an agent that generates different ways to express the same medical finding
    to help match against formal medical ontologies.
    """
    model = get_openai_model(settings.openai_default_model_small)  # Use small/fast model

    return Agent[None, list[str]](
        model=model,
        output_type=list[str],
        system_prompt="""We need to find terms that might match a radiology finding name in official medical ontologies that use formal medical terminology.

Our finding name might be expressed in different ways, which would prevent an exact match. For example, our finding might be named "quadriceps tendon rupture", but the ontology we're searching doesn't have that exact term, but it DOES have "quadriceps tendon tear". 

Generate a list of alternative terms that express the same medical concept. Include:
- Alternative medical terms (e.g., "tear" vs "rupture")
- Common abbreviations (e.g., "PE" for "pulmonary embolism")
- More general or specific variations
- Parent/broader terms that might categorize this finding

Example for "quadriceps tendon rupture":
["quadriceps tendon tear", "quad tendon rupture", "quadriceps tendon disruption", "torn quadriceps tendon", "quadriceps tendon injury", "quadriceps tendinopathy", "quadriceps tendon abnormality"]

Return 2-5 terms that would appear in formal medical ontologies. Keep it simple and practical.""",
    )


async def generate_finding_query_terms(finding_name: str, finding_description: str | None = None) -> list[str]:
    """Generate alternative search terms for matching a finding in medical ontologies.

    Stage 1 of the ontology concept search pipeline.
    Uses AI to generate different ways of expressing the same medical finding
    to improve matching against formal medical ontology databases.

    Args:
        finding_name: Name of the imaging finding
        finding_description: Optional detailed description (currently used for context if provided)

    Returns:
        List of search terms including the original finding name and alternatives
    """
    query_agent = create_query_generator_agent()

    # Build prompt - keep it simple and focused
    prompt = f"Imaging finding: {finding_name}"
    if finding_description:
        prompt += f"\n(Additional context: {finding_description})"
    prompt += "\n\nGenerate alternative terms to help match this finding in medical ontologies."

    try:
        result = await query_agent.run(prompt)
        query_terms = result.output

        # Always include the original finding name as the first term
        if finding_name not in query_terms:
            query_terms = [finding_name, *query_terms]

        logger.info(f"Generated {len(query_terms)} search queries for '{finding_name}'")
        return query_terms
    except Exception as e:
        logger.warning(f"Failed to generate query terms via agent: {e}, using fallback")
        # Fallback to just the finding name
        return [finding_name]


async def categorize_with_validation(
    finding_name: str,
    search_results: list[OntologySearchResult],
    query_terms: list[str],
) -> CategorizedConcepts:
    """Categorize search results with automatic validation.

    Stage 3 of the ontology concept search pipeline.
    Uses AI to categorize results into relevance tiers, then applies
    post-processing to ensure exact matches are never missed.

    Args:
        finding_name: Name of the imaging finding
        search_results: List of ontology search results to categorize
        query_terms: List of search terms used for querying

    Returns:
        CategorizedConcepts with concept IDs in each relevance tier
    """
    # Create categorization context
    categorization_context = CategorizationContext(
        finding_name=finding_name, search_results=search_results, query_terms=query_terms
    )

    # Use the categorization agent
    categorization_agent = create_categorization_agent()

    # Create a compact representation of results for the prompt
    compact_results = [{"id": r.concept_id, "text": r.concept_text} for r in search_results]

    categorize_prompt = (
        f"Categorize these ontology concepts for the imaging finding '{finding_name}':\n\n"
        f"{json.dumps(compact_results, indent=2)}\n\n"
        f"Return concept IDs in appropriate categories based on relevance."
    )

    # Run categorization
    try:
        categorization_result = await categorization_agent.run(categorize_prompt, deps=categorization_context)
        categorized = categorization_result.output

        # Apply post-processing to ensure exact matches are properly identified
        corrected_categorized = ensure_exact_matches_post_process(
            output=categorized, search_results=search_results, query_terms=query_terms
        )

        logger.info(
            f"Categorization complete: {len(corrected_categorized.exact_matches)} exact, "
            f"{len(corrected_categorized.should_include)} should include, "
            f"{len(corrected_categorized.marginal)} marginal"
        )
        return corrected_categorized
    except Exception as e:
        logger.error(f"Categorization failed: {e}")
        raise


def build_final_output(
    categorized: CategorizedConcepts,
    search_results: list[OntologySearchResult],
    max_exact_matches: int = 5,
    max_should_include: int = 10,
    max_marginal: int = 10,
) -> CategorizedOntologyConcepts:
    """Transform categorized concept IDs to final output format.

    Stage 4 of the ontology concept search pipeline.
    Converts concept IDs back to full OntologySearchResult objects and
    applies maximum limits to each category.

    Args:
        categorized: Categorized concept IDs from Stage 3
        search_results: Original search results to map IDs back to
        max_exact_matches: Maximum exact match concepts to return
        max_should_include: Maximum should-include concepts to return
        max_marginal: Maximum marginal concepts to return

    Returns:
        CategorizedOntologyConcepts ready for API response
    """
    # Create mapping from concept ID to full result object
    result_map = {r.concept_id: r for r in search_results}

    # Convert categorized IDs to full result objects with limits
    exact_matches = []
    should_include = []
    marginal_concepts = []

    # Map exact matches (apply limit)
    for concept_id in categorized.exact_matches[:max_exact_matches]:
        if concept_id in result_map:
            exact_matches.append(result_map[concept_id])
        else:
            logger.warning(f"Categorized concept ID {concept_id} not found in results")

    # Map should include (apply limit)
    for concept_id in categorized.should_include[:max_should_include]:
        if concept_id in result_map:
            should_include.append(result_map[concept_id])
        else:
            logger.warning(f"Categorized concept ID {concept_id} not found in results")

    # Map marginal concepts (apply limit)
    for concept_id in categorized.marginal[:max_marginal]:
        if concept_id in result_map:
            marginal_concepts.append(result_map[concept_id])
        else:
            logger.warning(f"Categorized concept ID {concept_id} not found in results")

    logger.info(
        f"Final output: {len(exact_matches)} exact matches, "
        f"{len(should_include)} should include, "
        f"{len(marginal_concepts)} marginal concepts"
    )

    return CategorizedOntologyConcepts(
        exact_matches=exact_matches,
        should_include=should_include,
        marginal_concepts=marginal_concepts,
        search_summary=categorized.rationale,
        excluded_anatomical=[],  # Could be populated if we track filtered anatomical concepts
    )


async def match_ontology_concepts(
    finding_name: str,
    finding_description: str | None = None,
    exclude_anatomical: bool = True,
    max_exact_matches: int = 5,
    max_should_include: int = 10,
    max_marginal: int = 10,
    ontologies: list[str] | None = None,
) -> CategorizedOntologyConcepts:
    """
    Match finding to relevant ontology concepts using BioOntology API.

    This is the main orchestration function that coordinates a 4-stage pipeline:
    1. Generate comprehensive query terms using AI
    2. Execute search using BioOntology API
    3. Categorize results with automatic validation
    4. Transform to final output format with limits

    Args:
        finding_name: Name of the finding model
        finding_description: Optional detailed description
        exclude_anatomical: Whether to filter out anatomical concepts
        max_exact_matches: Maximum exact match concepts to return
        max_should_include: Maximum should-include concepts
        max_marginal: Maximum marginal concepts to consider
        ontologies: Optional list of ontology acronyms to search (default: SNOMEDCT, RADLEX, LOINC)

    Returns:
        Categorized ontology concepts with rationale

    Raises:
        ValueError: If BioOntology API key is not configured
    """
    logger.info(f"Starting ontology concept matching for: {finding_name}")

    try:
        # Stage 1: Generate comprehensive search terms
        query_terms = await generate_finding_query_terms(finding_name, finding_description)

        # Stage 2: Execute search using BioOntology API
        search_results = await execute_ontology_search(
            query_terms=query_terms,
            exclude_anatomical=exclude_anatomical,
            base_limit=30,  # Cast wider net initially
            max_results=7,  # Focus on top results (plus exact matches)
            ontologies=ontologies,  # Pass through ontologies parameter
        )

        # Stage 3: Categorize with automatic validation
        categorized = await categorize_with_validation(
            finding_name=finding_name, search_results=search_results, query_terms=query_terms
        )

        # Stage 4: Transform to final output format
        return build_final_output(
            categorized=categorized,
            search_results=search_results,
            max_exact_matches=max_exact_matches,
            max_should_include=max_should_include,
            max_marginal=max_marginal,
        )

    except Exception as e:
        logger.error(f"Error in ontology concept matching: {e}")
        raise


__all__ = [
    "CategorizationContext",
    "CategorizedConcepts",
    "CategorizedOntologyConcepts",
    "build_final_output",
    "categorize_with_validation",
    "create_categorization_agent",
    "create_query_generator_agent",
    "ensure_exact_matches_post_process",
    "execute_ontology_search",
    "generate_finding_query_terms",
    "match_ontology_concepts",
]
