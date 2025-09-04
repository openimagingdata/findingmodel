"""
Ontology Concept Search Tool

Uses two specialized Pydantic AI agents to find and categorize relevant medical concepts
from ontology databases, excluding anatomical structures.

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
from findingmodel.tools.ontology_search import OntologySearchClient, OntologySearchResult, normalize_concept


class RawSearchResults(BaseModel):
    """Output from search agent."""

    search_terms_used: list[str] = Field(description="List of search terms that were actually used")
    all_results: list[OntologySearchResult] = Field(description="All unique search results found across tables")
    anatomical_filtered: list[str] = Field(
        default_factory=list, description="Anatomical concepts that were filtered out"
    )


class QueryTerms(BaseModel):
    """Structured query terms for comprehensive medical concept search."""

    primary_term: str = Field(description="The main medical finding name as provided")
    synonyms: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Medical synonyms and alternative names for the finding (max 10)",
    )
    related_terms: list[str] = Field(
        default_factory=list, max_length=5, description="Related medical conditions or findings (max 5)"
    )

    @property
    def all_terms(self) -> list[str]:
        """Return all unique terms in priority order: primary, synonyms, then related."""
        terms = [self.primary_term]

        # Add synonyms that aren't duplicates of primary
        for synonym in self.synonyms:
            if synonym.lower() != self.primary_term.lower() and synonym not in terms:
                terms.append(synonym)

        # Add related terms that aren't duplicates
        for related in self.related_terms:
            if related.lower() not in [t.lower() for t in terms] and related not in terms:
                terms.append(related)

        return terms


class SearchQueries(BaseModel):
    """Search queries for medical concept search."""

    queries: list[str] = Field(description="List of medical search terms, synonyms, and related conditions")


class CategorizedConcepts(BaseModel):
    """Categorized concept IDs with validation constraints."""

    exact_matches: list[str] = Field(description="Concept IDs that exactly match the finding", max_length=5)
    should_include: list[str] = Field(description="Concept IDs that should be included", max_length=10)
    marginal: list[str] = Field(description="Concept IDs that are marginally relevant", max_length=10)
    rationale: str = Field(description="Brief categorization rationale")


class SimpleCategorizedConcepts(BaseModel):
    """Simple categorized concepts for fast processing."""

    exact_matches: list[str] = Field(description="Concept IDs that exactly match the finding")
    should_include: list[str] = Field(description="Concept IDs that should be included")
    marginal_concepts: list[str] = Field(description="Concept IDs that are marginally relevant")
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
class SearchContext:
    """Context for search operations."""

    search_client: OntologySearchClient
    finding_name: str
    finding_description: str | None
    exclude_anatomical: bool


@dataclass
class CategorizationContext:
    """Context for categorization agent with dependencies."""

    finding_name: str
    search_results: list[OntologySearchResult]
    query_terms: QueryTerms


async def generate_search_queries(finding_name: str, finding_description: str | None = None) -> list[str]:
    """Generate search queries using the query generator agent.

    Args:
        finding_name: The name of the medical finding
        finding_description: Optional detailed description of the finding

    Returns:
        List of search queries in priority order
    """
    query_agent = create_query_generator_agent()

    # Create structured prompt
    prompt = f"Medical finding: {finding_name}"
    if finding_description:
        prompt += f"\nDescription: {finding_description}"
    prompt += "\n\nGenerate comprehensive search terms for ontology database queries."

    try:
        result = await query_agent.run(prompt)

        # Get all terms from the structured output
        # The QueryTerms model already handles deduplication and priority ordering
        queries = result.output.all_terms

        # Limit to a reasonable number for search performance
        return queries[:15]  # Increased limit since we have better structured output

    except Exception as e:
        logger.warning(f"Failed to generate query terms via agent: {e}, using fallback")
        # Simple fallback - just the original term
        return [finding_name]


async def execute_ontology_search(
    query_terms: QueryTerms,
    client: OntologySearchClient,
    exclude_anatomical: bool = True,
    base_limit: int = 30,
    max_results: int = 12,
) -> list[OntologySearchResult]:
    """Execute ontology search with smart result selection strategy.

    This function executes a parallel search across ontology databases and applies
    a smart selection strategy to ensure both high-scoring results and exact matches
    are included in the final result set.

    Strategy:
    1. Execute parallel search with all query terms (primary, synonyms, related)
    2. Sort all results by score in descending order
    3. Take the top N results by score
    4. Check remaining results for exact matches of the primary term
    5. Add any missing exact matches (important for cases like RID5350 for "pneumonia")
    6. Apply concept text normalization to all results

    Args:
        query_terms: Structured query terms containing primary term, synonyms, and related terms
        client: OntologySearchClient instance for database access
        exclude_anatomical: Whether to filter out anatomical structure concepts (default: True)
        base_limit: Initial limit per query for casting a wider net (default: 30)
        max_results: Maximum number of results to return after selection (default: 12)

    Returns:
        List of OntologySearchResult objects with normalized concept text,
        sorted by relevance score with exact matches guaranteed to be included.

    Raises:
        Exception: If the search operation fails
    """
    try:
        # Get all search terms from the QueryTerms object
        search_queries = query_terms.all_terms
        logger.info(f"Executing ontology search with {len(search_queries)} terms")

        # Execute parallel search across all ontology tables
        search_results = await client.search_parallel(
            queries=search_queries,
            tables=None,  # Search all available tables
            limit_per_query=base_limit,  # Cast wider net initially
            filter_anatomical=exclude_anatomical,
        )

        logger.info(f"Search returned {len(search_results)} total results")

        if not search_results:
            logger.warning("No search results found")
            return []

        # Sort all results by score (descending)
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)

        # Take the top N results by score
        top_results = sorted_results[:max_results]
        selected_ids = {r.concept_id for r in top_results}

        # Check remaining results for exact matches of the primary term
        # This ensures critical exact matches aren't missed due to lower scores
        primary_term_lower = query_terms.primary_term.lower()
        exact_matches_added = 0

        for result in sorted_results[max_results:]:
            # Normalize the concept text for comparison
            normalized_text = normalize_concept(result.concept_text).lower()

            # Check if this is an exact match for the primary term
            if normalized_text == primary_term_lower and result.concept_id not in selected_ids:
                top_results.append(result)
                selected_ids.add(result.concept_id)
                exact_matches_added += 1
                logger.info(
                    f"Added exact match for primary term '{query_terms.primary_term}': "
                    f"{result.concept_id} ({result.concept_text}) [score: {result.score:.3f}]"
                )

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


# Search agent no longer needed - using programmatic query generation


def create_old_categorization_agent() -> Agent[SearchContext, CategorizedOntologyConcepts]:
    """Create the categorization agent optimized for speed.

    DEPRECATED: Use create_categorization_agent() for better validation.
    """

    model = get_openai_model(settings.openai_default_model)

    return Agent[SearchContext, CategorizedOntologyConcepts](
        model=model,
        output_type=CategorizedOntologyConcepts,
        deps_type=SearchContext,
        system_prompt="""Medical ontology categorization. Analyze concepts for finding relevance.

Categories:
1. **exact_matches**: Direct matches to finding name
2. **should_include**: Highly relevant concepts  
3. **marginal_concepts**: Peripherally relevant

Prioritize exact name matches. Be concise in rationale.""",
    )


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
        system_prompt="""You are a medical ontology categorization expert.

Analyze the provided ontology concepts and categorize them by relevance to the medical finding.

Categories:
1. **exact_matches** (max 5): Concepts that directly match the finding name (case-insensitive)
   - CRITICAL: Any concept with text that exactly equals the finding name MUST go here
   - Include concepts whose normalized text exactly matches the finding name or its synonyms
   - These are the most important - never miss an exact match!
   
2. **should_include** (max 10): Highly relevant related concepts
   - Closely related medical conditions
   - Specific subtypes or variants
   - Concepts that medical professionals would strongly associate
   
3. **marginal** (max 10): Peripherally related concepts
   - Broader categories or parent conditions
   - Related but distinct conditions
   - Concepts with weaker associations

IMPORTANT: 
- Return only the concept IDs in each category
- Prioritize exact name matches above all else
- Be concise in your rationale
- A concept can only appear in one category""",
    )


def ensure_exact_matches_post_process(
    output: CategorizedConcepts,
    search_results: list[OntologySearchResult],
    query_terms: QueryTerms,
) -> CategorizedConcepts:
    """Post-process categorization output to ensure exact matches are properly identified.

    This function ensures that any concept whose normalized text exactly
    matches the finding name or its query terms is included in exact_matches.
    If any are missing, it automatically corrects the categorization.

    Args:
        output: The categorization output from the agent
        search_results: List of search results to check for exact matches
        query_terms: Original query terms used for searching

    Returns:
        CategorizedConcepts with exact matches properly identified and corrected
    """
    # Get the context data
    query_terms_lower = [term.lower() for term in query_terms.all_terms]

    # Find concepts that should be exact matches
    missing_exact_matches = []

    for result in search_results:
        # Normalize the concept text for comparison (already normalized in search)
        normalized_text = result.concept_text.lower()

        # Check if this is an exact match for any query term
        is_exact_match = False
        matched_term = None

        for query_term in query_terms_lower:
            if normalized_text == query_term:
                is_exact_match = True
                matched_term = query_term
                break

        # If it's an exact match but not in the exact_matches list, track it
        if is_exact_match and result.concept_id not in output.exact_matches:
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


def create_fast_categorization_agent() -> Agent[SearchContext, SimpleCategorizedConcepts]:
    """Create a fast categorization agent with simple output model.

    DEPRECATED: Use create_categorization_agent() for better validation.
    """

    model = get_openai_model(settings.openai_default_model)

    return Agent[SearchContext, SimpleCategorizedConcepts](
        model=model,
        output_type=SimpleCategorizedConcepts,
        deps_type=SearchContext,
        system_prompt="""Categorize medical concepts by relevance to finding.
Return concept IDs only in each category. Be fast and accurate.""",
    )


def create_query_generator_agent() -> Agent[None, QueryTerms]:
    """Create agent for generating comprehensive medical query terms.

    Returns an agent that generates structured query terms for ontology search,
    focusing on terms that would appear in medical ontologies like SNOMED-CT,
    RadLex, ICD-10, and LOINC.
    """
    model = get_openai_model("gpt-4o-mini")  # Using mini model for efficiency

    return Agent[None, QueryTerms](
        model=model,
        output_type=QueryTerms,
        system_prompt="""You are a medical terminology expert specializing in ontology search.

Your task is to generate comprehensive search terms for medical findings that will be used 
to query medical ontology databases (SNOMED-CT, RadLex, ICD-10, LOINC).

Given a medical finding name and optional description, generate:

1. **Primary Term**: Keep the original finding name exactly as provided

2. **Synonyms** (up to 10): Generate medical synonyms focusing on:
   - Standard medical terminology variants
   - Common clinical abbreviations (e.g., "PE" for "pulmonary embolism")
   - Alternative spellings (British vs American medical terms)
   - Formal/informal medical terms
   - Historical or legacy terminology still in use
   - Plural/singular variations if clinically relevant
   
3. **Related Terms** (up to 5): Include closely related medical concepts:
   - Parent conditions (e.g., "pneumonia" for "lobar pneumonia")
   - Subtypes or specific variants
   - Associated pathological processes
   - Differential diagnoses commonly grouped together

IMPORTANT GUIDELINES:
- Focus on terms that would actually appear in medical ontologies
- Avoid overly specific anatomical qualifiers unless inherent to the finding
- Include both abbreviated and full forms when applicable
- Prioritize commonly used clinical terminology
- Ensure terms are medically accurate and clinically relevant
- Do not include treatment or procedure terms unless they are part of the finding name

Example:
For "pulmonary embolism":
- Synonyms: ["PE", "pulmonary thromboembolism", "lung embolism", "pulmonary embolus", ...]
- Related: ["venous thromboembolism", "thromboembolism", "embolism", ...]
""",
    )


def create_synonym_generation_agent() -> Agent[None, SearchQueries]:
    """Create agent for generating medical synonyms and related terms.

    DEPRECATED: Use create_query_generator_agent() instead for better structured output.
    """
    model = get_openai_model(settings.openai_default_model)

    return Agent[None, SearchQueries](
        model=model,
        output_type=SearchQueries,
        system_prompt="""Generate medical synonyms and related search terms.

For a given medical finding, provide 5-8 relevant search terms including:
- Direct synonyms
- Alternative medical terminology  
- Related conditions
- Common abbreviations
- Anatomically specific variants

Always include the original term first.""",
    )


async def generate_query_terms(finding_name: str, finding_description: str | None = None) -> QueryTerms:
    """Generate comprehensive search terms for ontology queries.

    Stage 1 of the ontology concept search pipeline.
    Uses AI to generate synonyms, abbreviations, and related medical terms.

    Args:
        finding_name: Name of the medical finding
        finding_description: Optional detailed description

    Returns:
        QueryTerms object with primary term, synonyms, and related terms
    """
    query_agent = create_query_generator_agent()

    # Build prompt
    prompt = f"Medical finding: {finding_name}"
    if finding_description:
        prompt += f"\nDescription: {finding_description}"
    prompt += "\n\nGenerate comprehensive search terms for ontology database queries."

    try:
        result = await query_agent.run(prompt)
        query_terms = result.output
        logger.info(f"Generated {len(query_terms.all_terms)} search queries for '{finding_name}'")
        return query_terms
    except Exception as e:
        logger.warning(f"Failed to generate query terms via agent: {e}, using fallback")
        # Fallback to simple QueryTerms with just the finding name
        return QueryTerms(primary_term=finding_name, synonyms=[], related_terms=[])


async def categorize_with_validation(
    finding_name: str,
    search_results: list[OntologySearchResult],
    query_terms: QueryTerms,
) -> CategorizedConcepts:
    """Categorize search results with automatic validation.

    Stage 3 of the ontology concept search pipeline.
    Uses AI to categorize results into relevance tiers, then applies
    post-processing to ensure exact matches are never missed.

    Args:
        finding_name: Name of the medical finding
        search_results: List of ontology search results to categorize
        query_terms: Original query terms used for searching

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
    compact_results = [{"id": r.concept_id, "text": r.concept_text, "score": round(r.score, 3)} for r in search_results]

    categorize_prompt = (
        f"Categorize these ontology concepts for the medical finding '{finding_name}':\n\n"
        f"{json.dumps(compact_results, indent=2)}\n\n"
        f"Return concept IDs in appropriate categories based on medical relevance."
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


async def search_ontology_concepts(
    finding_name: str,
    finding_description: str | None = None,
    exclude_anatomical: bool = True,
    max_exact_matches: int = 5,
    max_should_include: int = 10,
    max_marginal: int = 10,
) -> CategorizedOntologyConcepts:
    """
    Search for relevant ontology concepts across all tables.

    This is the main orchestration function that coordinates a 4-stage pipeline:
    1. Generate comprehensive query terms using AI
    2. Execute smart search with exact match guarantee
    3. Categorize results with automatic validation
    4. Transform to final output format with limits

    Args:
        finding_name: Name of the finding model
        finding_description: Optional detailed description
        exclude_anatomical: Whether to filter out anatomical concepts
        max_exact_matches: Maximum exact match concepts to return
        max_should_include: Maximum should-include concepts
        max_marginal: Maximum marginal concepts to consider

    Returns:
        Categorized ontology concepts with rationale
    """
    logger.info(f"Starting ontology concept search for: {finding_name}")

    # Initialize search client
    client = OntologySearchClient(
        lancedb_uri=settings.lancedb_uri,
        api_key=settings.lancedb_api_key.get_secret_value() if settings.lancedb_api_key else None,
    )

    try:
        await client.connect()

        # Stage 1: Generate comprehensive search terms
        query_terms = await generate_query_terms(finding_name, finding_description)

        # Stage 2: Execute smart search with exact match guarantee
        search_results = await execute_ontology_search(
            query_terms=query_terms,
            client=client,
            exclude_anatomical=exclude_anatomical,
            base_limit=30,  # Cast wider net initially
            max_results=7,  # Focus on top results (plus exact matches)
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

    finally:
        client.disconnect()


__all__ = [
    "CategorizationContext",
    "CategorizedConcepts",
    "CategorizedOntologyConcepts",
    "QueryTerms",
    "build_final_output",
    "categorize_with_validation",
    "create_categorization_agent",
    "create_query_generator_agent",
    "ensure_exact_matches_post_process",
    "execute_ontology_search",
    "generate_query_terms",
    "search_ontology_concepts",
]
