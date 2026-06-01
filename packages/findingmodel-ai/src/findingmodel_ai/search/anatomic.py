"""
Anatomic Location Search Tool

Uses a query-generation agent plus deterministic index search to gather candidate anatomic
locations. Final applicability decisions belong to the metadata anatomy decision agent.
"""

import re
from typing import Literal

from anatomic_locations import AnatomicLocationIndex
from anatomic_locations.models import AnatomicLocation
from findingmodel.protocols import OntologySearchResult, normalize_concept
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from findingmodel_ai import logger
from findingmodel_ai.config import settings

SupportLevel = Literal[
    "direct_source",
    "source_inferred_query",
    "parent_of_supported",
    "child_of_supported",
    "ontology_context",
    "current_metadata",
    "search_only",
]

_SUPPORT_RANK: dict[SupportLevel, int] = {
    "direct_source": 0,
    "source_inferred_query": 1,
    "parent_of_supported": 2,
    "ontology_context": 3,
    "child_of_supported": 4,
    "current_metadata": 5,
    "search_only": 6,
}


def _convert_to_ontology_results(locations: list[AnatomicLocation]) -> list[OntologySearchResult]:
    """Convert AnatomicLocation objects to OntologySearchResult.

    Args:
        locations: List of AnatomicLocation objects from index search

    Returns:
        List of OntologySearchResult objects for compatibility with existing interfaces
    """
    return [
        OntologySearchResult(
            concept_id=loc.id,
            concept_text=loc.description,
            score=0.0,  # AnatomicLocationIndex doesn't expose search scores
            table_name="anatomic_locations",
        )
        for loc in locations
    ]


def _append_unique_term(terms: list[str], term: str | None) -> None:
    if not term:
        return
    cleaned = " ".join(term.split())
    if not cleaned:
        return
    if cleaned.lower() not in {existing.lower() for existing in terms}:
        terms.append(cleaned)


_GENERIC_LOCATION_SUFFIXES = (
    " abnormality",
    " abnormalities",
    " assessment",
    " assessments",
    " catheter",
    " catheters",
    " classification",
    " classifications",
    " disease",
    " diseases",
    " finding",
    " findings",
    " index",
    " lesion",
    " lesions",
    " line",
    " lines",
    " mass",
    " masses",
    " measurement",
    " measurements",
    " score",
    " sign",
    " tube",
    " tubes",
)

_ANATOMIC_TERM_VARIANTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("vertebra", "vertebrae"), ("vertebral column", "spine")),
)


def _humanize_term(term: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_/()]+", " ", term)).strip()


def _clean_location_phrase(term: str) -> str:
    return re.sub(r"^(the|a|an)\s+", "", term.strip(), flags=re.IGNORECASE)


def _term_variants(term: str) -> list[str]:
    variants: list[str] = []
    humanized = _humanize_term(term)
    if not humanized:
        return variants
    variants.append(humanized)

    lowered = humanized.lower()
    for preposition in (" in ", " of ", " at ", " within "):
        if preposition in lowered:
            tail = _clean_location_phrase(humanized[lowered.rindex(preposition) + len(preposition) :])
            if tail:
                variants.append(tail)

    for suffix in _GENERIC_LOCATION_SUFFIXES:
        if lowered.endswith(suffix):
            variants.append(humanized[: -len(suffix)])

    for triggers, additions in _ANATOMIC_TERM_VARIANTS:
        if any(_matches_anatomic_variant_trigger(lowered, trigger) for trigger in triggers):
            variants.extend(additions)

    return variants


def _matches_anatomic_variant_trigger(lowered_term: str, trigger: str) -> bool:
    if trigger.endswith("*"):
        return re.search(rf"\b{re.escape(trigger[:-1])}", lowered_term) is not None
    return re.search(rf"\b{re.escape(trigger)}\b", lowered_term) is not None


def _append_term_variants(terms: list[str], values: list[str] | None) -> None:
    for value in values or []:
        for term in _term_variants(value):
            _append_unique_term(terms, term)


def _description_phrases(description: str | None) -> list[str]:
    if not description:
        return []
    for separator in (";", ",", "."):
        description = description.replace(separator, "|")
    return [phrase for phrase in description.split("|") if 2 <= len(phrase.split()) <= 6]


def _explicit_context_terms(
    finding_name: str,
    description: str | None,
    synonyms: list[str] | None,
    attribute_labels: list[str] | None,
    locality_labels: list[str] | None,
    source_labels: list[str] | None = None,
) -> list[str]:
    terms: list[str] = []
    _append_term_variants(terms, [finding_name])
    _append_term_variants(terms, synonyms)
    _append_term_variants(terms, source_labels)
    _append_term_variants(terms, attribute_labels)
    _append_term_variants(terms, locality_labels)
    _append_term_variants(terms, _description_phrases(description))
    return terms[:20]


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


class AnatomicLocationCandidate(BaseModel):
    """One anatomic candidate plus why it was offered."""

    location: OntologySearchResult
    support_level: SupportLevel
    matched_terms: list[str] = Field(default_factory=list)
    broader_candidate_ids: list[str] = Field(default_factory=list)


class AnatomicCandidateSearchResponse(BaseModel):
    """Candidate-gathering response for anatomy assignment."""

    candidates: list[AnatomicLocationCandidate] = Field(default_factory=list)
    query_terms: list[str] = Field(default_factory=list)
    reasoning: str = ""


async def generate_anatomic_query_terms(
    finding_name: str,
    finding_description: str | None = None,
    *,
    synonyms: list[str] | None = None,
    attribute_labels: list[str] | None = None,
    locality_labels: list[str] | None = None,
    source_labels: list[str] | None = None,
) -> AnatomicQueryTerms:
    """Generate anatomic location search terms for a finding.

    First identifies the most appropriate anatomic location,
    then generates ontology term variations.

    Args:
        finding_name: Name of the finding
        finding_description: Optional detailed description

    Returns:
        List of anatomic location search terms
    """
    agent = Agent[None, AnatomicQueryTerms](
        model=settings.get_agent_model("anatomic_search"),
        output_type=AnatomicQueryTerms,
        system_prompt="""You are an anatomic location specialist for medical imaging findings.
        
Given a medical finding, you must:
1. First identify the REGION and PRIMARY anatomic location where this finding occurs
2. Then generate 3-5 ontology term variations for that location

- Focus on formal medical terminology used in ontologies.
- Do NOT include acronyms or layman terms. 
- Do NOT include bare adjectives (e.g., "abdominal", "cervical")--we're looking for nouns/noun phrases
- Do NOT separately search for left and right; only search for general terms.
- Include source-backed parent terms when they cover the whole modeled scope.
- When a word can name different anatomy in different contexts, use the finding description,
  synonyms, source ontology labels, and attribute labels to choose the intended anatomic meaning.
  Search for the intended organ or system, not every homonym.
- For generic vessel findings with no named territory, search broad vessel-system terms rather than
  lists of named vessels. A generic artery-to-vein connection should search vessel systems, not
  femoral vein, portal vein, or other site-specific vessels.
- Translate vascular relationship words into participating systems for search. For example,
  arteriovenous or artery-to-vein findings should search `arterial system` and other broad vascular
  system terms; do not search only the disease/finding name.
- For a tunneled central venous catheter described as having a subcutaneous tunnel, include
  "anterior chest wall" and "chest wall" as candidate search terms.

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

Example:
Finding: "arterial stent"
Primary location: arterial system
Region: "Body"
Terms: ["arterial system", "artery"]

Example:
Finding: "generic vascular connection"
Primary location: vascular system
Region: "Body"
Terms: ["vascular system", "arterial system", "venous system"]

Example:
Finding: "arteriovenous fistulas"
Description: "Abnormal connection between an artery and a vein."
Primary location: arterial and venous systems
Region: "Body"
Terms: ["arterial system", "vascular system"]

Return ONLY the region and list of terms, nothing else.""",
    )

    prompt = f"Finding: {finding_name}"
    if finding_description:
        prompt += f"\nDescription: {finding_description}"
    if synonyms:
        prompt += f"\nSynonyms: {', '.join(synonyms)}"
    if attribute_labels:
        prompt += f"\nAttribute/locality labels: {', '.join(attribute_labels)}"
    if locality_labels:
        prompt += f"\nAdditional locality labels: {', '.join(locality_labels)}"
    if source_labels:
        prompt += f"\nSource ontology labels: {', '.join(source_labels)}"

    try:
        result = await agent.run(prompt)
        terms = result.output.terms

        for term in _explicit_context_terms(
            finding_name,
            finding_description,
            synonyms,
            attribute_labels,
            locality_labels,
            source_labels,
        ):
            _append_unique_term(terms, term)

        logger.info(f"Generated {len(terms)} anatomic query terms for '{finding_name}'")
        return result.output
    except Exception as e:
        logger.warning(f"Failed to generate anatomic query terms: {e}, using fallback")
        # Fallback to just the finding name
        return AnatomicQueryTerms(
            region=None,
            terms=_explicit_context_terms(
                finding_name,
                finding_description,
                synonyms,
                attribute_labels,
                locality_labels,
                source_labels,
            ),
        )


async def execute_anatomic_search(
    query_info: AnatomicQueryTerms,
    index: AnatomicLocationIndex,
    limit: int = 30,
) -> list[OntologySearchResult]:
    """Execute search on anatomic_locations table with region and sided filtering.

    Returns results from hybrid search filtered by region and sided.

    Args:
        query_info: AnatomicQueryTerms containing terms and optional region
        index: AnatomicLocationIndex instance
        limit: Maximum results per query term (default 30)

    Returns:
        List of OntologySearchResult objects with normalized concept text
    """
    exact_locations = _exact_locations(query_info.terms, index)

    # Batch search all terms with a single embedding API call.
    batch_results = await index.search_batch(
        query_info.terms,
        limit=limit,
        region=query_info.region,
        sided_filter=["generic", "nonlateral"],  # Only generic or nonlateral sided
    )

    # Flatten and deduplicate by ID
    seen_ids: set[str] = set()
    unique_locations: list[AnatomicLocation] = []

    for loc in exact_locations:
        _append_location_and_relatives(loc, index, seen_ids=seen_ids, locations=unique_locations)
    for locations in batch_results.values():
        for loc in locations:
            _append_location_and_relatives(loc, index, seen_ids=seen_ids, locations=unique_locations)

    # Convert to OntologySearchResult format
    results = _convert_to_ontology_results(unique_locations)

    # Normalize concept text for all results
    for result in results:
        result.concept_text = normalize_concept(result.concept_text)

    logger.info(f"Found {len(results)} anatomic location results")
    return results


async def execute_anatomic_candidate_search(
    query_info: AnatomicQueryTerms,
    index: AnatomicLocationIndex,
    *,
    direct_terms: list[str],
    ontology_context_terms: list[str] | None = None,
    limit: int = 30,
) -> list[AnatomicLocationCandidate]:
    exact_locations = _exact_locations_with_terms(query_info.terms, index)
    batch_results = await index.search_batch(
        query_info.terms,
        limit=limit,
        region=None,
        sided_filter=["generic", "nonlateral"],
    )

    candidates: dict[str, AnatomicLocationCandidate] = {}
    direct_term_set = {normalize_concept(term) for term in direct_terms}
    ontology_term_set = {normalize_concept(term) for term in ontology_context_terms or []}

    for loc, term in exact_locations:
        normalized_term = normalize_concept(term)
        if normalized_term in direct_term_set:
            support_level: SupportLevel = "direct_source"
        elif normalized_term in ontology_term_set:
            support_level = "ontology_context"
        else:
            support_level = "source_inferred_query"
        ancestors = loc.get_containment_ancestors(index)[:3]
        _add_candidate(
            candidates,
            loc,
            support_level=support_level,
            matched_terms=[term],
            broader_candidate_ids=[ancestor.id for ancestor in ancestors],
        )
        for ancestor in ancestors:
            _add_candidate(
                candidates,
                ancestor,
                support_level="parent_of_supported",
                broader_candidate_ids=[],
            )
        for child in index.get_children_of(loc.id)[:5]:
            if str(child.laterality).split(".")[-1].lower() in {"generic", "nonlateral"}:
                _add_candidate(
                    candidates,
                    child,
                    support_level="child_of_supported",
                    broader_candidate_ids=[loc.id],
                )

    for locations in batch_results.values():
        for loc in locations:
            _add_candidate(candidates, loc, support_level="search_only")

    results = list(candidates.values())
    for candidate in results:
        candidate.location.concept_text = normalize_concept(candidate.location.concept_text)
    logger.info(f"Found {len(results)} anatomic candidate locations")
    return results


def _exact_locations(terms: list[str], index: AnatomicLocationIndex) -> list[AnatomicLocation]:
    locations: list[AnatomicLocation] = []
    for term in terms:
        try:
            location = index.get(term)
        except KeyError:
            continue
        if str(location.laterality).split(".")[-1].lower() in {"generic", "nonlateral"}:
            locations.append(location)
    return locations


def _exact_locations_with_terms(terms: list[str], index: AnatomicLocationIndex) -> list[tuple[AnatomicLocation, str]]:
    locations: list[tuple[AnatomicLocation, str]] = []
    for term in terms:
        try:
            location = index.get(term)
        except KeyError:
            continue
        if str(location.laterality).split(".")[-1].lower() in {"generic", "nonlateral"}:
            locations.append((location, term))
    return locations


def _add_candidate(
    candidates: dict[str, AnatomicLocationCandidate],
    loc: AnatomicLocation,
    *,
    support_level: SupportLevel,
    matched_terms: list[str] | None = None,
    broader_candidate_ids: list[str] | None = None,
) -> None:
    existing = candidates.get(loc.id)
    if existing is None:
        candidates[loc.id] = AnatomicLocationCandidate(
            location=OntologySearchResult(
                concept_id=loc.id,
                concept_text=normalize_concept(loc.description),
                score=0.0,
                table_name="anatomic_locations",
            ),
            support_level=support_level,
            matched_terms=matched_terms or [],
            broader_candidate_ids=broader_candidate_ids or [],
        )
        return
    if _SUPPORT_RANK[support_level] < _SUPPORT_RANK[existing.support_level]:
        existing.support_level = support_level
    for term in matched_terms or []:
        if term not in existing.matched_terms:
            existing.matched_terms.append(term)
    for candidate_id in broader_candidate_ids or []:
        if candidate_id not in existing.broader_candidate_ids:
            existing.broader_candidate_ids.append(candidate_id)


def _append_location_and_relatives(
    loc: AnatomicLocation,
    index: AnatomicLocationIndex,
    *,
    seen_ids: set[str],
    locations: list[AnatomicLocation],
) -> None:
    if loc.id not in seen_ids:
        seen_ids.add(loc.id)
        locations.append(loc)
    for ancestor in loc.get_containment_ancestors(index)[:3]:
        if ancestor.id not in seen_ids:
            seen_ids.add(ancestor.id)
            locations.append(ancestor)
    for child in index.get_children_of(loc.id)[:5]:
        if child.id not in seen_ids and str(child.laterality).split(".")[-1].lower() in {"generic", "nonlateral"}:
            seen_ids.add(child.id)
            locations.append(child)


class LocationSearchResponse(BaseModel):
    """Output from matching agent."""

    primary_location: OntologySearchResult = Field(description="Best primary anatomic location")
    alternate_locations: list[OntologySearchResult] = Field(description="Good alternate locations", max_length=8)
    reasoning: str = Field(description="Clear reasoning for selections made")


def create_location_selection_agent() -> Agent[None, LocationSearchResponse]:
    """Create agent for selecting best anatomic locations from search results.

    Returns:
        Agent configured for location selection
    """
    return Agent[None, LocationSearchResponse](
        model=settings.get_agent_model("anatomic_select"),
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
    synonyms: list[str] | None = None,
    attribute_labels: list[str] | None = None,
    locality_labels: list[str] | None = None,
    source_labels: list[str] | None = None,
    index: AnatomicLocationIndex | None = None,
    use_duckdb: bool = True,
) -> AnatomicCandidateSearchResponse:
    """Find relevant anatomic locations for a finding using 3-stage pipeline.

    Pipeline stages:
    1. Generate query terms using AI
    2. Execute direct search on anatomic_locations table
    3. Select best locations using AI agent

    Args:
        finding_name: Name of the finding (e.g., "PCL tear")
        description: Optional detailed description
        use_duckdb: Use DuckDB client if True, LanceDB if False (default True)

    Returns:
        Candidate locations with source-support evidence for the anatomy decision agent
    """
    logger.info(f"Starting anatomic location search for: {finding_name}")

    # Stage 1: Generate query terms
    query_info = await generate_anatomic_query_terms(
        finding_name,
        description,
        synonyms=synonyms,
        attribute_labels=attribute_labels,
        locality_labels=locality_labels,
        source_labels=source_labels,
    )
    logger.info(f"Generated query terms: {query_info.terms}, region: {query_info.region}")
    direct_terms = _explicit_context_terms(
        finding_name,
        description,
        synonyms,
        None,
        None,
        source_labels,
    )
    ontology_context_terms = _append_context_terms([], locality_labels)

    # Stage 2: Execute search with AnatomicLocationIndex
    if index is not None:
        candidates = await execute_anatomic_candidate_search(
            query_info,
            index,
            direct_terms=direct_terms,
            ontology_context_terms=ontology_context_terms,
        )
    elif use_duckdb:
        async with AnatomicLocationIndex() as owned_index:
            candidates = await execute_anatomic_candidate_search(
                query_info,
                owned_index,
                direct_terms=direct_terms,
                ontology_context_terms=ontology_context_terms,
            )
    else:
        logger.error("DuckDB is the only supported backend for anatomic location search")
        raise ValueError("DuckDB is required for anatomic location search")

    if not candidates:
        logger.warning(f"No search results found for '{finding_name}'")
        return AnatomicCandidateSearchResponse(
            candidates=[],
            query_terms=query_info.terms,
            reasoning=f"No anatomic locations found for '{finding_name}'.",
        )

    logger.info(f"Anatomic candidate gathering complete for '{finding_name}': candidates={len(candidates)}")
    return AnatomicCandidateSearchResponse(
        candidates=candidates,
        query_terms=query_info.terms,
        reasoning="Candidates gathered from source text, ontology context, hierarchy relatives, and search results.",
    )


def _append_context_terms(terms: list[str], values: list[str] | None) -> list[str]:
    _append_term_variants(terms, values)
    return terms


def _retain_exact_term_locations(
    response: LocationSearchResponse,
    search_results: list[OntologySearchResult],
    query_info: AnatomicQueryTerms,
) -> LocationSearchResponse:
    selected_ids = {response.primary_location.concept_id}
    selected_ids.update(location.concept_id for location in response.alternate_locations)
    exact_terms = {normalize_concept(term) for term in query_info.terms}
    retained_exact = [
        location
        for location in search_results
        if location.concept_text in exact_terms and location.concept_id not in selected_ids
    ]
    if not retained_exact:
        return response
    return response.model_copy(update={"alternate_locations": [*retained_exact, *response.alternate_locations][:8]})
