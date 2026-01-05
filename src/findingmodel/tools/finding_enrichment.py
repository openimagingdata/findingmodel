"""
Finding Enrichment System

Provides structured models and types for enriching finding models with comprehensive metadata
including ontology codes, body regions, etiologies, imaging modalities, subspecialties, and anatomic locations.

This module defines the core data structures used throughout the finding enrichment workflow.
"""

import asyncio
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent
from typing_extensions import Literal

from findingmodel import logger
from findingmodel.config import ModelTier, settings
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import DuckDBIndex
from findingmodel.index_code import IndexCode
from findingmodel.tools.ontology_search import OntologySearchResult

# Type aliases for constrained value sets

BodyRegion = Literal["ALL", "Head", "Neck", "Chest", "Breast", "Abdomen", "Arm", "Leg"]
"""Body region categories used in radiology imaging.

- ALL: Finding applies to entire body or multiple regions
- Head: Cranial and intracranial structures
- Neck: Cervical region
- Chest: Thorax including lungs and mediastinum
- Breast: Breast tissue
- Abdomen: Abdominal cavity and retroperitoneum
- Arm: Upper extremity
- Leg: Lower extremity
"""

Modality = Literal["XR", "CT", "MR", "US", "PET", "NM", "MG", "RF", "DSA"]
"""Imaging modality codes.

- XR: Radiography (plain X-rays)
- CT: Computed Tomography
- MR: Magnetic Resonance Imaging
- US: Ultrasound (including Doppler)
- PET: Positron Emission Tomography
- NM: Nuclear Medicine (single-photon/SPECT)
- MG: Mammography
- RF: Fluoroscopy (real-time X-ray)
- DSA: Digital Subtraction Angiography
"""

Subspecialty = Literal["AB", "BR", "CA", "CH", "ER", "GI", "GU", "HN", "IR", "MI", "MK", "NR", "OB", "OI", "PD", "VI"]
"""Radiology subspecialty codes.

- AB: Abdominal Radiology
- BR: Breast Imaging
- CA: Cardiac Imaging
- CH: Chest/Thoracic Imaging
- ER: Emergency Radiology
- GI: Gastrointestinal Radiology
- GU: Genitourinary Radiology
- HN: Head & Neck Imaging
- IR: Interventional Radiology
- MI: Molecular Imaging/Nuclear Medicine
- MK: Musculoskeletal Radiology
- NR: Neuroradiology
- OB: OB/Gyn Radiology
- OI: Oncologic Imaging
- PD: Pediatric Radiology
- VI: Vascular Imaging
"""

# Etiology taxonomy - comprehensive list of finding etiology categories
ETIOLOGIES: list[str] = [
    "inflammatory:infectious",
    "inflammatory",
    "neoplastic:benign",
    "neoplastic:malignant",
    "neoplastic:metastatic",
    "neoplastic:potential",  # indeterminate lesions, incidentalomas
    "traumatic",  # acute injury
    "post-traumatic",  # sequelae of prior injury
    "vascular:ischemic",
    "vascular:hemorrhagic",
    "vascular:thrombotic",
    "vascular:aneurysmal",
    "degenerative",
    "metabolic",
    "congenital",
    "developmental",
    "autoimmune",
    "toxic",
    "mechanical",  # obstruction, herniation, torsion
    "iatrogenic:post-operative",
    "iatrogenic:post-radiation",
    "iatrogenic:device-related",
    "iatrogenic:medication-related",
    "idiopathic",
    "normal-variant",
]
"""Comprehensive taxonomy of finding etiologies.

Includes hierarchical categories with colon-separated subtypes (e.g., inflammatory:infectious).
Multiple etiologies may apply to a single finding.
"""


class FindingEnrichmentResult(BaseModel):
    """Comprehensive enrichment metadata for an imaging finding.

    Contains structured metadata including ontology codes, anatomic classifications,
    etiologies, relevant modalities, and subspecialties. This result is designed for
    human review and validation before integration into finding models.
    """

    finding_name: str = Field(
        description="Name of the finding being enriched",
        min_length=1,
    )

    oifm_id: str | None = Field(
        default=None,
        description="Open Imaging Finding Model ID if the finding exists in the index (e.g., OIFM_AI_000001)",
    )

    snomed_codes: list[IndexCode] = Field(
        default_factory=list,
        description="SNOMED CT codes for this finding (disorder/finding codes only, excluding anatomic locations)",
    )

    radlex_codes: list[IndexCode] = Field(
        default_factory=list,
        description="RadLex codes for this finding (finding codes only, excluding anatomic locations)",
    )

    body_regions: Annotated[
        list[BodyRegion],
        Field(
            default_factory=list,
            description="Primary body regions where this finding occurs (from predefined BodyRegion list)",
        ),
    ]

    etiologies: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Etiology categories for this finding (from ETIOLOGIES taxonomy). "
            "Multiple categories may apply (e.g., a finding may be both infectious and vascular:thrombotic)",
        ),
    ]

    modalities: Annotated[
        list[Modality],
        Field(
            default_factory=list,
            description="Imaging modalities where this finding is typically visualized",
        ),
    ]

    subspecialties: Annotated[
        list[Subspecialty],
        Field(
            default_factory=list,
            description="Radiology subspecialties most relevant to this finding",
        ),
    ]

    anatomic_locations: list[OntologySearchResult] = Field(
        default_factory=list,
        description="Anatomic locations where this finding occurs (from ontology search with concept IDs and scores)",
    )

    enrichment_timestamp: datetime = Field(
        description="Timestamp when this enrichment was performed",
    )

    model_used: str = Field(
        description="AI model string used for enrichment (e.g., 'openai:gpt-5-mini', 'anthropic:claude-sonnet-4-5')",
    )

    model_tier: str = Field(
        description="AI model tier used for enrichment (e.g., 'small', 'base', 'full')",
    )

    @field_validator("etiologies")
    @classmethod
    def validate_etiologies(cls, v: list[str]) -> list[str]:
        """Validate that all etiologies are in the allowed ETIOLOGIES list."""
        invalid = [etiology for etiology in v if etiology not in ETIOLOGIES]
        if invalid:
            raise ValueError(f"Invalid etiology categories: {invalid}. Must be from ETIOLOGIES list.")
        return v


async def search_ontology_codes_for_finding(
    finding_name: str, description: str | None = None
) -> tuple[list[IndexCode], list[IndexCode]]:
    """Search for ontology codes (SNOMED CT and RadLex) that represent a finding.

    Wraps match_ontology_concepts() with exclude_anatomical=True to filter out
    anatomic location codes and returns only disorder/finding codes.

    Args:
        finding_name: Name of the imaging finding (e.g., "pneumonia", "liver lesion")
        description: Optional detailed description for context

    Returns:
        Tuple of (snomed_codes, radlex_codes) as IndexCode objects.
        Both lists may be empty if no matches found.

    Raises:
        Exception: If the ontology search fails

    Example:
        >>> snomed, radlex = await search_ontology_codes_for_finding("pneumonia")
        >>> print(f"Found {len(snomed)} SNOMED codes and {len(radlex)} RadLex codes")
        Found 3 SNOMED codes and 2 RadLex codes
    """
    from findingmodel.tools.ontology_concept_match import match_ontology_concepts

    logger.info(f"Searching ontology codes for finding: {finding_name}")

    try:
        categorized_results = await match_ontology_concepts(
            finding_name=finding_name,
            finding_description=description,
            exclude_anatomical=True,
        )

        # Collect high-confidence results (exact + should_include)
        all_results = [
            *categorized_results.exact_matches,
            *categorized_results.should_include,
        ]

        # Separate by ontology system
        snomed_codes: list[IndexCode] = []
        radlex_codes: list[IndexCode] = []

        for result in all_results:
            index_code = result.as_index_code()
            if index_code.system == "SNOMEDCT":
                snomed_codes.append(index_code)
            elif index_code.system == "RADLEX":
                radlex_codes.append(index_code)

        logger.info(f"Found {len(snomed_codes)} SNOMED CT and {len(radlex_codes)} RadLex codes")

        return (snomed_codes, radlex_codes)

    except Exception as e:
        logger.error(f"Error searching ontology codes: {e}")
        raise


# ==========================================================================================
# Enrichment Context & Agent Implementation
# ==========================================================================================


class EnrichmentContext(BaseModel):
    """Dependency context for the enrichment agent.

    Contains all relevant information about the finding being enriched,
    including pre-fetched search results and existing data from the index.
    """

    finding_name: str = Field(
        description="Name of the finding being enriched",
    )

    finding_description: str | None = Field(
        default=None,
        description="Detailed description of the finding, if available",
    )

    existing_codes: list[IndexCode] = Field(
        default_factory=list,
        description="Existing ontology codes from index or search",
    )

    existing_model: FindingModelFull | None = Field(
        default=None,
        description="Existing FindingModelFull from index, if found",
    )

    # Pre-fetched search results (provided to agent for context)
    snomed_codes: list[IndexCode] = Field(
        default_factory=list,
        description="SNOMED CT codes found for this finding",
    )

    radlex_codes: list[IndexCode] = Field(
        default_factory=list,
        description="RadLex codes found for this finding",
    )

    anatomic_locations: list[OntologySearchResult] = Field(
        default_factory=list,
        description="Anatomic locations identified for this finding",
    )


class EnrichmentClassification(BaseModel):
    """Structured output from the enrichment agent.

    Contains the agent's classifications for body regions, etiologies,
    modalities, and subspecialties.

    Note: This is the agent's output type (Phase 4). The complete FindingEnrichmentResult
    is assembled in Phase 5 by combining this classification with ontology codes and
    anatomic locations from other tools. This separation allows the agent to focus on
    classification while other components handle structured data retrieval.
    """

    body_regions: Annotated[
        list[BodyRegion],
        Field(
            default_factory=list,
            description="Primary body regions where this finding occurs",
        ),
    ]

    etiologies: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Etiology categories for this finding (from ETIOLOGIES taxonomy)",
        ),
    ]

    modalities: Annotated[
        list[Modality],
        Field(
            default_factory=list,
            description="Imaging modalities where this finding is typically visualized",
        ),
    ]

    subspecialties: Annotated[
        list[Subspecialty],
        Field(
            default_factory=list,
            description="Radiology subspecialties most relevant to this finding",
        ),
    ]

    reasoning: str = Field(
        description="Brief explanation of the classification decisions",
    )

    @field_validator("etiologies")
    @classmethod
    def validate_etiologies(cls, v: list[str]) -> list[str]:
        """Validate that all etiologies are in the allowed ETIOLOGIES list."""
        invalid = [etiology for etiology in v if etiology not in ETIOLOGIES]
        if invalid:
            raise ValueError(f"Invalid etiology categories: {invalid}. Must be from ETIOLOGIES list.")
        return v


class CategorizedOntologyCodes(BaseModel):
    """Ontology codes categorized by relevance to a finding.

    Used by the unified enrichment classifier to structure ontology code selection
    from raw search results.
    """

    exact_matches: Annotated[
        list[IndexCode],
        Field(
            default_factory=list,
            max_length=5,
            description="Concepts whose meaning is identical to the finding name (max 5)",
        ),
    ]

    should_include: Annotated[
        list[IndexCode],
        Field(
            default_factory=list,
            max_length=10,
            description="Closely related concepts - subtypes, variants, strong associations (max 10)",
        ),
    ]

    marginal: Annotated[
        list[IndexCode],
        Field(
            default_factory=list,
            max_length=10,
            description="Peripherally related - broader categories, distinct but related (max 10)",
        ),
    ]


class AnatomicLocationSelection(BaseModel):
    """Single anatomic location for a finding - the highest-level container.

    Used by the unified enrichment classifier to select the most appropriate
    anatomic location from raw search results. Returns exactly ONE location
    that encompasses all places where this finding can occur.
    """

    location: OntologySearchResult = Field(
        description="The single highest-level anatomic container that encompasses ALL locations where this finding can occur",
    )


class UnifiedEnrichmentOutput(BaseModel):
    """Complete structured output from the unified enrichment classifier.

    This model represents the output from the unified enrichment classifier that
    combines ontology code selection, anatomic location selection, and finding
    classification into a single AI agent call. It replaces the multi-stage
    pipeline with a single prompt that processes raw search results.

    The unified classifier receives:
    - Finding name and description
    - Raw SNOMED/RadLex search results
    - Raw anatomic location search results
    - Any existing model data

    And produces comprehensive enrichment in one call.
    """

    ontology_codes: CategorizedOntologyCodes = Field(
        description="Ontology codes categorized by relevance (exact/should_include/marginal)",
    )

    anatomic_location: AnatomicLocationSelection = Field(
        description="Single anatomic location for this finding (highest-level container)",
    )

    body_regions: Annotated[
        list[BodyRegion],
        Field(
            default_factory=list,
            description="Primary body regions where this finding occurs",
        ),
    ]

    etiologies: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Etiology categories for this finding (from ETIOLOGIES taxonomy)",
        ),
    ]

    modalities: Annotated[
        list[Modality],
        Field(
            default_factory=list,
            description="Imaging modalities where this finding is typically visualized",
        ),
    ]

    subspecialties: Annotated[
        list[Subspecialty],
        Field(
            default_factory=list,
            description="Radiology subspecialties most relevant to this finding",
        ),
    ]

    reasoning: str = Field(
        description="Clear explanation of all classification, selection, and categorization decisions",
    )

    @field_validator("etiologies")
    @classmethod
    def validate_etiologies(cls, v: list[str]) -> list[str]:
        """Validate that all etiologies are in the allowed ETIOLOGIES list."""
        invalid = [etiology for etiology in v if etiology not in ETIOLOGIES]
        if invalid:
            raise ValueError(f"Invalid etiology categories: {invalid}. Must be from ETIOLOGIES list.")
        return v


def _create_enrichment_system_prompt() -> str:
    """Create comprehensive system prompt for the enrichment agent.

    Returns:
        Complete system prompt with role, tasks, instructions, and taxonomy definitions.
    """
    return f"""You are a medical imaging finding enrichment specialist with expertise in radiology.

Your task is to analyze an imaging finding and classify it across multiple dimensions to aid in
organizing and categorizing findings in a medical imaging database.

ROLE & RESPONSIBILITIES:
- Classify the finding's primary body regions
- Identify all applicable etiologies (causes/origins)
- Determine relevant imaging modalities
- Identify appropriate subspecialties

INSTRUCTIONS:
1. Be precise and specific in your classifications
2. Multiple values are allowed and encouraged when appropriate
3. If uncertain about a classification, leave it blank rather than guessing
4. Consider the clinical context and common usage patterns in radiology
5. Provide brief reasoning for your decisions

CLASSIFICATION TAXONOMIES:

Body Regions:
- ALL: Finding applies to entire body or multiple regions
- Head: Cranial and intracranial structures
- Neck: Cervical region
- Chest: Thorax including lungs and mediastinum
- Breast: Breast tissue
- Abdomen: Abdominal cavity and retroperitoneum
- Arm: Upper extremity
- Leg: Lower extremity

Etiologies (multiple allowed):
{chr(10).join(f"- {etiology}" for etiology in ETIOLOGIES)}

Modalities:
- XR: Radiography (plain X-rays)
- CT: Computed Tomography
- MR: Magnetic Resonance Imaging
- US: Ultrasound (including Doppler)
- PET: Positron Emission Tomography
- NM: Nuclear Medicine (single-photon/SPECT)
- MG: Mammography
- RF: Fluoroscopy (real-time X-ray)
- DSA: Digital Subtraction Angiography

Subspecialties:
- AB: Abdominal Radiology
- BR: Breast Imaging
- CA: Cardiac Imaging
- CH: Chest/Thoracic Imaging
- ER: Emergency Radiology
- GI: Gastrointestinal Radiology
- GU: Genitourinary Radiology
- HN: Head & Neck Imaging
- IR: Interventional Radiology
- MI: Molecular Imaging/Nuclear Medicine
- MK: Musculoskeletal Radiology
- NR: Neuroradiology
- OB: OB/Gyn Radiology
- OI: Oncologic Imaging
- PD: Pediatric Radiology
- VI: Vascular Imaging

CONTEXT PROVIDED:
You will receive pre-fetched search results including:
- SNOMED CT and RadLex codes relevant to the finding
- Anatomic locations where the finding typically occurs
- Any existing data from the finding model database

Use this context along with your medical knowledge for the classifications.
Do NOT call any external tools - all necessary information is provided."""


def create_enrichment_agent(
    model_tier: ModelTier = "base",
    model: str | None = None,
) -> Agent[EnrichmentContext, EnrichmentClassification]:
    """Create the finding enrichment agent.

    The agent performs classification only - ontology and anatomic location searches
    are run in parallel BEFORE the agent is called, and results are provided in context.
    This avoids duplicate API calls and significantly improves performance.

    Args:
        model_tier: Model tier to use (defaults to "base")
        model: Optional model string override (e.g., 'openai:gpt-5', 'anthropic:claude-sonnet-4-5').
               If None, uses the configured default for the specified tier.

    Returns:
        Configured Pydantic AI agent for finding enrichment (no tools - classification only)
    """
    agent: Agent[EnrichmentContext, EnrichmentClassification] = Agent(
        model=model if model else settings.get_agent_model("enrich_classify", default_tier=model_tier),
        output_type=EnrichmentClassification,
        deps_type=EnrichmentContext,
        system_prompt=_create_enrichment_system_prompt(),
    )

    # NOTE: No tools registered - the agent only classifies based on context.
    # Ontology and anatomic location searches are performed in parallel
    # BEFORE the agent runs, and results are included in the context.
    # This design avoids duplicate API calls and improves performance significantly.

    return agent


# ==========================================================================================
# Main Enrichment Function
# ==========================================================================================


async def enrich_finding_unified(  # noqa: C901
    finding_name: str,
    description: str | None = None,
    model: str | None = None,
) -> UnifiedEnrichmentOutput:
    """Enrich a finding using the unified classifier approach.

    This is the optimized 3-LLM-call pipeline that:
    1. Runs parallel query generation for ontology and anatomic searches (small tier)
    2. Executes DuckDB searches for both
    3. Passes raw results to unified classifier (base tier) for comprehensive categorization

    This approach is faster and more coherent than the 5-call pipeline, reducing
    from ~25-30s to ~15-20s while maintaining or improving quality.

    Args:
        finding_name: Name of the imaging finding (e.g., "pneumonia", "liver lesion")
        description: Optional detailed description for context
        model: Optional model string override (e.g., 'openai:gpt-5', 'anthropic:claude-sonnet-4-5').
               If None, uses the configured default.

    Returns:
        UnifiedEnrichmentOutput with categorized codes, selected locations, and classifications

    Raises:
        RuntimeError: If critical workflow steps fail
        ConfigurationError: If AI provider is misconfigured

    Example:
        >>> result = await enrich_finding_unified("pneumonia")
        >>> print(f"Exact matches: {len(result.ontology_codes.exact_matches)}")
        >>> print(f"Anatomic location: {result.anatomic_location.location.concept_text}")
        >>> print(f"Body regions: {result.body_regions}")
    """
    from time import perf_counter

    from findingmodel.tools.anatomic_location_search import generate_anatomic_query_terms
    from findingmodel.tools.duckdb_search import DuckDBOntologySearchClient
    from findingmodel.tools.ontology_concept_match import generate_finding_query_terms
    from findingmodel.tools.prompt_template import load_prompt_template, render_agent_prompt

    timings: dict[str, float] = {}
    total_start = perf_counter()

    logger.info(f"Starting unified enrichment for: {finding_name}")

    # Step 1: Parallel query generation (LLM small tier)
    step1_start = perf_counter()
    logger.debug("Step 1: Parallel query generation")

    async def generate_ontology_queries() -> list[str]:
        """Generate ontology query terms with error handling."""
        try:
            return await generate_finding_query_terms(finding_name, description)
        except Exception as e:
            logger.warning(f"Ontology query generation failed, using fallback: {e}")
            return [finding_name]

    async def generate_anatomic_queries() -> tuple[
        list[str],
        Literal["Abdomen", "Neck", "Lower Extremity", "Breast", "Body", "Thorax", "Upper Extremity", "Head", "Pelvis"]
        | None,
    ]:
        """Generate anatomic query terms with error handling."""
        try:
            query_info = await generate_anatomic_query_terms(
                finding_name=finding_name,
                finding_description=description,
                model_tier="small",
            )
            return (query_info.terms, query_info.region)
        except Exception as e:
            logger.warning(f"Anatomic query generation failed, using fallback: {e}")
            return ([finding_name], None)

    # Execute query generation in parallel
    ontology_queries, (anatomic_queries, anatomic_region) = await asyncio.gather(
        generate_ontology_queries(),
        generate_anatomic_queries(),
    )
    timings["step1_query_gen"] = perf_counter() - step1_start

    logger.info(
        f"Query generation complete ({timings['step1_query_gen']:.1f}s): {len(ontology_queries)} ontology terms, "
        f"{len(anatomic_queries)} anatomic terms (region: {anatomic_region or 'none'})"
    )

    # Step 2: Execute searches (ontology via BioOntology API, anatomic via DuckDB)
    step2_start = perf_counter()
    logger.debug("Step 2: Executing searches")

    ontology_results: list[OntologySearchResult] = []
    anatomic_results: list[OntologySearchResult] = []

    # Search ontology codes (SNOMED/RadLex) via BioOntology API
    ontology_search_start = perf_counter()
    try:
        from findingmodel.tools.ontology_concept_match import execute_ontology_search

        ontology_results = await execute_ontology_search(
            query_terms=ontology_queries,
            exclude_anatomical=True,
            base_limit=30,
            max_results=12,
        )
        timings["step2a_ontology_search"] = perf_counter() - ontology_search_start
        logger.info(f"Found {len(ontology_results)} ontology results ({timings['step2a_ontology_search']:.1f}s)")
    except Exception as e:
        timings["step2a_ontology_search"] = perf_counter() - ontology_search_start
        logger.error(f"Ontology search failed: {e}")
        # Continue with empty results

    # Search anatomic locations via DuckDB
    anatomic_search_start = perf_counter()
    try:
        from findingmodel.tools.anatomic_location_search import AnatomicQueryTerms, execute_anatomic_search

        async with DuckDBOntologySearchClient() as client:
            query_info = AnatomicQueryTerms(region=anatomic_region, terms=anatomic_queries)
            anatomic_results = await execute_anatomic_search(query_info, client, limit=30)
            timings["step2b_anatomic_search"] = perf_counter() - anatomic_search_start
            logger.info(f"Found {len(anatomic_results)} anatomic results ({timings['step2b_anatomic_search']:.1f}s)")
    except Exception as e:
        timings["step2b_anatomic_search"] = perf_counter() - anatomic_search_start
        logger.error(f"Anatomic search failed: {e}")
        # Continue with empty results

    timings["step2_total_search"] = perf_counter() - step2_start

    # Step 3: Add RID39569 "whole body" to anatomic results if not present
    whole_body_id = "RID39569"
    if not any(result.concept_id == whole_body_id for result in anatomic_results):
        anatomic_results.append(
            OntologySearchResult(
                concept_id=whole_body_id,
                concept_text="whole body",
                score=0.0,  # Low score since it's a fallback
                table_name="anatomic_locations",
            )
        )
        logger.debug("Added RID39569 'whole body' to anatomic results")

    # Step 4: Format raw results as strings for the prompt
    logger.debug("Step 4: Formatting results for unified classifier")

    # Format ontology results as "SYSTEM CODE: display" strings
    # Note: We avoid calling as_index_code() here because IndexCode validation
    # may fail on short display values (e.g., "Or" < 3 chars). We just need the
    # system name for prompt formatting.
    from findingmodel.tools.ontology_search import TABLE_TO_INDEX_CODE_SYSTEM

    ontology_results_str = "\n".join([
        f"{TABLE_TO_INDEX_CODE_SYSTEM.get(result.table_name, result.table_name)} {result.concept_id}: {result.concept_text}"
        for result in ontology_results
    ])
    if not ontology_results_str:
        ontology_results_str = "(no ontology results found)"

    # Format anatomic results as "ID: description" strings
    anatomic_results_str = "\n".join([f"{result.concept_id}: {result.concept_text}" for result in anatomic_results])

    # Step 5: Load and render the unified enrichment classifier prompt
    step5_start = perf_counter()
    logger.debug("Step 5: Loading and rendering prompt template")
    template = load_prompt_template("unified_enrichment_classifier.xml.jinja")

    # Lookup existing model for context (optional, non-blocking)
    existing_model_str = ""
    index_lookup_start = perf_counter()
    try:
        index = DuckDBIndex(read_only=True)
        async with index:
            entry = await index.get(finding_name)
            if entry is not None:
                existing_model = await index.get_full(entry.oifm_id)
                existing_model_str = f"OIFM ID: {existing_model.oifm_id}\nName: {existing_model.name}"
                if existing_model.description:
                    existing_model_str += f"\nDescription: {existing_model.description}"
                logger.debug(f"Found existing model: {existing_model.oifm_id}")
    except Exception as e:
        logger.warning(f"Failed to lookup existing model (non-fatal): {e}")
    timings["step5_index_lookup"] = perf_counter() - index_lookup_start

    system_prompt, user_prompt = render_agent_prompt(
        template,
        finding_name=finding_name,
        description=description or "",
        ontology_results=ontology_results_str,
        anatomic_results=anatomic_results_str,
        existing_model=existing_model_str if existing_model_str else None,
    )
    timings["step5_template"] = perf_counter() - step5_start

    # Step 6: Create and run unified classifier agent (base tier)
    step6_start = perf_counter()
    logger.debug("Step 6: Running unified classifier agent")

    agent: Agent[None, UnifiedEnrichmentOutput] = Agent(
        model=model if model else settings.get_agent_model("enrich_unified", default_tier="base"),
        output_type=UnifiedEnrichmentOutput,
        system_prompt=system_prompt,
        retries=2,  # Allow 2 retries for validation errors
    )

    try:
        result = await agent.run(user_prompt)
        output = result.output
        timings["step6_classifier"] = perf_counter() - step6_start
        timings["total"] = perf_counter() - total_start

        logger.info(
            f"Unified classifier complete ({timings['step6_classifier']:.1f}s): "
            f"{len(output.ontology_codes.exact_matches)} exact codes, "
            f"{len(output.body_regions)} body regions, "
            f"{len(output.etiologies)} etiologies, "
            f"location: {output.anatomic_location.location.concept_text}"
        )

        # Log timing breakdown
        logger.info(
            f"Timing breakdown - Total: {timings['total']:.1f}s | "
            f"QueryGen: {timings.get('step1_query_gen', 0):.1f}s | "
            f"OntologySearch: {timings.get('step2a_ontology_search', 0):.1f}s | "
            f"AnatomicSearch: {timings.get('step2b_anatomic_search', 0):.1f}s | "
            f"IndexLookup: {timings.get('step5_index_lookup', 0):.1f}s | "
            f"Classifier: {timings.get('step6_classifier', 0):.1f}s"
        )

        return output

    except Exception as e:
        timings["step6_classifier"] = perf_counter() - step6_start
        timings["total"] = perf_counter() - total_start
        logger.error(f"Unified classifier agent failed after {timings['step6_classifier']:.1f}s: {e}")
        logger.info(
            f"Timing breakdown (failed) - Total: {timings['total']:.1f}s | "
            f"QueryGen: {timings.get('step1_query_gen', 0):.1f}s | "
            f"OntologySearch: {timings.get('step2a_ontology_search', 0):.1f}s | "
            f"AnatomicSearch: {timings.get('step2b_anatomic_search', 0):.1f}s | "
            f"IndexLookup: {timings.get('step5_index_lookup', 0):.1f}s | "
            f"Classifier: {timings.get('step6_classifier', 0):.1f}s"
        )
        raise RuntimeError(f"Enrichment classification failed: {e}") from e


async def enrich_finding(identifier: str, model: str | None = None) -> FindingEnrichmentResult:  # noqa: C901
    """Enrich a finding with comprehensive metadata.

    This is the main entry point for the enrichment workflow. It orchestrates:
    1. Index lookup to get existing finding data
    2. Parallel ontology code search and anatomic location search
    3. Agent-based classification of body regions, etiologies, modalities, and subspecialties
    4. Assembly of complete FindingEnrichmentResult with metadata

    The function uses parallel execution where possible and implements graceful degradation
    if individual components fail.

    Args:
        identifier: Either an OIFM ID (e.g., "OIFM_AI_000001") or finding name (e.g., "pneumonia")
        model: Optional model string override (e.g., 'openai:gpt-5', 'anthropic:claude-sonnet-4-5').
               If None, uses the configured default.

    Returns:
        FindingEnrichmentResult with all enrichment data and metadata

    Raises:
        RuntimeError: If critical workflow steps fail (e.g., database connection issues)
        ConfigurationError: If AI provider is misconfigured

    Example:
        >>> result = await enrich_finding("pneumonia")
        >>> print(f"Found {len(result.snomed_codes)} SNOMED codes")
        >>> print(f"Body regions: {result.body_regions}")
        >>> print(f"Anatomic locations: {[loc.concept_text for loc in result.anatomic_locations]}")
    """
    from findingmodel.tools.anatomic_location_search import find_anatomic_locations

    logger.info(f"Starting enrichment workflow for: {identifier}")

    # Step 1: Lookup finding in index
    logger.debug("Step 1: Looking up finding in index")
    try:
        index = DuckDBIndex(read_only=True)
        async with index:
            # Use public API - get() handles OIFM ID resolution internally
            # Tries in order: OIFM ID match, name match, slug match, synonym match
            entry = await index.get(identifier)

            if entry is not None:
                logger.debug(f"Resolved {identifier} to {entry.oifm_id}")
                # Get the full model
                existing_model = await index.get_full(entry.oifm_id)
                logger.info(f"Found existing model: {existing_model.oifm_id} ({existing_model.name})")
                finding_name = existing_model.name
                finding_description = existing_model.description
                oifm_id = existing_model.oifm_id
            else:
                logger.debug(f"Finding not found in index: {identifier}")
                logger.info(f"Finding not in index, treating '{identifier}' as finding name")
                existing_model = None
                finding_name = identifier
                finding_description = None
                oifm_id = None
    except Exception as e:
        logger.error(f"Error during index lookup: {e}")
        raise RuntimeError(f"Failed to lookup finding in index: {e}") from e

    # Step 2: Parallel tool execution for ontology codes and anatomic locations
    logger.debug("Step 2: Running parallel searches for ontology codes and anatomic locations")

    # Create tasks for parallel execution
    async def search_ontology_with_fallback() -> tuple[list[IndexCode], list[IndexCode]]:
        """Search ontology codes with error handling."""
        try:
            return await search_ontology_codes_for_finding(finding_name, finding_description)
        except Exception as e:
            logger.warning(f"Ontology code search failed (continuing with empty results): {e}")
            return ([], [])

    async def search_anatomic_with_fallback() -> list[OntologySearchResult]:
        """Search anatomic locations with error handling."""
        try:
            result = await find_anatomic_locations(
                finding_name=finding_name,
                description=finding_description,
                model_tier="small",
            )
            # Collect all locations (primary + alternates)
            locations = [result.primary_location]
            locations.extend(result.alternate_locations)
            return locations
        except Exception as e:
            logger.warning(f"Anatomic location search failed (continuing with empty results): {e}")
            return []

    # Execute in parallel with error handling
    # Defense-in-depth: wrap fallback calls with return_exceptions=True
    # to catch any exceptions that slip through the inner try/except blocks
    try:
        ontology_result, anatomic_result = await asyncio.gather(
            search_ontology_with_fallback(),
            search_anatomic_with_fallback(),
            return_exceptions=True,
        )

        # Type narrow and handle any exceptions that slipped through the fallback handlers
        snomed_codes: list[IndexCode]
        radlex_codes: list[IndexCode]
        anatomic_locations: list[OntologySearchResult]

        if isinstance(ontology_result, Exception):
            logger.error(f"Ontology search raised unhandled exception: {ontology_result}")
            snomed_codes, radlex_codes = [], []
        elif isinstance(ontology_result, tuple):
            snomed_codes, radlex_codes = ontology_result
        else:
            logger.error(f"Ontology search returned unexpected type: {type(ontology_result)}")
            snomed_codes, radlex_codes = [], []

        if isinstance(anatomic_result, Exception):
            logger.error(f"Anatomic search raised unhandled exception: {anatomic_result}")
            anatomic_locations = []
        elif isinstance(anatomic_result, list):
            anatomic_locations = anatomic_result
        else:
            logger.error(f"Anatomic search returned unexpected type: {type(anatomic_result)}")
            anatomic_locations = []

    except Exception as e:
        # Should not happen with return_exceptions=True, but be defensive
        logger.error(f"Unexpected error during parallel execution: {e}")
        snomed_codes, radlex_codes = [], []
        anatomic_locations = []

    logger.info(
        f"Parallel search complete: {len(snomed_codes)} SNOMED, "
        f"{len(radlex_codes)} RadLex, {len(anatomic_locations)} locations"
    )

    # Step 3: Create enrichment context for agent (includes pre-fetched results)
    logger.debug("Step 3: Creating enrichment context for agent")
    all_codes = [*snomed_codes, *radlex_codes]
    context = EnrichmentContext(
        finding_name=finding_name,
        finding_description=finding_description,
        existing_codes=all_codes,
        existing_model=existing_model,
        # Include pre-fetched results so agent doesn't need to call tools
        snomed_codes=snomed_codes,
        radlex_codes=radlex_codes,
        anatomic_locations=anatomic_locations,
    )

    # Step 4: Run enrichment agent for classification
    logger.debug("Step 4: Running enrichment agent")
    try:
        agent = create_enrichment_agent(model_tier="base", model=model)
        prompt = f"Classify the imaging finding: {finding_name}"
        if finding_description:
            prompt += f"\nDescription: {finding_description}"

        result = await agent.run(prompt, deps=context)
        classification = result.output

        logger.info(
            f"Agent classification complete: "
            f"{len(classification.body_regions)} regions, "
            f"{len(classification.etiologies)} etiologies, "
            f"{len(classification.modalities)} modalities, "
            f"{len(classification.subspecialties)} subspecialties"
        )
        logger.debug(f"Agent reasoning: {classification.reasoning}")

    except Exception as e:
        logger.error(f"Enrichment agent failed, using empty classifications: {e}")
        # Graceful degradation - return empty classification
        classification = EnrichmentClassification(
            body_regions=[],
            etiologies=[],
            modalities=[],
            subspecialties=[],
            reasoning=f"Agent classification failed: {e}",
        )

    # Step 5: Assemble complete FindingEnrichmentResult
    logger.debug("Step 5: Assembling final enrichment result")

    # Determine model used for metadata
    model_tier_str = "base"  # We use base tier for enrichment agent
    if model:
        model_used = model
    elif "enrich_classify" in settings.agent_model_overrides:
        model_used = settings.agent_model_overrides["enrich_classify"]
    else:
        model_used = settings.default_model

    enrichment_result = FindingEnrichmentResult(
        finding_name=finding_name,
        oifm_id=oifm_id,
        snomed_codes=snomed_codes,
        radlex_codes=radlex_codes,
        body_regions=classification.body_regions,
        etiologies=classification.etiologies,
        modalities=classification.modalities,
        subspecialties=classification.subspecialties,
        anatomic_locations=anatomic_locations,
        enrichment_timestamp=datetime.now(timezone.utc),
        model_used=model_used,
        model_tier=model_tier_str,
    )

    logger.info(f"Enrichment complete for '{finding_name}' ({oifm_id or 'not in index'})")

    return enrichment_result
