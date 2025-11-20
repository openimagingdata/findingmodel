"""
Finding Enrichment System

Provides structured models and types for enriching finding models with comprehensive metadata
including ontology codes, body regions, etiologies, imaging modalities, subspecialties, and anatomic locations.

This module defines the core data structures used throughout the finding enrichment workflow.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from typing_extensions import Literal

from findingmodel import logger
from findingmodel.config import ModelProvider, ModelTier, settings
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import DuckDBIndex
from findingmodel.index_code import IndexCode
from findingmodel.tools.common import get_model
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
    "traumatic-acute",  # acute injury
    "post-traumatic",  # sequelae of prior injury
    "iatrogenic:post-operative",
    "iatrogenic:post-radiation",
    "iatrogenic:device",
    "iatrogenic:medication-related",
    "vascular:ischemic",
    "vascular:hemorrhagic",
    "vascular:thrombotic",
    "degenerative",
    "congenital",
    "metabolic",
    "toxic",
    "mechanical",  # obstruction, herniation, torsion
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

    model_provider: str = Field(
        description="AI model provider used for enrichment (e.g., 'openai', 'anthropic')",
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


async def lookup_finding_in_index(identifier: str) -> FindingModelFull | None:
    """Look up a finding model in the DuckDB index by OIFM ID or name.

    This function attempts to retrieve a finding model from the index, first trying
    to match the identifier as an OIFM ID (exact match), then falling back to name-based
    search if not found. The function properly manages DuckDB connection lifecycle
    using a context manager.

    Args:
        identifier: Either an OIFM ID (e.g., "OIFM_AI_000001") or a finding name
                   (e.g., "pneumonia"). Name matching is case-insensitive and
                   supports synonym matching.

    Returns:
        FindingModelFull object if found, None if not found.
        Returns None rather than raising an error when the finding doesn't exist,
        as missing findings are a normal case in the enrichment workflow.

    Raises:
        RuntimeError: If there are database connection issues or other system errors.
                     Does NOT raise on missing findings (returns None instead).

    Example:
        >>> # Lookup by OIFM ID
        >>> model = await lookup_finding_in_index("OIFM_AI_000001")
        >>> if model:
        ...     print(f"Found: {model.name}")
        ...
        >>> # Lookup by name
        >>> model = await lookup_finding_in_index("pneumonia")
        >>> if model is None:
        ...     print("Finding not in index")
    """
    logger.debug(f"Looking up finding in index: {identifier}")

    index = DuckDBIndex(read_only=True)

    try:
        async with index:
            # Use public API - get() handles OIFM ID resolution internally
            # Tries in order: OIFM ID match, name match, slug match, synonym match
            entry = await index.get(identifier)

            if entry is None:
                logger.debug(f"Finding not found in index: {identifier}")
                return None

            logger.debug(f"Resolved {identifier} to {entry.oifm_id}")

            # Get the full model
            model = await index.get_full(entry.oifm_id)
            logger.debug(f"Retrieved full model: {model.oifm_id} ({model.name})")
            return model

    except Exception as e:
        # Log database connection errors but re-raise for caller to handle
        logger.error(f"Error looking up finding in index: {e}")
        raise RuntimeError(f"Database error while looking up finding '{identifier}': {e}") from e


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
    including existing data from the index if available.
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

TOOLS AVAILABLE:
You have access to tools for searching ontology codes and anatomic locations.
Use these to gather additional context if needed, but rely primarily on your
medical knowledge for the classifications."""


def create_enrichment_agent(
    model_tier: ModelTier = "base",
    provider: ModelProvider | None = None,
) -> Agent[EnrichmentContext, EnrichmentClassification]:
    """Create the finding enrichment agent.

    Args:
        model_tier: Model tier to use (defaults to "base")
        provider: AI model provider to use ("openai" or "anthropic"). If None, uses configured default.

    Returns:
        Configured Pydantic AI agent for finding enrichment
    """
    agent: Agent[EnrichmentContext, EnrichmentClassification] = Agent(
        model=get_model(model_tier, provider=provider),
        output_type=EnrichmentClassification,
        deps_type=EnrichmentContext,
        system_prompt=_create_enrichment_system_prompt(),
    )

    @agent.tool
    async def search_ontology_codes(ctx: RunContext[EnrichmentContext], query: str) -> str:
        """Search for ontology codes related to a finding concept.

        Use this to find SNOMED CT and RadLex codes that may provide additional
        context about the finding.

        Args:
            ctx: Agent run context with finding information
            query: Search query (finding name or concept)

        Returns:
            JSON string with search results
        """
        try:
            snomed_codes, radlex_codes = await search_ontology_codes_for_finding(
                finding_name=query,
                description=ctx.deps.finding_description,
            )

            result = {
                "snomed_codes": [code.model_dump() for code in snomed_codes],
                "radlex_codes": [code.model_dump() for code in radlex_codes],
            }

            logger.info(f"Ontology search for '{query}': {len(snomed_codes)} SNOMED, {len(radlex_codes)} RadLex")
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error in ontology search tool: {e}")
            return json.dumps({"error": str(e)})

    @agent.tool
    async def find_anatomic_location(ctx: RunContext[EnrichmentContext], finding: str) -> str:
        """Find anatomic locations for a finding.

        Use this to identify where in the body this finding typically occurs.

        Args:
            ctx: Agent run context with finding information
            finding: Finding name to search for

        Returns:
            JSON string with primary and alternate locations
        """
        from findingmodel.tools.anatomic_location_search import find_anatomic_locations

        try:
            # Use "small" tier for sub-tool regardless of parent agent tier
            # to keep costs down for this auxiliary search
            result = await find_anatomic_locations(
                finding_name=finding,
                description=ctx.deps.finding_description,
                model_tier="small",
            )

            locations = {
                "primary_location": result.primary_location.model_dump(),
                "alternate_locations": [loc.model_dump() for loc in result.alternate_locations],
                "reasoning": result.reasoning,
            }

            logger.info(f"Anatomic location search for '{finding}': {result.primary_location.concept_text}")
            return json.dumps(locations, indent=2)

        except Exception as e:
            logger.error(f"Error in anatomic location tool: {e}")
            return json.dumps({"error": str(e)})

    return agent


# ==========================================================================================
# Main Enrichment Function
# ==========================================================================================


async def enrich_finding(identifier: str, provider: ModelProvider | None = None) -> FindingEnrichmentResult:  # noqa: C901
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
        provider: AI model provider to use ("openai" or "anthropic"). If None, uses configured default.

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
        existing_model = await lookup_finding_in_index(identifier)
        if existing_model:
            logger.info(f"Found existing model: {existing_model.oifm_id} ({existing_model.name})")
            finding_name = existing_model.name
            finding_description = existing_model.description
            oifm_id = existing_model.oifm_id
        else:
            logger.info(f"Finding not in index, treating '{identifier}' as finding name")
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

    # Step 3: Create enrichment context for agent
    logger.debug("Step 3: Creating enrichment context for agent")
    all_codes = [*snomed_codes, *radlex_codes]
    context = EnrichmentContext(
        finding_name=finding_name,
        finding_description=finding_description,
        existing_codes=all_codes,
        existing_model=existing_model,
    )

    # Step 4: Run enrichment agent for classification
    logger.debug("Step 4: Running enrichment agent")
    try:
        agent = create_enrichment_agent(model_tier="base", provider=provider)
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

    # Determine model provider and tier for metadata
    effective_provider = provider or settings.model_provider
    model_tier_str = "base"  # We use base tier for enrichment agent

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
        model_provider=effective_provider,
        model_tier=model_tier_str,
    )

    logger.info(f"Enrichment complete for '{finding_name}' ({oifm_id or 'not in index'})")

    return enrichment_result
