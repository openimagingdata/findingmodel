"""
Finding Enrichment System - Agentic Tool-Calling Version

Alternative implementation that uses a single agent with tools for all lookups.
The agent decides when and how to call tools, rather than pre-fetching results.

This approach is simpler and lets the model orchestrate the workflow.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from findingmodel import logger
from findingmodel.config import ModelTier, settings
from findingmodel.finding_model import FindingModelFull
from findingmodel.index import DuckDBIndex
from findingmodel.index_code import IndexCode

# Import shared types from the main module
from findingmodel.tools.finding_enrichment import (
    ETIOLOGIES,
    BodyRegion,
    FindingEnrichmentResult,
    Modality,
    Subspecialty,
)
from findingmodel.tools.ontology_search import OntologySearchResult


class AgenticEnrichmentContext(BaseModel):
    """Context for the agentic enrichment agent."""

    finding_name: str
    finding_description: str | None = None
    oifm_id: str | None = None
    existing_model: FindingModelFull | None = None


class AgenticEnrichmentOutput(BaseModel):
    """Structured output from the agentic enrichment agent.

    The agent populates all fields after using tools to gather information.
    """

    body_regions: list[BodyRegion] = Field(
        default_factory=list,
        description="Primary body regions where this finding occurs",
    )

    etiologies: list[str] = Field(
        default_factory=list,
        description="Etiology categories from ETIOLOGIES taxonomy",
    )

    modalities: list[Modality] = Field(
        default_factory=list,
        description="Imaging modalities where this finding is typically visualized",
    )

    subspecialties: list[Subspecialty] = Field(
        default_factory=list,
        description="Radiology subspecialties most relevant to this finding",
    )

    snomed_codes: list[str] = Field(
        default_factory=list,
        description="SNOMED CT concept IDs for this finding",
    )

    radlex_codes: list[str] = Field(
        default_factory=list,
        description="RadLex concept IDs for this finding",
    )

    anatomic_location_ids: list[str] = Field(
        default_factory=list,
        description="Anatomic location concept IDs",
    )

    reasoning: str = Field(
        description="Brief explanation of classifications and tool usage",
    )


def _create_agentic_system_prompt() -> str:
    """Create system prompt for the agentic enrichment agent."""
    return f"""You are a medical imaging finding enrichment specialist.

Your task is to enrich an imaging finding with comprehensive metadata by:
1. Searching for relevant ontology codes (SNOMED CT and RadLex)
2. Finding appropriate anatomic locations
3. Classifying body regions, etiologies, modalities, and subspecialties

WORKFLOW:
1. FIRST, use the search_ontology_codes tool to find SNOMED and RadLex codes
2. THEN, use the find_anatomic_locations tool to identify where the finding occurs
3. FINALLY, based on the tool results and your medical knowledge, classify the finding

TOOLS AVAILABLE:
- search_ontology_codes: Find SNOMED CT and RadLex codes for the finding
- find_anatomic_locations: Find anatomic locations where the finding occurs

CLASSIFICATION TAXONOMIES:

Body Regions: ALL, Head, Neck, Chest, Breast, Abdomen, Arm, Leg

Etiologies (use exact strings):
{chr(10).join(f"- {e}" for e in ETIOLOGIES)}

Modalities: XR, CT, MR, US, PET, NM, MG, RF, DSA

Subspecialties: AB, BR, CA, CH, ER, GI, GU, HN, IR, MI, MK, NR, OB, OI, PD, VI

INSTRUCTIONS:
- Call BOTH tools before making classifications
- Use tool results to inform your classifications
- Multiple values are allowed for all categories
- If uncertain, leave categories empty rather than guessing
- Include concept IDs from tool results in your output"""


def create_agentic_enrichment_agent(
    model_tier: ModelTier = "base",
    model: str | None = None,
) -> Agent[AgenticEnrichmentContext, AgenticEnrichmentOutput]:
    """Create the agentic enrichment agent with tools.

    This agent uses tools to gather information, then classifies the finding.
    """
    agent: Agent[AgenticEnrichmentContext, AgenticEnrichmentOutput] = Agent(
        model=model if model else settings.get_agent_model("enrich_research", default_tier=model_tier),
        output_type=AgenticEnrichmentOutput,
        deps_type=AgenticEnrichmentContext,
        system_prompt=_create_agentic_system_prompt(),
    )

    @agent.tool
    async def search_ontology_codes(
        ctx: RunContext[AgenticEnrichmentContext],
    ) -> str:
        """Search for SNOMED CT and RadLex codes for the finding.

        Returns codes that represent this finding (excludes anatomic location codes).
        """
        from findingmodel.tools.finding_enrichment import search_ontology_codes_for_finding

        finding_name = ctx.deps.finding_name
        description = ctx.deps.finding_description

        logger.info(f"[Agentic] Tool call: search_ontology_codes for '{finding_name}'")

        try:
            snomed_codes, radlex_codes = await search_ontology_codes_for_finding(finding_name, description)

            # Format results for the agent
            result_parts = []

            if snomed_codes:
                snomed_strs = [f"  - {c.code}: {c.display}" for c in snomed_codes]
                result_parts.append("SNOMED CT codes:\n" + "\n".join(snomed_strs))
            else:
                result_parts.append("SNOMED CT codes: None found")

            if radlex_codes:
                radlex_strs = [f"  - {c.code}: {c.display}" for c in radlex_codes]
                result_parts.append("RadLex codes:\n" + "\n".join(radlex_strs))
            else:
                result_parts.append("RadLex codes: None found")

            return "\n\n".join(result_parts)

        except Exception as e:
            logger.warning(f"[Agentic] Ontology search failed: {e}")
            return f"Ontology search failed: {e}"

    @agent.tool
    async def find_anatomic_locations(
        ctx: RunContext[AgenticEnrichmentContext],
    ) -> str:
        """Find anatomic locations where this finding typically occurs.

        Returns ranked list of anatomic locations with concept IDs.
        """
        from findingmodel.tools.anatomic_location_search import (
            find_anatomic_locations as _find_locations,
        )

        finding_name = ctx.deps.finding_name
        description = ctx.deps.finding_description

        logger.info(f"[Agentic] Tool call: find_anatomic_locations for '{finding_name}'")

        try:
            result = await _find_locations(
                finding_name=finding_name,
                description=description,
                model_tier="small",
            )

            # Format results for the agent
            locations = [result.primary_location, *result.alternate_locations]

            if not locations or not locations[0]:
                return "No anatomic locations found"

            loc_strs = []
            for loc in locations:
                if loc:
                    loc_strs.append(f"  - {loc.concept_id}: {loc.concept_text} (score: {loc.score:.2f})")

            return "Anatomic locations:\n" + "\n".join(loc_strs)

        except Exception as e:
            logger.warning(f"[Agentic] Anatomic location search failed: {e}")
            return f"Anatomic location search failed: {e}"

    return agent


async def enrich_finding_agentic(identifier: str, model: str | None = None) -> FindingEnrichmentResult:
    """Enrich a finding using the agentic tool-calling approach.

    This version uses a single agent that calls tools as needed,
    rather than pre-fetching results in parallel.

    Args:
        identifier: Either an OIFM ID or finding name
        model: Optional model string override (e.g., 'openai:gpt-5', 'anthropic:claude-sonnet-4-5')

    Returns:
        FindingEnrichmentResult with all enrichment data
    """
    logger.info(f"[Agentic] Starting enrichment for: {identifier}")

    # Step 1: Lookup finding in index
    finding_name = identifier
    finding_description: str | None = None
    oifm_id: str | None = None
    existing_model: FindingModelFull | None = None

    try:
        index = DuckDBIndex(read_only=True)
        async with index:
            entry = await index.get(identifier)
            if entry is not None:
                existing_model = await index.get_full(entry.oifm_id)
                finding_name = existing_model.name
                finding_description = existing_model.description
                oifm_id = existing_model.oifm_id
                logger.info(f"[Agentic] Found in index: {oifm_id} ({finding_name})")
    except Exception as e:
        logger.warning(f"[Agentic] Index lookup failed: {e}")

    # Step 2: Create context and run agent
    context = AgenticEnrichmentContext(
        finding_name=finding_name,
        finding_description=finding_description,
        oifm_id=oifm_id,
        existing_model=existing_model,
    )

    agent = create_agentic_enrichment_agent(model_tier="base", model=model)

    prompt = f"Enrich the imaging finding: {finding_name}"
    if finding_description:
        prompt += f"\nDescription: {finding_description}"

    logger.info("[Agentic] Running agent with tools...")

    try:
        result = await agent.run(prompt, deps=context)
        output = result.output
        logger.info(
            f"[Agentic] Agent complete: {len(output.snomed_codes)} SNOMED, "
            f"{len(output.radlex_codes)} RadLex, {len(output.anatomic_location_ids)} locations"
        )
    except Exception as e:
        logger.error(f"[Agentic] Agent failed: {e}")
        raise

    # Step 3: Convert output to FindingEnrichmentResult
    # We need to convert code IDs back to IndexCode objects
    # For now, create minimal IndexCode objects from the IDs
    snomed_codes = [IndexCode(code=code, system="SNOMED_CT", display=code) for code in output.snomed_codes]
    radlex_codes = [IndexCode(code=code, system="RadLex", display=code) for code in output.radlex_codes]
    anatomic_locations = [
        OntologySearchResult(concept_id=loc_id, concept_text=loc_id, score=0.0, table_name="anatomic_locations")
        for loc_id in output.anatomic_location_ids
    ]

    if model:
        model_used = model
    elif "enrich_research" in settings.agent_model_overrides:
        model_used = settings.agent_model_overrides["enrich_research"]
    else:
        model_used = settings.default_model

    return FindingEnrichmentResult(
        finding_name=finding_name,
        oifm_id=oifm_id,
        snomed_codes=snomed_codes,
        radlex_codes=radlex_codes,
        body_regions=output.body_regions,
        etiologies=output.etiologies,
        modalities=output.modalities,
        subspecialties=output.subspecialties,
        anatomic_locations=anatomic_locations,
        enrichment_timestamp=datetime.now(timezone.utc),
        model_used=model_used,
        model_tier="base",
    )
