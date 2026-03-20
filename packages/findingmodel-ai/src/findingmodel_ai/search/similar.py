"""Similar Finding Models — 5-Phase Pipeline

Finds existing finding models that are similar enough to a proposed model
that editing them might be better than creating new ones.

Phases:
1. Fast-path: exact match by name/synonyms
2. LLM Planning: generate search terms + metadata hypotheses
3. Multi-Pass Search: unfiltered + metadata-filtered + related_models
4. LLM Selection: structured pick with rejection taxonomy
5. Assembly: build typed result
"""

from __future__ import annotations

import logfire
from findingmodel.index import FindingModelIndex as Index
from pydantic_ai import Agent

from findingmodel_ai import logger
from findingmodel_ai.config import settings
from findingmodel_ai.search.pipeline_helpers import (
    CandidatePool,
    MetadataHypothesis,
    SimilarModelMatch,
    SimilarModelPlan,
    SimilarModelRejection,
    SimilarModelResult,
    SimilarModelSelection,
    validate_selection_in_candidates,
)

# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are a medical terminology specialist. Given a proposed imaging finding,
generate 2-5 effective search terms for finding existing definitions that might
match or overlap, PLUS your best guess at the finding's metadata profile.

**Search term rules:**
- Use standard medical terminology, no acronyms
- Prefer MORE GENERAL terms — the index may use broader names
- NEVER more specific than the finding itself
- Use diverse phrasings for recall
- Do NOT generalize to meta-categories ("finding", "disease", "abnormality")
- Use "lesion" for focal findings, not for diffuse processes

**Metadata hypotheses:**
Fill in your best guesses. These are hints, not hard constraints — leave empty
lists when unsure rather than guessing incorrectly.
"""

SELECTION_SYSTEM_PROMPT = """\
You are an expert medical imaging informatics analyst. Given a proposed finding
and a set of candidate existing definitions, determine which (if any) are close
enough matches that editing them would be better than creating a new definition.

**Selection rules:**
- Match based on canonical clinical meaning, not surface string similarity
- A slightly MORE GENERAL candidate is acceptable and often preferred
  (convergence on common concept)
- A MORE SPECIFIC candidate is NOT acceptable (fragments what should be one group)
- When no match, MUST identify the closest candidate and classify the rejection reason
- Consider name, description, synonyms, tags, and metadata of each candidate
- "edit_existing" = proposed finding should be a synonym or revision of existing
- "create_new" = proposed finding is a distinct clinical concept

**Rejection taxonomy:**
- too_specific: candidate narrows beyond proposed finding
- too_broad: candidate too general to be useful
- wrong_concept: different clinical entity entirely
- definition_mismatch: name matches but meaning differs
- overlapping_scope: partial overlap, neither subsumes other
"""


def create_planning_agent() -> Agent[None, SimilarModelPlan]:
    """Create the Phase 2 planning agent."""
    return Agent[None, SimilarModelPlan](
        model=settings.get_agent_model("similar_plan"),
        output_type=SimilarModelPlan,
        instructions=PLANNING_SYSTEM_PROMPT,
    )


def create_selection_agent() -> Agent[None, SimilarModelSelection]:
    """Create the Phase 4 selection agent."""
    return Agent[None, SimilarModelSelection](
        model=settings.get_agent_model("similar_select"),
        output_type=SimilarModelSelection,
        instructions=SELECTION_SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _build_finding_description(
    finding_name: str,
    description: str | None = None,
    synonyms: list[str] | None = None,
) -> str:
    """Build a text summary of the proposed finding for LLM prompts."""
    parts = [f"Name: {finding_name}"]
    if description:
        parts.append(f"Description: {description}")
    if synonyms:
        parts.append(f"Synonyms: {', '.join(synonyms)}")
    return "\n".join(parts)


def _build_candidate_descriptions(pool: CandidatePool) -> str:
    """Format candidate pool entries for the selection agent prompt."""
    lines: list[str] = []
    for entry in pool.entries:
        parts = [f"- **{entry.oifm_id}** | {entry.name}"]
        if entry.description:
            parts.append(f"  Description: {entry.description}")
        if entry.synonyms:
            parts.append(f"  Synonyms: {', '.join(entry.synonyms)}")
        if entry.tags:
            parts.append(f"  Tags: {', '.join(entry.tags)}")
        if entry.body_regions:
            parts.append(f"  Body regions: {', '.join(br.value for br in entry.body_regions)}")
        if entry.entity_type:
            parts.append(f"  Entity type: {entry.entity_type.value}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def find_similar_models(
    finding_name: str,
    description: str | None = None,
    synonyms: list[str] | None = None,
    index: Index | None = None,
    *,
    existing_model_id: str | None = None,
) -> SimilarModelResult:
    """Find existing finding models similar to a proposed finding.

    5-phase pipeline: fast-path → LLM planning → multi-pass search →
    LLM selection → assembly.

    Args:
        finding_name: Name of the proposed finding
        description: Optional description
        synonyms: Optional list of synonyms
        index: FindingModelIndex (creates one if None)
        existing_model_id: If an existing model ID is available, adds related_models() pass
    """
    with logfire.span("find_similar_models", finding_name=finding_name):
        if index is None:
            index = Index()

        # === Phase 1: Fast-path ===
        with logfire.span("phase1_fast_path"):
            result = await _phase1_fast_path(index, finding_name, synonyms)
            if result is not None:
                logfire.info("Fast-path resolved", resolved=True)
                return result
            logfire.info("Fast-path resolved", resolved=False)

        # === Phase 2: LLM Planning ===
        with logfire.span("phase2_planning"):
            finding_desc = _build_finding_description(finding_name, description, synonyms)
            plan = await _phase2_planning(finding_desc)
            logfire.info(
                "Planning complete",
                terms_generated=len(plan.search_terms),
                has_metadata_hypotheses=bool(
                    plan.metadata_hypotheses.body_regions
                    or plan.metadata_hypotheses.modalities
                    or plan.metadata_hypotheses.entity_type
                    or plan.metadata_hypotheses.subspecialties
                ),
            )

        # === Phase 3: Multi-Pass Search ===
        with logfire.span("phase3_search"):
            pool = await _phase3_search(index, plan, existing_model_id)
            logfire.info("Search complete", candidate_count=len(pool), passes=pool.pass_counts)

        if len(pool) == 0:
            logger.info(f"No candidates found for '{finding_name}'")
            return SimilarModelResult(
                recommendation="create_new",
                search_passes=pool.pass_counts,
                metadata_hypotheses=plan.metadata_hypotheses,
            )

        # === Phase 4: LLM Selection ===
        with logfire.span("phase4_selection"):
            selection = await _phase4_selection(finding_desc, pool)
            logfire.info(
                "Selection complete",
                selected_count=len(selection.selected_ids),
                recommendation=selection.recommendation,
            )

        # === Phase 5: Assembly ===
        with logfire.span("phase5_assembly"):
            result = _phase5_assembly(selection, pool, plan.metadata_hypotheses)
            logfire.info("Assembly complete", recommendation=result.recommendation)

        return result


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


async def _phase1_fast_path(
    index: Index,
    finding_name: str,
    synonyms: list[str] | None,
) -> SimilarModelResult | None:
    """Phase 1: Check for exact match by name or synonyms."""
    # Check name
    existing = await index.get(finding_name)
    if existing:
        logger.info(f"Exact match found for '{finding_name}': {existing.oifm_id}")
        return SimilarModelResult(
            recommendation="edit_existing",
            matches=[SimilarModelMatch(entry=existing, match_reasoning="Exact name match in index")],
            search_passes={"fast_path": 1},
        )

    # Check synonyms (batch via index.get)
    if synonyms:
        for synonym in synonyms:
            existing = await index.get(synonym)
            if existing:
                logger.info(f"Synonym match found for '{synonym}': {existing.oifm_id}")
                return SimilarModelResult(
                    recommendation="edit_existing",
                    matches=[SimilarModelMatch(entry=existing, match_reasoning=f"Synonym match: '{synonym}'")],
                    search_passes={"fast_path": 1},
                )

    return None


async def _phase2_planning(finding_desc: str) -> SimilarModelPlan:
    """Phase 2: Generate search terms and metadata hypotheses via LLM."""
    agent = create_planning_agent()
    prompt = f"Generate search terms and metadata hypotheses for this proposed finding:\n\n{finding_desc}"
    result = await agent.run(prompt)
    return result.output


async def _phase3_search(
    index: Index,
    plan: SimilarModelPlan,
    existing_model_id: str | None,
) -> CandidatePool:
    """Phase 3: Multi-pass search merged into CandidatePool."""
    pool = CandidatePool()

    # Pass 1: Unfiltered text search (recall protection — ALWAYS runs)
    try:
        unfiltered_results = await index.search_batch(plan.search_terms, limit=6)
        for entries in unfiltered_results.values():
            pool.add_many(entries, "unfiltered_text")
    except (ValueError, Exception) as e:
        logger.warning(f"Unfiltered text search failed: {e}")

    # Pass 2: Metadata-filtered search (when hypotheses available)
    hyp = plan.metadata_hypotheses
    has_metadata_hints = hyp.body_regions or hyp.modalities or hyp.entity_type or hyp.subspecialties
    if has_metadata_hints:
        try:
            # Use first 2 terms for filtered pass
            filtered_terms = plan.search_terms[:2]
            filtered_results = await index.search_batch(
                filtered_terms,
                limit=6,
                body_regions=hyp.body_regions or None,
                applicable_modalities=hyp.modalities or None,
                entity_type=hyp.entity_type,
                subspecialties=hyp.subspecialties or None,
            )
            for entries in filtered_results.values():
                pool.add_many(entries, "metadata_filtered")
        except (ValueError, Exception) as e:
            logger.warning(f"Metadata-filtered search failed: {e}")

    # Pass 3: related_models() when caller provides an existing model ID
    if existing_model_id:
        try:
            related = await index.related_models(existing_model_id, limit=6)
            pool.add_many([entry for entry, _ in related], "related_models")
        except (KeyError, Exception) as e:
            logger.warning(f"related_models() failed for {existing_model_id}: {e}")

    return pool


async def _phase4_selection(
    finding_desc: str,
    pool: CandidatePool,
) -> SimilarModelSelection:
    """Phase 4: LLM selection from candidate pool."""
    agent = create_selection_agent()
    candidate_text = _build_candidate_descriptions(pool)

    prompt = f"""Analyze these candidate models for the proposed finding:

**Proposed Finding:**
{finding_desc}

**Candidates ({len(pool)} total):**
{candidate_text}

Select 0-3 models that are close enough matches, or recommend creating a new model.
If recommending create_new, identify the closest candidate and classify why it was rejected."""

    result = await agent.run(prompt)
    selection = result.output

    # Post-validation: remove hallucinated IDs
    selection = validate_selection_in_candidates(selection, pool)

    return selection


def _phase5_assembly(
    selection: SimilarModelSelection,
    pool: CandidatePool,
    metadata_hypotheses: MetadataHypothesis,
) -> SimilarModelResult:
    """Phase 5: Assemble typed result from validated selection."""
    matches: list[SimilarModelMatch] = []
    for oifm_id in selection.selected_ids:
        entry = pool.get(oifm_id)
        if entry:
            matches.append(SimilarModelMatch(entry=entry, match_reasoning=selection.reasoning))

    closest_rejection: SimilarModelRejection | None = None
    if selection.closest_rejection_id and selection.closest_rejection_reason:
        rejection_entry = pool.get(selection.closest_rejection_id)
        if rejection_entry:
            closest_rejection = SimilarModelRejection(
                entry=rejection_entry,
                rejection_reason=selection.closest_rejection_reason,
                reasoning=selection.reasoning,
            )

    # Coherence: if all selected IDs were removed by post-validation,
    # downgrade "edit_existing" to "create_new"
    recommendation = selection.recommendation
    if recommendation == "edit_existing" and not matches:
        logger.warning("Post-validation removed all selected IDs; downgrading to create_new")
        recommendation = "create_new"

    return SimilarModelResult(
        recommendation=recommendation,
        matches=matches,
        closest_rejection=closest_rejection,
        metadata_hypotheses=metadata_hypotheses,
        search_passes=pool.pass_counts,
    )
