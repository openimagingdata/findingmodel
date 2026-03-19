"""Shared infrastructure for the similar-model search pipeline.

Contains types, candidate pool management, and validation helpers
used across the 5-phase pipeline in similar.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from findingmodel.facets import BodyRegion, EntityType, Modality, Subspecialty
from findingmodel.index import IndexEntry
from pydantic import BaseModel, Field

from findingmodel_ai import logger


class ModelMatchRejectionReason(str, Enum):
    """Why a candidate model was not selected as a match."""

    TOO_SPECIFIC = "too_specific"
    TOO_BROAD = "too_broad"
    WRONG_CONCEPT = "wrong_concept"
    DEFINITION_MISMATCH = "definition_mismatch"
    OVERLAPPING_SCOPE = "overlapping_scope"


class MetadataHypothesis(BaseModel):
    """LLM-generated guesses about the finding's metadata profile.

    These are GUESSES — used to add a filtered search pass, not as hard gates.
    """

    body_regions: list[BodyRegion] = Field(default_factory=list)
    modalities: list[Modality] = Field(default_factory=list)
    entity_type: EntityType | None = None
    subspecialties: list[Subspecialty] = Field(default_factory=list)


class SimilarModelPlan(BaseModel):
    """Output of Phase 2: LLM planning step."""

    search_terms: list[str] = Field(min_length=2, max_length=5)
    metadata_hypotheses: MetadataHypothesis = Field(default_factory=MetadataHypothesis)


class SimilarModelSelection(BaseModel):
    """Output of Phase 4: LLM selection step."""

    selected_ids: list[str] = Field(default_factory=list, max_length=3)
    recommendation: Literal["edit_existing", "create_new"]
    reasoning: str
    closest_rejection_id: str | None = None
    closest_rejection_reason: ModelMatchRejectionReason | None = None


class SimilarModelMatch(BaseModel):
    """A matched model with reasoning."""

    entry: IndexEntry
    match_reasoning: str


class SimilarModelRejection(BaseModel):
    """The closest rejected candidate with classification."""

    entry: IndexEntry
    rejection_reason: ModelMatchRejectionReason
    reasoning: str


class SimilarModelResult(BaseModel):
    """Final result of find_similar_models() pipeline."""

    recommendation: Literal["edit_existing", "create_new"]
    matches: list[SimilarModelMatch] = Field(default_factory=list, max_length=3)
    closest_rejection: SimilarModelRejection | None = None
    metadata_hypotheses: MetadataHypothesis | None = None
    search_passes: dict[str, int] = Field(default_factory=dict)


MAX_CANDIDATES = 12


class CandidatePool:
    """Deduplicated candidate container with provenance tracking and max cap."""

    def __init__(self, max_size: int = MAX_CANDIDATES) -> None:
        self._entries: dict[str, IndexEntry] = {}
        self._provenance: dict[str, list[str]] = {}  # oifm_id -> [pass_name, ...]
        self._max_size = max_size

    def add(self, entry: IndexEntry, pass_name: str) -> None:
        """Add an entry from a search pass. Deduplicates by oifm_id."""
        if entry.oifm_id in self._entries:
            if pass_name not in self._provenance[entry.oifm_id]:
                self._provenance[entry.oifm_id].append(pass_name)
            return
        if len(self._entries) >= self._max_size:
            return
        self._entries[entry.oifm_id] = entry
        self._provenance[entry.oifm_id] = [pass_name]

    def add_many(self, entries: list[IndexEntry], pass_name: str) -> None:
        """Add multiple entries from a search pass."""
        for entry in entries:
            self.add(entry, pass_name)

    def get(self, oifm_id: str) -> IndexEntry | None:
        return self._entries.get(oifm_id)

    def contains(self, oifm_id: str) -> bool:
        return oifm_id in self._entries

    @property
    def entries(self) -> list[IndexEntry]:
        return list(self._entries.values())

    @property
    def pass_counts(self) -> dict[str, int]:
        """Return count of entries per search pass."""
        counts: dict[str, int] = {}
        for pass_names in self._provenance.values():
            for name in pass_names:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._entries)


def validate_selection_in_candidates(selection: SimilarModelSelection, pool: CandidatePool) -> SimilarModelSelection:
    """Validate that all selected IDs exist in the candidate pool.

    Hallucinated IDs are logged and removed.
    """
    valid_ids: list[str] = []
    for oifm_id in selection.selected_ids:
        if pool.contains(oifm_id):
            valid_ids.append(oifm_id)
        else:
            logger.warning(f"Post-validation: hallucinated ID '{oifm_id}' not in candidate pool, removing")

    if len(valid_ids) != len(selection.selected_ids):
        selection = selection.model_copy(update={"selected_ids": valid_ids})

    # Also validate closest_rejection_id
    if selection.closest_rejection_id and not pool.contains(selection.closest_rejection_id):
        logger.warning(
            f"Post-validation: hallucinated rejection ID '{selection.closest_rejection_id}' not in candidate pool"
        )
        selection = selection.model_copy(update={"closest_rejection_id": None, "closest_rejection_reason": None})

    return selection
