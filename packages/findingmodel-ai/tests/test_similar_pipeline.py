"""Tests for the find_similar_models() 5-phase pipeline (Slice 8).

Tests pipeline_helpers types, CandidatePool, validation, and
individual phase behavior using FunctionModel.
"""

from __future__ import annotations

import pytest
from findingmodel.index import IndexEntry
from findingmodel_ai.search.pipeline_helpers import (
    CandidatePool,
    MetadataHypothesis,
    ModelMatchRejectionReason,
    SimilarModelPlan,
    SimilarModelResult,
    SimilarModelSelection,
    validate_selection_in_candidates,
)
from findingmodel_ai.search.similar import _phase1_fast_path, _phase5_assembly
from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False


# ============================================================================
# Helpers
# ============================================================================


def _make_entry(oifm_id: str = "OIFM_TEST_000001", name: str = "test entry") -> IndexEntry:
    return IndexEntry(
        oifm_id=oifm_id,
        name=name,
        slug_name=name.replace(" ", "_"),
        filename=f"{name.replace(' ', '_')}.fm.json",
        file_hash_sha256="abc123",
    )


# ============================================================================
# CandidatePool tests
# ============================================================================


class TestCandidatePool:
    def test_add_deduplicates(self) -> None:
        pool = CandidatePool()
        entry = _make_entry("ID1", "model A")
        pool.add(entry, "pass1")
        pool.add(entry, "pass2")
        assert len(pool) == 1
        assert pool.pass_counts == {"pass1": 1, "pass2": 1}

    def test_max_cap(self) -> None:
        pool = CandidatePool(max_size=3)
        for i in range(5):
            pool.add(_make_entry(f"ID_{i}", f"model {i}"), "pass1")
        assert len(pool) == 3

    def test_add_many(self) -> None:
        pool = CandidatePool()
        entries = [_make_entry(f"ID_{i}", f"model {i}") for i in range(4)]
        pool.add_many(entries, "pass1")
        assert len(pool) == 4
        assert pool.pass_counts == {"pass1": 4}

    def test_contains_and_get(self) -> None:
        pool = CandidatePool()
        entry = _make_entry("ID1", "model A")
        pool.add(entry, "pass1")
        assert pool.contains("ID1")
        assert not pool.contains("ID_MISSING")
        assert pool.get("ID1") is entry
        assert pool.get("ID_MISSING") is None


# ============================================================================
# validate_selection_in_candidates tests
# ============================================================================


class TestValidateSelection:
    def test_valid_ids_pass_through(self) -> None:
        pool = CandidatePool()
        pool.add(_make_entry("ID1"), "pass1")
        pool.add(_make_entry("ID2", "other"), "pass1")

        selection = SimilarModelSelection(
            selected_ids=["ID1", "ID2"],
            recommendation="edit_existing",
            reasoning="test",
        )
        result = validate_selection_in_candidates(selection, pool)
        assert result.selected_ids == ["ID1", "ID2"]

    def test_hallucinated_ids_removed(self) -> None:
        pool = CandidatePool()
        pool.add(_make_entry("ID1"), "pass1")

        selection = SimilarModelSelection(
            selected_ids=["ID1", "HALLUCINATED_ID"],
            recommendation="edit_existing",
            reasoning="test",
        )
        result = validate_selection_in_candidates(selection, pool)
        assert result.selected_ids == ["ID1"]

    def test_hallucinated_rejection_id_nullified(self) -> None:
        pool = CandidatePool()
        pool.add(_make_entry("ID1"), "pass1")

        selection = SimilarModelSelection(
            selected_ids=[],
            recommendation="create_new",
            reasoning="test",
            closest_rejection_id="HALLUCINATED",
            closest_rejection_reason=ModelMatchRejectionReason.WRONG_CONCEPT,
        )
        result = validate_selection_in_candidates(selection, pool)
        assert result.closest_rejection_id is None
        assert result.closest_rejection_reason is None

    def test_valid_rejection_id_preserved(self) -> None:
        pool = CandidatePool()
        pool.add(_make_entry("ID1"), "pass1")

        selection = SimilarModelSelection(
            selected_ids=[],
            recommendation="create_new",
            reasoning="test",
            closest_rejection_id="ID1",
            closest_rejection_reason=ModelMatchRejectionReason.TOO_BROAD,
        )
        result = validate_selection_in_candidates(selection, pool)
        assert result.closest_rejection_id == "ID1"
        assert result.closest_rejection_reason == ModelMatchRejectionReason.TOO_BROAD


# ============================================================================
# Phase 1: Fast-path tests
# ============================================================================


class TestPhase1FastPath:
    @pytest.mark.asyncio
    async def test_exact_name_match(self) -> None:
        """Fast-path returns result on exact name match."""
        from unittest.mock import AsyncMock

        mock_index = AsyncMock()
        mock_entry = _make_entry("OIFM_TEST_001", "pneumothorax")
        mock_index.get = AsyncMock(return_value=mock_entry)

        result = await _phase1_fast_path(mock_index, "pneumothorax", None)

        assert result is not None
        assert result.recommendation == "edit_existing"
        assert len(result.matches) == 1
        assert result.matches[0].entry.oifm_id == "OIFM_TEST_001"

    @pytest.mark.asyncio
    async def test_synonym_match(self) -> None:
        """Fast-path returns result on synonym match."""
        from unittest.mock import AsyncMock

        mock_index = AsyncMock()
        # First call (name) returns None, second call (synonym) returns match
        mock_entry = _make_entry("OIFM_TEST_001", "pneumothorax")
        mock_index.get = AsyncMock(side_effect=[None, mock_entry])

        result = await _phase1_fast_path(mock_index, "air in chest", ["pneumothorax"])

        assert result is not None
        assert result.recommendation == "edit_existing"

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self) -> None:
        """Fast-path returns None when no exact match found."""
        from unittest.mock import AsyncMock

        mock_index = AsyncMock()
        mock_index.get = AsyncMock(return_value=None)

        result = await _phase1_fast_path(mock_index, "novel finding", ["new synonym"])
        assert result is None


# ============================================================================
# Phase 5: Assembly tests
# ============================================================================


class TestPhase5Assembly:
    def test_assembly_edit_existing(self) -> None:
        pool = CandidatePool()
        entry1 = _make_entry("ID1", "model A")
        pool.add(entry1, "pass1")

        selection = SimilarModelSelection(
            selected_ids=["ID1"],
            recommendation="edit_existing",
            reasoning="Close match",
        )
        result = _phase5_assembly(selection, pool, MetadataHypothesis())

        assert result.recommendation == "edit_existing"
        assert len(result.matches) == 1
        assert result.matches[0].entry.oifm_id == "ID1"
        assert result.closest_rejection is None

    def test_assembly_create_new_with_rejection(self) -> None:
        pool = CandidatePool()
        entry1 = _make_entry("ID1", "model A")
        pool.add(entry1, "pass1")

        selection = SimilarModelSelection(
            selected_ids=[],
            recommendation="create_new",
            reasoning="No good match",
            closest_rejection_id="ID1",
            closest_rejection_reason=ModelMatchRejectionReason.TOO_SPECIFIC,
        )
        result = _phase5_assembly(selection, pool, MetadataHypothesis())

        assert result.recommendation == "create_new"
        assert len(result.matches) == 0
        assert result.closest_rejection is not None
        assert result.closest_rejection.rejection_reason == ModelMatchRejectionReason.TOO_SPECIFIC

    def test_assembly_downgrades_edit_existing_with_no_matches(self) -> None:
        """If all selected IDs were hallucinated, recommendation must downgrade to create_new."""
        pool = CandidatePool()
        pool.add(_make_entry("ID1", "model A"), "pass1")

        # LLM said edit_existing but picked only hallucinated IDs (already removed by validation)
        selection = SimilarModelSelection(
            selected_ids=[],  # all removed by validate_selection_in_candidates
            recommendation="edit_existing",
            reasoning="Thought there was a match",
        )
        result = _phase5_assembly(selection, pool, MetadataHypothesis())

        assert result.recommendation == "create_new"
        assert len(result.matches) == 0

    def test_assembly_includes_search_passes(self) -> None:
        pool = CandidatePool()
        pool.add(_make_entry("ID1", "a"), "unfiltered_text")
        pool.add(_make_entry("ID2", "b"), "metadata_filtered")

        selection = SimilarModelSelection(
            selected_ids=["ID1"],
            recommendation="edit_existing",
            reasoning="Match",
        )
        result = _phase5_assembly(selection, pool, MetadataHypothesis())
        assert "unfiltered_text" in result.search_passes
        assert "metadata_filtered" in result.search_passes


# ============================================================================
# Result type tests
# ============================================================================


class TestResultTypes:
    def test_similar_model_result_serialization(self) -> None:
        result = SimilarModelResult(
            recommendation="create_new",
            matches=[],
            metadata_hypotheses=MetadataHypothesis(),
            search_passes={"unfiltered_text": 5, "metadata_filtered": 3},
        )
        data = result.model_dump()
        assert data["recommendation"] == "create_new"
        assert data["search_passes"]["unfiltered_text"] == 5
        # Round-trip
        restored = SimilarModelResult.model_validate(data)
        assert restored.recommendation == "create_new"

    def test_rejection_reason_enum_values(self) -> None:
        assert ModelMatchRejectionReason.TOO_SPECIFIC.value == "too_specific"
        assert ModelMatchRejectionReason.TOO_BROAD.value == "too_broad"
        assert ModelMatchRejectionReason.WRONG_CONCEPT.value == "wrong_concept"
        assert ModelMatchRejectionReason.DEFINITION_MISMATCH.value == "definition_mismatch"
        assert ModelMatchRejectionReason.OVERLAPPING_SCOPE.value == "overlapping_scope"

    def test_facet_hypothesis_defaults(self) -> None:
        hyp = MetadataHypothesis()
        assert hyp.body_regions == []
        assert hyp.modalities == []
        assert hyp.entity_type is None
        assert hyp.subspecialties == []

    def test_similar_model_plan_validation(self) -> None:
        plan = SimilarModelPlan(search_terms=["term1", "term2"])
        assert len(plan.search_terms) == 2
        assert plan.metadata_hypotheses is not None

    def test_similar_model_plan_min_terms(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SimilarModelPlan(search_terms=["only_one"])
