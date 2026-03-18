"""Tests for related_models() and RelatedModelWeights (Slice 5).

Tests score calculation, threshold filtering, self-exclusion, empty facets,
and integration tests against the populated fixture DB.

Test data reference (6 models in test_index.duckdb):
- OIFM_MSFT_134126 abdominal aortic aneurysm: abdomen, diagnosis, [AB,VI,ER], [CT,US,MR], aorta+abdominal_aorta
- OIFM_MSFT_573630 aortic dissection:          chest,   diagnosis, [CA,VI,ER], [CT,MR,XR], aorta
- OIFM_MSFT_356221 breast density:             breast,  measurement,[BR],       [MG],       female_breast
- OIFM_MSFT_156954 mammographic malignancy:    breast,  assessment, [BR],       [MG],       female_breast
- OIFM_MSFT_932618 pulmonary embolism:         chest,   diagnosis, [CH,ER],    [CT,XR],    lung
- OIFM_MSFT_367670 ventricular diameters:      chest,   measurement,[CA],       [US,CT,MR], heart
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from findingmodel.facets import (
    AgeProfile,
    AgeStage,
    BodyRegion,
    EntityType,
    ExpectedDuration,
    ExpectedTimeCourse,
    SexSpecificity,
    Subspecialty,
)
from findingmodel.index import FindingModelIndex, IndexEntry, RelatedModelWeights

# ============================================================================
# Helper: create minimal IndexEntry for scoring tests
# ============================================================================


def _make_entry(
    oifm_id: str = "OIFM_TEST_000001",
    name: str = "test entry",
    **kwargs: object,
) -> IndexEntry:
    """Create a minimal IndexEntry with specified facets."""
    defaults = {
        "slug_name": name.replace(" ", "_"),
        "filename": f"{name.replace(' ', '_')}.fm.json",
        "file_hash_sha256": "abc123",
    }
    defaults.update(kwargs)
    return IndexEntry(oifm_id=oifm_id, name=name, **defaults)


# ============================================================================
# _score_related unit tests
# ============================================================================


class TestScoreRelated:
    """Test the deterministic scoring function."""

    def test_identical_facets_max_score(self) -> None:
        """Two entries with identical facets should score highly."""
        source = _make_entry(
            oifm_id="SRC",
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.FINDING,
            subspecialties=[Subspecialty.CH],
        )
        candidate = _make_entry(
            oifm_id="CAND",
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.FINDING,
            subspecialties=[Subspecialty.CH],
        )
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        # body_regions=2.0 + entity_type=3.0 + subspecialties=2.0 = 7.0
        assert score == pytest.approx(7.0)

    def test_no_overlap_zero_score(self) -> None:
        """Entries with no shared facets score zero."""
        source = _make_entry(
            oifm_id="SRC",
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.FINDING,
        )
        candidate = _make_entry(
            oifm_id="CAND",
            body_regions=[BodyRegion.PELVIS],
            entity_type=EntityType.MEASUREMENT,
        )
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        assert score == 0.0

    def test_empty_facets_produce_zero(self) -> None:
        """When facets are None/empty, those dimensions contribute zero (not NaN)."""
        source = _make_entry(oifm_id="SRC")
        candidate = _make_entry(oifm_id="CAND")
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        assert score == 0.0
        import math

        assert not math.isnan(score)

    def test_partial_overlap_fractional_score(self) -> None:
        """Partial list overlap produces fractional score."""
        source = _make_entry(
            oifm_id="SRC",
            body_regions=[BodyRegion.CHEST, BodyRegion.ABDOMEN],
        )
        candidate = _make_entry(
            oifm_id="CAND",
            body_regions=[BodyRegion.CHEST, BodyRegion.PELVIS],
        )
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        # 1 overlap / max(2, 2) = 0.5, weighted by body_regions=2.0 → 1.0
        assert score == pytest.approx(1.0)

    def test_custom_weights(self) -> None:
        """Custom weights change the score calculation."""
        source = _make_entry(
            oifm_id="SRC",
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.FINDING,
        )
        candidate = _make_entry(
            oifm_id="CAND",
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.FINDING,
        )
        w = RelatedModelWeights(body_regions=10.0, entity_type=0.0)
        score = FindingModelIndex._score_related(source, candidate, w)
        assert score == pytest.approx(10.0)

    def test_anatomic_location_overlap(self) -> None:
        """Anatomic location IDs are string lists, overlap scored correctly."""
        source = _make_entry(oifm_id="SRC", anatomic_location_ids=["RID1301", "RID1302"])
        candidate = _make_entry(oifm_id="CAND", anatomic_location_ids=["RID1301", "RID1400"])
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        # 1 overlap / max(2, 2) = 0.5, weighted by 5.0 → 2.5
        assert score == pytest.approx(2.5)

    def test_age_overlap(self) -> None:
        """Age profile overlap contributes to score."""
        source = _make_entry(
            oifm_id="SRC",
            age_profile=AgeProfile(applicability=[AgeStage.ADULT, AgeStage.AGED]),
        )
        candidate = _make_entry(
            oifm_id="CAND",
            age_profile=AgeProfile(applicability=[AgeStage.ADULT]),
        )
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        # 1 overlap / max(2, 1) = 0.5, weighted by 1.0 → 0.5
        assert score == pytest.approx(0.5)

    def test_age_all_ages_both(self) -> None:
        """Both profiles with applicability='all_ages' score full overlap."""
        source = _make_entry(oifm_id="SRC", age_profile=AgeProfile(applicability="all_ages"))
        candidate = _make_entry(oifm_id="CAND", age_profile=AgeProfile(applicability="all_ages"))
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        assert score == pytest.approx(1.0)  # age_overlap weight

    def test_age_all_ages_vs_list(self) -> None:
        """'all_ages' vs a list of stages scores full overlap."""
        source = _make_entry(oifm_id="SRC", age_profile=AgeProfile(applicability="all_ages"))
        candidate = _make_entry(
            oifm_id="CAND",
            age_profile=AgeProfile(applicability=[AgeStage.ADULT]),
        )
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        assert score == pytest.approx(1.0)  # age_overlap weight

    def test_sex_match(self) -> None:
        """Sex specificity match is binary."""
        source = _make_entry(oifm_id="SRC", sex_specificity=SexSpecificity.FEMALE_SPECIFIC)
        candidate = _make_entry(oifm_id="CAND", sex_specificity=SexSpecificity.FEMALE_SPECIFIC)
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        assert score == pytest.approx(1.0)  # sex_match weight

    def test_time_course_match(self) -> None:
        """Time course duration match is binary."""
        source = _make_entry(
            oifm_id="SRC",
            expected_time_course=ExpectedTimeCourse(duration=ExpectedDuration.PERMANENT),
        )
        candidate = _make_entry(
            oifm_id="CAND",
            expected_time_course=ExpectedTimeCourse(duration=ExpectedDuration.PERMANENT),
        )
        w = RelatedModelWeights()
        score = FindingModelIndex._score_related(source, candidate, w)
        assert score == pytest.approx(1.0)  # time_course weight


# ============================================================================
# RelatedModelWeights model tests
# ============================================================================


class TestRelatedModelWeights:
    def test_defaults(self) -> None:
        w = RelatedModelWeights()
        assert w.anatomic_location_ids == 5.0
        assert w.entity_type == 3.0

    def test_override(self) -> None:
        w = RelatedModelWeights(anatomic_location_ids=10.0)
        assert w.anatomic_location_ids == 10.0
        assert w.entity_type == 3.0  # Other defaults unchanged


# ============================================================================
# Integration tests against populated fixture DB
# ============================================================================


@pytest_asyncio.fixture
async def index(prebuilt_db_path: Path) -> AsyncGenerator[FindingModelIndex, None]:
    """Load the pre-built test database (read-only)."""
    async with FindingModelIndex(prebuilt_db_path) as idx:
        yield idx


class TestRelatedModelsIntegration:
    """Integration tests for related_models() against the populated fixture DB."""

    @pytest.mark.asyncio
    async def test_aaa_related_to_aortic_dissection(self, index: FindingModelIndex) -> None:
        """AAA and aortic dissection share aorta (anatomic), diagnosis (entity_type),
        VI+ER (subspecialties), CT+MR (modalities). Should be top related."""
        results = await index.related_models("OIFM_MSFT_134126", min_score=1.0)
        ids = [entry.oifm_id for entry, _ in results]
        assert "OIFM_MSFT_573630" in ids  # aortic dissection
        assert results[0][0].oifm_id == "OIFM_MSFT_573630"

    @pytest.mark.asyncio
    async def test_breast_models_related(self, index: FindingModelIndex) -> None:
        """Breast density and mammographic malignancy share breast (body_region),
        BR (subspecialty), MG (modality), female_breast (anatomic)."""
        results = await index.related_models("OIFM_MSFT_356221", min_score=1.0)
        ids = [entry.oifm_id for entry, _ in results]
        assert "OIFM_MSFT_156954" in ids  # mammographic malignancy
        assert results[0][0].oifm_id == "OIFM_MSFT_156954"

    @pytest.mark.asyncio
    async def test_self_exclusion(self, index: FindingModelIndex) -> None:
        """Source model must not appear in its own results."""
        results = await index.related_models("OIFM_MSFT_134126", min_score=0.0)
        ids = [entry.oifm_id for entry, _ in results]
        assert "OIFM_MSFT_134126" not in ids

    @pytest.mark.asyncio
    async def test_not_found_raises(self, index: FindingModelIndex) -> None:
        """Non-existent model ID raises KeyError."""
        with pytest.raises(KeyError):
            await index.related_models("OIFM_FAKE_999999")

    @pytest.mark.asyncio
    async def test_min_score_filters(self, index: FindingModelIndex) -> None:
        """High min_score threshold excludes weakly related models."""
        results_low = await index.related_models("OIFM_MSFT_932618", min_score=0.0)
        results_high = await index.related_models("OIFM_MSFT_932618", min_score=10.0)
        assert len(results_high) <= len(results_low)

    @pytest.mark.asyncio
    async def test_anatomic_location_overlap_drives_relatedness(self, index: FindingModelIndex) -> None:
        """AAA and aortic dissection both have RID480 (aorta) — this should contribute
        to relatedness via _find_by_list_overlap on anatomic_location_ids."""
        results = await index.related_models("OIFM_MSFT_134126", min_score=0.0)
        scored = {entry.oifm_id: score for entry, score in results}
        # Aortic dissection shares RID480 (aorta) — should score higher than
        # models with no anatomic overlap
        if "OIFM_MSFT_573630" in scored and "OIFM_MSFT_356221" in scored:
            assert scored["OIFM_MSFT_573630"] > scored["OIFM_MSFT_356221"]
