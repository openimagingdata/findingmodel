"""Tests for metadata-assignment result and review types (Slice 6)."""

from datetime import UTC, datetime

import pytest
from findingmodel.finding_model import ChoiceAttributeIded, ChoiceValueIded, FindingModelFull
from findingmodel_ai.metadata.types import (
    AnatomicCandidate,
    FieldConfidence,
    MetadataAssignmentResult,
    MetadataAssignmentReview,
    OntologyCandidate,
    OntologyCandidateRejectionReason,
    OntologyCandidateRelationship,
    OntologyCandidateReport,
)
from oidm_common.models import IndexCode
from pydantic import ValidationError


def _make_minimal_model() -> FindingModelFull:
    return FindingModelFull(
        oifm_id="OIFM_TEST_000001",
        name="test finding",
        description="A test finding for metadata-assignment type tests.",
        attributes=[
            ChoiceAttributeIded(
                oifma_id="OIFMA_TEST_000001",
                name="Presence",
                values=[
                    ChoiceValueIded(value_code="OIFMA_TEST_000001.0", name="Present"),
                    ChoiceValueIded(value_code="OIFMA_TEST_000001.1", name="Absent"),
                ],
            ),
        ],
    )


def _make_minimal_review() -> MetadataAssignmentReview:
    return MetadataAssignmentReview(
        finding_name="test finding",
        assignment_timestamp=datetime.now(tz=UTC),
        model_used="openai:gpt-4o",
    )


# --- Type validation ---


class TestMetadataAssignmentResultCreation:
    def test_assign_metadata_result_creation(self) -> None:
        """Construct with a minimal FindingModelFull + MetadataAssignmentReview."""
        result = MetadataAssignmentResult(model=_make_minimal_model(), review=_make_minimal_review())
        assert result.model.oifm_id == "OIFM_TEST_000001"
        assert result.review.finding_name == "test finding"

    def test_assign_metadata_review_required_fields(self) -> None:
        """Omitting required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            MetadataAssignmentReview()  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            MetadataAssignmentReview(finding_name="x")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            MetadataAssignmentReview(finding_name="x", assignment_timestamp=datetime.now(tz=UTC))  # type: ignore[call-arg]

    def test_assign_metadata_review_minimal(self) -> None:
        """Only required fields — verify optional fields default correctly."""
        review = _make_minimal_review()
        assert review.ontology_candidates.canonical_codes == []
        assert review.ontology_candidates.review_candidates == []
        assert review.anatomic_candidates == []
        assert review.classification_rationale == ""
        assert review.field_confidence == {}
        assert review.timings == {}
        assert review.warnings == []
        assert review.oifm_id is None
        assert review.logfire_trace_id is None

    def test_ontology_candidate_relationship_values(self) -> None:
        """All 6 enum values are valid strings."""
        expected = {"exact_match", "clinically_substitutable", "narrower", "broader", "related", "complication"}
        actual = {v.value for v in OntologyCandidateRelationship}
        assert actual == expected

    def test_ontology_candidate_rejection_reason_values(self) -> None:
        """Ontology review rejection reasons are explicit and stable."""
        expected = {
            "too_specific",
            "too_broad",
            "wrong_concept",
            "definition_mismatch",
            "overlapping_scope",
        }
        actual = {v.value for v in OntologyCandidateRejectionReason}
        assert actual == expected

    def test_field_confidence_values(self) -> None:
        """All 3 enum values are valid strings."""
        expected = {"high", "medium", "low"}
        actual = {v.value for v in FieldConfidence}
        assert actual == expected


# --- Serialization ---


class TestSerialization:
    def test_assign_metadata_review_json_roundtrip(self) -> None:
        """model_dump_json() then model_validate_json() produces identical object."""
        review = MetadataAssignmentReview(
            oifm_id="OIFM_TEST_000001",
            finding_name="test finding",
            assignment_timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
            model_used="openai:gpt-4o",
            logfire_trace_id="trace_123",
            ontology_candidates=OntologyCandidateReport(
                canonical_codes=[
                    OntologyCandidate(
                        code=IndexCode(system="TEST", code="TEST_001"),
                        relationship=OntologyCandidateRelationship.EXACT_MATCH,
                        rationale="Direct match",
                    )
                ],
            ),
            anatomic_candidates=[
                AnatomicCandidate(
                    location=IndexCode(system="TEST", code="TEST_LOC_001", display="Test Location"),
                    selected=True,
                    rationale="Primary",
                ),
            ],
            classification_rationale="Test rationale",
            field_confidence={"description": FieldConfidence.HIGH},
            timings={"ontology_search": 1.23},
            warnings=["Something to note"],
        )
        json_str = review.model_dump_json()
        restored = MetadataAssignmentReview.model_validate_json(json_str)
        assert restored == review

    def test_assign_metadata_result_json_serializable(self) -> None:
        """Both .model and .review can be serialized independently."""
        result = MetadataAssignmentResult(model=_make_minimal_model(), review=_make_minimal_review())
        model_json = result.model.model_dump_json()
        review_json = result.review.model_dump_json()
        assert isinstance(model_json, str)
        assert isinstance(review_json, str)
        # Verify they parse back
        FindingModelFull.model_validate_json(model_json)
        MetadataAssignmentReview.model_validate_json(review_json)


# --- Separation invariant ---


class TestSeparationInvariant:
    def test_canonical_model_excludes_review_fields(self) -> None:
        """result.model must have ZERO keys from the review vocabulary."""
        result = MetadataAssignmentResult(model=_make_minimal_model(), review=_make_minimal_review())
        model_keys = set(result.model.model_dump().keys())
        review_only_keys = {
            "ontology_candidates",
            "timings",
            "classification_rationale",
            "field_confidence",
            "warnings",
            "anatomic_candidates",
            "assignment_timestamp",
            "model_used",
            "logfire_trace_id",
        }
        overlap = model_keys & review_only_keys
        assert overlap == set(), f"Model contains review fields: {overlap}"

    def test_review_does_not_contain_model_attributes(self) -> None:
        """Review must have ZERO keys that belong to the canonical model (except oifm_id)."""
        result = MetadataAssignmentResult(model=_make_minimal_model(), review=_make_minimal_review())
        review_keys = set(result.review.model_dump().keys())
        model_only_keys = {"attributes", "synonyms", "tags"}
        overlap = review_keys & model_only_keys
        assert overlap == set(), f"Review contains model fields: {overlap}"
