"""Evaluation suite for metadata assignment pipeline using pydantic-evals framework.

This module defines evaluation cases for assessing the assign_metadata() pipeline,
covering blank-start, wrong-existing-reassess, partial-existing-fill-blanks-only,
and existing-codes-and-anatomy scenarios.

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Each evaluator checks a specific aspect of metadata assignment quality
- Hybrid scoring: strict for non-negotiables (0.0 or 1.0), partial credit for quality (0.0-1.0)

EVALUATORS:
- ExecutionSuccessEvaluator: Strict pass/fail on successful run
- RequiredFieldCoverageEvaluator: Completeness for required metadata fields
- GoldMetadataMatchEvaluator: Compare final output to gold for must_match_fields
- PreservationSemanticsEvaluator: Locked fields unchanged in fill_blanks_only mode
- CandidateIntegrityEvaluator: Selected IDs from offered candidates only

LOGFIRE INTEGRATION:
Logfire observability is configured automatically in evals/__init__.py.
No manual instrumentation needed in this module.

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from findingmodel import (
    BodyRegion,
    EntityType,
    FindingModelFull,
    Modality,
    Subspecialty,
)
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, HasMatchingSpan
from pydantic_evals.reporting import EvaluationReport

# =============================================================================
# Data Types
# =============================================================================


class MetadataAssignmentInput(BaseModel):
    """Input for a metadata assignment evaluation case."""

    fixture_stem: str
    assignment_mode: str  # "reassess" or "fill_blanks_only"
    scenario: str  # "blank_start", "wrong_existing_reassess", "partial_existing_fill_blanks_only", "existing_codes_and_anatomy"


class MetadataAssignmentExpectedOutput(BaseModel):
    """Expected output for a metadata assignment evaluation case."""

    gold_fixture_stem: str
    must_match_fields: list[str] = Field(default_factory=list)
    locked_fields: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    expect_unknown_candidate_warnings: bool = False
    require_execution_spans: list[str] = Field(default_factory=list)


class MetadataAssignmentActualOutput(BaseModel):
    """Actual output from running a metadata assignment case."""

    model: FindingModelFull | None = None
    review: Any | None = None  # MetadataAssignmentReview (kept as Any for serialization)
    prepared_input_snapshot: dict[str, Any] = Field(default_factory=dict)
    offered_ontology_candidate_ids: list[str] = Field(default_factory=list)
    selected_ontology_candidate_ids: list[str] = Field(default_factory=list)
    offered_anatomic_candidate_ids: list[str] = Field(default_factory=list)
    selected_anatomic_candidate_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


# =============================================================================
# Fixture Loading & Preparation Helpers
# =============================================================================

EVAL_GOLD_DIR = Path(__file__).with_name("gold")
EVAL_MAX_CONCURRENCY = 3
EVALUATOR_WEIGHTS: dict[str, float] = {
    "GoldMetadataMatchEvaluator": 0.60,
    "RequiredFieldCoverageEvaluator": 0.15,
    "ExecutionSuccessEvaluator": 0.10,
    "PreservationSemanticsEvaluator": 0.10,
    "CandidateIntegrityEvaluator": 0.05,
}


def _has_non_empty_value(value: Any) -> bool:
    """Return True when a fixture field is intentionally populated."""
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return True


def _iter_gold_fixture_stems() -> list[str]:
    """Return all reviewed gold fixture stems in deterministic order."""
    return sorted(path.name.removesuffix(".fm.json") for path in EVAL_GOLD_DIR.glob("*.fm.json"))


def _load_gold_fixture(stem: str) -> FindingModelFull:
    """Load a reviewed gold fixture from packages/findingmodel-ai/evals/gold/{stem}.fm.json."""
    file_path = EVAL_GOLD_DIR / f"{stem}.fm.json"
    return FindingModelFull.model_validate_json(file_path.read_text(encoding="utf-8"))


def _prepare_blank_start(fm: FindingModelFull) -> FindingModelFull:
    """Blank all metadata, index_codes, and anatomic_locations."""
    return fm.model_copy(
        update={
            "body_regions": None,
            "subspecialties": None,
            "etiologies": None,
            "entity_type": None,
            "applicable_modalities": None,
            "expected_time_course": None,
            "age_profile": None,
            "sex_specificity": None,
            "index_codes": None,
            "anatomic_locations": None,
        }
    )


def _prepare_wrong_existing_reassess(fm: FindingModelFull) -> FindingModelFull:
    """Inject deliberately wrong metadata for reassess testing.

    Sets plausible-but-incorrect values to verify the pipeline overrides them.
    """
    return fm.model_copy(
        update={
            "body_regions": [BodyRegion.ABDOMEN],
            "subspecialties": [Subspecialty.SQ],
            "entity_type": EntityType.MEASUREMENT,
            "applicable_modalities": [Modality.US],
            "etiologies": None,
            "expected_time_course": None,
            "age_profile": None,
            "sex_specificity": None,
            "index_codes": None,
            "anatomic_locations": None,
        }
    )


def _prepare_partial_existing_fill_blanks_only(fm: FindingModelFull) -> FindingModelFull:
    """Preserve some fields, blank others for fill_blanks_only testing.

    Keeps body_regions and entity_type from the gold fixture (as locked fields),
    blanks everything else.
    """
    return fm.model_copy(
        update={
            # Preserve body_regions and entity_type from gold
            "subspecialties": None,
            "etiologies": None,
            "applicable_modalities": None,
            "expected_time_course": None,
            "age_profile": None,
            "sex_specificity": None,
            "index_codes": None,
            "anatomic_locations": None,
        }
    )


def _prepare_existing_codes_and_anatomy(fm: FindingModelFull) -> FindingModelFull:
    """Preserve index_codes and anatomic_locations from gold, blank structured metadata."""
    return fm.model_copy(
        update={
            "body_regions": None,
            "subspecialties": None,
            "etiologies": None,
            "entity_type": None,
            "applicable_modalities": None,
            "expected_time_course": None,
            "age_profile": None,
            "sex_specificity": None,
            # Keep index_codes and anatomic_locations from gold
        }
    )


def _snapshot_metadata(fm: FindingModelFull) -> dict[str, Any]:
    """Capture a snapshot of metadata fields for later comparison."""
    return {
        "body_regions": [v.value for v in fm.body_regions] if fm.body_regions else None,
        "subspecialties": [v.value for v in fm.subspecialties] if fm.subspecialties else None,
        "etiologies": [v.value for v in fm.etiologies] if fm.etiologies else None,
        "entity_type": fm.entity_type.value if fm.entity_type else None,
        "applicable_modalities": [v.value for v in fm.applicable_modalities] if fm.applicable_modalities else None,
        "expected_time_course": fm.expected_time_course.model_dump(mode="json") if fm.expected_time_course else None,
        "age_profile": fm.age_profile.model_dump(mode="json") if fm.age_profile else None,
        "sex_specificity": fm.sex_specificity.value if fm.sex_specificity else None,
        "index_codes": [c.model_dump(mode="json") for c in fm.index_codes] if fm.index_codes else None,
        "anatomic_locations": (
            [c.model_dump(mode="json") for c in fm.anatomic_locations] if fm.anatomic_locations else None
        ),
    }


SCENARIO_PREPARERS = {
    "blank_start": _prepare_blank_start,
    "wrong_existing_reassess": _prepare_wrong_existing_reassess,
    "partial_existing_fill_blanks_only": _prepare_partial_existing_fill_blanks_only,
    "existing_codes_and_anatomy": _prepare_existing_codes_and_anatomy,
}


# =============================================================================
# Evaluators
# =============================================================================


class ExecutionSuccessEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Strict pass/fail on whether the pipeline ran successfully.

    Returns:
        1.0 if execution succeeded (no error, model present)
        0.0 if execution failed
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> float:
        if ctx.output.error is not None:
            return 0.0
        if ctx.output.model is None:
            return 0.0
        return 1.0


class RequiredFieldCoverageEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Evaluate completeness for body_regions, subspecialties, entity_type, applicable_modalities.

    Returns:
        Proportion of required fields that are non-null/non-empty (0.0-1.0)
        1.0 if no required_fields specified in metadata
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> float:
        if ctx.metadata is None:
            return 1.0
        if not ctx.metadata.required_fields:
            return 1.0
        if ctx.output.error or ctx.output.model is None:
            return 0.0

        filled = 0
        for field_name in ctx.metadata.required_fields:
            value = getattr(ctx.output.model, field_name, None)
            if value is not None:
                # For list fields, also check non-empty
                if isinstance(value, list) and len(value) == 0:
                    continue
                filled += 1
        return filled / len(ctx.metadata.required_fields)


class GoldMetadataMatchEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Compare final output to gold fixture for must_match_fields.

    Comparison is normalized (sorted, case-insensitive for enums) and order-insensitive.

    Returns:
        Proportion of must_match_fields that match gold (0.0-1.0)
        1.0 if no must_match_fields specified
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> float:
        if ctx.metadata is None:
            return 1.0
        if not ctx.metadata.must_match_fields:
            return 1.0
        if ctx.output.error or ctx.output.model is None:
            return 0.0

        gold = _load_gold_fixture(ctx.metadata.gold_fixture_stem)
        matched = 0
        for field_name in ctx.metadata.must_match_fields:
            if _field_matches_normalized(ctx.output.model, gold, field_name):
                matched += 1
        return matched / len(ctx.metadata.must_match_fields)


class PreservationSemanticsEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Verify locked_fields are unchanged from prepared input in fill_blanks_only mode.

    Only active for fill_blanks_only cases with locked_fields specified.

    Returns:
        1.0 if all locked fields preserved or not applicable
        Proportion of preserved locked fields (0.0-1.0) otherwise
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> float:
        if ctx.metadata is None:
            return 1.0
        if not ctx.metadata.locked_fields:
            return 1.0
        if ctx.inputs.assignment_mode != "fill_blanks_only":
            return 1.0
        if ctx.output.error or ctx.output.model is None:
            return 1.0

        snapshot = ctx.output.prepared_input_snapshot
        if not snapshot:
            return 1.0

        preserved = 0
        for field_name in ctx.metadata.locked_fields:
            actual_snapshot = _snapshot_metadata(ctx.output.model)
            if actual_snapshot.get(field_name) == snapshot.get(field_name):
                preserved += 1
        return preserved / len(ctx.metadata.locked_fields)


class CandidateIntegrityEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Verify selected candidate IDs come from offered candidates.

    Also checks that unknown-candidate warnings only appear when expected.

    Returns:
        1.0 if all integrity checks pass
        0.0 if any selected ID was not in offered set, or unexpected warnings appear
    """

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> float:
        if ctx.output.error or ctx.output.model is None:
            return 1.0

        # Check ontology candidate integrity
        offered_ontology = set(ctx.output.offered_ontology_candidate_ids)
        for selected in ctx.output.selected_ontology_candidate_ids:
            if offered_ontology and selected not in offered_ontology:
                return 0.0

        # Check anatomic candidate integrity
        offered_anatomic = set(ctx.output.offered_anatomic_candidate_ids)
        for selected in ctx.output.selected_anatomic_candidate_ids:
            if offered_anatomic and selected not in offered_anatomic:
                return 0.0

        # Check unknown-candidate warnings
        expect_warnings = ctx.metadata.expect_unknown_candidate_warnings if ctx.metadata else False
        unknown_warnings = [w for w in ctx.output.warnings if "unknown" in w.lower() and "candidate" in w.lower()]
        if unknown_warnings and not expect_warnings:
            return 0.0

        return 1.0


# =============================================================================
# Normalized Field Comparison Helpers
# =============================================================================


def _normalize_field_value(value: Any) -> Any:
    """Normalize a field value for comparison: sort lists, lowercase enum values."""
    if value is None:
        return None
    if isinstance(value, list):
        normalized = [_normalize_field_value(v) for v in value]
        # Sort by string representation for order-insensitive comparison
        return sorted(normalized, key=str)
    if hasattr(value, "value"):
        # Enum
        return value.value.lower()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _field_matches_normalized(actual: FindingModelFull, gold: FindingModelFull, field_name: str) -> bool:
    """Compare a single field between actual and gold with normalization."""
    actual_val = _normalize_field_value(getattr(actual, field_name, None))
    gold_val = _normalize_field_value(getattr(gold, field_name, None))
    return actual_val == gold_val


# =============================================================================
# Task Execution Function
# =============================================================================


async def run_metadata_assignment_task(
    input_data: MetadataAssignmentInput,
) -> MetadataAssignmentActualOutput:
    """Execute a single metadata assignment evaluation case.

    Dataset.evaluate() automatically creates spans and captures inputs/outputs.
    Pydantic AI instrumentation captures agent/model/tool calls.
    """
    from findingmodel_ai.metadata.assignment import assign_metadata

    try:
        # Load and prepare the fixture
        gold_fm = _load_gold_fixture(input_data.fixture_stem)
        preparer = SCENARIO_PREPARERS[input_data.scenario]
        prepared_fm = preparer(gold_fm)
        prepared_snapshot = _snapshot_metadata(prepared_fm)

        # Run the pipeline
        fill_blanks_only = input_data.assignment_mode == "fill_blanks_only"
        result = await assign_metadata(prepared_fm, fill_blanks_only=fill_blanks_only)

        # Extract candidate info from review
        review = result.review

        offered_ontology_ids: list[str] = []
        selected_ontology_ids: list[str] = []
        if review.ontology_candidates:
            for candidate in review.ontology_candidates.canonical_codes:
                code = candidate.code
                cid = f"{code.system}:{code.code}"
                offered_ontology_ids.append(cid)
                selected_ontology_ids.append(cid)
            for candidate in review.ontology_candidates.review_candidates:
                code = candidate.code
                cid = f"{code.system}:{code.code}"
                offered_ontology_ids.append(cid)

        offered_anatomic_ids: list[str] = []
        selected_anatomic_ids: list[str] = []
        for candidate in review.anatomic_candidates:
            code = candidate.location
            cid = f"{code.system}:{code.code}"
            offered_anatomic_ids.append(cid)
            if candidate.selected:
                selected_anatomic_ids.append(cid)

        return MetadataAssignmentActualOutput(
            model=result.model,
            review=review.model_dump(mode="json"),
            prepared_input_snapshot=prepared_snapshot,
            offered_ontology_candidate_ids=offered_ontology_ids,
            selected_ontology_candidate_ids=selected_ontology_ids,
            offered_anatomic_candidate_ids=offered_anatomic_ids,
            selected_anatomic_candidate_ids=selected_anatomic_ids,
            warnings=review.warnings,
        )
    except Exception as e:
        return MetadataAssignmentActualOutput(error=str(e))


# =============================================================================
# Eval Case Definitions
# =============================================================================


def _default_required_fields() -> list[str]:
    return ["body_regions", "subspecialties", "entity_type", "applicable_modalities"]


def _default_must_match_fields() -> list[str]:
    return ["body_regions", "entity_type"]


def _default_span_assertions() -> list[str]:
    return [
        "assign_metadata.ontology_candidates",
        "assign_metadata.anatomic_candidates",
        "assign_metadata.classifier",
    ]


def _required_fields_for_gold(gold_fm: FindingModelFull) -> list[str]:
    """Require only the core fields that the reviewed gold fixture actually populates."""
    return [field_name for field_name in _default_required_fields() if _has_non_empty_value(getattr(gold_fm, field_name))]


def _must_match_fields_for_wrong_existing(gold_fm: FindingModelFull) -> list[str]:
    """Match subspecialties only when the reviewed gold fixture intentionally defines them."""
    must_match = ["body_regions", "entity_type"]
    if _has_non_empty_value(gold_fm.subspecialties):
        must_match.append("subspecialties")
    return must_match


def _build_case(
    *,
    fixture_stem: str,
    gold_fm: FindingModelFull,
    assignment_mode: str,
    scenario: str,
    must_match_fields: list[str],
    locked_fields: list[str] | None = None,
) -> Case[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]:
    """Build a metadata-assignment eval case from a reviewed gold fixture."""
    return Case(
        name=f"{fixture_stem}_{scenario}",
        inputs=MetadataAssignmentInput(
            fixture_stem=fixture_stem,
            assignment_mode=assignment_mode,
            scenario=scenario,
        ),
        metadata=MetadataAssignmentExpectedOutput(
            gold_fixture_stem=fixture_stem,
            must_match_fields=must_match_fields,
            locked_fields=locked_fields or [],
            required_fields=_required_fields_for_gold(gold_fm),
            require_execution_spans=_default_span_assertions(),
        ),
    )


def create_eval_cases() -> list[
    Case[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
]:
    """Create eval cases from the full reviewed gold fixture set."""
    cases: list[Case[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]] = []

    for fixture_stem in _iter_gold_fixture_stems():
        gold_fm = _load_gold_fixture(fixture_stem)

        cases.append(
            _build_case(
                fixture_stem=fixture_stem,
                gold_fm=gold_fm,
                assignment_mode="reassess",
                scenario="blank_start",
                must_match_fields=_default_must_match_fields(),
            )
        )

        cases.append(
            _build_case(
                fixture_stem=fixture_stem,
                gold_fm=gold_fm,
                assignment_mode="reassess",
                scenario="wrong_existing_reassess",
                must_match_fields=_must_match_fields_for_wrong_existing(gold_fm),
            )
        )

        fill_blank_must_match_fields = ["subspecialties"] if _has_non_empty_value(gold_fm.subspecialties) else []
        cases.append(
            _build_case(
                fixture_stem=fixture_stem,
                gold_fm=gold_fm,
                assignment_mode="fill_blanks_only",
                scenario="partial_existing_fill_blanks_only",
                must_match_fields=fill_blank_must_match_fields,
                locked_fields=["body_regions", "entity_type"],
            )
        )

        if _has_non_empty_value(gold_fm.index_codes) and _has_non_empty_value(gold_fm.anatomic_locations):
            cases.append(
                _build_case(
                    fixture_stem=fixture_stem,
                    gold_fm=gold_fm,
                    assignment_mode="reassess",
                    scenario="existing_codes_and_anatomy",
                    must_match_fields=_default_must_match_fields(),
                )
            )

    return cases


# =============================================================================
# Dataset Creation
# =============================================================================

all_cases = create_eval_cases()

evaluators: list[
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
    | HasMatchingSpan
] = [
    ExecutionSuccessEvaluator(),
    RequiredFieldCoverageEvaluator(),
    GoldMetadataMatchEvaluator(),
    PreservationSemanticsEvaluator(),
    CandidateIntegrityEvaluator(),
    # Span assertions: verify key pipeline stages executed
    HasMatchingSpan({"name_contains": "assign_metadata.ontology_candidates"}, "ontology_candidates_span"),
    HasMatchingSpan({"name_contains": "assign_metadata.anatomic_candidates"}, "anatomic_candidates_span"),
    HasMatchingSpan({"name_contains": "assign_metadata.classifier"}, "classifier_span"),
]

metadata_assignment_dataset: Dataset[
    MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
] = Dataset(cases=all_cases, evaluators=evaluators)


# =============================================================================
# Runner
# =============================================================================


async def run_metadata_assignment_evals() -> EvaluationReport[
    MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
]:
    """Run metadata assignment evaluation suite.

    Dataset.evaluate() automatically creates evaluation spans and captures
    all inputs, outputs, and scores for visualization in Logfire.
    """
    report = await metadata_assignment_dataset.evaluate(
        run_metadata_assignment_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
        progress=False,
    )
    return report


def weighted_case_score(
    case: Any,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted case score using evaluator-only weights.

    Span assertions are intentionally excluded from the weighted overall score.
    """
    active_weights = weights or EVALUATOR_WEIGHTS
    weighted_total = 0.0
    total_weight = 0.0

    for evaluator_name, weight in active_weights.items():
        score = case.scores.get(evaluator_name)
        if score is None:
            continue
        weighted_total += score.value * weight
        total_weight += weight

    return weighted_total / total_weight if total_weight else 0.0


def weighted_overall_score(
    report: EvaluationReport[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Average the weighted case scores across the metadata-assignment eval run."""
    if not report.cases:
        return 0.0
    case_scores = [weighted_case_score(case, weights=weights) for case in report.cases]
    return sum(case_scores) / len(case_scores)


if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()

    async def main() -> None:
        print("\nRunning metadata assignment evaluation suite...")
        print("=" * 80)

        report = await run_metadata_assignment_evals()

        print("\n" + "=" * 80)
        print("METADATA ASSIGNMENT EVALUATION RESULTS")
        print("=" * 80 + "\n")

        report.print(
            include_input=False,
            include_output=False,
            include_durations=True,
            width=120,
        )

        overall_score = weighted_overall_score(report)

        print("\n" + "=" * 80)
        print(f"WEIGHTED OVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\n")

    asyncio.run(main())
