"""Evaluation suite for metadata assignment pipeline using pydantic-evals framework.

This module defines evaluation cases for assessing the assign_metadata() pipeline,
covering blank-start, wrong-existing-reassess, partial-existing-fill-blanks-only,
and existing-codes-and-anatomy scenarios.

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Gate evaluators return assertions and are not quality scores
- Metadata quality evaluators return per-field scores (0.0-1.0)

EVALUATORS:
- ExecutionSuccessEvaluator: Gate for successful run
- RequiredFieldCoverageEvaluator: Gate for required metadata fields
- MetadataQualityEvaluator: Field-level metadata quality scores
- PreservationSemanticsEvaluator: Locked fields unchanged in fill_blanks_only mode
- CandidateIntegrityEvaluator: Gate for selected IDs from offered candidates only

LOGFIRE INTEGRATION:
Logfire observability is configured automatically in evals/__init__.py.
No manual instrumentation needed in this module.

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, HasMatchingSpan
from pydantic_evals.reporting import EvaluationReport

from evals.metadata_scoring import (
    score_code_selection,
    score_entity_type,
    score_expected_time_course,
    score_optional_field_match,
)

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
QUALITY_FIELD_WEIGHTS: dict[str, float] = {
    "quality.entity_type": 0.15,
    "quality.body_regions": 0.12,
    "quality.subspecialties": 0.14,
    "quality.applicable_modalities": 0.14,
    "quality.etiologies": 0.10,
    "quality.expected_time_course": 0.10,
    "quality.index_codes": 0.10,
    "quality.anatomic_locations": 0.10,
    "quality.age_profile": 0.025,
    "quality.sex_specificity": 0.025,
}
CORE_QUALITY_FIELDS = ("entity_type", "body_regions", "subspecialties", "applicable_modalities")


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
    """Gate whether the pipeline ran successfully."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> EvaluationReason:
        if ctx.output.error is not None:
            return EvaluationReason(value=False, reason=ctx.output.error)
        if ctx.output.model is None:
            return EvaluationReason(value=False, reason="assignment returned no model")
        return EvaluationReason(value=True, reason="assignment returned a parsed model")


class RequiredFieldCoverageEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Gate required metadata fields."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> EvaluationReason:
        if ctx.metadata is None:
            return EvaluationReason(value=True, reason="no required fields configured")
        if not ctx.metadata.required_fields:
            return EvaluationReason(value=True, reason="no required fields configured")
        if ctx.output.error or ctx.output.model is None:
            return EvaluationReason(value=False, reason="assignment did not return a model")

        missing: list[str] = []
        for field_name in ctx.metadata.required_fields:
            value = getattr(ctx.output.model, field_name, None)
            if value is None or (isinstance(value, list) and len(value) == 0):
                missing.append(field_name)
        if missing:
            return EvaluationReason(value=False, reason=f"missing required fields: {', '.join(missing)}")
        return EvaluationReason(value=True, reason="required fields present")


class MetadataQualityEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Return field-level metadata quality scores."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> dict[str, EvaluationReason]:
        if ctx.metadata is None:
            return {}
        if ctx.output.error or ctx.output.model is None:
            return {
                "quality.entity_type": EvaluationReason(
                    value=0.0,
                    reason="assignment did not return a model",
                )
            }

        gold = _load_gold_fixture(ctx.metadata.gold_fixture_stem)
        return _quality_field_scores(ctx.output.model, gold, ctx.output.prepared_input_snapshot)


class PreservationSemanticsEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Gate locked-field preservation in fill_blanks_only mode."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> EvaluationReason:
        if ctx.metadata is None:
            return EvaluationReason(value=True, reason="no locked fields configured")
        if not ctx.metadata.locked_fields:
            return EvaluationReason(value=True, reason="no locked fields configured")
        if ctx.inputs.assignment_mode != "fill_blanks_only":
            return EvaluationReason(value=True, reason="not a fill_blanks_only case")
        if ctx.output.error or ctx.output.model is None:
            return EvaluationReason(value=False, reason="assignment did not return a model")

        snapshot = ctx.output.prepared_input_snapshot
        if not snapshot:
            return EvaluationReason(value=False, reason="missing prepared input snapshot")

        changed: list[str] = []
        actual_snapshot = _snapshot_metadata(ctx.output.model)
        for field_name in ctx.metadata.locked_fields:
            if actual_snapshot.get(field_name) != snapshot.get(field_name):
                changed.append(field_name)
        if changed:
            return EvaluationReason(value=False, reason=f"locked fields changed: {', '.join(changed)}")
        return EvaluationReason(value=True, reason="locked fields preserved")


class CandidateIntegrityEvaluator(
    Evaluator[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]
):
    """Gate selected candidate IDs and unexpected unknown-candidate warnings."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
        ],
    ) -> EvaluationReason:
        if ctx.output.error or ctx.output.model is None:
            return EvaluationReason(value=False, reason="assignment did not return a model")

        # Check ontology candidate integrity
        offered_ontology = set(ctx.output.offered_ontology_candidate_ids)
        for selected in ctx.output.selected_ontology_candidate_ids:
            if offered_ontology and selected not in offered_ontology:
                return EvaluationReason(value=False, reason=f"selected unknown ontology candidate: {selected}")

        # Check anatomic candidate integrity
        offered_anatomic = set(ctx.output.offered_anatomic_candidate_ids)
        for selected in ctx.output.selected_anatomic_candidate_ids:
            if offered_anatomic and selected not in offered_anatomic:
                return EvaluationReason(value=False, reason=f"selected unknown anatomic candidate: {selected}")

        # Check unknown-candidate warnings
        expect_warnings = ctx.metadata.expect_unknown_candidate_warnings if ctx.metadata else False
        unknown_warnings = [w for w in ctx.output.warnings if "unknown" in w.lower() and "candidate" in w.lower()]
        if unknown_warnings and not expect_warnings:
            return EvaluationReason(value=False, reason="unexpected unknown-candidate warning")

        return EvaluationReason(value=True, reason="candidate integrity checks passed")


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
    return bool(actual_val == gold_val)


def _has_snapshot_value(snapshot: dict[str, Any], field_name: str) -> bool:
    return _has_non_empty_value(snapshot.get(field_name))


def _quality_field_applicable(
    actual: FindingModelFull,
    gold: FindingModelFull,
    field_name: str,
    prepared_snapshot: dict[str, Any],
) -> bool:
    if field_name in CORE_QUALITY_FIELDS:
        return True
    return any((
        _has_non_empty_value(getattr(actual, field_name, None)),
        _has_non_empty_value(getattr(gold, field_name, None)),
        _has_snapshot_value(prepared_snapshot, field_name),
    ))


def _code_keys(value: Any) -> list[str]:
    if not value:
        return []
    return [f"{code.system}:{code.code}" for code in value]


def _snapshot_code_keys(snapshot: dict[str, Any], field_name: str) -> list[str]:
    values = snapshot.get(field_name) or []
    return [f"{item['system']}:{item['code']}" for item in values]


def _score_field_match(
    actual: FindingModelFull,
    gold: FindingModelFull,
    field_name: str,
    prepared_snapshot: dict[str, Any],
) -> float:
    """Score field agreement with partial credit for lower-cost mistakes."""
    if field_name == "entity_type":
        return score_entity_type(actual.entity_type, gold.entity_type)
    if field_name == "expected_time_course":
        return score_expected_time_course(actual.expected_time_course, gold.expected_time_course)
    if field_name == "index_codes":
        return score_code_selection(
            _code_keys(actual.index_codes),
            _code_keys(gold.index_codes),
            existing=_snapshot_code_keys(prepared_snapshot, "index_codes"),
        )
    if field_name == "anatomic_locations":
        return score_code_selection(
            _code_keys(actual.anatomic_locations),
            _code_keys(gold.anatomic_locations),
            existing=_snapshot_code_keys(prepared_snapshot, "anatomic_locations"),
        )
    return score_optional_field_match(
        _normalize_field_value(getattr(actual, field_name, None)),
        _normalize_field_value(getattr(gold, field_name, None)),
    )


def _quality_field_scores(
    actual: FindingModelFull,
    gold: FindingModelFull,
    prepared_snapshot: dict[str, Any],
) -> dict[str, EvaluationReason]:
    scores: dict[str, EvaluationReason] = {}
    for score_name in QUALITY_FIELD_WEIGHTS:
        field_name = score_name.removeprefix("quality.")
        if not _quality_field_applicable(actual, gold, field_name, prepared_snapshot):
            continue
        score = _score_field_match(actual, gold, field_name, prepared_snapshot)
        actual_value = _normalize_field_value(getattr(actual, field_name, None))
        gold_value = _normalize_field_value(getattr(gold, field_name, None))
        scores[score_name] = EvaluationReason(
            value=score,
            reason=f"actual={actual_value!r}; gold={gold_value!r}",
        )
    return scores


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

    stage = "load_fixture"
    try:
        # Load and prepare the fixture
        gold_fm = _load_gold_fixture(input_data.fixture_stem)
        stage = "prepare_fixture"
        preparer = SCENARIO_PREPARERS[input_data.scenario]
        prepared_fm = preparer(gold_fm)
        prepared_snapshot = _snapshot_metadata(prepared_fm)

        # Run the pipeline
        stage = "assign_metadata"
        print(f"[metadata_assignment] {input_data.fixture_stem}/{input_data.scenario}: {stage}")
        fill_blanks_only = input_data.assignment_mode == "fill_blanks_only"
        result = await assign_metadata(prepared_fm, fill_blanks_only=fill_blanks_only)

        # Extract candidate info from review
        stage = "summarize_review"
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
        for anatomic_candidate in review.anatomic_candidates:
            code = anatomic_candidate.location
            cid = f"{code.system}:{code.code}"
            offered_anatomic_ids.append(cid)
            if anatomic_candidate.selected:
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
        return MetadataAssignmentActualOutput(error=f"{input_data.fixture_stem}/{input_data.scenario} {stage}: {e}")


# =============================================================================
# Eval Case Definitions
# =============================================================================


def _default_required_fields() -> list[str]:
    return ["entity_type"]


def _default_must_match_fields() -> list[str]:
    return ["body_regions", "entity_type", "applicable_modalities"]


def _default_span_assertions() -> list[str]:
    return [
        "assign_metadata.ontology_candidates",
        "assign_metadata.anatomic_candidates",
        "assign_metadata.focused_decisions",
    ]


def _required_fields_for_gold(gold_fm: FindingModelFull) -> list[str]:
    """Entity type is the only required metadata field."""
    _ = gold_fm
    return _default_required_fields()


def _must_match_fields_for_wrong_existing(gold_fm: FindingModelFull) -> list[str]:
    """Score broad metadata quality, adding subspecialty only when gold defines it."""
    must_match = _default_must_match_fields()
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


def create_eval_cases(
    *,
    fixture: str | None = None,
    fixture_sample: int | None = None,
    seed: int = 0,
    scenario: str | None = None,
    case_name: str | None = None,
    limit: int | None = None,
) -> list[Case[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]]:
    """Create eval cases from the full reviewed gold fixture set."""
    cases: list[Case[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]] = []

    fixture_stems = _iter_gold_fixture_stems()
    if fixture is not None:
        fixture_stems = [fixture_stem for fixture_stem in fixture_stems if fixture_stem == fixture]
    if fixture_sample is not None:
        sample_size = min(fixture_sample, len(fixture_stems))
        fixture_stems = sorted(random.Random(seed).sample(fixture_stems, sample_size))

    for fixture_stem in fixture_stems:
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

    if scenario is not None:
        cases = [case for case in cases if case.inputs.scenario == scenario]
    if case_name is not None:
        cases = [case for case in cases if case.name == case_name]
    if limit is not None:
        cases = cases[:limit]
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
    MetadataQualityEvaluator(),
    PreservationSemanticsEvaluator(),
    CandidateIntegrityEvaluator(),
    # Span assertions: verify key pipeline stages executed
    HasMatchingSpan({"name_contains": "assign_metadata.ontology_candidates"}, "ontology_candidates_span"),
    HasMatchingSpan({"name_contains": "assign_metadata.anatomic_candidates"}, "anatomic_candidates_span"),
    HasMatchingSpan({"name_contains": "assign_metadata.focused_decisions"}, "focused_decisions_span"),
]

metadata_assignment_dataset: Dataset[
    MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput
] = Dataset(cases=all_cases, evaluators=evaluators)


def build_metadata_assignment_dataset(
    *,
    fixture: str | None = None,
    fixture_sample: int | None = None,
    seed: int = 0,
    scenario: str | None = None,
    case_name: str | None = None,
    limit: int | None = None,
) -> Dataset[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]:
    """Build a filtered metadata-assignment dataset for CLI and task targets."""

    return Dataset(
        cases=create_eval_cases(
            fixture=fixture,
            fixture_sample=fixture_sample,
            seed=seed,
            scenario=scenario,
            case_name=case_name,
            limit=limit,
        ),
        evaluators=evaluators,
    )


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
        progress=True,
    )
    return report


async def run_filtered_metadata_assignment_evals(
    *,
    fixture: str | None = None,
    fixture_sample: int | None = None,
    seed: int = 0,
    scenario: str | None = None,
    case_name: str | None = None,
    limit: int | None = None,
) -> EvaluationReport[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput]:
    """Run a filtered metadata assignment evaluation suite."""

    dataset = build_metadata_assignment_dataset(
        fixture=fixture,
        fixture_sample=fixture_sample,
        seed=seed,
        scenario=scenario,
        case_name=case_name,
        limit=limit,
    )
    return await dataset.evaluate(
        run_metadata_assignment_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
        progress=True,
    )


def metadata_quality_case_score(
    case: Any,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted metadata-quality score for one case."""
    active_weights = weights or QUALITY_FIELD_WEIGHTS
    weighted_total = 0.0
    total_weight = 0.0

    for score_name, weight in active_weights.items():
        score = case.scores.get(score_name)
        if score is None:
            continue
        weighted_total += score.value * weight
        total_weight += weight

    return weighted_total / total_weight if total_weight else 0.0


def metadata_quality_overall_score(
    report: EvaluationReport[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Average metadata-quality scores across the metadata-assignment eval run."""
    if not report.cases:
        return 0.0
    case_scores = [metadata_quality_case_score(case, weights=weights) for case in report.cases]
    return sum(case_scores) / len(case_scores)


def metadata_gate_failures(
    report: EvaluationReport[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput],
) -> list[tuple[str, str, str | None]]:
    """Return failed gate assertions as case, gate, reason tuples."""

    failures: list[tuple[str, str, str | None]] = []
    for case in report.cases:
        for assertion_name, assertion in case.assertions.items():
            if assertion.value is False:
                failures.append((case.name or "<unnamed>", assertion_name, assertion.reason))
    return failures


def quality_field_averages(
    report: EvaluationReport[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput],
) -> dict[str, float]:
    """Return average score for each quality field present in the report."""

    averages: dict[str, float] = {}
    for score_name in QUALITY_FIELD_WEIGHTS:
        values = [case.scores[score_name].value for case in report.cases if score_name in case.scores]
        if values:
            averages[score_name] = sum(values) / len(values)
    return averages


def weakest_quality_fields(case: Any, *, limit: int = 3) -> list[str]:
    """Return the lowest quality field labels for one report case."""

    scored_fields = [
        (name, score.value)
        for name, score in case.scores.items()
        if name.startswith("quality.") and isinstance(score.value, int | float)
    ]
    return [
        f"{name.removeprefix('quality.')}={value:.2f}"
        for name, value in sorted(scored_fields, key=lambda item: item[1])[:limit]
    ]


def print_metadata_assignment_summary(
    report: EvaluationReport[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput],
) -> None:
    """Print gate status and metadata-quality summary."""

    failures = metadata_gate_failures(report)
    print("\nGATES: " + ("FAIL" if failures else "PASS"))
    if failures:
        print("Gate failures:")
        for case_name, assertion_name, reason in failures:
            detail = f" ({reason})" if reason else ""
            print(f"- {case_name}: {assertion_name}{detail}")

    print(f"\nMETADATA QUALITY: {metadata_quality_overall_score(report):.2f}")
    for score_name, average in quality_field_averages(report).items():
        print(f"{score_name.removeprefix('quality.')}: {average:.2f}")

    case_scores = [(case, metadata_quality_case_score(case)) for case in report.cases]
    print("Lowest scoring cases:")
    for case, score in sorted(case_scores, key=lambda item: item[1])[:5]:
        weak_fields = ", ".join(weakest_quality_fields(case))
        suffix = f" ({weak_fields})" if weak_fields else ""
        print(f"- {case.name or '<unnamed>'}: {score:.2f}{suffix}")


def _json_or_null(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True, default=str)


def _quality_score_reason(case: Any, score_name: str) -> str:
    score = case.scores.get(score_name)
    reason = getattr(score, "reason", None) if score is not None else None
    return "" if reason is None else str(reason)


def write_metadata_assignment_details(
    report: EvaluationReport[MetadataAssignmentInput, MetadataAssignmentActualOutput, MetadataAssignmentExpectedOutput],
    output_path: Path,
) -> None:
    """Write per-case gate and quality details for bounded assignment eval review."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "fixture",
        "scenario",
        "gates_passed",
        "metadata_quality",
        "entity_type_score",
        "body_regions_score",
        "subspecialties_score",
        "applicable_modalities_score",
        "etiologies_score",
        "expected_time_course_score",
        "index_codes_score",
        "anatomic_locations_score",
        "age_profile_score",
        "sex_specificity_score",
        "entity_type_detail",
        "body_regions_detail",
        "subspecialties_detail",
        "applicable_modalities_detail",
        "etiologies_detail",
        "expected_time_course_detail",
        "index_codes_detail",
        "anatomic_locations_detail",
        "age_profile_detail",
        "sex_specificity_detail",
        "error",
        "warnings",
    ]
    rows = []
    for case in report.cases:
        output = case.output
        rows.append({
            "case": case.name or "",
            "fixture": case.inputs.fixture_stem,
            "scenario": case.inputs.scenario,
            "gates_passed": not any(assertion.value is False for assertion in case.assertions.values()),
            "metadata_quality": f"{metadata_quality_case_score(case):.4f}",
            "entity_type_score": _score_csv_value(case, "quality.entity_type"),
            "body_regions_score": _score_csv_value(case, "quality.body_regions"),
            "subspecialties_score": _score_csv_value(case, "quality.subspecialties"),
            "applicable_modalities_score": _score_csv_value(case, "quality.applicable_modalities"),
            "etiologies_score": _score_csv_value(case, "quality.etiologies"),
            "expected_time_course_score": _score_csv_value(case, "quality.expected_time_course"),
            "index_codes_score": _score_csv_value(case, "quality.index_codes"),
            "anatomic_locations_score": _score_csv_value(case, "quality.anatomic_locations"),
            "age_profile_score": _score_csv_value(case, "quality.age_profile"),
            "sex_specificity_score": _score_csv_value(case, "quality.sex_specificity"),
            "entity_type_detail": _quality_score_reason(case, "quality.entity_type"),
            "body_regions_detail": _quality_score_reason(case, "quality.body_regions"),
            "subspecialties_detail": _quality_score_reason(case, "quality.subspecialties"),
            "applicable_modalities_detail": _quality_score_reason(case, "quality.applicable_modalities"),
            "etiologies_detail": _quality_score_reason(case, "quality.etiologies"),
            "expected_time_course_detail": _quality_score_reason(case, "quality.expected_time_course"),
            "index_codes_detail": _quality_score_reason(case, "quality.index_codes"),
            "anatomic_locations_detail": _quality_score_reason(case, "quality.anatomic_locations"),
            "age_profile_detail": _quality_score_reason(case, "quality.age_profile"),
            "sex_specificity_detail": _quality_score_reason(case, "quality.sex_specificity"),
            "error": output.error if output is not None else "",
            "warnings": _json_or_null(output.warnings if output is not None else []),
        })
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(tmp.name)
    temp_path.replace(output_path)


def _score_csv_value(case: Any, score_name: str) -> str:
    score = case.scores.get(score_name)
    return "" if score is None else f"{score.value:.4f}"


if __name__ == "__main__":
    import argparse
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()

    async def main() -> None:
        parser = argparse.ArgumentParser(description="Run metadata assignment evals.")
        parser.add_argument("--fixture", help="Run only one gold fixture stem.")
        parser.add_argument("--fixture-sample", type=int, help="Run a seeded sample of fixture stems.")
        parser.add_argument("--seed", type=int, default=0, help="Seed for --fixture-sample.")
        parser.add_argument("--scenario", choices=sorted(SCENARIO_PREPARERS), help="Run only one scenario.")
        parser.add_argument("--case", dest="case_name", help="Run one exact eval case name.")
        parser.add_argument("--limit", type=int, help="Run only the first N filtered cases.")
        parser.add_argument("--details-output", type=Path, help="Write per-case gate and quality details to CSV.")
        args = parser.parse_args()

        print("\nRunning metadata assignment evaluation suite...")
        print("=" * 80)

        report = await run_filtered_metadata_assignment_evals(
            fixture=args.fixture,
            fixture_sample=args.fixture_sample,
            seed=args.seed,
            scenario=args.scenario,
            case_name=args.case_name,
            limit=args.limit,
        )

        print("\n" + "=" * 80)
        print("METADATA ASSIGNMENT EVALUATION RESULTS")
        print("=" * 80 + "\n")

        report.print(
            include_input=False,
            include_output=False,
            include_durations=True,
            width=120,
        )

        print_metadata_assignment_summary(report)
        if args.details_output is not None:
            write_metadata_assignment_details(report, args.details_output)
            print(f"\nWrote details: {args.details_output}")
        if metadata_gate_failures(report):
            raise SystemExit(1)

    asyncio.run(main())
