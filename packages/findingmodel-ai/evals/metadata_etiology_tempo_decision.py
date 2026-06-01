"""Component evals for the metadata etiology-tempo decision sub-agent."""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from findingmodel import EntityType, EtiologyCode, ExpectedDuration, ExpectedTimeCourse, TimeCourseModifier
from findingmodel_ai.metadata.assignment import create_etiology_tempo_agent
from findingmodel_ai.metadata.decisions import EtiologyTempoDecision
from pydantic import BaseModel, Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from evals.metadata_scoring import (
    DURATION_ORDER,
    clamp_score,
    score_commission_sensitive_set_similarity,
    score_expected_time_course,
)

EVAL_MAX_CONCURRENCY = 2
EVAL_GOLD_DIR = Path(__file__).with_name("gold")
EVAL_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "etiology_tempo_reviewed_cases.json"
CASE_SET_CHOICES = ("pilot", "gold", "reviewed", "expanded", "all")
COMBINED_SCORE_NAME = "combined"


class EtiologyTempoDecisionInput(BaseModel):
    """Input payload for one etiology-tempo replay case."""

    payload: dict[str, Any]


class EtiologyTempoDecisionExpectedOutput(BaseModel):
    """Expected etiology-tempo output for one replay case."""

    etiologies: set[EtiologyCode] = Field(default_factory=set)
    expected_time_course: ExpectedTimeCourse | None = None


class EtiologyTempoDecisionActualOutput(BaseModel):
    """Observed etiology-tempo output for one replay case."""

    etiologies: set[EtiologyCode] = Field(default_factory=set)
    expected_time_course: ExpectedTimeCourse | None = None
    raw_output: EtiologyTempoDecision | None = None
    error: str | None = None


def _ontology_candidate(candidate_id: str, text: str, *, selected: bool = True) -> dict[str, Any]:
    system, _, _code = candidate_id.partition(":")
    return {
        "candidate_id": candidate_id,
        "text": text,
        "display": text,
        "table_name": system.lower(),
        "system": system,
        "source_bucket": "existing_index_codes",
        "default_relationship": "exact_match" if selected else "related",
        "default_selected_as_canonical": selected,
    }


def _payload(
    *,
    name: str,
    description: str,
    entity_type: EntityType,
    ontology: list[dict[str, Any]],
    synonyms: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "assignment_mode": "reassess",
        "finding_model": {
            "oifm_id": f"EVAL_{name.upper().replace(' ', '_')}",
            "name": name,
            "description": description,
            "synonyms": synonyms or [],
            "tags": tags or [],
            "existing_structured_metadata": {
                "body_regions": None,
                "subspecialties": None,
                "etiologies": None,
                "entity_type": entity_type.value,
                "applicable_modalities": None,
                "expected_time_course": None,
                "age_profile": None,
                "sex_specificity": None,
                "index_codes": [],
                "anatomic_locations": [],
            },
            "attributes": [],
        },
        "ontology_candidates": [candidate for candidate in ontology if candidate["default_selected_as_canonical"]],
        "anatomic_candidates": [],
        "task": "Assign only etiologies and expected_time_course.",
    }


def _tc(duration: ExpectedDuration, *modifiers: TimeCourseModifier) -> ExpectedTimeCourse:
    return ExpectedTimeCourse(duration=duration, modifiers=list(modifiers))


def _case(
    name: str,
    *,
    payload: dict[str, Any],
    etiologies: set[EtiologyCode] | None,
    expected_time_course: ExpectedTimeCourse | None,
) -> Case[EtiologyTempoDecisionInput, EtiologyTempoDecisionExpectedOutput]:
    return Case(
        name=name,
        inputs=EtiologyTempoDecisionInput(payload=payload),
        expected_output=EtiologyTempoDecisionExpectedOutput(
            etiologies=etiologies or set(),
            expected_time_course=expected_time_course,
        ),
    )


def _gold_case(fixture_stem: str) -> Case[EtiologyTempoDecisionInput, EtiologyTempoDecisionExpectedOutput]:
    data = json.loads((EVAL_GOLD_DIR / f"{fixture_stem}.fm.json").read_text(encoding="utf-8"))
    ontology = [
        _ontology_candidate(f"{code['system']}:{code['code']}", code["display"])
        for code in data.get("index_codes") or []
    ]
    time_course = (
        ExpectedTimeCourse.model_validate(data["expected_time_course"])
        if data.get("expected_time_course") is not None
        else None
    )
    return _case(
        f"gold_{fixture_stem}",
        payload=_payload(
            name=data["name"],
            description=data.get("description") or "",
            entity_type=EntityType(data["entity_type"]),
            ontology=ontology,
            synonyms=data.get("synonyms") or [],
            tags=data.get("tags") or [],
        ),
        etiologies={EtiologyCode(value) for value in data.get("etiologies") or []},
        expected_time_course=time_course,
    )


def _fixture_case(entry: dict[str, Any]) -> Case[EtiologyTempoDecisionInput, EtiologyTempoDecisionExpectedOutput]:
    """Build one etiology/time-course eval case from the reviewed JSON fixture."""

    ontology = [
        _ontology_candidate(f"{code['system']}:{code['code']}", code["display"])
        for code in entry.get("index_codes") or []
    ]
    expected = entry["expected"]
    time_course = (
        ExpectedTimeCourse.model_validate(expected["expected_time_course"])
        if expected.get("expected_time_course") is not None
        else None
    )
    return _case(
        f"{entry['source']}_{entry['slug']}",
        payload=_payload(
            name=entry["name"],
            description=entry.get("description") or "",
            entity_type=EntityType(entry["entity_type"]),
            ontology=ontology,
            synonyms=entry.get("synonyms") or [],
            tags=entry.get("tags") or [],
        ),
        etiologies={EtiologyCode(value) for value in expected.get("etiologies") or []},
        expected_time_course=time_course,
    )


def _load_fixture_entries(*, source: str | None = None) -> list[dict[str, Any]]:
    data = json.loads(EVAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    entries = data["cases"]
    if source is None:
        return entries
    return [entry for entry in entries if entry["source"] == source]


def _fixture_cases(
    *, source: str | None = None
) -> list[Case[EtiologyTempoDecisionInput, EtiologyTempoDecisionExpectedOutput]]:
    return [_fixture_case(entry) for entry in _load_fixture_entries(source=source)]


PILOT_CASES: list[Case[EtiologyTempoDecisionInput, EtiologyTempoDecisionExpectedOutput]] = [
    _case(
        "generic_axillary_mass_stays_null",
        payload=_payload(
            name="axillary mass",
            description="Abnormal soft tissue mass located in the axillary region.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("GAMUTS:16069", "axillary mass")],
        ),
        etiologies=None,
        expected_time_course=None,
    ),
    _case(
        "pericardial_effusion_stays_null_without_qualifier",
        payload=_payload(
            name="pericardial effusion",
            description="Accumulation of fluid in the pericardial cavity.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:373945007", "Pericardial effusion")],
        ),
        etiologies=None,
        expected_time_course=None,
    ),
    _case(
        "cervical_lymphadenopathy_carries_common_causes",
        payload=_payload(
            name="cervical lymphadenopathy",
            description="Abnormal enlargement of cervical lymph nodes.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:127086001", "Cervical lymphadenopathy")],
        ),
        etiologies={EtiologyCode.INFLAMMATORY, EtiologyCode.NEOPLASTIC_MALIGNANT},
        expected_time_course=_tc(ExpectedDuration.WEEKS, TimeCourseModifier.EVOLVING),
    ),
    _case(
        "fdg_avid_pulmonary_nodule_carries_pet_context_causes",
        payload=_payload(
            name="FDG-avid pulmonary nodule",
            description="Pulmonary nodule with increased FDG uptake on PET, suspicious for malignancy or inflammation.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("OIDM:FDG_NODULE", "FDG-avid pulmonary nodule")],
        ),
        etiologies={EtiologyCode.INFLAMMATORY, EtiologyCode.NEOPLASTIC_MALIGNANT},
        expected_time_course=_tc(ExpectedDuration.MONTHS, TimeCourseModifier.PROGRESSIVE),
    ),
    _case(
        "pulmonary_embolism_is_thrombotic_and_resolving",
        payload=_payload(
            name="pulmonary embolism",
            description="Embolic occlusion of a pulmonary artery.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:59282003", "Pulmonary embolism")],
        ),
        etiologies={EtiologyCode.VASCULAR_THROMBOTIC},
        expected_time_course=_tc(ExpectedDuration.WEEKS, TimeCourseModifier.RESOLVING),
    ),
    _case(
        "aortic_dissection_is_vascular_and_durable",
        payload=_payload(
            name="aortic dissection",
            description="Dissection flap involving the aorta.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:308546005", "Aortic dissection")],
        ),
        etiologies={EtiologyCode.VASCULAR},
        expected_time_course=_tc(ExpectedDuration.PERMANENT, TimeCourseModifier.EVOLVING),
    ),
    _case(
        "abdominal_aortic_aneurysm_is_aneurysmal_degenerative",
        payload=_payload(
            name="abdominal aortic aneurysm",
            description="Permanent aneurysmal dilation of the abdominal aorta, often degenerative.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:233985008", "Abdominal aortic aneurysm")],
        ),
        etiologies={EtiologyCode.VASCULAR_ANEURYSMAL, EtiologyCode.DEGENERATIVE},
        expected_time_course=_tc(ExpectedDuration.PERMANENT, TimeCourseModifier.PROGRESSIVE),
    ),
    _case(
        "pneumonia_is_infectious_and_resolving",
        payload=_payload(
            name="pneumonia",
            description="Infectious inflammation of lung parenchyma.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:233604007", "Pneumonia")],
        ),
        etiologies={EtiologyCode.INFLAMMATORY_INFECTIOUS},
        expected_time_course=_tc(ExpectedDuration.WEEKS, TimeCourseModifier.RESOLVING),
    ),
    _case(
        "acute_appendicitis_is_inflammatory_progressive",
        payload=_payload(
            name="acute appendicitis",
            description="Acute inflammation of the appendix.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:85189001", "Acute appendicitis")],
        ),
        etiologies={EtiologyCode.INFLAMMATORY},
        expected_time_course=_tc(ExpectedDuration.DAYS, TimeCourseModifier.PROGRESSIVE),
    ),
    _case(
        "kidney_stone_is_metabolic_stable_months",
        payload=_payload(
            name="kidney stone",
            description="Calculus in the kidney or urinary tract.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:95570007", "Kidney stone")],
        ),
        etiologies={EtiologyCode.METABOLIC},
        expected_time_course=_tc(ExpectedDuration.MONTHS, TimeCourseModifier.STABLE),
    ),
    _case(
        "distal_radius_fracture_is_acute_traumatic",
        payload=_payload(
            name="distal radius fracture",
            description="Acute fracture of the distal radius.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:263199001", "Fracture of distal end of radius")],
        ),
        etiologies={EtiologyCode.TRAUMATIC_ACUTE},
        expected_time_course=_tc(ExpectedDuration.WEEKS, TimeCourseModifier.RESOLVING),
    ),
    _case(
        "vertebral_compression_fracture_can_persist",
        payload=_payload(
            name="vertebral compression fracture",
            description="Compression deformity of a vertebral body from trauma or osteoporotic degeneration.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:42942008", "Compression fracture of vertebral column")],
        ),
        etiologies={EtiologyCode.TRAUMATIC_ACUTE, EtiologyCode.DEGENERATIVE},
        expected_time_course=_tc(ExpectedDuration.PERMANENT, TimeCourseModifier.STABLE),
    ),
    _case(
        "hip_osteoarthritis_is_degenerative_progressive",
        payload=_payload(
            name="hip osteoarthritis",
            description="Degenerative osteoarthritis of the hip joint.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:239872002", "Osteoarthritis of hip")],
        ),
        etiologies={EtiologyCode.DEGENERATIVE},
        expected_time_course=_tc(ExpectedDuration.YEARS, TimeCourseModifier.PROGRESSIVE),
    ),
    _case(
        "lumbar_disc_herniation_is_degenerative_mechanical",
        payload=_payload(
            name="lumbar disc herniation",
            description="Herniation of lumbar intervertebral disc causing mechanical displacement.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:202735001", "Lumbar disc prolapse")],
        ),
        etiologies={EtiologyCode.DEGENERATIVE, EtiologyCode.MECHANICAL},
        expected_time_course=_tc(ExpectedDuration.MONTHS, TimeCourseModifier.EVOLVING),
    ),
    _case(
        "primary_brain_tumor_is_benign_and_malignant",
        payload=_payload(
            name="primary brain tumor",
            description="Tumor originating within the brain.",
            entity_type=EntityType.DIAGNOSIS,
            ontology=[_ontology_candidate("SNOMEDCT:428061005", "Primary neoplasm of brain")],
        ),
        etiologies={EtiologyCode.NEOPLASTIC_BENIGN, EtiologyCode.NEOPLASTIC_MALIGNANT},
        expected_time_course=_tc(ExpectedDuration.YEARS, TimeCourseModifier.PROGRESSIVE),
    ),
    _case(
        "thyroid_nodule_is_benign_or_potential",
        payload=_payload(
            name="thyroid nodule",
            description="Discrete nodule in the thyroid gland, often benign but assessed for malignant potential.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:237495005", "Thyroid nodule")],
        ),
        etiologies={EtiologyCode.NEOPLASTIC_BENIGN, EtiologyCode.NEOPLASTIC_POTENTIAL},
        expected_time_course=_tc(ExpectedDuration.YEARS, TimeCourseModifier.STABLE),
    ),
    _case(
        "aortic_stent_is_iatrogenic_device_and_persistent",
        payload=_payload(
            name="aortic stent",
            description="Endovascular stent graft within the aorta.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:70512008", "Aortic stent")],
        ),
        etiologies={EtiologyCode.IATROGENIC_DEVICE},
        expected_time_course=_tc(ExpectedDuration.PERMANENT, TimeCourseModifier.STABLE),
    ),
    _case(
        "motion_artifact_has_no_etiology_or_tempo",
        payload=_payload(
            name="motion artifact",
            description="Image degradation from patient motion during acquisition.",
            entity_type=EntityType.TECHNIQUE_ISSUE,
            ontology=[_ontology_candidate("RADLEX:RID10311", "Motion artifact")],
        ),
        etiologies=None,
        expected_time_course=None,
    ),
    _case(
        "birads_assessment_has_no_etiology_or_tempo",
        payload=_payload(
            name="BI-RADS assessment",
            description="Structured breast imaging reporting and assessment category.",
            entity_type=EntityType.ASSESSMENT,
            ontology=[_ontology_candidate("RADLEX:RID10370", "BI-RADS assessment")],
        ),
        etiologies=None,
        expected_time_course=None,
    ),
    _case(
        "breast_density_has_tempo_but_no_etiology",
        payload=_payload(
            name="breast density",
            description="Relative amount of fibroglandular tissue in the breast.",
            entity_type=EntityType.FINDING,
            ontology=[_ontology_candidate("SNOMEDCT:129718006", "Breast density")],
        ),
        etiologies=None,
        expected_time_course=_tc(ExpectedDuration.YEARS, TimeCourseModifier.EVOLVING),
    ),
]


def score_etiology_tempo_output(
    expected: EtiologyTempoDecisionExpectedOutput,
    output: EtiologyTempoDecisionActualOutput,
) -> bool:
    if output.error is not None:
        return False
    return output.etiologies == expected.etiologies and output.expected_time_course == expected.expected_time_course


def etiology_score(
    expected: EtiologyTempoDecisionExpectedOutput,
    output: EtiologyTempoDecisionActualOutput,
) -> float:
    """Score etiology agreement for one output."""

    if output.error is not None:
        return 0.0
    return score_commission_sensitive_set_similarity(output.etiologies, expected.etiologies)


def time_course_score(
    expected: EtiologyTempoDecisionExpectedOutput,
    output: EtiologyTempoDecisionActualOutput,
) -> float:
    """Score combined expected time-course agreement for one output."""

    if output.error is not None:
        return 0.0
    return score_expected_time_course(output.expected_time_course, expected.expected_time_course)


def duration_score(actual: ExpectedTimeCourse | None, expected: ExpectedTimeCourse | None) -> float:
    """Score expected duration independently from modifiers."""

    if actual is None and expected is None:
        return 1.0
    if actual is None or expected is None:
        return 0.0
    if actual.duration == expected.duration:
        return 1.0
    if actual.duration is None or expected.duration is None:
        return 0.0
    distance = abs(DURATION_ORDER.index(actual.duration) - DURATION_ORDER.index(expected.duration))
    return clamp_score(1.0 - (distance * 0.25))


def modifier_score(actual: ExpectedTimeCourse | None, expected: ExpectedTimeCourse | None) -> float:
    """Score expected time-course modifiers independently from duration."""

    if actual is None and expected is None:
        return 1.0
    if actual is None or expected is None:
        return 0.0
    return score_commission_sensitive_set_similarity(
        actual.modifiers or [],
        expected.modifiers or [],
        missing_expected_credit=0.25,
    )


def combined_score(
    expected: EtiologyTempoDecisionExpectedOutput,
    output: EtiologyTempoDecisionActualOutput,
) -> float:
    """Return the weighted etiology/time-course score for one output."""

    if output.error is not None:
        return 0.0
    return clamp_score((0.60 * etiology_score(expected, output)) + (0.40 * time_course_score(expected, output)))


class EtiologyTempoEvaluator(
    Evaluator[EtiologyTempoDecisionInput, EtiologyTempoDecisionActualOutput, EtiologyTempoDecisionExpectedOutput]
):
    """Score etiology and time-course agreement with weighted partial credit."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            EtiologyTempoDecisionInput, EtiologyTempoDecisionActualOutput, EtiologyTempoDecisionExpectedOutput
        ],
    ) -> dict[str, float]:
        if ctx.expected_output is None:
            return {
                "etiologies": 0.0,
                "expected_time_course": 0.0,
                "duration": 0.0,
                "modifiers": 0.0,
                COMBINED_SCORE_NAME: 0.0,
            }
        return {
            "etiologies": etiology_score(ctx.expected_output, ctx.output),
            "expected_time_course": time_course_score(ctx.expected_output, ctx.output),
            "duration": duration_score(ctx.output.expected_time_course, ctx.expected_output.expected_time_course),
            "modifiers": modifier_score(ctx.output.expected_time_course, ctx.expected_output.expected_time_course),
            COMBINED_SCORE_NAME: combined_score(ctx.expected_output, ctx.output),
        }


async def run_etiology_tempo_decision_task(
    case_input: EtiologyTempoDecisionInput,
) -> EtiologyTempoDecisionActualOutput:
    """Run only the etiology-tempo decision agent for one replay payload."""

    agent = create_etiology_tempo_agent()
    payload = copy.deepcopy(case_input.payload)
    try:
        result = await agent.run(json.dumps(payload, indent=2))
    except Exception as exc:  # pragma: no cover - useful in eval reports
        return EtiologyTempoDecisionActualOutput(error=f"{type(exc).__name__}: {exc}")

    return EtiologyTempoDecisionActualOutput(
        etiologies=set(result.output.etiologies or []),
        expected_time_course=result.output.expected_time_course,
        raw_output=result.output,
    )


def select_etiology_tempo_cases(
    case_set: str,
) -> list[Case[EtiologyTempoDecisionInput, EtiologyTempoDecisionExpectedOutput]]:
    """Return etiology/time-course component cases for the requested corpus."""

    if case_set == "pilot":
        return PILOT_CASES
    if case_set == "gold":
        return _fixture_cases(source="gold")
    if case_set == "reviewed":
        return _fixture_cases(source="reviewed_clean_input")
    if case_set == "expanded":
        return _fixture_cases()
    if case_set == "all":
        return [*PILOT_CASES, *_fixture_cases()]
    raise ValueError(f"Unknown etiology tempo case set: {case_set}")


def build_etiology_tempo_dataset(
    *, case_set: str = "expanded"
) -> Dataset[
    EtiologyTempoDecisionInput,
    EtiologyTempoDecisionActualOutput,
    EtiologyTempoDecisionExpectedOutput,
]:
    """Build the etiology/time-course component dataset."""

    return Dataset(cases=select_etiology_tempo_cases(case_set), evaluators=[EtiologyTempoEvaluator()])


async def run_metadata_etiology_tempo_decision_evals(
    *,
    case_set: str = "expanded",
) -> EvaluationReport[
    EtiologyTempoDecisionInput,
    EtiologyTempoDecisionActualOutput,
    EtiologyTempoDecisionExpectedOutput,
]:
    """Run the etiology/time-course component replay suite."""

    return await build_etiology_tempo_dataset(case_set=case_set).evaluate(
        run_etiology_tempo_decision_task,
        max_concurrency=EVAL_MAX_CONCURRENCY,
    )


async def sample_etiology_tempo_cases(case_names: list[str], *, case_set: str, repeats: int) -> None:
    """Run selected cases repeatedly and print pass rates plus output variants."""

    case_by_name = {case.name: case for case in select_etiology_tempo_cases(case_set)}
    missing_names = [name for name in case_names if name not in case_by_name]
    if missing_names:
        raise SystemExit(f"Unknown case name(s): {', '.join(missing_names)}")

    for case_name in case_names:
        case = case_by_name[case_name]
        variants: Counter[tuple[tuple[str, ...], str]] = Counter()
        errors: Counter[str] = Counter()
        pass_count = 0
        for _ in range(repeats):
            output = await run_etiology_tempo_decision_task(case.inputs)
            if output.error is not None:
                errors[output.error] += 1
            else:
                etiology_values = tuple(sorted(value.value for value in output.etiologies))
                time_course = (
                    json.dumps(output.expected_time_course.model_dump(mode="json"), sort_keys=True)
                    if output.expected_time_course is not None
                    else "null"
                )
                variants[etiology_values, time_course] += 1
            if score_etiology_tempo_output(case.expected_output, output):
                pass_count += 1

        print(f"\n{case_name}: {pass_count}/{repeats} passed")
        if variants:
            print("  variants:")
            for (etiologies, time_course), count in variants.most_common():
                etiology_label = ", ".join(etiologies) if etiologies else "<empty>"
                print(f"  - {count}x etiologies=[{etiology_label}] expected_time_course={time_course}")
        if errors:
            print("  errors:")
            for error, count in errors.most_common():
                print(f"  - {count}x {error}")


def _score_value(case: Any, score_name: str) -> float | None:
    score = case.scores.get(score_name)
    if score is None:
        score = case.scores.get(f"EtiologyTempoEvaluator.{score_name}")
    return score.value if score is not None else None


def print_etiology_tempo_component_summary(report: EvaluationReport[Any, Any, Any]) -> None:
    """Print field-level etiology/time-course component summary."""

    if not report.cases:
        print("Etiology/time-course component eval: no cases")
        return

    score_names = ("etiologies", "expected_time_course", "duration", "modifiers", COMBINED_SCORE_NAME)
    print("\nEtiology/time-course component scores:")
    for score_name in score_names:
        values = [_score_value(case, score_name) for case in report.cases]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            print(f"{score_name}: {sum(numeric_values) / len(numeric_values):.2f}")

    scored_cases = [
        (case, _score_value(case, COMBINED_SCORE_NAME))
        for case in report.cases
        if _score_value(case, COMBINED_SCORE_NAME) is not None
    ]
    print("Lowest scoring cases:")
    for case, score in sorted(scored_cases, key=lambda item: item[1] or 0.0)[:10]:
        weak_fields = []
        for field_name in ("etiologies", "expected_time_course", "duration", "modifiers"):
            field_score = _score_value(case, field_name)
            if field_score is not None and field_score < 1.0:
                weak_fields.append(f"{field_name}={field_score:.2f}")
        suffix = f" ({', '.join(weak_fields)})" if weak_fields else ""
        print(f"- {case.name or '<unnamed>'}: {score:.2f}{suffix}")


def _json_or_null(value: Any) -> str:
    if value is None:
        return "null"
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True)


def _etiology_values(values: Iterable[EtiologyCode] | None) -> list[str]:
    return sorted(value.value for value in values or [])


def miss_labels(
    expected: EtiologyTempoDecisionExpectedOutput,
    output: EtiologyTempoDecisionActualOutput,
) -> list[str]:
    """Return deterministic labels describing etiology/time-course mismatches."""

    if output.error is not None:
        return ["execution_error"]

    labels: list[str] = []
    expected_etiologies = set(expected.etiologies or [])
    actual_etiologies = set(output.etiologies or [])
    missing_etiologies = expected_etiologies - actual_etiologies
    extra_etiologies = actual_etiologies - expected_etiologies
    if missing_etiologies:
        labels.append("missing_expected_etiology")
    if extra_etiologies:
        labels.append("extra_unsupported_etiology")
    if missing_etiologies and extra_etiologies:
        labels.append("wrong_etiology_family_or_subtype")

    expected_tc = expected.expected_time_course
    actual_tc = output.expected_time_course
    if expected_tc is None and actual_tc is not None:
        labels.append("extra_time_course")
    elif expected_tc is not None and actual_tc is None:
        labels.append("missing_time_course")
    elif expected_tc is not None and actual_tc is not None:
        if actual_tc.duration != expected_tc.duration:
            labels.append("wrong_duration")
        if set(actual_tc.modifiers or []) != set(expected_tc.modifiers or []):
            labels.append("wrong_modifier")

    return labels


def write_etiology_tempo_details(report: EvaluationReport[Any, Any, Any], output_path: Path) -> None:
    """Write per-case expected/actual values and miss labels as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in report.cases:
        expected = case.expected_output
        output = case.output
        if expected is None or output is None:
            continue
        rows.append({
            "case": case.name or "",
            "combined": _score_value(case, COMBINED_SCORE_NAME),
            "etiologies_score": _score_value(case, "etiologies"),
            "expected_time_course_score": _score_value(case, "expected_time_course"),
            "duration_score": _score_value(case, "duration"),
            "modifiers_score": _score_value(case, "modifiers"),
            "expected_etiologies": ";".join(_etiology_values(expected.etiologies)),
            "actual_etiologies": ";".join(_etiology_values(output.etiologies)),
            "expected_time_course": _json_or_null(expected.expected_time_course),
            "actual_time_course": _json_or_null(output.expected_time_course),
            "miss_labels": ";".join(miss_labels(expected, output)),
            "error": output.error or "",
        })

    fieldnames = [
        "case",
        "combined",
        "etiologies_score",
        "expected_time_course_score",
        "duration_score",
        "modifiers_score",
        "expected_etiologies",
        "actual_etiologies",
        "expected_time_course",
        "actual_time_course",
        "miss_labels",
        "error",
    ]
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(tmp.name)
    temp_path.replace(output_path)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--case-set",
            choices=CASE_SET_CHOICES,
            default="expanded",
            help=(
                "Case corpus: pilot hand cases, gold fixtures, reviewed clean-input cases, "
                "expanded gold+reviewed fixture, or all including pilot cases."
            ),
        )
        parser.add_argument(
            "--sample-case",
            action="append",
            default=[],
            help="Run one named case repeatedly and summarize output variants. May be passed more than once.",
        )
        parser.add_argument(
            "--repeats",
            type=int,
            default=5,
            help="Number of repetitions for each --sample-case.",
        )
        parser.add_argument(
            "--details-output",
            type=Path,
            help="Write per-case expected/actual values and miss labels to a CSV file.",
        )
        args = parser.parse_args()
        if args.sample_case:
            asyncio.run(sample_etiology_tempo_cases(args.sample_case, case_set=args.case_set, repeats=args.repeats))
        else:
            print(f"\nRunning etiology/time-course component evaluation suite ({args.case_set})...")
            print("=" * 80)
            report = asyncio.run(
                run_metadata_etiology_tempo_decision_evals(
                    case_set=args.case_set,
                )
            )
            report.print(include_input=False, include_expected_output=False, include_durations=True)
            print_etiology_tempo_component_summary(report)
            if args.details_output is not None:
                write_etiology_tempo_details(report, args.details_output)
                print(f"\nWrote details: {args.details_output}")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
