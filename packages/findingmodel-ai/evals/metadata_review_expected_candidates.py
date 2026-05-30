"""Extract conservative expected-metadata candidates from latest human feedback."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_SUMMARY = Path(__file__).with_name("fixtures") / "metadata_review_feedback_summary.json"
DEFAULT_OUTPUT = Path(__file__).with_name("fixtures") / "metadata_review_expected_candidates.json"
DEFAULT_MD_OUTPUT = Path(__file__).parents[3] / "docs" / "reviews" / "metadata-review-expected-candidates-2026-05-24.md"


def explicit_duration(lowered: str) -> str | None:
    """Return explicit single duration from lowered review text."""

    for duration in ("permanent", "years", "months", "weeks", "days", "hours"):
        if duration in lowered:
            return duration
    return None


def extract_time_course(comment: str) -> dict[str, Any] | None:
    """Extract a conservative time-course hint from reviewer text."""

    lowered = comment.lower()
    range_patterns = (
        (("weeks/months", "weeks to months"), ["weeks", "months"]),
        (("months/years", "months years", "months to years"), ["months", "years"]),
    )
    for markers, candidates in range_patterns:
        if any(marker in lowered for marker in markers):
            return {
                "duration_candidates": candidates,
                "source_span": next(marker for marker in markers if marker in lowered),
            }

    duration = explicit_duration(lowered)

    if duration is None:
        return None

    modifiers = []
    if "progressive" in lowered:
        modifiers.append("progressive")
    if "resolving" in lowered or "resolve" in lowered:
        modifiers.append("resolving")
    return {"duration": duration, "modifiers": modifiers, "source_span": duration}


def extract_age_profile(comment: str) -> dict[str, Any] | None:
    """Extract an age-profile hint when the reviewer used explicit age language."""

    lowered = comment.lower()
    if "not in adolescence" in lowered:
        return {"forbidden": ["adolescent"]}
    if any(marker in lowered for marker in ("all ages", "all-ages", "any age", "all agents", "age profile is all")):
        age_result: dict[str, Any] = {"applicability": "all_ages"}
        if "more common" in lowered and ("elderly" in lowered or "aged" in lowered):
            age_result["more_common_in"] = ["aged"]
        return age_result
    if "not really more common" in lowered:
        return None

    stages = []
    for marker, stage in (
        ("adolescent", "adolescent"),
        ("adolescence", "adolescent"),
        ("adult", "adult"),
        ("middle-aged", "middle_aged"),
        ("elderly", "aged"),
        ("aged", "aged"),
        ("child", "child"),
        ("children", "child"),
        ("newborn", "infant"),
        ("infant", "infant"),
    ):
        if marker in lowered and stage not in stages:
            stages.append(stage)

    if stages:
        result: dict[str, Any] = {"applicability": stages}
        if "more common" in lowered and ("elderly" in lowered or "aged" in lowered):
            result["more_common_in"] = ["aged"]
        return result
    return None


def extract_sex_specificity(comment: str) -> str | None:
    """Extract sex-specificity hints."""

    lowered = comment.lower()
    if "sex-neutral" in lowered or "sex neutral" in lowered or "sex specificity is neutral" in lowered:
        return "sex-neutral"
    if "female" in lowered:
        return "female-specific"
    if re.search(r"\bmale\b", lowered):
        return "male-specific"
    return None


def extract_etiology_hints(comment: str) -> dict[str, list[str]]:
    """Extract etiology words as review hints, not final coded values."""

    lowered = comment.lower()
    expected = []
    forbidden = []

    for marker in ("not inflammatory", "not ischemic", "not congenital", "not be congenital", "not neoplastic:benign"):
        if marker in lowered:
            forbidden.append(marker.removeprefix("not be ").removeprefix("not "))

    positive_markers = (
        "vascular",
        "post-infectious",
        "post-exposure",
        "post-treatment",
        "heart failure",
        "cardiovascular",
        "degenerative",
    )
    for marker in positive_markers:
        if marker in lowered:
            expected.append(marker)
    if ("inflammatory" in lowered or "inflammation" in lowered) and "not inflammatory" not in lowered:
        expected.append("inflammatory")
    if "congenital" in lowered and not any(marker in lowered for marker in ("not congenital", "not be congenital")):
        expected.append("congenital")

    result: dict[str, list[str]] = {}
    if expected:
        result["etiology_hints"] = expected
    if forbidden:
        result["forbidden_etiology_hints"] = forbidden
    return result


def extract_expected_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Extract conservative expected metadata from one feedback summary record."""

    comment = record["comment"]
    fields = set(record["affected_fields"])
    expected: dict[str, Any] = {}

    if "expected_time_course" in fields:
        time_course = extract_time_course(comment)
        if time_course is not None:
            expected["expected_time_course"] = time_course
    if "age_profile" in fields:
        age_profile = extract_age_profile(comment)
        if age_profile is not None:
            expected["age_profile"] = age_profile
    if "sex_specificity" in fields:
        sex_specificity = extract_sex_specificity(comment)
        if sex_specificity is not None:
            expected["sex_specificity"] = sex_specificity
    if "etiologies" in fields:
        etiology_hints = extract_etiology_hints(comment)
        if etiology_hints:
            expected.update(etiology_hints)

    return expected


def build_candidates(summary_path: Path) -> dict[str, Any]:
    """Build expected-metadata candidate records."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = []
    for record in summary["records"]:
        expected = extract_expected_metadata(record)
        if not expected:
            continue
        records.append({
            "source_record_id": record["source_record_id"],
            "path": record["path"],
            "title": record["title"],
            "affected_fields": record["affected_fields"],
            "extracted_expected_metadata": expected,
            "source_comment": record["comment"],
            "extraction_confidence": "low",
            "promotion_status": "candidate",
            "requires_human_promotion": True,
        })

    field_counts = Counter(field for record in records for field in record["extracted_expected_metadata"])
    return {
        "version": 1,
        "source_summary": str(summary_path),
        "counts": {
            "candidate_records": len(records),
            "extracted_fields": dict(sorted(field_counts.items())),
        },
        "records": records,
    }


def markdown_table(candidates: dict[str, Any]) -> str:
    """Render expected-metadata candidates for human promotion review."""

    lines = [
        "# Metadata Review Expected Candidates",
        "",
        "Status: Generated promotion review aid",
        "Date: 2026-05-24",
        "",
        "These are conservative metadata hints extracted from latest human feedback.",
        "They are candidates, not gold, until explicitly promoted.",
        "",
        "## Counts",
        "",
        f"- Candidate records: {candidates['counts']['candidate_records']}",
        "",
        "Extracted fields:",
        "",
    ]
    for field, count in candidates["counts"]["extracted_fields"].items():
        lines.append(f"- {field}: {count}")

    lines.extend([
        "",
        "## Candidate Records",
        "",
        "| Finding | Extracted expected metadata | Human comment | Promotion status |",
        "| --- | --- | --- | --- |",
    ])
    for record in candidates["records"]:
        expected = json.dumps(record["extracted_expected_metadata"], sort_keys=True)
        comment = record["source_comment"].replace("\n", "<br>").replace("|", "\\|")
        lines.append(f"| `{record['source_record_id']}` | `{expected}` | {comment} | {record['promotion_status']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    candidates = build_candidates(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text(markdown_table(candidates), encoding="utf-8")
    print(f"wrote {candidates['counts']['candidate_records']} candidate records to {args.output} and {args.md_output}")


if __name__ == "__main__":
    main()
