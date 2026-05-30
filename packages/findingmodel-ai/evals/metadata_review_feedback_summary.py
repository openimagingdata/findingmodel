"""Summarize latest human feedback records from the review evidence register."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_REGISTER = Path(__file__).with_name("fixtures") / "metadata_review_evidence_register.json"
DEFAULT_JSON_OUTPUT = Path(__file__).with_name("fixtures") / "metadata_review_feedback_summary.json"
DEFAULT_MD_OUTPUT = Path(__file__).parents[3] / "docs" / "reviews" / "metadata-review-feedback-summary-2026-05-24.md"

FIELD_LABEL_RULES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"expected_time_course"}), "time_course"),
    (frozenset({"anatomic_locations"}), "anatomic_location"),
    (frozenset({"age_profile", "sex_specificity"}), "age_or_sex_applicability"),
    (frozenset({"index_codes"}), "index_code"),
    (frozenset({"etiologies"}), "etiology"),
    (frozenset({"applicable_modalities"}), "modality"),
    (frozenset({"subspecialties", "body_regions"}), "domain_or_region"),
    (frozenset({"entity_type"}), "entity_type"),
)

COMMENT_LABEL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("wrong", "way off", "inappropriate", "not the same", "should not"), "incorrect_assignment"),
    (("why no", "where are", "don't we have", "aren't there", "should have"), "missing_assignment"),
    (("can't localize", "too general", "right atrium", "fibula", "tibia"), "over_specific_or_bad_code"),
    (("description", "skip this one", "not even an imaging finding"), "source_model_issue"),
    (("radelement", "snomed", "radlex", "code"), "code_mapping_issue"),
)


def latest_records(register: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the latest review event for each source path."""

    by_path: dict[str, dict[str, Any]] = {}
    for record in register["records"]:
        path = record["path"]
        current = by_path.get(path)
        if current is None or (record["human_review"].get("updated_at") or "") >= (
            current["human_review"].get("updated_at") or ""
        ):
            by_path[path] = record
    return list(by_path.values())


def classify_feedback(record: dict[str, Any]) -> list[str]:
    """Assign coarse action labels to one feedback record."""

    comment = record["human_review"]["comment"].lower()
    fields = set(record["affected_fields"])
    labels: set[str] = set()

    for field_names, label in FIELD_LABEL_RULES:
        if fields & field_names:
            labels.add(label)

    for markers, label in COMMENT_LABEL_RULES:
        if any(marker in comment for marker in markers):
            labels.add(label)

    return sorted(labels or {"manual_review"})


def disposition_queue(labels: list[str]) -> str:
    """Assign the next disposition queue for one feedback record."""

    label_set = set(labels)
    if "source_model_issue" in label_set:
        return "source-model-issue"
    if label_set & {"code_mapping_issue", "over_specific_or_bad_code"}:
        return "code/anatomy-review"
    if label_set & {
        "time_course",
        "etiology",
        "age_or_sex_applicability",
        "modality",
        "domain_or_region",
        "entity_type",
    }:
        return "expected-metadata-extraction"
    if label_set & {"anatomic_location", "index_code"}:
        return "expected-code-or-location-extraction"
    return "manual-disposition"


def next_action(queue: str) -> str:
    """Return the next review action for one disposition queue."""

    if queue == "source-model-issue":
        return "Open a source-model cleanup item; do not convert this feedback into prompt tuning."
    if queue == "code/anatomy-review":
        return "Review code or anatomic-location expectation before adding eval coverage."
    if queue == "expected-code-or-location-extraction":
        return "Extract expected code/location candidates and require human promotion."
    if queue == "expected-metadata-extraction":
        return "Extract expected metadata candidates and require human promotion."
    return "Manual disposition required."


def build_summary(register_path: Path) -> dict[str, Any]:
    """Build feedback summary from the register."""

    register = json.loads(register_path.read_text(encoding="utf-8"))
    feedback = [record for record in latest_records(register) if record["human_review"]["status"] == "feedback"]
    records = []
    for record in feedback:
        labels = classify_feedback(record)
        records.append({
            "id": record["id"],
            "source_record_id": record["source_record_id"],
            "path": record["path"],
            "title": record["title"],
            "affected_fields": record["affected_fields"],
            "labels": labels,
            "disposition_queue": disposition_queue(labels),
            "triage_status": "unresolved",
            "owner_or_queue": disposition_queue(labels),
            "next_action": next_action(disposition_queue(labels)),
            "eval_candidate": disposition_queue(labels)
            in {"expected-metadata-extraction", "expected-code-or-location-extraction"},
            "source_model_issue": disposition_queue(labels) == "source-model-issue",
            "comment": record["human_review"]["comment"],
        })

    label_counts = Counter(label for record in records for label in record["labels"])
    disposition_counts = Counter(record["disposition_queue"] for record in records)
    field_counts = Counter(field for record in records for field in record["affected_fields"])
    return {
        "version": 1,
        "source_register": str(register_path),
        "counts": {
            "latest_feedback": len(records),
            "labels": dict(sorted(label_counts.items())),
            "dispositions": dict(sorted(disposition_counts.items())),
            "affected_fields": dict(sorted(field_counts.items())),
        },
        "records": records,
    }


def markdown_table(summary: dict[str, Any]) -> str:
    """Render a concise review table."""

    lines = [
        "# Metadata Review Feedback Summary",
        "",
        "Status: Generated review aid",
        "Date: 2026-05-24",
        "",
        "This table summarizes latest human-feedback records from the review evidence register.",
        "",
        "## Counts",
        "",
        f"- Latest feedback records: {summary['counts']['latest_feedback']}",
        "",
        "Disposition counts:",
        "",
    ]
    for disposition, count in summary["counts"]["dispositions"].items():
        lines.append(f"- {disposition}: {count}")

    lines.extend([
        "",
        "Affected-field counts:",
        "",
    ])
    for field, count in summary["counts"]["affected_fields"].items():
        lines.append(f"- {field}: {count}")

    lines.extend([
        "",
        "## Feedback Records",
        "",
        "| Finding | Fields | Labels | Disposition queue | Comment |",
        "| --- | --- | --- | --- | --- |",
    ])
    for record in summary["records"]:
        comment = record["comment"].replace("\n", "<br>").replace("|", "\\|")
        lines.append(
            "| "
            f"`{record['source_record_id']}` | "
            f"{', '.join(record['affected_fields']) or '-'} | "
            f"{', '.join(record['labels'])} | "
            f"{record['disposition_queue']} | "
            f"{comment} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(args.register)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text(markdown_table(summary), encoding="utf-8")
    print(
        f"wrote {summary['counts']['latest_feedback']} latest-feedback records to "
        f"{args.json_output} and {args.md_output}"
    )


if __name__ == "__main__":
    main()
