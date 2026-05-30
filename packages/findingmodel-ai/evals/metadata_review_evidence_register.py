"""Build the metadata review evidence register from human review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_EXPORTS = (
    Path(
        "/Users/talkasab/repos/findingmodels-metadata/"
        ".metadata-runs/review-exports/"
        "talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json"
    ),
    Path(
        "/Users/talkasab/repos/findingmodels-metadata/"
        ".metadata-runs/phase5-targeted-review-hardened-v3/"
        "talkasab-metadata-enrichment-review-responses.json"
    ),
)
DEFAULT_OUTPUT = Path(__file__).with_name("fixtures") / "metadata_review_evidence_register.json"
REGISTER_VERSION = 1


def file_sha256(path: Path) -> str:
    """Return a stable content hash for a source artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state(path: Path) -> dict[str, str | None]:
    """Capture enough git state to make local provenance auditable."""

    try:
        repo_root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        head = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return {"repo_root": None, "head": None}

    return {"repo_root": repo_root, "head": head}


def normalize_status(status: str) -> str:
    """Normalize UI review statuses into register statuses."""

    if status == "approved":
        return "approved"
    if status == "feedback":
        return "feedback"
    return "unresolved"


def source_id(source_export: Path) -> str:
    """Return a compact source id for known human review exports."""

    path = str(source_export)
    if "phase5-targeted-review-hardened-v3" in path:
        return "phase5_targeted_hardened_v3"
    if "review-exports" in path:
        return "pilot_review"
    return source_export.stem.replace("-", "_")


def infer_affected_fields(comment: str) -> list[str]:
    """Infer coarse affected fields from reviewer free text without inventing expected answers."""

    lowered = comment.lower()
    field_markers = {
        "anatomic_locations": ("anatomic", "anatomy", "location", "locations", "digit", "hand"),
        "applicable_modalities": (
            "modality",
            "modalities",
            "ultrasound",
            " ct",
            "ct,",
            "ct.",
            "ct or",
            "cta",
            " mr",
            "mr,",
            "mr.",
            " xr",
            "xr,",
            " us ",
            "us,",
            "us.",
        ),
        "age_profile": ("age", "ages", "all-ages", "adolescence", "adolescent", "adult", "child", "children"),
        "body_regions": ("body region", "body_regions"),
        "entity_type": ("entity type", "finding", "diagnosis", "assessment"),
        "etiologies": (
            "etiology",
            "etiologies",
            "cause",
            "causes",
            "vascular",
            "inflammatory",
            "inflammation",
            "ischemic",
            "cardiovascular",
            "heart failure",
            "congenital",
            "neoplastic",
            "benign",
            "post-infectious",
            "post-exposure",
            "post-treatment",
        ),
        "expected_time_course": (
            "time course",
            "timecourse",
            "duration",
            "persistent",
            "transient",
            "resolves",
            "weeks",
            "months",
            "years",
            "permanent",
            "progressive",
        ),
        "index_codes": ("index code", "snomed", "gamut", "radlex", "code"),
        "sex_specificity": ("sex", "sex-neutral", "female", "male"),
        "subspecialties": ("subspecialty", "subspecialties", "subspeciality", "gu imaging"),
    }
    return [field for field, markers in field_markers.items() if any(marker in lowered for marker in markers)]


def latest_records_by_path(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return latest review event per path using review update timestamps."""

    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        path = record["path"]
        current = latest.get(path)
        if current is None or (record["human_review"].get("updated_at") or "") >= (
            current["human_review"].get("updated_at") or ""
        ):
            latest[path] = record
    return latest


def build_register(source_exports: list[Path]) -> dict[str, Any]:
    """Build the review evidence register from human review exports."""

    records = []
    generated_from = []
    provenance_sources = []
    dropped_records: list[dict[str, Any]] = []
    for source_export in source_exports:
        export = json.loads(source_export.read_text(encoding="utf-8"))
        responses = export["responses"]
        source = source_id(source_export)
        source_counts = Counter(normalize_status(response["status"]) for response in responses)
        generated_from.append({
            "path": str(source_export),
            "sha256": file_sha256(source_export),
            "kind": "human_review_export",
            "summary": export.get("summary"),
            "input_response_count": len(responses),
            "imported_record_count": len(responses),
            "dropped_record_count": 0,
            "imported_status_counts": {
                "approved": source_counts["approved"],
                "feedback": source_counts["feedback"],
                "unresolved": source_counts["unresolved"],
            },
        })
        provenance_sources.append({
            "source_artifact": str(source_export),
            "reviewer": export.get("reviewer"),
            "dataset": export.get("dataset"),
            "exported_at": export.get("exported_at"),
            "app_version": export.get("app_version"),
        })
        for index, response in enumerate(responses, start=1):
            status = normalize_status(response["status"])
            comment = response.get("comment") or ""
            records.append({
                "id": f"{source}:{response['item_id']}",
                "source_artifact": str(source_export),
                "source_record_index": index,
                "source_record_id": response["item_id"],
                "path": response["path"],
                "title": response["title"],
                "human_review": {
                    "raw_status": response["status"],
                    "status": status,
                    "comment": comment,
                    "first_reviewed_at": response.get("first_reviewed_at"),
                    "updated_at": response.get("updated_at"),
                },
                "affected_fields": infer_affected_fields(comment),
                "expected_metadata": None,
                "disposition": "unresolved" if status == "feedback" else "baseline_approved",
                "notes": [],
            })

    counts = Counter(record["human_review"]["status"] for record in records)
    latest_by_path = latest_records_by_path(records)
    effective_counts = Counter(record["human_review"]["status"] for record in latest_by_path.values())
    for record in records:
        latest = latest_by_path[record["path"]]
        is_latest = record["id"] == latest["id"]
        record["is_latest"] = is_latest
        record["superseded_by"] = None if is_latest else latest["id"]
    return {
        "version": REGISTER_VERSION,
        "generated_from": generated_from,
        "provenance": {
            "source_repo": git_state(source_exports[0].parent),
            "tool_repo": git_state(Path(__file__).parent),
            "sources": provenance_sources,
        },
        "counts": {
            "total_review_events": len(records),
            "unique_items": len(latest_by_path),
            "approved": counts["approved"],
            "feedback": counts["feedback"],
            "unresolved": counts["unresolved"],
            "effective_approved": effective_counts["approved"],
            "effective_feedback": effective_counts["feedback"],
            "effective_unresolved": effective_counts["unresolved"],
        },
        "dropped_records": dropped_records,
        "usage": {
            "authority": "human_review_only",
            "latest_review_policy": "Use the latest event per path by human_review.updated_at.",
            "machine_extraction_policy": (
                "Derived summaries and candidate extraction are triage aids only; they are not gold "
                "until explicitly promoted by human review."
            ),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export", type=Path, action="append", dest="source_exports")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_exports = args.source_exports or list(DEFAULT_EXPORTS)
    register = build_register(source_exports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(register, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} with {register['counts']['total_review_events']} review events "
        f"over {register['counts']['unique_items']} unique items "
        f"({register['counts']['effective_approved']} latest approved, "
        f"{register['counts']['effective_feedback']} latest feedback)"
    )


if __name__ == "__main__":
    main()
