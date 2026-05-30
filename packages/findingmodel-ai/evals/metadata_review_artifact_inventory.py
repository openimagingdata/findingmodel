"""Inventory metadata review artifacts used by the cleanup/readiness plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DATA_REPO = Path("/Users/talkasab/repos/findingmodels-metadata")
TOOL_REPO = Path(__file__).parents[3]
DEFAULT_OUTPUT = Path(__file__).with_name("fixtures") / "metadata_review_artifact_inventory.json"

KNOWN_FILES = (
    (
        DATA_REPO / ".metadata-runs/review-exports/talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json",
        "human_review_export",
        True,
        "Authoritative 150-record pilot human review export.",
    ),
    (
        DATA_REPO / ".metadata-runs/phase5-targeted-review-hardened-v3/"
        "talkasab-metadata-enrichment-review-responses.json",
        "human_review_export",
        True,
        "Authoritative 30-record targeted follow-up human review export.",
    ),
    (
        DATA_REPO / ".metadata-runs/phase5-targeted-review-hardened-v3-data/review-data.json",
        "human_review_package_input",
        False,
        "Generated metadata payload provenance for the 30-record targeted follow-up human review.",
    ),
    (
        DATA_REPO / ".metadata-runs/pilot-enrichment/before-after",
        "human_review_payload_snapshots",
        False,
        "Generated metadata payload provenance for the 150-record pilot human review.",
    ),
    (
        DATA_REPO / ".metadata-runs/pilot-review-ingest.json",
        "human_review_ingest_summary",
        False,
        "Derived ingest summary for the pilot human review export.",
    ),
    (
        DATA_REPO / ".metadata-runs/phase5-targeted-v3-review-ingest.json",
        "human_review_ingest_summary",
        False,
        "Derived ingest summary for the targeted follow-up human review export.",
    ),
    (
        DATA_REPO / ".metadata-runs/phase6-nongmts-gmts-review-v1/review-decisions.json",
        "subagent_triage",
        False,
        "Supporting phase-6 subagent triage, not human gold.",
    ),
    (
        DATA_REPO / ".metadata-runs/phase6-nongmts-gmts-review-v1/review-data/review-data.json",
        "review_package_input",
        False,
        "Review package input for phase-6 triage.",
    ),
    (
        DATA_REPO / "evals/regression_floor/regression-floor-v1.json",
        "regression_floor",
        False,
        "Data-repo regression floor to port or explicitly replace in later eval work.",
    ),
    (
        DATA_REPO / "evals/regression_floor/manifest.json",
        "regression_floor_manifest",
        False,
        "Manifest for the data-repo regression floor.",
    ),
    (
        TOOL_REPO / "packages/findingmodel-ai/evals/fixtures/etiology_tempo_reviewed_cases.json",
        "reviewed_eval_fixture",
        False,
        "Mixed etiology/time-course eval fixture; contains reviewed cases and comparison-derived overrides.",
    ),
)


def file_sha256(path: Path) -> str:
    """Return a stable content hash for one artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(path: Path) -> str:
    """Return a stable content hash for files under one artifact directory."""

    digest = hashlib.sha256()
    for file_path in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def summarize_json(path: Path) -> dict[str, Any]:
    """Return small count-oriented summary for known JSON artifact shapes."""

    data = json.loads(path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {}
    if isinstance(data, dict):
        for key in ("summary", "status_counts", "counts"):
            if key in data:
                summary[key] = data[key]
        for key in ("responses", "actionable_feedback", "records", "items", "cases"):
            if key in data and isinstance(data[key], list):
                summary[key] = len(data[key])
    elif isinstance(data, list):
        summary["records"] = len(data)
    return summary


def summarize_directory(path: Path) -> dict[str, Any]:
    """Return small count-oriented summary for known directory artifact shapes."""

    summary: dict[str, Any] = {"files": sum(1 for child in path.rglob("*") if child.is_file())}
    before_after = path.name == "before-after"
    if before_after:
        summary["before_snapshots"] = len(list(path.glob("*.before.json")))
        summary["after_snapshots"] = len(list(path.glob("*.after.json")))
    return summary


def gold_fixture_summary() -> dict[str, Any]:
    """Summarize tool-repo gold fixture files."""

    gold_dir = TOOL_REPO / "packages/findingmodel-ai/evals/gold"
    paths = sorted(gold_dir.glob("*.fm.json"))
    return {
        "path": str(gold_dir),
        "kind": "tool_repo_gold_fixtures",
        "authoritative_human_review": True,
        "role": "Existing manually curated gold fixtures.",
        "exists": gold_dir.exists(),
        "count": len(paths),
        "files": [path.name for path in paths],
    }


def pilot_payload_cross_check() -> dict[str, Any]:
    """Check that pilot payload snapshots cover every pilot review response id."""

    response_path = (
        DATA_REPO / ".metadata-runs/review-exports/talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json"
    )
    payload_dir = DATA_REPO / ".metadata-runs/pilot-enrichment/before-after"
    response_ids: set[str] = set()
    if response_path.exists():
        response_data = json.loads(response_path.read_text(encoding="utf-8"))
        response_ids = {
            response["item_id"]
            for response in response_data.get("responses", [])
            if isinstance(response.get("item_id"), str)
        }
    after_ids = (
        {path.name.removesuffix(".after.json") for path in payload_dir.glob("*.after.json")}
        if payload_dir.exists()
        else set()
    )
    return {
        "name": "pilot_after_payloads_match_review_responses",
        "response_artifact": str(response_path),
        "payload_artifact": str(payload_dir),
        "response_count": len(response_ids),
        "after_payload_count": len(after_ids),
        "missing_after_payloads": sorted(response_ids - after_ids),
        "extra_after_payloads": sorted(after_ids - response_ids),
        "passes": response_ids == after_ids and bool(response_ids),
    }


def build_inventory() -> dict[str, Any]:
    """Build review artifact inventory."""

    artifacts = []
    missing = []
    for path, kind, authoritative, role in KNOWN_FILES:
        exists = path.exists()
        record: dict[str, Any] = {
            "path": str(path),
            "kind": kind,
            "authoritative_human_review": authoritative,
            "role": role,
            "exists": exists,
        }
        if exists:
            record["sha256"] = directory_sha256(path) if path.is_dir() else file_sha256(path)
            if path.is_dir():
                record["summary"] = summarize_directory(path)
            elif path.suffix == ".json":
                record["summary"] = summarize_json(path)
        else:
            missing.append(str(path))
        artifacts.append(record)

    artifacts.append(gold_fixture_summary())
    unresolved_expected_artifacts: list[dict[str, Any]] = []
    authoritative_count = sum(1 for artifact in artifacts if artifact["authoritative_human_review"])
    return {
        "version": 1,
        "policy": {
            "authority": "Only human-reviewed artifacts are authoritative.",
            "supporting_evidence": (
                "Ingest summaries, review-package inputs, subagent triage, and regression-floor "
                "runs are preserved as provenance or later eval material, not as gold."
            ),
        },
        "counts": {
            "artifacts": len(artifacts),
            "missing_artifacts": len(missing) + len(unresolved_expected_artifacts),
            "authoritative_artifacts": authoritative_count,
        },
        "missing_artifacts": missing,
        "unresolved_expected_artifacts": unresolved_expected_artifacts,
        "cross_checks": [pilot_payload_cross_check()],
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inventory = build_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {inventory['counts']['artifacts']} artifact records to {args.output} "
        f"({inventory['counts']['missing_artifacts']} missing)"
    )


if __name__ == "__main__":
    main()
