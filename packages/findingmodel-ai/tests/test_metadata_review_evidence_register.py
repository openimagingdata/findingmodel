"""Tests for the human review evidence register."""

import json
from pathlib import Path

BACKFILL_CSV_PATH = (
    Path(__file__).parents[3] / "notebooks" / "data" / "brain_volumetry_anatomic_code_display_backfill_2026-05-10.csv"
)
REGISTER_PATH = Path(__file__).parents[1] / "evals" / "fixtures" / "metadata_review_evidence_register.json"
OVERLAP_PATH = Path(__file__).parents[1] / "evals" / "fixtures" / "metadata_review_source_overlap.json"
SUMMARY_PATH = Path(__file__).parents[1] / "evals" / "fixtures" / "metadata_review_feedback_summary.json"
CANDIDATES_PATH = Path(__file__).parents[1] / "evals" / "fixtures" / "metadata_review_expected_candidates.json"
APPROVED_OUTPUTS_PATH = Path(__file__).parents[1] / "evals" / "fixtures" / "metadata_review_approved_outputs.json"
SOURCE_APPLY_MANIFEST_PATH = Path(__file__).parents[1] / "evals" / "fixtures" / "metadata_source_apply_manifest.json"
ARTIFACT_INVENTORY_PATH = Path(__file__).parents[1] / "evals" / "fixtures" / "metadata_review_artifact_inventory.json"


def test_review_evidence_register_preserves_human_review_counts() -> None:
    data = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))

    assert data["version"] == 1
    assert data["counts"] == {
        "total_review_events": 180,
        "unique_items": 150,
        "approved": 67,
        "feedback": 113,
        "unresolved": 0,
        "effective_approved": 67,
        "effective_feedback": 83,
        "effective_unresolved": 0,
    }
    assert len(data["records"]) == 180
    assert len({record["id"] for record in data["records"]}) == 180
    assert len({record["path"] for record in data["records"]}) == 150
    assert data["dropped_records"] == []
    assert data["usage"]["authority"] == "human_review_only"
    assert [source["imported_record_count"] for source in data["generated_from"]] == [150, 30]
    assert sum(1 for record in data["records"] if record["is_latest"]) == 150
    assert all(record["superseded_by"] is None for record in data["records"] if record["is_latest"])


def test_review_evidence_register_feedback_records_keep_comments() -> None:
    data = json.loads(REGISTER_PATH.read_text(encoding="utf-8"))

    feedback_records = [record for record in data["records"] if record["human_review"]["status"] == "feedback"]

    assert len(feedback_records) == 113
    assert all(record["human_review"]["comment"] for record in feedback_records)
    assert all(record["disposition"] == "unresolved" for record in feedback_records)


def test_review_source_overlap_preserves_gate_a_counts() -> None:
    data = json.loads(OVERLAP_PATH.read_text(encoding="utf-8"))

    assert data["counts"] == {
        "total_modified_defs": 160,
        "approved": 67,
        "feedback": 83,
        "not_in_register": 10,
    }
    assert len(data["records"]) == 160
    assert data["data_repo_head"]
    assert "feedback and unreviewed overlap must not be treated as approved" in data["gate_a_policy"]


def test_review_feedback_summary_preserves_latest_feedback_records() -> None:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert data["counts"]["latest_feedback"] == 83
    assert len(data["records"]) == 83
    assert data["counts"]["dispositions"]["expected-metadata-extraction"] > 0
    assert data["counts"]["dispositions"]["code/anatomy-review"] > 0
    assert data["counts"]["dispositions"]["source-model-issue"] > 0
    assert all(record["triage_status"] == "unresolved" for record in data["records"])
    assert all(record["next_action"] for record in data["records"])


def test_review_expected_candidates_extract_conservative_metadata_hints() -> None:
    data = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))

    assert data["counts"]["candidate_records"] > 0
    assert data["counts"]["extracted_fields"]["expected_time_course"] > 0
    assert data["counts"]["extracted_fields"]["age_profile"] > 0
    assert data["counts"]["extracted_fields"]["sex_specificity"] > 0
    assert all(record["promotion_status"] == "candidate" for record in data["records"])
    assert all(record["requires_human_promotion"] for record in data["records"])


def test_review_approved_outputs_preserve_latest_approved_source_metadata() -> None:
    data = json.loads(APPROVED_OUTPUTS_PATH.read_text(encoding="utf-8"))

    assert data["counts"]["approved_outputs"] == 67
    assert data["counts"]["snapshot_sources"] == {
        "current_data_repo_def": 0,
        "pilot_after_payload": 46,
        "review_package_payload": 21,
    }
    assert len(data["records"]) == 67
    assert data["metadata_fields"] == [
        "entity_type",
        "body_regions",
        "subspecialties",
        "etiologies",
        "applicable_modalities",
        "expected_time_course",
        "age_profile",
        "sex_specificity",
        "index_codes",
        "anatomic_locations",
        "tags",
    ]
    assert all(record["review_id"] for record in data["records"])
    assert all(record["source_sha256"] for record in data["records"])
    assert data["usage"]["authority"] == "latest_human_approved_only"
    assert "not a gold fixture" in data["usage"]["not_gold_policy"]
    assert all(record["reviewed_payload_sha256"] for record in data["records"])


def test_source_apply_manifest_reconciles_reviewed_source_application() -> None:
    manifest = json.loads(SOURCE_APPLY_MANIFEST_PATH.read_text(encoding="utf-8"))
    approved_outputs = json.loads(APPROVED_OUTPUTS_PATH.read_text(encoding="utf-8"))
    overlap = json.loads(OVERLAP_PATH.read_text(encoding="utf-8"))

    assert manifest["counts"] == {
        "total_records": 78,
        "human_approved_metadata": 67,
        "index_code_display_backfill": 11,
    }
    assert manifest["reconciliation"] == {
        "source_overlap_total_modified_defs": 160,
        "source_overlap_approved": 67,
        "source_overlap_feedback_not_applied": 83,
        "source_overlap_not_in_register_not_applied_as_enrichment": 10,
        "applied_human_approved_metadata": 67,
        "applied_index_code_display_backfills": 11,
        "note": (
            "The 160-record overlap was an audit of generated source diffs. Only "
            "latest human-approved metadata is applied as enrichment; index-code "
            "display backfills are separately authorized and field-limited."
        ),
    }

    approved_manifest_paths = {
        record["path"] for record in manifest["records"] if record["classification"] == "human_approved_metadata"
    }
    approved_output_paths = {record["path"] for record in approved_outputs["records"]}
    assert approved_manifest_paths == approved_output_paths

    overlap_status_by_path = {record["path"]: record["review_status"] for record in overlap["records"]}
    assert {
        overlap_status_by_path[record["path"]]
        for record in manifest["records"]
        if record["classification"] == "human_approved_metadata"
    } == {"approved"}


def test_source_apply_manifest_limits_nonapproved_records_to_index_code_display_backfills() -> None:
    manifest = json.loads(SOURCE_APPLY_MANIFEST_PATH.read_text(encoding="utf-8"))
    backfills = [record for record in manifest["records"] if record["classification"] == "index_code_display_backfill"]

    assert len(backfills) == 11
    assert {record["source_overlap_status"] for record in backfills} == {
        "feedback",
        "not_in_register",
    }
    assert sum(1 for record in backfills if record["source_overlap_status"] == "feedback") == 1
    assert all(
        record["allowed_change"]
        == "Add display strings to existing index_codes only; do not change systems, codes, or enrichment metadata."
        for record in backfills
    )
    assert all(record["diff_evidence"]["unchanged_except_index_code_display_additions"] for record in backfills)
    assert all(record["diff_evidence"]["before_sha256"] for record in backfills)
    assert all(record["diff_evidence"]["after_sha256"] for record in backfills)
    assert all(record["diff_evidence"]["added_index_code_displays"] for record in backfills)

    expected_csv = str(BACKFILL_CSV_PATH.relative_to(Path(__file__).parents[3]))
    assert {record["evidence"]["display_backfill_csv"] for record in backfills} == {expected_csv}
    for record in backfills:
        added_displays = [
            {
                "system": row["system"],
                "code": row["code"],
                "display": row["display"],
            }
            for row in record["diff_evidence"]["added_index_code_displays"]
        ]
        assert record["evidence"]["csv_rows"] == added_displays


def test_review_artifact_inventory_accounts_for_known_artifacts() -> None:
    data = json.loads(ARTIFACT_INVENTORY_PATH.read_text(encoding="utf-8"))

    assert data["counts"] == {
        "artifacts": 12,
        "authoritative_artifacts": 3,
        "missing_artifacts": 0,
    }
    assert data["missing_artifacts"] == []
    assert data["unresolved_expected_artifacts"] == []
    assert data["cross_checks"] == [
        {
            "name": "pilot_after_payloads_match_review_responses",
            "response_artifact": (
                "/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/review-exports/"
                "talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json"
            ),
            "payload_artifact": (
                "/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/pilot-enrichment/before-after"
            ),
            "response_count": 150,
            "after_payload_count": 150,
            "missing_after_payloads": [],
            "extra_after_payloads": [],
            "passes": True,
        }
    ]
    assert data["policy"]["authority"] == "Only human-reviewed artifacts are authoritative."
    kinds = {artifact["kind"] for artifact in data["artifacts"]}
    assert {
        "human_review_export",
        "human_review_payload_snapshots",
        "human_review_ingest_summary",
        "subagent_triage",
        "regression_floor",
        "reviewed_eval_fixture",
        "tool_repo_gold_fixtures",
    } <= kinds
