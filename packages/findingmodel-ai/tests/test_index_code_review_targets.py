"""Tests for reviewed index-code/anatomic-location target fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from evals.index_code_review_targets import (
    ANATOMIC_LOCATION_SYSTEM,
    DEFAULT_APPROVED,
    DEFAULT_WORKSHEET,
    INDEX_CODE_SYSTEMS,
    build_targets,
)
from evals.metadata_readiness_run import reviewed_gold_for_item

FIXTURES = Path(__file__).parents[1] / "evals" / "fixtures"
TARGETS_PATH = FIXTURES / "index_code_review_targets.json"
APPROVED_OUTPUTS_PATH = FIXTURES / "metadata_review_approved_outputs.json"


def test_index_code_review_targets_are_reproducible_from_pruned_worksheet() -> None:
    generated = build_targets(DEFAULT_WORKSHEET, DEFAULT_APPROVED)
    checked_in = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))

    assert checked_in == generated


def test_index_code_review_targets_validate_reviewed_fixture_shape() -> None:
    data = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))

    assert data["counts"] == {
        "records": 67,
        "index_code_targets": 128,
        "anatomic_location_targets": 69,
        "empty_index_code_records": 3,
        "empty_anatomic_location_records": 2,
        "skipped_non_anatomic_location_candidates": 1,
    }
    assert data["target_fields"] == ["index_codes", "anatomic_locations"]
    assert data["empty_targets"]["index_codes"] == [
        "abdominal_clips",
        "basal_cistern_effacement",
        "breast_malignancy_risk",
    ]
    assert data["empty_targets"]["anatomic_locations"] == [
        "radiolucent_urinary_calculus",
        "tunneled_catheter",
    ]
    assert len(data["records"]) == 67


def test_index_code_review_targets_keep_systems_in_expected_fields() -> None:
    data = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))

    for record in data["records"]:
        for code in record["index_codes"]:
            assert code["system"] in INDEX_CODE_SYSTEMS
        for code in record["anatomic_locations"]:
            assert code["system"] == ANATOMIC_LOCATION_SYSTEM


def test_index_code_review_targets_record_basal_cistern_gap_without_fake_anatomy() -> None:
    data = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    basal = next(record for record in data["records"] if record["item_id"] == "basal_cistern_effacement")

    assert basal["index_codes"] == []
    assert basal["anatomic_locations"] == [
        {
            "code": "RID6383_RID9080",
            "display": "intracranial head",
            "system": "ANATOMICLOCATIONS",
        }
    ]
    assert data["known_gaps"] == [
        {
            "decision": (
                "Exclude RADLEX:RID9865 from anatomic_locations for now; do not invent an "
                "ANATOMICLOCATIONS basal cistern code."
            ),
            "field": "anatomic_locations",
            "gap": "missing_anatomic_location_cistern_concept",
            "item_id": "basal_cistern_effacement",
        }
    ]
    assert data["skipped_candidates"] == [
        {
            "candidate": {
                "code": "RID9865",
                "display": "basal cistern",
                "system": "RADLEX",
            },
            "field": "anatomic_locations",
            "item_id": "basal_cistern_effacement",
            "name": "basal cistern effacement",
            "reason": (
                "Local anatomic_locations does not currently contain basal, basilar, "
                "perimesencephalic, subarachnoid, or generic cistern entries."
            ),
        }
    ]


def test_reviewed_gold_for_item_overlays_only_code_and_anatomy_targets() -> None:
    approved = json.loads(APPROVED_OUTPUTS_PATH.read_text(encoding="utf-8"))["records"]
    item = next(record for record in approved if record["item_id"] == "abdominal_clips")

    gold = reviewed_gold_for_item(item)

    assert item["metadata"]["index_codes"] is None
    assert item["metadata"]["anatomic_locations"] == [
        {
            "code": "RID32954",
            "display": "abdominal cavity",
            "system": "ANATOMICLOCATIONS",
        }
    ]
    assert gold["index_codes"] == []
    assert gold["anatomic_locations"] == [
        {
            "code": "RID56",
            "display": "abdomen",
            "system": "ANATOMICLOCATIONS",
        }
    ]
    assert gold["entity_type"] == item["metadata"]["entity_type"]
    assert gold["applicable_modalities"] == item["metadata"]["applicable_modalities"]
