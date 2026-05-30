"""Snapshot latest human-approved generated metadata outputs for Gate A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_DATA_REPO = Path("/Users/talkasab/repos/findingmodels-metadata")
DEFAULT_OVERLAP = Path(__file__).with_name("fixtures") / "metadata_review_source_overlap.json"
DEFAULT_REGISTER = Path(__file__).with_name("fixtures") / "metadata_review_evidence_register.json"
DEFAULT_OUTPUT = Path(__file__).with_name("fixtures") / "metadata_review_approved_outputs.json"

REVIEW_PACKAGES = {
    "96bd0ace1aed": DEFAULT_DATA_REPO / ".metadata-runs/phase5-targeted-review-hardened-v3-data/review-data.json",
}
PILOT_AFTER_PAYLOADS = DEFAULT_DATA_REPO / ".metadata-runs/pilot-enrichment/before-after"

METADATA_FIELDS = (
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
)


def file_sha256(path: Path) -> str:
    """Return a stable content hash for an approved output source file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_sha256(path: Path) -> str:
    """Return a stable content hash for one reviewed payload source."""

    if path.is_file():
        return file_sha256(path)
    digest = hashlib.sha256()
    for file_path in sorted(child for child in path.rglob("*") if child.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def approved_overlap_records(overlap_path: Path) -> list[dict[str, Any]]:
    """Return latest-approved source-overlap records."""

    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    return [record for record in overlap["records"] if record["review_status"] == "approved"]


def load_register_records(register_path: Path) -> dict[str, dict[str, Any]]:
    """Return review register records by id."""

    register = json.loads(register_path.read_text(encoding="utf-8"))
    return {record["id"]: record for record in register["records"]}


def load_review_items() -> dict[str, dict[str, dict[str, Any]]]:
    """Return review-package items by dataset id and item id for available packages."""

    package_items = {}
    if PILOT_AFTER_PAYLOADS.exists():
        package_items["69ede7dd9df4"] = {
            path.name.removesuffix(".after.json"): json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(PILOT_AFTER_PAYLOADS.glob("*.after.json"))
        }
    for dataset_id, path in REVIEW_PACKAGES.items():
        if not path.exists():
            continue
        package = json.loads(path.read_text(encoding="utf-8"))
        package_items[dataset_id] = {item["id"]: item for item in package["items"]}
    return package_items


def dataset_id(review_record: dict[str, Any]) -> str | None:
    """Return the dataset id for one review register record."""

    source = review_record["source_artifact"]
    if "phase5-targeted-review-hardened-v3" in source:
        return "96bd0ace1aed"
    if "review-exports" in source:
        return "69ede7dd9df4"
    return None


def normalize_metadata_item(item: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata fields from a reviewed package item."""

    metadata = {field: item.get(field) for field in METADATA_FIELDS}
    for field in ("index_codes", "anatomic_locations"):
        values = metadata.get(field)
        if not isinstance(values, list):
            continue
        metadata[field] = [
            {
                **value,
                "display": value.get("display") or value.get("label"),
            }
            for value in values
            if isinstance(value, dict)
        ]
    return metadata


def reviewed_payload_source(dataset_id: str | None) -> tuple[str, Path | None]:
    """Return the source label and path for one reviewed payload dataset."""

    if dataset_id == "69ede7dd9df4":
        return "pilot_after_payload", PILOT_AFTER_PAYLOADS
    if dataset_id in REVIEW_PACKAGES:
        return "review_package_payload", REVIEW_PACKAGES[dataset_id]
    raise ValueError(f"Unknown review dataset id for approved output snapshot: {dataset_id!r}")


def snapshot_approved_outputs(data_repo: Path, overlap_path: Path, register_path: Path) -> dict[str, Any]:
    """Snapshot metadata fields from latest human-approved generated source outputs."""

    register_by_id = load_register_records(register_path)
    review_items = load_review_items()
    records = []
    for overlap_record in approved_overlap_records(overlap_path):
        review_record = register_by_id[overlap_record["review_id"]]
        record_dataset_id = dataset_id(review_record)
        snapshot_source, review_package_path = reviewed_payload_source(record_dataset_id)
        reviewed_item = review_items.get(record_dataset_id or "", {}).get(overlap_record["item_id"])
        source_path = data_repo / overlap_record["path"]
        if reviewed_item is None:
            raise RuntimeError(
                "Approved human-review record has no reviewed payload snapshot: "
                f"dataset={record_dataset_id}, item={overlap_record['item_id']}"
            )
        review_package_sha256 = artifact_sha256(review_package_path) if review_package_path else None
        reviewed_payload_sha256 = hashlib.sha256(json.dumps(reviewed_item, sort_keys=True).encode("utf-8")).hexdigest()
        metadata = normalize_metadata_item(reviewed_item)
        records.append({
            "item_id": overlap_record["item_id"],
            "path": overlap_record["path"],
            "review_id": overlap_record["review_id"],
            "review_dataset_id": record_dataset_id,
            "snapshot_source": snapshot_source,
            "review_package_path": str(review_package_path) if review_package_path else None,
            "review_package_sha256": review_package_sha256,
            "reviewed_payload_sha256": reviewed_payload_sha256,
            "source_sha256": file_sha256(source_path),
            "name": reviewed_item.get("name"),
            "description": reviewed_item.get("description"),
            "synonyms": reviewed_item.get("synonyms") or [],
            "metadata": metadata,
        })

    source_counts = {
        "review_package_payload": sum(1 for record in records if record["snapshot_source"] == "review_package_payload"),
        "pilot_after_payload": sum(1 for record in records if record["snapshot_source"] == "pilot_after_payload"),
        "current_data_repo_def": sum(1 for record in records if record["snapshot_source"] == "current_data_repo_def"),
    }
    return {
        "version": 1,
        "source_data_repo": str(data_repo),
        "source_overlap": str(overlap_path),
        "source_register": str(register_path),
        "usage": {
            "authority": "latest_human_approved_only",
            "not_gold_policy": (
                "This is a preservation snapshot of generated source metadata for latest-approved "
                "records. It is not a gold fixture and is not an approved source-application command."
            ),
        },
        "counts": {
            "approved_outputs": len(records),
            "snapshot_sources": source_counts,
        },
        "metadata_fields": list(METADATA_FIELDS),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-repo", type=Path, default=DEFAULT_DATA_REPO)
    parser.add_argument("--overlap", type=Path, default=DEFAULT_OVERLAP)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snapshot = snapshot_approved_outputs(args.data_repo, args.overlap, args.register)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {snapshot['counts']['approved_outputs']} approved outputs to {args.output}")


if __name__ == "__main__":
    main()
