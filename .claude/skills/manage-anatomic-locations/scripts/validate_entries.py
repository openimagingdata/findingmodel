#!/usr/bin/env python3
"""Validate anatomic locations source JSON for schema conformance and referential integrity.

Usage:
    python .claude/skills/manage-anatomic-locations/scripts/validate_entries.py notebooks/data/anatomic_locations_noembed.json
    python .claude/skills/manage-anatomic-locations/scripts/validate_entries.py notebooks/data/anatomic_locations_noembed.json --ids RID34566,RID34566_RID5824
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Data types ---

REF_FIELDS = ("containedByRef", "partOfRef", "leftRef", "rightRef", "unsidedRef")
REF_ARRAY_FIELDS = ("containsRefs", "hasPartsRefs")
VALID_REGIONS = {
    "Head",
    "Neck",
    "Thorax",
    "Abdomen",
    "Pelvis",
    "Upper Extremity",
    "Lower Extremity",
    "Spine",
}


@dataclass
class Issue:
    entry_id: str
    level: str  # "error" or "warning"
    message: str


@dataclass
class ValidationResult:
    issues: list[Issue] = field(default_factory=list)

    def error(self, entry_id: str, msg: str) -> None:
        self.issues.append(Issue(entry_id, "error", msg))

    def warning(self, entry_id: str, msg: str) -> None:
        self.issues.append(Issue(entry_id, "warning", msg))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]


# --- Validators ---


def validate_schema(entry: dict, result: ValidationResult) -> None:
    """Check required fields and field types."""
    eid = entry.get("_id", "<missing>")

    # Required fields
    if "_id" not in entry:
        result.error(eid, "Missing required field: _id")
    if "description" not in entry:
        result.error(eid, "Missing required field: description")

    # Type checks
    if "_id" in entry and not isinstance(entry["_id"], str):
        result.error(eid, f"_id must be string, got {type(entry['_id']).__name__}")
    if "description" in entry and not isinstance(entry["description"], str):
        result.error(eid, f"description must be string, got {type(entry['description']).__name__}")

    # Region validation
    if "region" in entry and entry["region"] not in VALID_REGIONS:
        result.warning(eid, f"Unrecognized region: {entry['region']!r} (known: {', '.join(sorted(VALID_REGIONS))})")

    # Ref field structure
    for ref_field in REF_FIELDS:
        if ref_field in entry:
            ref = entry[ref_field]
            if not isinstance(ref, dict):
                result.error(eid, f"{ref_field} must be an object, got {type(ref).__name__}")
            elif "id" not in ref or "display" not in ref:
                result.error(eid, f"{ref_field} must have 'id' and 'display' fields")

    # Ref array field structure
    for ref_field in REF_ARRAY_FIELDS:
        if ref_field in entry:
            refs = entry[ref_field]
            if not isinstance(refs, list):
                result.error(eid, f"{ref_field} must be an array, got {type(refs).__name__}")
            else:
                for i, ref in enumerate(refs):
                    if not isinstance(ref, dict):
                        result.error(eid, f"{ref_field}[{i}] must be an object")
                    elif "id" not in ref or "display" not in ref:
                        result.error(eid, f"{ref_field}[{i}] must have 'id' and 'display' fields")

    # Synonyms
    if "synonyms" in entry:
        if not isinstance(entry["synonyms"], list):
            result.error(eid, f"synonyms must be an array, got {type(entry['synonyms']).__name__}")
        elif not all(isinstance(s, str) for s in entry["synonyms"]):
            result.error(eid, "All synonyms must be strings")

    # Codes
    if "codes" in entry:
        if not isinstance(entry["codes"], list):
            result.error(eid, f"codes must be an array, got {type(entry['codes']).__name__}")
        else:
            for i, code in enumerate(entry["codes"]):
                if not isinstance(code, dict):
                    result.error(eid, f"codes[{i}] must be an object")
                elif "system" not in code or "code" not in code:
                    result.error(eid, f"codes[{i}] must have 'system' and 'code' fields")

    # SNOMED consistency
    if "snomedId" in entry and "snomedDisplay" not in entry:
        result.warning(eid, "Has snomedId but missing snomedDisplay")
    if "snomedDisplay" in entry and "snomedId" not in entry:
        result.warning(eid, "Has snomedDisplay but missing snomedId")

    # Advisory warnings
    if "snomedId" not in entry:
        result.warning(eid, "No SNOMED code — consider adding one")
    if "region" not in entry:
        result.warning(eid, "No region specified")


def validate_referential_integrity(entries: list[dict], result: ValidationResult, filter_ids: set[str] | None) -> None:
    """Check that all refs point to existing entries and bidirectional refs are consistent."""
    id_set = {e["_id"] for e in entries if "_id" in e}
    by_id = {e["_id"]: e for e in entries if "_id" in e}

    for entry in entries:
        eid = entry.get("_id", "<missing>")
        if filter_ids and eid not in filter_ids:
            continue

        # Check single refs exist
        for ref_field in REF_FIELDS:
            if ref_field in entry and isinstance(entry[ref_field], dict):
                ref_id = entry[ref_field].get("id")
                if ref_id and ref_id not in id_set:
                    result.error(eid, f"{ref_field}.id = {ref_id!r} not found in entries")

        # Check array refs exist
        for ref_field in REF_ARRAY_FIELDS:
            if ref_field in entry and isinstance(entry[ref_field], list):
                for ref in entry[ref_field]:
                    if isinstance(ref, dict):
                        ref_id = ref.get("id")
                        if ref_id and ref_id not in id_set:
                            result.error(eid, f"{ref_field} references {ref_id!r} which is not found in entries")

    # Bidirectional checks — only for filtered entries or all
    check_ids = filter_ids or id_set

    for eid in check_ids:
        if eid not in by_id:
            continue
        entry = by_id[eid]

        # containsRefs <-> containedByRef
        if "containsRefs" in entry and isinstance(entry["containsRefs"], list):
            for ref in entry["containsRefs"]:
                if not isinstance(ref, dict):
                    continue
                child_id = ref.get("id")
                if child_id and child_id in by_id:
                    child = by_id[child_id]
                    child_container = child.get("containedByRef", {})
                    if isinstance(child_container, dict) and child_container.get("id") != eid:
                        result.warning(
                            eid,
                            f"containsRefs lists {child_id!r} but its containedByRef points to "
                            f"{child_container.get('id')!r}, not {eid!r}",
                        )

        # containedByRef -> parent's containsRefs (advisory)
        if "containedByRef" in entry and isinstance(entry["containedByRef"], dict):
            parent_id = entry["containedByRef"].get("id")
            if parent_id and parent_id in by_id:
                parent = by_id[parent_id]
                parent_contains = parent.get("containsRefs", [])
                if isinstance(parent_contains, list):
                    child_ids = {r.get("id") for r in parent_contains if isinstance(r, dict)}
                    if eid not in child_ids and parent_contains:
                        # Only warn if parent has containsRefs at all (many don't)
                        result.warning(
                            eid,
                            f"containedByRef points to {parent_id!r} but parent's containsRefs "
                            f"does not list {eid!r} (parent may not track all children)",
                        )

        # hasPartsRefs <-> partOfRef
        if "hasPartsRefs" in entry and isinstance(entry["hasPartsRefs"], list):
            for ref in entry["hasPartsRefs"]:
                if not isinstance(ref, dict):
                    continue
                part_id = ref.get("id")
                if part_id and part_id in by_id:
                    part = by_id[part_id]
                    part_of = part.get("partOfRef", {})
                    if isinstance(part_of, dict) and part_of.get("id") != eid:
                        result.warning(
                            eid,
                            f"hasPartsRefs lists {part_id!r} but its partOfRef points to "
                            f"{part_of.get('id')!r}, not {eid!r}",
                        )

        # Laterality cross-refs
        _check_laterality_consistency(entry, by_id, result)


def _check_laterality_consistency(entry: dict, by_id: dict, result: ValidationResult) -> None:
    """Check laterality cross-references are consistent."""
    eid = entry.get("_id", "")

    # Generic entry: has leftRef AND rightRef
    if "leftRef" in entry and "rightRef" in entry:
        left_id = entry["leftRef"].get("id") if isinstance(entry["leftRef"], dict) else None
        right_id = entry["rightRef"].get("id") if isinstance(entry["rightRef"], dict) else None

        # Left entry should have unsidedRef pointing back
        if left_id and left_id in by_id:
            left_entry = by_id[left_id]
            unsided = left_entry.get("unsidedRef", {})
            if isinstance(unsided, dict) and unsided.get("id") != eid:
                result.warning(
                    eid,
                    f"leftRef {left_id!r} has unsidedRef pointing to "
                    f"{unsided.get('id')!r}, not back to {eid!r}",
                )

        # Right entry should have unsidedRef pointing back
        if right_id and right_id in by_id:
            right_entry = by_id[right_id]
            unsided = right_entry.get("unsidedRef", {})
            if isinstance(unsided, dict) and unsided.get("id") != eid:
                result.warning(
                    eid,
                    f"rightRef {right_id!r} has unsidedRef pointing to "
                    f"{unsided.get('id')!r}, not back to {eid!r}",
                )

    # Left entry (has rightRef + unsidedRef, no leftRef)
    if "rightRef" in entry and "unsidedRef" in entry and "leftRef" not in entry:
        right_id = entry["rightRef"].get("id") if isinstance(entry["rightRef"], dict) else None
        if right_id and right_id in by_id:
            right_entry = by_id[right_id]
            # Right counterpart should have leftRef pointing back to us
            left_ref = right_entry.get("leftRef", {})
            if isinstance(left_ref, dict) and left_ref.get("id") != eid:
                result.warning(
                    eid,
                    f"rightRef {right_id!r} has leftRef pointing to "
                    f"{left_ref.get('id')!r}, not back to {eid!r}",
                )

    # Right entry (has leftRef + unsidedRef, no rightRef)
    if "leftRef" in entry and "unsidedRef" in entry and "rightRef" not in entry:
        left_id = entry["leftRef"].get("id") if isinstance(entry["leftRef"], dict) else None
        if left_id and left_id in by_id:
            left_entry = by_id[left_id]
            # Left counterpart should have rightRef pointing back to us
            right_ref = left_entry.get("rightRef", {})
            if isinstance(right_ref, dict) and right_ref.get("id") != eid:
                result.warning(
                    eid,
                    f"leftRef {left_id!r} has rightRef pointing to "
                    f"{right_ref.get('id')!r}, not back to {eid!r}",
                )

    # ID pattern consistency
    if eid.endswith("_RID5824") and "unsidedRef" not in entry:
        result.warning(eid, "Left-lateralized ID (_RID5824) but no unsidedRef")
    if eid.endswith("_RID5825") and "unsidedRef" not in entry:
        result.warning(eid, "Right-lateralized ID (_RID5825) but no unsidedRef")


def validate_duplicates(entries: list[dict], result: ValidationResult) -> None:
    """Check for duplicate IDs."""
    seen: dict[str, int] = {}
    for entry in entries:
        eid = entry.get("_id", "<missing>")
        seen[eid] = seen.get(eid, 0) + 1

    for eid, count in seen.items():
        if count > 1:
            result.error(eid, f"Duplicate _id: appears {count} times")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate anatomic locations source JSON")
    parser.add_argument("file", help="Path to anatomic_locations_noembed.json")
    parser.add_argument(
        "--ids",
        help="Comma-separated list of IDs to focus validation on (still validates schema for all)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        print("Error: Expected a JSON array at the top level")
        sys.exit(1)

    filter_ids: set[str] | None = None
    if args.ids:
        filter_ids = set(args.ids.split(","))

    result = ValidationResult()

    # Schema validation (all entries, or filtered)
    for entry in entries:
        eid = entry.get("_id", "<missing>")
        if filter_ids and eid not in filter_ids:
            continue
        validate_schema(entry, result)

    # Duplicate check (always all entries)
    validate_duplicates(entries, result)

    # Referential integrity
    validate_referential_integrity(entries, result, filter_ids)

    # Report
    print(f"\nValidated {len(entries)} entries", end="")
    if filter_ids:
        print(f" (focused on {len(filter_ids)} IDs)", end="")
    print()
    print(f"  Errors:   {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")

    if result.errors:
        print(f"\n{'='*60}")
        print("ERRORS (must fix)")
        print(f"{'='*60}")
        for issue in result.errors:
            print(f"  [{issue.entry_id}] {issue.message}")

    if result.warnings:
        print(f"\n{'='*60}")
        print("WARNINGS (advisory)")
        print(f"{'='*60}")
        for issue in result.warnings:
            print(f"  [{issue.entry_id}] {issue.message}")

    if not result.errors and not result.warnings:
        print("\nAll checks passed!")

    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
