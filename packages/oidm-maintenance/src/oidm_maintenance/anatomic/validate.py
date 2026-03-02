"""Validate anatomic-locations source JSON before building the database.

Provides Pydantic-based record validation and cross-record relationship checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anatomic_locations.models.enums import AnatomicRegion
from anatomic_locations.models.location import AnatomicRef
from loguru import logger
from oidm_common.models.index_code import IndexCode
from pydantic import BaseModel, Field


class AnatomicSourceRecord(BaseModel):
    """Pydantic model for a single record in the anatomic-locations source JSON."""

    id: str = Field(alias="_id", min_length=1)
    description: str = Field(min_length=1)
    region: AnatomicRegion | None = None

    snomed_id: str | None = Field(default=None, alias="snomedId")
    snomed_display: str | None = Field(default=None, alias="snomedDisplay")
    acr_common_id: str | None = Field(default=None, alias="acrCommonId")
    anatomic_locations_id: str | None = Field(default=None, alias="anatomicLocationsId")

    codes: list[IndexCode] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)

    contained_by_ref: AnatomicRef | None = Field(default=None, alias="containedByRef")
    part_of_ref: AnatomicRef | None = Field(default=None, alias="partOfRef")
    left_ref: AnatomicRef | None = Field(default=None, alias="leftRef")
    right_ref: AnatomicRef | None = Field(default=None, alias="rightRef")
    unsided_ref: AnatomicRef | None = Field(default=None, alias="unsidedRef")

    contains_refs: list[AnatomicRef] = Field(default_factory=list, alias="containsRefs")
    has_parts_refs: list[AnatomicRef] = Field(default_factory=list, alias="hasPartsRefs")

    definition: str | None = None
    sex_specific: str | None = Field(default=None, alias="sexSpecific")

    model_config = {"populate_by_name": True}


def _check_record_warnings(rec: AnatomicSourceRecord) -> list[str]:
    """Return non-fatal warnings for a parsed record."""
    warns: list[str] = []
    if rec.snomed_id and not rec.snomed_display:
        warns.append(f"{rec.id}: snomedId present without snomedDisplay")
    if len(rec.synonyms) != len(set(rec.synonyms)):
        removed = len(rec.synonyms) - len(set(rec.synonyms))
        warns.append(f"{rec.id}: {removed} duplicate synonym(s)")
    return warns


@dataclass
class ValidationResult:
    """Collects errors and warnings from source-JSON validation."""

    parse_errors: dict[str, list[str]] = field(default_factory=dict)
    relationship_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.parse_errors and not self.relationship_errors

    @property
    def total_errors(self) -> int:
        return sum(len(v) for v in self.parse_errors.values()) + len(self.relationship_errors)

    def summary(self) -> str:
        lines: list[str] = []
        if self.ok:
            lines.append("Validation passed.")
        else:
            lines.append(f"Validation failed with {self.total_errors} error(s).")
        if self.parse_errors:
            lines.append(f"  Parse errors in {len(self.parse_errors)} record(s)")
        if self.relationship_errors:
            lines.append(f"  {len(self.relationship_errors)} relationship error(s)")
        if self.warnings:
            lines.append(f"  {len(self.warnings)} warning(s)")
        return "\n".join(lines)


def _parse_records(raw_records: list[dict[str, Any]], result: ValidationResult) -> list[AnatomicSourceRecord]:
    """Parse raw dicts into validated AnatomicSourceRecord models."""
    records: list[AnatomicSourceRecord] = []
    for i, raw in enumerate(raw_records):
        record_id = raw.get("_id", f"<index {i}>")
        try:
            rec = AnatomicSourceRecord.model_validate(raw)
            result.warnings.extend(_check_record_warnings(rec))
            records.append(rec)
        except Exception as exc:
            result.parse_errors.setdefault(record_id, []).append(str(exc))
    return records


def _check_ref_targets(
    rec: AnatomicSourceRecord,
    by_id: dict[str, AnatomicSourceRecord],
    all_ids: set[str],
    result: ValidationResult,
) -> None:
    """Check that all ref targets exist and have matching display names."""
    for field_name in ("contained_by_ref", "part_of_ref", "left_ref", "right_ref", "unsided_ref"):
        ref: AnatomicRef | None = getattr(rec, field_name)
        if ref is None:
            continue
        if ref.id not in all_ids:
            result.relationship_errors.append(f"{rec.id}.{field_name}: target {ref.id} not found")
        elif ref.display != by_id[ref.id].description:
            result.relationship_errors.append(
                f"{rec.id}.{field_name}: display '{ref.display}' != target description '{by_id[ref.id].description}'"
            )

    for field_name in ("contains_refs", "has_parts_refs"):
        ref_list: list[AnatomicRef] = getattr(rec, field_name)
        for ref in ref_list:
            if ref.id not in all_ids:
                result.relationship_errors.append(f"{rec.id}.{field_name}: target {ref.id} not found")
            elif ref.display != by_id[ref.id].description:
                result.relationship_errors.append(
                    f"{rec.id}.{field_name}: display '{ref.display}' "
                    f"!= target description '{by_id[ref.id].description}'"
                )


def _check_laterality(
    rec: AnatomicSourceRecord,
    by_id: dict[str, AnatomicSourceRecord],
    result: ValidationResult,
) -> None:
    """Check laterality reciprocity for unsided parent records.

    A record with BOTH leftRef and rightRef and no unsidedRef is the unsided parent.
    Its left/right targets should have unsidedRef pointing back to it.
    """
    if rec.unsided_ref is not None or rec.left_ref is None or rec.right_ref is None:
        return

    if rec.left_ref.id in by_id:
        left_unsided = by_id[rec.left_ref.id].unsided_ref
        if left_unsided is None or left_unsided.id != rec.id:
            result.relationship_errors.append(
                f"{rec.id}: leftRef -> {rec.left_ref.id} but target's unsidedRef doesn't point back"
            )
    if rec.right_ref.id in by_id:
        right_unsided = by_id[rec.right_ref.id].unsided_ref
        if right_unsided is None or right_unsided.id != rec.id:
            result.relationship_errors.append(
                f"{rec.id}: rightRef -> {rec.right_ref.id} but target's unsidedRef doesn't point back"
            )


def _check_parts_bidirectional(
    rec: AnatomicSourceRecord,
    by_id: dict[str, AnatomicSourceRecord],
    result: ValidationResult,
) -> None:
    """Check that hasPartsRefs targets have partOfRef pointing back."""
    for part_ref in rec.has_parts_refs:
        if part_ref.id not in by_id or part_ref.id == rec.id:
            continue
        part = by_id[part_ref.id]
        if part.part_of_ref is None or part.part_of_ref.id != rec.id:
            result.relationship_errors.append(
                f"{rec.id}: hasPartsRefs includes {part_ref.id} but target's partOfRef doesn't point back"
            )


def _check_cycles(by_id: dict[str, AnatomicSourceRecord], result: ValidationResult) -> None:
    """Detect cycles in containedByRef / partOfRef chains (self-refs are OK)."""
    for chain_field in ("contained_by_ref", "part_of_ref"):
        for start_id in by_id:
            visited: set[str] = set()
            current = start_id
            while True:
                ref = getattr(by_id.get(current), chain_field, None) if current in by_id else None
                if ref is None or ref.id == current:
                    break
                if ref.id in visited:
                    result.relationship_errors.append(f"Cycle detected in {chain_field} chain starting at {start_id}")
                    break
                visited.add(current)
                current = ref.id


def validate_relationships(records: list[AnatomicSourceRecord], result: ValidationResult) -> None:
    """Cross-record relationship validation."""
    by_id: dict[str, AnatomicSourceRecord] = {}
    for rec in records:
        if rec.id in by_id:
            result.relationship_errors.append(f"Duplicate _id: {rec.id}")
        else:
            by_id[rec.id] = rec

    all_ids = set(by_id.keys())
    for rec in by_id.values():
        _check_ref_targets(rec, by_id, all_ids, result)
        _check_laterality(rec, by_id, result)
        _check_parts_bidirectional(rec, by_id, result)

    _check_cycles(by_id, result)


def validate_anatomic_json(path: Path) -> ValidationResult:
    """Validate an anatomic-locations source JSON file.

    Args:
        path: Path to the source JSON file.

    Returns:
        ValidationResult with all errors and warnings.
    """
    result = ValidationResult()

    logger.info(f"Validating {path}")
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        result.parse_errors["<file>"] = [f"Expected JSON array, got {type(raw_data).__name__}"]
        return result

    logger.info(f"Parsing {len(raw_data)} records")
    records = _parse_records(raw_data, result)

    logger.info(f"Checking relationships among {len(records)} valid records")
    validate_relationships(records, result)

    logger.info(result.summary())
    return result
