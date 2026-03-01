"""Tests for anatomic-locations source JSON validation."""

import json
from pathlib import Path

import pytest
from oidm_maintenance.anatomic.validate import (
    AnatomicSourceRecord,
    ValidationResult,
    _check_record_warnings,
    validate_anatomic_json,
    validate_relationships,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_record(**overrides: object) -> dict[str, object]:
    """Return a minimal valid source record, with optional overrides."""
    base: dict[str, object] = {"_id": "RID001", "description": "test location"}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# AnatomicSourceRecord — field-level validation
# ---------------------------------------------------------------------------


class TestAnatomicSourceRecord:
    def test_minimal_valid(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record())
        assert rec.id == "RID001"
        assert rec.description == "test location"
        assert rec.codes == []
        assert rec.synonyms == []

    def test_missing_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            AnatomicSourceRecord.model_validate({"description": "no id"})

    def test_empty_id_fails(self) -> None:
        with pytest.raises(ValidationError):
            AnatomicSourceRecord.model_validate({"_id": "", "description": "empty id"})

    def test_missing_description_fails(self) -> None:
        with pytest.raises(ValidationError):
            AnatomicSourceRecord.model_validate({"_id": "RID001"})

    def test_valid_region(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(region="Thorax"))
        assert rec.region is not None
        assert rec.region.value == "Thorax"

    def test_invalid_region_fails(self) -> None:
        with pytest.raises(ValidationError):
            AnatomicSourceRecord.model_validate(_minimal_record(region="InvalidRegion"))

    def test_snomed_fields(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(snomedId="12345", snomedDisplay="Structure of thing"))
        assert rec.snomed_id == "12345"
        assert rec.snomed_display == "Structure of thing"

    def test_codes_parsed(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(codes=[{"system": "FMA", "code": "54880"}]))
        assert len(rec.codes) == 1
        assert rec.codes[0].system == "FMA"

    def test_invalid_code_fails(self) -> None:
        with pytest.raises(ValidationError):
            AnatomicSourceRecord.model_validate(
                _minimal_record(codes=[{"system": "X", "code": "1"}])  # too short
            )

    def test_synonyms_no_duplicates(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(synonyms=["alpha", "beta"]))
        assert rec.synonyms == ["alpha", "beta"]

    def test_ref_fields_parsed(self) -> None:
        rec = AnatomicSourceRecord.model_validate(
            _minimal_record(
                containedByRef={"id": "RID002", "display": "parent"},
                partOfRef={"id": "RID003", "display": "whole"},
            )
        )
        assert rec.contained_by_ref is not None
        assert rec.contained_by_ref.id == "RID002"
        assert rec.part_of_ref is not None
        assert rec.part_of_ref.id == "RID003"

    def test_laterality_refs(self) -> None:
        rec = AnatomicSourceRecord.model_validate(
            _minimal_record(
                leftRef={"id": "RID001_L", "display": "left thing"},
                rightRef={"id": "RID001_R", "display": "right thing"},
            )
        )
        assert rec.left_ref is not None
        assert rec.right_ref is not None

    def test_list_refs_parsed(self) -> None:
        rec = AnatomicSourceRecord.model_validate(
            _minimal_record(
                containsRefs=[{"id": "RID010", "display": "child"}],
                hasPartsRefs=[{"id": "RID011", "display": "part"}],
            )
        )
        assert len(rec.contains_refs) == 1
        assert len(rec.has_parts_refs) == 1

    def test_sex_specific_field(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(sexSpecific="Male"))
        assert rec.sex_specific == "Male"

    def test_definition_field(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(definition="A structure."))
        assert rec.definition == "A structure."

    def test_anatomic_locations_id_field(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(anatomicLocationsId="RID001"))
        assert rec.anatomic_locations_id == "RID001"


# ---------------------------------------------------------------------------
# _check_record_warnings
# ---------------------------------------------------------------------------


class TestRecordWarnings:
    def test_no_warnings_for_clean_record(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record())
        assert _check_record_warnings(rec) == []

    def test_snomed_id_without_display(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(snomedId="12345"))
        warns = _check_record_warnings(rec)
        assert len(warns) == 1
        assert "snomedId present without snomedDisplay" in warns[0]

    def test_snomed_with_display_no_warning(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(snomedId="12345", snomedDisplay="Structure"))
        assert _check_record_warnings(rec) == []

    def test_duplicate_synonyms(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(synonyms=["alpha", "alpha"]))
        warns = _check_record_warnings(rec)
        assert len(warns) == 1
        assert "duplicate synonym" in warns[0]

    def test_unique_synonyms_no_warning(self) -> None:
        rec = AnatomicSourceRecord.model_validate(_minimal_record(synonyms=["alpha", "beta"]))
        assert _check_record_warnings(rec) == []


# ---------------------------------------------------------------------------
# validate_relationships
# ---------------------------------------------------------------------------


def _make_records(*dicts: dict[str, object]) -> list[AnatomicSourceRecord]:
    return [AnatomicSourceRecord.model_validate(d) for d in dicts]


class TestValidateRelationships:
    def test_valid_pair(self) -> None:
        recs = _make_records(
            {"_id": "A", "description": "alpha", "containedByRef": {"id": "B", "display": "beta"}},
            {"_id": "B", "description": "beta"},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert result.ok

    def test_duplicate_ids(self) -> None:
        recs = _make_records(
            {"_id": "A", "description": "first"},
            {"_id": "A", "description": "second"},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("Duplicate _id" in e for e in result.relationship_errors)

    def test_missing_ref_target(self) -> None:
        recs = _make_records(
            {"_id": "A", "description": "alpha", "containedByRef": {"id": "MISSING", "display": "gone"}},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("not found" in e for e in result.relationship_errors)

    def test_display_mismatch(self) -> None:
        recs = _make_records(
            {"_id": "A", "description": "alpha", "containedByRef": {"id": "B", "display": "wrong name"}},
            {"_id": "B", "description": "beta"},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("display" in e and "wrong name" in e for e in result.relationship_errors)

    def test_laterality_reciprocity_ok(self) -> None:
        recs = _make_records(
            {
                "_id": "UNSIDED",
                "description": "thing",
                "leftRef": {"id": "LEFT", "display": "left thing"},
                "rightRef": {"id": "RIGHT", "display": "right thing"},
            },
            {"_id": "LEFT", "description": "left thing", "unsidedRef": {"id": "UNSIDED", "display": "thing"}},
            {"_id": "RIGHT", "description": "right thing", "unsidedRef": {"id": "UNSIDED", "display": "thing"}},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert result.ok

    def test_laterality_missing_backref(self) -> None:
        """Unsided parent (has both leftRef+rightRef, no unsidedRef) expects targets to point back."""
        recs = _make_records(
            {
                "_id": "UNSIDED",
                "description": "thing",
                "leftRef": {"id": "LEFT", "display": "left thing"},
                "rightRef": {"id": "RIGHT", "display": "right thing"},
            },
            {"_id": "LEFT", "description": "left thing"},  # no unsidedRef back
            {"_id": "RIGHT", "description": "right thing", "unsidedRef": {"id": "UNSIDED", "display": "thing"}},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("unsidedRef doesn't point back" in e for e in result.relationship_errors)

    def test_sided_cross_ref_is_ok(self) -> None:
        """Sided records cross-referencing each other via leftRef/rightRef is valid."""
        recs = _make_records(
            {
                "_id": "UNSIDED",
                "description": "thing",
                "leftRef": {"id": "LEFT", "display": "left thing"},
                "rightRef": {"id": "RIGHT", "display": "right thing"},
            },
            {
                "_id": "LEFT",
                "description": "left thing",
                "unsidedRef": {"id": "UNSIDED", "display": "thing"},
                "rightRef": {"id": "RIGHT", "display": "right thing"},
            },
            {
                "_id": "RIGHT",
                "description": "right thing",
                "unsidedRef": {"id": "UNSIDED", "display": "thing"},
                "leftRef": {"id": "LEFT", "display": "left thing"},
            },
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert result.ok

    def test_has_parts_bidirectional_ok(self) -> None:
        recs = _make_records(
            {"_id": "WHOLE", "description": "whole", "hasPartsRefs": [{"id": "PART", "display": "part"}]},
            {"_id": "PART", "description": "part", "partOfRef": {"id": "WHOLE", "display": "whole"}},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert result.ok

    def test_has_parts_missing_backref(self) -> None:
        recs = _make_records(
            {"_id": "WHOLE", "description": "whole", "hasPartsRefs": [{"id": "PART", "display": "part"}]},
            {"_id": "PART", "description": "part"},  # no partOfRef back
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("partOfRef doesn't point back" in e for e in result.relationship_errors)

    def test_self_ref_is_ok(self) -> None:
        """Self-references in containedByRef/partOfRef are common and should not be flagged."""
        recs = _make_records(
            {
                "_id": "A",
                "description": "alpha",
                "containedByRef": {"id": "A", "display": "alpha"},
                "partOfRef": {"id": "A", "display": "alpha"},
            },
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert result.ok

    def test_cycle_in_contained_by(self) -> None:
        recs = _make_records(
            {"_id": "A", "description": "alpha", "containedByRef": {"id": "B", "display": "beta"}},
            {"_id": "B", "description": "beta", "containedByRef": {"id": "C", "display": "gamma"}},
            {"_id": "C", "description": "gamma", "containedByRef": {"id": "A", "display": "alpha"}},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("Cycle" in e for e in result.relationship_errors)

    def test_list_ref_missing_target(self) -> None:
        recs = _make_records(
            {"_id": "A", "description": "alpha", "containsRefs": [{"id": "MISSING", "display": "gone"}]},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("not found" in e for e in result.relationship_errors)

    def test_list_ref_display_mismatch(self) -> None:
        recs = _make_records(
            {"_id": "A", "description": "alpha", "containsRefs": [{"id": "B", "display": "wrong"}]},
            {"_id": "B", "description": "beta"},
        )
        result = ValidationResult()
        validate_relationships(recs, result)
        assert not result.ok
        assert any("display" in e for e in result.relationship_errors)


# ---------------------------------------------------------------------------
# validate_anatomic_json (end-to-end with temp files)
# ---------------------------------------------------------------------------


class TestValidateAnatomicJson:
    def test_valid_file(self, tmp_path: Path) -> None:
        data = [
            {"_id": "A", "description": "alpha"},
            {"_id": "B", "description": "beta", "containedByRef": {"id": "A", "display": "alpha"}},
        ]
        p = tmp_path / "valid.json"
        p.write_text(json.dumps(data))
        result = validate_anatomic_json(p)
        assert result.ok

    def test_not_a_list(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}')
        result = validate_anatomic_json(p)
        assert not result.ok
        assert "<file>" in result.parse_errors

    def test_mixed_valid_and_invalid(self, tmp_path: Path) -> None:
        data = [
            {"_id": "A", "description": "alpha"},
            {"_id": "", "description": "empty id"},  # parse error
        ]
        p = tmp_path / "mixed.json"
        p.write_text(json.dumps(data))
        result = validate_anatomic_json(p)
        assert not result.ok
        assert result.parse_errors

    def test_warnings_collected(self, tmp_path: Path) -> None:
        data = [
            {"_id": "A", "description": "alpha", "snomedId": "12345"},
        ]
        p = tmp_path / "warn.json"
        p.write_text(json.dumps(data))
        result = validate_anatomic_json(p)
        assert result.ok  # warnings don't cause failure
        assert len(result.warnings) == 1
        assert "snomedId" in result.warnings[0]

    def test_summary_counts(self) -> None:
        r = ValidationResult()
        r.parse_errors["X"] = ["err1"]
        r.relationship_errors.append("rel1")
        r.warnings.append("warn1")
        assert r.total_errors == 2
        assert not r.ok
        assert "2 error" in r.summary()
