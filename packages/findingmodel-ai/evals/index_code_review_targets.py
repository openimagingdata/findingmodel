"""Build reviewed index-code/anatomic-location targets from a pruned worksheet."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_WORKSHEET = Path(__file__).with_name("index_code_candidate_review_curated_2026-06-05.txt")
DEFAULT_APPROVED = Path(__file__).with_name("fixtures") / "metadata_review_approved_outputs.json"
DEFAULT_OUTPUT = Path(__file__).with_name("fixtures") / "index_code_review_targets.json"

INDEX_CODE_SYSTEMS = {
    "SNOMEDCT",
    "RADLEX",
    "LOINC",
    "GAMUTS",
    "GMTS",
    "radelement",
    "RADELEMENT",
    "CDES",
}
ANATOMIC_LOCATION_SYSTEM = "ANATOMICLOCATIONS"

KNOWN_NON_ANATOMIC_LOCATION_GAPS = {
    ("basal cistern effacement", "RADLEX", "RID9865"): (
        "Local anatomic_locations does not currently contain basal, basilar, "
        "perimesencephalic, subarachnoid, or generic cistern entries."
    ),
}

CODE_RE = re.compile(r"^  - (?P<system>[^:]+):(?P<code>[^|]+) \| (?P<display>.*?)(?: \[source\])?$")


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _parse_code(line: str) -> dict[str, str]:
    match = CODE_RE.match(line)
    if match is None:
        raise ValueError(f"Malformed candidate line: {line!r}")
    return {
        "system": match.group("system"),
        "code": match.group("code").strip(),
        "display": match.group("display").strip(),
    }


def _new_section(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "input": "",
        "synonyms": [],
        "index_codes": [],
        "anatomic_locations": [],
    }


def _update_section_metadata(section: dict[str, Any], line: str) -> bool:
    if line.startswith("Input: "):
        section["input"] = line.removeprefix("Input: ").strip()
        return True
    if line.startswith("Synonyms: "):
        section["synonyms"] = [
            synonym.strip() for synonym in line.removeprefix("Synonyms: ").split(";") if synonym.strip()
        ]
        return True
    return False


def _candidate_mode(line: str) -> str | None:
    if line == "Index code candidates:":
        return "index_codes"
    if line == "Anatomic location candidates:":
        return "anatomic_locations"
    return None


def _append_candidate(section: dict[str, Any], mode: str | None, line: str) -> None:
    if not line.startswith("  - "):
        return
    if mode is None:
        raise ValueError(f"Candidate line before a field heading in {section['name']!r}: {line!r}")
    section[mode].append(_parse_code(line))


def parse_worksheet(path: Path) -> list[dict[str, Any]]:
    """Parse the curated text worksheet into section records."""

    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    mode: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current is not None:
                sections.append(current)
            current = _new_section(line.removeprefix("## ").strip())
            mode = None
            continue
        if current is None:
            continue
        if _update_section_metadata(current, line):
            continue
        new_mode = _candidate_mode(line)
        if new_mode is not None:
            mode = new_mode
            continue
        _append_candidate(current, mode, line)
    if current is not None:
        sections.append(current)
    return sections


def _dedupe_codes(codes: list[dict[str, str]], *, section_name: str, field: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped = []
    for code in codes:
        key = f"{code['system']}:{code['code']}"
        if key in seen:
            raise ValueError(f"Duplicate {field} code in {section_name!r}: {key}")
        seen.add(key)
        deduped.append(code)
    return deduped


def approved_records_by_name(path: Path) -> dict[str, dict[str, Any]]:
    """Return approved-output records keyed by normalized display name."""

    approved = json.loads(path.read_text(encoding="utf-8"))
    records = approved["records"]
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        name = str(record["name"])
        key = _norm(name)
        if key in by_name:
            raise ValueError(f"Duplicate approved-output record name: {name!r}")
        by_name[key] = record
    return by_name


def _validate_index_codes(section: dict[str, Any]) -> list[dict[str, str]]:
    codes = _dedupe_codes(section["index_codes"], section_name=section["name"], field="index_codes")
    for code in codes:
        if code["system"] not in INDEX_CODE_SYSTEMS:
            raise ValueError(f"Unsupported index-code system in {section['name']!r}: {code['system']}")
    return codes


def _validate_anatomic_locations(section: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept = []
    skipped = []
    for code in _dedupe_codes(section["anatomic_locations"], section_name=section["name"], field="anatomic_locations"):
        if code["system"] == ANATOMIC_LOCATION_SYSTEM:
            kept.append(code)
            continue
        gap_key = (section["name"], code["system"], code["code"])
        if gap_key not in KNOWN_NON_ANATOMIC_LOCATION_GAPS:
            raise ValueError(f"Unsupported anatomic-location system in {section['name']!r}: {code['system']}")
        skipped.append(code)
    return kept, skipped


def build_targets(worksheet_path: Path, approved_path: Path) -> dict[str, Any]:
    """Build machine-readable reviewed targets from the pruned worksheet."""

    sections = parse_worksheet(worksheet_path)
    approved_by_name = approved_records_by_name(approved_path)
    records = []
    skipped_non_anatomic = []
    for section in sections:
        approved = approved_by_name.get(_norm(section["name"]))
        if approved is None:
            raise ValueError(f"Worksheet section does not map to approved output: {section['name']!r}")
        index_codes = _validate_index_codes(section)
        anatomic_locations, skipped = _validate_anatomic_locations(section)
        for code in skipped:
            skipped_non_anatomic.append({
                "item_id": approved["item_id"],
                "name": approved["name"],
                "field": "anatomic_locations",
                "candidate": code,
                "reason": KNOWN_NON_ANATOMIC_LOCATION_GAPS[section["name"], code["system"], code["code"]],
            })
        records.append({
            "item_id": approved["item_id"],
            "name": approved["name"],
            "index_codes": index_codes,
            "anatomic_locations": anatomic_locations,
        })

    approved_ids = {record["item_id"] for record in approved_by_name.values()}
    target_ids = {record["item_id"] for record in records}
    if len(records) != len(approved_ids) or target_ids != approved_ids:
        missing = sorted(approved_ids - target_ids)
        extra = sorted(target_ids - approved_ids)
        raise ValueError(f"Worksheet/approved-output mismatch: missing={missing}, extra={extra}")

    index_system_counts = Counter(code["system"] for record in records for code in record["index_codes"])
    anatomy_system_counts = Counter(code["system"] for record in records for code in record["anatomic_locations"])
    empty_index = sorted(record["item_id"] for record in records if not record["index_codes"])
    empty_anatomy = sorted(record["item_id"] for record in records if not record["anatomic_locations"])
    return {
        "version": 1,
        "source_worksheet": str(worksheet_path),
        "source_approved_outputs": str(approved_path),
        "usage": {
            "authority": "human_pruned_code_anatomy_targets",
            "overlay_policy": (
                "Use as a scoring-time overlay for index_codes and anatomic_locations only; "
                "do not mutate metadata_review_approved_outputs.json from this fixture."
            ),
        },
        "counts": {
            "records": len(records),
            "index_code_targets": sum(len(record["index_codes"]) for record in records),
            "anatomic_location_targets": sum(len(record["anatomic_locations"]) for record in records),
            "empty_index_code_records": len(empty_index),
            "empty_anatomic_location_records": len(empty_anatomy),
            "skipped_non_anatomic_location_candidates": len(skipped_non_anatomic),
        },
        "target_fields": ["index_codes", "anatomic_locations"],
        "system_counts": {
            "index_codes": dict(sorted(index_system_counts.items())),
            "anatomic_locations": dict(sorted(anatomy_system_counts.items())),
        },
        "empty_targets": {
            "index_codes": empty_index,
            "anatomic_locations": empty_anatomy,
        },
        "known_gaps": [
            {
                "item_id": "basal_cistern_effacement",
                "field": "anatomic_locations",
                "gap": "missing_anatomic_location_cistern_concept",
                "decision": (
                    "Exclude RADLEX:RID9865 from anatomic_locations for now; do not invent an "
                    "ANATOMICLOCATIONS basal cistern code."
                ),
            }
        ],
        "skipped_candidates": skipped_non_anatomic,
        "records": sorted(records, key=lambda record: record["item_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    parser.add_argument("--approved", type=Path, default=DEFAULT_APPROVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    targets = build_targets(args.worksheet, args.approved)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(targets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {targets['counts']['records']} reviewed code/anatomy targets to {args.output}")


if __name__ == "__main__":
    main()
