"""Compare generated data-repo source diffs with the human review register."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_DATA_REPO = Path("/Users/talkasab/repos/findingmodels-metadata")
DEFAULT_REGISTER = Path(__file__).with_name("fixtures") / "metadata_review_evidence_register.json"
DEFAULT_OUTPUT = Path(__file__).with_name("fixtures") / "metadata_review_source_overlap.json"


def modified_definition_paths(data_repo: Path) -> list[str]:
    """Return modified `defs/*.fm.json` paths from the data repo."""

    output = subprocess.check_output(
        ["git", "-C", str(data_repo), "status", "--short", "defs"],
        text=True,
    )
    paths = []
    for line in output.splitlines():
        status = line[:2]
        path = line[3:]
        if status.strip() == "M" and path.startswith("defs/") and path.endswith(".fm.json"):
            paths.append(path)
    return sorted(paths)


def git_head(data_repo: Path) -> str | None:
    """Return the data repo HEAD used for source-overlap capture."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(data_repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def load_register(register_path: Path) -> dict[str, dict[str, Any]]:
    """Index review register records by data-repo source path."""

    register = json.loads(register_path.read_text(encoding="utf-8"))
    latest: dict[str, dict[str, Any]] = {}
    for record in register["records"]:
        path = record["path"]
        current = latest.get(path)
        if current is None or (record["human_review"].get("updated_at") or "") >= (
            current["human_review"].get("updated_at") or ""
        ):
            latest[path] = record
    return latest


def build_overlap(data_repo: Path, register_path: Path) -> dict[str, Any]:
    """Build a source-overlap report for Gate A."""

    register_by_path = load_register(register_path)
    records = []
    for path in modified_definition_paths(data_repo):
        review_record = register_by_path.get(path)
        if review_record is None:
            review_status = "not_in_register"
            review_id = None
        else:
            review_status = review_record["human_review"]["status"]
            review_id = review_record["id"]
        records.append({
            "path": path,
            "item_id": Path(path).name.removesuffix(".fm.json"),
            "review_status": review_status,
            "review_id": review_id,
        })

    counts = Counter(record["review_status"] for record in records)
    return {
        "version": 1,
        "data_repo": str(data_repo),
        "data_repo_head": git_head(data_repo),
        "register": str(register_path),
        "gate_a_policy": (
            "This file audits generated source diffs against latest human review status. "
            "Approved overlap may be preserved for later approved-output application; feedback "
            "and unreviewed overlap must not be treated as approved source changes."
        ),
        "counts": {
            "total_modified_defs": len(records),
            "approved": counts["approved"],
            "feedback": counts["feedback"],
            "not_in_register": counts["not_in_register"],
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-repo", type=Path, default=DEFAULT_DATA_REPO)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    overlap = build_overlap(args.data_repo, args.register)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overlap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} with {overlap['counts']['total_modified_defs']} modified defs "
        f"({overlap['counts']['approved']} approved, {overlap['counts']['feedback']} feedback, "
        f"{overlap['counts']['not_in_register']} not in register)"
    )


if __name__ == "__main__":
    main()
