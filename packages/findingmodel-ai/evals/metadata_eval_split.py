"""Build the authority-aware reviewed-record split manifest.

Separates the two orthogonal axes the readiness process depends on:

- **allowed_use**: what a record may be used for.
  - `quality_gold`  — the 67 human-approved records (have an agreed-correct answer to score against).
  - `guidance`      — the 83 feedback records (signal for tuning, NEVER scored as gold).
- **split**: only quality_gold rows are split for measurement.
  - `dev`           — used during the iteration loop.
  - `heldout_test`  — frozen overfitting tripwire; touched only at candidate-ready milestones.
  - `excluded`      — guidance rows (not part of any quality split).

Writeback authority is a separate axis entirely: all 67 approved remain source-writeback authority
regardless of dev/heldout split (that is governed by the approved-output manifest + Gate A, not here).

Held-out is a COARSE tripwire — 67 is thin, so the fresh certification batch + component evals + the
35 gold fixtures carry the real weight. The dev/heldout fraction here is intentionally tunable.

Run: PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_eval_split
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
APPROVED = FIXTURES / "metadata_review_approved_outputs.json"
FEEDBACK = FIXTURES / "metadata_review_feedback_summary.json"
OUT = FIXTURES / "metadata_eval_split_manifest.json"

HELDOUT_FRACTION = 0.20  # tunable; coarse tripwire only
SEED = 20260602


def _first(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str) and value:
        return value
    return "none"


def _stratum(md: dict) -> dict[str, str]:
    return {
        "entity_type": _first(md.get("entity_type")),
        "subspecialty": _first(md.get("subspecialties")),
        "body_region": _first(md.get("body_regions")),
    }


def build() -> dict:
    approved = json.loads(APPROVED.read_text())["records"]
    feedback = json.loads(FEEDBACK.read_text())["records"]

    records: list[dict] = []

    # quality_gold: the 67 approved. Stratified dev/heldout split by entity_type.
    by_entity: dict[str, list[dict]] = {}
    for r in approved:
        md = r.get("metadata") or {}
        by_entity.setdefault(_first(md.get("entity_type")), []).append(r)

    rng = random.Random(SEED)
    heldout_ids: set[str] = set()
    for group in (g for _, g in sorted(by_entity.items())):
        ordered = sorted(group, key=lambda r: r["item_id"])
        n_heldout = round(len(ordered) * HELDOUT_FRACTION)
        for r in rng.sample(ordered, n_heldout):
            heldout_ids.add(r["item_id"])

    for r in approved:
        md = r.get("metadata") or {}
        records.append({
            "item_id": r["item_id"],
            "path": r["path"],
            "latest_status": "approved",
            "allowed_use": "quality_gold",
            "split": "heldout_test" if r["item_id"] in heldout_ids else "dev",
            "stratify": _stratum(md),
            "source_sha256": r.get("source_sha256"),
            "reviewed_payload_sha256": r.get("reviewed_payload_sha256"),
        })

    # guidance: the 83 feedback. Never gold; excluded from quality splits.
    for r in feedback:
        records.append({
            "item_id": r["id"],
            "path": r.get("path"),
            "latest_status": "feedback",
            "allowed_use": "guidance",
            "split": "excluded",
            "affected_fields": r.get("affected_fields"),
            "source_model_issue": r.get("source_model_issue", False),
        })

    counts = {
        "total": len(records),
        "by_allowed_use": dict(Counter(r["allowed_use"] for r in records)),
        "by_split": dict(Counter(r["split"] for r in records)),
        "heldout_by_entity_type": dict(
            Counter(r["stratify"]["entity_type"] for r in records if r["split"] == "heldout_test")
        ),
    }
    return {
        "version": 1,
        "seed": SEED,
        "heldout_fraction": HELDOUT_FRACTION,
        "policy": {
            "quality_gold": "67 approved records; the only rows scored as gold.",
            "guidance": "83 feedback records; tuning signal only, never scored as gold.",
            "writeback_authority": "separate axis — governed by the approved-output manifest + Gate A, not this split.",
            "heldout": "coarse overfitting tripwire; fresh certification batch carries the real weight.",
        },
        "counts": counts,
        "records": records,
    }


if __name__ == "__main__":
    manifest = build()
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(manifest["counts"], indent=2))
