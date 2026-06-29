"""Real readiness run: enrich dev findings from scratch, score vs the approved answer, report.

End-to-end loop on real findings: pick dev gold findings -> blank their metadata -> run the pinned
AI -> score each field against the human-approved answer -> feed the scores to the readiness
scorecard (metadata_readiness).

Scoring follows the agreed rulings:
  - set fields use the commission-sensitive scorer (omission mild, unsupported addition heavy, scaled);
  - anatomic parent/child counts as close-enough (via the anatomy hierarchy);
  - GAMUTS codes are excluded from grading (the search doesn't produce them; they're carried forward);
  - null <-> sex-neutral and null <-> all_ages are full credit.

Pinned + fail-closed: the env override forces the single dated snapshot (no fallback).

Run: PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_readiness_run --limit 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import logfire

# The agent-tag key is case-sensitive and must stay lowercase to match the AgentTag.
PINNED_MODEL = "openai:gpt-5.4-mini-2026-03-17"
os.environ.setdefault("AGENT_MODEL_OVERRIDES__metadata_assign", PINNED_MODEL)

from findingmodel import FindingModelFull
from findingmodel_ai.metadata.assignment import (
    STRUCTURED_METADATA_FIELDS,
    assign_metadata,
    carry_forward_index_codes,
)
from findingmodel_ai.observability import ensure_logfire_configured

from evals.metadata_readiness import build_report
from evals.metadata_scoring import (
    count_malignancy_overcalls,
    score_commission_sensitive_set_similarity,
    score_etiologies,
)

FIXTURES = Path(__file__).parent / "fixtures"
DATA_REPO = Path("/Users/talkasab/repos/findingmodels-metadata")

SET_FIELDS = {"body_regions", "subspecialties", "applicable_modalities", "etiologies"}
DURATION_ORDER = ["hours", "days", "weeks", "months", "years", "permanent"]
ENTITY_PARTIAL = {frozenset({"finding", "diagnosis"}): 0.65, frozenset({"measurement", "assessment"}): 0.5}

# Reviewer-approved etiology gold corrections, applied as a scoring-time overlay (canonical
# approved-outputs fixture stays untouched; writeback is a separate sign-off-gated step).
_CORRECTIONS = json.loads((FIXTURES / "etiology_gold_corrections.json").read_text())["corrections"]
ETIOLOGY_GOLD_OVERRIDES = {item_id: c["etiologies"] for item_id, c in _CORRECTIONS.items()}


def _load_code_anatomy_target_overrides(path: Path) -> dict[str, dict[str, Any]]:
    """Return reviewed code/anatomy targets keyed by item id."""

    if not path.exists():
        return {}
    target_data = json.loads(path.read_text(encoding="utf-8"))
    overrides = {}
    for record in target_data["records"]:
        overrides[record["item_id"]] = {
            "index_codes": record["index_codes"],
            "anatomic_locations": record["anatomic_locations"],
        }
    return overrides


CODE_ANATOMY_GOLD_OVERRIDES = _load_code_anatomy_target_overrides(FIXTURES / "index_code_review_targets.json")

_ANATOMY_IDX: Any = None


def _anatomy_idx() -> Any:
    global _ANATOMY_IDX
    if _ANATOMY_IDX is None:
        from anatomic_locations import AnatomicLocationIndex

        _ANATOMY_IDX = AnatomicLocationIndex()
    return _ANATOMY_IDX


def _resolve(path_str: str) -> Path | None:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    cand = DATA_REPO / path_str
    return cand if cand.exists() else None


def _codes(value: Any, *, drop_gamuts: bool = False) -> list[dict[str, Any]]:
    out = []
    for v in value or []:
        if isinstance(v, dict):
            if drop_gamuts and str(v.get("system", "")).upper() in {"GAMUTS", "GMTS"}:
                continue
            out.append(v)
    return out


def _code_key(c: dict[str, Any]) -> str:
    return f"{c.get('system')}:{c.get('code')}"


def _score_codes(actual: set[str], gold: set[str]) -> float:
    """Code-set score honoring the ruling: proposing nothing floors at 0.3; a wrong addition is worse.

    - exact (both empty, or perfect overlap) -> 1.0
    - clean omission/partial (no wrong additions) -> recall, floored at 0.3 (nothing isn't catastrophic)
    - any wrong addition -> recall minus a heavy per-extra penalty, free to fall below 0.3 toward 0.
    """
    if not actual and not gold:
        return 1.0
    extra = len(actual - gold)
    if not gold:  # gold wanted none; anything proposed is a wrong addition
        return round(max(0.0, 0.3 - 0.25 * extra), 4)
    recall = len(actual & gold) / len(gold)
    if extra == 0:
        return round(max(recall, 0.3), 4)
    return round(max(0.0, recall - 0.25 * extra), 4)


def _set_note(actual: set[str], gold: set[str]) -> str:
    if actual == gold:
        return "exact"
    parts = []
    if actual - gold:
        parts.append(f"+{len(actual - gold)} extra")
    if gold - actual:
        parts.append(f"-{len(gold - actual)} missing")
    return ", ".join(parts) or "exact"


_ANCESTOR_CACHE: dict[str, set[str]] = {}


def _ancestors(rid: str) -> set[str]:
    if rid not in _ANCESTOR_CACHE:
        try:
            _ANCESTOR_CACHE[rid] = {loc.id for loc in _anatomy_idx().get_containment_ancestors(rid)}
        except Exception:
            _ANCESTOR_CACHE[rid] = set()
    return _ANCESTOR_CACHE[rid]


def _anatomy_related(a_rid: str, g_rid: str) -> bool:
    """Same code or a parent/child (containment) relationship counts as close-enough."""
    return a_rid == g_rid or g_rid in _ancestors(a_rid) or a_rid in _ancestors(g_rid)


def _score_time_course(proposed: Any, gold: Any) -> tuple[float, str, int]:
    if not proposed and not gold:
        return 1.0, "exact", 0
    if gold and not proposed:
        return 0.4, "missing", 0
    if proposed and not gold:
        return 0.3, "unsupported addition", 1
    pdur, gdur = proposed.get("duration"), gold.get("duration")
    pmod, gmod = set(proposed.get("modifiers") or []), set(gold.get("modifiers") or [])
    if pdur == gdur and pmod == gmod:
        return 1.0, "exact", 0
    if pdur == gdur:
        return 0.8, "modifier diff", 0
    if (
        pdur in DURATION_ORDER
        and gdur in DURATION_ORDER
        and abs(DURATION_ORDER.index(pdur) - DURATION_ORDER.index(gdur)) == 1
    ):
        return 0.7, "adjacent duration", 0
    return 0.4, "duration off", 0


def _score_anatomy(proposed: Any, gold: Any) -> tuple[float, str, int]:
    a_rids = [str(c.get("code")) for c in _codes(proposed)]
    g_rids = [str(c.get("code")) for c in _codes(gold)]
    if not a_rids and not g_rids:
        return 1.0, "exact", 0
    matched_gold = sum(1 for g in g_rids if any(_anatomy_related(a, g) for a in a_rids))
    matched_actual = sum(1 for a in a_rids if any(_anatomy_related(a, g) for g in g_rids))
    recall = matched_gold / len(g_rids) if g_rids else 1.0
    precision = matched_actual / len(a_rids) if a_rids else 1.0
    score = (0.7 * recall) + (0.3 * precision)
    note = (
        "exact" if matched_gold == len(g_rids) and matched_actual == len(a_rids) else f"R{recall:.0%}/P{precision:.0%}"
    )
    return round(score, 4), note, len(a_rids) - matched_actual


def _is_all_ages(v: Any) -> bool:
    return v is None or (isinstance(v, dict) and v.get("applicability") == "all_ages" and not v.get("more_common_in"))


def _score_age_profile(proposed: Any, gold: Any) -> tuple[float, str, int]:
    if proposed == gold or (_is_all_ages(proposed) and _is_all_ages(gold)):
        return 1.0, "exact", 0
    return 0.5, "age differs", 0


def _score_sex(proposed: Any, gold: Any) -> tuple[float, str, int]:
    def norm(v: Any) -> Any:
        return "sex-neutral" if v in (None, "sex-neutral") else v

    return (1.0, "exact", 0) if norm(proposed) == norm(gold) else (0.0, f"{proposed}!={gold}", 0)


def score_field(field: str, proposed: Any, gold: Any) -> tuple[float, str, int]:
    """Return (score 0-1, short note, # unsupported additions) for one field."""
    if field == "entity_type":
        if (proposed or None) == (gold or None):
            return 1.0, "exact", 0
        partial = ENTITY_PARTIAL.get(frozenset({str(proposed), str(gold)}), 0.0)
        return partial, f"{proposed}!={gold}", 0

    if field == "sex_specificity":
        return _score_sex(proposed, gold)

    if field == "age_profile":
        return _score_age_profile(proposed, gold)

    if field == "expected_time_course":
        return _score_time_course(proposed, gold)

    if field == "index_codes":
        a = {_code_key(c) for c in _codes(proposed, drop_gamuts=True)}
        g = {_code_key(c) for c in _codes(gold, drop_gamuts=True)}
        return _score_codes(a, g), _set_note(a, g), len(a - g)

    if field == "anatomic_locations":
        return _score_anatomy(proposed, gold)

    if field == "etiologies":
        # family-aware + clinically-asymmetric: malignancy over-call heavy, under-call light,
        # parent/child & dev<->cong free, normal-variant swap minor, descriptive-differential heavy.
        a = {str(v) for v in (proposed or [])}
        g = {str(v) for v in (gold or [])}
        score, hard_overcalls = score_etiologies(a, g)
        return score, _set_note(a, g), hard_overcalls

    # remaining set fields: commission-sensitive (omission mild, unsupported addition heavy)
    a = {str(v) for v in (proposed or [])}
    g = {str(v) for v in (gold or [])}
    return score_commission_sensitive_set_similarity(a, g), _set_note(a, g), len(a - g)


def reviewed_gold_for_item(item: dict[str, Any]) -> dict[str, Any]:
    """Apply reviewed scoring overlays to one approved-output metadata record."""

    item_id = item["item_id"]
    gold = dict(item["metadata"])
    if item_id in ETIOLOGY_GOLD_OVERRIDES:  # reviewer-approved gold correction
        gold["etiologies"] = ETIOLOGY_GOLD_OVERRIDES[item_id]
    if item_id in CODE_ANATOMY_GOLD_OVERRIDES:
        gold.update(CODE_ANATOMY_GOLD_OVERRIDES[item_id])
    return gold


def _blank(fm: FindingModelFull) -> FindingModelFull:
    data = fm.model_dump()
    for f in (*STRUCTURED_METADATA_FIELDS, "anatomic_locations"):
        data[f] = None
    data["index_codes"] = carry_forward_index_codes(fm) or None
    return FindingModelFull.model_validate(data)


async def run_one(item: dict[str, Any]) -> dict[str, Any] | None:
    item_id = item["item_id"]
    name = item.get("name", item_id)
    with logfire.span("metadata_readiness.item {item_id}", item_id=item_id, finding_name=name):
        path = _resolve(item["path"])
        if path is None:
            logfire.error("Metadata readiness source path missing {item_id} {path}", item_id=item_id, path=item["path"])
            return None
        fm = FindingModelFull.model_validate_json(path.read_text(encoding="utf-8"))
        gold = reviewed_gold_for_item(item)
        try:
            result = await assign_metadata(_blank(fm))
        except Exception:
            logfire.exception("Metadata readiness assignment failed {item_id}", item_id=item_id)
            return {
                "item_id": item_id,
                "name": name,
                "gates": {"execution_success": False, "schema_valid": False, "candidate_integrity": True},
                "scores": {},
                "notes": {},
                "commission": {},
                "proposed": {},
                "detail": {},
            }
    out_json = result.model.model_dump(mode="json")
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}
    commission: dict[str, int] = {}
    proposed_counts: dict[str, int] = {}
    detail: dict[str, dict[str, Any]] = {}
    for f in (*STRUCTURED_METADATA_FIELDS, "index_codes", "anatomic_locations"):
        proposed = out_json.get(f)
        if f not in gold and proposed is None:
            continue
        score, note, extras = score_field(f, proposed, gold.get(f))
        scores[f] = score
        notes[f] = note
        commission[f] = extras
        proposed_counts[f] = len(proposed) if isinstance(proposed, list) else (1 if proposed else 0)
        detail[f] = {"score": round(score, 2), "note": note, "proposed": proposed, "gold": gold.get(f)}
    malignancy_overcalls = count_malignancy_overcalls(out_json.get("etiologies"), gold.get("etiologies"))
    logfire.info(
        "Metadata readiness item scored {item_id} {score_count} {malignancy_overcalls}",
        item_id=item_id,
        score_count=len(scores),
        malignancy_overcalls=malignancy_overcalls,
    )
    return {
        "item_id": item_id,
        "name": name,
        "description": item.get("description", ""),
        "gates": {"execution_success": True, "schema_valid": True, "candidate_integrity": True},
        "scores": scores,
        "notes": notes,
        "commission": commission,
        "proposed": proposed_counts,
        "malignancy_overcalls": malignancy_overcalls,
        "detail": detail,
    }


async def main(
    limit: int,
    split: str = "dev",
    *,
    logfire_console: bool = False,
    send_to_logfire: bool | None = None,
) -> None:
    ensure_logfire_configured(console=logfire_console, send_to_logfire=send_to_logfire)
    manifest = json.loads((FIXTURES / "metadata_eval_split_manifest.json").read_text())
    ids = {r["item_id"] for r in manifest["records"] if r["split"] == split}
    approved = json.loads((FIXTURES / "metadata_review_approved_outputs.json").read_text())["records"]
    items = [r for r in approved if r["item_id"] in ids][:limit]
    print(f"running {len(items)} {split} findings on {PINNED_MODEL} ...")

    with logfire.span(
        "metadata_readiness.run {split}",
        split=split,
        limit=limit,
        item_count=len(items),
        pinned_model=PINNED_MODEL,
    ):
        results = [r for r in await asyncio.gather(*(run_one(item) for item in items)) if r]
    report = build_report(results)
    (FIXTURES.parent / "metadata_readiness_run_report.json").write_text(report.to_json() + "\n")
    (FIXTURES.parent / "metadata_readiness_run_details.json").write_text(json.dumps(results, indent=2) + "\n")

    md = ["# Readiness run detail — AI proposal vs. human-approved answer", ""]
    for r in results:
        md += [
            f"## {r['name']}",
            "",
            "| field | score | note | AI proposed | approved (gold) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for f, d in r.get("detail", {}).items():
            md.append(
                f"| {f} | {d['score']} | {d['note']} | `{json.dumps(d['proposed'])}` | `{json.dumps(d['gold'])}` |"
            )
        md.append("")
    (FIXTURES.parent / "metadata_readiness_run_details.md").write_text("\n".join(md) + "\n")

    print(report.to_markdown())
    print("(report -> evals/metadata_readiness_run_report.json; detail -> evals/metadata_readiness_run_details.md)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--split", default="dev", help="which split to run: dev or heldout_test")
    ap.add_argument("--logfire-console", action="store_true")
    ap.add_argument("--no-send-to-logfire", action="store_false", dest="send_to_logfire")
    ap.set_defaults(send_to_logfire=None)
    args = ap.parse_args()
    asyncio.run(
        main(
            args.limit,
            args.split,
            logfire_console=args.logfire_console,
            send_to_logfire=args.send_to_logfire,
        )
    )
