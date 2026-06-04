"""Readiness report: turn per-field quality scores into a pass/fail verdict.

Each finding contributes a 0-1 score per field (graduated, not pass/fail). The runner computes those
scores with the shared scoring helpers (`metadata_scoring.py`) plus the agreed rulings:
  - set fields are commission-sensitive: omitting a value is a mild ding, adding an unsupported one
    is the heavy penalty, scaled by count;
  - anatomic parent/child counts as close-enough; GAMUTS codes are out of scope;
  - null <-> sex-neutral and null <-> all_ages count as full credit.

This module owns the thresholds and the verdict only. Thresholds are defaults — tune in
`docs/metadata/enrichment/readiness-gates.md`.

Per-finding record shape:
    {
      "item_id": str, "name": str,
      "gates": {"execution_success": bool, "schema_valid": bool, "candidate_integrity": bool},
      "scores": {"<field>": float},      # 0-1 quality per field
      "notes": {"<field>": str},          # short label per field, for the failure-class rollup / detail
      "commission": {"<field>": int},     # # of unsupported additions (numerator of the commission rate)
      "proposed": {"<field>": int},       # # of values proposed (denominator of the commission rate)
    }

Self-test: PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_readiness
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

FIELD_FLOORS: dict[str, float] = {
    "entity_type": 0.90,
    "body_regions": 0.90,
    "anatomic_locations": 0.85,
    "subspecialties": 0.85,
    "applicable_modalities": 0.85,
    "index_codes": 0.85,
    "etiologies": 0.75,
    "expected_time_course": 0.75,
    "age_profile": 0.85,
    "sex_specificity": 0.95,
}
# Etiology is gated by its family-aware score + the malignancy tripwire below, NOT this blunt cap
# (a raw additions-rate over-counts defensible siblings / gold-lag — see the etiology rubric review).
COMMISSION_FIELDS = {"expected_time_course", "subspecialties", "applicable_modalities"}
COMMISSION_CAP = 0.05  # unsupported additions as a share of proposed values
MALIGNANCY_OVERCALL_CAP = 0  # the cardinal sin: asserting malignancy absent from gold must be ~0
FAILURE_CLASS_CAP = 0.10  # no single low-score note may recur in > this share of records
FAIL_THRESHOLD = 0.5  # a per-field score below this is a "miss" for failure-class bucketing
HARD_GATES = ("execution_success", "schema_valid", "candidate_integrity")


@dataclass
class ReadinessReport:
    n_records: int
    hard_gate_pass: bool
    hard_gate_failures: dict[str, int]
    field_scores: dict[str, float]
    field_floor_pass: dict[str, bool]
    commission_rates: dict[str, float]
    commission_pass: dict[str, bool]
    failure_class_breaches: dict[str, float]
    malignancy_overcalls: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.hard_gate_pass
            and all(self.field_floor_pass.values())
            and all(self.commission_pass.values())
            and not self.failure_class_breaches
            and self.malignancy_overcalls <= MALIGNANCY_OVERCALL_CAP
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "passed": self.passed,
                "n_records": self.n_records,
                "hard_gates": {"pass": self.hard_gate_pass, "failures": self.hard_gate_failures},
                "malignancy_overcalls": self.malignancy_overcalls,
                "field_scores": self.field_scores,
                "field_floor_pass": self.field_floor_pass,
                "commission_rates": self.commission_rates,
                "commission_pass": self.commission_pass,
                "failure_class_breaches": self.failure_class_breaches,
            },
            indent=2,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        lines = [f"# Readiness Report ({'PASS' if self.passed else 'FAIL'})", "", f"- records: {self.n_records}"]
        lines.append(f"- hard gates: {'pass' if self.hard_gate_pass else 'FAIL ' + str(self.hard_gate_failures)}")
        mal_ok = self.malignancy_overcalls <= MALIGNANCY_OVERCALL_CAP
        lines.append(f"- malignancy over-calls: {self.malignancy_overcalls} {'✓' if mal_ok else '✗ (cardinal sin)'}")
        lines += ["", "| field | score | floor | commission | ok |", "| --- | --- | --- | --- | --- |"]
        for fld in sorted(self.field_scores):
            comm = self.commission_rates.get(fld)
            comm_s = "-" if comm is None else f"{comm:.2f}"
            ok = self.field_floor_pass.get(fld, True) and self.commission_pass.get(fld, True)
            lines.append(
                f"| {fld} | {self.field_scores[fld]:.2f} | {FIELD_FLOORS.get(fld, 0.0):.2f} | {comm_s} | "
                f"{'✓' if ok else '✗'} |"
            )
        if self.failure_class_breaches:
            lines += [
                "",
                "**Recurring misses over cap:** "
                + ", ".join(f"{k} {v:.0%}" for k, v in sorted(self.failure_class_breaches.items())),
            ]
        return "\n".join(lines) + "\n"


def build_report(records: list[dict[str, Any]]) -> ReadinessReport:
    n = len(records)
    hard_failures: Counter[str] = Counter()
    score_sum: defaultdict[str, float] = defaultdict(float)
    score_count: defaultdict[str, int] = defaultdict(int)
    commission_hits: defaultdict[str, int] = defaultdict(int)
    proposed_total: defaultdict[str, int] = defaultdict(int)
    miss_notes: defaultdict[str, set[str]] = defaultdict(set)

    for r in records:
        for g in HARD_GATES:
            if not r.get("gates", {}).get(g, True):
                hard_failures[g] += 1
        for fld, score in r.get("scores", {}).items():
            score_sum[fld] += score
            score_count[fld] += 1
            if score < FAIL_THRESHOLD:
                miss_notes[r.get("notes", {}).get(fld, fld)].add(r.get("item_id", "?"))
        for fld, hits in r.get("commission", {}).items():
            commission_hits[fld] += hits
        for fld, count in r.get("proposed", {}).items():
            proposed_total[fld] += count

    field_scores = {f: score_sum[f] / score_count[f] for f in score_count}
    field_floor_pass = {f: field_scores[f] >= FIELD_FLOORS.get(f, 0.0) for f in field_scores}
    commission_rates = {
        f: (commission_hits[f] / proposed_total[f] if proposed_total.get(f) else 0.0)
        for f in COMMISSION_FIELDS
        if f in score_count
    }
    commission_pass = {f: rate <= COMMISSION_CAP for f, rate in commission_rates.items()}
    breaches = {note: len(items) / n for note, items in miss_notes.items() if n and len(items) / n > FAILURE_CLASS_CAP}
    malignancy_overcalls = sum(int(r.get("malignancy_overcalls", 0)) for r in records)

    return ReadinessReport(
        n_records=n,
        hard_gate_pass=not hard_failures,
        hard_gate_failures=dict(hard_failures),
        field_scores=field_scores,
        field_floor_pass=field_floor_pass,
        commission_rates=commission_rates,
        commission_pass=commission_pass,
        malignancy_overcalls=malignancy_overcalls,
        failure_class_breaches=breaches,
    )


def _selftest() -> None:
    def rec(scores: dict[str, float], **kw: object) -> dict[str, Any]:
        base = {
            "item_id": kw.get("item_id", "x"),
            "gates": {"execution_success": True, "schema_valid": True, "candidate_integrity": True},
            "scores": scores,
            "notes": kw.get("notes", {}),
            "commission": kw.get("commission", {}),
            "proposed": kw.get("proposed", {}),
            "malignancy_overcalls": kw.get("malignancy_overcalls", 0),
        }
        return base

    perfect = [rec({"entity_type": 1.0, "etiologies": 1.0}, item_id=f"g{i}") for i in range(10)]
    assert build_report(perfect).passed

    # graduated: a field averaging below its floor fails it
    weak = [rec({"entity_type": 1.0, "etiologies": 0.5}, item_id=f"w{i}") for i in range(10)]
    assert not build_report(weak).passed

    # hard-gate failure fails regardless of scores
    gate = [rec({"entity_type": 1.0}, item_id="z")]
    gate[0]["gates"]["execution_success"] = False
    assert not build_report(gate).passed

    # commission cap (still applies to subspecialties/modalities/time_course): unsupported additions trip it
    comm = [
        rec({"subspecialties": 0.9}, item_id=f"c{i}", commission={"subspecialties": 1}, proposed={"subspecialties": 2})
        for i in range(10)
    ]
    assert not build_report(comm).passed

    # etiology is NO LONGER gated by the blunt commission cap (only by score + malignancy tripwire)
    eti_comm = [
        rec({"etiologies": 0.9}, item_id=f"e{i}", commission={"etiologies": 1}, proposed={"etiologies": 2})
        for i in range(10)
    ]
    assert build_report(eti_comm).passed

    # malignancy tripwire: a single asserted-malignant over-call fails the run (cardinal sin)
    mal = [rec({"etiologies": 0.95}, item_id="m", malignancy_overcalls=1)]
    assert not build_report(mal).passed

    print("metadata_readiness self-test: PASS")


if __name__ == "__main__":
    _selftest()
