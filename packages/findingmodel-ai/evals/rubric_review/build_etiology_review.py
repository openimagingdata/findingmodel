"""Generate the etiology rubric-review CSVs from readiness run snapshots.

Aggregates every etiology disagreement (over-call / omission / mixed) across the saved run
snapshots, dedupes per finding with a frequency count, joins my first-pass verdict, and writes two
CSVs for human review:

- etiology_disagreements.csv : one row per finding where agent != curator, with my proposed verdict
  + empty columns for the reviewer to fill (YOUR_VERDICT, CORRECT_ETIOLOGY, YOUR_SEVERITY, NOTES).
- etiology_family_map.csv    : the bounded parent/child + sibling structure to settle once.

Run: PYTHONPATH=packages/findingmodel-ai uv run python -m evals.rubric_review.build_etiology_review
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

# Snapshots captured during validation (3 pre-override + 3 post-override = 6 runs).
SNAPSHOTS = [f"/tmp/tc_run{i}.json" for i in (1, 2, 3)] + [f"/tmp/tc2_run{i}.json" for i in (1, 2, 3)]
OUT_DIR = Path(__file__).parent

# My first-pass verdict per finding: (verdict, severity, rationale).
# verdict vocabulary: family-parentchild | family-sibling | wrong-trio | wrong-off | under-call |
#                     curator-overcall | defensible | unsure
# severity (by the reviewer's principle — family diff < completely-off): small | moderate | large | none | ?
VERDICTS: dict[str, tuple[str, str, str]] = {
    # --- over-call only (agent added, never dropped) ---
    "abnormal sternomanubrial synchondrosis": (
        "unsure",
        "?",
        "could be acquired (degenerative/inflammatory), not clearly formation; dev+cong may over-reach",
    ),
    "large orbit": (
        "unsure",
        "?",
        "developmental vs curator congenital (trio); but large orbit can be acquired (NF1/mass)",
    ),
    "Mastectomy Breast Implant": (
        "family-sibling",
        "small",
        "iatrogenic:device vs post-operative; both true, device arguably better",
    ),
    "unilateral hilar enlargement": (
        "wrong-off",
        "large",
        "malignant+inflammatory+vascular differential pasted on a descriptive finding",
    ),
    "vertebral coronal cleft": ("family-sibling", "small", "developmental vs congenital; genuine formation variant"),
    "arterial tortuosity": (
        "wrong-off",
        "large",
        "acquired/degenerative not congenital (reviewer-confirmed); correct=degenerative, curator under-called",
    ),
    "dental periapical opacity": (
        "unsure",
        "?",
        "opacity often condensing osteitis/osseous dysplasia; inflammatory plausible but uncertain",
    ),
    "breast soft tissue lesion": (
        "unsure",
        "?",
        "prompt says unspecified lesion -> benign+malignant; curator blank; whose call?",
    ),
    "vertebral compression fracture": (
        "defensible",
        "small",
        "osteoporotic/insufficiency = degenerative is defensible; curator focused traumatic",
    ),
    "breast calcification cluster": (
        "unsure",
        "?",
        "clusters can flag malignancy; neoplastic:potential defensible or over-reach?",
    ),
    # --- mixed (added AND dropped across runs) ---
    "acute lung injury and ards in children": (
        "under-call",
        "moderate",
        "missed main causes (infectious/toxic/inflammatory); occasional wrong-add trauma",
    ),
    "arterial rupture": (
        "family-parentchild",
        "small",
        "gave parent 'vascular' instead of child 'vascular:aneurysmal'",
    ),
    "early intrauterine pregnancy": (
        "wrong-off",
        "large",
        "normal pregnancy not a malformation; should be normal-variant, substituted dev+cong",
    ),
    "fetal chest mass": ("under-call", "large", "missed benign+malignant+formation; only added potential"),
    "focal shadowing pancreatic lesion": (
        "family-sibling",
        "small",
        "malignant vs curator potential; same neoplastic family, more confident",
    ),
    "increased resistance index of renal transplant": (
        "unsure",
        "?",
        "parent vascular vs specific children + missed iatrogenic; should a measurement carry etiology?",
    ),
    "lung sutures": ("family-sibling", "small", "iatrogenic:post-operative vs device; both true"),
    "omega sella": ("wrong-trio", "moderate", "normal variant; should be normal-variant not dev+cong"),
    "cardiac valve thickening": (
        "under-call",
        "moderate",
        "missed degenerative/inflammatory commonly; added autoimmune (rheumatic, defensible)",
    ),
    "sternal fixation": ("family-sibling", "small", "iatrogenic:post-operative vs device; both true"),
    # --- omission only (agent dropped curator labels) ---
    "abnormal right paratracheal stripe": (
        "curator-overcall",
        "none",
        "descriptive finding; curator pasted infectious+malignant; agent right to omit (mirror of hilar enlargement)",
    ),
    "AO Spine Subaxial Cervical Spine injury Classification": (
        "unsure",
        "?",
        "it's a classification (no etiology?) vs curator traumatic:acute; should classifications carry etiology?",
    ),
    "infratentorial intracranial tumor in a child": (
        "under-call",
        "moderate",
        "a tumor should get neoplastic benign+malignant; agent under-called",
    ),
    "large vascular grooves of skull": (
        "family-sibling",
        "small",
        "missed developmental (kept normal-variant); trio sibling",
    ),
    "short thin distal phalanx of thumb": (
        "family-sibling",
        "small",
        "captured congenital+developmental, missed normal-variant; trio",
    ),
    "fracture": (
        "family-sibling",
        "small",
        "kept traumatic:acute, missed traumatic:sequela; sibling (sequela is a stretch)",
    ),
    "air in esophagus": ("unsure", "?", "curator normal-variant; air can be normal or pathologic; agent omitted"),
    "Pneumonia": (
        "family-parentchild",
        "small",
        "kept inflammatory:infectious, dropped parent inflammatory; correct per no-parent+child rule",
    ),
}

FAMILY_MAP_ROWS = [
    (
        "vascular",
        "parent/child",
        "vascular -> vascular:ischemic/hemorrhagic/thrombotic/aneurysmal",
        "parent given when a child is present (or vice versa) = MATCH, no penalty",
    ),
    (
        "vascular",
        "sibling",
        "vascular:ischemic vs :hemorrhagic vs :thrombotic vs :aneurysmal",
        "wrong sibling = PARTIAL (~0.5)",
    ),
    (
        "neoplastic",
        "sibling",
        "neoplastic:benign vs :malignant vs :metastatic vs :potential",
        "wrong sibling = PARTIAL; benign<->malignant a bigger miss than potential<->malignant?",
    ),
    ("traumatic", "sibling", "traumatic:acute vs :sequela", "wrong sibling = PARTIAL"),
    ("inflammatory", "parent/child", "inflammatory -> inflammatory:infectious", "parent/child = MATCH"),
    (
        "iatrogenic",
        "sibling",
        "iatrogenic:device/post-operative/post-radiation/medication-related",
        "device<->post-operative near-MATCH; others PARTIAL",
    ),
    (
        "formation trio",
        "sibling cluster",
        "congenital ~ developmental ~ normal-variant",
        "PARTIAL — but is normal-variant (not-a-disease) far from congenital/developmental (anomaly)? YOUR CALL",
    ),
    ("cross-family", "unrelated", "e.g. inflammatory vs neoplastic vs vascular", "full error, no credit"),
]


def _fmt(counter: dict[str, int]) -> str:
    return "; ".join(f"{k}(x{v})" for k, v in sorted(counter.items(), key=lambda kv: -kv[1]))


def build() -> None:
    runs = [json.loads(Path(p).read_text()) for p in SNAPSHOTS]
    agg: dict[str, dict] = defaultdict(
        lambda: {"added": defaultdict(int), "missed": defaultdict(int), "curator": [], "desc": "", "runs": 0}
    )
    for r in runs:
        for rec in r:
            d = rec.get("detail", {}).get("etiologies")
            if not d:
                continue
            a = {str(x) for x in (d["proposed"] or [])}
            g = {str(x) for x in (d["gold"] or [])}
            if a == g:
                continue
            e = agg[rec["name"]]
            e["curator"] = sorted(g)
            e["desc"] = (rec.get("description") or "").strip().replace("\n", " ")[:200]
            e["runs"] += 1
            for x in a - g:
                e["added"][x] += 1
            for x in g - a:
                e["missed"][x] += 1

    rows = []
    for name, e in sorted(agg.items(), key=lambda kv: (-kv[1]["runs"], kv[0].lower())):
        added, missed = e["added"], e["missed"]
        kind = "mixed" if added and missed else ("over-call" if added else "omission")
        verdict, severity, why = VERDICTS.get(name, ("(needs review)", "?", ""))
        rows.append({
            "finding": name,
            "type": kind,
            "freq": f"{e['runs']}/6",
            "curator_had": "; ".join(e["curator"]) or "(none)",
            "agent_added": _fmt(added) or "-",
            "agent_missed": _fmt(missed) or "-",
            "my_verdict": verdict,
            "my_severity": severity,
            "my_rationale": why,
            "YOUR_VERDICT": "",
            "CORRECT_ETIOLOGY": "",
            "YOUR_SEVERITY": "",
            "NOTES": "",
            "description": e["desc"],
        })

    cols = [
        "finding",
        "type",
        "freq",
        "curator_had",
        "agent_added",
        "agent_missed",
        "my_verdict",
        "my_severity",
        "my_rationale",
        "YOUR_VERDICT",
        "CORRECT_ETIOLOGY",
        "YOUR_SEVERITY",
        "NOTES",
        "description",
    ]
    out1 = OUT_DIR / "etiology_disagreements.csv"
    with out1.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    out2 = OUT_DIR / "etiology_family_map.csv"
    with out2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "relationship", "members", "proposed_rubric_treatment", "YOUR_CALL", "NOTES"])
        for fam, rel, mem, treat in FAMILY_MAP_ROWS:
            w.writerow([fam, rel, mem, treat, "", ""])

    print(f"wrote {out1}  ({len(rows)} disagreements)")
    print(f"wrote {out2}  ({len(FAMILY_MAP_ROWS)} family rules)")
    n_unsure = sum(1 for r in rows if r["my_verdict"] in ("unsure", "(needs review)"))
    print(f"  rows needing your ruling (unsure): {n_unsure}")


if __name__ == "__main__":
    build()
