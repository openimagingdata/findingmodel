# Holistic etiology/tempo prompt improvement plan

Status: Complete (2026-06-03)
Date: 2026-06-03
Owner prompt: `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/etiology_tempo.md`
Base candidate: `etiology_tempo.alternate.md` (better-structured sibling)

## Outcome (validated over multiple 54-finding dev runs)

**Scope note:** this workstream addressed only `etiologies` and `expected_time_course`. The overall
readiness verdict is still **FAIL** — `age_profile` and `index_codes` remain below floor (separate
agents), and the blunt commission cap still flags `subspecialties`/`applicable_modalities`. Do not
read these results as an overall readiness pass.

- **expected_time_course**: 0.69 → **0.76–0.78** (clears 0.75 floor); omissions ~18 → ~9. The
  sectioned commit-by-default framing + a null-trigger override (measurements/scores/classifications
  stay null) did it; prior wording tweaks had plateaued at 0.71.
- **etiologies**: ~0.74 → **0.91–0.93** (floor 0.75). Driven by (a) family-aware + clinically
  asymmetric scoring (`score_etiologies` in `metadata_scoring.py`), (b) 8 reviewer-approved gold
  corrections applied as a scoring overlay (`fixtures/etiology_gold_corrections.json`), (c) the
  prompt's indeterminate-mass → `neoplastic:potential` rule.
- **Malignancy over-calls eliminated** (cardinal sin): 0 across runs; now a hard tripwire gate.
- Etiology's blunt 5%-of-additions commission cap **retired** — it over-counted defensible
  siblings / gold-lag; etiology is gated by score + the malignancy tripwire instead.
- No regression in passing fields. Out of scope (still failing, separate agents): age_profile,
  index_codes; and the SAME blunt commission cap now also mis-flags subspecialties/modalities —
  generalizing the family-aware fix to those is the recommended next step.

Rubric review artifacts in `evals/rubric_review/`. Pending separate step: canonical writeback of
the 8 gold corrections to the approved `.fm.json` records (currently a scoring overlay only).

## Review-driven refinements (independent adversarial review, 2026-06-04)

An independent review probed the scoring and gates. What we adopted:

- **Unjustified additions now cost.** A cross-family etiology added to an otherwise-correct finding
  drops it below the 0.75 floor (cross-family-add penalty raised to 0.60). Closes the gap where
  "one correct label + one unrelated junk label" still scored 0.75 and passed.
- **Three gold calls settled.** breast calcification cluster = `neoplastic:potential` (and removed
  the contradicting "calcification clusters → null" example from the prompt); infratentorial tumor
  = `neoplastic:potential`; Pneumonia = `inflammatory:infectious`. abnormal sternomanubrial
  synchondrosis encoded as `congenital`/`developmental` so the heavier penalty does not ding a call
  the reviewer judged defensible.

What we deliberately did NOT adopt:

- **A "missed-malignancy" gate.** `etiologies` is finding-type metadata (which processes can cause a
  finding), not a per-patient diagnosis; under-calling stays light per the reviewing radiologist's
  ruling. The over-call malignancy tripwire (asserting malignancy absent from gold) remains the
  asymmetry that matters.
- **Model-pin enforcement plumbing** (assert-resolved-equals-pin, snapshot re-verification) — YAGNI
  for dev-set iteration.

## Evidence (54 dev findings, pinned `gpt-5.4-mini-2026-03-17`)

Both fields fail, but in **opposite directions** — this is the governing insight.

### expected_time_course — systematic UNDER-commitment
- Score 0.69–0.71 vs 0.75 floor; ~18–20 of 54 are pure omissions (null where curator committed).
- Robust to wording: targeted edits and the restructured alternate both plateaued at 0.71.
- Residual omissions are durable **structural** findings: omega sella, large orbit, breast
  calcification cluster, cardiac valve thickening, vertebral compression fracture, etc. The agent
  treats null as the safe default and the prompt is weighted toward abstention.

### etiologies — BOTH over- and under-commitment (commission over cap)
- Score ~0.72–0.76; commission rate 0.35–0.43 vs 0.05 cap. 31 exact, **7 inventions, 13 omissions, 3 mixed**.
- **Invention pattern 1 — differential pasted onto descriptive findings:** breast calcification
  cluster → benign+malignant+potential; unilateral hilar enlargement → inflammatory+benign+malignant+vascular
  (curator: none in both).
- **Invention pattern 2 — speculative formation labels:** arterial tortuosity, sternomanubrial
  synchondrosis → congenital+developmental (curator: none).
- **Omission pattern — the formation trio:** the `congenital` / `developmental` / `normal-variant`
  cluster drives ~9 of 23 etiology misses. hypoplastic fibula & craniosynostosis (agent congenital,
  missed developmental); large vascular grooves & short thin distal phalanx (wrong member of trio);
  large orbit (missed congenital). The current "age context alone doesn't imply congenital/
  developmental" rule is overcorrecting on genuine structural malformations.
- **Omission pattern — real masses under-called:** fetal chest mass (missed neoplastic), focal
  shadowing pancreatic lesion (missed neoplastic:potential) — the "descriptive → null" rule applied
  too aggressively to actual mass concepts.

### Why one lever won't work
A uniform "commit more" fixes time_course but worsens etiology invention. The two fields share a
root miscalibration ("what counts as clearly implied") but need **opposite nudges**: push time_course
toward commitment; give etiology *sharper boundaries* (commit on structural malformations and real
masses; abstain on descriptive appearance).

## Plan

Build on `etiology_tempo.alternate.md` (principle-first structure + closing checklist is the better
architecture). Apply two field-specific levers, then validate against noise.

### Lever 1 — time_course: commitment-default with narrow null-triggers ("commit-or-justify", prompt form)
Invert the framing so commitment is the default for any **named persistent finding class**, and null
is the *exception* that requires a specific trigger. Concretely:
- State the decision order explicitly: first decide whether the finding **class** has a characteristic
  imaging persistence; assign null only if it does not.
- Make the null-triggers a short, closed list: pure measurement number / score / index, classification
  or assessment scale, technique-only or administrative concept, artifact, or persistence that depends
  entirely on an unspecified cause.
- Add a generalized (NOT case-named) cue that durable **structural / morphologic** findings —
  deformity, dysplasia/hypoplasia, calcification, chronic enlargement, fixed anatomic variant,
  implanted hardware — are persistent by nature and should commit (permanent/years).

### Lever 2 — etiologies: sharpen the confusable clusters (fix invention AND omission together)
- **Formation trio rule:** structural malformations and anomalies of formation take
  `developmental` and/or `congenital`; fixed benign anatomic variants take `normal-variant`; these
  commonly co-occur (congenital+developmental, or developmental+normal-variant). Carve this as the
  explicit exception to "age context alone doesn't imply congenital/developmental" — the trigger is
  the **finding being a malformation**, not patient age.
- **Descriptive over-reach guard (tighten):** descriptive appearance findings (calcification
  clusters, asymmetric enlargement, density/signal patterns) keep null etiologies — do not attach a
  benign/malignant/inflammatory differential. Reinforce in the closing checklist.
- **Real-mass commitment:** a named mass/tumor/neoplasm concept commits neoplastic labels per the
  existing benign+malignant / specific-category rules, even when the surrounding description is terse.
- Leave the strong "no inference from anatomy / modality / generic association" rules intact; they
  are not the problem.

### Lever 3 (escalation, only if prompt-only plateaus) — schema-level commit-or-justify for time_course
If Lever 1 still leaves time_course under floor after validation, force the abstention decision into
the structured output (e.g. a required brief persistence rationale, or an explicit
"is-persistent-class" boolean). This is a code/schema change (`metadata/types.py`) — defer until the
prompt-only path is proven insufficient. YAGNI until then.

## Constraints (project rules)
- Do NOT paste eval finding names or answers into the prompt — generalize the categories, not cases.
- Do not duplicate schema/enum values as new rules; the structured-output schema is the spec.
- Keep edits principle-based and minimal; prefer sharpening existing rules over bulk additions.

## Validation
Single runs cannot beat the noise (±2 findings / ±0.03 observed). For the revised prompt:
1. Run the readiness scorecard 3× on the 54-finding dev set; report mean + range per field.
2. Gates to clear: time_course omissions materially below 18 AND score ≥ floor across runs;
   etiologies commission rate trending toward the 0.05 cap with no rise in omissions; no regression
   in already-passing fields (body_regions, entity_type, sex_specificity, anatomic_locations).
3. Cross-check the formation-trio and descriptive-guard changes on their specific evidence findings
   (did the known inventions stop / the known omissions resolve) without case-baking the prompt.
4. If gates hold, promote the revised alternate to active; archive this plan as Complete.

## Open decisions for the user
- Adopt the alternate's structure as the base now, or apply levers to the current edited prompt?
- Approve Lever 3 (schema change) as a pre-authorized fallback, or stop at prompt-only and reassess?
- Validation depth: 3 runs as above, or more given the noise floor?
