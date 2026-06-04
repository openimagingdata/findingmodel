# Independent review request: etiology/tempo enrichment tuning + scoring rubric

You are an independent, skeptical reviewer. Before we commit a body of work, we want you to check
whether it's sound — or whether we've fooled ourselves. Be adversarial: your job is to find what's
wrong, weak, or self-serving, not to rubber-stamp. You have full read access to this repo and can
run the eval harness.

## What the system does

`findingmodel-ai` enriches radiology "finding models" with structured metadata. One agent
(`etiology_tempo`) assigns two fields: `etiologies` (the process types that cause a finding) and
`expected_time_course` (how long the finding persists on imaging). We measure quality on a 54-finding
dev set drawn from 67 human-curated "gold" records, via a readiness scorecard.

## The problem we were solving

Both fields were failing their quality floors. `expected_time_course` systematically under-committed
(left null ~18/54 times where the curator assigned a value). `etiologies` both over- and
under-called, and the raw "over-call rate" was ~35%.

## What we changed (the work under review)

1. **Prompt** — `src/findingmodel_ai/metadata/prompts/etiology_tempo.md` (and `.alternate.md`):
   restructured into two opposite-default sections. Time-course now **commits by default** (null only
   for a closed trigger list: measurements/scores/classifications/artifacts/cause-dependent).
   Etiology stays conservative, plus a "formation trio" rule (developmental/congenital/normal-variant)
   and an **"indeterminate mass/lesion → `neoplastic:potential`, never assert malignant"** rule.
2. **Scoring** — `evals/metadata_scoring.py::score_etiologies`: a family-aware, clinically-asymmetric
   set scorer. Parent/child and developmental↔congenital are free; siblings partial; **over-calling
   malignancy is heavy; under-calling (missing a label) is light**.
3. **Gates** — `evals/metadata_readiness.py`: retired the blunt "≤5% of additions" commission cap for
   etiology; instead gate on the family-aware score + a **malignancy-over-call tripwire** (must be 0).
4. **Gold corrections** — `evals/fixtures/etiology_gold_corrections.json`: 8 reviewer-approved
   changes to the human gold, applied as a scoring-time *overlay* (canonical fixture untouched).
5. Runner wiring — `evals/metadata_readiness_run.py`; review artifacts — `evals/rubric_review/`.

Result: time_course 0.69→~0.77, etiologies ~0.74→~0.92, malignancy over-calls 0, no regressions.
Full write-up: `docs/plans/holistic-etiology-tempo-prompt-improvement-2026-06-03.md`.
Re-run: `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_readiness_run --limit 100`
(pinned, fail-closed; needs OPENAI_API_KEY). Scorer unit cases: read `score_etiologies` and try your own.

## Scrutinize these specifically (we are most worried about these)

1. **Circularity / teaching-to-the-test.** We corrected 8 human-gold answers, and several moved
   *toward what the agent produced* (e.g. indeterminate masses → `neoplastic:potential`). Are those
   corrections defensible on independent radiology grounds, or did we lower the bar to flatter the
   model? Check each entry in `etiology_gold_corrections.json` against the finding's description and
   say which you'd reject. This is our #1 concern.
2. **A dangerous asymmetry.** The scorer penalizes *over-calling* malignancy heavily but treats
   *under-calling* (missing a malignant etiology the gold had) as "light." Missing a cancer is
   clinically dangerous. Is "under-call = light" defensible for the malignant case, or have we
   under-weighted a real safety risk? Should there be a "missed-malignancy" tripwire too?
3. **Did we remove a gate just to pass?** We retired etiology's commission cap because it
   "over-counted defensible siblings." Verify that the family-aware score + malignancy tripwire
   actually still catch genuine over-reach (e.g. a differential pasted onto a descriptive finding),
   rather than just deleting the thing that was failing. Construct a case that *should* fail and check
   that it does.
4. **Overfitting to 54 dev findings.** The prompt edits and gold fixes were derived from this exact
   set. Are the prompt rules genuinely principle-based and generalizable, or tuned to these cases?
   (No eval case names should appear in the prompt — verify.) Note: a held-out/fresh certification
   batch is the real test and has NOT been run yet — is that a blocking gap before committing?
5. **Commit-by-default for time-course.** Does pushing the agent to commit a persistence by default
   risk systematic *over*-commitment at scale on findings unlike the dev set? Is the "null-trigger
   override" list complete enough?
6. **Scoring design soundness.** Read `score_etiologies` end to end. Are the penalty weights
   internally consistent? Any edge cases (empty sets, all-omission, parent+child both present,
   metastatic vs malignant) that score surprisingly? Is the per-finding mean-of-slots aggregation
   reasonable?
7. **Two sources of truth.** The gold overlay diverges from the canonical approved `.fm.json` until a
   separate writeback. Is that a maintenance hazard? Is the overlay approach the right call?
8. **Arbitrary thresholds.** Floors (etiology 0.75, time_course 0.75, etc.) and the penalty constants
   — justified, or round numbers? Does the verdict hinge on them?

## What to return

- A bottom-line judgment: are we on the right track, or should we reconsider before committing?
- A list of specific findings/files where you disagree, each with the concrete reason.
- Any gold correction you'd reject, and why.
- The single most important thing we should fix or verify before committing or scaling.
Be concrete and cite file paths / line numbers. Prefer "this is wrong because X" over general praise.
