# Plan: Metadata Full-Suite Miss Triage

## Goal

Identify the highest-leverage, lowest-complexity fixes for the remaining metadata-assignment eval misses after the targeted prompt-example update.

## Status

- In progress
- 2026-04-12: full live metadata-assignment eval rerun produced weighted overall score `0.9273`
- 2026-04-12: all non-gold evaluators were perfect; remaining misses are all `GoldMetadataMatchEvaluator` mismatches
- 2026-04-12: largest misses are the `0.40` `fill_blanks_only` cases, which indicate complete failure on the must-match field(s)
- 2026-04-12: inspected the original six `0.40` `fill_blanks_only` misses and found a shared
  pattern:
  - under-selection of `ER` for acute trauma / kidney stone
  - over-selection of `VA` for non-vessel endpoint diagnoses and vascular-composition lesions
  - one persistent gold/prompt tension on `pyloric_stenosis` (`gold=["PD","GI"]` vs prompt/user
    steering favoring `ER`)
- 2026-04-12: patched the prompt to:
  - force fuller `subspecialties` reasoning in `fill_blanks_only`
  - narrow `VA` to vessel-centered findings
  - explicitly teach `ACL tear`, `kidney stone`, `cerebral infarction`, `liver hemangioma`, and
    `pulmonary embolism`
- 2026-04-12: reran the original six-case `fill_blanks_only` slice
  - improved from `0.40` on all six to `1.00` on five cases
  - `pyloric_stenosis_partial_existing_fill_blanks_only` remains at `0.40`
- 2026-04-12: full live suite rerun after the prompt patch improved to weighted overall score
  `0.9410` with `24` non-perfect cases, versus prior `0.9273` with `31` non-perfect cases
- 2026-04-12: this is an overall improvement, but not a clean no-regression pass; several different
  non-perfect cases appeared in the rerun and should be treated as potential regressions or
  instability candidates until rechecked

## Triage Order

1. Inspect the `0.40` cases first:
   - `acl_tear_partial_existing_fill_blanks_only`
   - `cerebral_infarction_partial_existing_fill_blanks_only`
   - `kidney_stone_partial_existing_fill_blanks_only`
   - `liver_hemangioma_partial_existing_fill_blanks_only`
   - `pulmonary_embolism_partial_existing_fill_blanks_only`
   - `pyloric_stenosis_partial_existing_fill_blanks_only`
2. For each case, compare:
   - gold must-match fields
   - actual output
   - existing locked fields in the prepared input
   - whether the failure is prompt-underinstruction, eval-expectation mismatch, or retrieval/candidate drift
3. If the pattern is shared across several cases, make the smallest prompt or assembly fix that addresses the cluster.
4. Re-run the worst-case slice before touching the full suite again.
5. Update this plan with the diagnosed pattern and chosen fix.

## Working Hypotheses

- `fill_blanks_only` may still be under-filling `subspecialties` when only `body_regions` and `entity_type` are locked.
- Some misses may reflect persistent under-selection of additive subspecialties such as `ER`, `NR`, `PD`, or `VA`.
- Some misses may be ontology/anatomic over-reliance rather than direct use of the modeled finding identity.
