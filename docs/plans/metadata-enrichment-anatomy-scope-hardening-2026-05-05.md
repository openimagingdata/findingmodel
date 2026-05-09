# Metadata Enrichment Anatomy-Scope Hardening

Date: 2026-05-05
Status: Special-case assignment code removed; general fix still needed

## Purpose

Address the next unblocked Phase 6 anatomy issue without hard-coded finding-name or code-pair
assignment tables. The corrected regression-floor targets include `radiolucent_urinary_calculus`
using `urinary tract` and `tunneled_catheter` using `anterior chest wall`, but the tool must reach
those through candidate generation, candidate evidence, and model selection rather than deterministic
single-case remapping.

## Plan

1. Reverted: Implement the clarified parent-covers-parts rule deterministically after anatomic
   candidate selection.
2. Reverted: Add the same scope rules to the assignment decision payload so final candidate
   selection sees them even when anatomic candidates are tempting but over-specific or too broad.
3. Complete: Add focused tests that lock the prompt/payload guidance.
4. Complete: Run focused assignment/anatomic tests and lint touched files.
5. Complete: Refresh the primary metadata repo wheelhouse and rerun the targeted regression subset.
6. Complete: Update this plan and the Phase 6 hardening plan with rerun evidence.
7. Complete: Follow up on regression-floor drift by narrowing the change set.
8. Complete: Rerun the targeted floor with only upstream anatomic selector prompt hardening.
9. Complete: Correct regression-floor expectations for `radiolucent_urinary_calculus` and
   `tunneled_catheter`.
10. Reverted: Replace the prior over-broad scope prompt work with deterministic
    parent-covers-parts normalization.
11. Complete: Rerun the corrected floor against the reverted special-case code to document the
    current floor.
12. Active: Replace special-case assignment behavior with general candidate-generation and
    candidate-filtering improvements.

## Implementation Status

- Special-case assignment remapping for `radiolucent_urinary_calculus`/`urinary tract` and
  `tunneled_catheter`/`anterior chest wall` was removed.
- The anatomic selection prompt was restored to the prior specificity guidance; no prompt-content
  test remains.
- The attempted `anatomic_selection_guidance` assignment payload addition was removed after the v2
  targeted run showed broader regression-floor drift. The remaining code change is limited to the
  upstream anatomic selector prompt.
- Focused verification passed:
  `uv run pytest packages/findingmodel-ai/tests/test_anatomic_search.py packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q`
- Ruff passed for the touched support files.

## Targeted Rerun Evidence

Primary repo run:
`.metadata-runs/phase6-targeted-v2/run`

- Batch status: 12 `dry_run_success`, 0 failures.
- Regression-floor comparison:
  `docs/plans/metadata-enrichment-regression-floor-results-phase6-targeted-v2-2026-05-05.md`
- Result: not acceptable as a floor pass yet.
- Floor mismatches increased from 2 in `phase6-targeted-v1` to 5 in `phase6-targeted-v2`.
- `tunneled_catheter` no longer mismatched on anatomy, but `radiolucent_urinary_calculus` still
  mismatched on anatomy and new strict mismatches appeared in `index_codes`, `subspecialties`,
  `expected_time_course`, and `age_profile`.
- Descriptor probes did not produce deterministic audit flags in v2. `hypoechoic_liver_lesion`
  assigned `US`; `osseous_lucent_lesion` no longer assigned `MR`, so the MR-overreach audit did not
  fire.

## Next Step

Treat the v2 output as regression evidence, not as an accepted fix. The next engineering-local work
is to add narrower deterministic validators or retry checks for regression-floor drift classes
rather than broadening the anatomy prompt further.

Follow-up v3 run with only the upstream anatomic selector prompt change:

- Primary repo run: `.metadata-runs/phase6-targeted-v3/run`
- Batch status: 12 `dry_run_success`, 0 failures.
- Regression-floor comparison:
  `docs/plans/metadata-enrichment-regression-floor-results-phase6-targeted-v3-2026-05-05.md`
- Result: not accepted as a floor pass.
- Floor mismatches: 4 strict mismatches across `index_codes`, `age_profile`, and
  `anatomic_locations`.
- Descriptor probes: `osseous_lucent_lesion` again produced the intended MR-overreach audit flag;
  `hypoechoic_liver_lesion` assigned `US` and produced no audit flag.

## Correction

The earlier analysis had the target backward for two regression-floor anatomy cases. Correct targets:

- `radiolucent_urinary_calculus`: `urinary tract` only.
- `tunneled_catheter`: `anterior chest wall`.

After correcting the floor, v1 has one strict mismatch: `radiolucent_urinary_calculus` still adds
contained urinary sites alongside `urinary tract`. The next change targets exactly that behavior:
when a parent location fully covers multiple candidate parts, select the parent alone.

## Deterministic Rule Evidence

Support package focused verification:

- `uv run pytest packages/findingmodel-ai/tests/test_anatomic_search.py packages/findingmodel-ai/tests/test_assign_metadata.py -q`
  passed: 11 tests.
- Ruff passed for touched assignment/search/test files.

Primary repo run before removing the special-case code:
`.metadata-runs/phase6-targeted-v6-floor/run`

- Batch status: 10 `dry_run_success`, 0 failures.
- `radiolucent_urinary_calculus` output only `ANATOMICLOCATIONS:RID204` / `urinary tract`, but that
  was achieved by special-case assignment code and is not an acceptable implementation pattern.
- Floor comparison:
  `docs/plans/metadata-enrichment-regression-floor-results-phase6-targeted-v6-floor-2026-05-05.md`
- The floor still had 4 strict mismatches from other classes, including `tunneled_catheter`
  selecting `thorax`.

## Current Constraint

Do not implement metadata-field fixes as hard-coded finding-name, text-blob, or code-pair assignment
tables. Fixes must be general tool behavior: better candidates, better evidence supplied to the
model, generic validation of invalid output shape, or documented regression cases that evaluate the
model behavior.
