# Plan: Metadata Assignment Full Gold Suite Expansion

Status: In Progress (2026-04-10)

## Goal

Make the metadata-assignment eval suite actually validate the reviewed gold corpus instead of a tiny starter matrix.

## Problem Statement

The current metadata-assignment eval suite has two structural gaps:

1. it hard-codes an initial matrix of only `7` eval cases in `packages/findingmodel-ai/evals/metadata_assignment.py`
2. it loads fixtures from `packages/findingmodel/tests/data/defs/` rather than the reviewed gold fixture set in `packages/findingmodel-ai/evals/gold/`

That means the reported metadata-assignment eval results do not represent the reviewed gold standards.

## Inputs

- reviewed metadata comments: `metadata-standard-review.json`
- reviewed gold fixtures: `packages/findingmodel-ai/evals/gold/*.fm.json`
- current metadata-assignment eval suite: `packages/findingmodel-ai/evals/metadata_assignment.py`
- shared eval helpers: `packages/findingmodel-ai/evals/utils.py`
- prior prompt-review notes: `docs/plans/gold-standards-and-enrichment-prompt-followup-2026-04-09.md`

## Deliverables

1. metadata-assignment evals loading fixtures from `packages/findingmodel-ai/evals/gold/`
2. eval case generation covering the full reviewed gold set
3. required-field expectations derived from the actual gold fixture contents where needed
4. a fresh Logfire-instrumented eval run against the expanded suite
5. documentation updates capturing the suite expansion, results, and any newly exposed prompt issues

## Workstreams

### Workstream A: Replace The Wrong Fixture Source

- stop using `packages/findingmodel/tests/data/defs/` as the metadata-assignment gold source
- load gold fixtures directly from `packages/findingmodel-ai/evals/gold/`
- make the source path explicit in the eval code so future runs are not ambiguous

### Workstream B: Expand Case Generation To The Full Gold Set

- replace the hard-coded `7` starter cases with generated cases from all reviewed gold fixtures
- include at least one `blank_start` case for every gold fixture
- retain scenario coverage for reassess and fill-blanks semantics across the gold set wherever the fixture contents support it
- keep case naming deterministic so Logfire traces remain easy to inspect

### Workstream C: Make Expectations Match Fixture Reality

- derive required fields from the gold fixture instead of assuming every fixture has every metadata field populated
- avoid requiring fields such as `subspecialties` or `anatomic_locations` when the reviewed gold intentionally leaves them blank
- keep the must-match checks focused on fields that are meant to be gold-stable across the suite

### Workstream D: Validate With Logfire And Fold Findings Back Into Prompt Review

- rerun the expanded suite with Logfire enabled
- inspect the root eval trace plus representative individual case traces
- record whether failures are structural, fixture-related, or prompt-related
- update the prompt-review planning doc with any newly exposed, concrete prompt problems

## Implementation Plan

1. Add this plan to `docs/plans/`.
2. Patch `packages/findingmodel-ai/evals/metadata_assignment.py` to read from `packages/findingmodel-ai/evals/gold/`.
3. Replace the hand-written starter matrix with generated cases from the reviewed gold fixtures.
4. Derive per-case expectations from the gold fixture contents where that avoids false failures.
5. Run focused validation on the expanded suite.
6. Inspect Logfire traces for the expanded run, including individual case spans.
7. Update this plan and any related docs with the actual outcome and next prompt-improvement work.

## Validation

Success criteria:

- the suite covers all reviewed gold fixtures in `packages/findingmodel-ai/evals/gold/`
- Logfire shows the expanded suite running those fixtures, not the old six-file defs set
- failures, if any, are attributable to real metadata-assignment behavior rather than stale eval wiring
- documentation reflects the true scope of the eval suite after the change

## Progress Update (2026-04-10)

- Confirmed that `packages/findingmodel-ai/evals/metadata_assignment.py` currently defines an initial `7`-case matrix.
- Confirmed that `_load_gold_fixture()` currently resolves through `evals.utils.load_fm_json()`, which reads from `packages/findingmodel/tests/data/defs/`.
- Confirmed that the reviewed gold corpus contains `35` fixtures in `packages/findingmodel-ai/evals/gold/`.
- Confirmed that the reviewed gold corpus is nearly fully populated for metadata-assignment use:
  - `35/35` have `index_codes`
  - `34/35` have `anatomic_locations`
  - `35/35` have `body_regions`
  - `34/35` have `subspecialties`
  - `35/35` have `entity_type`
  - `35/35` have `applicable_modalities`
- The one reviewed gold fixture without `anatomic_locations` and `subspecialties` is `motion_artifact`, so the expanded suite needs dynamic expectations rather than blanket required fields.
- Patched `packages/findingmodel-ai/evals/metadata_assignment.py` to:
  - load fixtures from `packages/findingmodel-ai/evals/gold/`
  - generate cases from the full reviewed gold set
  - derive required fields from the reviewed gold fixture contents
- Current expanded suite size:
  - `35` reviewed fixtures
  - `139` generated cases

## Expanded Run Findings (2026-04-10)

### Run 1: Full Gold Matrix With Default-Style Concurrency

- The first expanded `139`-case run failed for infrastructure reasons rather than metadata quality.
- All `139/139` case spans failed with:
  - `[Errno 24] Too many open files`
- This was not a prompt issue and not a retry-policy issue.
- The immediate fix was to lower eval concurrency for this suite.

### Run 2: Full Gold Matrix With Lower Concurrency

- After lowering `EVAL_MAX_CONCURRENCY` to `2`, the expanded run produced meaningful case outcomes.
- Result summary from Logfire trace `019d76e45e37f6776bff9b79bb2a85e6`:
  - `139` total cases
  - `126` successful executions
  - `13` failed executions
- All `13` remaining execution failures had the same error:
  - `Exceeded maximum retries (1) for output validation`
- Every one of those remaining execution failures occurred in `partial_existing_fill_blanks_only` cases.

### Actual Metadata Problems Exposed By The Successful Cases

The expanded run exposed prompt/data-structure issues that need attention before increasing retries:

- `fill_blanks_only` under-fills blank structured metadata even when evidence is available.
  - Example: `abdominal_aortic_aneurysm_partial_existing_fill_blanks_only`
  - Gold subspecialties: `AB|VI|ER`
  - Run output in the expanded trace: only `VI`
  - Direct rerun also showed unstable partial under-fill (`VI|ER`)
- Subspecialty mapping is still drifting between neighboring enums.
  - `acute_appendicitis_blank_start` returned `GI|ER` while gold expects `AB|ER`
  - `cardiomegaly_wrong_existing_reassess` returned only `CH` while gold expects `CA|CH`
  - `pericardial_effusion_wrong_existing_reassess` returned `CH|ER` while gold expects `CA|CH`
- `entity_type` drift is still present.
  - `cardiomegaly_wrong_existing_reassess` returned `diagnosis` while gold expects `finding`
  - `pericardial_effusion_wrong_existing_reassess` returned `diagnosis` while gold expects `finding`
- Global technique issues are not being grounded cleanly.
  - `motion_artifact_blank_start` returned `entity_type=technique_issue` correctly
  - but left `body_regions` null instead of the reviewed gold `whole_body`

## Immediate Conclusions

- The first blocker was eval execution pressure, which is now addressed by lowering concurrency.
- The next blocker is not “just give it more retries.”
- The expanded run shows prompt/data-structure issues already worth addressing:
  - clearer `fill_blanks_only` guidance to fill all supported blank metadata, not just one blank field
  - stronger enum-boundary guidance for `AB` vs `GI`, `CA` vs `CH`, and when `ER` is additive
  - stronger `finding` vs `diagnosis` guidance for descriptive observations such as cardiomegaly and effusions
  - explicit handling for global technique artifacts and `whole_body`
- Retry tuning should be considered only after those prompt/data-structure fixes are made, or at most as a secondary follow-up for the residual `partial_existing_fill_blanks_only` validation failures.
