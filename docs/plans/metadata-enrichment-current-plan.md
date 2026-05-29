# Metadata Enrichment Cleanup Plan

Status: Active

## Goal

Get the metadata-enrichment branch and sibling data repo to a coherent, reviewable, git-clean state
so we can move toward supervised corpus enrichment. The immediate milestone is not a broad corpus
run. It is a cleaned-up branch with consolidated docs, preserved human-review evidence, validated
Gate A source-apply artifacts, rationalized prompts/evals, and only approved data-source changes.

No commits are made without explicit permission.

## Current Direction

The evidence layer already exists and currently reconciles. Treat those artifacts as validated work
to protect, not as work to regenerate casually.

The work proceeds in five reviewable commits:

1. Docs consolidation and archive mining.
2. Review evidence fixtures and Gate A validation tests.
3. Tool, prompt, and eval refactor.
4. Data-repo review/apply tooling.
5. Data-repo approved baseline and display repairs.

Commit 4 must land before Commit 5, or both must be reviewed as a tightly ordered pair, because the
approved data baseline depends on the data-side apply/audit tooling.

## Current Facts To Preserve

Human review:

- 150 unique reviewed records.
- 180 review events after targeted follow-up.
- Latest effective status: 67 approved and 83 feedback.
- The original 46 approved grew to 67 after targeted follow-up review updated 21 records to
  approved.
- Feedback records remain valuable but are not source-writeback authority.
- 57 expected-candidate records were extracted from feedback; not all 83 feedback records yield
  candidate expected values.

Source overlap and writeback:

- The old generated source overlap covered 160 modified definitions.
- Of those, 67 had latest approval, 83 had feedback, and 10 were not in the review register.
- Active source writeback is limited to 67 approved metadata records.
- A separate 11-record display-label backfill is allowed only because it adds missing `display`
  strings to existing `index_codes`.
- The intended data-source baseline is 78 definitions plus regenerated text/index output.

Tool/eval state:

- The active direction is split-agent metadata assignment.
- `entity_type` is the only required field.
- `clear_fields` is not part of the output contract.
- Execution/schema checks are gates; metadata quality is scored separately.
- Commission errors are more costly than omissions when they create unsupported groupings.
- Review-derived decision standards must be preserved as reusable rules, not only as counts,
  fixtures, or archived transcripts.
- Current bounded evals pass gates but still show weaker etiology/time-course and some
  age/anatomic-location behavior.

## Slice 1: Documentation Architecture And Archive Mining

Active documentation must have one coherent entry point, one metadata reference area, one enrichment
process area, and one active execution plan.

Target active documentation structure:

- `docs/metadata/README.md`: starting point for metadata docs.
- `docs/metadata/fields.md`: canonical metadata field reference plus field decision standards.
- `docs/metadata/subspecialties.md`: RSNA subspecialty keep/drop policy, MI/NM/SQ guidance, and
  horizontal-domain rules.
- `docs/metadata/enrichment/README.md`: enrichment process entry point.
- `docs/metadata/enrichment/evaluation.md`: gates, weighted scoring, eval sources, current weak
  fields.
- `docs/metadata/enrichment/prompt-guidance.md`: prompt construction rules only.
- `docs/metadata/enrichment/human-review-and-writeback.md`: human authority, review artifacts,
  Gate A, source writeback.
- `docs/metadata/enrichment/database-artifacts-and-package-pinning.md`: current-compatible and
  metadata-aware DB artifact strategy, local wheelhouse use, package release gate, and publish
  requirements.
- `docs/plans/metadata-enrichment-current-plan.md`: active execution plan only.

Required migrations and dispositions:

- Move/fold the legacy top-level metadata field reference into `docs/metadata/fields.md`.
- Move/fold the legacy top-level RSNA subspecialty code reference into
  `docs/metadata/subspecialties.md`.
- Archive the dated RSNA alignment plan after its durable content lands in the subspecialty
  reference.
- Archive the old gold-standard worksheet after confirming later adjudications supersede its
  conflicting values.
- Archive the old proposed-prompt examples as prompt-spam style.
- Remove the old standalone field-decision standards page after its durable rules are folded into
  field/subspecialty docs.
- Update `docs/canonical-structured-metadata-and-enrichment-rewrite.md` so it is clearly
  architecture/design context, not the current cleanup plan.
- Inventory every current `docs/*.md` and `docs/plans/*.md` file with keep/migrate/archive/defer
  disposition before staging the docs commit.

Archive mining requirements:

- Add a mined-content table to `docs/archive/metadata-enrichment/README.md`.
- For every archived metadata-enrichment plan/review, record: source file, durable content kept,
  destination doc, and stale content intentionally left historical.
- Use the completed sub-agent archive audit as input.
- After the docs are refactored, ask a sub-agent for an oppositional review focused on lost
  decisions, stale conflicts, and bad doc navigation, then address concrete findings.

Documentation checks before the docs commit:

- No active links to the old nonexistent `decision-standards` path.
- No remaining active dependence on the old standalone field-decision standards page after it is
  folded into the new reference docs.
- No active top-level RSNA doc reference.
- No active doc says body regions, subspecialties, or modalities are required.
- No active prompt docs preserve old eval-case spam.
- Archive index shows where durable content was mined.
- `CURRENT_PROGRESS_LOG.md`, unrelated provider plans, local config, and scratch artifacts are
  explicitly excluded from commits.

### Current Docs Disposition

Keep active:

- `docs/anatomic-locations.md`
- `docs/configuration.md`
- `docs/database-management.md`
- `docs/duckdb-development.md`
- `docs/logfire_observability_guide.md`
- `docs/manifest_schema.md`
- `docs/mcp_server.md`
- `docs/canonical-structured-metadata-and-enrichment-rewrite.md`, after status/link cleanup
- `docs/plans/metadata-enrichment-current-plan.md`
- `docs/metadata/enrichment/database-artifacts-and-package-pinning.md`
- unrelated active plans that are outside metadata enrichment, unless separately reviewed

Migrate into metadata reference:

- legacy top-level metadata field reference -> `docs/metadata/fields.md`
- legacy top-level RSNA subspecialty code reference -> `docs/metadata/subspecialties.md`

Archive as metadata-enrichment history:

- `docs/archive/metadata-enrichment/active-docs/gold-standard-review.md`
- `docs/archive/metadata-enrichment/active-docs/proposed-prompt-examples.md`
- `docs/archive/metadata-enrichment/plans/rsna-subspecialty-alignment-2026-04-12.md`
- dated metadata-enrichment plans already moved under `docs/archive/metadata-enrichment/`

Exclude from this commit unless separately justified:

- `CURRENT_PROGRESS_LOG.md`
- local agent config
- unrelated provider plans
- scratch/generated review logs not referenced by the durable artifact inventory

## Slice 2: Validate Existing Review Evidence And Gate A Fixtures

Do not recreate the evidence layer unless validation fails. The fixtures already exist under
`packages/findingmodel-ai/evals/fixtures/` and currently reconcile.

Required files to validate and stage:

- `metadata_review_evidence_register.json`: 180 review events, 150 unique, 67 approved,
  83 effective feedback, provenance, usage policy, drop logging.
- `metadata_review_artifact_inventory.json`: 12 artifacts, no missing artifacts.
- `metadata_review_approved_outputs.json`: exactly 67 approved-output records.
- `metadata_review_feedback_summary.json`: 83 latest-feedback records.
- `metadata_review_expected_candidates.json`: 57 candidate records, each mechanically marked
  non-authoritative with candidate/promotion fields such as `promotion_status` and
  `requires_human_promotion`.
- `metadata_source_apply_manifest.json`: exactly 78 records, 67 `human_approved_metadata` plus
  11 `index_code_display_backfill`.
- `metadata_review_source_overlap.json`: 160-record overlap audit with Gate A policy and data-repo
  HEAD pin.

Slice 2 pre-flight checks:

- Confirm the data-repo HEAD pin in the fixtures matches live `../findingmodels-metadata` HEAD.
- Confirm manifest `path` values exactly match the 78 modified `defs/*.fm.json` files in the data
  repo.
- Confirm manifest `text_path` values exactly match the 78 modified `text/*.md` files in the data
  repo.
- Confirm `index.md` is the only allowed additional derived corpus file.
- Confirm fixture counts reconcile: 67 + 83 = 150; 67 + 83 + 10 = 160; 67 + 11 = 78.
- Do not rerun fixture generators unless one of these checks fails.

Gate A tests must prove:

- every approved metadata source change maps to the 67 approved-output records;
- the only non-approved records in the manifest are the 11 display backfills;
- no feedback or unreviewed record is promoted to source writeback;
- data-source path lists are manifest-derived, not selected from `git diff`.

## Slice 3: Tool, Prompt, And Eval Refactor

Finish making the tool state match the documentation.

Required behavior:

- split metadata decision surfaces remain the active direction;
- prompt loading uses concise external prompts;
- prompt tests verify loading/schema/path behavior, not brittle exact prompt text;
- no output contract includes `clear_fields`;
- only `entity_type` is structurally required;
- metadata smoke eval passes before legacy cleanup/removal is staged;
- LLM graders are removed from the durable baseline and documented only as possible future
  diagnostics.

Gate B:

- Before deleting fallback or legacy metadata paths, prove the configured split-agent
  `assign_metadata(...)` path works end to end on a real case using the configured model path.
- Mocked unit tests alone do not satisfy this gate.

Clarify LLM grader removal scope before staging:

- Remove obsolete LLM grader eval files from the active test/eval path.
- Remove or update any live `metadata_grade` config/TOML agent surface if it no longer has a
  supported role.
- Keep future LLM review only as documented optional diagnostics: a sub-agent-style critique that
  classifies misses and suggests whether fixes belong in prompts, schema, scoring, gold data, or
  source-model cleanup.

Eval reporting must show:

- gates separately from quality scores;
- per-field scores;
- commission-sensitive set scoring;
- lowest-scoring cases and failure classes;
- enough details output to guide prompt/gold/scoring fixes without pretending every case must score
  1.00.

Tool/eval checks before commit:

- targeted metadata pytest suite;
- metadata smoke eval on the configured `assign_metadata(...)` path;
- bounded metadata assignment eval with a named fixed case set or regression floor;
- component evals for rationalized prompts;
- review evidence register tests;
- scoring tests;
- Gate A/source manifest tests.

Assign non-doc working-tree files before staging:

- `Taskfile.yml`: Slice 3 if it only wires eval/test commands.
- model config/TOML changes: Slice 3 only if tied to active metadata agents; otherwise restore or
  defer.
- `.codex/config.toml`: local-only unless proven project-relevant; do not commit by default.
- `CHANGELOG.md`: include only concise outside-user-visible changes, or leave unstaged until a final
  user-facing commit requires it.

## Slice 4: Data Repo Approved Baseline

In `../findingmodels-metadata`, land only manifest-backed data changes.

Slice 4 pre-flight checks:

- Re-run the Gate A drift check immediately before staging data-repo changes.
- If the live data-repo HEAD no longer matches the fixture pin, stop and regenerate/revalidate
  overlap and manifest before proceeding.
- If modified defs/text no longer exactly match the manifest, stop and reconcile before proceeding.

Allowed source changes:

- 67 latest-approved metadata records.
- 11 index-code display-only repairs.
- regenerated derived `text/*.md` and `index.md` corresponding to those source changes.

Required display-only audit:

- For each of the 11 `index_code_display_backfill` records, parse before/after JSON and assert the
  diff only adds `display` keys within existing `index_codes` entries.
- The audit must prove no systems/codes are added, removed, or changed, and no other metadata fields
  are touched.
- The display-backfill CSV at `notebooks/data/brain_volumetry_anatomic_code_display_backfill_2026-05-10.csv`
  should be committed only if it is necessary provenance for the 11 repairs; otherwise archive or
  exclude explicitly.

Required derived-output step:

- Regenerate `text/*.md` and `index.md` from the final 78 source definitions.
- Confirm regenerated text/index either match the current working-tree derived diffs or replace them.
- Do not trust existing text/index diffs as-is.

Not allowed:

- broad generated 160-record source diffs;
- feedback records;
- not-in-register records;
- scratch logs;
- local config.

Data checks before commit:

- approved-output dry run;
- source manifest audit;
- display-only repair audit;
- text/index regeneration check;
- data validator;
- final `git status --short` review in the data repo.

## Slice 5: Database Artifact And Package Release Readiness

This slice is not part of the immediate approved-baseline source commit, but it is part of the
overall path to corpus enrichment.

The executable, step-by-step release procedure (build → validate → publish → rollback) lives in
`docs/metadata/enrichment/release-runbook.md`; the strategy/rationale stays in
`docs/metadata/enrichment/database-artifacts-and-package-pinning.md`. The preserved direction below
is the summary.

Preserved direction:

- Build current-compatible `finding_models` and metadata-aware `finding_models_metadata` artifacts
  from the same final enriched `findingmodels` source commit.
- Keep pilot-only and partial-review DB builds validation-only; do not publish them.
- Validate `finding_models` against the legacy schema contract and current runtime.
- Validate `finding_models_metadata` against metadata columns, full enriched JSON,
  `database_metadata` provenance, and metadata-aware runtime behavior.
- Use local metadata-aware wheels in the data repo only during branch work.
- Before final data merge and DB publish, release or otherwise make available metadata-aware package
  versions and update data-repo scripts from local wheelhouse paths to released package pins.
- Rehearse manifest updates locally and publish without overwriting either manifest key.

## Commit Acceptance Criteria

Each proposed commit must have:

- explicit pathspecs;
- a concise external-facing commit message;
- verification results;
- explicit exclusions for scratch/local files.

If a commit is accidentally overstaged or includes the wrong files, recover by unstaging/resetting
that commit before proceeding; do not patch around a bad commit.

## Current Verification Baseline

Recent checks that should remain reproducible before commits:

- targeted metadata pytest suite: 51 passed;
- `task evals:metadata:smoke`: gates passed;
- `task evals:metadata`: gates passed and wrote assignment/etiology details CSVs under `/tmp`;
- data repo `uv run scripts/validator.py`: passed after approved-output application and display
  backfill;
- approved-output dry run selected 67 records and reported them already applied;
- data-side metadata scripts compiled successfully.

Current bounded quality results should be treated as diagnostic:

- assignment quality about 0.80;
- ontology, anatomy, entity type near 1.00;
- patient applicability about 0.92;
- subspecialty and modality high but not perfect;
- etiology/time-course about 0.75.

## Next Decision Points

- Finish and review the documentation refactor before asking permission for the docs commit.
- Commit review evidence and the source-apply manifest only after Gate A tests pass.
- Commit tool/eval refactor after targeted tests and bounded metadata evals pass.
- Commit data tooling and approved source baseline after validator and manifest audits pass.
- After the two repos are git-clean, use the visible eval failure classes to decide the next prompt
  or gold-data tuning step.
- Before any database publication, revalidate package pins and both database artifact paths.
