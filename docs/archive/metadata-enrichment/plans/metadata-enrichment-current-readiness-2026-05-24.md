# Metadata Enrichment Current Readiness Plan

Status: Active cleanup/readiness plan
Date: 2026-05-24

## Purpose

This is the current source of truth for cleaning up the metadata-enrichment work and getting back to
a usable supervised enrichment tool for the finding model corpus.

The goal is not more prompt tuning for its own sake. The goal is to:

- preserve the human review information already produced;
- make the current tool-repo branch reviewable;
- clear untrusted generated source changes from the data repo;
- prove an approved-output path from reviewed evidence to source updates;
- reach a good-enough enrichment process that can safely move through the corpus.

## Current State

This work spans two repos:

- Tool/package repo: `/Users/talkasab/repos/findingmodel-metadata`
- Corpus/data repo: `/Users/talkasab/repos/findingmodels-metadata`

Inspected state:

- `findingmodel-metadata` is on `feature/metadata-cleanup`.
- The tool repo has a large uncommitted split-agent/eval/prompt cleanup in progress.
- The tool repo has 35 checked-in metadata gold fixtures.
- The tool repo has `packages/findingmodel-ai/evals/fixtures/etiology_tempo_reviewed_cases.json`
  with 106 etiology/time-course cases.
- Targeted metadata prompt/scoring tests passed recently with `34 passed`.
- `findingmodels-metadata` is on `findingmodels-metadata`.
- The data repo has 160 modified `defs/*.fm.json` and 160 regenerated `text/*.md` files.
- Latest human-review overlap for the 160 modified source definitions is 67 approved, 83 feedback,
  and 10 not in the review register.
- The data repo also contains useful metadata scripts, review tooling, and raw review artifacts.

The generated source diffs are not authoritative. Human review is authoritative.

## Slice Status

- Slice 1: Documentation/inventory closeout complete as of 2026-05-25 after oppositional review
  cleanup.
- Slice 2: Evidence preservation and triage closeout is structurally complete as of 2026-05-25
  after oppositional review cleanup. The pilot reviewed payload is represented by the
  `pilot-enrichment/before-after/*.after.json` snapshots, and approved-output preservation no longer
  falls back to dirty source definitions.
- Slice 3: Tool-repo code/eval cleanup is complete as of 2026-05-25 after Gate B proof, legacy-path
  cleanup, details-output wiring, bounded metadata evals, and oppositional review.
- Slice 4: Data-repo cleanup and approved writeback path is complete as of 2026-05-25 after
  generated source quarantine, source cleanup, approved-output application, validator regeneration,
  and oppositional review.

Current inventory:

- The working-tree inventory is tracked in
  `docs/plans/metadata-enrichment-working-tree-inventory-2026-05-24.md`.
- Tool repo status at inventory time: 31 modified, 3 deleted, 32 untracked paths.
- Data repo status at inventory time: 328 modified and 34 untracked paths.
- Data repo generated source output currently includes 160 modified `defs/*.fm.json` files, 160
  modified `text/*.md` files, and `index.md`.

## Locked Decisions

- Human review is the only authoritative gold/review source.
- Subagent triage is supporting evidence only.
- All known human review artifacts must be inventoried before building the review evidence register.
- The checked-in normalized artifact is called the **review evidence register**.
- The review evidence register lives primarily in the tool repo because it must drive evals and
  prompt/tool improvement.
- Approved records are baseline/reference evidence.
- Feedback records must be captured and dispositioned before any new review burden.
- Generated data-repo source diffs must be quarantined, not trusted as gold.
- Do not run another corpus batch until existing review feedback has been consumed.
- Readiness is based on structural gates plus reviewed-regression coverage, not a single score.
- Current boolean LLM graders are not useful readiness infrastructure. Future LLM help, if any,
  should be an optional review-analysis subagent, not an eval score contributor.
- No commits are made without explicit permission.

## Slice Ordering And Gates

The cleanup has four slices. They are ordered because two hard gates prevent data loss and false
cleanup.

1. Planning and documentation consolidation.
2. Review evidence register and eval harvest.
3. Tool-repo code/eval cleanup.
4. Data-repo source quarantine and approved writeback path.

Gate A blocks Slice 4:

- Before clearing or quarantining generated data-repo source diffs, the Slice 2 review evidence
  register must preserve enough latest-approved human-reviewed evidence to reproduce the 67
  latest-approved source changes through the approved-output path.

Gate B blocks destructive Slice 3 cleanup:

- Before deleting fallback or legacy paths, prove the current split-agent `assign_metadata(...)`
  path works end to end on a real case using the configured model path. Mocked unit tests alone do
  not satisfy this gate.

## Slice 1: Planning And Documentation Consolidation

Create one current plan and stop treating older run-result plans as active direction.

Implementation:

- Keep this file as the active plan.
- Create one consolidated old-plan/history document per repo for metadata-enrichment planning.
- Pull still-relevant decisions from old plans into this file or a stable reference doc.
- Mark active-looking older metadata plans as superseded by this plan.
- Preserve reusable reference docs:
  - metadata field documentation;
  - eval scoring guidance;
  - setup documentation;
  - database schema documentation;
  - supervised review prompt/context only if still used as a reference.
- Convert run-result plan files into summarized evidence, not active plans.
- Review `CHANGELOG.md` and remove internal-only development-log entries.
- Keep unrelated work separate, especially the untracked OpenRouter plan and repo-local config
  changes if they are not part of metadata cleanup.

Working-tree inventory required before any commit:

- Use `docs/plans/metadata-enrichment-working-tree-inventory-2026-05-24.md` as the current
  inventory.
- Update that file before any commit if paths are added, removed, or reclassified.

## Slice 2: Review Evidence Register And Eval Harvest

This is the load-bearing preservation step.

Create the review evidence register under the tool repo, likely under
`packages/findingmodel-ai/evals/fixtures/`.

Register entries must include:

- stable record id;
- source artifact;
- human review status and feedback text when present;
- affected fields;
- expected metadata or explicit policy decision when known;
- disposition: `implemented`, `eval-covered`, `guidance-only`, `rejected`, `obsolete`, or
  `unresolved`;
- raw provenance pointer or copied evidence record.

Inventory and harvest all human review artifacts, not just the known 150-response pilot export.

Known artifacts to include or reconcile:

- Human pilot review export:
  `.metadata-runs/review-exports/talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json`
  with 150 responses: 46 approved, 104 feedback.
- Human targeted follow-up review export:
  `.metadata-runs/phase5-targeted-review-hardened-v3/talkasab-metadata-enrichment-review-responses.json`
  with 30 responses: 21 approved, 9 feedback. These 30 records duplicate pilot records and update
  their latest effective status.
- Pilot review ingestion summary:
  `.metadata-runs/pilot-review-ingest.json`
- Pilot enrichment before/after payload snapshots:
  `.metadata-runs/pilot-enrichment/before-after/*.after.json` and `*.before.json`, with 150 after
  snapshots that exactly match the 150 pilot human-review response ids.
- Targeted follow-up ingestion summary:
  `.metadata-runs/phase5-targeted-v3-review-ingest.json`
- Existing 35 tool-repo gold fixtures.
- Existing etiology/time-course reviewed fixture with 106 cases.
- Data-repo regression floor:
  `evals/regression_floor/regression-floor-v1.json`
- Phase 6 subagent triage:
  `.metadata-runs/phase6-nongmts-gmts-review-v1/review-decisions.json`
  with 400 supporting triage records. These are not authoritative gold.

Current Slice 2 progress:

- `packages/findingmodel-ai/evals/metadata_review_evidence_register.py` builds the first checked-in
  review evidence register from the 150-response human pilot export and the 30-response targeted
  follow-up export.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_evidence_register.json` currently
  preserves 180 human review events across 150 unique records: 67 approved and 113 feedback review
  events; latest effective status is 67 approved and 83 feedback.
- The register keeps reviewer comments, timestamps, source paths, source artifact hashes, per-export
  imported counts, dropped-record accounting, source repo state, and coarse affected-field
  inference.
- `packages/findingmodel-ai/evals/metadata_review_artifact_inventory.py` builds the review artifact
  inventory.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_artifact_inventory.json` currently
  accounts for 12 known review/eval/provenance artifact groups. It marks authoritative human-review
  decisions separately from supporting evidence such as generated payload snapshots, ingest
  summaries, review-package inputs, subagent triage, and regression-floor runs.
- The pilot reviewed payload is preserved as the 150 `pilot-enrichment/before-after/*.after.json`
  snapshots, not as a single `review-data.json` package. The targeted follow-up review package
  payload is present and hashed.
- The artifact inventory cross-check confirms the 150 pilot `.after.json` ids exactly match the 150
  pilot human-review response ids.
- Latest feedback field buckets are summarized in
  `packages/findingmodel-ai/evals/fixtures/metadata_review_feedback_summary.json` and
  `docs/reviews/metadata-review-feedback-summary-2026-05-24.md`.
- `packages/findingmodel-ai/evals/metadata_review_source_overlap.py` builds the Gate A source-overlap
  audit.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_source_overlap.json` currently confirms
  160 modified data-repo definitions: 67 latest-approved, 83 latest-feedback, and 10 not in the
  register. It records the data-repo HEAD used for the audit.
- The 10 modified definitions not in the register are all brain-volume region/structure volume
  records: cerebellum/brainstem, cingulate cortex, deep gray nuclei, frontal lobe, global brain,
  hippocampal asymmetry, occipital lobe, parietal lobe, temporal lobe, and ventricular system.
- `packages/findingmodel-ai/evals/metadata_review_feedback_summary.py` builds a latest-feedback
  triage summary.
- `docs/reviews/metadata-review-feedback-summary-2026-05-24.md` summarizes the 83 latest-feedback
  records for review. Current disposition queue counts are: 61 expected-metadata extraction, 11
  code/anatomy review, 9 expected-code-or-location extraction, and 2 source-model issues.
- Current affected-field counts in latest feedback are: expected time course 46, anatomic locations
  28, age profile 15, sex specificity 11, index codes 9, etiologies 9, modalities 7, subspecialties
  4, entity type 2, body regions 1.
- `packages/findingmodel-ai/evals/metadata_review_expected_candidates.py` extracts conservative
  expected-metadata candidates from latest feedback without promoting them to gold.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_expected_candidates.json` currently
  contains 57 candidate records, with extracted hints for expected time course 45, age profile 14,
  sex specificity 11, etiology 6, and forbidden etiology 4. These records require human promotion
  and preserve ambiguous range phrases instead of silently converting every range to one schema value.
- `packages/findingmodel-ai/evals/metadata_review_approved_outputs.py` snapshots latest-approved
  generated source metadata for Gate A preservation.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_approved_outputs.json` currently
  preserves metadata fields for 67 latest-approved generated source outputs, with source file hashes
  and review ids. This preserves the approved overlap before data-repo source diffs are
  quarantined, but it is not itself a gold fixture or the Slice 4 approved-output application
  command. Twenty-one records are snapshotted from the targeted review-package payload; 46 pilot
  records are snapshotted from the pilot enrichment `.after.json` payloads.

Completeness requirements:

- Count every input source.
- Assert register count equals input count for each harvested source unless a record is explicitly
  dropped.
- Record every dropped record and reason.
- Record the data-repo commit/state used for harvest.
- Avoid fresh-checkout dead links: either copy minimal raw review evidence into the tool repo or
  explicitly document local-only provenance. Current state: minimal human review evidence is copied
  into the register; source artifact paths and hashes remain as local provenance.

Forward conversion:

- Clear expected answers become eval cases.
- Generalizable lessons become prompt guidance.
- Repeated failures become code/tool issues.
- Source-model problems become data-repo issues.
- Ambiguous cases remain unresolved with owner/field.

Do not use generated `defs/*.fm.json` diffs as gold. Human-reviewed judgments are gold; generated
source diffs are not.

Slice 2 Gate A closeout status:

- The previously suspected missing reviewed-payload blocker is resolved. The pilot dataset
  `69ede7dd9df4` is represented by `pilot-enrichment/before-after/*.after.json`, which matches all
  150 pilot human-review response ids.
- Remaining Slice 2 work is to promote, reject, or explicitly leave unresolved the 57
  expected-metadata candidate records. Current state leaves them explicitly unresolved and requiring
  human promotion.
- The 10 modified data-repo definitions that are not represented in the human review register still
  need to be quarantined or otherwise handled before Slice 4 source cleanup.

## Slice 3: Tool-Repo Code And Eval Cleanup

Keep the current split-agent direction and make it reviewable.

Closeout status as of 2026-05-25:

- Gate B is satisfied. `uv run task evals:metadata:smoke` ran the configured `assign_metadata(...)`
  path on two abdominal-aortic-aneurysm cases; all gates passed and metadata quality was 0.84.
- Obsolete legacy etiology-only and imaging-workflow eval paths were removed from the active
  worktree, and live aggregate-classifier naming was replaced with split-agent assignment naming.
- Brittle prompt text assertions were removed; prompt tests now keep prompt loading/path-safety
  coverage and structured-output checks.
- Bounded metadata evals are runnable through `uv run task evals:metadata`. The latest run passed
  all end-to-end gates, wrote assignment details to `/tmp/metadata-assignment-bounded-details.csv`,
  wrote etiology/time-course details to `/tmp/metadata-etiology-tempo-bounded-details.csv`, and
  reported: assignment quality 0.80, ontology 1.00, anatomy 1.00, entity type 1.00, patient
  applicability 0.92, subspecialty domain 1.00, modality applicability 0.93, etiology/time-course
  0.76.
- Oppositional review found remaining active `classifier` naming and insufficient assignment
  details output. Both were fixed: active naming now uses assignment/decision language, and the
  assignment details CSV includes per-field actual-vs-gold score reasons for review.
- Final bounded verification after those fixes passed through `uv run task evals:metadata`;
  end-to-end assignment quality averaged 0.795 on the bounded 8-case run and etiology/time-course
  quality averaged 0.754 on 106 cases.
- Validation passed: ruff on touched metadata/eval/test files, mypy on touched metadata/eval files,
  49 targeted metadata tests, `task evals:metadata:smoke`, and bounded `task evals:metadata`.

Before destructive cleanup:

- Satisfy Gate B with a live/configured end-to-end `assign_metadata(...)` proof on at least one real
  case.
- Confirm whether `task evals:metadata:smoke` satisfies this gate. It calls
  `evals.metadata_assignment --limit 2`, which calls `assign_metadata(...)`, but the run must use the
  configured model path.

Implementation cleanup:

- Keep `assign_metadata(...)` as the canonical API.
- Keep external prompt files.
- Remove obsolete legacy eval paths and dead aggregate-classifier remnants after Gate B.
- Remove brittle prompt-content tests except minimal loader/security checks.
- Preserve schema descriptions in Pydantic output models.
- Preserve conservative optional-field behavior: only `entity_type` is required.
- Preserve `clear_fields` removal.
- Preserve existing index/anatomic code carry-forward policy for ties and uncertainty.

Eval cleanup:

- Keep execution/schema gates separate from metadata quality scores.
- Keep component evals as diagnostics.
- Make the review evidence register the central regression source.
- Require details output for broad etiology/time-course and end-to-end assignment runs.
- Stop prompt tuning unless a reviewed-regression cluster supports a general rule.
- Remove current boolean LLM graders from readiness thinking. If LLM help returns later, implement it
  as an optional review-analysis subagent that classifies misses/feedback into actionable categories
  with evidence.

Verification:

- Targeted metadata pytest suite.
- Ruff on touched files.
- Mypy.
- `task evals:metadata:smoke`.
- Bounded `task evals:metadata` with an explicit reproducible bound.
- Full metadata eval only after bounded results are interpretable.

## Slice 4: Data-Repo Cleanup And Approved Writeback Path

Slice 4 is last because it clears generated source diffs.

Closeout status as of 2026-05-25:

- The full generated source/text/index diff set was quarantined before cleanup at
  `../findingmodels-metadata/.metadata-runs/slice4-quarantine-2026-05-25/`.
  The patch covers 160 `defs/*.fm.json`, 160 `text/*.md`, and `index.md` changes.
- The unapproved generated `defs/`, `text/`, and `index.md` changes were cleared from the active
  data-repo branch.
- Approved source changes now land through
  `../findingmodels-metadata/scripts/metadata_apply_approved_outputs.py`, which reads the tool-repo
  `metadata_review_approved_outputs.json` fixture, verifies the review-package and reviewed-payload
  hashes, refuses items not present in approved outputs, writes auditable before/after/source hashes
  and before/after JSON artifacts, and validates each updated source model before writing.
- The command applied 67 human-approved records from the approved-output fixture. A refusal check
  against a feedback record returned `refused` because that item is not present in the approved
  fixture; the command's authority boundary is the approved-output fixture.
- The metadata-aware validator also required display values on existing model-level index codes for
  11 brain-volumetry records. Those were handled as deterministic display-label backfills only: no
  new codes or semantic metadata were added to feedback/unreviewed records.
- `uv run scripts/validator.py` passed after approved-output application, display backfill, and
  Markdown/index regeneration.
- The resulting data-repo source diffs are limited to 67 approved metadata records plus 11
  display-only validation repairs, with regenerated `text/` files for those 78 definitions and
  `index.md`.
- Oppositional review found missing explicit source/provenance hash handling in the approved-output
  command. This was fixed by verifying review-package/reviewed-payload hashes before accepting
  records and by recording before/after/source hashes plus the source-hash policy in the apply
  report.

Prerequisite:

- Gate A is satisfied.

Quarantine policy:

- Preserve metadata scripts and review tooling.
- Before clearing generated source/text diffs, capture the full generated diff set to a durable
  quarantine branch or patch artifact.
- Then clear generated `defs/*.fm.json`, `text/*.md`, and `index.md` changes from the active branch.
- Keep the active data-repo branch tooling-only plus documentation/setup changes until approved
  source application is proven.
- Remove duplicate run-result docs only after their useful facts are consolidated.
- Keep raw `.metadata-runs` artifacts as provenance, but do not depend on them as the active plan.

Approved writeback path:

- Source files change only through an explicit approved-output application command.
- The command refuses records that are not present in the approved-output fixture.
- Applying approved records produces auditable before/after artifacts.
- Validator passes after application.
- Markdown/index regeneration happens only after approved source changes.

No new batch runs until:

- review evidence register completeness checks pass;
- current tool eval structural gates pass;
- reviewed-feedback failures are resolved or explicitly accepted;
- data repo has no unapproved generated source diffs;
- approved-output application works end to end.

## Acceptance Gate For Next Corpus Batch

Before any next corpus batch:

- Current plan and old-plan consolidation are complete.
- Review evidence register is checked in and validated structurally and for completeness.
- Existing human feedback is eval-covered, implemented, rejected with rationale, obsolete, or
  unresolved with owner/field.
- Regression floor has been ported/imported into the tool-repo evaluation structure or explicitly
  replaced.
- Split-agent proof-of-life passed.
- Assignment eval gates pass.
- Weak fields are characterized rather than hidden behind a headline score.
- Data repo has no unapproved source diffs.
- Approved-output application path works end to end.

## Commit Strategy

No commits without explicit permission.

Recommended slices:

- Slice 1: docs consolidation only.
- Slice 2: review evidence register and harvested fixtures.
- Slice 3: tool cleanup split by concern:
  - legacy/eval removal;
  - prompt-test cleanup;
  - eval harness/reporting cleanup.
- Slice 4: data repo cleanup split by concern:
  - quarantine generated source diffs;
  - approved-output application path;
  - re-land approved records through that path.
