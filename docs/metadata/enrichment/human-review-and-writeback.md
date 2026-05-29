# Human Review And Writeback

Human review is the authority for metadata enrichment. Generated metadata is a proposal until a
human-approved record or a separate field-limited repair rule makes it eligible for source
writeback.

## Review Decisions

Human review actions:

- `approved`: eligible for source writeback.
- `skipped`: excluded from the current enrichment pass.
- `feedback`: not eligible for writeback until corrected or later approved.

Sub-agent triage can help organize review queues, but it is not authority. It should only propose
queues such as accept, skip, needs attention, or suspected tool problem.

## Human Review Artifacts

Known authoritative human review artifacts are:

- Pilot review export:
  `.metadata-runs/review-exports/talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json`
  with 150 responses: 46 approved and 104 feedback.
- Targeted follow-up review export:
  `.metadata-runs/phase5-targeted-review-hardened-v3/talkasab-metadata-enrichment-review-responses.json`
  with 30 responses: 21 approved and 9 feedback.

The targeted follow-up records duplicate pilot records and update the latest effective review
status. Together these exports represent:

- 180 human review events;
- 150 unique reviewed records;
- 67 latest-approved records;
- 83 latest-feedback records.

Supporting artifacts are preserved separately. They include pilot before/after payload snapshots,
review ingestion summaries, generated review HTML, data-repo regression-floor records, and sub-agent
triage. These can explain or reproduce decisions, but they do not override human review.

## Feedback Work Queues

Latest feedback records are preserved for future work, but they are not source-writeback authority.

The 83 latest-feedback records are currently grouped as:

- expected-metadata extraction: 61;
- code/anatomy review: 11;
- expected-code-or-location extraction: 9;
- source-model issue: 2.

Affected-field counts:

- expected_time_course: 46;
- anatomic_locations: 28;
- age_profile: 15;
- sex_specificity: 11;
- etiologies: 9;
- index_codes: 9;
- applicable_modalities: 7;
- subspecialties: 4;
- entity_type: 2;
- body_regions: 1.

The 57 expected-candidate records come from the extractable subset of this feedback. The remaining
feedback still matters as source-model, code/anatomy, or review-queue evidence.

## Feedback Routing

Feedback is not source-writeback authority and never becomes one automatically. Each feedback
record moves down one of these paths; none of them write source directly:

1. **Human promotion to approved.** A reviewer re-reviews the corrected finding and approves it. It
   then enters the approved-output snapshot and becomes eligible for writeback through Gate A like
   any approved record. This is the only path to source change.
2. **Conversion to general eval/prompt guidance.** When a feedback record exposes a *general* rule
   (not a one-off), the rule is added to the field decision standards (`fields.md` /
   `subspecialties.md`) and/or covered by a component eval or gold fixture. Do not paste the missed
   case into a prompt — encode the general rule and let the eval hold the example (see
   `evaluation.md`).
3. **Source-model issue.** Records flagged as source-model problems (2 currently) are corrections to
   the finding definition itself, not enrichment decisions; they are handled as authoring fixes.
4. **Deferred.** Records with insufficient evidence stay in the feedback queue, tracked, until one
   of the paths above applies.

The 57 expected-candidate records are the mechanically-extractable subset, carried in
`metadata_review_expected_candidates.json` and marked non-authoritative (`promotion_status`,
`requires_human_promotion`). They are hints for paths 1-2, not gold. A candidate becomes gold only
after human promotion; otherwise it informs general guidance or is dropped.

## Current Fixture Baseline

Current tool-repo fixtures preserve the review evidence:

- `packages/findingmodel-ai/evals/fixtures/metadata_review_evidence_register.json`
  stores the normalized human review events and latest status.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_artifact_inventory.json`
  records known review/eval/provenance artifact groups.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_feedback_summary.json`
  summarizes latest feedback records by affected field and disposition queue.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_expected_candidates.json`
  stores conservative candidate hints extracted from feedback without promoting them to gold.
- `packages/findingmodel-ai/evals/fixtures/metadata_review_approved_outputs.json`
  stores source metadata from the 67 latest-approved generated outputs with provenance hashes.
- `packages/findingmodel-ai/evals/fixtures/metadata_source_apply_manifest.json`
  records the 78 data-source changes eligible for the approved baseline.

Expected-candidate records are mechanically marked non-authoritative with fields such as
`promotion_status` and `requires_human_promotion`.

## Current Approved Source Baseline

The current data-source baseline is:

- 67 human-approved metadata records.
- 11 index-code display backfills.

The 11 backfills are not metadata enrichment decisions. They only add human-readable `display`
strings to existing `index_codes` entries without changing systems, codes, or other metadata
fields.

## 160-To-78 Reconciliation

The older 160-record source-overlap report is audit evidence:

- 67 records had latest human approval.
- 83 records had feedback and must not be applied as approved metadata.
- 10 records were not in the human review register and must not be applied as enrichment.

The 160-record report explains why broad generated source diffs were not trusted. It is not a
writeback manifest.

The active data-repo source baseline is narrower than the old 160-record overlap:

- the 83 latest-feedback records are excluded from source writeback;
- the 10 not-in-register records are excluded as enrichment decisions;
- 67 latest-approved records are eligible for metadata writeback;
- 11 separate index-code display backfills are eligible because they only add missing display labels
  to existing codes required by validation.

This yields 78 changed definitions in the intended data-source commit: 67 approved metadata records
plus 11 display-only repairs. Their corresponding `text/*.md` files and `index.md` are regenerated
outputs, not independent review decisions.

## Gate A

Gate A is the source-application gate. It must pass before committing source metadata changes.

Gate A checks:

- the review evidence register loads and has expected counts;
- the approved-output snapshot has exactly 67 records;
- the source-apply manifest has exactly 78 records;
- the live data-repo HEAD matches the fixture pin;
- manifest `path` values exactly match modified `defs/*.fm.json` files;
- manifest `text_path` values exactly match modified `text/*.md` files;
- `index.md` is the only allowed additional derived corpus file;
- all approved metadata records in the manifest are present in the approved-output snapshot;
- the only non-approved manifest records are the 11 index-code display backfills;
- no feedback or unreviewed record is treated as metadata enrichment;
- display backfills only add `display` strings to existing `index_codes`;
- every manifest `defs/*.fm.json` path has the expected corresponding `text/*.md` path.

Gate A exists to prevent accidental source commits based on a dirty working tree. It must be
possible to explain every changed source file from the manifest and review evidence.

## Gate B

Gate B blocks destructive tool cleanup. Before deleting fallback or legacy metadata paths, the
configured split-agent `assign_metadata(...)` path must work end to end on a real case through the
metadata smoke eval or an equivalent configured-model run. Mocked unit tests alone do not satisfy
this gate.

## Display-Only Backfill Audit

The manifest's display-backfill policy is not enough by itself. Before committing the 11 display
repairs, an audit must parse before/after JSON and prove:

- only `display` keys are added;
- additions occur only inside existing `index_codes` entries;
- no code system or code value changes;
- no index code entry is added or removed;
- no other metadata field changes.

## Approved Writeback Command

The data repo has a dedicated approved-output application command:

- `../findingmodels-metadata/scripts/metadata_apply_approved_outputs.py`

The command reads the approved-output fixture from the tool repo. It verifies review-package and
reviewed-payload hashes, refuses records that are not present in approved outputs, records before
and after hashes, writes before/after JSON artifacts, and validates each updated source model before
writing.

A refusal check against a feedback record returned `refused`, which is the intended authority
boundary: feedback records are evidence for improvement, not approved source changes.

## Quarantine Policy

Generated source/text/index diffs that are not approved must be quarantined before cleanup, not
silently discarded. The current quarantine artifact is in the data repo at:

- `.metadata-runs/slice4-quarantine-2026-05-25/`

That quarantine captures the broad generated patch for 160 `defs/*.fm.json`, 160 `text/*.md`, and
`index.md`. It is historical evidence and a recovery aid, not an approved source baseline.

## Writeback Principle

Source commits must be derived from the source-apply manifest, not from `git diff`. If a changed
source file is not in the manifest, it is not part of the approved baseline.

Before any source commit, run the manifest checks, display-only audit, data validator, text/index
regeneration check, and a status/path audit that proves only manifest-backed source files and their
regenerated text/index outputs are staged.
