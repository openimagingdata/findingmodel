# Metadata Enrichment Plan History

Status: Historical consolidation; superseded by
`docs/plans/metadata-enrichment-current-plan.md`
Date: 2026-05-24

## Purpose

This file consolidates the older metadata-enrichment planning documents in this repo so they no
longer compete with the current readiness plan.

The active plan is:

- `docs/plans/metadata-enrichment-current-plan.md`

Older plans are evidence and history. They should not be used as instructions for new work unless a
decision has been pulled forward into the active plan or into a stable reference document.

## Current Active Direction

The current direction is:

- preserve human review evidence first;
- build a checked-in review evidence register;
- stabilize the split-agent `assign_metadata(...)` implementation;
- quarantine generated data-repo source changes only after approved evidence is safely captured;
- use reviewed-regression coverage and structural gates to decide readiness for the next corpus
  batch.

## Still-Relevant Decisions Pulled Forward

- Structured metadata belongs on canonical `FindingModel` records, not in disposable sidecar output.
- `assign_metadata(...)` is the canonical package API.
- Review/provenance artifacts remain separate from canonical source JSON.
- Only human-approved records are eligible for source writeback.
- Human feedback and approved records are more valuable than generated source diffs.
- Evals are diagnostic instruments: execution/schema gates are separate from metadata quality.
- Component evals help debug split agents but do not prove end-to-end readiness by themselves.
- Prompts should use general rules and illustrative examples, not eval-case spam.
- The etiology/time-course work is not the main project; it is one component inside supervised
  enrichment readiness.

## Historical Plan Groups

### Canonical schema and platform rewrite

- `docs/canonical-structured-metadata-and-enrichment-rewrite.md`
- `tasks/canonical-structured-metadata-implementation-plan.md`
- `docs/plans/metadata-pipeline-polish.md`

These documents explain the canonical metadata model, index/database changes, and the move away from
sidecar enrichment. They remain useful architectural background. The unfinished operational work has
been pulled into the current readiness plan.

### Coordinated two-repo enrichment and release

- `docs/plans/coordinated-metadata-enrichment-and-dual-db-release-2026-04-26.md`
- `docs/plans/coordinated-metadata-enrichment-implementation-log-2026-04-26.md`

These captured the original two-repo release path and implementation chronology. They contain useful
facts about data-repo scripts, pilot review, and DB build strategy, but they are no longer the active
execution plan.

### Prompt and eval iteration

- `docs/plans/gold-standards-and-enrichment-prompt-followup-2026-04-09.md`
- `docs/plans/metadata-assignment-full-gold-suite-expansion-2026-04-10.md`
- `docs/plans/metadata-assignment-next-iteration-2026-04-11.md`
- `docs/plans/metadata-assignment-targeted-prompt-honing-2026-04-11.md`
- `docs/plans/metadata-full-suite-miss-triage-2026-04-12.md`
- `docs/plans/metadata-targeted-example-pack-2026-04-12.md`
- `docs/plans/metadata-prompt-improvements-from-head-ct-traces.md`
- `docs/plans/metadata-prompt-repair-2026-05-11.md`
- `docs/plans/metadata-eval-scoring-and-prompt-cleanup-2026-05-13.md`
- `docs/plans/etiology-tempo-verifiable-tuning-2026-05-18.md`

These documents record useful prompt/eval lessons. They should feed the review evidence register and
stable eval-scoring docs, not remain active plans.

### Tool hardening and supervised review

- `docs/plans/pilot-feedback-tooling-hardening-subplan-2026-05-05.md`
- `docs/plans/metadata-enrichment-anatomy-scope-hardening-2026-05-05.md`
- `docs/plans/metadata-enrichment-right-sized-tool-2026-05-06.md`
- `docs/plans/metadata-enrichment-supervised-review-prompt-2026-05-10.md`

These contain the strongest history of the right-sized split-agent tool, review handoff, and
supervised writeback policy. The durable decisions have been pulled into the current readiness plan.

### Related or separate workstreams

- `docs/plans/rsna-subspecialty-alignment-2026-04-12.md`
- `docs/plans/per-agent-model-config.md`
- `docs/plans/simplify-model-config.md`
- `docs/plans/add-openrouter-provider-2026-05-19.md`
- `docs/plans/logical-commit-split-2026-04-12.md`

These are not the active metadata cleanup plan. If they are still needed, handle them as separate
workstreams or stable reference material.

## Stable References To Keep Active

- `docs/metadata-eval-scoring.md`
- `docs/finding-model-metadata-fields.md`
- `docs/configuration.md`
- `docs/database-management.md`
- `docs/database-schemas/finding_models_legacy_2026-01-28.schema.json`
- `docs/rsna-subspecialty-codes.md`

## Closeout Audit: 2026-05-25

- Checked metadata/prompt/enrichment/gold/pilot plan files in this repo.
- Active execution belongs to
  `docs/plans/metadata-enrichment-current-plan.md`.
- `docs/plans/metadata-enrichment-working-tree-inventory-2026-05-24.md` is the active temporary
  inventory for this branch cleanup.
- `docs/plans/metadata-pipeline-polish.md` is already marked complete and remains historical
  architecture context.
- All other metadata-enrichment/prompt/eval/gold/pilot plans found in the closeout pass are marked
  superseded for active execution.
