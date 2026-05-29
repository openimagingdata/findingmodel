# Metadata Enrichment

Start here for the current metadata-enrichment process.

The active direction is to make the enrichment tool good enough for supervised corpus-scale use.
The tool proposes structured metadata, preserves human-review evidence, and applies only reviewed
source changes through explicit gates.

## Current Documents

- Metadata reference: `docs/metadata/README.md`
- Field definitions and standards: `docs/metadata/fields.md`
- Subspecialty policy: `docs/metadata/subspecialties.md`
- Current execution plan: `docs/plans/metadata-enrichment-current-plan.md`
- Human review and writeback: `docs/metadata/enrichment/human-review-and-writeback.md`
- Evaluation policy: `docs/metadata/enrichment/evaluation.md`
- Prompt guidance: `docs/metadata/enrichment/prompt-guidance.md`
- Agent architecture: `docs/metadata/enrichment/enrichment-agent-architecture.md`
- Database artifacts and package pinning:
  `docs/metadata/enrichment/database-artifacts-and-package-pinning.md`
- Historical archive index: `docs/archive/metadata-enrichment/README.md`

## Current State

Two repositories are involved:

- Tool/package repo: `/Users/talkasab/repos/findingmodel-metadata`
- Corpus/data repo: `/Users/talkasab/repos/findingmodels-metadata`

The tool repo contains metadata-enrichment code, prompts, evals, review-evidence fixtures, and
planning docs. The data repo contains the finding model corpus and generated source/text/index
changes that must be kept separate from reviewed writeback.

The durable baseline is human review, not generated output. Generated source diffs and sub-agent
triage can be useful evidence, but they are not authority for source changes.

Current reviewed-source application is limited to:

- 67 human-approved metadata records from the review evidence register and approved-output snapshot.
- 11 index-code display backfills that only add `display` strings to existing `index_codes`.

The older 160-record source-overlap report is audit evidence. It is not an approved writeback list.

## What Has Been Consolidated

The active docs replace overlapping dated plans and review reports. The important durable facts
pulled forward are:

- only human review is authoritative;
- review-derived decision standards live in `docs/metadata/fields.md` and
  `docs/metadata/subspecialties.md`;
- latest human review covers 150 unique records and 180 review events;
- latest effective review status is 67 approved and 83 feedback;
- 67 approved records are preserved in `metadata_review_approved_outputs.json`;
- 83 feedback records are preserved for future eval/tool improvement, not source writeback;
- 57 expected-candidate records are preserved as non-authoritative candidates requiring human
  promotion;
- 10 generated source changes in the old 160-record overlap were not in the human review register;
- data-source writeback is blocked unless it is manifest-backed and passes Gate A;
- `entity_type` is the only required metadata field;
- optional metadata fields may be null when the finding does not justify a value;
- unsupported additions are more costly than omissions when they create false groupings;
- the obsolete `clear_fields` output contract is gone.

## Current Technical Baseline

The active enrichment implementation is split into seven focused **assignment agents** instead of
one broad classifier — each owns part of the structured output, has a lean external prompt, and is
evaluated independently. An **orchestrator** (`assign_metadata()`) gathers ontology and anatomic
candidates from **search agents**, runs the assignment agents, assembles their decisions, and an
**auditor** flags likely problems for human review. The seven assignment agents are entity_type,
etiology/time-course, patient applicability, subspecialty domain, modality applicability,
ontology (index codes), and anatomy (anatomic locations + body regions).

For the full design — pipeline flow, per-agent responsibilities, search agents, auditor, output
contract, and model configuration — see
`docs/metadata/enrichment/enrichment-agent-architecture.md`. Terminology is defined in the
repo-root `CONTEXT.md`.

The current eval direction is score-based, not all-or-nothing. Execution/schema checks are gates;
metadata judgment is scored separately. Current bounded runs pass gates but still show weaker
fields, especially etiology/time-course and some age/anatomic-location behavior. That is useful
signal, not a reason to hide results behind a single pass/fail claim.

The data repo has an approved writeback path that applies only the approved-output fixture and
refuses feedback/unreviewed records. Source changes are expected to be selected from a manifest,
not from whatever happens to be in `git diff`.

Database distribution is a later release gate, not part of the immediate docs/review-evidence
commit. The preserved plan is to build two DuckDB artifacts from the same final enriched source
commit: current-compatible `finding_models` and metadata-aware `finding_models_metadata`. During
branch work, metadata-aware data-repo scripts use local wheels from `.metadata-runs/wheelhouse`; the
final data branch must move to released package pins before DB publication.

## What Is Not Ready Yet

The project is not ready for a broad corpus run until:

- the current two-repo branch cleanup is committed in coherent, reviewable commits;
- review evidence and source-apply manifests are checked in and verified;
- data-repo source changes are limited to the 67 approved records plus 11 display backfills;
- feedback-derived candidates are either promoted by human review, converted into eval guidance, or
  explicitly deferred;
- eval reporting makes weak fields and failure classes visible.
- database build and publish gates are revalidated against both the current-compatible and
  metadata-aware artifact paths.

## Reading Rule

If a reader is trying to decide what to do next, read the current plan first, then use the reference
docs above for policy details. Dated plans in `docs/archive` are provenance and recovery material,
not active instructions.
