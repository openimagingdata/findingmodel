# Plan: Coordinated Metadata Enrichment and Dual-Database Release

Status: In Progress (2026-04-26)

## Summary

We need to finish the metadata-enrichment work without blocking the rest of development and without
breaking the currently published `findingmodel` users. The canonical finding model source files live
in the separate `openimagingdata/findingmodels` repository, while this repository owns the Python
packages and reusable tooling that validate, enrich, index, and publish those models.

The agreed release strategy is:

1. rebase this branch on local `dev` before doing more work
2. prepare metadata-aware package/tooling changes in this repository
3. create a `findingmodels` branch named `findingmodels-metadata`
4. add high-level enrichment and database-production scripts in `findingmodels`
5. enrich canonical `findingmodels/defs/*.fm.json` directly on that branch
6. publish two DuckDB artifacts from the same enriched canonical source during a transition period:
   - `finding_models`: current-compatible artifact with the exact current published DB schema
   - `finding_models_metadata`: metadata-aware artifact for `findingmodel 2.0.0`

All publishing is manual on a maintainer machine for this plan. CI wiring is intentionally deferred,
but scripts should be written so they can later be called from CI without redesign.

The plan also introduces two quality-control assets that should outlive the initial enrichment run:

- a persisted ontology lookup cache/database used for code fact-checking and reproducible review
- a reusable Pydantic AI enrichment-auditor agent that flags likely metadata or ontology-code issues

## Key Decisions and Rationale

### Use a `findingmodels` Worktree Branch for Canonical Data Changes

Decision: create/use a `findingmodels` branch named `findingmodels-metadata`, based on updated
`origin/main`, and modify canonical `defs/*.fm.json` directly there.

Rationale: the `.fm.json` files are canonical source data, not generated artifacts of this
repository. The data change belongs in the data repository. A dedicated branch gives us a normal Git
review surface for schema updates, enriched definitions, generated markdown, generated index files,
and repo-local operational scripts.

### Keep High-Level Corpus Scripts in `findingmodels`

Decision: put corpus-level scripts in `findingmodels/scripts/`, not in this repository.

Rationale: scripts that select pilot files, run enrichment over the corpus, regenerate the data repo
schema/docs, and build/publish DBs from that repo's `defs/` directory are operational scripts for the
data repository. This repository should provide reusable package APIs and command primitives; it
should not own the corpus workflow.

### Use Local Wheels During Iteration, Then Released Pins

Decision: during preparation, build local wheels from this branch and have `findingmodels` scripts use
those local wheels. Do not commit local path hacks or machine-specific dependency wiring. Before the
final data branch is merged, switch metadata scripts to released package pins.

Rationale: local wheels test the packages as installable artifacts without forcing premature
publication. They are more reproducible than `PYTHONPATH` or editable path injection, and they avoid
committing temporary local paths to the data repository.

### Keep Current-Compatible DB Publishing Alive

Decision: for a transition period, publish both DB outputs from the same enriched `defs/` source.

Rationale: downstream users of the current `findingmodel` package still need the existing
`finding_models` DB contract. New metadata-aware users need a separate artifact. Publishing both lets
canonical source move forward while keeping current users functional.

### Define "Legacy" as Exact Current Published DB Schema

Decision: the current-compatible artifact means exact current published DuckDB table/column schema,
published under the existing `finding_models` manifest key. It does not include metadata columns.

Verified current published `finding_models` schema from the live manifest artifact:

- tables: `attributes`, `finding_model_json`, `finding_models`, `model_organizations`,
  `model_people`, `organizations`, `people`, `synonyms`, `tags`
- `finding_models` columns: `oifm_id`, `slug_name`, `name`, `filename`, `file_hash_sha256`,
  `description`, `search_text`, `embedding`, `created_at`, `updated_at`
- no metadata columns, no helper metadata table

Rationale: old/current clients should not see or hydrate new metadata columns. This makes enum/data
compatibility concerns irrelevant for the current-compatible DB artifact.

Implementation note: create a checked-in schema contract artifact for this legacy/current-compatible
DB shape, and keep it until we stop publishing the non-metadata DB. This should be generated from the
actual live artifact or the verified matching builder and stored in the repository, for example under
`docs/database-schemas/`. The artifact should include table names, column names/types/nullability,
primary keys, and indexes. Legacy build validation must compare against this artifact rather than
against prose in this plan.

### Use Old Pinned Tooling for the Current-Compatible DB

Decision: the `findingmodels` legacy DB script should use PEP 723 script dependencies pinned to the
old/current package set:

- `findingmodel==1.0.4`
- `oidm-common==0.2.7`
- `oidm-maintenance` from Git URL pinned to commit
  `75afd39a400419dcfaf7c8d4a34f065b4d804e0d`

Use the actual repository URL:

```text
git+https://github.com/openimagingdata/findingmodel.git@75afd39a400419dcfaf7c8d4a34f065b4d804e0d#subdirectory=packages/oidm-maintenance
```

Rationale: `oidm-maintenance` is not published on PyPI, but commit `75afd39...` is local `main`,
explicitly titled `Merge dev: oidm-common-0.2.7, findingmodel-1.0.4, oidm-maintenance-0.2.5`, and
contains the current published DB schema. Running old tooling against enriched source files naturally
projects models through the old `FindingModelFull` shape when it parses/dumps stored JSON, so the
current-compatible DB remains old-shape without a separate metadata-stripping implementation.

This Git dependency pattern has been sandbox-tested with `uv run --isolated --with ...`; `uv`
successfully built and imported `oidm-maintenance`, `findingmodel`, `oidm-common`, and
`anatomic-locations` from the pinned commit.

### Release Metadata-Aware Package Line as `findingmodel 2.0.0`

Decision: target `findingmodel 2.0.0` for the metadata-aware release line.

Rationale: this work changes the public model schema, DB default behavior, and package/runtime
assumptions. A major version communicates that users should treat this as a new compatibility line.

### Default New Package to `finding_models_metadata`

Decision: `findingmodel 2.0.0` should resolve the new metadata-aware DB by default using manifest key
`finding_models_metadata`.

Rationale: new package users should receive the artifact that matches the new metadata-aware runtime.
Current users stay on `findingmodel 1.0.4` and keep resolving `finding_models`.

### Defer CI

Decision: do not implement CI in this plan. All enrichment, review, DB build, and publish actions are
manual on a maintainer machine.

Rationale: CI raises separate questions about secrets, publishing permissions, branch trust, and
artifact retention. Those are important but not necessary to unblock the immediate enrichment and
release workflow. Scripts should still be designed so they can later be called from CI.

### Persist an Ontology Lookup Cache

Decision: create and keep a durable ontology lookup cache/database for ontology codes encountered
during enrichment and review.

Rationale: ontology identifiers and preferred terms are intended to be stable. Repeated live lookups
are slow, harder to reproduce, and make review/auditing dependent on external service availability.
The cache should preserve lookup evidence used to select or reject `index_codes`, especially because
hallucinated ontology codes are a high-impact failure mode. The cache is not a transient run artifact;
it is reusable infrastructure for enrichment, auditing, future evals, and manual review.

The cache should store, at minimum:

- ontology system
- code
- preferred display term
- synonyms/labels returned by the lookup source when available
- source service and source URL or concept URI
- lookup timestamp
- raw normalized response or enough structured fields to audit the match later
- whether the concept was used as a canonical selected code, related candidate, rejected candidate,
  or fact-check evidence

### Add an Enrichment Auditor Agent

Decision: add a separate Pydantic AI auditor agent for QA. It should not write canonical model JSON.
It should inspect enriched outputs and produce structured flags for human review and run metrics.

Rationale: the primary enrichment agent can make plausible-looking mistakes, especially with ontology
codes. A second pass with a narrower "find problems" prompt creates an independent safety check and a
reusable review asset. The auditor is a triage tool, not the final authority.

The auditor must, at minimum:

- fact-check `index_codes` against ontology lookup evidence
- flag nonexistent or hallucinated codes
- flag wrong ontology-system/code pairings
- flag display terms that do not match preferred terms when the lookup provides one
- flag codes that are merely related when the model stored them as canonical exact/substitutable
  `index_codes`
- flag internally inconsistent metadata, such as modality/subspecialty/body-region mismatches
- flag likely over-broad etiologies or inappropriate age/sex/time-course assignments
- output severity, field, evidence, and a concise rationale for each flag

Auditor output should feed the HTML review package and summary metrics. If pilot or full-run auditor
flags are high-volume or clustered by issue type, pause and revise prompts/tooling before proceeding.

## Phase 1: Rebase and Stabilize This Repository

### Why This Phase Exists

This branch is not up to date with local `dev`. Local `dev` includes commit `504a42a`, which removes
FastEmbed/local embedding support and simplifies runtime/build behavior to OpenAI-only. Any plan that
continues to reason about old embedding-profile complexity will be wrong. Rebase first so all later
tooling and database decisions are made against the current development baseline.

### Required Work

1. Rebase `feature/metadata-cleanup` onto local `dev`.
2. Resolve conflicts in favor of `dev`'s OpenAI-only embedding simplification.
3. Preserve the metadata model, eval, prompt, RSNA subspecialty, and gold-standard work already on
   this branch unless a conflict reveals a direct incompatibility.
4. Run focused tests for changed packages:
   - `packages/findingmodel`
   - `packages/findingmodel-ai`
   - `packages/oidm-common`
   - `packages/oidm-maintenance`
5. Run the metadata assignment eval suite with Logfire enabled only if the rebase resolution changes
   enrichment-affecting code, such as assignment prompts, metadata assignment behavior, eval fixtures,
   eval harness code, or metadata model-routing configuration.
6. If no enrichment-affecting code changed, explicitly document why full evals and Logfire trace
   checks are deferred until the next enrichment/prompt/tooling change or the pre-pilot gate.
7. Verify the current published DB schema by downloading the live manifest artifact and running schema
   inspection against the actual DuckDB file.
8. Create the checked-in legacy schema contract artifact from the verified current-compatible DB
   schema.
9. Update this plan with any rebase consequences that change the implementation sequence.

### Done Criteria

- `feature/metadata-cleanup` is rebased onto local `dev`.
- No unresolved conflicts remain.
- OpenAI-only behavior from `dev` is preserved.
- Focused package tests pass or any failures are understood and resolved in this workstream.
- Metadata evals either run successfully with Logfire traces available for inspection, or are
  explicitly deferred because no enrichment-affecting code changed during the rebase resolution.
- The legacy/current-compatible DB schema contract artifact is checked in.
- Any changed assumptions are documented in this plan before proceeding.

### Phase 1 Execution Update (2026-04-26)

Status: completed and committed. Detailed commands, verification output, and schema-capture notes
are in `docs/plans/coordinated-metadata-enrichment-implementation-log-2026-04-26.md`.

Summary:

- Rebasing onto local `dev` completed without changing enrichment prompts, assignment logic, eval
  fixtures, or model routing.
- Package-suite tests passed for the rebase scope.
- Full metadata evals and Logfire smoke checks were intentionally skipped because no
  enrichment-affecting code changed.
- The current published DB schema was captured as the legacy/current-compatible schema contract.

## Phase 2: Finalize Package Capabilities in This Repository

Status: Implemented locally and package-suite validation passing (2026-04-26); awaiting review and
commit.

### Why This Phase Exists

The `findingmodels` branch will depend on package behavior from this repository. Before enriching
canonical data, we need the model schema, assignment behavior, DB builders, and publish primitives to
be coherent enough that the data repo scripts can call them reliably.

### Phase 2 Execution Approach

Start with an audit against the checklist below before making substantive implementation changes.
This avoids assuming package capabilities that may not actually exist after the rebase.

Initial audit targets:

- optional metadata fields and generated JSON Schema support in `findingmodel`
- markdown/schema rendering behavior expected by the `findingmodels` validator
- stable external metadata assignment API and review artifacts in `findingmodel-ai`
- metadata-aware DuckDB build and manifest-key targeting in `oidm-maintenance`
- durable ontology lookup cache support
- enrichment auditor agent support

Full metadata evals and Logfire trace review are not part of the opening audit unless the audit leads
to prompt, assignment, ontology, auditor, model-routing, or eval-harness changes. They remain required
before pilot enrichment or after any enrichment-affecting implementation change.

Any implementation work in this phase should update this plan with the changed status and should
identify whether focused tests, full metadata evals, or traced Logfire inspection are required by the
kind of change made.

### Phase 2 Execution Update (2026-04-26)

Status: completed and committed. Detailed audit notes, implementation chronology, and validation
commands are in `docs/plans/coordinated-metadata-enrichment-implementation-log-2026-04-26.md`.

Summary:

- Package models, schema generation, markdown rendering, assignment API, metadata-aware DB build,
  ontology-cache support, and auditor support were audited after rebase.
- Gaps were closed for DB provenance, publish target parameterization, durable ontology lookup
  evidence, and the enrichment auditor.
- The ontology cache decision was settled on DuckDB.
- The auditor was intentionally kept as a lightweight Pydantic AI sanity checker backed by
  deterministic ontology-cache lookup evidence.
- `FINDINGMODEL_DB_MANIFEST_KEY` was added while preserving the current default manifest key until
  the `findingmodel 2.0.0` release gate.
- Broad package-scope validation passed after the Phase 2 changes.
- Full metadata assignment evals and Logfire trace review were intentionally skipped because the
  Phase 2 work did not alter assignment prompts, assignment behavior, ontology selection, eval
  fixtures, eval harness code, or model routing.

### Required Work

1. Confirm the already-built `FindingModelFull` metadata fields survive the rebase and remain
   optional:
   - `body_regions`
   - `subspecialties`
   - `etiologies`
   - `entity_type`
   - `applicable_modalities`
   - `expected_time_course`
   - `age_profile`
   - `sex_specificity`
   - `anatomic_locations`
2. Confirm `FindingModelFull.model_json_schema()` includes the metadata fields and supporting enum
   definitions.
3. Confirm the markdown rendering used by `findingmodels/scripts/validator.py` renders metadata fields
   for enriched models.
4. Ensure `findingmodel-ai` exposes a stable metadata assignment API that a repo-local batch script
   can call directly.
5. Ensure assignment review output includes enough information for audit:
   - selected ontology candidates
   - selected anatomic candidates
   - warnings
   - confidence fields
   - Logfire trace IDs where available
6. Parameterize metadata-aware publish tooling so it can target manifest key
   `finding_models_metadata` instead of hardcoding `finding_models`.
7. Ensure `oidm-maintenance` can build the metadata-aware DB shape from `.fm.json` files after the
   rebase.
8. Decide and implement the metadata DB metadata/version table with:
   - DB kind/schema identifier
   - source `findingmodels` commit
   - build timestamp
   - package/tooling versions
   - embedding model information
9. Add ontology cache/database support that can persist lookup evidence for index-code selection and
   fact-checking.
10. Add the enrichment auditor agent and structured auditor output models.
11. Prepare package versioning for the metadata-aware line, targeting `findingmodel 2.0.0`, but do
    not change the active package-line default manifest key before the release gate.

### Done Criteria

- Current un-enriched `.fm.json` fixtures validate with the new optional schema.
- Schema generation contains all metadata fields and supporting definitions.
- Metadata assignment API is callable from external scripts without relying on private internals.
- Metadata-aware DB build produces a readable DB with metadata columns populated when source models
  have metadata.
- Metadata-aware publish flow can write/update `finding_models_metadata` without touching
  `finding_models`.
- Ontology lookup evidence can be cached and re-used.
- Auditor agent can run on enriched model JSON and produce structured review flags.
- Documentation and changelog updates needed for package users are identified and updated for the
  completed Phase 2 package changes.

## Phase 3: Build Local Wheels for Data-Repo Iteration

Status: Completed (2026-04-26).

### Why This Phase Exists

The `findingmodels` scripts need to run against metadata-aware packages before those packages are
published. We need a reproducible local mechanism that exercises the same installed package shape
users will eventually get, without committing local paths or prematurely publishing packages.

### Required Work

1. Build local wheels from this branch for the packages needed by the `findingmodels` scripts.
2. Store those wheels in a local, uncommitted wheelhouse.
3. Document the exact command used to build the wheelhouse.
4. Document the exact `uv run` invocation pattern for scripts that need the wheelhouse.
5. Verify a small script launched from the `findingmodels` checkout imports the local wheel versions,
   not PyPI versions.

### Done Criteria

- Local wheelhouse exists and contains the metadata-aware package artifacts.
- A `findingmodels` script can run with those wheels without editing committed dependency files.
- The script reports/imports the expected local package versions.
- No machine-specific local path wiring is committed to either repository.

### Phase 3 Execution Update (2026-04-26)

Status: completed. Detailed wheel paths, build commands, and verification notes are in
`docs/plans/coordinated-metadata-enrichment-implementation-log-2026-04-26.md`.

Summary:

- Local wheels were built for all packages needed by the data-repo scripts.
- Package build-system requirements were updated to match the active `uv_build` release line.
- A `--no-sources` wheel build succeeded as a publish-readiness check.
- The wheel environment verified imports and expected metadata-aware package capabilities.
- No committed dependency file, lockfile, or local path wiring changed in either repository.

### Phase 4 Wheel Usage Decision

Decision: metadata-aware `findingmodels` scripts should use PEP 723 inline metadata with
`[tool.uv.sources]` entries pointing at local wheel files under ignored `.metadata-runs/` paths. This
lets maintainers run scripts with plain `uv run scripts/<script>.py` while still installing the
unpublished metadata-aware wheels from this repository.

Rationale: testing showed uv accepts local wheel sources in script headers, including relative paths
resolved from the script location. This is cleaner and less error-prone than requiring every command
to repeat a long list of `--with /tmp/.../*.whl` arguments. It also avoids committing absolute
machine-specific paths. The wheel files themselves remain untracked run artifacts.

Implementation pattern for metadata-aware scripts:

```toml
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "findingmodel",
#   "findingmodel-ai",
#   "oidm-common",
#   "oidm-maintenance",
#   "anatomic-locations",
# ]
# [tool.uv.sources]
# findingmodel = { path = "../.metadata-runs/wheelhouse/current/findingmodel-1.0.4-py3-none-any.whl" }
# "findingmodel-ai" = { path = "../.metadata-runs/wheelhouse/current/findingmodel_ai-0.2.1-py3-none-any.whl" }
# "oidm-common" = { path = "../.metadata-runs/wheelhouse/current/oidm_common-0.2.7-py3-none-any.whl" }
# "oidm-maintenance" = { path = "../.metadata-runs/wheelhouse/current/oidm_maintenance-0.2.5-py3-none-any.whl" }
# "anatomic-locations" = { path = "../.metadata-runs/wheelhouse/current/anatomic_locations-0.2.5-py3-none-any.whl" }
# ///
```

The legacy/current-compatible DB script is the exception: it should keep its pinned published/Git
dependencies from the legacy-tooling decision because it intentionally proves compatibility with the
current published DB schema.

## Phase 4: Prepare the `findingmodels-metadata` Branch

### Why This Phase Exists

The canonical source and repo-local operations belong in `findingmodels`. This branch will be the
review surface for schema support, high-level scripts, enriched JSON, generated markdown, and generated
index files. It must stay internally synchronized.

### Required Work

1. Update `findingmodels-main` from `origin/main`.
2. Create or update branch `findingmodels-metadata`.
3. Limit the source corpus for this work to `defs/*.fm.json`.
4. Do not include `conflicts/` files in enrichment or DB production for this plan.
5. Add `.metadata-runs/` to `.gitignore`.
6. Add repo-local scripts under `findingmodels/scripts/`:
   - pilot selection script
   - enrichment batch script
   - HTML review-package generation script
   - human-review ingestion script
   - auditor-run script or auditor integration in the enrichment/review workflow
   - legacy DB build/publish script
   - metadata DB build/publish script
7. Keep assignment and audit orchestration simple. Do not add a package-level `assign_and_audit()`
   helper unless duplication grows beyond the current batch script's direct `assign_metadata()` then
   `audit_enrichment()` sequence.
8. Keep detailed run outputs under `.metadata-runs/`, untracked.
9. Add a guard or dependency pin so `scripts/validator.py` cannot silently run with
   `findingmodel<2.0.0` once enriched metadata fields are present.
10. Regenerate `schema/finding_model.schema.json` using the metadata-aware package.
11. Manually update `schema/finding_model_schema.md` to document the new optional metadata fields.
12. Run the existing validator so generated `text/*.md`, `index.md`, and `ids.json` are synchronized.

### Script Requirements

The pilot selection script must:

- use deterministic stratified/seeded sampling
- select exactly 150 representative `defs/*.fm.json` files unless fewer are eligible
- stratify across filename/name/tag-derived buckets so the pilot covers broad clinical and metadata
  patterns rather than only previously reviewed gold examples
- include bucket coverage for common anatomy/body-region signals, modality-specific names,
  pediatric terms, vascular findings, oncologic/tumor findings, trauma/fracture, measurements and
  classifications, artifacts/technique issues, broad nonspecific abnormality labels, and common
  thoracoabdominal/neuro/MSK/GU/GI/breast categories
- fill remaining slots with seeded random sampling after stratified quotas are met
- write the selected file list to `.metadata-runs/`
- avoid using reviewed gold answers as prompt examples or hidden labels

The enrichment batch script must:

- operate on `defs/*.fm.json`
- default to concurrency `3`
- call package-level `findingmodel-ai` assignment APIs
- write updated `.fm.json` files directly to the branch
- leave a file unchanged on unresolved failure
- continue after per-file failures
- record failures in `.metadata-runs/`
- write per-file review JSON
- write before/after metadata snapshots
- write status JSONL
- record Logfire trace IDs when available
- support resuming or skipping already-completed files
- write or update ontology cache entries for ontology concepts used or considered during enrichment
- run or enqueue the auditor agent for enriched files
- preserve ontology candidate relationship and rejection-reason evidence in the cache so later review
  can distinguish selected codes from rejected related/broader/narrower candidates

The HTML review-package script must:

- use the same review-tool pattern as the gold-standard review process in `../review_tool`, not a
  static report page
- generate a standalone single-file HTML reviewer plus supporting `review-data.json` for pilot and
  full-run review sets
- default to a stable current-review location at `.metadata-runs/review-current/` so the human
  reviewer has one obvious page to open; run-specific output directories are optional archival
  artifacts, not the primary interaction model
- use an inbox-style sidebar, one active item at a time, local draft persistence, keyboard shortcuts,
  and JSON download hand-back in the same general interaction model as the existing review tool
- keep the top finding display focused on the reviewer-facing enriched finding metadata
- render structured fields such as age profile and expected time course as readable label/value
  content, not raw JSON blobs
- show a run warning/error box only when assignment warnings or auditor flags exist; do not show
  empty "none" boxes
- show field confidence in a collapsed accordion
- show run details in a collapsed accordion
- keep lower-level evidence such as raw before/after metadata, metadata diffs, ontology candidate
  review, index-code cache evidence, and raw audit JSON out of the default reviewer surface
- require completed enrichment artifacts for every included item; the review package must fail loudly
  if an enriched snapshot or metadata-review artifact is missing rather than silently falling back to
  source-only model JSON
- export structured reviewer responses in the review-tool JSON format so they can be handed back and
  ingested without scraping browser state

The human-review ingestion script must:

- read the structured review JSON file exported by the HTML review tool
- summarize raw tool statuses (`approved`, `feedback`, unfinished items) and normalize them into the
  actionable buckets needed for prompt, tooling, ontology-cache, or source-model follow-up
- produce a concrete fix list for prompts, package code, ontology cache corrections, or source model
  edits
- treat repeated hallucinated ontology codes, missing cache evidence, and code/display mismatches as
  prompt/tooling issues to resolve before expanding the run
- block progression from pilot to full enrichment until every pilot review item has been reviewed in
  the tool and every feedback item has been triaged into concrete follow-up work

The legacy DB build/publish script must:

- use PEP 723 script dependencies
- pin `findingmodel`, `oidm-common`, `oidm-maintenance`, and `anatomic-locations` by Git URL to
  current-main commit `75afd39a400419dcfaf7c8d4a34f065b4d804e0d`
- build from the enriched `defs/` source
- produce the exact current published DB schema
- publish/update manifest key `finding_models`

The metadata DB build/publish script must:

- use local wheelhouse during iteration
- later switch to released metadata-aware package pins
- build from the enriched `defs/` source
- produce metadata-aware DB schema
- include the DB metadata/version table
- publish/update manifest key `finding_models_metadata`

### Done Criteria

- `findingmodels-metadata` branch exists from updated main.
- The data repo has scripts for pilot selection, enrichment, legacy DB production, and metadata DB
  production.
- The data repo has scripts for HTML review-package generation and review JSON ingestion.
- `schema/finding_model.schema.json` reflects metadata-aware `FindingModelFull`.
- `schema/finding_model_schema.md` documents the new optional fields.
- `scripts/validator.py` is guarded or pinned against metadata-stripping old package execution.
- Validator runs cleanly with local metadata-aware package wheels.
- Generated files are synchronized with `defs/`.

### Phase 4 Completion and Phase 5 Handoff

Status: completed locally on the `findingmodels-metadata` branch of the `findingmodels-metadata`
repository.
Detailed smoke commands, local artifact paths, validation output, and review-generator cleanup notes
are in `docs/plans/coordinated-metadata-enrichment-implementation-log-2026-04-26.md`.

Relevant local commits:

- `findingmodels-metadata`: `04c8ba5 feat: add metadata enrichment workflow tooling`
- `findingmodels-metadata`: `6700a8c chore: pin metadata-aware schema and validation tooling`
- `findingmodels-metadata`: `fce7966 chore: regenerate finding markdown with stable filenames`
- `findingmodel-metadata`: `5af26a8 docs: split metadata enrichment plan and implementation log`

Phase 5 should primarily run in `/Users/talkasab/repos/findingmodels-metadata` on branch
`findingmodels-metadata`. It uses local metadata-aware wheels built from
`/Users/talkasab/repos/findingmodel-metadata`; see
`/Users/talkasab/repos/findingmodels-metadata/docs/metadata-enrichment-setup.md` for wheelhouse and
environment setup.

Implemented scripts:

- `scripts/metadata_select_pilot.py`
- `scripts/metadata_assign_batch.py`
- `scripts/metadata_audit.py`
- `scripts/metadata_review_package.py`
- `scripts/metadata_ingest_review.py`
- `scripts/build_legacy_findingmodel_db.py`
- `scripts/build_metadata_findingmodel_db.py`

Implementation decisions:

- `.metadata-runs/` is ignored and is the local home for the wheelhouse, pilot manifests, enrichment
  artifacts, review apps, ontology cache, and database smoke outputs.
- Metadata-aware scripts use PEP 723 `[tool.uv.sources]` entries that point to local wheel files under
  `.metadata-runs/wheelhouse/current/`.
- The legacy DB script intentionally does not use local metadata-aware wheels. It pins the current
  compatible package set to the chosen current-main Git commit so it proves the current published DB
  contract separately from the metadata-aware DB path. This is explicit because uv resolves
  `oidm-maintenance` Git dependencies with sibling packages from the same Git source.
- `scripts/output_schema.py` and `scripts/validator.py` are pinned to local metadata-aware wheels
  during this branch work so generated schemas and docs cannot silently come from the published
  non-metadata model.
- `scripts/validator.py` now has a metadata-aware package guard and preserves the existing
  `.fm.json` filename when regenerating markdown and `index.md`. This avoids metadata loss from an
  old package and avoids broken generated links for files whose model names do not map exactly to the
  existing filenames.
- `scripts/metadata_assign_batch.py` has an explicit `--logfire` switch that calls the package
  Logfire configuration helper before assignment starts. Without this switch, it behaves like the
  package CLI default and does not assume cloud tracing is configured.
- `scripts/metadata_assign_batch.py` writes source `.fm.json` files only after assignment, audit, and
  review artifact generation succeed, so audit failures do not leave partially accepted source
  changes.
- `scripts/output_schema.py` and `scripts/validator.py` both guard that they are running with a
  metadata-aware package before regenerating schema or source-derived files.
- The reviewer-facing interaction model is one stable current-review location:
  `.metadata-runs/review-current/index.html`.
- The metadata review generator should remain a thin adapter from completed enrichment outputs to
  `review-data.json` plus the HTML template. It should not become a workflow framework.
- The data repo now includes `docs/metadata-enrichment-setup.md` explaining how to populate the local
  wheelhouse and run the basic enrichment/review smoke commands.
- `scripts/metadata_review_template.html` is marked as an adaptation of `../review_tool` so future
  maintainers know it is intentionally forked.
- No package-level `assign_and_audit()` helper was added; the current duplication is limited to the
  batch script's direct assignment-then-audit sequence and is not worth another package API yet.

Validation and smoke status:

- Help commands resolve for the new Phase 4 scripts.
- One-model metadata-aware and current-compatible DuckDB smoke builds succeeded.
- Schema regeneration and validation succeeded with local metadata-aware wheels.
- A one-item live dry-run enrichment smoke succeeded with `.env`, Logfire, and ontology cache.
- A real three-item dry-run enrichment smoke succeeded with concurrency `3`.
- The three-item run generated a multi-item review app at `.metadata-runs/review-current/index.html`.
- Review ingestion smoke checks succeeded for approved and feedback exports.
- The review generator was simplified after review and received a targeted subagent fix for
  append-only `status.jsonl` dedupe and relative CLI path normalization.
- Follow-up code review fixes tightened source-write ordering, transient retry behavior, standalone
  audit safety, stable review IDs, review export paths, schema guard coverage, metadata DB provenance,
  and setup documentation.

Phase 5 command shape:

```bash
uv run scripts/metadata_select_pilot.py \
  --defs-dir defs \
  --target-count 150 \
  --output-dir .metadata-runs/pilot
```

```bash
uv run --env-file .env scripts/metadata_assign_batch.py \
  --manifest .metadata-runs/pilot/pilot_manifest.json \
  --run-dir .metadata-runs/pilot-enrichment \
  --ontology-cache .metadata-runs/pilot-ontology-cache.duckdb \
  --concurrency 3 \
  --logfire
```

```bash
uv run --env-file .env scripts/metadata_review_package.py \
  --run-dir .metadata-runs/pilot-enrichment
```

Open `.metadata-runs/review-current/index.html` for human review. Export the review JSON from that
page, then ingest it:

```bash
uv run scripts/metadata_ingest_review.py <exported-review-json> \
  --output .metadata-runs/pilot-review-ingest.json
```

Then run:

```bash
uv run scripts/validator.py
```

Important handoff rules:

- Pilot-enriched source files are branch working state for review and iteration, not publishable
  release artifacts.
- Do not proceed to the tool-improvement phase only because scripts ran successfully. The human review export must be
  ingested, every feedback item must be triaged, and every pilot item must be accepted, fixed, or
  explicitly deferred with rationale.
- Use `.metadata-runs/review-current/index.html` as the reviewer-facing page. Do not make raw run
  artifacts the primary review surface.
- `IndexCode.code.minLength == 2` in the regenerated schema is expected; it comes from the current
  `oidm-common` `IndexCode` model.

## Phase 5: Pilot Enrichment

### Why This Phase Exists

Before running enrichment over 2,149 models, we need a representative proof that the prompt, assignment
API, run artifacts, validator, generated docs, and DB production path work together on real canonical
source files. The pilot is a quality and workflow gate, not a publishable dataset.

### Required Work

1. Use the pilot selection script to choose about 150 representative models.
2. Run enrichment on real `findingmodels/defs/*.fm.json` files selected by the pilot manifest, using
   concurrency `3`.
3. Use bounded retries:
   - allow assignment-internal validation retries
   - retry the whole file once for transient/API failures
4. Leave unresolved failed files unchanged.
5. Continue the batch after failures.
6. Write all review artifacts to `.metadata-runs/`.
7. Populate or update the ontology cache with lookup evidence used during enrichment.
8. Run the enrichment auditor agent on real enriched pilot outputs and review every resulting flag.
9. Generate the standalone HTML human-review app plus dataset bundle.
10. Complete human review through the review app and export the structured review JSON.
11. Ingest the review JSON and produce the concrete fix list.
12. Run the `findingmodels` validator after the pilot batch and any accepted fixes.
13. Review all failure records, warnings, low-confidence outputs, auditor flags, and representative
    Logfire traces.
14. Update prompts/tooling only for systematic issues.
15. If auditor output is too noisy or too weak during pilot review, first improve the auditor prompt;
    add a separate `metadata_audit` model-routing tag only if the existing `metadata_assign` route is
    a demonstrated bottleneck.
16. If ontology-code fact checking reveals missing cache coverage, add cache-first lookup/fill steps
    before expanding to the full corpus rather than asking the auditor to infer ontology facts.

### Current Phase 5 Status (2026-05-01)

Phase 5 remains active. Pilot enrichment, validator execution, review-app generation, human review,
and review-export ingestion are complete. The pilot is not ready for database-build testing or
broader enrichment because the complete review surfaced systematic assignment and review-signal
problems that must be used to improve the tool first.

Current pilot state:

- 150 pilot items enriched and reviewed.
- Review export complete: 46 approved, 104 feedback, 0 drafts, 0 remaining.
- Review export copied in the data repo to
  `.metadata-runs/review-exports/talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json`.
- Review ingestion summary written to `.metadata-runs/pilot-review-ingest.json`.
- The next phase is to turn the reviewed feedback into better automation: every feedback item
  must be fixed, explicitly deferred with rationale, or marked not applicable with rationale;
  package/tool hardening must be completed; and targeted reruns must show the systematic issues are
  controlled before database-build testing or broader enrichment.

### Phase 5 Recovery Plan

The pilot review showed that the toolchain is useful but not yet safe for full-corpus enrichment.
Most feedback clusters around expected time course, anatomic location selection, age/sex specificity,
and ontology/index-code quality. The full run must wait for a recovery pass.

#### 1. Track Recovery State

- Keep this umbrella plan as the decision source for Phase 5 recovery.
- Keep execution facts, artifact paths, review counts, and observed feedback themes in
  `docs/plans/coordinated-metadata-enrichment-implementation-log-2026-04-26.md`.
- Keep data-repo operational status in
  `findingmodels-metadata/docs/plans/metadata-enrichment-phase-5-pilot-2026-04-27.md`.
- Do not treat raw `.metadata-runs/` files as the primary review surface; use them as supporting
  artifacts behind the review app and ingestion summary.

#### 2. Harden Validation and Confidence Output

- Add a shared `ConfidenceFieldKey` type for real metadata fields only:
  `body_regions`, `subspecialties`, `etiologies`, `entity_type`, `applicable_modalities`,
  `expected_time_course`, `age_profile`, `sex_specificity`, `anatomic_locations`, and `index_codes`.
- Use that type for `MetadataAssignmentDecision.field_confidence` and
  `MetadataAssignmentReview.field_confidence`; unknown keys must fail Pydantic validation and trigger
  agent retry during assignment.
- In the assignment output validator, require confidence for every field the decision sets, clears,
  or materially changes:
  - structured metadata fields with non-null decision values;
  - `index_codes` when ontology decisions affect canonical code selection;
  - `anatomic_locations` when anatomic decisions select locations.
- Add a final assembly warning if a changed field somehow lacks confidence so the review UI can flag
  the item.
- Add a targeted `FindingModelFull` validator requiring non-empty `display` on model-level
  `index_codes` and `anatomic_locations`, after scanning package fixtures and data-repo `defs/` for
  migration needs.

#### 3. Improve Assignment Semantics

- Expand expected-time-course guidance using pilot-derived patterns:
  - congenital/fixed anomalies and calcifications are usually permanent and often stable unless the
    finding is biologically progressive;
  - masses and neoplasms are usually months/years and often progressive;
  - acute injuries and inflammatory findings are often weeks/months and resolving or evolving;
  - devices, tubes, lines, and catheters are not permanent; choose weeks/months or months/years based
    on the device class;
  - measurements, classifications, and assessments should have null time course unless the modeled
    finding itself has temporal behavior.
- Strengthen age/sex defaults:
  - default to `sex-neutral` unless the anatomy or finding identity is intrinsically sex-specific;
  - default to `all_ages` unless the finding identity truly constrains age applicability;
  - handle fetal and pregnancy findings explicitly without conflating fetal applicability with patient
    sex specificity.
- Tighten ontology guidance:
  - do not store broader, narrower, related, procedure, exam, or modality-specific codes as canonical
    unless they are true equivalents for the modeled finding;
  - preserve non-canonical candidates in review output rather than `index_codes`.

#### 4. Improve Anatomic Candidate Generation

- Extend anatomic search to use explicit context from finding name, description, synonyms, and
  attribute/locality labels, not only model-generated query terms.
- Attempt exact/synonym resolution for obvious anatomy terms before semantic search.
- Include useful parent/common-ancestor candidates from the anatomic hierarchy so broad findings are
  not forced into overly specific locations.
- Prefer clinically useful scope over the most specific matched code.
- Emit warnings and low confidence when explicit anatomy appears present but no plausible anatomic
  location is selected.
- Add optional shared `AnatomicLocationIndex` parameters through `find_anatomic_locations()` and
  `assign_metadata()` so batch runs do not reopen the index per item.

#### 5. Expand Auditor Deterministic Checks

- Rename the deterministic auditor helper from ontology-only language to a general deterministic flag
  pass.
- Keep the existing ontology evidence checks.
- Add deterministic flags for:
  - anatomy implies body region;
  - anatomy implies sex specificity;
  - measurement, assessment, and technique issue entities should usually not have etiologies or
    intrinsic time course;
  - PET modality should align with `MI`, and `MI` should align with PET/NM when appropriate.
- Do not make `audit_enrichment()` silently open an anatomic DB. F1/F2 anatomy checks require a
  caller-supplied `AnatomicLocationIndex`; otherwise they are skipped.
- Update the data repo batch script to open one shared `AnatomicLocationIndex` and pass it to both
  assignment and audit.

#### 6. Resolve Pilot Feedback

- Extend or supplement `metadata_ingest_review.py` to produce a tracked resolution worksheet grouped
  by field/theme.
- For each of the 104 feedback items, mark exactly one outcome:
  - fixed in source;
  - explicitly deferred with rationale;
  - not applicable with rationale.
- Store rationale in a tracked data-repo plan or review-resolution document, not only ignored
  `.metadata-runs` output.
- After fixes, run the data repo validator and regenerate source-derived files.

#### 7. Targeted Rerun Before Full Corpus

- Rebuild local wheels and refresh the data repo wheelhouse after package changes.
- Run a dry-run targeted rerun through the normal batch/review pipeline on a subset covering the
  pilot failure modes:
  - anatomy-heavy: `abnormal_right_paratracheal_stripe`, `air_in_esophagus`, `axillary_mass`,
    `basal_cistern_effacement`, `aortic_measurements`, `disrupted_epiphyseal_metaphyseal_junction`,
    `sacroiliac_joint_disease`, `vertebral_coronal_cleft`;
  - time-course-heavy: `arterial_tortuosity`, `breast_calcification_cluster`, `fracture`,
    `pulmonary_artery_catheterization`, `tunneled_catheter`, `striated_nephrogram`,
    `early_intrauterine_pregnancy`;
  - age/sex/fetal: `acute_lung_injury_and_ards_in_children`, `fetal_chest_mass`,
    `intrauterine_growth_retardation`, `posterior_fossa_cystic_lesion`, `t2_hyperintense_renal_mass`;
  - ontology/modality/entity: `breast_malignancy_risk`, `breast_soft_tissue_lesion`,
    `mastectomy_breast_implant`, `osseous_lucent_lesion`, `traumatic_pneumatocele`,
    `focal_shadowing_pancreatic_lesion`, `increased_resistance_index_of_renal_transplant`,
    `pulmonary_vascular_engorgement`, `radiolucent_urinary_calculus`, `soft_tissue_abnormality`.
- Generate the review app for the targeted rerun and inspect it there, not from raw JSON.
- Proceed only if confidence keys are valid, changed fields have confidence, anatomy/time-course
  regressions are materially reduced, and no new systematic class of failures appears.

Status update after targeted v3 review and source patch pass:

- The hardened v3 targeted run completed all 30 targeted items and human review was ingested.
- The 30 targeted outputs were accepted into source working state, the 9 v3 reviewer feedback items
  were corrected, and a broader 150-item pilot source-patch pass resolved or explicitly deferred the
  remaining pilot feedback rows.
- The current post-feedback targeted review app is source-derived at
  `.metadata-runs/phase5-targeted-review-resolved-v1/index.html`; use it as the current local
  review surface if the Phase 5 recovery gate needs visual confirmation.
- The deterministic auditor now reports zero flags across the 150 pilot records when run with the
  merged Phase 5 recovery ontology cache.

#### 8. Incorporate Mechanistic-Check Work

The pilot run produced useful mechanistic-check work in the data repo, but that work should not
remain as a separate post-hoc script or a parallel review artifact. Fold useful rules into the normal
package pipeline before broader enrichment:

1. Treat `findingmodels-metadata/scripts/metadata_mechanistic_check.py` and its hints TOML as a
   temporary analysis aid, not a production pipeline step.
2. Keep only the high-signal rules:
   - `field_confidence` keys must be real metadata field names.
   - `FindingModelFull.index_codes` and `FindingModelFull.anatomic_locations` entries must have
     non-empty `display` values.
   - Deterministic auditor flags should cover anatomy-to-body-region alignment,
     anatomy-to-sex-specificity alignment, non-disease entity constraints, and PET/MI pairing.
3. Drop the regex name-pattern hint family unless pilot reviewers identify a concrete need for it.
4. Put hard schema constraints in Pydantic:
   - tighten `MetadataAssignmentReview.field_confidence` and `MetadataAssignmentDecision.field_confidence`
     to reject unknown keys at parse time;
   - add a targeted `FindingModelFull` validator for missing displays on model-level canonical codes,
     after scanning package fixtures and data-repo `defs/` for migration needs.
5. Put soft or data-dependent checks in the package auditor:
   - rename the deterministic helper from ontology-only language to a general deterministic flag pass;
   - keep the current ontology evidence checks;
   - add deterministic flag helpers for the four retained mechanistic rule families;
   - refine the LLM auditor prompt so it focuses on semantic judgment not already covered by
     deterministic checks.
6. Do not make `audit_enrichment()` silently open an anatomic-locations database when no index is
   passed. F1/F2 anatomy checks should require a supplied `AnatomicLocationIndex`; callers that do not
   pass one simply skip those deterministic checks. Batch enrichment should open one shared index and
   pass it through explicitly.
7. Add focused tests before deleting the temporary script:
   - unit tests for every new deterministic flag helper, using stubbed anatomic-index responses;
   - schema tests for invalid `field_confidence` keys;
   - schema tests for missing displays on model-level `index_codes` and `anatomic_locations`;
   - a small pilot-output verification run to confirm the new flags are useful and not noisy.
8. Update the data repo batch script to pass a shared `AnatomicLocationIndex` into `audit_enrichment()`
   once the package signature supports it.
9. Delete the standalone mechanistic checker and hints TOML only after replacement package behavior is
   verified against pilot outputs.
10. Preserve the new `py.typed` package markers if they were added during this work; they are useful
    packaging hygiene independent of the temporary script.
11. Update package documentation, the package CHANGELOG, and this implementation log with the final
    validation/auditor behavior and any breaking validation changes.

This recovery work is a gate before any database-build testing or broader enrichment. Do not
advance while mechanistic findings are only captured in a data-repo script or ignored raw JSONL
output, or while the complete human review remains unresolved.

### Done Criteria

- Pilot selection manifest exists.
- Pilot enrichment run artifacts are complete.
- Ontology cache entries exist for pilot ontology evidence.
- Auditor output exists for pilot enriched files.
- HTML review app exists and includes auditor flags plus ontology evidence.
- Human review was performed from the review app, not by ad hoc inspection of raw JSON alone.
- Human review JSON has been exported and ingested.
- Every successfully enriched pilot file validates.
- Generated files are synchronized after validation.
- Every pilot output has explicit human review status.
- No unresolved pilot failure is ignored.
- Every pilot review item is accepted, fixed, or explicitly deferred with rationale.
- Any prompt/tooling changes are justified by concrete pilot findings, not by overfitting to a single
  example.
- Auditor prompt/model-routing changes are based on pilot findings and do not introduce avoidable
  workflow complexity.
- All 104 feedback items from the complete pilot review are fixed, explicitly deferred with
  rationale, or marked not applicable with rationale.
- The assignment confidence schema rejects non-metadata keys and changed fields have confidence.
- The package assignment, anatomic search, and auditor changes from the Phase 5 recovery plan are
  implemented and validated.
- The targeted rerun review app shows no unresolved systematic anatomy, time-course, age/sex, or
  ontology-code failure pattern.
- Useful mechanistic-check findings have been incorporated into Pydantic validation and the package
  auditor, or explicitly deferred with rationale in this plan.
- The temporary standalone mechanistic checker is deleted or clearly retained only as a documented
  diagnostic aid outside the enrichment gate.


## Phase 6: Improve the Enrichment Tool Using the 150 Reviewed Examples

### Why This Phase Exists

The 150-item pilot review was expensive evidence about how the enrichment tool fails on real finding
models. The next step is not database building and not enriching the rest of the corpus. The next
step is to use that review to make the automated tool better, and to prove the improvement from clean
inputs before it is used more broadly.

The detailed working plan for this phase lives in:

```text
docs/plans/pilot-feedback-tooling-hardening-subplan-2026-05-05.md
```

That document is the task-level plan for this phase.

### Required Work

1. Build a comparison script that can distinguish true tool mistakes from reasonable differences,
   source-data gaps, and cases needing human judgment.
2. Triage the current clean-input differences from the reviewed pilot records using that comparison.
3. Create a version-controlled regression set from reviewed examples so prompt/tool changes cannot
   silently break cases that were already acceptable.
4. Pin the assignment model for repeatable testing. The current target is `gpt-5.4-mini`, preferably
   snapshot `gpt-5.4-mini-2026-03-17` if the current configuration supports that.
5. Refactor the assignment prompt:
   - correct the time-course rule so it uses the observable imaging duration, not the upper end of a
     range by default;
   - keep `field_confidence` guidance in one dedicated section;
   - keep only 3-4 examples, chosen because they teach reviewed tricky distinctions;
   - remove single-case patches and move those cases into tests or evals.
6. Remove the prompt instruction that favors one ontology system. Relevant hits from every searched
   ontology should remain available.
7. Move reliable rules into validation or auditing code instead of depending on prompt memory:
   - invalid confidence keys;
   - missing confidence for changed fields;
   - time course or etiology on measurements, scores, classifications, assessments,
     recommendations, and technique issues;
   - modality-language conflicts;
   - parent/child anatomy conflicts;
   - ontology cache display conflicts.
8. Improve anatomy selection so classification, score, and assessment models use the anatomy at the
   declared scope instead of mixing parent anatomy with component parts.
9. Update the auditor prompt so it names the pilot-derived problems it should catch after
   deterministic checks.
10. Rerun targeted reviewed examples from clean inputs, plus the regression set, and compare against
    reviewed outcomes.
11. Update the feedback-to-tooling coverage document only when there is actual tool evidence, not
    merely corrected source files.

### Done Criteria

- The comparison script exists, is repeatable, and writes reviewable output.
- The reviewed pilot differences are triaged as true tool errors, reasonable differences,
  source-data gaps, explicitly deferred items, or human-decision items.
- The regression set exists in the repo and is used for prompt/tool changes.
- The selected assignment model and reasoning setting are recorded in rerun output.
- Prompt changes are backed by reviewed examples and current OpenAI guidance, not by one-off case
  patches.
- Validation and auditing catch reliable repeated problems without relying only on the LLM prompt.
- Targeted clean-input reruns show the major reviewed failure patterns are controlled.
- Remaining differences are documented as acceptable, deferred with rationale, source-data blocked,
  or requiring human/domain decision.
- The coverage matrix reflects tool evidence.
- The larger corpus remains untouched during this phase.

## Phase 7: Test Both Database Builders on the Pilot-Only Enriched Repository

### Why This Phase Exists

After the pilot and tool-improvement work, only the reviewed pilot files should be enriched. The
repository is not publishable yet, but this partial state is useful for proving both database build
paths before changing the rest of the corpus.

### Required Work

1. Build the current-compatible `finding_models` DB from the pilot-only enriched `defs/` directory
   using the legacy script.
2. Compare its table/column schema to
   `docs/database-schemas/finding_models_legacy_2026-01-28.schema.json`.
3. Confirm an old/current `findingmodel` runtime can open and query the artifact.
4. Build the `finding_models_metadata` DB from the same pilot-only enriched `defs/` directory using
   the metadata script.
5. Confirm metadata columns are populated for enriched pilot models and null/empty for untouched
   models.
6. Confirm metadata-aware runtime can open, browse, search, and retrieve full models from the artifact.
7. Confirm auditor-reviewed `index_codes` appear correctly in the metadata DB with preferred display
   terms where the ontology cache provides them.
8. Do not publish either pilot-state artifact.

### Done Criteria

- Current-compatible pilot-state DB builds successfully.
- Current-compatible pilot-state DB matches
  `docs/database-schemas/finding_models_legacy_2026-01-28.schema.json`.
- Current `findingmodel` runtime can read the current-compatible pilot-state DB.
- Metadata-aware pilot-state DB builds successfully.
- Metadata-aware runtime can read/query the metadata-aware pilot-state DB.
- The pilot-state DBs are explicitly marked as validation-only and not published.

## Phase 8: Enrich the Remaining Finding Models

### Why This Phase Exists

The final publishable metadata DB requires the canonical source corpus to be enriched, not just a
pilot subset. The full run should reuse the pilot-proven workflow and should produce auditable run
artifacts for risk-based review.

### Required Work

1. Run enrichment across all `defs/*.fm.json`.
2. Use concurrency `3` unless the pilot shows a concrete reason to change it.
3. Keep bounded retry behavior from the pilot.
4. Leave unresolved failed files unchanged.
5. Continue after per-file failures.
6. Block publication until all failures are resolved or explicitly removed from scope.
7. Run the validator after the full batch.
8. Populate/update the ontology cache for full-run ontology evidence.
9. Run the enrichment auditor over full-run outputs.
10. Generate the HTML review package for all failures, warnings, low-confidence outputs, auditor
    flags, and a seeded sample of clean outputs.
11. Ingest human review JSON and produce a fix list.
12. Re-run metadata evals in this repository and inspect representative Logfire traces.
13. Update schema/docs/generated files as needed after fixes.

### Done Criteria

- Full enrichment run artifacts are complete.
- Ontology cache has lookup evidence for selected and audited index codes.
- Auditor output is complete for the reviewed full-run scope.
- Human review JSON for the full-run review scope has been ingested.
- No unresolved per-file failures remain for publishable source.
- `findingmodels` validator passes.
- Generated `text/*.md`, `index.md`, and `ids.json` are synchronized.
- Risk-based review is complete.
- Metadata evals still pass acceptable structural gates, and semantic misses are understood.

## Phase 9: Release Metadata-Aware Package Versions

### Why This Phase Exists

The data repo scripts must not merge in a state that depends on unpublished local wheels. Before the
final data branch merge and DB publish, metadata-aware package versions need to be released so scripts
can use normal version pins.

### Required Work

1. Finalize package documentation:
   - `packages/findingmodel/README.md`
   - `packages/findingmodel-ai/README.md`
   - database/configuration docs
   - metadata field docs
2. Update `CHANGELOG.md` with concise user-facing changes.
3. Bump/release `findingmodel` as `2.0.0`.
4. Release any matching package versions needed by `findingmodels` scripts.
5. Update `findingmodels` metadata scripts from local wheelhouse usage to released package pins.
6. Change the `findingmodel 2.0.0` runtime default manifest key to `finding_models_metadata` in the
   same release-preparation commit as the package version bump and documentation updates.
7. Re-run the relevant `findingmodels` scripts with released pins.

### Done Criteria

- Metadata-aware packages are released or otherwise available through the intended package channel.
- `findingmodels` scripts no longer require local wheelhouse usage for final operation.
- Package docs and changelog describe the new metadata fields, DB key, and compatibility behavior.
- Data repo validation still passes with released package pins.

## Phase 10: Build and Publish Final Databases

### Why This Phase Exists

The enriched source must produce two production artifacts from the same canonical commit. Publishing is
manual for now, and the manifest must carry enough metadata to identify source/build provenance.

### Required Work

1. From the final `findingmodels` source state, build the current-compatible DB using the legacy script.
2. From the same source state, build the metadata-aware DB using the metadata script.
3. Validate the current-compatible DB:
   - table/column schema matches `docs/database-schemas/finding_models_legacy_2026-01-28.schema.json`
   - record count matches source scope
   - current `findingmodel` runtime can open/query it
   - stored full model JSON is old/current-compatible through old tooling
4. Validate the metadata-aware DB:
   - metadata columns exist
   - full enriched JSON is stored
   - DB metadata/version table exists and is correct
   - `findingmodel 2.0.0` runtime can open/query/browse/search it
5. Publish current-compatible DB under manifest key `finding_models`.
6. Publish metadata-aware DB under manifest key `finding_models_metadata`.
7. Preserve and back up the ontology cache used as evidence for the published metadata DB.
8. Add source/build metadata to both manifest entries:
   - source `findingmodels` commit
   - tooling/package versions
   - schema kind
   - record count
   - hash
   - build timestamp
9. Back up the manifest before update.
10. Before publishing, rehearse the manifest update shape locally and confirm it contains both
    artifact keys without overwriting either entry.
11. Verify post-publish download, hash validation, runtime open/query behavior, and representative
    search/browse behavior for both artifacts.

### Done Criteria

- Both DBs are built from the same enriched source commit.
- `finding_models` remains compatible with current users.
- `finding_models_metadata` works with the metadata-aware package.
- Manifest contains both entries with correct provenance metadata.
- Manual post-publish download, hash, runtime query, and representative search/browse checks pass for
  both artifacts.

## Phase 11: Documentation Review and Plan Closeout

### Why This Phase Exists

This work spans package APIs, data repository schema, enriched content, DB publishing, and user-facing
configuration. The final docs must describe the true shipped state, not the intermediate plan.

### Required Work

1. Update this plan with final results and mark it complete only after publish verification.
2. Review active docs for stale assumptions about:
   - metadata fields
   - manifest keys
   - DB schema
   - enrichment workflow
   - package version behavior
   - CI status
3. Update user-facing changelog entries concisely.
4. Document that CI is deferred and publishing is currently manual.
5. Record follow-up items for CI wiring, if still desired.

### Done Criteria

- This plan reflects the final state and is marked complete.
- User-facing docs match the shipped behavior.
- Changelog entries describe what changed for external users.
- Follow-up CI work is captured separately and not mixed into this plan's completion criteria.

## Global Acceptance Criteria

- This repository is rebased on local `dev` and uses the OpenAI-only embedding baseline.
- `findingmodel 2.0.0` supports optional metadata fields and defaults to `finding_models_metadata`.
- `findingmodels` branch `findingmodels-metadata` contains synchronized schema, scripts, enriched
  `defs/`, generated markdown, index, and ID files.
- Pilot enrichment of about 150 representative models is fully manually reviewed.
- Full enrichment has no unresolved failures and passes risk-based review.
- HTML review package workflow is used for pilot review and full-run risk-based review.
- Enrichment auditor flags are reviewed and resolved or deferred with rationale.
- Ontology cache contains retained lookup evidence for selected/audited index codes.
- Current-compatible DB is built with old pinned tooling and matches the current published DB schema.
- Metadata-aware DB is built with new tooling and exposes enriched metadata.
- Both DB artifacts are produced from the same enriched canonical source.
- Publishing is manual, verified after upload, and records source/build provenance in the manifest.

## Explicit Non-Goals

- Do not implement CI in this plan.
- Do not enrich files under `findingmodels/conflicts/`.
- Do not make old/current clients parse new metadata fields.
- Do not publish a mixed pilot DB.
- Do not commit local wheelhouse paths or machine-specific dependency hacks.
- Do not use reviewed gold answers as hidden prompt examples for production enrichment.
- Do not let auditor-agent flags automatically rewrite canonical `.fm.json` files without human or
  scripted follow-up review.
