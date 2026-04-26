# Plan: Coordinated Metadata Enrichment and Dual-Database Release

Status: Draft (2026-04-26)

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
5. Run the metadata assignment eval suite with Logfire enabled.
6. Inspect representative Logfire traces, not just aggregate scores.
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
- Metadata evals run successfully with Logfire traces available for inspection.
- The legacy/current-compatible DB schema contract artifact is checked in.
- Any changed assumptions are documented in this plan before proceeding.

## Phase 2: Finalize Package Capabilities in This Repository

### Why This Phase Exists

The `findingmodels` branch will depend on package behavior from this repository. Before enriching
canonical data, we need the model schema, assignment behavior, DB builders, and publish primitives to
be coherent enough that the data repo scripts can call them reliably.

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
11. Prepare package versioning for the metadata-aware line, targeting `findingmodel 2.0.0`.

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
- Documentation and changelog updates needed for package users are identified.

## Phase 3: Build Local Wheels for Data-Repo Iteration

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
7. Keep detailed run outputs under `.metadata-runs/`, untracked.
8. Add a guard or dependency pin so `scripts/validator.py` cannot silently run with
   `findingmodel<2.0.0` once enriched metadata fields are present.
9. Regenerate `schema/finding_model.schema.json` using the metadata-aware package.
10. Manually update `schema/finding_model_schema.md` to document the new optional metadata fields.
11. Run the existing validator so generated `text/*.md`, `index.md`, and `ids.json` are synchronized.

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

The HTML review-package script must:

- use the same general pattern as the gold-standard review process
- generate a static review package for pilot and full-run review sets
- include original model metadata state, enriched model metadata state, and concise before/after diffs
- include selected `index_codes`, anatomic locations, and ontology lookup evidence from the cache
- include auditor flags and severity
- include assignment review warnings and confidence fields
- capture human reviewer status and comments in a structured review JSON file

The human-review ingestion script must:

- read the structured review JSON file
- summarize accepted, rejected, corrected, deferred, and needs-discussion items
- produce a concrete fix list for prompts, package code, ontology cache corrections, or source model
  edits
- block progression from pilot to full enrichment until every pilot review item is accepted, fixed, or
  explicitly deferred with rationale

The legacy DB build/publish script must:

- use PEP 723 script dependencies
- pin `findingmodel==1.0.4`
- pin `oidm-common==0.2.7`
- pin `oidm-maintenance` by Git URL to commit
  `75afd39a400419dcfaf7c8d4a34f065b4d804e0d`
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

## Phase 5: Pilot Enrichment

### Why This Phase Exists

Before running enrichment over 2,149 models, we need a representative proof that the prompt, assignment
API, run artifacts, validator, generated docs, and DB production path work together on real canonical
source files. The pilot is a quality and workflow gate, not a publishable dataset.

### Required Work

1. Use the pilot selection script to choose about 150 representative models.
2. Run enrichment on the selected files using concurrency `3`.
3. Use bounded retries:
   - allow assignment-internal validation retries
   - retry the whole file once for transient/API failures
4. Leave unresolved failed files unchanged.
5. Continue the batch after failures.
6. Write all review artifacts to `.metadata-runs/`.
7. Populate or update the ontology cache with lookup evidence used during enrichment.
8. Run the enrichment auditor agent on pilot outputs.
9. Generate the static HTML human-review package.
10. Complete human review through the HTML package and export the structured review JSON.
11. Ingest the review JSON and produce the concrete fix list.
12. Run the `findingmodels` validator after the pilot batch and any accepted fixes.
13. Review all failure records, warnings, low-confidence outputs, auditor flags, and representative
    Logfire traces.
14. Update prompts/tooling only for systematic issues.

### Done Criteria

- Pilot selection manifest exists.
- Pilot enrichment run artifacts are complete.
- Ontology cache entries exist for pilot ontology evidence.
- Auditor output exists for pilot enriched files.
- HTML review package exists and includes auditor flags plus ontology evidence.
- Human review JSON has been exported and ingested.
- Every successfully enriched pilot file validates.
- Generated files are synchronized after validation.
- Every pilot output has explicit human review status.
- No unresolved pilot failure is ignored.
- Every pilot review item is accepted, fixed, or explicitly deferred with rationale.
- Any prompt/tooling changes are justified by concrete pilot findings, not by overfitting to a single
  example.

## Phase 6: Mixed-Source Dual DB Proof

### Why This Phase Exists

After the pilot, the `defs/` directory is intentionally mixed: some files enriched, most untouched.
This mixed state is not publishable, but it is useful for proving both database production paths can
operate from the same canonical source layout before full enrichment.

### Required Work

1. Build the current-compatible `finding_models` DB from the mixed `defs/` directory using the legacy
   script.
2. Compare its table/column schema to the current published DB schema.
3. Confirm an old/current `findingmodel` runtime can open and query the artifact.
4. Build the `finding_models_metadata` DB from the same mixed `defs/` directory using the metadata
   script.
5. Confirm metadata columns are populated for enriched pilot models and null/empty for untouched
   models.
6. Confirm metadata-aware runtime can open, browse, search, and retrieve full models from the artifact.
7. Confirm auditor-reviewed `index_codes` appear correctly in the metadata DB with preferred display
   terms where the ontology cache provides them.
8. Do not publish either mixed-source artifact.

### Done Criteria

- Legacy mixed-source DB builds successfully.
- Legacy mixed-source DB has the exact current published table/column schema.
- Current `findingmodel` runtime can read the legacy mixed-source DB.
- Metadata mixed-source DB builds successfully.
- Metadata-aware runtime can read/query the metadata mixed-source DB.
- The mixed-source DBs are explicitly marked as validation-only and not published.

## Phase 7: Full Corpus Enrichment

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
10. Generate a review package for all failures, warnings, low-confidence outputs, auditor flags, and
    a seeded sample of clean outputs.
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

## Phase 8: Release Package Line

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
6. Re-run the relevant `findingmodels` scripts with released pins.

### Done Criteria

- Metadata-aware packages are released or otherwise available through the intended package channel.
- `findingmodels` scripts no longer require local wheelhouse usage for final operation.
- Package docs and changelog describe the new metadata fields, DB key, and compatibility behavior.
- Data repo validation still passes with released package pins.

## Phase 9: Final DB Build and Manual Publish

### Why This Phase Exists

The enriched source must produce two production artifacts from the same canonical commit. Publishing is
manual for now, and the manifest must carry enough metadata to identify source/build provenance.

### Required Work

1. From the final `findingmodels` source state, build the current-compatible DB using the legacy script.
2. From the same source state, build the metadata-aware DB using the metadata script.
3. Validate the current-compatible DB:
   - exact current published table/column schema
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
10. Verify post-publish download and hash validation.

### Done Criteria

- Both DBs are built from the same enriched source commit.
- `finding_models` remains compatible with current users.
- `finding_models_metadata` works with the metadata-aware package.
- Manifest contains both entries with correct provenance metadata.
- Manual post-publish download/read checks pass.

## Phase 10: Documentation Review and Plan Closeout

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
