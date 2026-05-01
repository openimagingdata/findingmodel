# Implementation Log: Coordinated Metadata Enrichment and Dual-Database Release

This log records execution details for
`docs/plans/coordinated-metadata-enrichment-and-dual-db-release-2026-04-26.md`.
The plan should stay focused on decisions, phases, requirements, and gates. This file holds command
history, smoke-test notes, trace IDs, and local artifact paths.

## Phase 1: Rebase and Stabilize

Rebase status:

- `feature/metadata-cleanup` was rebased onto local `dev`.
- Conflict resolution preserved `dev`'s OpenAI-only embedding baseline.
- Manual conflict resolutions were limited to `scripts/check_anatomic_embeddings.py` and
  `scripts/benchmark_models.py`.
- The metadata assignment prompt, assignment implementation, eval harness, eval fixtures, and
  `metadata_assign` model-routing configuration were not manually changed as part of conflict
  resolution.

Verification:

- `uv run ruff check scripts/check_anatomic_embeddings.py scripts/benchmark_models.py`: passed.
- `uv run pytest packages/findingmodel/tests`: `387 passed, 2 skipped`.
- `uv run pytest packages/oidm-common/tests`: `142 passed`.
- `uv run pytest packages/oidm-maintenance/tests`: `154 passed`.
- `uv run pytest packages/findingmodel-ai/tests`: `247 passed, 10 skipped`.
- One expected-style warning was emitted because a unit test entered a Logfire span without global
  Logfire configuration.

Metadata eval / Logfire decision:

- Full metadata assignment evals were not run during Phase 1 because the conflict resolution did not
  change enrichment-affecting code.
- A Logfire smoke run was also not run. Traced metadata evals remain required before pilot
  enrichment or after any prompt, assignment, ontology, auditor, or model-routing change that could
  affect enrichment outputs.

Legacy schema contract:

- The current published DB schema was verified from the live `finding_models` DuckDB artifact listed
  in the manifest as version `2026-01-28`.
- The legacy/current-compatible schema contract artifact was added at
  `docs/database-schemas/finding_models_legacy_2026-01-28.schema.json`.

## Phase 2: Package Capabilities

Opening audit findings:

- `FindingModelFull` and `FindingModelBase` include the planned optional metadata fields.
- `FindingModelFull.model_json_schema()` includes metadata fields and supporting enum/model
  definitions.
- Markdown rendering already includes model-level codes and structured metadata fields.
- `findingmodel-ai` exposes `assign_metadata()` as a package-level callable.
- Metadata assignment review output includes model used, assignment mode, Logfire trace ID, ontology
  candidates, anatomic candidates, rationale, field confidence, timings, and warnings.
- The metadata-aware DB builder already stores structured metadata columns and hydrates them through
  `FindingModelIndex`.

Gaps found:

- No metadata-aware DB build provenance/version table existed.
- Findingmodel publish tooling still hardcoded the `finding_models` manifest key and artifact path.
- Durable ontology lookup evidence caching did not exist.
- No dedicated enrichment auditor agent existed.

Implemented:

- Added a `database_metadata` table to metadata-aware findingmodel DB builds.
- Parameterized findingmodel publish tooling with `manifest_key`, `s3_prefix`, and `artifact_name`.
- Added CLI options for metadata DB build provenance and metadata artifact publishing targets.
- Added a DuckDB-backed ontology lookup evidence cache in `findingmodel-ai`.
- Added optional ontology-cache recording to `assign_metadata()` without changing assignment
  decisions or prompts.
- Added `--ontology-cache` to the `findingmodel-ai assign-metadata` CLI.
- Added a lightweight enrichment auditor wrapper with deterministic cache lookup, deterministic
  missing-evidence/code-display flags, and a Pydantic AI sanity-check pass.
- Added `FINDINGMODEL_DB_MANIFEST_KEY` for manifest key override.
- Preserved the current runtime default manifest key as `finding_models`.
- Deferred the default flip to `finding_models_metadata` until the `findingmodel 2.0.0` release gate.
- Refactored the ontology lookup cache onto the shared DuckDB connection helper and a held
  connection/context-manager lifecycle.
- Added `relationship` and `rejection_reason` columns to ontology cache evidence while the cache
  schema is still uncommitted.
- Updated relevant package READMEs, database management docs, and user-facing changelog entries.

Design decisions made during Phase 2:

- The ontology lookup cache uses DuckDB, not SQLite, to keep local durable artifacts/query tooling
  consistent with the rest of the stack.
- The enrichment auditor remains a lightweight Pydantic AI wrapper, not a large workflow system.
- The auditor prompt reviews an already enriched model and flags likely issues; it does not
  re-enrich or act as final authority.
- The auditor gets deterministic ontology-code evidence from the DuckDB cache and must not invent
  ontology facts.

Verification highlights:

- Focused DB/publish tests passed after initial implementation.
- Additional `findingmodel-ai` and `oidm-maintenance` tests passed after ontology cache/auditor work.
- Manifest-key override tests passed.
- Broader package-scope validation passed:
  - `uv run ruff check packages/findingmodel packages/findingmodel-ai packages/oidm-maintenance packages/oidm-common`
  - `uv run pytest packages/findingmodel/tests`: `388 passed, 2 skipped`.
  - `uv run pytest packages/findingmodel-ai/tests`: `253 passed, 10 skipped`.
  - `uv run pytest packages/oidm-common/tests`: `142 passed`.
  - `uv run pytest packages/oidm-maintenance/tests`: `158 passed`.
- Repeated assignment unit-test runs emitted the same Logfire-not-configured warning from a unit test
  that enters a Logfire span without global configuration.

Metadata eval / Logfire decision:

- Full metadata assignment evals and Logfire trace review were not run for Phase 2 DB/publish/cache
  plumbing because those changes did not alter assignment prompts, assignment behavior, ontology
  selection logic, eval fixtures, eval harness code, or metadata model routing.

## Phase 3: Local Wheelhouse

Initial wheelhouse path:

```text
/tmp/findingmodel-metadata-wheelhouse/a8b21b0
```

Initial build command:

```bash
uv build --all-packages --wheel --out-dir /tmp/findingmodel-metadata-wheelhouse/a8b21b0 --no-create-gitignore
```

Built wheels:

```text
anatomic_locations-0.2.5-py3-none-any.whl
findingmodel-1.0.4-py3-none-any.whl
findingmodel_ai-0.2.1-py3-none-any.whl
oidm_common-0.2.7-py3-none-any.whl
oidm_maintenance-0.2.5-py3-none-any.whl
```

`uv build` emitted a `uv_build` compatibility warning because package build-system requirements
still targeted `uv_build>=0.10,<0.11`.

Packaging correction:

- Updated package `[build-system].requires` entries to `uv_build>=0.11.7,<0.12`.
- Rebuilt with no compatibility warning.
- Confirmed `uv build --all-packages --wheel --no-sources` also succeeded.

Final wheelhouse path:

```text
/tmp/findingmodel-metadata-wheelhouse/phase3-uvbuild-0.11.7
```

Verification:

- Imports resolved for `findingmodel`, `findingmodel-ai`, `oidm-common`, `oidm-maintenance`, and
  `anatomic-locations`.
- Installed package provenance confirmed `direct_url.json` entries pointing to local wheelhouse file
  URLs for all five wheels.
- Verified metadata-aware behavior from the wheel environment:
  - `FindingModelConfig().db_manifest_key == "finding_models"`
  - `findingmodel_ai.metadata.OntologyLookupCache` is importable
  - `findingmodel_ai.metadata.audit_enrichment` is importable
  - `oidm_maintenance.findingmodel.build.build_findingmodel_database` exposes `schema_name`,
    `schema_version`, and `source_commit` parameters.
- No committed dependency file, lockfile, or local path wiring changed in either repository.

## Phase 4: `findingmodels-metadata` Branch Preparation

Implemented scripts:

- `scripts/metadata_select_pilot.py`
- `scripts/metadata_assign_batch.py`
- `scripts/metadata_audit.py`
- `scripts/metadata_review_package.py`
- `scripts/metadata_ingest_review.py`
- `scripts/build_legacy_findingmodel_db.py`
- `scripts/build_metadata_findingmodel_db.py`

Important implementation notes:

- `.metadata-runs/` is ignored and is the local home for wheelhouse, pilot manifests, enrichment
  outputs, review apps, ontology cache, and database smoke outputs.
- Metadata-aware scripts use PEP 723 `[tool.uv.sources]` entries pointing at local wheel files under
  `.metadata-runs/wheelhouse/current/`.
- The legacy DB script intentionally uses pinned current-compatible Git dependencies, not local
  metadata-aware wheels.
- `scripts/output_schema.py` and `scripts/validator.py` are pinned to local metadata-aware wheels
  during this branch work.
- `scripts/validator.py` preserves existing `.fm.json` filenames when regenerating markdown and
  `index.md`.
- `scripts/metadata_assign_batch.py` has an explicit `--logfire` switch.
- The reviewer-facing path is `.metadata-runs/review-current/index.html`.

Smoke and validation notes:

- `metadata_select_pilot.py` selected five files and wrote a pilot manifest.
- The first `metadata_review_package.py` attempt generated the wrong kind of static page; inspection
  of `../review_tool` established that the intended artifact is a standalone review app plus
  `review-data.json`.
- Help commands resolved for metadata audit, review ingestion, batch assignment, and both DB build
  scripts.
- `metadata_ingest_review.py` accepts review-tool export JSON with `responses[]`.
- One-model metadata-aware and current-compatible DuckDB smoke builds succeeded.
- `output_schema.py` regenerated the metadata-aware JSON schema successfully.
- `validator.py` completed successfully with local metadata-aware wheels.
- Ruff passed for new and modified Phase 4 scripts.

Live enrichment smoke:

- A no-key dry-run exercised failure recording and source preservation, then failed immediately with
  the expected provider-key configuration error.
- A one-item live dry-run with `.env`, Logfire, and ontology cache succeeded for
  `defs/abdominal_abscess.fm.json`.
- One-item trace ID: `019dcb847b6da968b268cf239fc781fa`.
- Logfire trace review showed the assignment span took about 15 seconds: ontology candidate
  gathering about 7.7 seconds, anatomic candidate gathering about 8.4 seconds, classifier about
  6.6 seconds, and final assembly negligible.

Review app smoke:

- The review app renders readable structured metadata, optional run warnings/errors, collapsed field
  confidence, and collapsed run details.
- Empty warning sections are suppressed.
- Raw before/after metadata, metadata diffs, ontology candidate review, cache evidence, and raw audit
  JSON are not dumped into the default reviewer surface.
- The generator rejects requested paths that do not have completed enrichment outputs.
- A three-path smoke attempt correctly failed when two paths lacked enriched snapshots.
- A real three-item dry-run enrichment smoke completed successfully with concurrency `3` for
  `defs/abdominal_abscess.fm.json`, `defs/aortic_dissection.fm.json`, and
  `defs/adrenal_nodule.fm.json`.
- Three-item run directory: `.metadata-runs/smoke-live-enrichment-3item-v2`.
- The three-item run was regenerated into `.metadata-runs/review-current/index.html`.
- The three-item `review-data.json` contains item IDs `abdominal_abscess`, `adrenal_nodule`, and
  `aortic_dissection`.
- The auditor raised one or more flags for each of the three smoke outputs, so each item showed a
  visible run warning section.

Review ingestion smoke:

- Approved review export ingestion produced a complete, non-actionable summary.
- Feedback review export ingestion produced a complete-but-actionable summary with
  `requires_follow_up=true` and `blocks_progression=false`.

Known cleanup after log split:

- The Phase 4 review-package generator was rewritten to be a thin adapter after the first version was
  judged overcomplicated.
- A subagent review found the rewrite understandable and appropriately scoped, with one required fix:
  dedupe append-only `status.jsonl` rows by latest successful path to avoid duplicate review items.
- A worker implemented that dedupe plus relative `--run-dir`/`--output-dir` normalization in
  `scripts/metadata_review_package.py`, and Ruff passed for that script.
- `docs/metadata-enrichment-setup.md` was added in the data repo so the local wheelhouse and basic
  smoke workflow are reproducible without reading this implementation log.
- `scripts/metadata_review_template.html` was marked as an adaptation of `../review_tool`.
- A package-level `assign_and_audit()` helper was deliberately not added. The current orchestration is
  a single direct assignment-then-audit sequence inside the batch script; adding a public helper now
  would add package API surface before there is enough duplicated workflow to justify it.
- Follow-up Phase 4 code review fixes:
  - `metadata_assign_batch.py` now runs audit before writing source files, records tracebacks on
    failures, checks `--skip-completed` before parsing, and retries transient whole-file failures once
    by default.
  - `output_schema.py` now has the metadata-aware package guard used by `validator.py`.
  - `metadata_audit.py` now requires explicit paths or `--all` and uses bounded concurrency.
  - `metadata_review_package.py` now uses a stable review ID based on run directory and source item
    identity rather than timestamps/timings.
  - The review export now includes source `path`.
  - Metadata DB source commit provenance marks dirty worktrees and the development schema version
    defaults to `2.0.0-dev`.
  - `IndexCode.code.minLength == 2` was confirmed from `oidm-common` source and is expected in the
    regenerated schema.

## Phase 5: Pilot Enrichment Support Fixes

During the first 150-item pilot run, three package-level issues surfaced before human review:

- DuckDB extension cache writes defaulted to the user home in sandboxed local runs. The pilot command
  now sets `HOME` to the data repo's ignored `.metadata-runs/home` directory.
- Ontology lookup/cache paths could open unnecessary DuckDB FTS/VSS extensions and could leave
  path-owned cache connections open if assignment or audit failed. The ontology cache now opens
  without search/vector extensions, and assignment/audit close path-owned cache connections in
  `finally` blocks.
- Ontology labels such as `T1` and `T2` are valid ontology text but invalid `IndexCode.display`
  values because `display` has `minLength == 3`. `OntologySearchResult.as_index_code()` now omits
  too-short displays rather than failing validation.

Verification:

- `uv run ruff check packages/findingmodel/src/findingmodel/protocols.py
  packages/findingmodel/tests/test_protocols.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/auditor.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/ontology_cache.py
  packages/oidm-common/src/oidm_common/duckdb/connection.py`: passed.
- `uv run pytest packages/findingmodel/tests/test_protocols.py`: `1 passed`.
- `uv run pytest packages/oidm-common/tests/test_duckdb.py`: `36 passed`.
- `uv run pytest packages/findingmodel-ai/tests/test_ontology_cache.py
  packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_enrichment_auditor.py`: `10 passed`.
- Broader package verification after the pilot fixes:
  - `uv run pytest packages/findingmodel/tests`: `389 passed, 2 skipped`.
  - `uv run pytest packages/oidm-common/tests`: `142 passed`.
  - `uv run pytest packages/findingmodel-ai/tests`: `253 passed, 10 skipped`.
- Local wheels were rebuilt into `/tmp/findingmodel-metadata-wheelhouse/current` and recopied into
  the data repo wheelhouse.

Pilot status after fixes:

- All 150 pilot items have completed enrichment artifacts.
- `.metadata-runs/review-current/index.html` was regenerated from the completed pilot run.
- The data repo validator completed successfully after pilot enrichment.
- Human review export ingestion is complete, but Phase 6 remains blocked until the actionable
  feedback is either fixed or explicitly deferred with rationale and the required tool hardening is
  complete.

Pilot human review received 2026-05-01:

- Review export copied in the data repo to
  `.metadata-runs/review-exports/talkasab-mgh-harvard-edu-metadata-enrichment-review-responses.json`.
- Review ingestion summary written to `.metadata-runs/pilot-review-ingest.json`.
- The export is complete: 150 total, 150 done, 46 approved, 104 feedback, 0 drafts, 0 remaining.
- The review is therefore sufficient as a complete pilot review artifact, but the feedback rate is
  high enough that broader enrichment should not proceed before tool changes.

Feedback themes from the 104 actionable comments:

- Expected time course is the dominant issue. About half of feedback comments mention missing or
  incorrect duration/progression/resolution. The prompt needs stronger defaults and examples for
  congenital/permanent findings, masses/neoplasms, acute injuries, calcifications, devices/tubes, and
  measurements/classifications.
- Anatomic location selection is the next largest issue. Reviewers repeatedly flagged missing obvious
  anatomy, wrong anatomy, or overly specific anatomy. Examples include missing esophagus, mediastinum,
  axilla, aorta, hippocampus, sacroiliac joints, spine, larynx, kidney/renal cortex, and urinary tract;
  over-specific selections such as head of fibula, right atrium, and sacrum were also flagged.
- Age and sex specificity are often too restrictive or omitted. The prompt should default to
  sex-neutral and all/any age unless the finding itself, not just a common demographic, truly
  constrains applicability. Pregnancy/fetal findings need explicit handling so fetal applicability is
  not conflated with patient sex specificity.
- Ontology/index-code issues remain meaningful. Feedback called out missing expected codes, codes that
  were too broad or too specific, modality-specific codes stored on multi-modality findings, and
  inappropriate BI-RADS/classification codes.
- Etiology, modality, and subspecialty issues were less common but still systematic enough to require
  prompt and auditor attention before full-corpus enrichment.

Tooling implications before larger corpus:

- Tighten `field_confidence` to actual metadata fields only. The pilot review data contains invalid
  confidence keys such as `ontology_decisions` and `anatomic_decisions`, while reviewer comments on
  `anatomic_locations` were not represented as low/non-high confidence on that field.
- Require confidence entries for every metadata field the agent sets, clears, or materially changes;
  the review UI should flag changed fields with missing confidence as needing attention.
- Improve anatomic candidate generation and selection:
  - search explicit anatomy terms from the finding name, description, synonyms, and attribute/locality
    labels before relying on model-generated query terms alone;
  - include parent/common-ancestor locations from the anatomic hierarchy so broad findings do not get
    forced into too-specific locations;
  - if an explicit anatomy term cannot be resolved, emit a warning and low confidence rather than
    silently leaving `anatomic_locations` empty;
  - prefer clinically useful scope over the most specific matched code.
- Strengthen ontology candidate handling:
  - do not store broader/narrower/related or modality-specific codes as canonical `index_codes`;
  - improve recall for RadElement/RadLex/SNOMED/LOINC candidates where the finding name strongly
    implies an available code;
  - surface "no plausible code found" as a warning when reviewer-facing terminology strongly suggests
    an ontology concept should exist.
- Fold the useful mechanistic-check rules into package validation/auditing before Phase 6:
  field-confidence key validation, model-level display validation for canonical codes, deterministic
  anatomy/body-region and anatomy/sex checks, non-disease entity constraints, and PET/MI pairing.
- Expand the auditor and review UI around the patterns the pilot actually exposed. The auditor raised
  warnings for 51 items, but 64 feedback items had no run warning and 11 approved items did have a
  warning, so auditor output is useful triage but not yet a reliable substitute for human review.
- After changes, rerun a smaller targeted pilot subset containing anatomy-heavy, time-course-heavy,
  device/tube, breast, spine, vascular, and pediatric/fetal findings before running the full corpus.

Recovery planning decisions recorded 2026-05-01:

- Current phase remains Phase 5. Phase 6 and full-corpus enrichment remain blocked.
- Approved gate: harden tooling, fix or defer all pilot feedback, then rerun a targeted subset before
  any full-corpus enrichment.
- Approved pilot-feedback policy: each of the 104 actionable feedback items must be fixed, explicitly
  deferred with rationale, or marked not applicable with rationale.
- Approved time-course strategy: improve prompt guidance plus auditor flags; do not add broad
  deterministic time-course defaults that silently overwrite model judgment.
- The targeted rerun should cover anatomy-heavy, time-course-heavy, device/tube, breast, spine,
  vascular, ontology/modality, and pediatric/fetal cases before broader enrichment resumes.
- The larger coordinated plan now contains the authoritative Phase 5 recovery plan; this log records
  the facts and decisions that motivated it.
