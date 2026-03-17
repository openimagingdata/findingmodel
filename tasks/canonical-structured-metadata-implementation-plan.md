# Canonical Structured Metadata Implementation Plan

**Status:** In progress — Slice 1, Slice 2, and Slice 3 complete, pending commit

**Design Spec:** [../docs/canonical-structured-metadata-and-enrichment-rewrite.md](../docs/canonical-structured-metadata-and-enrichment-rewrite.md)

**This plan is the execution document.** The design doc above remains the source of truth for architecture, canonical values, and semantic rules.

**Supersedes for this workstream:** [facets-implementation-plan.md](./facets-implementation-plan.md)

## Goal

Implement the canonical structured metadata rewrite as a sequence of small, reviewable slices that can be:
- coded independently where possible
- tested before the next slice starts
- documented as they land
- parallelized without overlapping ownership more than necessary

## Execution Rules

- Treat the design doc as authoritative. If implementation pressure reveals a design gap, update the design doc first, then continue.
- Land this work in discrete slices. Do not combine core schema, retrieval API, enrichment rewrite, and backfill into one change.
- Every slice must include:
  - code changes
  - targeted tests
  - documentation updates for the surfaces changed in that slice
- Prefer additive internal scaffolding first, then coordinated public-surface migrations.
- Do not preserve old enrichment pathways with compatibility shims unless the design doc is changed explicitly.

## Parallel Work Map

### Lane A: Core Model Surface
- `packages/findingmodel/src/findingmodel/finding_model.py`
- `packages/findingmodel/src/findingmodel/__init__.py`
- new shared facet type module(s) in `packages/findingmodel/src/findingmodel/`
- markdown/template/editor/stub round-trip code

### Lane B: Index and Build Surface
- `packages/findingmodel/src/findingmodel/index.py`
- `packages/findingmodel/src/findingmodel/protocols.py`
- `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`
- relevant DuckDB tests and fixtures

### Lane C: Enrichment Rewrite
- `packages/findingmodel-ai/src/findingmodel_ai/enrichment/`
- `packages/findingmodel-ai/src/findingmodel_ai/search/ontology.py`
- `packages/findingmodel-ai/src/findingmodel_ai/search/anatomic.py`
- enrichment tests/evals/scripts

### Lane D: Retrieval and Similar-Model Rewrite
- `packages/findingmodel-ai/src/findingmodel_ai/search/similar.py`
- `packages/findingmodel-ai/evals/similar_models.py`
- CLI/MCP callers after retrieval APIs stabilize

## Dependency Order

1. Slice 0 must land first.
2. Slice 1 unlocks all other lanes.
3. Slice 2 and Slice 3 can proceed in parallel after Slice 1.
4. Slice 4 depends on Slice 3.
5. Slice 5 depends on Slice 3 and should land before Slice 8.
6. Slice 6 can start after Slice 1, but Slice 7 depends on Slice 6 and Slice 3.
7. Slice 8 depends on Slice 4, Slice 5, and enough of Slice 7 to provide the new candidate-planning/output types.
8. Slice 9 depends on Slice 7 and Slice 8.
9. Slice 10 is last.

## Slice 0: Create and Align Planning Artifacts

**Scope**
- Create this implementation plan.
- Add a pointer from the design doc to this plan.
- Record the public surfaces that must migrate.

**Files**
- `tasks/canonical-structured-metadata-implementation-plan.md`
- `docs/canonical-structured-metadata-and-enrichment-rewrite.md`

**Acceptance Gate**
- The implementation plan exists and clearly separates design from execution.
- The design doc links to this plan.

**Docs Gate**
- No additional docs required beyond the plan and design-doc pointer.

## Slice 1: Add Shared Canonical Types and Model Fields ✓

**Status:** Complete

**Scope**
- Add shared facet types to core `findingmodel`.
- Extend `FindingModelBase` and `FindingModelFull` with the canonical optional fields.
- Capture `IndexCode` semantic rules in code docstrings/comments where appropriate.
- Export the new types from `findingmodel.__init__` if they are intended to be public.

**Primary ownership**
- Lane A

**Files changed**
- `packages/findingmodel/src/findingmodel/facets.py` — new module with 9 enums, 2 Pydantic models, normalization helpers with BeforeValidator integration
- `packages/findingmodel/src/findingmodel/finding_model.py` — 8 optional metadata fields on FindingModelBase and FindingModelFull
- `packages/findingmodel/src/findingmodel/__init__.py` — 11 new public exports
- `packages/findingmodel/tests/test_facets.py` — 85 tests
- `packages/findingmodel/README.md` — updated with structured metadata docs

**Design doc updates during implementation**
- `ExpectedDuration` replaces `TimeCourseDuration` — concrete time-range bins (hours/days/weeks/months/years/permanent) replace vague qualitative labels (rapid/short_term/intermediate/long_term)
- `TimeCourseModifier` revised — dropped `relapsing` and `waxing_waning`, added `recurrent` and `fluctuating` (aligned with SNOMED CT)
- Duration semantics clarified: upper bound on how long the finding typically remains visible on imaging

**Tests**
- model validation for all new canonical types
- serialization/deserialization with missing fields and populated fields
- normalization helper tests for legacy values
- normalization-on-load tests (legacy values auto-normalize via BeforeValidator)
- AgeProfile empty-list rejection

**Acceptance Gate**
- ✓ The core model schema matches the design doc.
- ✓ Older JSON still loads with new fields absent.
- ✓ Canonical type validation is covered directly.

**Docs Gate**
- ✓ Design doc updated for ExpectedDuration/TimeCourseModifier revision.
- ✓ `packages/findingmodel/README.md` updated with structured metadata field table and examples.

## Slice 2: Minimal Markdown Output and Authoring Surface ✓

**Status:** Complete

**Scope**
- Add structured metadata to one-way Markdown rendering and the existing authoring export surface.
- Keep Markdown tooling usable without treating Markdown as a canonical round-trip contract.

**Primary ownership**
- Lane A

**Files changed**
- `packages/findingmodel/src/findingmodel/facets.py` — added NormalizedAgeProfile, format_age_profile(), format_time_course()
- `packages/findingmodel/src/findingmodel/fm_md_template.py` — 8 conditional Jinja blocks for metadata fields
- `packages/findingmodel/src/findingmodel/finding_model.py` — age_profile uses NormalizedAgeProfile; both as_markdown() pass metadata to template
- `packages/findingmodel-ai/src/findingmodel_ai/authoring/editor.py` — _render_structured_metadata_lines() for editor export
- `packages/findingmodel/tests/test_facets.py` — 10 new tests (formatting helpers, NormalizedAgeProfile)
- `packages/findingmodel/tests/test_models.py` — markdown rendering tests, canary round-trip test
- `packages/findingmodel-ai/tests/test_model_editor.py` — editor export with metadata test

**Scope decisions**
- Follow-up overhaul tracked separately in `tasks/markdown-output-overhaul-plan.md`.
- `create_stub.py` — no changes. Stubs are intentionally minimal; metadata is added via enrichment.
- `markdown_in.py` remains an AI-assisted outline importer, not a faithful importer for exported Markdown.
- `edit_model_markdown()` remains convenience tooling for human/LLM edits, not a canonical interchange path.
- Markdown field ordering: entity type → body regions → modalities → subspecialties → etiologies → time course → age profile → sex specificity

**Tests**
- formatting helpers for AgeProfile and ExpectedTimeCourse
- NormalizedAgeProfile on model load (legacy string → AgeProfile)
- markdown rendering with all 8 metadata fields (Base and Full models)
- editor export with metadata fields
- canary content test: JSON → model → markdown renders the expected metadata labels and values

**Acceptance Gate**
- ✓ Model → markdown renders all metadata fields correctly
- ✓ Editor export includes metadata lines
- Markdown output is explicitly treated as presentation / convenience authoring text, not canonical serialized state

**Docs Gate**
- ✓ README updated with structured metadata field table (done in Slice 1)
- ✓ Markdown docs/configuration updated to describe these APIs as presentation / convenience tooling
- ✓ Implementation plan updated with scope decisions and follow-up overhaul pointer

## Slice 3: Extend DuckDB Build, Storage, and Hydration ✓

**Status:** Complete

**Scope**
- Add the new queryable columns to the maintainer build and read side.
- Extend `IndexEntry` and hydration so the structured fields are available from the index.
- Keep facets as structured columns only; do not duplicate them into FTS/search_text or embedding_text.

**Primary ownership**
- Lane B

**Files changed**
- `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py` — added structured metadata DuckDB columns, build-side value extraction, and normalized `index_code_keys`
- `packages/findingmodel/src/findingmodel/index.py` — extended `IndexEntry` and hydration for canonical facet fields, `ExpectedTimeCourse`, `AgeProfile`, `anatomic_location_ids`, and `index_code_keys`
- `packages/oidm-maintenance/tests/test_findingmodel_build.py` — schema, populated/null hydration, and “not in search/embedding text” coverage
- `packages/findingmodel/tests/data/test_index.duckdb` — regenerated committed fixture with the updated schema
- `docs/canonical-structured-metadata-and-enrichment-rewrite.md` — updated to keep facets structured rather than duplicated into retrieval text

**Files likely in scope**
- `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`
- `packages/findingmodel/src/findingmodel/index.py`

**Tests**
- exact schema/DDL assertions
- build round-trip tests
- hydration tests for populated and null facet columns
- `packages/oidm-maintenance/tests/test_findingmodel_build.py`
- `packages/findingmodel/tests/test_findingmodel_index.py`

**Acceptance Gate**
- ✓ Database build and read-side hydration fully support the canonical fields.
- ✓ `index_code_keys` are normalized consistently as `system:code`.
- ✓ FTS/search text and embedding text remain focused on free-text clinical content rather than duplicated facet values.

**Docs Gate**
- ✓ Design doc updated to reflect the “structured columns only” retrieval decision.

## Slice 4: Add `browse(...)` and Facet-Aware Search APIs

**Scope**
- Add `browse(...)` to the index API with pagination-friendly return shape.
- Extend `search(...)` and `search_batch(...)` with facet filters.
- Keep `tags` as ALL-of while other multi-value facets use OR-within-facet semantics.

**Primary ownership**
- Lane B

**Files likely in scope**
- `packages/findingmodel/src/findingmodel/index.py`
- `packages/findingmodel/src/findingmodel/protocols.py`

**Tests**
- `browse(...)` semantics
- filter-only browse replacing `search("", tags=...)`
- SQL pushdown behavior for search and batch search
- exact-match behavior under facet filters

**Acceptance Gate**
- Filter-only callers no longer depend on empty-query `search(...)`.
- Batch candidate generation can use the same facet filter surface when needed.

**Docs Gate**
- Update `packages/findingmodel/README.md` for the new retrieval API.

## Slice 5: Add Deterministic `related_models(...)`

**Scope**
- Add the core deterministic related-model API.
- Implement the provisional scoring weights from the design doc.
- Bootstrap a dedicated related-model evaluation fixture from the existing similar-model eval cases.

**Primary ownership**
- Lane D

**Files likely in scope**
- `packages/findingmodel/src/findingmodel/index.py`
- `packages/findingmodel-ai/evals/similar_models.py`
- new related-model fixture/eval assets as needed

**Tests**
- overlap/scoring tests
- threshold tests
- gold-case-set evaluation coverage

**Acceptance Gate**
- `related_models(...)` is usable independently of the LLM path.
- Scoring is covered by deterministic tests and eval assets.

**Docs Gate**
- Update retrieval docs if `related_models(...)` is public.

## Slice 6: Define New Enrichment Result and Review Types ✓

**Status:** Complete

**Scope**
- Define `EnrichModelResult`, `EnrichModelReview`, and any helper types needed for typed candidate gathering and review output.
- Keep the review artifact JSON-serializable and explicitly separate from canonical model JSON.
- All new types go in a new `types.py` module, importing canonical facet types from `findingmodel`.

**Explicitly out of scope**
- Do NOT modify `unified.py`, `agentic.py`, or their tests. The old enrichment modules are being replaced wholesale (Slice 7), not migrated. Retrofitting canonical types into dead-end code has no value.
- The duplicated facet literals in the old modules will disappear when those modules are removed in Slice 9.

**Primary ownership**
- Lane C

**Files changed**
- `packages/findingmodel-ai/src/findingmodel_ai/enrichment/types.py` (new) — 8 types: FieldConfidence, OntologyCandidateRelationship, OntologyCandidate, OntologyCandidateReport, AnatomicCandidate, EnrichModelReview, EnrichModelResult
- `packages/findingmodel-ai/src/findingmodel_ai/enrichment/__init__.py` — added exports for new types
- `packages/findingmodel-ai/tests/test_enrichment_types.py` (new) — 9 tests

**Design decisions**
- `AnatomicCandidate.location` uses `IndexCode` to avoid duplicating system/code/display fields
- `EnrichModelResult.model` is `FindingModelFull` (not a union) — enrichment operates on registered models
- `model_tier` uses `Literal["base", "small", "full"]` to keep types.py dependency-free from config
- No raw search results stored — Logfire instrumentation handles traceability

**Tests**
- ✓ Type validation (required fields, defaults, enum values)
- ✓ Serialization round-trip (JSON lossless)
- ✓ Separation invariant (review data excluded from model dump, model data excluded from review dump)

**Acceptance Gate**
- ✓ Type surface defined and stable for the pipeline rewrite
- ✓ Review-only data does not leak into canonical model JSON

**Docs Gate**
- ✓ No design doc update needed — types match existing design doc spec
- ✓ Implementation plan updated

## Slice 7: Implement `enrich_model(...)`

**Scope**
- Build the new canonical enrichment entrypoint around `FindingModel` updates.
- Keep candidate gathering deterministic and coded.
- Use typed structured inputs/outputs for the narrow classifier step.
- Apply canonical normalization during enrichment.

**Primary ownership**
- Lane C

**Files likely in scope**
- new module(s) under `packages/findingmodel-ai/src/findingmodel_ai/enrichment/`
- salvage helpers in `packages/findingmodel-ai/src/findingmodel_ai/search/`

**Tests**
- unit tests with `TestModel` and `FunctionModel`
- integration coverage for ontology/anatomic lookup composition
- review artifact generation tests

**Acceptance Gate**
- `enrich_model(...)` returns an updated canonical model plus separate review data.
- The old sidecar-style enrichment result is no longer the primary product internally.

**Docs Gate**
- Update `packages/findingmodel-ai/README.md` to introduce the new entrypoint once it is usable.

## Slice 8: Rewrite `find_similar_models()`

**Scope**
- Rework candidate gathering to use:
  - exact checks
  - unfiltered text search
  - optional facet-guided filtered search passes
  - `related_models(...)` when an existing canonical model ID is available
- Keep the planner step narrow and typed.
- Keep the final LLM decision limited to "edit existing vs create new".

**Primary ownership**
- Lane D

**Files likely in scope**
- `packages/findingmodel-ai/src/findingmodel_ai/search/similar.py`
- `packages/findingmodel-ai/evals/similar_models.py`

**Tests**
- exact match behavior
- new-finding scenario behavior
- recall protection when facet hypotheses are wrong
- evaluation coverage on the updated candidate pipeline

**Acceptance Gate**
- `find_similar_models()` no longer relies on text-only search-term generation alone.
- Facet hypotheses improve retrieval without becoming hard gates.

**Docs Gate**
- Update `packages/findingmodel-ai/README.md` examples for similar-model lookup.

## Slice 9: Public Surface Migration and Canonical CLI

**Scope**
- Add the canonical CLI for single-model enrichment and bulk backfill.
- Define review-artifact write location and overwrite/resume behavior.
- Update or remove ad hoc enrichment scripts and old exported entrypoints.
- Update CLI/MCP callers that moved from filter-only `search(...)` to `browse(...)`.

**Primary ownership**
- Lane C for enrichment CLI
- Lane B/D for retrieval caller migrations

**Files likely in scope**
- `packages/findingmodel-ai/src/findingmodel_ai/cli.py`
- `packages/findingmodel-ai/scripts/*`
- `packages/findingmodel/src/findingmodel/cli.py`
- `packages/findingmodel/src/findingmodel/mcp_server.py`
- `packages/findingmodel-ai/src/findingmodel_ai/enrichment/__init__.py`

**Tests**
- CLI smoke tests
- MCP-facing API tests where applicable
- migration tests for retired/renamed public entrypoints

**Acceptance Gate**
- One canonical enrichment CLI exists.
- Old enrichment codepaths are removed or intentionally retired in one coordinated pass.

**Docs Gate**
- Update package READMEs and top-level docs for the final public surface.

## Slice 10: Backfill, Fixtures, and Final Documentation Review

**Scope**
- Backfill the working corpus into `.fm.json` files using atomic writes.
- Rebuild DuckDB fixtures and published DB artifacts from the updated models.
- Curate canonical example files so they obey the new `IndexCode` semantics.
- Do a final docs pass to ensure the design doc, this plan, READMEs, and `CHANGELOG.md` match shipped behavior.

**Primary ownership**
- Shared, after all prior slices land

**Tests**
- full-corpus rebuild path
- atomic write behavior
- fixture rebuild verification

**Acceptance Gate**
- Backfilled models, rebuilt fixtures, and docs all reflect the same final workflow.

**Docs Gate**
- Mark this implementation plan complete.
- Update `CHANGELOG.md` with user-visible changes only.
- Move or archive superseded planning docs if needed.

## Suggested Review Boundaries

- PR 1: Slice 1
- PR 2: Slice 2
- PR 3: Slice 3
- PR 4: Slice 4
- PR 5: Slice 5
- PR 6: Slice 6 + Slice 7
- PR 7: Slice 8
- PR 8: Slice 9
- PR 9: Slice 10

Combine adjacent slices only if ownership is the same and the test surface remains easy to review.

## Ready-to-Assign Parallel Work

After Slice 1 lands:
- Worker A: Slice 2
- Worker B: Slice 3
- Worker C: Slice 6

After Slice 3 lands:
- Worker D: Slice 4
- Worker E: Slice 5

After Slices 4, 5, and 6 are stable:
- Worker F: Slice 7
- Worker G: Slice 8

## Final Completion Checklist

- [ ] Canonical facet types live in core `findingmodel`
- [ ] `FindingModelBase` and `FindingModelFull` carry the new structured metadata fields
- [ ] `IndexCode` semantics are enforced at the model/review boundary
- [ ] DuckDB build and hydration support the canonical fields
- [ ] `browse(...)`, facet-aware `search(...)`, and facet-aware `search_batch(...)` are implemented
- [ ] `related_models(...)` exists and is tested against a dedicated evaluation set
- [ ] `enrich_model(...)` is the canonical enrichment entrypoint
- [ ] `find_similar_models()` uses the new deterministic candidate pipeline
- [ ] canonical enrichment CLI/backfill flow exists
- [ ] docs and changelog match the final shipped behavior
