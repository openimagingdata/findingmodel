# Canonical Structured Metadata Implementation Plan

**Status:** In progress — Slices 1–8 and 9-A complete. Slice 9-B and Slice 10 remain.

**Design Spec:** [../docs/canonical-structured-metadata-and-enrichment-rewrite.md](../docs/canonical-structured-metadata-and-enrichment-rewrite.md)

**This plan is the execution document.** The design doc above remains the source of truth for architecture, canonical values, and semantic rules.

**Supersedes for this workstream:** [facets-implementation-plan.md](./facets-implementation-plan.md)

## Remaining Work Summary

At this point the core structured-metadata architecture should be treated as implemented and usable via its intended public surfaces:

- `findingmodel_ai.metadata.assign_metadata(...)`
- `MetadataAssignmentResult` / `MetadataAssignmentReview`
- `FindingModelIndex.browse(...)`
- `FindingModelIndex.search(...)` / `search_batch(...)`
- `FindingModelIndex.related_models(...)`

The remaining work is now primarily in two buckets:

1. Tooling migration
   - update `findingmodel` MCP tools to expose facet-aware browsing
   - update `findingmodel` CLI to expose `browse()` / `related_models()`
   - add bulk metadata-assignment/backfill CLI flow with resume/overwrite behavior

2. Content backfill
   - run metadata assignment across the corpus
   - review and persist canonical model updates / review sidecars
   - rebuild fixtures and published DB artifacts from backfilled content
   - do final documentation cleanup once the corpus and artifacts reflect the final state

Forward work should generally treat the metadata core as a black box unless backfill reveals a concrete defect that requires reopening core behavior.

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
- shared metadata type module(s) (`facets.py`) in `packages/findingmodel/src/findingmodel/`
- markdown/template/editor/stub round-trip code

### Lane B: Index and Build Surface
- `packages/findingmodel/src/findingmodel/index.py`
- `packages/findingmodel/src/findingmodel/protocols.py`
- `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`
- relevant DuckDB tests and fixtures

### Lane C: Metadata Assignment Rewrite
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/`
- `packages/findingmodel-ai/src/findingmodel_ai/search/ontology.py`
- `packages/findingmodel-ai/src/findingmodel_ai/search/anatomic.py`
- metadata-assignment tests/evals/scripts

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
- Add shared structured metadata types to core `findingmodel`.
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
- `packages/findingmodel/src/findingmodel/index.py` — extended `IndexEntry` and hydration for structured metadata fields, `ExpectedTimeCourse`, `AgeProfile`, `anatomic_location_ids`, and `index_code_keys`
- `packages/oidm-maintenance/tests/test_findingmodel_build.py` — schema, populated/null hydration, and “not in search/embedding text” coverage
- `packages/findingmodel/tests/data/test_index.duckdb` — regenerated committed fixture with the updated schema
- `docs/canonical-structured-metadata-and-enrichment-rewrite.md` — updated to keep facets structured rather than duplicated into retrieval text

**Files likely in scope**
- `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`
- `packages/findingmodel/src/findingmodel/index.py`

**Tests**
- exact schema/DDL assertions
- build round-trip tests
- hydration tests for populated and null metadata columns
- `packages/oidm-maintenance/tests/test_findingmodel_build.py`
- `packages/findingmodel/tests/test_findingmodel_index.py`

**Acceptance Gate**
- ✓ Database build and read-side hydration fully support the canonical fields.
- ✓ `index_code_keys` are normalized consistently as `system:code`.
- ✓ FTS/search text and embedding text remain focused on free-text clinical content rather than duplicated metadata values.

**Docs Gate**
- ✓ Design doc updated to reflect the “structured columns only” retrieval decision.

## Slice 4: `browse()` + Metadata-Aware Search APIs ✓

**Status:** Complete

**No major scope change** — this is the foundation for Slices 5 and 8.

**Primary ownership**
- Lane B

**Add to `packages/findingmodel/src/findingmodel/index.py`:**

1. `browse(*, body_regions, subspecialties, etiologies, entity_type, applicable_modalities, sex_specificity, tags, limit, offset) -> tuple[list[IndexEntry], int]`
   - Pure SQL filter, no FTS/embeddings
   - OR-within-field, AND-across-fields, ALL-of for tags
   - Pagination via limit/offset with total count

2. Extend `search()` signature with same metadata filter params
   - Filters push into SQL WHERE (pre-ranking, not post-ranking)
   - Uses DuckDB `list_has_any()` for array columns

3. Extend `search_batch()` signature with same metadata filter params
   - Critical for Slice 8: enables metadata-guided candidate gathering alongside unfiltered search

4. Internal `_build_facet_where_clause()` shared by all three methods

**Also update `packages/findingmodel/src/findingmodel/protocols.py`** — update protocol definitions to match.

**Tests**
- browse with single field, multiple fields, empty result
- search with facets narrows results
- search_batch with facets
- SQL pushdown verification (filter before ranking, not after)
- tags retain ALL-of while facets use OR-within

**Acceptance Gate**
- Filter-only callers no longer depend on empty-query `search(...)`.
- Batch candidate generation can use the same metadata filter surface when needed.

**Docs Gate**
- Update `packages/findingmodel/README.md` for the new retrieval API.

## Slice 5: Deterministic `related_models()` ✓

**Status:** Complete

**A first-class standalone API** — useful for browsing related models in MCP tools, CLI exploration, and as a candidate source for Slice 8's pipeline. No LLM calls; purely deterministic metadata-overlap scoring.

**Primary ownership**
- Lane D

**Add to `packages/findingmodel/src/findingmodel/index.py`:**

```python
async def related_models(
    self,
    model_id: str,
    *,
    limit: int = 10,
    min_score: float = 3.0,
    weights: RelatedModelWeights | None = None,
) -> list[tuple[IndexEntry, float]]:
```

**Configurable weights** via `RelatedModelWeights(BaseModel)`:
- `anatomic_location_ids: float = 5.0`
- `index_code_keys: float = 4.0`
- `entity_type: float = 3.0`
- `body_regions: float = 2.0`
- `etiologies: float = 2.0`
- `subspecialties: float = 2.0`
- `applicable_modalities: float = 1.0`
- `age_overlap: float = 1.0`
- `sex_match: float = 1.0`
- `time_course: float = 1.0`

Uses `browse()` internally to prefilter candidates sharing at least one metadata field with the source model, then scores the prefiltered set.

**Tests**
- Score calculation with known inputs
- Threshold filtering
- Self-exclusion (source model not in results)
- Empty facets produce zero for that dimension (not NaN)

**Acceptance Gate**
- `related_models(...)` is usable independently of the LLM path.
- Scoring is covered by deterministic tests.

**Docs Gate**
- Update retrieval docs if `related_models(...)` is public.

## Slice 6: Define New Metadata-Assignment Result and Review Types ✓

**Status:** Complete

**Scope**
- Define `MetadataAssignmentResult`, `MetadataAssignmentReview`, and any helper types needed for typed candidate gathering and review output.
- Keep the review artifact JSON-serializable and explicitly separate from canonical model JSON.
- All new types go in a new `types.py` module, importing structured metadata types from `findingmodel`.

**Explicitly out of scope**
- Do NOT modify `unified.py`, `agentic.py`, or their tests. The old enrichment modules are being replaced wholesale (Slice 7), not migrated. Retrofitting canonical types into dead-end code has no value.
- The duplicated metadata literals in the old modules will disappear when those modules are removed in Slice 9.

**Primary ownership**
- Lane C

**Files changed**
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/types.py` — typed metadata-assignment result and review models
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/__init__.py` — public exports for metadata-assignment types
- `packages/findingmodel-ai/tests/test_metadata_types.py` — coverage for metadata-assignment types

**Design decisions**
- `AnatomicCandidate.location` uses `IndexCode` to avoid duplicating system/code/display fields
- `MetadataAssignmentResult.model` is `FindingModelFull` (not a union) — metadata assignment operates on registered models
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

## Slice 7: Implement `assign_metadata(...)` ✓

**Status:** Complete

**Scope**
- Build the new canonical metadata-assignment entrypoint around `FindingModel` updates.
- Keep candidate gathering deterministic and coded.
- Use typed structured inputs/outputs for the narrow classifier step.
- Apply canonical normalization during metadata assignment.
- Add careful, tiered Logfire observability for the metadata-assignment run and major substeps.

**Primary ownership**
- Lane C

**Files changed**
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py` — canonical metadata-assignment entrypoint, deterministic phase helpers, narrow classifier agent, and Logfire tracing
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/types.py` — `logfire_trace_id` and ontology rejection taxonomy support on review candidates
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/__init__.py` — exported `assign_metadata(...)` and the rejection taxonomy
- `packages/findingmodel-ai/tests/test_assign_metadata.py` — canonical assembly, prompt-shape, and fast-path tests
- `packages/findingmodel-ai/tests/test_metadata_types.py` — coverage for trace correlation and ontology rejection reason enums
- `packages/findingmodel-ai/README.md` — documented the new canonical entrypoint

**Files likely in scope**
- new module(s) under `packages/findingmodel-ai/src/findingmodel_ai/metadata/`
- salvage helpers in `packages/findingmodel-ai/src/findingmodel_ai/search/`

**Implementation notes**
- Add a new canonical entrypoint, `async assign_metadata(finding_model: FindingModelFull, *, model_tier: ModelTier = "small") -> MetadataAssignmentResult`. Model selection goes through agent tags and `AGENT_MODEL_OVERRIDES`; no per-call model override parameter.
- Keep file I/O, review-artifact persistence, and bulk backfill out of this slice. Slice 7 returns typed in-memory results only.
- Preserve the incoming model identity and authored text (`oifm_id`, `slug_name`, `name`, `description`, `synonyms`, `attributes`, `tags`) unless deterministic code or the narrow classifier explicitly updates a canonical metadata field.
- Add field-aware fast paths:
  - skip ontology gathering when canonical `index_codes` are already present
  - skip anatomic gathering when canonical `anatomic_locations` are already present
  - skip the classifier entirely when all canonical metadata/code/location targets are already populated
- Candidate gathering remains coded:
  - ontology candidates from `search/ontology.py`
  - anatomic candidates from `search/anatomic.py`
  - no raw search result blobs stored on the canonical model
- Use one narrow typed classifier output model for only the ambiguous canonical decisions:
  - structured metadata values
  - ontology/anatomic candidate selection decisions where deterministic ranking alone is insufficient
  - field-level rationale/confidence for the review artifact
- Ontology selection should follow the explicit generality rule borrowed from the coding pipeline:
  - a slightly broader match can be acceptable when it preserves grouping of equivalent findings
  - a narrower match is not acceptable
- Review-only ontology candidates should carry a typed rejection taxonomy where applicable:
  - `too_specific`
  - `too_broad`
  - `wrong_concept`
  - `definition_mismatch`
  - `overlapping_scope`
- Instrument the entrypoint with tiered Logfire spans:
  - one top-level metadata-assignment run span
  - child spans for ontology gathering, anatomic gathering, classifier decision, and final model assembly
  - structured summary events/warnings at major boundaries, but no noisy per-candidate trace spam by default
- Capture the top-level Logfire `trace_id` when available and store it on `MetadataAssignmentReview.logfire_trace_id` so review artifacts can be correlated back to the full trace.
- Normalize all selected values through the canonical `findingmodel` types before assembling the final `FindingModelFull`.
- Default the narrow structured-output LLM steps to `small` tier unless an override is configured; allow bumping to `base`/`full` via `AGENT_MODEL_OVERRIDES` or the entrypoint argument.
- Export the new entrypoint from `findingmodel_ai.metadata`.
- Pull Slice 9 forward for this surface: remove the old `findingmodel_ai.enrichment` package rather than preserving it alongside the new metadata-assignment API.

**Tests**
- unit tests with `TestModel` and `FunctionModel`
- integration coverage for ontology/anatomic lookup composition
- review artifact generation tests

**Acceptance Gate**
- ✓ `assign_metadata(...)` returns an updated canonical model plus separate review data.
- ✓ The old sidecar-style result is no longer the primary product internally.
- ✓ The metadata-assignment run emits tiered Logfire traces and includes a trace handle on the review object when available.
- ✓ Field-aware fast paths avoid unnecessary LLM work when canonical data is already present.
- ✓ Review data clearly distinguishes canonical ontology selections from rejected candidates and records typed rejection reasons where applicable.

**Docs Gate**
- ✓ `packages/findingmodel-ai/README.md` updated to introduce the new entrypoint.

## Slice 8: Rewrite `find_similar_models()` — 5-Phase Pipeline ✓

**Status:** Complete

Adopts the 5-phase pipeline architecture from the coding project.

**Primary ownership**
- Lane D

**New shared infrastructure** (`packages/findingmodel-ai/src/findingmodel_ai/search/pipeline_helpers.py`):
- `ModelMatchRejectionReason` enum: `too_specific`, `too_broad`, `wrong_concept`, `definition_mismatch`, `overlapping_scope`
- `CandidatePool` — deduplicated candidate container with provenance tracking and max cap
- `validate_selection_in_candidates()` — post-selection check for hallucinated IDs
- `MetadataHypothesis` model — typed container for LLM-generated metadata guesses
- Result types: `SimilarModelMatch`, `SimilarModelRejection`, `SimilarModelResult`

**5-Phase Pipeline** (`packages/findingmodel-ai/src/findingmodel_ai/search/similar.py` rewrite):

1. **Phase 1: Fast-path** — exact match by name/synonyms via `index.get()`, return immediately if found
2. **Phase 2: LLM Planning** — single small-model call generating `SimilarModelPlan` (search terms + metadata hypotheses)
3. **Phase 3: Multi-Pass Search** — unfiltered text search (recall protection) + metadata-filtered search + optional `related_models()`
4. **Phase 4: LLM Selection** — structured output `SimilarModelSelection` with rejection taxonomy and asymmetric generality rule
5. **Phase 5: Assembly** — build `SimilarModelResult` from validated selections

**Agent tags**: `similar_plan` (replaces `similar_search`), `similar_select` (replaces `similar_assess`)

**What gets removed**: `create_search_agent()`, `create_term_generation_agent()`, `create_analysis_agent()`, `search_models_tool()`, `SearchStrategy`, `SimilarModelAnalysis`, `_generate_search_terms_with_fallback()`

**Tests**
- Fast-path: exact name/synonym match returns immediately
- Phase 2: plan generation with FunctionModel
- Phase 3: multi-pass search merges correctly, deduplicates, caps at MAX_CANDIDATES
- Phase 3: recall protection — wrong metadata hypothesis doesn't collapse results
- Phase 4: selection with each rejection reason
- Phase 4: post-validation catches hallucinated IDs
- Integration: full pipeline with `@pytest.mark.callout`

**Acceptance Gate**
- `find_similar_models()` no longer relies on text-only search-term generation alone.
- Metadata hypotheses improve retrieval without becoming hard gates.
- Post-selection validation catches hallucinated IDs.

**Docs Gate**
- Update `packages/findingmodel-ai/README.md` examples for similar-model lookup.

## Slice 9-A: Canonical CLI and Legacy Removal ✓

**Status:** Complete

**Scope**
- Add canonical CLI for single-model metadata assignment.
- Remove old enrichment package, entrypoints, and scripts.
- Rename `enrichment/` → `metadata/`, `enrich_model()` → `assign_metadata()`.
- Add shared runtime Logfire initializer for executable paths.

**Primary ownership**
- Lane C

**Files changed**
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/` — new package (renamed from `enrichment/`)
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py` — renamed from `canonical.py`
- `packages/findingmodel-ai/src/findingmodel_ai/metadata/types.py` — renamed types (`MetadataAssignmentResult`, etc.)
- `packages/findingmodel-ai/src/findingmodel_ai/observability.py` — shared Logfire + HTTPX instrumentation
- `packages/findingmodel-ai/src/findingmodel_ai/cli.py` — `assign-metadata` command with `--logfire` and review output
- `packages/findingmodel-ai/src/findingmodel_ai/__init__.py` — exports `metadata` and `observability` instead of `enrichment`
- `packages/findingmodel-ai/src/findingmodel_ai/config.py` — agent tag `metadata_assign` replaces `enrich_classify`
- Deleted: `enrichment/unified.py`, `enrichment/agentic.py`, `enrichment/__init__.py`, `enrichment/types.py`, old batch scripts, `test_finding_enrichment.py`, `test_enrichment_types.py`

**Acceptance Gate**
- ✓ One canonical `findingmodel-ai assign-metadata` CLI exists with `--logfire` and optional review JSON output
- ✓ Old enrichment codepaths removed in one coordinated pass
- ✓ Executable paths can opt into Logfire with agent + outbound HTTP traces via `ensure_logfire_configured()`

**Docs Gate**
- ✓ README, CHANGELOG, configuration docs updated for the rename and new CLI

## Slice 9-B: MCP/CLI Caller Migration and Bulk Backfill

**Status:** Not started

**Scope**
- Update MCP server tools to use `browse()` for metadata-filtered listing.
- Update `findingmodel` CLI to expose `browse()` and `related_models()`.
- Add bulk backfill CLI (`assign-metadata --batch` or similar) with resume/overwrite behavior.
- Define review-artifact write location convention for bulk runs.

**Primary ownership**
- Lane B/D for retrieval caller migrations
- Lane C for bulk backfill

**Files likely in scope**
- `packages/findingmodel/src/findingmodel/mcp_server.py`
- `packages/findingmodel/src/findingmodel/cli.py`
- `packages/findingmodel-ai/src/findingmodel_ai/cli.py`

**Tests**
- MCP tool tests for browse/metadata-filtered listing
- CLI smoke tests for bulk backfill
- Resume/overwrite behavior tests

**Acceptance Gate**
- MCP server exposes metadata-based browsing.
- Bulk backfill CLI exists with atomic writes and resume support.
- Review artifacts have a defined write location convention.

**Docs Gate**
- Update MCP server documentation for new tools.
- Update CLI documentation for bulk backfill.

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

- [ ] Structured metadata types live in core `findingmodel`
- [ ] `FindingModelBase` and `FindingModelFull` carry the new structured metadata fields
- [ ] `IndexCode` semantics are enforced at the model/review boundary
- [ ] DuckDB build and hydration support the canonical fields
- [ ] `browse(...)`, metadata-aware `search(...)`, and metadata-aware `search_batch(...)` are implemented
- [ ] `related_models(...)` exists and is tested against a dedicated evaluation set
- [ ] `assign_metadata(...)` is the canonical metadata-assignment entrypoint
- [ ] `find_similar_models()` uses the new deterministic candidate pipeline
- [ ] canonical enrichment CLI/backfill flow exists
- [ ] docs and changelog match the final shipped behavior
