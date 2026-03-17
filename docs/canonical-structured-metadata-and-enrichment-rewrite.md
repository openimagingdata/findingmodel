# Canonical Structured Metadata and Enrichment Rewrite

## Status

- Proposed
- Supersedes the old enrichment implementation plan now archived at `docs/archive/finding-enrichment-implementation-plan.md`
- Supersedes the older enrichment PRD now archived at `docs/archive/finding-enrichment-prd.md`
- Implementation tracking lives in `tasks/canonical-structured-metadata-implementation-plan.md`

## Summary

- The repo has two older enrichment pathways.
- The original program-orchestrated pathway came first and better matches the repo's current AI workflow guidance.
- The later agentic pathway was an alternate experiment.
- Neither older pathway is acceptable as the canonical future implementation because both produce sidecar enrichment output instead of making structured metadata part of `FindingModel`.
- The new direction is to build one new enrichment tool from scratch around the model we actually want, salvage only the useful search/prompt pieces from the old work, and remove both older enrichment pathways with no compatibility shims.

## Goals

- Make structured metadata canonical `FindingModel` state rather than disposable enrichment output.
- Support fast, structured browse and related-model retrieval on top of that canonical metadata.
- Keep broad anatomy, precise anatomy, ontology codes, and retrieval behavior coherent across the model layer, enrichment pipeline, and DuckDB index.
- Keep review/provenance artifacts separate from the canonical model JSON.

## Non-Goals

- Keeping either old enrichment path as a supported entrypoint.
- Storing raw candidate buckets or LLM reasoning directly on `FindingModel`.
- Overloading `search()` to do both query-driven search and filter-only browse.

## Canonical Model Shape

Shared facet types move into core `findingmodel` so both `findingmodel` and `findingmodel-ai` use one schema.

Add these optional fields to `FindingModelBase` and `FindingModelFull`:

- `body_regions: list[BodyRegion] | None`
- `subspecialties: list[Subspecialty] | None`
- `etiologies: list[EtiologyCode] | None`
- `entity_type: EntityType | None`
- `applicable_modalities: list[Modality] | None`
- `expected_time_course: ExpectedTimeCourse | None`
- `age_profile: AgeProfile | None`
- `sex_specificity: SexSpecificity | None`

Canonical facet values:

- `BodyRegion = {head, neck, chest, breast, abdomen, pelvis, spine, upper_extremity, lower_extremity, whole_body}`
- `Subspecialty = {AB, BR, CA, CH, ER, GI, GU, HN, IR, MI, MK, NR, OB, OI, PD, VI}`
- `Modality = {XR, CT, MR, US, PET, NM, MG, RF, DSA}`, with `CR` and `DX` normalized to `XR`
- `EntityType = {finding, diagnosis, grouping, measurement, assessment, recommendation, technique_issue}`
- `SexSpecificity = {male-specific, female-specific, sex-neutral}`
- `AgeStage = {newborn, infant, preschool_child, child, adolescent, young_adult, adult, middle_aged, aged}`
- `AgeApplicability = "all_ages" | list[AgeStage]`
- `AgeProfile = {applicability: AgeApplicability, more_common_in: list[AgeStage] | None}`
- The age stages are disjoint, MeSH-derived bins chosen for faceting rather than raw overlapping MeSH descriptors
- `ExpectedTimeCourse = {duration: ExpectedDuration | None, modifiers: list[TimeCourseModifier]}`
- `ExpectedDuration = {hours, days, weeks, months, years, permanent}` — upper bound on how long the finding typically remains visible on imaging
- `TimeCourseModifier = {progressive, stable, evolving, resolving, intermittent, fluctuating, recurrent}`
- `EtiologyCode = {inflammatory, inflammatory:infectious, neoplastic:benign, neoplastic:malignant, neoplastic:metastatic, neoplastic:potential, traumatic:acute, traumatic:sequela, vascular:ischemic, vascular:hemorrhagic, vascular:thrombotic, vascular:aneurysmal, degenerative, metabolic, congenital, developmental, autoimmune, toxic, mechanical, iatrogenic:post-operative, iatrogenic:post-radiation, iatrogenic:device, iatrogenic:medication-related, idiopathic, normal-variant}`

Keep these existing fields as the canonical standardized-code layer:

- `anatomic_locations`
- `index_codes`

Broad anatomy belongs in `body_regions`. Precise anatomy belongs in ontology-backed `anatomic_locations`.

`IndexCode` semantics:

- `IndexCode` remains a simple ontology identifier object: `system`, `code`, and optional `display`
- relationship semantics do not live on `IndexCode` itself
- on canonical `FindingModel.index_codes`, each code must be an exact match or a clinically substitutable near-equivalent for the full model concept
- do not store merely related, broader, narrower, complication-specific, severity-specific, or temporally qualified codes in canonical `index_codes` unless the model itself is defined at that narrower level
- non-exact ontology candidates belong in the separate enrichment review artifact, where they can be categorized explicitly
- if the project later needs canonical storage of non-exact code relationships, add a separate typed wrapper rather than overloading `IndexCode`

Normalization rule for legacy body-region values:

- existing title-cased legacy values normalize to the lowercase canonical equivalent
- `ALL` remains a special legacy alias that normalizes to `whole_body`

Do not store ontology candidate buckets, raw search results, prompt traces, or model reasoning on `FindingModel`.

## Canonical Enrichment Tool

Add one new AI entrypoint in `findingmodel-ai`:

```python
async def enrich_model(
    model: FindingModelBase | FindingModelFull,
    *,
    tier: ModelTier = "base",
) -> EnrichModelResult
```

`EnrichModelResult` contains:

- `model`: the updated model with canonical structured metadata filled in
- `review`: a separate QA artifact with raw candidates, normalization notes, timings, and review-oriented reasoning

`review` contract in v1:

- Define a typed `EnrichModelReview` Pydantic model in `findingmodel-ai`
- Make it fully JSON-serializable so CLI/backfill workflows can persist it without ad hoc shaping
- Return it from `enrich_model(...)` directly; file writing stays outside the core function
- Canonical CLI/backfill writes review JSON to a separate artifact location, not into `.fm.json`
- Default bulk-backfill layout should be deterministic by model identity, e.g. `<artifact_root>/<oifm_id>.enrich-review.json`
- The plan must update all current scripts that assume a sidecar enrichment blob or custom output JSON shape

Design rules:

- Build this tool fresh around the canonical model update path.
- Reuse only the useful parts of the old work: ontology search helpers, anatomic search helpers, and prompt/example material.
- Keep orchestration programmatic.
- Query generation, candidate gathering, normalization, and deterministic fallback behavior live in code.
- The model-facing classifier step receives typed structured input and returns typed canonical output.
- The tool updates a `FindingModel`; it does not write a sidecar enrichment blob as the primary product.

## Search, Browse, and Relatedness

Keep `search(query: str, ...)` query-driven.

Add a separate `browse(...)` API for filter-only browsing.

`browse(...)` should return `tuple[list[IndexEntry], int]`, matching the current pagination pattern of `all(...)`.

`browse(...)` is the replacement for all current empty-query/tag-only lookup patterns. As part of the rewrite, update:

- `FindingModelIndex`
- exported `Index` protocol surface
- CLI and MCP server call sites
- tests that currently exercise filter-only search via `search("", tags=...)`
- AI search consumers that need filtered candidate gathering

Both `search(...)` and `browse(...)` support filters for:

- `body_regions`
- `subspecialties`
- `etiologies`
- `entity_type`
- `applicable_modalities`
- `age_applicability`
- `age_more_common_in`
- `sex_specificity`
- `time_course_durations`
- `time_course_modifiers`
- `tags`

Also extend `search_batch(...)` with the same facet filter surface where batch candidate generation still needs it; do not leave `find_similar_models()` or anatomic/ontology helpers stranded on a narrower search API.

Filter semantics:

- OR within a single multi-value facet
- AND across different facets
- `tags` keeps current ALL-of semantics

DuckDB mirrors the canonical model fields into queryable columns:

- scalar columns: `entity_type VARCHAR`, `expected_time_course_duration VARCHAR`, `sex_specificity VARCHAR`, `age_applicability_scope VARCHAR`
- list columns: `body_regions VARCHAR[]`, `subspecialties VARCHAR[]`, `etiologies VARCHAR[]`, `applicable_modalities VARCHAR[]`, `age_applicability_stages VARCHAR[]`, `age_more_common_in VARCHAR[]`, `expected_time_course_modifiers VARCHAR[]`, `anatomic_location_ids VARCHAR[]`, `index_code_keys VARCHAR[]`

`index_code_keys` uses the normalized compound form `"{system}:{code}"` for deterministic build/query behavior.

`search(...)` pushes facet filters into FTS/vector candidate generation before ranking.

Add `related_models(model_id, *, limit=10)` as a deterministic core API.

Initial candidate prefilter:

- exclude self
- if the source model has anatomy or codes, prefilter to models sharing at least one of `body_regions`, `anatomic_location_ids`, or `index_code_keys`

Initial scoring weights:

- anatomic location overlap: `5`
- index-code overlap: `4`
- same entity type: `3`
- body-region overlap: `2`
- etiology overlap: `2`
- subspecialty overlap: `2`
- modality overlap: `1`
- age overlap: `1`
- sex match: `1`
- time-course overlap: `1`

Release rules:

- minimum score `3`
- default return count `10`
- weights are provisional and must be tuned against a gold case set before release
- bootstrap that gold case set from the existing `packages/findingmodel-ai/evals/similar_models.py` cases, then promote the retained subset into a dedicated related-model fixture/eval asset owned by this rewrite

Rewrite `find_similar_models()` in `findingmodel-ai` to use `search(...)`, `browse(...)`, and `related_models()` for candidate gathering, leaving the LLM only to make the final "edit existing vs create new" judgment.

Design note for `find_similar_models()`:

- when the caller has an existing canonical model ID, use `related_models(model_id, ...)` as one candidate source
- when the caller only has a proposed new finding name/description/synonyms, keep a narrow typed planning step that emits:
  - alternate search terms
  - optional high-confidence facet hypotheses (for example `body_regions`, `applicable_modalities`, `entity_type`)
- use those facet hypotheses to add filtered candidate-gathering passes via `search(...)` and/or `browse(...)`, but do not rely on them as the only gate
- always preserve an unfiltered text-search path so incorrect facet guesses do not collapse recall
- deterministic code merges and scores the union of exact-match checks, text-search candidates, facet-filtered candidates, and `related_models(...)` candidates before the final LLM judgment

## Implementation Phases

### Phase 1: Write and align the docs

- Write this plan into `docs/canonical-structured-metadata-and-enrichment-rewrite.md`
- Archive the old plan under `docs/archive/`
- Mark the older facets/enrichment planning docs as superseded where relevant
- Record the implementation inventory for all affected public surfaces before coding:
  - `findingmodel` Index/CLI/MCP/README
  - `findingmodel-ai` enrichment exports, search exports, scripts, evals, and README
  - test files that currently assume filter-only `search(...)` or old enrichment result types
- Update package README docs and `CHANGELOG.md` as later phases land

### Phase 2: Add the canonical schema

- Add the shared facet types to core `findingmodel`
- Extend `FindingModelBase` and `FindingModelFull` with the new optional fields
- Update JSON serialization, round-trip loading, markdown export/import, editor flows, and stub generation
- Normalize legacy values during load and enrichment:
  - `ALL -> whole_body`
  - `Arm -> upper_extremity`
  - `Leg -> lower_extremity`
  - `traumatic -> traumatic:acute`
  - `post-traumatic -> traumatic:sequela`
  - `iatrogenic:device-related -> iatrogenic:device`
  - `CR -> XR`
  - `DX -> XR`
- Normalize legacy age labels into the new structure:
  - `any age -> {"applicability": "all_ages"}`
  - overlapping or free-text age bins map through one explicit normalization table checked into the repo and covered by tests
  - "more common in" and "can occur in" signals are stored separately

### Phase 3: Build the new enrichment tool

- Implement `enrich_model(...)` from scratch in `findingmodel-ai`
- Reuse only the salvageable search/prompt assets from the older work
- Keep evidence gathering deterministic and coded
- Keep the classifier narrow, typed, and model-oriented
- Write review artifacts separately from canonical model JSON
- Replace the current exported enrichment entrypoints in one coordinated pass:
  - `findingmodel_ai.enrichment.enrich_finding`
  - `findingmodel_ai.enrichment.enrich_finding_unified`
  - `findingmodel_ai.enrichment.enrich_finding_agentic`
  - scripts and tests that currently depend on `FindingEnrichmentResult`

### Phase 4: Rebuild indexing and retrieval around the canonical fields

- Extend the maintainer DuckDB build with the new columns
- Extend read-side hydration and `IndexEntry`
- Add `browse(...)`
- Extend `search(...)` with facet-aware candidate generation and ranking inputs
- Keep canonical facets in structured columns for filtering, browsing, and retrieval logic; do not duplicate them into FTS search text or embedding text by default

### Phase 5: Add related-model retrieval and authoring integration

- Add `related_models(...)`
- Rework `find_similar_models()` onto the new deterministic candidate pipeline
- Add one canonical CLI for single-model enrichment and bulk backfill
- Define the review-artifact write location and overwrite/resume behavior for that CLI before implementation starts
- Remove the old ad hoc enrichment scripts and codepaths

### Phase 6: Backfill, validate, and clean up

- Backfill the current corpus directly into `.fm.json` files with the new tool
- Use file-by-file atomic writes
- Rebuild DuckDB fixtures and published DB artifacts from the updated models
- Run a final documentation review so the plan, READMEs, and `CHANGELOG.md` match the final shipped workflow

## Test Plan

- Test the new enums, typed models, and normalization rules directly
- Test that older model JSON still loads and reserializes under the new code
- Test markdown and editor round-trips for the new fields
- Test enrichment mapping and QA artifact generation with `FunctionModel` and `TestModel`, plus callout integration coverage
- Test exact DuckDB DDL and hydration for the new columns
- Test `browse(...)` and facet-aware `search(...)` semantics, including SQL prefiltering behavior
- Replace current empty-query tag-filter search tests with explicit `browse(...)` coverage rather than silently preserving the old overload
- Test CLI/MCP/public API migration points so there is no orphaned caller still expecting the retired enrichment or filter-only search behavior
- Test related-model scoring against a gold case set before release
- Test backfill atomic writes and a full-corpus rebuild path

## Defaults and Assumptions

- No compatibility shims
- No schema version field in v1
- Filter-only browse is first-class, but it uses `browse(...)`, not `search(query=None)`
- Broad anatomy lives in `body_regions`
- Precise anatomy lives in `anatomic_locations`
- Review artifacts are kept outside canonical `.fm.json` model files
