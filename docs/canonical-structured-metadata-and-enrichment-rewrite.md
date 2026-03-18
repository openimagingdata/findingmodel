# Canonical Structured Metadata and Enrichment Rewrite

## Status

- Proposed
- Supersedes the old enrichment implementation plan now archived at `docs/archive/finding-enrichment-implementation-plan.md`
- Supersedes the older enrichment PRD now archived at `docs/archive/finding-enrichment-prd.md`

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
- `EntityType = {finding, diagnosis, grouping, measurement, assessment, recommendation}`
- `SexSpecificity = {male-specific, female-specific, sex-neutral}`
- `AgeStage = {newborn, infant, preschool_child, child, adolescent, young_adult, adult, middle_aged, aged}`
- `AgeApplicability = "all_ages" | list[AgeStage]`
- `AgeProfile = {applicability: AgeApplicability, more_common_in: list[AgeStage] | None}`
- The age stages are disjoint, MeSH-derived bins chosen for faceting rather than raw overlapping MeSH descriptors
- `ExpectedTimeCourse = {duration: TimeCourseDuration | None, modifiers: list[TimeCourseModifier]}`
- `TimeCourseDuration = {rapid, short_term, intermediate, long_term, permanent}`
- `TimeCourseModifier = {resolving, evolving, progressive, stable, intermittent, waxing_waning, relapsing}`
- `EtiologyCode = {inflammatory, inflammatory:infectious, neoplastic:benign, neoplastic:malignant, neoplastic:metastatic, neoplastic:potential, traumatic:acute, traumatic:sequela, vascular:ischemic, vascular:hemorrhagic, vascular:thrombotic, vascular:aneurysmal, degenerative, metabolic, congenital, developmental, autoimmune, toxic, mechanical, iatrogenic:post-operative, iatrogenic:post-radiation, iatrogenic:device, iatrogenic:medication-related, idiopathic, normal-variant}`

Keep these existing fields as the canonical standardized-code layer:

- `anatomic_locations`
- `index_codes`

Broad anatomy belongs in `body_regions`. Precise anatomy belongs in ontology-backed `anatomic_locations`.

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

Both `search(...)` and `browse(...)` support filters for:

- `body_regions`
- `subspecialties`
- `etiologies`
- `entity_types`
- `applicable_modalities`
- `age_applicability`
- `age_more_common_in`
- `sex_specificity`
- `time_course_durations`
- `tags`

Filter semantics:

- OR within a single multi-value facet
- AND across different facets
- `tags` keeps current ALL-of semantics

DuckDB mirrors the canonical model fields into queryable columns:

- scalar columns: `entity_type VARCHAR`, `expected_time_course_duration VARCHAR`, `sex_specificity VARCHAR`, `age_applicability_scope VARCHAR`
- list columns: `body_regions VARCHAR[]`, `subspecialties VARCHAR[]`, `etiologies VARCHAR[]`, `applicable_modalities VARCHAR[]`, `age_applicability_stages VARCHAR[]`, `age_more_common_in VARCHAR[]`, `expected_time_course_modifiers VARCHAR[]`, `anatomic_location_ids VARCHAR[]`, `index_code_keys VARCHAR[]`

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

Rewrite `find_similar_models()` in `findingmodel-ai` to use `search(...)`, `browse(...)`, and `related_models()` for candidate gathering, leaving the LLM only to make the final "edit existing vs create new" judgment.

## Implementation Phases

### Phase 1: Write and align the docs

- Write this plan into `docs/canonical-structured-metadata-and-enrichment-rewrite.md`
- Archive the old plan under `docs/archive/`
- Mark the older facets/enrichment planning docs as superseded where relevant
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
  - overlapping or free-text age bins map to the nearest canonical `AgeStage` values
  - "more common in" and "can occur in" signals are stored separately

### Phase 3: Build the new enrichment tool

- Implement `enrich_model(...)` from scratch in `findingmodel-ai`
- Reuse only the salvageable search/prompt assets from the older work
- Keep evidence gathering deterministic and coded
- Keep the classifier narrow, typed, and model-oriented
- Write review artifacts separately from canonical model JSON

### Phase 4: Rebuild indexing and retrieval around the canonical fields

- Extend the maintainer DuckDB build with the new columns
- Extend read-side hydration and `IndexEntry`
- Add `browse(...)`
- Extend `search(...)` with facet-aware candidate generation and ranking inputs
- Extend search text and embedding text with human-readable structured labels, not just raw ontology IDs

### Phase 5: Add related-model retrieval and authoring integration

- Add `related_models(...)`
- Rework `find_similar_models()` onto the new deterministic candidate pipeline
- Add one canonical CLI for single-model enrichment and bulk backfill
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
- Test related-model scoring against a gold case set before release
- Test backfill atomic writes and a full-corpus rebuild path

## Defaults and Assumptions

- No compatibility shims
- No schema version field in v1
- Filter-only browse is first-class, but it uses `browse(...)`, not `search(query=None)`
- Broad anatomy lives in `body_regions`
- Precise anatomy lives in `anatomic_locations`
- Review artifacts are kept outside canonical `.fm.json` model files
