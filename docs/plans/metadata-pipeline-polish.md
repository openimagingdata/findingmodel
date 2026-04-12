# Plan: Finish the `findingmodel` Type Reorganization and Complete Metadata Reassessment

Status: Complete (2026-03-24)

Source draft: `~/.claude/plans/eager-twirling-scroll.md`

## Summary

This work has four linked outcomes:

1. complete the internal `findingmodel` type split
2. migrate the repo to top-level `findingmodel` imports
3. make `assign_metadata()` reassess existing metadata by default
4. add deterministic eval, benchmark, and documentation coverage

Supported Python API after this work is top-level `from findingmodel import ...` only. `findingmodel.types` is internal layout. `findingmodel.facets` and `findingmodel.finding_model` are removed once the repo migration is complete.

As phases land, update this plan, the active metadata rewrite doc, package README and CLI docs, and `CHANGELOG.md`. At the end, do a final documentation review so the plan status, reference docs, and user-facing change notes match the shipped state.

## Verification Update (2026-03-24)

- Focused verification passed for the reorganized `findingmodel` surface and metadata-assignment path:
  - `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata_modes.py packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_cli_assign_metadata.py`
  - `uv run pytest packages/findingmodel/tests/test_type_restructure.py packages/findingmodel/tests/test_metadata.py packages/findingmodel/tests/test_models.py`
  - `uv run pytest packages/findingmodel/tests/test_create_stub.py packages/findingmodel/tests/test_findingmodel_index.py packages/findingmodel/tests/test_index_validation.py packages/findingmodel/tests/test_related_models.py packages/findingmodel/tests/test_browse_and_metadata_search.py`
  - `uv run pytest packages/findingmodel-ai/tests/test_metadata_types.py packages/findingmodel-ai/tests/test_tools.py packages/findingmodel-ai/tests/test_model_editor.py`
  - `uv run pytest packages/oidm-maintenance/tests/test_findingmodel_build.py`
- `task evals:metadata_assignment` now runs successfully with all `HasMatchingSpan` assertions passing in every case and a rerun overall score of `0.99`.
- Follow-up tuning added explicit reassess vs `fill_blanks_only` mode guidance to the classifier prompt, clarified `entity_type` and `body_regions` decision rules, and made the top-level `findingmodel.__all__` export list match the repo-used public types.
- Residual note: metadata gold-match quality improved substantially, but reruns still showed occasional variability in the `pulmonary_embolism_wrong_existing_reassess` case for full subspecialty recovery (`CH` vs `CH`+`ER`). No contract, validator, or span-execution failures were observed.

## Restructure Spec

Replace the current internal layout with:

```text
packages/findingmodel/src/findingmodel/
  __init__.py
  _id_gen.py
  types/
    __init__.py
    metadata.py
    attributes.py
    models.py
  finding_info.py
  abstract_finding_model.py
  create_stub.py
  index.py
  index_validation.py
  tools/
```

### `types/metadata.py`

Move the entire canonical metadata layer from `facets.py` into `types/metadata.py`. This file owns:

- enums and structured models: `BodyRegion`, `Subspecialty`, `Modality`, `EntityType`, `SexSpecificity`, `AgeStage`, `AgeProfile`, `ExpectedDuration`, `TimeCourseModifier`, `ExpectedTimeCourse`, `EtiologyCode`
- legacy normalization maps and normalizers: `BODY_REGION_LEGACY_MAP`, `ETIOLOGY_LEGACY_MAP`, `MODALITY_LEGACY_MAP`, `AGE_LABEL_LEGACY_MAP`, `normalize_body_region()`, `normalize_etiology()`, `normalize_modality()`, `normalize_age_label()`
- annotated normalizing aliases: `NormalizedBodyRegionList`, `NormalizedEtiologyList`, `NormalizedModalityList`, `NormalizedAgeProfile`
- formatting helpers: `format_age_profile()`, `format_time_course()`

### `types/attributes.py`

Move the attribute layer from `finding_model.py` into `types/attributes.py`. This file owns:

- `AttributeType`
- attribute and value aliases and constraints: `AttributeId`, `AttributeValueCode`, `IndexCodeList`, `AttributeNameStr`, `AttributeDescriptionStr`, `RequiredBool`, `MaxSelectedInt`, `MinimumNumeric`, `MaximumNumeric`, `UnitStr`
- shared helpers: `_index_codes_str()`, `fix_max_selected_validator()`, `ATTRIBUTE_FIELD_DESCRIPTION`
- models: `ChoiceValue`, `ChoiceValueIded`, `ChoiceAttribute`, `ChoiceAttributeIded`, `NumericAttribute`, `NumericAttributeIded`
- unions: `Attribute`, `AttributeIded`

### `types/models.py`

Move the model layer from `finding_model.py` into `types/models.py`. This file owns:

- model-only aliases and constraints: `NameString`, `DescriptionString`, `SynonymSequence`, `TagSequence`, `OifmIdStr`, `Contributor`, `ATTRIBUTES_FIELD_DESCRIPTION`
- models: `FindingModelBase`, `FindingModelFull`
- markdown rendering methods remain on the model classes
- `FindingModelFull.index_codes_str` remains here because it is model behavior

### `._id_gen.py`

Move ID generation into private `._id_gen.py`. This file owns:

- `ID_LENGTH`
- `_random_digits()`
- `generate_oifm_id()`
- `generate_oifma_id()`

Do not export `generate_oifm_id()`, `generate_oifma_id()`, or `._id_gen` at the top level. They are internal only. Current repo usage is internal to `index.py`.

### `types/__init__.py`

`types/__init__.py` re-exports moved symbols for internal package use only. It is not documented or treated as stable API.

### `findingmodel/__init__.py`

`findingmodel/__init__.py` becomes the only supported package surface and must explicitly re-export every symbol the repo will use from the top level after the migration:

- metadata types and helpers: `AgeProfile`, `AgeStage`, `BodyRegion`, `EntityType`, `EtiologyCode`, `ExpectedDuration`, `ExpectedTimeCourse`, `Modality`, `SexSpecificity`, `Subspecialty`, `TimeCourseModifier`, `format_age_profile`, `format_time_course`
- attribute and model types: `AttributeType`, `ChoiceValue`, `ChoiceValueIded`, `ChoiceAttribute`, `ChoiceAttributeIded`, `NumericAttribute`, `NumericAttributeIded`, `FindingModelBase`, `FindingModelFull`, `IndexCodeList`
- existing package exports that stay: `FindingInfo`, `Index`, `RelatedModelWeights`, `IndexCode`, `WebReference`, `EmbeddingCache`, `create_model_stub_from_info`, `get_settings`, `logger`, `tools`

### Mandatory migration targets inside `packages/findingmodel`

- `abstract_finding_model.py`: import `AttributeType` from `.types.attributes`
- `create_stub.py`: import model and attribute types from `.types.models` and `.types.attributes`
- `fm_md_template.py`: import model and attribute types from `.types.models` and `.types.attributes`
- `index.py`: import metadata enums and models from `.types.metadata`, model classes from `.types.models`, and `_random_digits` from `._id_gen`
- `index_validation.py`: import `FindingModelFull` from `.types.models`
- `tools/__init__.py`: import model classes from `.types.models`
- `tools/index_codes.py`: import `ChoiceAttributeIded`, `FindingModelFull`, and `IndexCodeList` from `.types.attributes` and `.types.models`

### Mandatory repo migration targets outside `packages/findingmodel`

- `packages/findingmodel-ai/src`: `cli.py`, `metadata/types.py`, `authoring/editor.py`, `authoring/markdown_in.py`, `search/pipeline_helpers.py`
- `packages/findingmodel-ai/evals`: existing eval suites that import `finding_model`
- `packages/findingmodel-ai/scripts`
- `packages/findingmodel-ai/tests`
- `packages/oidm-maintenance/src` and tests
- `packages/findingmodel/tests`

### Import migration rule

- inside `packages/findingmodel`, use relative imports from `.types.*` and `._id_gen`
- outside `packages/findingmodel`, use top-level `findingmodel` imports only

### Naming cleanup rule

- rename schema and state uses of "facet" to "metadata" in active docs, test names, helper names, and comments
- do not rename search and filter faceting code purely for aesthetics
- `test_facets.py` becomes the metadata-type test file
- browse and search tests that are actually about facet filtering keep that terminology

### Deletion order

The deletion order is fixed:

1. add the new modules with identical behavior
2. switch `findingmodel/__init__.py` to re-export from the new modules
3. migrate all repo imports
4. update active tests, docs, and names
5. run grep gates
6. delete `facets.py` and `finding_model.py`

### Grep deletion gates

- no active repo imports from `findingmodel.facets`
- no active repo imports from `findingmodel.finding_model`
- no internal package imports from the removed files

## Metadata Pipeline and Evals Spec

Change the API to:

```python
async def assign_metadata(
    finding_model: FindingModelFull,
    *,
    fill_blanks_only: bool = False,
) -> MetadataAssignmentResult
```

### Metadata behavior

Default mode is reassess-and-update:

- always gather ontology candidates
- always gather anatomic candidates
- always run the classifier
- always present existing structured metadata, `index_codes`, and `anatomic_locations` as draft context

`fill_blanks_only=True` is the preservation mode:

- still gather and still classify
- only populate currently-empty fields, codes, and anatomy
- never overwrite populated fields
- ignore `clear_fields` with warning

Keep the decision model minimal:

- retain the current typed metadata fields
- add `clear_fields` limited to `body_regions`, `subspecialties`, `etiologies`, `entity_type`, `applicable_modalities`, `expected_time_course`, `age_profile`, `sex_specificity`
- omitted field means unchanged
- non-null field means replace in reassess mode or fill-if-empty in fill-blanks mode

Standardize candidate IDs everywhere to `SYSTEM:CODE`. That exact string format is used in prompt payloads, merged existing candidate state, classifier decisions, warning logic, and final application. Raw `concept_id` strings are no longer valid decision IDs.

Add `MetadataAssignmentReview.assignment_mode` with values `reassess` and `fill_blanks_only`.

Add `@agent.output_validator` plus `ModelRetry` for:

- hallucinated ontology and anatomic candidate IDs
- missing required fields in reassess mode
- missing currently-blank required fields in fill-blanks mode

### Eval suite

Add `packages/findingmodel-ai/evals/metadata_assignment.py` using the same structure as the existing eval suites:

- `MetadataAssignmentInput`
- `MetadataAssignmentExpectedOutput`
- `MetadataAssignmentActualOutput`
- `MetadataAssignmentCase`
- inline evaluators only
- `Dataset`
- `run_metadata_assignment_evals()`
- `ensure_instrumented()` in `__main__`

Gold outputs come from existing fixture files in `packages/findingmodel/tests/data/defs/{fixture_stem}.fm.json`. Do not hand-maintain duplicated expected metadata dicts in the eval file.

`MetadataAssignmentInput` fields are fixed:

- `fixture_stem`
- `assignment_mode` as `reassess` or `fill_blanks_only`
- `scenario` as `blank_start`, `wrong_existing_reassess`, `partial_existing_fill_blanks_only`, or `existing_codes_and_anatomy`

`MetadataAssignmentExpectedOutput` fields are fixed:

- `gold_fixture_stem`
- `must_match_fields`
- `locked_fields`
- `required_fields`
- `expect_unknown_candidate_warnings`
- `require_execution_spans`

`MetadataAssignmentActualOutput` fields are fixed:

- final `FindingModelFull`
- final `MetadataAssignmentReview`
- prepared input snapshot
- offered ontology candidate IDs
- offered anatomic candidate IDs
- selected ontology candidate IDs
- selected anatomic candidate IDs
- warnings
- error string

Keep all prep helpers local to this eval file:

- `_load_gold_fixture()`
- `_prepare_blank_start()`
- `_prepare_wrong_existing_reassess()`
- `_prepare_partial_existing_fill_blanks_only()`
- `_prepare_existing_codes_and_anatomy()`

### Initial eval case matrix

- `pulmonary_embolism_blank_start`: blank all metadata, codes, and anatomy; mode `reassess`; final output must match gold for every populated metadata field and canonical code and location
- `abdominal_aortic_aneurysm_blank_start`: same
- `breast_density_blank_start`: same; this is the measurement, female-specific, age-profile gold case
- `aortic_dissection_blank_start`: same; this is the vascular diagnosis case without etiology gold
- `pulmonary_embolism_wrong_existing_reassess`: inject `body_regions=[ABDOMEN]`, `subspecialties=[AB]`, `entity_type=MEASUREMENT`, `applicable_modalities=[US]`, `etiologies=[INFLAMMATORY_INFECTIOUS]`, `sex_specificity=FEMALE_SPECIFIC`; mode `reassess`; final structured metadata must return to gold
- `breast_density_partial_existing_fill_blanks_only`: preserve gold `body_regions`, `entity_type`, and `applicable_modalities`; blank `age_profile`, `sex_specificity`, and `anatomic_locations`; mode `fill_blanks_only`; preserved fields must remain unchanged and blank fields must be repopulated to gold
- `abdominal_aortic_aneurysm_existing_codes_and_anatomy_reassess`: preserve gold `index_codes` and `anatomic_locations`, blank structured metadata, mode `reassess`; final codes and anatomy must still match gold and no unknown-candidate warnings may appear

### Eval evaluators

Implement exactly five evaluators:

- `ExecutionSuccessEvaluator`: strict pass/fail on successful run
- `RequiredFieldCoverageEvaluator`: completeness for `body_regions`, `subspecialties`, `entity_type`, `applicable_modalities`
- `GoldMetadataMatchEvaluator`: compare final output to the gold fixture for `must_match_fields`, using normalized dumps and order-insensitive set comparisons
- `PreservationSemanticsEvaluator`: compare `locked_fields` to the prepared input snapshot; only attach to fill-blanks-only cases
- `CandidateIntegrityEvaluator`: strict pass/fail; selected ontology and anatomic IDs must come from offered candidates or preserved starting selections, and warnings must not contain unknown-candidate messages unless the case explicitly expects them

Add built-in `HasMatchingSpan` assertions for:

- `assign_metadata.ontology_candidates`
- `assign_metadata.anatomic_candidates`
- `assign_metadata.classifier`

These apply to every case in the suite and are the eval proof that reassessment no longer silently skips work.

### Task and benchmark wiring

Add `evals:metadata_assignment` to `Taskfile.yml` and include it in `task evals`.

Extend `scripts/benchmark_models.py` rather than inventing a second benchmark surface:

- keep the existing `metadata-assign` comparison set
- split metadata prep into `blank_start` and `improve_existing`
- keep the current fixture-backed findings already used by the benchmark script
- verify default-mode traces include the gather and classifier spans in both benchmark modes

## Test and Verification

Replace the current fast-path skip test with `test_assign_metadata_reassesses_populated_model` and assert ontology gathering, anatomic gathering, and classifier execution all occur on a complete model in default mode.

Add unit tests for `fill_blanks_only`:

- preserves populated structured metadata
- fills blank structured metadata
- preserves populated `index_codes` and `anatomic_locations`
- ignores `clear_fields` with warning

Add unit tests for the `SYSTEM:CODE` candidate-ID contract:

- prompt payload uses `SYSTEM:CODE`
- decisions using `SYSTEM:CODE` apply to both gathered and pre-existing candidates
- unknown candidate IDs trigger validator retry or error

Add validator tests using `TestModel` and `FunctionModel`:

- missing required fields retry in reassess mode
- only currently-blank required fields retry in fill-blanks mode
- hallucinated candidate IDs retry

Add restructure verification tests:

- top-level imports work for every documented and repo-used public symbol
- `FindingModelBase` and `FindingModelFull` round-trip behavior is unchanged
- markdown rendering is unchanged
- attribute validation is unchanged
- ID generation behavior is unchanged after moving to `._id_gen`

Rename and retarget tests as part of the cleanup:

- `test_facets.py` becomes the metadata-type suite
- model-only tests stop importing old submodules
- search and filter tests keep facet terminology only where they actually test faceting

Final verification commands:

- `task check`
- `task test:findingmodel`
- `task test:findingmodel-ai`
- `task test:maintenance`
- `task evals:metadata_assignment`
- `task evals`
- `task verify:install`

## Review Notes (March 22, 2026)

Reviewed against current codebase state for consistency:

1. **`MetadataAssignmentReview` does NOT have a `model_tier` field.** It has `model_used: str` which is correct. No cleanup needed on that type. (An earlier draft incorrectly flagged this.)

2. **`_id_gen.py` naming is correct** — `generate_oifm_id()` and `generate_oifma_id()` are only called internally by `finding_model.py`. Only `_random_digits()` is imported externally (by `index.py`). Making the module private is the right call.

3. **Naming cleanup nuance is good** — the distinction between "facet" as metadata-type naming (rename) vs "facet" as a filtering concept in search/browse (keep) is the right call. `browse()` and `search()` do facet-based filtering; that's standard terminology.

4. **`SYSTEM:CODE` candidate ID standardization** is new scope beyond the original punch list. Implementers should note this touches the prompt payload format, decision model, warning logic, and final application — not just a cosmetic rename.

5. **Current model defaults are Logfire-verified and should be preserved.** The `metadata_assign` agent uses `gpt-5.4-mini/low` as primary (classification task). The upstream agents (ontology_search, anatomic_search, ontology_match, anatomic_select) use `gpt-5.4-nano` (generative) and `gpt-5.4-mini` (classification) per the March 22 full-model-shootout benchmarks. These are configured in `supported_models.toml` and should not be changed by this work unless benchmarks show a reason.

6. **Files referenced in the restructure spec all exist** — `abstract_finding_model.py`, `create_stub.py`, `index_validation.py`, `tools/index_codes.py` are all present in the current codebase.

7. **The `fm_md_template.py` module** exists in the package but is not mentioned in the restructure migration targets. It imports from `finding_model.py` and will need updating:
   ```
   packages/findingmodel/src/findingmodel/fm_md_template.py
   ```

## Assumptions and Defaults

- supported public imports are top-level `findingmodel` imports only
- `findingmodel.types` and `._id_gen` are internal structure, not public API
- removing `findingmodel.facets` and `findingmodel.finding_model` is intentional and happens in the same change once grep gates pass
- candidate IDs are standardized to `SYSTEM:CODE` everywhere
- default metadata mode is reassess-and-update; preservation is opt-in via `fill_blanks_only`
- the initial metadata eval suite is deterministic, fixture-backed, and local to `metadata_assignment.py`; no reusable evaluator module and no LLM-judge scoring in this slice
