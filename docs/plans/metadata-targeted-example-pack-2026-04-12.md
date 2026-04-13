# Plan: Targeted Metadata Prompt Examples

## Goal

Propose a compact set of field-focused prompt examples that teach the metadata-assignment model the
remaining weak spots without embedding gold eval answers, and lock in ontology lookup behavior so
selected `index_codes` retain a populated `display`.

## Status

- Implemented
- 2026-04-12: audited the active ontology lookup path and confirmed candidate conversion already
  carries `IndexCode.display` via `OntologySearchResult.as_index_code()`
- 2026-04-12: added focused metadata-assignment test assertions covering lookup-derived
  `index_codes`, ontology review candidates, and anatomic candidates retaining `display`
- 2026-04-12: integrated partial field-targeted teaching snippets into the active
  metadata-assignment prompt for `etiologies`, `age_profile`, `sex_specificity`,
  `expected_time_course`, and `index_codes`
- 2026-04-12: added `display` to compact ontology/anatomic candidate payloads so prompt-side
  candidate context and final `IndexCode` objects use the same preferred term
- 2026-04-12: focused tests passed:
  - `packages/findingmodel-ai/tests/test_assign_metadata.py`
  - `packages/findingmodel-ai/tests/test_assign_metadata_modes.py`
- 2026-04-12: targeted live metadata-assignment eval slice passed at weighted score `1.0000`
  across:
  - `breast_density`
  - `benign_prostatic_hyperplasia`
  - `ovarian_cyst`
  - `pyloric_stenosis`
  - `pulmonary_embolism`
  - `fdg_avid_pulmonary_nodule`
  and scenarios:
  - `blank_start`
  - `existing_codes_and_anatomy`

## Scope

- propose micro-examples for:
  - `etiologies`
  - `age_profile`
  - `sex_specificity`
  - `expected_time_course`
  - `index_codes`
- keep the examples partial and field-targeted rather than fully enriched unless a full example is
  necessary to teach a boundary
- avoid using reviewed gold-suite cases as prompt examples
- verify that ontology candidate lookup and canonical code assembly preserve `IndexCode.display`

## Proposed Example Targets

### `etiologies`

- `mediastinal lymphadenopathy`
  - teach: `etiologies=["inflammatory:infectious","inflammatory","neoplastic:malignant"]`
  - reason: broad chest node enlargement should not be forced to a single mechanism and should not
    have etiologies left blank
- `bone island (enostosis)`
  - teach: `etiologies=["normal-variant"]`
- `post-radiation enteritis`
  - teach: `etiologies=["iatrogenic:post-radiation"]`

### `age_profile`

- `necrotizing enterocolitis`
  - teach: `applicability=["newborn","infant"]`, `more_common_in=["newborn"]`
- `slipped capital femoral epiphysis`
  - teach: `applicability=["child","adolescent"]`, `more_common_in=["adolescent"]`
- `degenerative lumbar facet arthropathy`
  - teach: `applicability="all_ages"`, `more_common_in=["middle_aged","aged"]`

### `sex_specificity`

- `prostate abscess`
  - teach: `sex_specificity="male-specific"`
- `endometrial polyp`
  - teach: `sex_specificity="female-specific"`
- `renal cyst`
  - teach: `sex_specificity="sex-neutral"`

### `expected_time_course`

- `pulmonary contusion`
  - teach: `expected_time_course={duration:"weeks", modifiers:["resolving"]}`
- `atheromatous plaque burden`
  - teach: `expected_time_course={duration:"permanent", modifiers:["progressive"]}`
- `bone mineral density T-score`
  - teach: `expected_time_course=null`

### `index_codes`

- `pulmonary nodule`
  - teach exact-vs-broader-vs-narrower ontology acceptance
- `bone mineral density T-score`
  - teach exact LOINC measurement code preference and rejection of diagnosis codes when the finding
    is a measurement rather than the disease itself
- `mammographic architectural distortion`
  - teach finding-code acceptance and procedure-code rejection
- `developmental venous anomaly`
  - teach rejection of clinically adjacent but non-equivalent vascular malformation codes

## Implementation Notes

- partial examples must be labeled explicitly so omitted fields are not interpreted as nulls
- ontology examples should include code system, code, and preferred display term in the rendered
  candidate payload
- tests should assert that selected canonical `index_codes` keep `display`
- tests should also assert that ontology review candidates keep `display`

## Final Example Targets

### `etiologies`

- `mediastinal lymphadenopathy`
- `bone island (enostosis)`
- `post-radiation enteritis`

### `age_profile`

- `necrotizing enterocolitis`
- `slipped capital femoral epiphysis`
- `degenerative lumbar facet arthropathy`

### `sex_specificity`

- `prostate abscess`
- `endometrial polyp`
- `renal cyst`

### `expected_time_course`

- `pulmonary contusion`
- `atheromatous plaque burden`
- `bone mineral density T-score`

### `index_codes`

- `pulmonary nodule`
- `bone mineral density T-score`
- `mammographic architectural distortion`
- `developmental venous anomaly`
