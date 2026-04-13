# Plan: Gold Standards And Enrichment Prompt Follow-Up

Status: In Progress (2026-04-09)

## Goal

Use the reviewer feedback in `metadata-standard-review.json` to:

1. update the gold-standard eval fixtures so they reflect the reviewed metadata expectations
2. expand ontology gold standards so reviewed fixtures show LOINC and RadLex codes where appropriate, not only SNOMED
3. review the existing enrichment prompts and identify concrete prompt changes that are already justified by the reviewed outputs and comments

## Inputs

- reviewer comments: `metadata-standard-review.json`
- current gold-standard summary: `docs/gold-standard-review.md`
- metadata assignment evals: `packages/findingmodel-ai/evals/metadata_assignment.py`
- gold fixtures: `packages/findingmodel-ai/evals/gold/*.fm.json`
- ontology/enrichment code paths and prompts in `packages/findingmodel-ai/src/findingmodel_ai/search/` and related metadata assignment modules

## Deliverables

1. updated gold fixtures in `packages/findingmodel-ai/evals/gold/`
2. fixture ontology index codes expanded where good LOINC and/or RadLex codes exist
3. any required eval expectation updates if the revised gold standards change matching behavior
4. a prompt-review section in this plan capturing prompt changes that are already warranted before further experimentation
5. documentation updates reflecting what changed and why

## Workstreams

### Workstream A: Apply Reviewed Gold-Standard Corrections

Use `metadata-standard-review.json` as the source of truth for fixture corrections.

Examples already called out by review:

- age applicability corrections:
  - `abdominal_aortic_aneurysm`
  - `aortic_dissection`
  - `cardiomegaly`
  - `cerebral_infarction`
  - `coronary_artery_calcification`
  - `meningioma`
  - `ovarian_cyst`
  - `pneumothorax`
- synonym cleanup:
  - `aortic_dissection` should not include `aortic rupture`
  - `distal_radius_fracture` should not include `colles fracture` as a synonym
- etiology corrections:
  - `aortic_dissection` should not be `vascular:hemorrhagic`
  - `benign_prostatic_hyperplasia` should be `neoplastic:benign`
  - `cardiomegaly` should not use `degenerative`
  - `pneumothorax` should consider iatrogenic support where appropriate
- modality corrections:
  - `distal_radius_fracture` should include `MR`
  - `motion_artifact` should include `XR` and `MG`
  - `pleural_effusion` should include `MR`
- time-course corrections:
  - `cerebral_infarction` should evolve over weeks to months rather than be simply stable/permanent
  - `kidney_stone` duration should exceed weeks
  - `primary_lung_malignancy` duration should be years or permanent
  - `ovarian_cyst` may be years depending on etiology

### Workstream B: Expand Ontology Gold Standards

Current concern from review:

- examples appear to show SNOMED codes only
- reviewed standards should show LOINC and RadLex codes wherever there are good, non-forced matches

Planned changes:

- inspect current `index_codes` in all reviewed gold fixtures
- add additional ontology codes where there are clearly appropriate equivalents or clinically substitutable concepts
- prefer high-quality, non-forced additions
- preserve SNOMED as the anchor when it is the strongest canonical code, but do not stop there

Priority targets:

- fixtures with strong RadLex presence expected from radiology usage
- fixtures with established LOINC observation-answer concepts or finding terms
- fixtures where reviewer specifically objected to a weak mapping, such as:
  - `fdg_avid_pulmonary_nodule`
  - `motion_artifact`

### Workstream C: Review Enrichment Prompts For Immediate Changes

Use the reviewed fixture issues plus the earlier trace-review work to identify prompt changes that are already justified now.

Initial hypotheses to test against the current prompts:

- the enrichment path may be over-prioritizing a single “best” ontology family instead of returning multiple good ontology-family matches
- synonym generation may be too permissive about near-neighbors and subtypes
- ontology categorization may be too willing to keep weak or broader concepts in a way that biases downstream canonical examples
- modality and age-profile enrichment may still be too eager to compress commonness into applicability

Expected prompt-review output:

- concrete prompt changes we already want
- concrete prompt changes we are deliberately deferring pending more evidence

## Implementation Plan

1. Update this plan as the fixture review is translated into concrete file changes.
2. Read the remaining review comments in `metadata-standard-review.json` and build a fixture-by-fixture edit list.
3. Inspect the current gold fixture JSON files for the reviewed cases.
4. Update the gold fixture metadata fields to match the review comments.
5. Inspect existing ontology index-code coverage across the gold fixtures.
6. Add LOINC and RadLex codes to gold fixtures where there are good reviewed additions to make.
7. Check `packages/findingmodel-ai/evals/metadata_assignment.py` for any expectation changes needed because gold standards now contain broader ontology code sets.
8. Review the current enrichment prompts and document immediate prompt changes in this plan before making any prompt edits.
9. Run focused evals/tests covering the updated gold standards.
10. Update related docs so the repository reflects the final reviewed standard.

## Validation

Success criteria:

- reviewed fixture metadata reflects the reviewer comments
- gold fixtures show multi-ontology coverage where appropriate instead of defaulting to SNOMED-only examples
- weak or explicitly rejected ontology mappings are removed from gold standards
- prompt-review recommendations are tied to specific reviewed errors, not generic speculation
- relevant evals still run and clearly report any intentional expectation shifts

## Progress Update (2026-04-09)

- Applied reviewer-driven metadata corrections across the reviewed gold fixture set.
- Updated reviewed fixtures for:
  - age applicability and more-common-in fixes
  - synonym cleanup where subtype or non-equivalent terms were being treated as synonyms
  - etiology, modality, subspecialty, and time-course corrections
  - broader spine anatomic targets for generic spine fixtures
- Expanded ontology coverage in the reviewed set using current ontology search results.
- Current reviewed-set ontology coverage after the update:
  - `15` of `27` reviewed fixtures now include more than one ontology family in `index_codes`
  - added LOINC and/or RadLex only where the available match looked acceptable enough to serve as a gold-standard example
- Validation completed:
  - all `27` modified gold fixtures load successfully as `FindingModelFull`
  - `task evals:metadata_assignment` completed with overall score `0.99`
  - all cases executed successfully and all span assertions passed
  - the only imperfect case in that run was `aortic_dissection_blank_start`, where `GoldMetadataMatchEvaluator` scored `0.5`
  - an immediate direct rerun of that case matched the gold `body_regions` and `entity_type`, so the mismatch currently looks like run-to-run variability rather than a stable broken fixture

## Prompt Review Notes

Reviewed prompt surfaces:

- `packages/findingmodel-ai/src/findingmodel_ai/search/ontology.py`
  - `create_query_generator_agent()`
  - `create_categorization_agent()`
- `packages/findingmodel-ai/src/findingmodel_ai/prompt_templates/unified_enrichment_classifier.xml.jinja`

Concrete prompt changes already justified by the reviewed output:

1. Stop explicitly prioritizing SNOMED over other exact ontology matches.

Current issue:

- both the ontology categorization prompt and the unified enrichment template explicitly bias toward SNOMED
- this directly conflicts with the desired gold-standard behavior of showing LOINC and RadLex where good equivalents exist

Wanted change:

- replace “prioritize SNOMED” with:
  - keep all true exact matches across ontology families
  - when multiple exact matches exist, preserve cross-system coverage rather than collapsing to SNOMED-only
  - prefer clinically equivalent finding concepts, not one ontology family by default

2. Reduce broader-term and subtype drift in query generation.

Current issue:

- the query generator prompt explicitly asks for:
  - broader terms
  - more general or specific variations
  - parent/broader terms that might categorize the finding
- this is consistent with reviewer objections such as:
  - `FDG-avid pulmonary nodule` drifting toward `solitary pulmonary nodule`
  - subtype concepts being treated too much like synonyms

Wanted change:

- make the generator prioritize:
  - lexical variants
  - clinically interchangeable synonyms
  - standard abbreviations only when canonical
- demote broader categories and subtypes to backup-only behavior rather than normal output

3. Make subtype and narrower-concept handling stricter in ontology categorization.

Current issue:

- the current `should_include` guidance explicitly welcomes subtypes and variants
- reviewer feedback repeatedly distinguishes:
  - subtype vs synonym
  - more specific concept vs equivalent concept

Wanted change:

- explicitly state:
  - subtypes are not synonyms
  - narrower concepts should not be shown as exact matches
  - narrower concepts should only be retained when they are intentionally useful as related concepts, not canonical examples
- add negative examples:
  - `Colles fracture` is not a synonym for `distal radius fracture`
  - `supraspinatus tear` is not a synonym for `rotator cuff tear`

4. Exclude specimen, procedure, and impression concepts more aggressively.

Current issue:

- ontology search still surfaces concepts that are not the finding itself:
  - procedure concepts
  - specimen-like concepts such as `pleural fluid`
  - impression/report terms

Wanted change:

- extend exclusion guidance to explicitly reject:
  - specimen concepts when the finding is the pathologic state

## Additional Prompt Findings From Expanded Metadata-Assignment Evals (2026-04-10)

The expanded full-gold metadata-assignment eval run added a few concrete prompt problems beyond the earlier gold review.

1. `fill_blanks_only` needs to say "fill all supported blank metadata fields," not merely "do not overwrite existing fields."

Observed issue:

- successful `fill_blanks_only` runs can still leave supported blank fields under-filled
- example: `abdominal_aortic_aneurysm_partial_existing_fill_blanks_only` returned only `VI` or `VI|ER` instead of the reviewed gold `AB|VI|ER`

Wanted change:

- explicitly instruct the classifier that in `fill_blanks_only` mode it should populate every blank structured field that is clearly supported by the finding and candidate evidence
- emphasize that preserving locked fields does not mean being minimal about the blank ones

2. Subspecialty enum boundaries need sharper guidance.

Observed issue:

- `acute_appendicitis` drifted to `GI|ER` instead of `AB|ER`
- `cardiomegaly` and `pericardial_effusion` drifted away from `CA|CH`
- vascular findings can over-collapse to `VI` while dropping the body-region specialty such as `AB`

Wanted change:

- add explicit enum-boundary examples for:
  - `AB` vs `GI`
  - `CA` vs `CH`
  - when `VI` should be additive rather than replacing the regional specialty
  - when `ER` is additive rather than a substitute for the core specialty

3. `entity_type` needs stronger finding-vs-diagnosis counterexamples.

Observed issue:

- `cardiomegaly` and `pericardial_effusion` reassess cases were pushed to `diagnosis`

Wanted change:

- add explicit examples showing that descriptive imaging observations can remain `finding` even when they are clinically important and often discussed like diagnoses

4. Global technique issues need explicit `whole_body` handling.

Observed issue:

- `motion_artifact_blank_start` correctly reached `technique_issue` but left `body_regions` null instead of `whole_body`

Wanted change:

- add a rule that generalized technique artifacts and global acquisition problems may use `whole_body` when no narrower anatomic region is appropriate

5. Retry tuning should be deferred until after the above prompt tightening.

Observed issue:

- the expanded run still had `13` execution failures, all `partial_existing_fill_blanks_only`, all `Exceeded maximum retries (1) for output validation`

Interpretation:

- this does justify looking at retry budget later
- but because the same run also exposed real semantic under-fill and enum-boundary mistakes, raising retries first would risk masking unstable prompt/schema behavior rather than fixing it
  - report-section / impression concepts
  - procedures or interventions even if text overlap is high

5. Add cross-system exact-match examples to the prompt examples.

Current issue:

- existing examples reinforce SNOMED-first behavior
- they do not show the desired “multiple exact ontology systems can all be right” pattern clearly enough

Wanted change:

- add examples like:
  - `pulmonary embolism` -> SNOMED + LOINC + RadLex exact matches
  - `thyroid nodule` -> SNOMED + LOINC + RadLex exact matches
  - `motion artifact` -> RadLex exact match and rejection of generic `artifact`

6. Tighten the unified enrichment template’s `should_include` semantics.

Current issue:

- the XML template currently defines `should_include` as “subtypes, variants, strong associations”
- this encourages the exact category boundary problems seen in review

Wanted change:

- rewrite `should_include` guidance to favor:
  - clinically adjacent but non-equivalent concepts only when genuinely helpful
  - no automatic inclusion of narrower concepts
  - no automatic inclusion of parent categories just because they are related

7. Keep the ontology-family expansion conservative.

Current issue:

- not every finding has a good LOINC or RadLex exact equivalent
- forcing cross-system coverage would create weak gold standards

Wanted change:

- add guidance that LOINC/RadLex should be included when the match is genuinely exact or clinically substitutable
- do not force non-SNOMED codes where the available concept is broader, specimen-based, procedural, or otherwise not the same finding

## Documentation Follow-Up

When the work lands:

- update this plan with the final fixture list touched
- update `docs/gold-standard-review.md` if it is still meant to reflect the current gold state
- review `CHANGELOG.md` only if the resulting gold-standard changes materially alter externally visible evaluation expectations
