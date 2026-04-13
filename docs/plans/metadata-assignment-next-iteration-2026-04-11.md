# Plan: Metadata Assignment Next Iteration

Status: In Progress (2026-04-11)

## Goal

Use the just-completed full-suite rerun to drive the next prompt iteration for metadata assignment, focusing on the remaining semantic mismatches rather than rerunning broadly without a concrete hypothesis.

## Trigger

The full metadata-assignment suite completed under Logfire trace `019d7e60c9668f1de39ed980fc3193eb` with:

- `139` cases
- weighted overall score `0.90`
- `GoldMetadataMatchEvaluator` average `0.831`
- execution, coverage, preservation, and candidate-integrity all at `1.00`

That means the remaining issues are still semantic. They are concentrated enough that the next pass should be based on direct inspection of the failing cases.

## Inputs

- full-suite Logfire trace: `019d7e60c9668f1de39ed980fc3193eb`
- current metadata-assignment prompt:
  - `packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py`
- current metadata-assignment eval harness:
  - `packages/findingmodel-ai/evals/metadata_assignment.py`
- reviewed metadata field definitions:
  - `docs/finding-model-metadata-fields.md`
- prior targeted prompt-honing plan:
  - `docs/plans/metadata-assignment-targeted-prompt-honing-2026-04-11.md`

## Deliverables

1. a concrete analysis of the remaining failure clusters from the latest full-suite run
2. a revised prompt that addresses those residual clusters directly
3. targeted traced validation for the revised prompt
4. updated docs capturing what changed and what remains

## Workstreams

### Workstream A: Failure Review

- inspect the latest full-suite failures from Logfire
- group them by scenario and semantic pattern
- compare actual outputs to the reviewed gold fixtures

### Workstream B: Prompt / Evaluator Follow-Up

- decide whether the remaining issues are:
  - prompt clarity problems
  - missing examples
  - evaluator blind spots
- patch only what is justified by the trace review

### Workstream C: Targeted Validation

- rerun only the cases that represent the dominant residual clusters
- inspect Logfire traces again before deciding on another full-suite rerun

### Workstream D: Documentation

- update this plan with the concrete failure clusters, the chosen fixes, and the targeted outcomes
- update related docs if the recommended prompt strategy materially changes

## Initial Hypothesis

From the current run summary, the remaining issues appear to cluster as:

- `partial_existing_fill_blanks_only`
- `wrong_existing_reassess`
- a smaller number of `blank_start` and `existing_codes_and_anatomy` mismatches

The first likely target is broader subspecialty guidance, because many of the residual `fill_blanks_only` misses appear to involve specialized domain labels beyond the smaller set of examples already added in the previous pass.

## Implementation Plan

1. Write this plan into `docs/plans/`.
2. Pull the imperfect cases from the latest full-suite trace.
3. Compare representative actual outputs against the gold fixtures.
4. Research any additional prompt tactics only if the failure review shows a real gap.
5. Patch the prompt and, if needed, evaluator expectations.
6. Run targeted traced validation on the residual clusters.
7. Update this plan with results and next steps.

## Failure Clusters Confirmed From Trace Review

Confirmed against full-suite trace `019d7e60c9668f1de39ed980fc3193eb` and the current gold fixture
files:

- entity-type drift:
  - `intracranial_hemorrhage`, `pneumothorax`, and `cardiomegaly` were still being promoted to
    `diagnosis` when the gold fixtures require `finding`
- body-region overreach:
  - `rotator_cuff_tear` drifted from shoulder anatomy to `chest`
  - `meningioma` drifted to `["head","spine"]`
  - `kidney_stone` sometimes widened from `abdomen` to `["abdomen","pelvis"]`
- subspecialty under/over-selection:
  - missing additive tags such as `GU`, `OB`, `OI`, `MI`, `MK`, `PD`
  - over-adding `AB`, `CA`, `CH`, or `VI` when the narrower or more relevant set was already clear
- modality overreach:
  - `pneumothorax` was still drifting toward `MR`

## Prompt Changes Chosen For This Iteration

- weaken the ontology-driven pressure toward `diagnosis` and explicitly state that exact ontology
  matches can still correspond to `finding` in this schema
- make body-region mapping rules more literal:
  - shoulder -> `upper_extremity`
  - ovary/uterus/adnexa/prostate -> `pelvis`
  - do not widen to alternate regions just because the description mentions possible variants
- expand the subspecialty guidance for:
  - `GU` + `OB`
  - `OI` + `MI`
  - `MK` on shoulder/spine/chest-wall trauma
  - `PD` + `GI`
- add direct examples for the concrete residual miss patterns rather than relying on abstract prose

## Targeted Validation Round 1

- traced targeted batch completed under Logfire trace `019d7e7ad62d7075f539e0198e8fb50f`
- case count: `15`
- weighted overall score: `0.94`
- improved to full gold on:
  - `cardiomegaly_wrong_existing_reassess`
  - `cervical_lymphadenopathy_partial_existing_fill_blanks_only`
  - `coronary_artery_calcification_partial_existing_fill_blanks_only`
  - `fdg_avid_pulmonary_nodule_partial_existing_fill_blanks_only`
  - `hepatocellular_carcinoma_partial_existing_fill_blanks_only`
  - `intracranial_hemorrhage_blank_start`
  - `kidney_stone_partial_existing_fill_blanks_only`
  - `ovarian_cyst_partial_existing_fill_blanks_only`
  - `pneumothorax_blank_start`
  - `primary_lung_malignancy_partial_existing_fill_blanks_only`
  - `pyloric_stenosis_partial_existing_fill_blanks_only`
  - `rotator_cuff_tear_wrong_existing_reassess`
  - `vertebral_compression_fracture_partial_existing_fill_blanks_only`
- remaining misses after round 1:
  - `aortic_dissection_partial_existing_fill_blanks_only`
    - still over-added `CH`
  - `meningioma_blank_start`
    - still widened to `head|spine`
    - still over-added `OI` / `MK`
    - still over-added `XR`

## Follow-Up Prompt Adjustment

- make `CH` exclusion stronger for thoracic vascular disease and add an explicit `aortic dissection`
  example
- narrow `OI` so it is not implied for every benign neoplasm
- add an explicit `meningioma` example to prevent region / subspecialty / modality overreach

## Method Correction

- the first prompt pass in this iteration used examples that were too close to eval-suite gold
  cases
- that creates an overfitting risk and is not an acceptable benchmarking setup
- corrected approach for the next measured run:
  - keep gold fixtures eval-only
  - encode lessons as generalized rules
  - use only synthetic or otherwise non-eval examples in the prompt
