# Plan: Metadata Assignment Targeted Prompt Honing

Status: Completed (2026-04-11)

## Goal

Improve the metadata-assignment prompt so it performs better on the concrete failure patterns exposed by the reviewed gold corpus, without masking prompt weaknesses by immediately broadening retries or rerunning the full suite.

## Problem Statement

The latest full reviewed-corpus eval run is structurally stable, but it still shows semantic mismatches concentrated in a few repeatable patterns:

1. `fill_blanks_only` cases still under-fill clearly supported metadata
2. `wrong_existing_reassess` cases can remain anchored to incorrect existing metadata instead of overriding it
3. subspecialty assignment is still drifting across nearby labels such as `AB` vs `GI` and `CA` vs `CH`
4. some descriptive observations still drift toward the wrong `entity_type`
5. generalized technique issues can over-generalize modality coverage

Those are prompt-clarity problems first. They should be addressed with clearer decision instructions, explicit positive and negative examples, and targeted traced runs before any retry-policy changes.

## Inputs

- current metadata-assignment prompt: `packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py`
- reviewed metadata field definitions: `docs/finding-model-metadata-fields.md`
- reviewed gold fixtures: `packages/findingmodel-ai/evals/gold/*.fm.json`
- expanded eval harness: `packages/findingmodel-ai/evals/metadata_assignment.py`
- prior follow-up docs:
  - `docs/plans/metadata-assignment-full-gold-suite-expansion-2026-04-10.md`
  - `docs/plans/gold-standards-and-enrichment-prompt-followup-2026-04-09.md`
  - `docs/plans/metadata-prompt-improvements-from-head-ct-traces.md`
- Logfire traces from recent targeted and full eval runs
- current primary-source prompt guidance gathered during this pass

## Deliverables

1. a revised metadata-assignment prompt with clearer field-boundary and mode-specific guidance
2. documented prompt-improvement rationale tied to observed failures, not generic prompt advice
3. targeted Logfire-traced runs covering the known problem cases
4. plan/documentation updates recording what improved, what did not, and what remains open

## Workstreams

### Workstream A: Translate Observed Failures Into Prompt Requirements

- enumerate the concrete failure modes from the reviewed-corpus traces
- anchor the prompt changes to repo-local field semantics instead of ad hoc intuition
- separate prompt issues from retry-policy or execution-policy issues

### Workstream B: Research Current Prompt-Clarity Guidance

- use current primary-source documentation to review:
  - how to make structured extraction instructions more explicit
  - when to use positive and negative examples
  - how to reduce ambiguity in schema-constrained output tasks
- keep the research focused on tactics that can actually improve this prompt

### Workstream C: Tighten The Assignment Prompt

- sharpen reassess-mode guidance so wrong existing metadata is treated as replaceable
- sharpen `fill_blanks_only` guidance so the model completes all clearly supported blank fields
- make subspecialty selection logic explicitly additive and field-semantic rather than anatomy-only
- add concrete examples for high-confusion cases
- tighten modality guidance for generalized technique issues

### Workstream D: Validate With Targeted Logfire Runs

- run only the specific cases that exercise the known failure patterns
- inspect the traced prompts, outputs, and any intermediate spans
- confirm whether the revised language changes the actual decision behavior

### Workstream E: Update Documentation

- update this plan with actual outcomes
- update related prompt-review docs if the new findings materially change the recommended next steps
- review whether `CHANGELOG.md` should be updated once the scope is complete

## Targeted Cases

Initial focus set for this pass:

- `abdominal_aortic_aneurysm_partial_existing_fill_blanks_only`
- `acute_appendicitis_blank_start`
- `pericardial_effusion_wrong_existing_reassess`
- `cardiomegaly_wrong_existing_reassess`
- `subdural_hematoma_wrong_existing_reassess`
- `motion_artifact_blank_start`

## Implementation Plan

1. Write this plan into `docs/plans/`.
2. Re-read the current prompt, field-definition docs, and failing-case outputs together.
3. Review current primary-source prompt guidance and extract only the tactics that apply here.
4. Revise the metadata-assignment prompt with clearer decision procedures and examples.
5. Run targeted cases with Logfire tracing enabled.
6. Inspect traces and compare prompt text to outputs for those cases.
7. Record concrete improvements, remaining misses, and any justified next-step changes.
8. Review whether related docs and `CHANGELOG.md` need final updates.

## Validation

Success criteria:

- targeted cases show fewer semantic mismatches than the immediately prior prompt
- traces show the model following the intended field boundaries more consistently
- prompt changes are concrete enough that an intelligent but naive colleague could apply the same rules
- documentation reflects the actual results of the pass rather than planned intent only

## Progress Update (2026-04-11)

- Created this plan for a targeted prompt-honing pass instead of another broad rerun.
- Confirmed the current failure cluster is semantic rather than execution-related:
  - the latest full reviewed-corpus run completed without execution failures
  - remaining misses are concentrated in subspecialty selection, reassess behavior, fill-blanks completeness, and a few entity/modality boundary cases
- Reviewed current primary-source guidance from:
  - OpenAI prompt engineering docs
  - OpenAI GPT-5.4 prompt-guidance docs, especially the guidance for `gpt-5.4-mini`
  - OpenAI structured-outputs docs
  - Pydantic AI output-validator docs
- The specific tactics adopted from those sources were:
  - make the output contract more explicit instead of relying on schema alone
  - use clearer execution-order and mode-specific rules for small, literal models
  - add concrete positive/negative examples for the known ambiguous cases
  - use output validation to reject structurally invalid “fixes” instead of letting them score as successful runs

## Implemented Changes

### Prompt Changes

- Rewrote the metadata-assignment system prompt into explicit sections:
  - objective
  - assignment-mode contract
  - field rules
  - subspecialty rules
  - canonical examples
  - candidate rules
  - output discipline
- Added explicit reassess guidance that existing metadata is provisional and may be wrong.
- Added explicit fill-blanks guidance that the model should fill every clearly supported blank field, not stop after one.
- Added an explicit rule that `classification_rationale` is not a substitute for populating corrected structured fields.
- Added explicit examples for:
  - `acute appendicitis`
  - `abdominal aortic aneurysm`
  - `pericardial effusion`
  - `cardiomegaly`
  - `subdural hematoma`
  - `motion artifact`
- Tightened wording around:
  - `AB` vs `GI`
  - `CA` vs `CH`
  - additive `ER` and `VI`
  - routine/direct modalities vs merely possible modalities
  - keeping etiologies short for base findings
  - leaving `subspecialties` null for generalized technique issues

### Validator / Eval Changes

- Tightened the reassess-mode validator so required fields cannot be “fixed” by clearing them without replacement.
- Tightened fill-blanks validation so blank required fields must actually be populated in the projected result.
- Added the core required-field list to the decision prompt payload for clearer model-side grounding.
- Updated the metadata-assignment eval overall-score weighting so `GoldMetadataMatchEvaluator` contributes about `60%` of the weighted total.
  - implemented weights:
    - `GoldMetadataMatchEvaluator`: `0.60`
    - `RequiredFieldCoverageEvaluator`: `0.15`
    - `ExecutionSuccessEvaluator`: `0.10`
    - `PreservationSemanticsEvaluator`: `0.10`
    - `CandidateIntegrityEvaluator`: `0.05`
- Added/updated focused unit tests so the stricter validator contract is covered.

## Targeted Trace Results

### Pass 1: Six-Case Targeted Run

- Logfire trace: `019d7e517d279f2f01d9bc9a489225b9`
- Cases:
  - `abdominal_aortic_aneurysm_partial_existing_fill_blanks_only`
  - `acute_appendicitis_blank_start`
  - `pericardial_effusion_wrong_existing_reassess`
  - `cardiomegaly_wrong_existing_reassess`
  - `subdural_hematoma_wrong_existing_reassess`
  - `motion_artifact_blank_start`
- Weighted overall score for that batch: `0.90`
- Confirmed improvements:
  - `acute_appendicitis` corrected to `AB|ER`
  - `cardiomegaly_wrong_existing_reassess` corrected to `chest`, `CA|CH`, `finding`, `XR|CT`
  - `subdural_hematoma_wrong_existing_reassess` corrected to `head`, `NR|ER`, `diagnosis`, `CT|MR`
  - `pericardial_effusion_wrong_existing_reassess` corrected the main reassess fields instead of staying anchored to abdomen/measurement/US
- Residual issues from this pass:
  - `abdominal_aortic_aneurysm` still missed `ER` and over-added `XR`
  - `motion_artifact` still over-generalized modalities and etiologies
  - `pericardial_effusion` still overfilled modalities/etiologies even though the gold-match score for that scenario was perfect

### Pass 2: Three-Case Residual Run

- Logfire trace: `019d7e5300f497f26e161d671e2a2274`
- Cases:
  - `abdominal_aortic_aneurysm_partial_existing_fill_blanks_only`
  - `motion_artifact_blank_start`
  - `pericardial_effusion_wrong_existing_reassess`
- Weighted overall score for that batch: `1.00`
- Confirmed improvements:
  - `abdominal_aortic_aneurysm` now returns `AB|VI|ER` with `CT|US|MR`
  - `pericardial_effusion` now returns `US|CT|MR` without the prior X-ray overreach and with conservative etiologies
- New residual issue exposed:
  - `motion_artifact` now had the right modalities and null etiologies, but started inventing subspecialties for a generalized technique issue

### Pass 3: One-Case Motion-Artifact Verification

- Logfire trace: `019d7e541ca27d2abe19a73ca7d8dbec`
- Case:
  - `motion_artifact_blank_start`
- Confirmed final targeted output:
  - `body_regions=["whole_body"]`
  - `subspecialties=null`
  - `entity_type="technique_issue"`
  - `applicable_modalities=["XR","CT","MR","MG"]`
  - `etiologies=null`

## Conclusions

- The prompt changes materially improved the real failure modes seen in the reviewed-corpus traces.
- The validator tightening was justified:
  - it prevented reassess-mode outputs that merely cleared wrong required fields instead of replacing them
  - it forced the tests to reflect the actual minimum contract expected from the classifier
- The weighted overall score now reflects metadata correctness much more honestly than the prior flat average.

## Remaining Gap

- The targeted traces showed that some scenario-level gold checks still do not examine every field that matters.
- In particular, manual trace review caught modality/etiology overshoot in cases whose `GoldMetadataMatchEvaluator` still scored `1.0` because those fields were not in `must_match_fields` for that scenario.
- Follow-up worth considering:
  - tighten scenario-specific `must_match_fields` for selected targeted cases where overfill risk is known
  - or add a dedicated “overreach” evaluator for modalities / etiologies / subspecialties in scenarios where the reviewed gold is intentionally specific

## Documentation Review

- Updated this plan with the actual implemented changes and traced outcomes.
- Did not update `CHANGELOG.md` in this pass because the work is internal prompt/eval tuning rather than an external user-facing behavior change that would be meaningful in the project changelog.
