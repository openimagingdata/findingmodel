# Metadata Prompt Repair Plan

> Superseded for active execution by `docs/plans/metadata-enrichment-current-plan.md`. Keep this file as historical evidence only; pull any still-useful decisions into the active plan or a stable reference doc before acting on them.



## Status

- 2026-05-11: Started pilot implementation. The first checkpoint is `subspecialty_domain`; the
  remaining prompt rewrites are blocked on human review of the pilot prompt and eval results.
- 2026-05-12: Revised the pilot prompt to define subspecialty from the reporting radiologist's
  perspective and removed worklist/workflow framing.
- 2026-05-12: Human review accepted broader overlay behavior for cardiac/chest and
  oncologic-suspicion cases; expanding the pilot eval set before proceeding to other prompts.
- 2026-05-12: Expanded `subspecialty_domain` evals to 43 curated cases. After three concise prompt
  clarifications, the focused deterministic eval passed all cases on two consecutive runs.
- 2026-05-12: Removed `clear_fields` from the pilot subspecialty-domain output contract. Added
  repeat-sampling support for selected eval cases and used it to resolve stochastic overlay
  variants in cervical lymphoid tissue, central venous catheter malposition, and testicular torsion.
- 2026-05-12: Started the next focused prompt pilot, `modality_applicability`, using the
  subspecialty-domain pilot pattern: concise Markdown prompt, focused decision model, agent factory,
  curated evals, and review documentation before production wiring.
- 2026-05-12: Completed the `modality_applicability` pilot surface with a concise prompt, focused
  output model, agent factory, 29-case curated eval suite, repeat-sampling support, and passing
  deterministic evals.
- 2026-05-12: Started the `etiology_tempo` pilot. This combines `etiologies` and
  `expected_time_course`, explicitly using prompt-master to distill the current etiology prompt and
  the time-course portion of the identity prompt before adding the concise pilot prompt and evals.
- 2026-05-12: Completed the `etiology_tempo` pilot surface with a focused output model, prompt,
  agent factory, 20-case eval suite, repeat-sampling support, and passing deterministic evals.

## Decisions

- Use existing prompts as source material, not text to preserve.
- Keep no numeric prompt length cap; use human review plus focused evals as the anti-bloat gate.
- Rewrite all active enrichment prompt surfaces eventually, but do one pilot first.
- Split current subspecialty/modality behavior into `subspecialty_domain` and
  `modality_applicability`.
- Combine etiology and time-course prompt repair into `etiology_tempo`.
- For `etiology_tempo`, use common cause buckets radiologists would reasonably carry for a finding,
  not all theoretically possible causes.
- For `etiology_tempo`, treat time course as whether a prior finding should still be expected on a
  later study, based on observable imaging persistence.
- Do not add deterministic metadata assignment or audit rules for domain decisions.
- Do not change model or reasoning configuration for the assignment pipeline in this pass.
- Use fixture assertions plus a separate stronger LLM grader for prompt-repair evals.
- Keep the current 400-record review package diagnostic-only until regenerated after prompt repair.

## Pilot Checkpoint

The pilot must produce:

- a Markdown prompt file for `subspecialty_domain`;
- a focused eval suite with required and forbidden subspecialty assertions;
- an LLM grader hook using a dedicated grader model config;
- a review document containing the old prompt source material, distilled rules, new prompt, eval
  cases, and actual eval results.

Do not proceed to the rest of the prompts until the pilot prompt style is accepted or revised.

## Full Refactor After Pilot Approval

- Move active enrichment prompts to Markdown package resources.
- Quarantine the legacy aggregate metadata prompt so production does not silently use it.
- Add `modality_applicability` as its own focused agent.
- Compress ontology, anatomy, identity/time-course, etiology, patient applicability, auditor, and
  search prompts around one job each.
- Move most examples into eval fixtures; keep only a few high-signal examples inline.

## Verification Expectations

- Unit tests for prompt loading, resource inclusion, decision models, and pilot agent wiring.
- Focused prompt-repair evals pass fixture assertions and LLM grader checks.
- Existing metadata tests continue to pass for unchanged production behavior.
- Final documentation update records completed prompt rewrites and remaining rerun steps.
