# Metadata Eval Scoring And Prompt Cleanup Plan

> Superseded for active execution by `docs/plans/metadata-enrichment-current-plan.md`. Keep this file as historical evidence only; pull any still-useful decisions into the active plan or a stable reference doc before acting on them.



## Goal

Refactor metadata enrichment evals from brittle exact-match checks into weighted diagnostic scores, then clean up the active metadata prompts so examples illustrate general rules instead of encoding eval misses.

## Decisions

- Cover all metadata enrichment evals.
- Use shared scoring helpers with per-suite weights.
- Prefer recall for set-valued clinical metadata: missing an important expected value usually costs more than adding a plausible extra value.
- Treat forbidden labels as severe errors; treat labels outside allowed sets as precision penalties.
- Keep gold fixtures as reference targets and score softly unless human review identifies a true gold error.
- Use evals as report-first quality instruments. Hard failures are for execution/schema breakage, not ordinary submaximal quality scores.
- Keep optional LLM graders diagnostic only.
- Retire executable legacy aggregate, combined subspecialty/modality, and etiology-only agent paths.
- Rewire production assignment to entity_type, subspecialty_domain, modality_applicability, and etiology_tempo split agents.
- In reassess mode, the owning split agent's null value clears optional metadata.
- Use prompt-master guidance for concise active prompts with general rules and non-eval-name examples.
- Treat `entity_type` as the only required field.
- Remove `clear_fields`; in reassess mode, a split-agent null clears the optional field it owns.
- Carry existing index and anatomic codes forward as selected-by-default review candidates. Silence,
  ties, and uncertainty keep existing codes; anatomic deselection of existing codes requires an
  auditable reason.

## Completed So Far

- [x] Add permanent metadata eval scoring guide.
- [x] Add shared metadata scoring helpers.
- [x] Refactor metadata eval suites to use weighted component scores and summaries.
- [x] Add focused entity_type eval.
- [x] Add `task evals:metadata`.
- [x] Retire legacy combined and etiology-only eval modules.
- [x] Remove executable legacy aggregate classifier fallback and archive distilled rules in docs.
- [x] Rename/narrow identity decision to entity_type.
- [x] Rewire production assignment to split pilots, with etiology_tempo owning etiology and time course.
- [x] Refactor the three pilot prompts to remove exact eval-case examples.
- [x] Update tests, eval docs, and changelog.
- [x] Remove `clear_fields` from live metadata decisions, prompts, assembly, docs, and tests.
- [x] Add metadata-assignment CLI filters and progress/stage output.
- [x] Add smoke, bounded, and full metadata eval task tiers.
- [x] Run targeted tests, lint, type checking, and metadata evals; record final status here.

## Historical Status Correction

- Implemented shared weighted metadata scoring and report summaries, including abstention credit
  for optional blanks and carry-forward handling for existing code extras.
- Rewired production metadata assignment to split entity-type, subspecialty-domain, modality-applicability, and etiology/time-course agents.
- Retired the executable aggregate classifier prompt plus the combined imaging and etiology-only eval modules.
- Removed `clear_fields`; reassess now applies split-agent nulls to optional owned fields and warns
  when existing metadata is cleared.
- Existing index and anatomic codes are selected-by-default review candidates. Existing anatomic
  code deselection now requires an explicit reason.
- Added `--fixture`, `--scenario`, `--case`, and `--limit` filters to `evals.metadata_assignment`.
- Added `task evals:metadata:smoke`, bounded `task evals:metadata`, and `task evals:metadata:full`.
- Verification passed:
  - targeted pytest: `26 passed`.
  - ruff on touched metadata/eval/test files.
  - `uv run mypy`.
  - `task evals:metadata:smoke`.
  - bounded `task evals:metadata`.

The current assignment eval headline score is not acceptable as a metadata-quality metric. It
still mixes pass/fail gates such as execution success, required-field presence, candidate integrity,
and fill-blanks preservation into the weighted score. Those are gates, not quality grades.

## Next Phase: Eval Scoring And Reporting Repair

- [x] Treat execution success, parsed output presence, required `entity_type`, candidate ID integrity,
  fill-blanks preservation, and required spans as boolean gates.
- [x] Gate failures must be reported separately and cause the assignment eval CLI/task to exit
  non-zero.
- [x] Remove gate checks from the headline score.
- [x] Replace the assignment headline with metadata quality only.
- [x] Report per-field quality scores:
  - `quality.entity_type`
  - `quality.body_regions`
  - `quality.subspecialties`
  - `quality.applicable_modalities`
  - `quality.etiologies`
  - `quality.expected_time_course`
  - `quality.index_codes`
  - `quality.anatomic_locations`
  - `quality.age_profile`
  - `quality.sex_specificity`
- [x] Applicability rules:
  - `entity_type`, `body_regions`, `subspecialties`, and `applicable_modalities` are always scored.
  - Other fields are scored only when gold, actual, or starting metadata has a non-empty value.
  - Optional fields where gold and actual are both null do not inflate the headline denominator.
- [x] Default quality weights:
  - entity type `0.15`
  - body regions `0.12`
  - subspecialties `0.14`
  - applicable modalities `0.14`
  - etiologies `0.10`
  - expected time course `0.10`
  - index codes `0.10`
  - anatomic locations `0.10`
  - age profile `0.025`
  - sex specificity `0.025`
- [x] Renormalize weights across the fields scored for each case.
- [x] Preserve existing scoring semantics:
  - set fields use recall-weighted precision/recall;
  - optional blank vs gold-populated field gets conservative abstention credit;
  - existing code extras carried forward are not penalized;
  - newly added extra codes get a moderate precision penalty;
  - entity type keeps limited partial credit for nearby distinctions.
- [x] Replace assignment eval summary output with:
  - `GATES: PASS` or `GATES: FAIL`;
  - gate failure list by case;
  - `METADATA QUALITY: <score>`;
  - per-field averages;
  - lowest-scoring cases with weakest field/reason.

### Eval Scoring Repair Verification

- `uv run --package findingmodel-ai pytest packages/findingmodel-ai/tests/test_metadata_scoring.py packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_metadata_prompt_repair_pilot.py -q`
  passed with `29 passed`.
- `uv run ruff check packages/findingmodel-ai/evals/metadata_assignment.py packages/findingmodel-ai/evals/metadata_scoring.py packages/findingmodel-ai/tests/test_metadata_scoring.py`
  passed.
- `uv run mypy` passed.
- `task evals:metadata:smoke` passed and reported `GATES: PASS`, `METADATA QUALITY: 0.87`.
- Bounded `task evals:metadata` passed. The assignment portion reported `GATES: PASS`,
  `METADATA QUALITY: 0.80`, and exposed weak fields including `expected_time_course`,
  `etiologies`, `applicable_modalities`, and selected code/anatomy misses.

## Next Phase: Broader End-To-End Assignment Evals

- Treat `evals.metadata_assignment` as the real platform eval because it calls `assign_metadata()`
  end to end.
- Keep subagent-only evals as prompt diagnostics, not platform proof.
- [x] Add representative sampling so bounded evals are not just the first fixtures:
  - `--fixture-sample N`
  - `--seed`
  - sampling happens at fixture-stem level, then includes the selected fixtures' scenarios.
- [x] Rerun smoke assignment eval after scoring repair.
- [x] Rerun seeded bounded assignment eval after scoring repair.
- [ ] Run a broader/full assignment eval when practical and review the lowest cases.

## Next Phase: Prompt Review Based On End-To-End Results

- Review prompts only after the repaired end-to-end quality report exists.
- Start with `etiology_tempo`, because current diagnostic scores show it is weakest.
- For each miss, decide whether the issue is model behavior, prompt ambiguity, scoring defect, or
  gold fixture issue.
- Derive general rules only; do not add eval-case names or case-specific patches to prompts.

## Next Phase: Remaining Prompt Rationalization

- Rationalize remaining inline prompts one at a time in this order:
  1. `entity_type`
  2. `patient_applicability`
  3. `anatomy_decision`
  4. `ontology_decision`
- Move each prompt to a named prompt file, keep the output contract explicit, remove obsolete
  language, and validate with bounded end-to-end assignment evals.

## Next Phase: Finish Metadata Prompt Rationalization

- [ ] Externalize the remaining inline metadata prompts:
  - [x] `entity_type`
  - [x] `patient_applicability`
  - [x] `ontology_decision`
  - [x] `anatomy_decision`
- [x] Use prompt-master guidance:
  - small prompt that does the job;
  - explicit output contract;
  - general rules before examples;
  - no obsolete routing/worklist/workflow language;
  - no exact eval-case names as prompt examples.
- [x] Keep the prompt files as the source of truth and make every focused metadata agent load them
  through `load_metadata_prompt()`.
- [x] Expand prompt infrastructure tests to cover every focused metadata prompt and focused agent.
- [x] Add focused patient-applicability eval coverage.
- [x] Expand focused ontology eval coverage for general canonical-code selection failures.
- [x] Add the patient-applicability eval to bounded and full metadata eval tasks.
- [x] Run formatting, linting, type checking, focused prompt tests, focused component evals, and
  bounded metadata evals.
- [x] Update this plan and the changelog with the final state.

### Prompt Rationalization Verification

- All focused metadata agents now load prompt files from
  `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/`.
- Prompt infrastructure tests passed with `19 passed`.
- Targeted metadata tests passed with `41 passed`.
- `uv run ruff check` passed on touched metadata/eval/test files.
- `uv run mypy` passed.
- Focused component eval results:
  - entity type: `1.00`
  - ontology decision: `1.00`
  - anatomy decision: `1.00`
  - patient applicability: `0.84` to `0.92` across runs, with remaining misses isolated to
    age/default applicability decisions rather than execution/schema failures.
  - subspecialty domain: `1.00`
  - modality applicability: `1.00`
  - etiology tempo: `0.82`
- Bounded metadata assignment rerun reached `GATES: PASS` with metadata-quality reporting intact.
  A later aggregate `task evals:metadata` run stalled in the etiology/tempo child process and was
  terminated; rerunning `evals.metadata_etiology_tempo_decision` directly completed successfully.

## Next Phase: Etiology/Tempo Prompt-Master Cleanup

- [x] Refactor `etiology_tempo.md` for prompt-master hygiene:
  - preserve the current field contract;
  - keep the long-end-of-common-observable-time-course framing;
  - organize rules by decision priority;
  - keep general mappings, not eval-case examples;
  - keep the prompt longer than the other focused prompts only where the domain complexity requires it.
- [x] Verify prompt length and exact eval-case-name hygiene.
- [x] Run prompt infrastructure tests and the focused etiology/tempo eval.
- [x] Record the final result here.

### Etiology/Tempo Cleanup Verification

- `etiology_tempo.md` remains 64 lines, with clearer section order and no exact focused eval
  case-name hits.
- Prompt infrastructure test passed with `19 passed`.
- Ruff passed on the updated prompt test.
- Focused etiology/tempo eval completed with weighted overall `0.81`; this is essentially
  unchanged from the prior `0.82` and keeps the same known low-scoring cases visible for future
  gold review or prompt tuning.

## Next Phase: Etiology/Tempo Readability Repair

- [x] Rewrite `etiology_tempo.md` around readable general rules with specific examples.
- [x] Put the highest-impact rules first:
  - do not guess causes;
  - distinguish broad common cause buckets from differentials;
  - choose persistence on imaging, not symptom duration;
  - use the long common time course;
  - keep examples in service of general rules.
- [x] Run prompt hygiene tests and focused etiology/tempo eval.
- [x] Record the result here.

### Etiology/Tempo Readability Verification

- Rewrote the prompt around a plain-language goal and five priority rules, with examples attached
  to general rules rather than listed as missed cases.
- Prompt length is `66` lines.
- Prompt infrastructure tests passed with `22 passed`.
- Focused etiology/tempo eval completed with weighted overall `0.81`, recovering from an initial
  over-conservative rewrite and matching the prior cleaned-prompt band.

## Next Phase: Etiology/Tempo Naive-Agent Clarity

- [x] Rewrite the prompt opening so a fresh agent understands why the fields exist.
- [x] State that etiologies are used to find relationships between findings and group related
  findings.
- [x] Replace private shorthand such as "bucket" with plain language.
- [x] Run prompt hygiene tests and focused etiology/tempo eval.

### Etiology/Tempo Naive-Agent Verification

- Prompt opening now states that `etiologies` are cause labels for finding relationships/grouping
  and `expected_time_course` is for judging whether a prior finding should still be visible.
- Prompt length is `70` lines.
- Prompt infrastructure tests passed with `22 passed`.
- Focused etiology/tempo eval completed with weighted overall `0.81`.

## Next Phase: Etiology/Tempo Decision Standard

- [x] Improve Pydantic schema descriptions instead of adding a prompt-level JSON schema example.
- [x] Make the etiology decision standard explicit:
  - finding text states the process;
  - selected canonical ontology encodes the process;
  - radiology convention treats the process as inherent to the finding class.
- [x] Rework descriptive-finding examples so they read as general category guidance, not eval-case
  patches.
- [x] Run prompt tests and focused etiology/tempo eval.

### Etiology/Tempo Decision Standard Verification

- Updated `EtiologyTempoDecision` schema descriptions for `etiologies`, `expected_time_course`,
  and `field_confidence`.
- Prompt now states the etiology standard explicitly: use a label when the finding text states the
  process, selected canonical ontology encodes it, or radiology convention treats it as inherent.
- Prompt length is `70` lines.
- Prompt infrastructure tests passed with `22 passed`.
- Ruff passed on touched Python files.
- Focused etiology/tempo eval completed with weighted overall `0.81`.

## Next Phase: Etiology Rule Distillation And Larger Eval

- [x] Replace tautological etiology mappings with decision-focused rules that resolve likely
  mistakes.
- [x] Keep examples tied to general rules, not eval case names.
- [x] Run prompt tests and focused etiology/tempo eval.
- [x] Run a larger metadata assignment eval set to expose tuning targets.
- [x] Record the focused and larger-eval results here.

### Etiology Rule Distillation Verification

- Used prompt-master guidance to keep the prompt concise, explicit, and organized around general
  decision rules rather than case-name patches.
- `etiology_tempo.md` is `70` lines.
- Prompt infrastructure tests passed with `22 passed`.
- Focused etiology/tempo eval completed with weighted overall `0.81`. The score remained in the
  prior band, with persistent low cases concentrated around active inflammation, descriptive
  states, and some gold expectations that may need human review.
- Larger end-to-end assignment eval:
  `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_assignment --fixture-sample 10 --seed 20260517`
  completed with `GATES: PASS` and `METADATA QUALITY: 0.81`.
- Larger eval per-field averages:
  - entity type `0.97`
  - body regions `0.98`
  - subspecialties `0.93`
  - applicable modalities `0.83`
  - etiologies `0.74`
  - expected time course `0.50`
  - index codes `0.83`
  - anatomic locations `0.68`
  - age profile `0.26`
  - sex specificity `1.00`
- Lowest larger-eval cases were mainly rotator cuff tear and lumbar disc herniation scenarios,
  driven by expected time course, index-code/anatomic-code selection, and age profile misses.
