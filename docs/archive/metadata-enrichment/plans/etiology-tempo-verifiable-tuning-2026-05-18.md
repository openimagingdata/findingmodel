# Etiology/Time-Course Verifiable Tuning Plan

> Superseded for active execution by `docs/plans/metadata-enrichment-current-plan.md`. Keep this file as historical evidence only; pull any still-useful decisions into the active plan or a stable reference doc before acting on them.



## Summary

Stop tuning the etiology/time-course prompt against the current 39-case hand-curated component eval.
Build a larger, versioned eval corpus from reviewed data, run the current prompt as a baseline,
triage misses by concrete failure type, then make prompt changes only when they improve declared
failure classes without hiding regressions.

## Historical Implementation Checklist

- [x] Create a checked-in etiology/time-course fixture from all package gold fixtures plus the 73
  clean-input reviewed cases from the sibling data repo.
- [x] Dedupe by normalized fixture/file slug, with package gold winning over sibling review data.
- [x] Keep reviewed expected values unchanged and flag questionable values for later adjudication.
- [x] Rename eval docs/output language from "focused eval" to "component eval".
- [x] Add etiology/time-course corpus filters: `pilot`, `gold`, `reviewed`, `expanded`, and `all`.
- [x] Add separate score outputs for etiologies, time-course, duration, modifier, and combined score.
- [x] Add a details output option with expected/actual values and deterministic miss labels.
- [x] Run prompt hygiene tests, component evals, lint/type checks, and sampled end-to-end assignment.
- [x] Record baseline results and remaining adjudication targets here.

## Acceptance Rules

- Prompt edits must state general rules, not case-specific exceptions.
- Eval scoring should penalize mistakes of commission more than mistakes of omission: unsupported
  extra labels create false grouping relationships, while missing labels are usually easier to
  recover through review or later enrichment.
- A prompt edit is accepted only if it improves a declared miss group and causes no meaningful
  regression:
  - no gate failures;
  - no new zero-score cases outside the targeted miss group;
  - no drop greater than `0.02` in either field-level average;
  - target miss group improves by at least 3 cases or by at least `0.10` average score.

## Current Decisions

- First expanded target: all 35 package gold fixtures plus the 73 structured reviewed clean-input
  cases.
- Reviewed values stay authoritative for scoring until humans adjudicate them.
- The generated fixture lives in this repo so evals do not depend on the sibling data repo at run
  time.
- The prompt should stay concise and rule-based; expanded evidence belongs in eval fixtures and
  review reports, not in the prompt.

## Baseline Results

- Generated fixture:
  `packages/findingmodel-ai/evals/fixtures/etiology_tempo_reviewed_cases.json`.
- Fixture counts:
  - package gold: `35`;
  - reviewed clean-input records seen: `73`;
  - reviewed records skipped because package gold wins: `2`;
  - reviewed records with expected etiology/time-course values corrected from the comparison
    artifact: `40`;
  - total expanded cases: `106`.
- Pilot component run:
  `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set pilot --details-output /tmp/etiology-tempo-pilot-details.csv`
  - combined `0.84`;
  - etiologies `0.85`;
  - expected time course `0.82`;
  - duration `0.85`;
  - modifiers `0.70`.
- Expanded component run:
  `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set expanded --details-output /tmp/etiology-tempo-expanded-details-v2.csv`
  - combined `0.74`;
  - etiologies `0.73`;
  - expected time course `0.75`;
  - duration `0.78`;
  - modifiers `0.66`;
  - package gold combined average `0.79`;
  - reviewed-clean-input combined average `0.71`.
- Expanded miss-label counts:
  - extra unsupported etiology: `59`;
  - wrong duration: `45`;
  - wrong modifier: `30`;
  - missing expected etiology: `17`;
  - wrong etiology family/subtype: `12`;
  - missing time course: `7`;
  - extra time course: `5`;
  - exact/perfect: `19`.

## Immediate Tuning Targets

- Primary prompt target: over-assignment of etiologies for broad/descriptive findings.
- Secondary prompt target: time-course modifier choice, especially stable/evolving/progressive.
- Adjudication candidates before prompt changes: cases with expected null etiology/time course but
  clinically plausible broad labels, and gold cases where our current policy may conflict with older
  fixture values.

## Prompt Review Updates

- Independent prompt review supported the revised section structure:
  etiology guidelines, etiology heuristics, time-course guidelines, and time-course heuristics.
- Applied the first four review fixes:
  - broad descriptive findings only support grouping labels when the name or description clearly
    implies them;
  - unspecified tumor/neoplasm can carry both benign and malignant labels, but explicitly benign,
    malignant, or metastatic diagnoses should use only the stated category;
  - metastatic disease should use `neoplastic:metastatic` without also adding
    `neoplastic:malignant` unless a separate primary malignancy process is represented;
  - generic lesions should not receive a years-long time course merely because they could persist.
- Did not duplicate the full output schema in the prompt. The etiology/time-course agent is created
  with `output_type=EtiologyTempoDecision`, so the Pydantic schema and field descriptions are
  supplied through the structured-output call path.
- Removed the hemangioma/vascular-malformation rule without replacement. It was narrow,
  potentially overfit, and the non-overfit version collapsed into a tautology.
- Updated etiology/time-course scoring to prefer conservative outputs:
  unsupported extra etiologies and modifiers are penalized more heavily than omitted expected
  values, and extra time course on an expected-null case is worse than omitting an expected time
  course.
- Honed-prompt runs under the commission-sensitive scorer:
  - expanded corpus:
    `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set expanded --details-output /tmp/etiology-tempo-expanded-details-honed-commission-sensitive.csv`
    produced combined `0.70`, etiologies `0.65`, expected time course `0.76`, duration `0.76`,
    modifiers `0.61`;
  - pilot corpus:
    `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set pilot --details-output /tmp/etiology-tempo-pilot-details-honed-commission-sensitive.csv`
    produced combined `0.83`, etiologies `0.84`, expected time course `0.83`, duration `0.84`,
    modifiers `0.70`.
- Added a first prompt-fix pass for the highest-confidence miss classes:
  - appearance, distribution, signal, density, size, and enhancement patterns should not be turned
    into differential etiologies;
  - artifacts, measurements, indices, classifications, assessment scales, recommendations, and
    technique-only concepts should not carry etiologies or expected time course.
- First prompt-fix pass results:
  - expanded corpus:
    `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set expanded --details-output /tmp/etiology-tempo-expanded-details-prompt-fix1.csv`
    produced combined `0.73`, etiologies `0.70`, expected time course `0.78`, duration `0.78`,
    modifiers `0.60`;
  - expanded miss-label deltas versus the honed commission-sensitive run:
    extra unsupported etiology `53 -> 43`, extra time course `5 -> 2`, perfect cases `21 -> 24`;
  - pilot corpus:
    `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set pilot --details-output /tmp/etiology-tempo-pilot-details-prompt-fix1.csv`
    produced combined `0.83`, etiologies `0.85`, expected time course `0.81`, duration `0.84`,
    modifiers `0.60`.
- Human adjudication pass applied:
  - generic pleural/pericardial effusion and transudative pleural effusion now expect null etiology
    and null time course;
  - FDG-avid pulmonary nodule now expects `inflammatory` and `neoplastic:malignant`;
  - hepatic hemangioma now expects `neoplastic:benign`;
  - aortic dissection expects parent `vascular`;
  - cardiomegaly and neonatal heart failure expect `cardiac`;
  - prolonged cerebral vascular transit time remains parent `vascular`;
  - cysts remain null etiology, pediatric age context alone does not imply congenital/developmental,
    and null etiology may still coexist with non-null time course for persistent finding classes.
- Added matching prompt clarifications for cardiac/vascular mechanism labels, lymphadenopathy,
  cysts/effusions, pediatric age context, and null-etiology/non-null-time-course cases.
- After the first adjudicated run, tightened prompt wording for remaining commission errors:
  descriptive fluid findings should not infer etiologies or time course from location/character
  alone, age words including fetal/newborn/infant/child/pediatric do not imply
  congenital/developmental labels, and FDG avidity is not `neoplastic:potential`.
- Second adjudicated prompt run:
  - expanded corpus:
    `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set expanded --details-output /tmp/etiology-tempo-expanded-details-adjudicated-prompt2.csv`
    produced combined `0.74`, etiologies `0.72`, expected time course `0.78`, duration `0.79`,
    modifiers `0.62`;
  - pilot corpus after aligning pilot cases with adjudicated policies:
    `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set pilot --details-output /tmp/etiology-tempo-pilot-details-adjudicated-prompt2-pilotfix.csv`
    produced combined `0.84`, etiologies `0.82`, expected time course `0.87`, duration `0.91`,
    modifiers `0.70`.
- Sampled end-to-end assignment after adjudication:
  `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_assignment --fixture-sample 10 --seed 20260517`
  completed with `GATES: PASS` and `METADATA QUALITY: 0.81`.
  Field averages included etiologies `0.68` and expected time course `0.59`; remaining low cases
  are mostly index-code/anatomic-location/age-profile misses, with one FDG-avid pulmonary nodule
  etiology miss still visible in the sample.

## Verification

- Prompt/eval infrastructure tests:
  `PYTHONPATH=packages/findingmodel-ai uv run --package findingmodel-ai pytest packages/findingmodel-ai/tests/test_metadata_prompt_repair_pilot.py packages/findingmodel-ai/tests/test_metadata_etiology_tempo_eval.py -q`
  passed with `24 passed`.
- Ruff:
  `uv run ruff check packages/findingmodel-ai/evals/metadata_etiology_tempo_decision.py packages/findingmodel-ai/tests/test_metadata_etiology_tempo_eval.py`
  passed.
- Type checking: `uv run mypy` passed.
- Sampled end-to-end assignment:
  `PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_assignment --fixture-sample 10 --seed 20260517`
  completed with `GATES: PASS` and `METADATA QUALITY: 0.80`.
- Sampled end-to-end field averages:
  - entity type `0.97`;
  - body regions `0.98`;
  - subspecialties `0.90`;
  - applicable modalities `0.81`;
  - etiologies `0.77`;
  - expected time course `0.55`;
  - index codes `0.83`;
  - anatomic locations `0.57`;
  - age profile `0.23`;
  - sex specificity `1.00`.
