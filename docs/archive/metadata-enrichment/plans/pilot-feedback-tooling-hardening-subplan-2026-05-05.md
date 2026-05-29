# Metadata Enrichment Tool Hardening Plan

> Superseded for active execution by `docs/plans/metadata-enrichment-current-plan.md`. Keep this file as historical evidence only; pull any still-useful decisions into the active plan or a stable reference doc before acting on them.



Date: 2026-05-05
Status: Integrated revision with prompt-suite assessment incorporated

## Purpose

We ran a 150-record pilot metadata enrichment pass and had it reviewed by a human domain expert. The
goal now is not merely to polish those 150 records. The goal is to use that expensive review to make
the automated enrichment tool less likely to repeat the same mistakes on a larger corpus.

The larger corpus run remains blocked until we have evidence that the tool has absorbed the major
review lessons or can reliably flag/defer cases it should not decide automatically.

## Start Here For Implementation

This document is the task-level plan for
`Phase 6: Improve the Enrichment Tool Using the 150 Reviewed Examples` in the umbrella plan.

The immediate implementation task is **not** another prompt rewrite and **not** a larger corpus run.
Start by building the grading-aware comparison script in the primary metadata repo:

```text
/Users/talkasab/repos/findingmodels-metadata/scripts/metadata_compare_clean_rerun.py
```

Then use that script to triage the clean-input rerun differences before making more prompt/tool
changes.

Important constraints for the implementer:

- Do not start enrichment on the broader corpus.
- Do not commit without explicit permission.
- Do not revert unrelated dirty working-tree changes.
- Treat `.metadata-runs/` files as local run evidence, not publishable source artifacts.
- Treat the 150 pilot-enriched `defs/` and `text/` files in the primary metadata repo as review and
  iteration working state, not final publishable content.
- Use plain progress language: state what was changed, what was verified, and what remains blocked.
  Avoid internal shorthand when reporting status.

## Current State

Supporting package repo:

- `/Users/talkasab/repos/findingmodel-metadata`
- Branch: `feature/metadata-cleanup`
- Owns reusable enrichment code, prompt, anatomy search, ontology handling, validators, auditors,
  and tests.

Primary metadata repo:

- `/Users/talkasab/repos/findingmodels-metadata`
- Branch: `findingmodels-metadata`
- Owns canonical `defs/*.fm.json`, generated `text/*.md`, review scripts, and run artifacts.

Current facts:

- Human review of the 150-record pilot is complete.
- Pilot source corrections are applied or deferred with rationale.
- A coverage matrix exists to separate source corrections from actual tooling evidence.
- A 73-record clean-input rerun was created from preserved pre-correction inputs.
- That rerun completed mechanically, but exact comparison against reviewed source fields found 76
  reviewed-field differences across 58 of 73 records.
- A separate current-prompt reference packet exists at
  `/tmp/metadata-enrichment-current-prompts-reference-2026-05-05.md`.
- External prompt-suite review identified the assignment prompt as too long and internally
  contradictory on time course, and identified several rules that should move out of prompt text
  into deterministic checks, candidate filtering, validators, or evals.
- Official OpenAI documentation was checked for `gpt-5.4-mini` and GPT-5.4 prompt guidance:
  - `gpt-5.4-mini` is documented as a faster, lower-cost GPT-5.4 variant for high-volume workloads
    and has a pinned snapshot `gpt-5.4-mini-2026-03-17`.
  - The current models page recommends `gpt-5.4-mini` or `gpt-5.4-nano` when optimizing for
    latency and cost.
  - GPT-5.4 guidance says smaller models are highly steerable but more literal; prompts should put
    critical rules first, use numbered decision rules, define ambiguity behavior, and specify the
    output package directly.

The important conclusion:

> The tool is more stable than before, but mechanical success is not correctness. We need a better
> comparison and triage loop before further prompt/tool iteration.

## Core Principle

Reviewed cases should become regression examples, not prompt exceptions.

Bad prompt pattern:

- "Newborn duodenal obstruction should include ultrasound."
- "Upper cervical spine classification should not include atlas and axis."
- "Osseous lucent lesion should not include MR."

Better pattern:

- Extract a reusable rule.
- Put the specific case in a test/eval.
- Use a clean-input rerun to verify the rule actually improves behavior.

Examples:

- Pediatric GI obstruction: routine modality choices depend on neonatal/pediatric context; use the
  case as a regression test, not as a hard-coded prompt fact.
- Modality-specific language: terms like `echogenic`, `hypodense`, `T2 hyperintense`, `FDG-avid`,
  and `lucent` imply modality constraints. This belongs mostly in deterministic checks or auditor
  logic, not a one-off prompt sentence about lucency.
- Spine classifications: select anatomy at the scope being classified. If both a parent segment and
  component parts are returned, candidate filtering or deterministic post-checks may be better than
  asking the LLM to ignore tempting candidates.

## Revised Approach

The next step is not more prompt editing. The next step is to make the comparison itself meaningful.

Exact set equality is too crude for several metadata fields. Some differences are true errors; some
are adjacent or defensible; some reflect missing terminology or ontology evidence; some may reveal
that the reviewed source correction itself needs reconsideration.

The workflow should be:

1. Build a grading-aware comparison script.
2. Triage the current clean-input mismatches.
3. Convert the prompt-suite assessment into a scoped implementation checklist.
4. Fix issues at the lowest reliable layer.
5. Rerun targeted subsets plus a regression-floor subset.
6. Update the coverage matrix only when tool evidence exists.

## Step 1: Build A Grading-Aware Comparison Script

Add a tracked primary-repo script:

```text
scripts/metadata_compare_clean_rerun.py
```

Inputs:

- coverage matrix;
- clean-input rerun directory;
- reviewed source `defs/*.fm.json`;
- metadata-review JSON from the rerun;
- deterministic audit outputs;
- optionally the anatomic index and ontology cache.

Outputs:

- machine-readable JSON summary;
- concise Markdown summary for tracked docs;
- mismatch table suitable for triage.

The script should report:

- records compared;
- batch status counts;
- assignment warnings;
- deterministic audit flags;
- mismatch counts by field, item, and theme;
- confidence level for each mismatched field;
- whether a mismatch is exact, adjacent, missing, extra, hierarchical, unrelated, or needs review.

Field grading:

- `expected_time_course`: grade exact, adjacent-shorter, adjacent-longer, distant,
  missing-expected, extra-unexpected, modifier-difference.
- `anatomic_locations`: use the anatomy hierarchy where available; grade exact, parent,
  child, sibling/nearby, unrelated, extra-broad, extra-narrow, missing-reviewed.
- `index_codes`: do not rely on string equality alone; grade same code, likely equivalent,
  broader, narrower, related-only, modality-specific overreach, unsupported extra, or
  needs-ontology-review.
- `applicable_modalities`: distinguish missing routine modality from extra nonroutine modality and
  descriptor-modality conflict.
- `etiologies`: distinguish missing core etiology, extra speculative etiology, adjacent broad
  etiology, wrong mechanism, and non-disease-entity etiology.
- `age_profile`, `sex_specificity`, `body_regions`, and `subspecialties`: mostly exact, but still
  report missing versus extra separately.

This script is the immediate next implementation task.

## Step 2: Triage The Current Mismatches

After the comparison script exists, triage the 76 reviewed-field differences from the clean-input
rerun.

Each mismatch should receive one disposition:

- `tool error`
- `judgment call - reviewer preferred`
- `judgment call - tool defensible`
- `reviewed source needs reconsideration`
- `terminology/source-data blocked`
- `comparison too strict`

Only these should drive tool changes:

- `tool error`
- high-confidence `judgment call - reviewer preferred` where the preference generalizes beyond the
  one reviewed item.

Do not tune the prompt to:

- defensible alternatives;
- reviewer preference that does not generalize;
- terminology gaps;
- comparison artifacts.

## Step 3: Fix Issues At The Right Layer

Do not put every lesson into the assignment prompt. Use the lowest reliable layer.

### Prompt

Use prompt text for clinical judgment rules that require interpretation:

- broad age and sex defaults;
- observable imaging time course;
- distinguishing intrinsic etiology from a broad differential;
- routine modality selection when deterministic modality-language cues are insufficient;
- choosing scope when candidates are not structurally comparable.

Prompt text should be compact and general.

The prompt-suite review and OpenAI's GPT-5.4 guidance add a concrete refactor target:

- Restructure the metadata assignment instructions as identity, mode, decision principles, compact
  field rules, and output discipline.
- Remove the synthetic examples, partial-field snippets, and index-code teaching snippets from the
  live prompt; convert representative cases into evals/regression examples.
- Replace the current time-course rule that says to choose the upper bound. The replacement rule is:
  choose the duration that matches the typical observable imaging window of the finding itself.
- Put `field_confidence` guidance in one dedicated location in the assignment prompt; leave the
  detailed enforcement to validators.
- Compact the subspecialty section to principles and common co-occurrences. Do not keep long lists
  of single-case patches in the live prompt.
- In the ontology prompts, remove hardcoded ontology-system preference from model instructions.
  The target behavior is to include relevant hits from every searched ontology, then rank or
  deduplicate by relevance and evidence rather than by ontology brand. Remove routine all-caps
  emphasis.
- In the anatomy selection prompt, replace "sweet spot" wording with explicit scope rules:
  specific named structure, organ/region for regional findings, and declared scope for
  classification/score/assessment models.
- In the auditor prompt, enumerate the pilot-derived patterns it should catch after deterministic
  checks, but keep deterministic rules implemented deterministically.

For `gpt-5.4-mini`, do not over-correct by making the prompt terse. "Compact" means removing noisy
case patches and duplicated guidance while keeping explicit numbered decision rules, clear ambiguity
behavior, and a precise output contract. Prompt edits should be implemented only after the comparison
script can show whether outputs improved, regressed, or merely changed.

### Structured Validators

Use validators for outputs that are invalid or semantically inconsistent:

- invalid confidence keys;
- missing confidence for changed fields;
- unknown candidate IDs;
- required fields left empty;
- measurement, assessment, recommendation, or technique issue carrying etiologies;
- measurement, score, classification, assessment, recommendation, or technique issue carrying
  intrinsic time course.

### Candidate Generation And Filtering

Use candidate filtering when the candidate set tempts the LLM into predictable mistakes:

- classification/score/assessment models returning both a scope-level anatomy and component parts;
- spine classifications returning segment anatomy plus atlas/axis/neck;
- broad findings getting overly localized candidates from incidental description text;
- missing anatomy terms causing bad fallbacks.

Filtering should be gated and conservative. Do not globally suppress useful candidates.

### Deterministic Audit

Use deterministic checks for reliable rules:

- modality-language conflicts:
  - `echogenic` -> US;
  - `hypodense`, `hyperdense`, `attenuating` -> CT;
  - `T1/T2 hyperintense` -> MR;
  - `FDG-avid` -> PET;
  - `lucent`, `lucency`, `radiolucent` -> XR/CT, with MR overreach flagged;
- PET without molecular imaging subspecialty;
- molecular imaging subspecialty without PET/NM;
- anatomy/body-region incompatibility;
- sex-specific anatomy conflicts;
- non-disease entity with etiology;
- selected parent and child anatomy when model scope implies only one should be canonical;
- canonical index-code display text that conflicts with ontology cache evidence.

Implementation status:

- Descriptor-modality deterministic audit checks are implemented for echogenic/hypoechoic,
  hypodense/hyperdense/attenuation, T1/T2 hyperintense, FDG, and lucency/radiolucency language.
- Lucency/radiolucency now flags MR overreach and missing XR/CT support.
- Focused auditor tests cover lucency MR overreach and missing US for hypoechoic descriptor text.
- Phase 6 targeted rerun evidence in the primary metadata repo confirms the descriptor checks on
  clean inputs: `osseous_lucent_lesion` produced one MR-overreach audit flag, while
  `hypoechoic_liver_lesion` assigned `US` and produced no audit flag.

### Ontology Decision Logic

Ontology exactness should not live only in broad prompt wording.

The decision step should enforce or flag:

- canonical codes must be exact or clinically substitutable;
- broader, narrower, related, procedure/history, exam, and modality-specific concepts should not be
  canonical unless the model scope truly matches;
- related concepts should remain review evidence;
- source codes should be preserved when they are more exact than generated candidates.
- there should be no ontology-system preference in the prompt. Relevant hits from every searched
  ontology should remain available; ranking should be by conceptual relevance and evidence quality.

If cross-ontology equivalence is not computable, mark it as `needs-ontology-review` rather than
forcing exact mismatch.

## Step 4: Define A Regression-Floor Subset

Every targeted rerun must include a fixed set of records that were already acceptable. This prevents
prompt drift.

Store the selected regression-floor records and their reviewed canonical outputs in version control:

```text
evals/regression_floor/
```

Pick records by category, not convenience. Each selected record should name the field or behavior it
guards and should include the reviewed expected output used by the comparison script.

The regression-floor subset should cover:

- one time-course case;
- one anatomy case;
- one measurement/assessment case;
- one device/tube/line case;
- one broad whole-body or soft-tissue case;
- one PET/molecular imaging case;
- one ontology exactness case;
- one pediatric case;
- one sex-specific anatomy case;
- one age-neutral/all-ages case.

A targeted fix is not acceptable if it creates new mismatches, warnings, or deterministic audit flags
in the regression-floor subset.

## Step 5: Pin The Model During Iteration

Rerun deltas are not meaningful if the assignment model changes underneath the prompt.

Use this target configuration for the next prompt/tool iteration unless a later repo or API
constraint prevents it:

- model: `gpt-5.4-mini`
- snapshot, if supported by current configuration: `gpt-5.4-mini-2026-03-17`
- reasoning setting: `none` for the first controlled comparison, matching the decision not to use
  o-series/reasoning-mode prompting assumptions
- API shape: keep the existing Pydantic AI integration unless the repo needs a separate migration

Implementation status:

- `metadata_assign` is now pinned in `supported_models.toml` to
  `openai:gpt-5.4-mini-2026-03-17`.
- `metadata_assign` now uses `reasoning = "none"` for the primary OpenAI model.
- The primary metadata repo has a seed regression floor under `evals/regression_floor/`.

Each rerun summary should record:

- assignment model id;
- support package wheel source;
- ontology cache path;
- audit mode;
- run directory;
- comparison script version or commit.

If model pinning is not currently possible through configuration, fix that workflow issue before
using mismatch deltas as evidence. If the assignment model cannot be pinned, defer prompt iteration;
mismatch deltas without a pinned model cannot distinguish prompt effects from model drift.

## Step 6: Iterate By Failure Class

After grading and triage, address failure classes in this order:

1. Comparison infrastructure and triage.
2. Regression-floor eval set and model pinning.
3. Deterministic validators and checks that should not be prompt rules:
   - modality-language conflicts;
   - parent/child anatomy selection;
   - non-disease entity constraints;
   - field-confidence validity;
   - ontology cache display conflicts.
4. Metadata assignment prompt compaction:
   - shorter structure;
   - one field-confidence section;
   - keep only 3-4 examples that illustrate otherwise tricky distinctions known to need
     reinforcement; move all other examples into evals;
   - compact subspecialty principles;
   - corrected time-course rule.
5. Anatomy selection prompt plus candidate filtering:
   - explicit scope-by-model-kind rules;
   - classification/score/assessment handling;
   - no parent plus contained child unless truly distinct.
6. Ontology query/categorization cleanup:
   - remove hardcoded SNOMEDCT preference from prompt;
   - preserve relevant hits from all searched ontology systems;
   - reduce all-caps emphasis;
   - reduce candidate volume where practical;
   - keep canonical code rule as exact or clinically substitutable.
7. Auditor prompt update:
   - enumerate pilot-derived patterns that may survive deterministic checks;
   - require concrete offending value plus contradicting evidence.
8. Targeted rerun plus regression-floor comparison.

Time course has the largest raw mismatch count, but raw exact-match counts are not reliable enough
to justify editing it first in isolation. The known contradictory "upper bound" instruction should
be removed during the prompt-compaction change set, then evaluated with graded comparison and
regression-floor cases.

## Time-Course Rule, Restated

Do not use "always choose the upper bound" as the rule.

Use this instead:

> Choose the duration that matches how long the imaging finding typically remains observable. This
> may differ from the duration of the underlying clinical process. When evidence gives a range,
> choose the value that best represents the expected observable imaging window, not automatically
> the longest possible value.

Working guidance:

- `weeks`: acute inflammatory/infectious/traumatic findings expected to evolve or resolve.
- `months`: subacute findings, temporary devices, healing changes, slower inflammatory findings.
- `years`: chronic lesions, nodules, neoplasms, masses, and long-term findings that persist but are
  not inherently permanent.
- `permanent`: fixed congenital/developmental anomaly, structural absence, calcification, retained
  surgical change, healed deformity.
- Devices, tubes, lines, and catheters: choose by expected dwell, not by whether the hardware itself
  is physically durable.
- Measurements, scores, classifications, assessments, recommendations, and technique issues have
  null time course.

Specific reviewed cases should test these rules; they should not be copied into the prompt as
exceptions.

The assignment prompt should use this wording or an equivalent compact form:

```text
- `expected_time_course` reflects how long the imaging finding itself remains observable, not the
  duration of the underlying clinical process.
- When evidence offers a range, choose the value that best matches the typical observable imaging
  window. Do not default to the upper bound.
- Fixed structural change, calcification, healed deformity, congenital absence, and retained
  surgical material are often permanent. Chronic lesions, nodules, masses, and neoplasms are often
  years rather than permanent unless the imaging trace is fixed.
- Measurements, scores, classifications, assessments, recommendations, and technique issues have
  null time course.
```

## Coverage Matrix Updates

Rows should move out of `source corrected; needs clean-input tool evidence` only when:

- the clean-input rerun reproduces the reviewed correction;
- the tool flags the issue deterministically for review;
- the mismatch is triaged as defensible and documented;
- the item is deferred with rationale.

Do not mark a row covered merely because the corrected source file validates.

## Prompt-Specific Implementation Notes

The prompt-suite assessment should be treated as a scoped refactor, not an invitation to keep adding
rules.

Official OpenAI prompt guidance changes the interpretation of "shorten the prompt":

- Use clear markdown sectioning and numbered rules.
- Put the most important rules early.
- Keep examples concise and easy to scan.
- State what to do when evidence is missing or ambiguous.
- Keep outputs compact and structured with an explicit output contract.
- For this project, do not remove clinically necessary specificity merely to reduce line count.

### Assignment Prompt

Target structure:

1. Identity and task boundary.
2. Assignment mode contract.
3. Non-negotiable decision principles and ambiguity behavior.
4. Compact field rules, one field at a time.
5. Candidate selection rules.
6. One field-confidence section.
7. Output contract and failure/warning behavior.

Move out of the prompt:

- synthetic contrast examples;
- partial-field teaching snippets;
- index-code teaching snippets;
- single-case subspecialty/body-region patches.

Keep 3-4 examples only when they illustrate an otherwise tricky distinction that the review showed
needs reinforcement. All other cases should become eval records or focused tests.

Implementation status:

- The contradictory "choose the upper expected visible duration" time-course instruction has been
  replaced with the observable-imaging-window rule.
- The live assignment prompt now explicitly says not to default to the upper bound.
- Chronic lesions, nodules, neoplasms, masses, and soft-tissue tumors are now framed as usually
  years rather than permanent unless the imaging trace is fixed.

### Subspecialty Prompt Guidance

Use the radiology reading-workflow rule:

> Include a subspecialty when a radiologist of that subspecialty would routinely read or report this
> finding. Organ membership alone is not enough.

Keep only common co-occurrence guidance:

- cardiac/pericardial findings: `CA`, adding `CH` only when thoracic interpretation is core;
- pulmonary, pleural, mediastinal, rib, and chest-wall findings: `CH`;
- vessel-centered findings: `VA`;
- gynecologic/adnexal/obstetric findings: `OB` and `GU` may co-occur;
- oncologic staging/surveillance/nodal findings: `OI`;
- PET/FDG-routine findings: `MI`; planar/SPECT-centered findings: `NM`;
- pediatric-specific entities: `PD` plus organ-system code;
- acute/urgent presentations: `ER` plus organ-system code.

Keep code-name reminders that prevent invalid output: `VA`, never `VI`; no `AB`.

### Anatomy Selection Prompt Guidance

Add the clarified parent-covers-parts rule without hard-coded metadata assignment tables:

- if multiple candidate locations are parts of one parent location that fully covers the finding,
  choose the parent location rather than the separate parts;
- do not return both a parent and its contained parts when the parent is the canonical scope;
- keep the existing specificity guidance otherwise, because `tunneled_catheter` should use
  `anterior chest wall`, not broad `thorax`;
- implement this through candidate generation/evidence and model selection, not finding-name or
  code-pair remapping.

Implementation status:

- The anatomic selection prompt keeps the existing specificity guidance.
- Special-case assignment remapping for `radiolucent_urinary_calculus`/`urinary tract` and
  `tunneled_catheter`/`anterior chest wall` was removed as an unacceptable implementation pattern.
- An attempted `anatomic_selection_guidance` addition to the assignment decision payload was removed
  after the v2 targeted run showed broader regression-floor drift. The current change is scoped to
  the upstream anatomic selector prompt.
- Focused assignment/anatomic tests and Ruff checks pass for the touched support files.

### Auditor Prompt Guidance

The auditor should name pilot-derived patterns that may survive deterministic checks:

- broader, narrower, procedure-coded, exam-coded, or modality-specific canonical `index_codes`;
- index-code display text conflicting with ontology cache evidence;
- parent plus child anatomy where the model does not span both distinctly;
- modality-language conflicts not caught deterministically;
- PET/MI/NM pairing problems;
- time course or etiology on non-disease entities;
- unsupported broad etiology lists;
- invalid or missing `field_confidence` for changed anatomy or ontology fields.

Each auditor flag must include the offending value and the contradicting evidence.

## Commit Strategy

Do not commit without explicit permission.

Recommended sequence:

1. Support package checkpoint:
   - current confidence validation, deterministic audit support, anatomy improvements, validators,
     prompt hardening, and tests.
2. Primary workflow/docs checkpoint:
   - coverage matrix generator;
   - clean-input comparison script once added;
   - clean-input rerun analysis;
   - primary script changes.
   - Exclude the 150 pilot `defs/` and `text/` corrections unless explicitly approved.
3. Comparison/triage commit:
   - grading-aware comparison output and mismatch triage docs.
4. Targeted hardening commits by class:
   - anatomy;
   - ontology;
   - modality;
   - time course;
   - etiology/subspecialty.
5. Pilot metadata correction commit:
   - only after readiness is documented.

## Decisions Needed Before Encoding Certain Rules

These are actual domain/policy decisions. Resolved decisions are recorded here so implementation
does not re-open them accidentally.

1. Are reviewed source corrections preferred targets or absolute truth?
   - Proposed default: preferred targets, with documented defensible alternatives allowed.
2. What time-course differences count as adjacent rather than wrong?
   - Proposed default: weeks/months, months/years, and years/permanent are adjacent labels, but
     still require category-specific triage.
3. Should measurement/classification/assessment entities always have null time course?
   - Resolved: yes. Measurements, classifications, scores, assessments, recommendations, and
     technique issues have null `expected_time_course`.
4. What counts as cross-ontology equivalence for index codes?
   - Proposed default: only explicit evidence or mappings; otherwise mark as
     `needs-ontology-review`.
5. How should transudative pleural effusion etiologies be handled?
   - Proposed default: do not encode from the current mismatch until clinically reconsidered.
6. Which assignment model id is pinned for this iteration cycle?
   - Resolved target after official-doc check: use `gpt-5.4-mini`, preferably pinned to
     `gpt-5.4-mini-2026-03-17` if supported by current configuration. Use `reasoning=none` for the
     first controlled comparison. Do not use o-series/reasoning-mode prompting assumptions.
7. Where should `field_confidence` guidance live?
   - Resolved: one dedicated section in the assignment prompt; validators enforce the detailed
     requirements.
8. How many examples should remain in the assignment prompt?
   - Resolved: keep 3-4 examples only if they illustrate tricky distinctions that need
     reinforcement; move the rest to evals.
9. Should ontology hits prefer one ontology system?
   - Resolved: no. Include relevant hits from every searched ontology system. The prompt should not
     prefer SNOMEDCT or any other ontology by name.

## Official OpenAI Guidance Incorporated

Sources checked on 2026-05-05:

- `https://developers.openai.com/api/docs/models/gpt-5.4-mini`
- `https://developers.openai.com/api/docs/models`
- `https://developers.openai.com/api/docs/guides/prompt-guidance`
- `https://developers.openai.com/api/docs/guides/prompting`

Plan implications:

- `gpt-5.4-mini` is an appropriate target for high-volume work when lower latency/cost matters, but
  the model page also exposes a snapshot. Prompt iteration should pin the snapshot when possible.
- GPT-5.4-class models support reasoning settings, including `none`; this plan starts with `none`
  because the project decision is not to use o-series/reasoning-mode prompting assumptions.
- OpenAI notes that `none` can work well for action-selection and tool-discipline tasks, while
  `low` or `medium` can help when nuanced interpretation or ambiguity dominates. Do not vary this
  setting during the first controlled prompt comparison; if `none` underperforms after prompt and
  validator fixes, test `low` as a separate, pinned eval condition.
- The prompt refactor should not simply make prompts shorter. For `gpt-5.4-mini`, use explicit
  numbered rules, ambiguity instructions, and structured output contracts.
- OpenAI's prompt docs recommend concise examples in a scannable block and rerunning linked evals
  whenever prompts change. That directly supports the 3-4-example cap plus regression-floor evals.

## Readiness Gate

Before a larger corpus run:

- grading-aware comparison exists and is repeatable;
- current mismatches are triaged;
- major tool-error classes are addressed at the right layer;
- targeted reruns plus regression-floor checks pass;
- remaining mismatches are documented as defensible, deferred, terminology-blocked, or requiring
  human/domain decision;
- coverage matrix reflects tool evidence, not just source edits.

Until then, the larger corpus run remains blocked.

Current targeted-rerun status:

- `.metadata-runs/phase6-targeted-v1/run` completed 12 dry-run assignments with no batch failures.
- After correcting the regression-floor anatomy targets for `tunneled_catheter` and
  `radiolucent_urinary_calculus`, the tracked regression-floor comparison script in the primary
  metadata repo compared 10 floor records and 95 fields.
- Before deterministic normalization, the corrected floor had 1 strict mismatch:
  `radiolucent_urinary_calculus` returned contained urinary sites alongside the desired parent
  `urinary tract`.
- No floor record produced deterministic audit flags.
- The next engineering-local failure class is anatomy scope/candidate filtering; the larger corpus
  run remains blocked until that class is addressed or explicitly deferred.

After anatomy-scope prompt hardening:

- `.metadata-runs/phase6-targeted-v2/run` completed 12 dry-run assignments with no batch failures.
- The floor comparison found 5 strict mismatches, up from 2 in v1.
- `tunneled_catheter` no longer mismatched on anatomy, but `radiolucent_urinary_calculus` still
  mismatched on anatomy and new strict mismatches appeared in `breast_calcification_cluster`,
  `aortic_measurements`, and `focal_shadowing_pancreatic_lesion`.
- This is not an accepted regression-floor pass. Treat the v2 output as evidence that broad prompt
  changes still need narrower validators, retry checks, or scoped candidate filtering before a
  larger corpus run.

After narrowing to only the upstream anatomic selector prompt:

- `.metadata-runs/phase6-targeted-v3/run` completed 12 dry-run assignments with no batch failures.
- The floor comparison found 4 strict mismatches, still worse than the 2-mismatch v1 floor.
- `osseous_lucent_lesion` again produced the intended MR-overreach deterministic audit flag;
  `hypoechoic_liver_lesion` assigned `US` and produced no audit flag.
- The anatomy-scope prompt change remains blocked on regression-floor acceptance. The next decision
  is whether to abandon prompt-only anatomy hardening and pivot to deterministic candidate
  filtering/validators, or keep the prompt change while adding retry/validator support to restore
  the floor.

Special-case deterministic normalization was tried and then removed:

- `.metadata-runs/phase6-targeted-v6-floor/run` completed 10 floor assignments with no batch
  failures, but the urinary-tract improvement in that run came from hard-coded assignment
  normalization.
- That implementation was removed. Continue with general candidate-generation, candidate-filtering,
  and ontology/anatomy evidence improvements only.
