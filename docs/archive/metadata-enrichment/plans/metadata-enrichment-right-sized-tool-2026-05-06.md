# Plan: Right-Size Metadata Enrichment

> Superseded for active execution by `docs/plans/metadata-enrichment-current-plan.md`. Keep this file as historical evidence only; pull any still-useful decisions into the active plan or a stable reference doc before acting on them.



Status: Superseded for active execution; retained as historical evidence
Date: 2026-05-06

## Purpose

Simplify the metadata-enrichment tool without discarding the useful Phase 6 prompt and evaluation
work. The goal is a smaller, more robust implementation with focused prompts, clear ownership, and
sanity validation, not a literal line-count target.

This plan supersedes prompt-only hardening as the next implementation path. It preserves the
grading-aware comparison work, mismatch triage, regression floor, model pinning, and prompt lessons
already documented in the Phase 6 planning artifacts.

## Preserved Lessons

- Use the smallest prompt that passes evals, with clear sectioning, important rules early, compact
  output contracts, and examples only for measured failure modes.
- Do not use reviewed gold answers as hidden prompt examples. Reviewed cases belong in evals and
  regression floors.
- Preserve `reassess` and `fill_blanks_only` semantics: `reassess` may replace existing metadata;
  `fill_blanks_only` preserves populated fields and fills every clearly supported blank field.
- Preserve bounded source context when available: source collection, known anatomy scope, known
  modality bias, and curated-hierarchy status.
- Preserve core prompt decisions:
  - observable imaging duration for time course;
  - exact ontology matches do not automatically force `diagnosis`;
  - canonical ontology codes must be exact or clinically substitutable;
  - anatomy selection must support parent-covers-parts without hard-coded case remapping;
  - modalities are routine direct evaluation modes, not every theoretically visible mode;
  - subspecialties reflect radiology reading workflow, not organ membership alone;
  - etiologies are intrinsic/common mechanisms, not speculative differentials.

## Architecture

Keep the public `assign_metadata(...)` API, CLI behavior, `MetadataAssignmentResult`, and review JSON
shape. Internally, replace the monolithic assignment classifier with focused decision components:

1. Anatomy agent: searches/selects anatomic locations and assigns `body_regions`. It owns
   weak-candidate rejection, parent-covers-parts, classification/score/measurement scope, spine
   segment scope, broad soft-tissue scope, device placement, and body-region mapping from selected
   anatomy.
2. Ontology agent: searches and selects canonical index codes plus review candidates. It owns
   relationship labels and must not prefer one ontology system by name.
3. Identity agent: assigns `entity_type` and `expected_time_course`.
4. Etiology agent: assigns only `etiologies`, using the selected ontology/anatomy and identity
   output as context.
5. Patient Applicability agent: assigns `age_profile` and `sex_specificity`.
6. Imaging Workflow agent: assigns `subspecialties` and `applicable_modalities`.

The orchestrator only runs components, passes compact evidence, merges output, respects assignment
mode, dedupes codes, validates schema/candidate IDs/enums, records provenance, and emits review
warnings.

## Implementation Phases

1. Create this plan and update it as implementation progresses.
2. Refactor assignment prompts into focused agent instructions while retaining the reviewed prompt
   lessons above.
3. Remove brittle deterministic semantic logic:
   - no finding-name, ontology-ID, anatomy-ID, or text-blob metadata assignment;
   - no hand-maintained anatomy adjective-to-location expansion table;
   - no silent exact-match post-processing that overrides ontology judgment;
   - no prompt-content tests.
4. Keep deterministic logic only for sanity and review warnings:
   - schema/enums/candidate IDs;
   - duplicate candidates;
   - ontology display evidence conflicts;
   - descriptor-modality review flags;
   - parent/child review flags;
   - non-disease entity etiology conflicts.
5. Make `field_confidence` optional: validate keys/enums if present; warn rather than retry when a
   changed field lacks confidence.
6. Update focused tests and regression-floor expectations to exercise behavior rather than prompt
   wording.
7. Run focused package verification, then targeted examples and the regression floor where local
   credentials/artifacts allow it.

## Readiness Gate

Before writing enriched metadata back to the full corpus, use this go/no-go rubric. Dry runs can
continue before every item is green; source writes should not.

Mechanical readiness:

- Focused package verification passes: targeted ruff and `task test:findingmodel-ai -- -q`.
- The next larger dry run has no systemic failure class. Isolated failures are acceptable only when
  each is classified as source-data validation, transient provider/runtime failure, or a specific
  tool bug with a follow-up fix.
- Normal single-record runs complete under the configured record timeout. Repeated timeout patterns
  are a no-go until understood.
- No forbidden metadata assignment code exists: no finding-name, ontology-ID, anatomy-ID, or text
  keyword logic that forces field values.

Audit readiness:

- Deterministic audit flags are zero or explicitly triaged with source evidence.
- No audit flag class repeats across unrelated records.
- Any audit false positive is addressed by narrowing the check or documenting why it should remain
  review-only.

Qualitative readiness:

- A reviewed random sample from the larger dry run shows sane metadata across findings, diagnoses,
  measurements, assessments, devices, anatomy variants, and broad system findings.
- Anatomy follows the parent/child rule: if a parent location covers all involved parts, select the
  parent; otherwise select the specific affected structure without over-narrowing to unsupported
  children.
- Ontology `index_codes` represent the modeled concept, not merely a target anatomy, component,
  imaging view, procedure, or related differential.
- Optional fields are conservative. Etiologies and expected time course require direct support from
  the modeled finding itself, not differential context, source tags alone, or plausible clinical
  associations.
- Existing bad metadata can be cleared in reassess mode when the review rationale says it is
  unsupported.
- Anatomy and workflow fields are direct and routine: body region matches selected anatomy,
  subspecialty reflects reading workflow, and modalities match the rationale and source support.

Operational readiness:

- Dry-run artifacts include before/after JSON, review JSON, audit JSON, and status JSONL for every
  processed record.
- The team can inspect failures and rerun targeted subsets without changing source files.
- The plan document records the latest dry-run path, outcome, unresolved issues, and next decision.

Go/no-go threshold for the next large dry run:

- At least 200 dry-run records.
- 0 systemic failures.
- 0 unresolved deterministic audit flags.
- No unresolved recurring qualitative failure class in a reviewed sample of at least 25 mixed
  records.
- Any source-data validation failures are listed explicitly and are either repaired, excluded from
  the write pass, or handled by a preflight report.

## Supervised Writeback Workflow

The enrichment tool is not intended to write directly to source files without supervision. Use each
larger dry run as a proposed metadata batch, then decide which records are safe to promote.

Each subagent-reviewed record should receive one queue label:

- `proposed_accept`: the source concept is in scope and the enriched metadata is supported well
  enough to write back.
- `proposed_skip`: the source concept is out of scope for this enrichment pass, such as a clinical scale,
  laboratory/clinical-only concept, template/protocol/module, or other record that is not a single
  imaging-observable finding model.
- `needs_attention`: the source concept is probably in scope, but one or more fields need
  domain judgment before writeback.
- `suspected_tool_problem`: the record shows a repeated tool or prompt failure pattern that should
  be fixed before similar records are promoted.

Human review actions are deliberately smaller:

- `approved`: eligible for source writeback.
- `skipped`: excluded from this enrichment pass.
- `feedback`: not applied yet; needs correction, domain input, or prompt/tool follow-up.

First-pass review should classify records, not hand-edit every output. The reviewer should use
field-level callouts for the reason a record is not immediately accepted:

- `index_codes`: selected canonical codes are anatomy-only, procedure-only, broader/narrower,
  qualifier-dropping, or otherwise not the modeled concept.
- `anatomic_locations` / `body_regions`: selected anatomy is unsupported, too broad, too narrow, or
  body region does not follow the selected anatomy.
- `etiologies`: causal buckets are assigned from differential diagnosis, weak tags, or plausible
  association rather than the modeled finding itself.
- `applicable_modalities` / `subspecialties`: values come from routine-practice inference or source
  workflow noise rather than direct support for the modeled finding.
- `entity_type`, `expected_time_course`, `age_profile`, or `sex_specificity`: values need policy or
  domain review.

Promotion rule:

- Only human `approved` records should be applied to `defs/*.fm.json`.
- Human `skipped` records stay untouched and should be logged so they are not repeatedly re-reviewed in the
  same pass.
- Human `feedback` records stay untouched until corrections are made or a later review changes them
  to `approved` or `skipped`.
- `suspected_tool_problem` patterns feed the next focused prompt/tool improvement cycle.

This review layer is separate from metadata assignment. It must not become brittle assignment code
that forces metadata values from finding names, ontology IDs, or keyword lists. It is a supervised
promotion contract: decide whether a generated artifact is safe to write, and use repeated failure
patterns to improve the focused sub-agent prompts.

The subagent prompt and its surrounding review context are maintained in
`docs/plans/metadata-enrichment-supervised-review-prompt-2026-05-10.md`. That document is the source
of truth for the review criteria, examples, output schema, and collation rules used for supervised
batch review.

## Small-Batch Sanity Review Prompt

Use this prompt for subagents that review small batches of enriched dry-run artifacts. Assign each
subagent a concrete run directory and 5-10 records. The subagent should read the `.after.json`,
`.before.json`, `.metadata-review.json`, `.audit.json`, and relevant source `defs/*.fm.json` files
for its assigned records.

```text
You are reviewing a small batch of metadata-enriched finding model records for corpus-write
readiness. Your job is not to rewrite the data and not to propose broad architecture changes. Your
job is to identify concrete metadata quality problems in the assigned records and summarize whether
this batch supports readiness for a larger write pass.

Inputs:
- Run directory: <RUN_DIR>
- Assigned records: <RECORD_STEMS_OR_PATHS>
- Source repo: /Users/talkasab/repos/findingmodels-metadata
- Tool repo: /Users/talkasab/repos/findingmodel-metadata

For each assigned record, inspect:
- before-after/<stem>.before.json
- before-after/<stem>.after.json
- reviews/<stem>.metadata-review.json
- audits/<stem>.audit.json
- defs/<stem>.fm.json when needed for source context

Look for these issue classes:
1. Mechanical failure: missing artifact, failed status, invalid JSON, required metadata still null,
   or review/audit files inconsistent with the output.
2. Ontology issue: model-level index_codes are anatomy-only, target-only, procedure-only,
   component-only, broader/narrower/related concepts, or otherwise not canonical for the modeled
   finding. Exact anatomy concepts may be acceptable as anatomic_locations, not necessarily as
   index_codes.
3. Anatomy issue: selected anatomy is too broad, too narrow, missing, includes unsupported children,
   or fails the parent-covers-parts rule. Parent-covers-parts means if multiple parts are involved
   and a parent location actually covers all of them, select the parent instead of the separate
   parts.
4. Body-region issue: body_regions do not match selected anatomy or incorrectly use whole_body for a
   localized organ/vessel. Use whole_body only for true system-level or nonlocalized findings.
5. Modality issue: applicable_modalities include speculative modalities, omit directly supported
   routine modalities, or contradict the rationale.
6. Subspecialty issue: subspecialties are inferred from body region alone, omit an obvious reading
   workflow, or include unrelated workflows.
7. Optional-field issue: etiologies, expected_time_course, age_profile, or sex_specificity are
   filled from differential context, source tags alone, or plausible associations rather than direct
   support. Be especially skeptical of generic masses, lesions, abnormalities, enhancement patterns,
   effusions, and measurements.
8. Entity-type issue: entity_type is inconsistent with the model: measurement/assessment/finding/
   diagnosis/recommendation/technique_issue/grouping.
9. Existing-metadata clear issue: before.json had unsupported existing metadata that should have
   been cleared in reassess mode but survived, or supported existing metadata was incorrectly
   cleared.
10. Rationale/output mismatch: the rationale says a value is unsupported or should be omitted, but
    the structured output still contains it.

Do not flag mere preference differences. Flag a record only when you can state the concrete field,
the observed output, and why the source/review evidence does not support it.

Output exactly this structure:

Batch: <batch label>
Records reviewed: <count>
Overall readiness signal: green | yellow | red

Findings:
- <severity: high|medium|low> <record stem> <field>: <concrete issue>
  Evidence: <specific before/after/review/audit/source evidence>
  Suggested next action: <fix prompt/structural issue/source data/review manually/no tool change>

Patterns:
- <recurring issue class, if any>

Good examples:
- <record stem>: <brief note on why it looked sane>

Open questions:
- <only questions that block interpretation from local context; otherwise "None">

Readiness recommendation:
- <one concise paragraph saying whether this batch supports a larger dry run or source write, and
  what must happen first>
```

## Status Log

- 2026-05-06: Plan created. Implementation starting with focused prompt/pipeline refactor.
- 2026-05-06: Focused decision-agent refactor implemented behind the existing `assign_metadata(...)`
  API. Production assignment now uses anatomy, ontology, identity, and workflow/applicability
  agents, then assembles the existing review/result shape.
- 2026-05-06: Removed the hand-maintained anatomy adjective-to-location expansion table. Anatomic
  query preparation now keeps generic text cleanup only.
- 2026-05-06: Removed ontology categorization post-processing that silently promoted exact text
  matches after the model decision. Ontology applicability now remains with the ontology decision
  agent.
- 2026-05-06: `field_confidence` is now optional at runtime. Invalid keys still fail schema
  validation, but missing confidence entries produce review warnings rather than model retries.
- 2026-05-06: Focused verification passed:
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_assign_metadata_modes.py packages/findingmodel-ai/tests/test_anatomic_search.py packages/findingmodel-ai/tests/test_ontology_search.py -q`
  and
  `uv run ruff check packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py packages/findingmodel-ai/src/findingmodel_ai/search/anatomic.py packages/findingmodel-ai/src/findingmodel_ai/search/ontology.py packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_assign_metadata_modes.py packages/findingmodel-ai/tests/test_anatomic_search.py packages/findingmodel-ai/tests/test_ontology_search.py`.
- 2026-05-06: Audited the slow full-suite behavior. The slow tests are intentional `callout`
  coverage, especially `test_model_smoke.py`, which hits live model providers for every configured
  model and reasoning level. The correct local verification path is the existing task target, not raw
  package pytest. Verification passed: `task test:findingmodel-ai -- -q -x` with 200 passed and 85
  callouts deselected. Run `task test:findingmodel-ai-full` when live provider connectivity is the
  explicit goal.
- 2026-05-06: Regression-floor v1 exposed mechanical failures where a focused agent requested
  `clear_fields` for required fields. Required-field clears are now ignored with review warnings
  unless a replacement value is also supplied, so the orchestrator preserves valid existing data
  instead of failing after focused agents have already completed.
- 2026-05-06: Focused verification after the required-field clear fix passed:
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q`,
  targeted `ruff check`, and `task test:findingmodel-ai -- -q -x` with 200 passed and 85 callouts
  deselected.
- 2026-05-06: Candidate-field clears from focused agents are now ignored with review warnings, so
  ontology and anatomy decisions cannot mechanically erase selected candidate fields. Existing
  supported identity, patient applicability, and imaging workflow fields are preserved when a
  focused agent omits them in reassess mode; non-disease entity etiologies are sanitized with
  warnings rather than producing assignment failures.
- 2026-05-06: Anatomy candidate gathering now runs after ontology matching and receives compact
  ontology labels as locality context. Exact anatomic term hits are retained and prioritized in the
  offered candidate set, which keeps explicit parent locations such as `urinary tract` available
  for the anatomy agent without hard-coded finding-name rules.
- 2026-05-06: Focused prompt hardening incorporated reviewed lessons for parent-covers-parts
  anatomy, device course/placement anatomy, ontology qualifier preservation, esophageal workflow
  routing, radiolucent urinary calculus modalities/routing, and breast calcification-cluster
  exactness.
- 2026-05-06: Targeted reviewed examples verified after the final candidate-flow fix:
  `radiolucent_urinary_calculus` produced exactly `urinary tract`, CT/US, GU+ER, abdomen;
  `breast_calcification_cluster` produced only `SNOMEDCT:129769006` for index codes; and
  `tunneled_catheter` produced `anterior chest wall`.
- 2026-05-06: Regression floor v15 passed with 10/10 dry-run successes, 0 strict mismatches, and 0
  deterministic audit flags. Report:
  `/Users/talkasab/repos/findingmodels-metadata/docs/plans/metadata-enrichment-regression-floor-results-phase6-right-sized-v15-2026-05-06.md`.
- 2026-05-06: Final focused package verification passed:
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_assign_metadata_modes.py packages/findingmodel-ai/tests/test_anatomic_search.py -q`
  with 33 passed; targeted `ruff check` passed; `task test:findingmodel-ai -- -q` passed with 207
  passed, 85 callouts deselected, and one Logfire no-config warning.
- 2026-05-06: First 200-record dry run completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v16-200/run`.
  Initial result was 197 dry-run successes and 3 timeout failures. The failures were
  `aortic_atherosclerosis`, `ascites_in_an_infant_or_child`, and
  `asymmetrically_small_bones_of_one_hand`; tracebacks showed record-window cancellation during
  ontology and anatomic embedding/model calls, not schema/domain conflicts.
- 2026-05-06: Timeout follow-up completed. Batch retries no longer retry the batch runner's own
  record timeout, and OpenAI embedding clients now use explicit timeout/retry settings. Rebuilt
  wheels and reran the 3 timed-out records with no whole-record retry in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v17-timeout-recheck/run`;
  all 3 completed successfully with 0 deterministic audit flags.
- 2026-05-06: Verification after timeout fixes passed:
  `task test:oidm-common -- -q` with 140 passed and 2 deselected; targeted `ruff check` passed for
  `packages/oidm-common/src/oidm_common/embeddings/generation.py` and
  `/Users/talkasab/repos/findingmodels-metadata/scripts/metadata_assign_batch.py`.
- 2026-05-06: Deterministic audit review of the 197 successful initial 200-record artifacts found 7
  flags. Follow-up narrowed false-positive abdomen/pelvis audit behavior for appendix/sigmoid bowel
  findings and tightened focused guidance for generic arterial-system findings and vascular devices.
  Rechecked flagged records in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v18-audit-flag-recheck/run`
  and
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v19-arterial-stent-recheck/run`;
  final rechecked deterministic audit flags were 0.
- 2026-05-06: Final verification after 200-record follow-up passed:
  `task test:findingmodel-ai -- -q` with 208 passed, 85 callouts deselected, and one Logfire
  no-config warning.
- 2026-05-06: Fresh 15-record qualitative run completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v38-fresh-15-qual/run`.
  The run had 15/15 mechanical successes and 0 deterministic audit flags, but qualitative review
  found unresolved assignment problems, so the tool is not ready for a larger corpus pass.
- 2026-05-06: Removed an attempted keyword-based validator change from the focused identity and
  workflow/applicability agents. Validators must remain structural sanity checks over schema,
  candidate IDs, enum consistency, and selected structured evidence; they must not inspect finding
  text for specific words and force metadata outcomes. Focused verification after removal passed
  with 44 tests and targeted ruff.
- 2026-05-06: Tightened focused prompts for the fresh-slice failure pattern without adding
  keyword-specific assignment code. Identity and workflow/applicability agents now explicitly
  require output values to agree with their rationale, reject use of rejected candidates as support,
  avoid deriving subspecialty from body region alone, and restrict `clear_fields` to metadata field
  names. Anatomic query generation now asks the model to resolve ambiguous anatomy from source
  context rather than searching every homonym. Focused verification passed with 44 tests and
  targeted ruff.
- 2026-05-06: Second fresh-slice cleanup pass broadened candidate offering without adding semantic
  special cases: ontology categorization now receives more top candidates, anatomic candidate search
  no longer lets a single generated region filter suppress plausible candidates, identity guidance
  requires medium/high support before outputting optional etiologies, and anatomy guidance clarifies
  that broad region phrases do not outrank specific organs/systems/vessel sets that preserve the
  full modeled anatomy. Focused verification passed with 65 tests and targeted ruff.
- 2026-05-06: Third fresh-slice cleanup pass added a structural assembly rule: low-confidence
  optional fields are ignored with review warnings instead of being written. The identity and
  workflow/applicability prompts now also state that rejected, broader, narrower, related, and
  review-only ontology candidates cannot support their metadata fields. Ontology query generation
  now asks for formal wording variants that preserve anatomy and qualifiers. Focused verification
  passed with 65 tests and targeted ruff.
- 2026-05-06: v43 six-record recheck had 5/6 mechanical successes and confirmed several
  improvements: adrenal subspecialty was suppressed when low confidence, aortic atherosclerosis
  recovered the generic SNOMED code, abnormal ventricular configuration selected lateral ventricle
  without cardiac routing, and AV fistula no longer included XR/RF. Remaining work items were one
  hallucinated candidate-ID failure and medium-confidence invented etiologies. Candidate-ID output is
  now filtered structurally before assembly. At that point non-null etiology assignments required
  high confidence while still allowing explicit clears; the 2026-05-07 dedicated etiology-agent
  change supersedes that by accepting medium/high etiology support and still dropping low-confidence
  values. Focused verification passed with 65 tests and targeted ruff.
- 2026-05-06: v44 six-record recheck completed with 6/6 mechanical successes and 0 audit flags.
  Etiology overfill was fixed for the reviewed issue set. Remaining qualitative issues were
  medium-confidence time-course overfill, pediatric routing added from possible congenital context,
  broader-than-needed modalities, and anatomy sometimes preferring a bare region over a more specific
  offered structure. Prompt guidance now addresses those general patterns, and non-null
  `expected_time_course` assignments now require high confidence. Focused verification passed with
  65 tests and targeted ruff.
- 2026-05-06: Shifted the next workstream away from mechanics-driven rerun loops and back to
  prompt/response quality. Prompt updates now require agents to classify optional fields as directly
  supported, contradicted, or unclear before outputting values; require rationale support for every
  non-null optional field; treat generic base findings as null for etiology/time course unless the
  cause or natural history is part of the modeled finding; reject anatomy-only ontology concepts as
  canonical abnormality codes; and instruct anatomy selection to prefer the affected structure over a
  bare region when both are offered. Fast ruff check passed for the prompt edits.
- 2026-05-06: v46 prompt-response check showed improved conservative optional-field behavior, but
  also showed that generic vessel findings still need better broad-system candidate search and that
  generic parent-vessel anatomy can be over-split into named segments. Prompt updates now ask anatomy
  query generation to search vessel-system terms for generic vessel connections, ask anatomy
  selection to choose a parent vessel alone when generic parent and segment candidates are offered,
  and clarify that etiology is cause, not the organ system containing the finding. Fast ruff check
  passed for the prompt edits.
- 2026-05-06: v47 vessel-focused response check improved `arteriovenous_fistulas` back to
  `arterial system` and kept `abnormal_abdominal_vessels` on `set of abdominal vessels` with no
  etiology/time-course overfill. It also exposed a body-region response problem: named vessels such
  as `aorta` were still being treated as `whole_body`. Body-region prompt guidance now
  distinguishes true system-level anatomy from named vessels and tells the agent not to use
  `whole_body` for a named vessel or organ. Fast ruff check passed.
- 2026-05-06: v48 aortic-only response check still chose `whole_body` for `aorta`, so body-region
  prompt guidance now gives concrete named-vessel examples: aorta spans chest/abdomen, carotid
  spans neck/head, pulmonary artery is chest, while arterial/vascular system can be whole body.
  Fast ruff check passed. The aortic response remains the next qualitative item to recheck.
- 2026-05-06: v49 aortic-only response check fixed body region (`chest` + `abdomen`) and preserved
  generic `aorta` anatomy, but the workflow/applicability response still contradicted itself by
  outputting `XR` while its rationale said XR was unsupported. Prompt guidance now includes an
  explicit final self-check: remove any modality, subspecialty, age, or sex value that the rationale
  criticizes or says should be omitted. Fast ruff check passed.
- 2026-05-06: v50 aortic-only response check kept `aorta`, `chest` + `abdomen`, `VA`, and null
  expected time course. Remaining aortic judgments are qualitative rather than mechanical:
  degenerative etiology and CT/XR modality support. Next response-quality check is the 15-record
  fresh slice.
- 2026-05-06: v51-v57 fresh-slice cleanup moved the remaining work back to prompt/response quality
  rather than deterministic assignment code. Changes made:
  focused prompts now explicitly prevent internal response fields from being used as `clear_fields`;
  the auditor was reduced to source-backed consistency checks and LLM review, removing opinionated
  deterministic modality/subspecialty heuristics; anatomy search now prompts generic artery-vein
  relationships toward system-level vascular search terms; workflow prompts require the JSON
  modality list to match the rationale; and identity prompts now treat cause-list language such as
  "common causes include" or "possibly due to" as differential context, not etiology support.
  Focused ruff and pytest checks passed after these edits.
- 2026-05-06: v57 15-record stability check completed with 15/15 dry-run successes and 0
  deterministic audit flags in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v57-fresh-15-stability-check/run`.
  Qualitative review showed the targeted failure classes corrected: `abnormal_intracranial_enhancement`
  retained null etiologies, `arteriovenous_fistulas` selected `arterial system`, `airspace_consolidation`
  stayed on XR only, `abnormal_abdominal_vessels` selected `set of abdominal vessels`, and
  `anisospondyly` selected spine with null etiology/time course. Remaining differences in the sample
  are judgment calls to watch in a broader run, especially entity type for named disease-like findings
  and optional time course on fixed structural absence.
- 2026-05-06: v58 random 50-record dry run completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v58-random-50-check/run`
  with 46/50 successes. Three failures were tool-owned and were fixed: null ontology display handling
  in anatomic filtering, required modality guidance for records with direct modality tags, and removal
  of retry-style parent/child anatomy validation that belonged in prompt guidance. The fourth failure,
  `occipital_lobe_region_volumes`, is an invalid source record: existing model-level index codes
  `RADLEX:RID6502` and `SNOMEDCT:31065004` lack required display values before enrichment starts.
- 2026-05-07: Resolved the `occipital_lobe_region_volumes` pre-enrichment validation failure by
  adding reviewed display values to the source record:
  `RADLEX:RID6502` -> `occipital lobe` and `SNOMEDCT:31065004` ->
  `occipital lobe structure`. v62/v64 rechecks of the record completed successfully with 0
  deterministic audit flags.
- 2026-05-06: v59 failure recheck passed for the three tool-owned failures:
  `choanal_stenosis`, `t1_hyperintense_liver_lesion`, and
  `congenital_renal_or_ureteral_anomaly` all completed with 0 deterministic audit flags.
- 2026-05-06: v60-v61 qualitative follow-up fixed the one audit-flagged random-slice output and a
  stale existing time-course issue. Body-region guidance now maps male genital anatomy such as
  testis to `pelvis`, so `macroorchidism` rechecked with `testis`, `pelvis`, `GU`, US/MR, and 0
  audit flags.
  Reassess assembly now honors explicit optional identity clears unless the model marks the clear
  low/medium confidence, so `pericardial_effusion` clears unsupported existing
  `expected_time_course`.
- 2026-05-06: Final focused verification after v58-v61 fixes passed: targeted ruff passed;
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q` passed with 30 tests; and
  `task test:findingmodel-ai -- -q` passed with 204 tests, 85 callouts deselected, and one Logfire
  no-config warning.

## Historical Status

Source-support hardening, fresh-slice cleanup, and a 50-record random dry-run follow-up are
implemented. The tool is ready to try a larger dry-run corpus slice. Source writes should still stay
disabled until the next larger dry run is inspected qualitatively.

- Anatomy search now returns candidate locations with evidence labels instead of making the final
  primary/alternate decision itself.
- Existing `anatomic_locations` are treated as current metadata context, not as source proof. This
  matters because those fields are exactly what reassessment is allowed to correct.
- Source ontology labels, finding name, description, and synonyms provide the strongest anatomy
  support. Attribute option values still help find candidates, but they no longer outrank a covering
  top-level location.
- The focused anatomy agent remains the only component that decides final anatomy and body regions.
  The orchestrator now adds only generic sanity validation:
  - selected system-level anatomy cannot be combined with named component locations;
  - selected system-level anatomy should map to `whole_body` body-region output rather than component
    regions;
  - the source support level consistency check warns when selected anatomy is narrower or weaker than
    available support.
- Focused prompt updates preserve reviewed lessons without deterministic per-finding assignment
  rules: parent covers children when it covers the full scope; tunneled catheter course/entry can be
  anterior chest wall; generic aortic findings should not narrow to thoracic/abdominal aorta without
  source support; generic vascular-device findings should use system-level anatomy when appropriate;
  generic masses/calcifications should not receive broad differential etiologies; breast anatomy
  maps to the `breast` body region rather than `chest`; nearby landmarks or partial anatomy should
  not be selected when they do not cover the modeled anatomy.
- The deterministic auditor has been narrowed back to structural checks only. Removed text-hint
  body-region exceptions, descriptor/modality substring checks, and the unsafe assumption that
  anatomic-index `Body` means finding-model `whole_body`. The anatomic-index-region to
  finding-model-body-region check was also removed after the 200-record run showed it produced noisy
  boundary-region flags instead of dependable corpus readiness signals.
- Focused validators are structural only. Text-keyword validators that attempted to reject specific
  vocabulary patterns were removed.
- The latest prompt cleanup targets general support failures from the fresh-slice review: rejected
  candidates cannot justify patient applicability or imaging workflow metadata, rationale/output
  contradictions should be resolved by nulling or clearing the unsupported field, body region alone
  does not imply an organ-system subspecialty, and ambiguous anatomy should be disambiguated from
  source context by the query agent.
- Candidate gathering has been loosened where the prior flow could hide the right answer: the
  ontology classifier sees more returned candidates, and anatomy embedding search is not constrained
  by a single generated region. The anatomy agent still decides final applicability from offered
  candidates.
- The current priority is prompt/response behavior, not more mechanical proof. The agents should be
  conservative about optional fields and must distinguish direct support from plausible clinical
  context in their rationales.

Verification:

- Focused tests and lint passed:
  `uv run pytest packages/findingmodel-ai/tests/test_anatomic_search.py packages/findingmodel-ai/tests/test_assign_metadata.py packages/findingmodel-ai/tests/test_assign_metadata_modes.py packages/findingmodel-ai/tests/test_enrichment_auditor.py -q`
  and targeted `ruff check`.
- Full package verification passed:
  `task test:findingmodel-ai -- -q` with 206 passed, 85 callouts deselected, and one Logfire
  no-config warning.
- Final focused corpus dry run passed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v37-clean-auditor-final-focused/run`:
  11 processed, 0 failures, 0 deterministic audit flags.
- 200-record dry run completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v65-random-200-check/run`:
  200 processed, 196 successes, 4 failures, 5 deterministic audit flags. The four failures all
  passed on focused rerun in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v66-failure-recheck/run`,
  so they are being treated as batch-load/retry robustness work rather than confirmed content
  failures.
- After removing the noisy anatomic-index body-region check and tightening general anatomy/workflow
  prompts, the five flagged records passed focused rerun with 0 failures and 0 deterministic audit
  flags in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v69-audit-flag-recheck-updated-wheel/run`.
- A follow-up qualitative subset after the latest prompt cleanup passed mechanically in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v70-qualitative-recheck/run`;
  remaining review attention is on qualitative choices such as boundary body-region breadth,
  partial anatomic landmarks, and optional workflow labels.
- Three qualitative review subagents checked 26 v65 records across anatomy/body-region,
  assessment/pediatric/spine, and diagnosis/lesion/optional-field batches. All three returned
  `yellow`, not `green`: the tool was good enough for another larger dry run, but not source writes.
  Recurring issues were over-broad or over-narrow anatomy for generic/site-flexible findings,
  rationale/output mismatches on optional fields, broader ontology codes that dropped fetal
  qualifiers, entity type misses for grading scales and aneurysms, and modality additions from
  routine-practice inference rather than source support.
- Prompt updates from the subagent review remain general rather than per-record assignment code:
  generic site-flexible lesions use `whole_body` instead of enumerating or choosing a common site;
  possible causes such as lymphadenopathy do not become anatomy for a mass; fetal qualifiers must be
  preserved in canonical ontology; named scales/classifications are assessments; aneurysms are
  diagnoses; source modality tags outrank routine-practice guesses; fetal age should not be mapped
  to maternal adult age bins.
- Focused rerun of the subagent-flagged records passed mechanically in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v72-residual-qualitative-recheck/run`:
  5 processed, 0 failures, 0 deterministic audit flags. The key prior problems improved:
  `hemangioma` now uses `whole_body`; `mass_in_the_porta_hepatis` no longer selects hepatoportal
  lymph nodes; `fetal_hepatosplenomegaly` keeps only the fetal exact code; `arterial_aneurysm` is
  `diagnosis`.
- Fresh 200-record rerun with the current wheel completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v73-random-200-current-wheel/run`:
  200 processed, 199 successes, 1 timeout, 0 deterministic audit flags. The single timeout
  (`ribbon_like_ribs`) passed on targeted rerun in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v74-single-failure-recheck/run`.
  Successful v73 records had median duration 23.161 seconds and maximum duration 43.7 seconds.
- Current qualitative spot checks from v73 show the main reviewed failure classes are improved:
  breast clips maps to `breast`/`BR`; BI-RADS and ESCC are assessments; hemangioma uses
  `whole_body`; fetal hepatosplenomegaly keeps only the fetal exact code; arterial aneurysm is a
  diagnosis; the prior v65 timeout records completed in the larger run. Remaining qualitative review
  risk is stochastic anatomy narrowing when the correct specific location is not offered, seen in
  one v73 `mass_in_the_porta_hepatis` output that included a hepatoportal lymph-node candidate.
  A generic search-only anatomy guard was tried and rejected because it also removed legitimate
  breast anatomy; this should stay prompt-level or candidate-evidence work, not brittle filtering.
- Full local package verification passed after the current prompt changes:
  `task test:findingmodel-ai -- -q` with 202 passed, 85 callouts deselected, and one Logfire
  no-config warning.
- Deterministic dry-run artifact promotion is implemented in the data repo as
  `scripts/metadata_apply_dry_run_outputs.py`. It defaults to report-only mode, requires an
  explicit approved selector, refuses failed/audit-flagged/missing/invalid artifacts, refuses
  source files that no longer match `before.json`, and can apply accepted `after.json` artifacts
  with `--write` without calling the enrichment agents. Temp-directory verification covered
  report-only, write, idempotent already-applied, and source-drift refusal paths.
- 500-record dry run completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v76-random-500-current-wheel/run`:
  500 processed, 460 successes, 40 failures, and 0 deterministic audit flags. Serial retry of the
  40 failures in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v77-failure-recheck/run`
  recovered 36. The remaining work items were `hypoglycemia`,
  `benign_synovial_lesion`, `multiple_hypointense_liver_lesions`, and
  `global_brain_volumes`.
- Targeted follow-up fixed the `global_brain_volumes` source validation failure by adding reviewed
  display values for the model-level brain codes, tightened general prompt guidance for broad and
  site-variable findings, and added generic structural safety checks for low-confidence required
  fields, low-confidence anatomy selections, and non-exact canonical ontology codes on assessment or
  measurement models. These are sanity checks over structured evidence, not finding-text or
  ontology-ID assignment rules.
- Latest targeted rechecks:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v84-targeted-assembly-recheck/run`
  produced sane `global_brain_volumes` output with unsupported model-level anatomy codes cleared
  from `index_codes`; and
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v85-anatomy-confidence-recheck/run`
  produced sane `benign_synovial_lesion` output with no unsupported joint anatomy. `hypoglycemia`
  now fails safely because the source record has no medium/high-confidence imaging modality support,
  instead of writing weak CT/XR metadata.
- Fresh current-wheel 200-record verification completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v86-random-200-current-wheel-post-safety/run`:
  200 processed, 191 successes, 9 safety-check failures, and 0 deterministic audit flags. Successful
  rows had median duration 23.65 seconds and maximum duration 74.631 seconds. The nine failures were
  all explicit safety refusals: low-confidence required `body_regions`, low-confidence anatomy used
  for localized body region, non-exact canonical ontology for assessment/measurement, or
  low-confidence required `applicable_modalities`.
- Human review page for the 191 successful v86 outputs is generated at
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v86-human-review-191/index.html`.
  The page embeds the review data directly, and the preserved dataset is at
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v86-human-review-191-data/review-data.json`.
- Qualitative spot checks from v86 showed the main reviewed behaviors are improved enough to stop
  prompt tuning for this phase: breast calcification cluster stayed exact with breast anatomy; breast
  clips stayed breast/BR; BI-RADS stayed assessment; breast density stayed measurement; axillary mass
  avoided lymph-node narrowing; radiodense urinary calculus selected urinary tract; spine hardware
  selected spine. Residual review items remain, including judgment calls around broad vascular/device
  body regions, optional workflow labels, and records that are clinical/laboratory rather than
  image-observable. These should be handled by review/exclusion during promotion rather than another
  prompt-edit loop unless they recur as a clear class in source-write review.

Qualitative checks from the final focused run:

- `radiolucent_urinary_calculus`: `urinary tract` only.
- `tunneled_catheter`: `anterior chest wall`.
- `aortic_stent`: `aorta`, not `thoracic aorta`.
- `arterial_stent`: `arterial system`, `whole_body`, exact `Arterial stent` code.
- `acquired_fused_vertebrae`: `spine`, not cervical-only anatomy.
- `axillary_mass`: `upper extremity`, no axillary lymph-node narrowing, no broad etiology list.
- `breast_calcification_cluster`: exact `Calcification cluster` code, no neoplastic etiology.

## Superseded Next Step

Prepare controlled promotion rather than continuing prompt tuning:

- treat the v86 safety failures as exclude/manual-review records unless source evidence is added;
- build an approved manifest from successful dry-run artifacts after human spot review;
- apply accepted `.after.json` artifacts with `scripts/metadata_apply_dry_run_outputs.py --write`
  rather than regenerating them;
- continue to use dry-run plus targeted qualitative review for larger source-write batches, but do
  not reopen prompt tuning unless a repeated failure class appears.

Historical review:

- 2026-05-07: Current state is ready for a larger dry-run slice, not for source writes. Mechanical
  readiness is good on the 200-record rerun, but qualitative review still needs to watch anatomy
  narrowing for generic/site-flexible findings.
- 2026-05-07: 500-record dry run and targeted failure follow-up are complete. The next gate is a
  fresh current-wheel verification slice plus qualitative review, because prompt and structural
  safety checks changed after the 500-record run.
- 2026-05-07: Fresh v86 200-record verification after the safety changes is complete. Current
  recommendation is to stop prompt tuning for this phase and move to controlled promotion of reviewed
  successful artifacts, with safety-failed records excluded or manually reviewed.
- 2026-05-07: Generated a single-file human review page for the 191 successful v86 outputs.
- 2026-05-07: v86 human-review sampling found etiology underfill: only 53 of 191 successful records
  had non-null etiologies. The next implementation step is to move etiology decisions out of the
  identity prompt into a dedicated focused etiology agent. The agent should select a small set of
  broad, common causal buckets when the modeled concept supports them, keep generic broad
  abnormalities null, and rely on structural sanity checks only. Medium/high etiology support should
  be accepted; low-confidence etiologies should still be dropped.
- 2026-05-07/08: Dedicated etiology agent implemented. Identity now assigns only entity type and
  expected time course; etiology has its own focused prompt and structural checks. Medium/high
  etiology confidence is accepted and low-confidence etiologies are dropped. Focused verification
  passed with `uv run ruff check packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py` and `uv run pytest
  packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q` with 33 tests.
- 2026-05-07/08: Targeted dry-run checks after the etiology split:
  - v87 targeted run exposed one code bug: `ExpectedTimeCourse` identity context was serialized as
    though it were an enum. Fixed by JSON-serializing the structured time-course object.
  - v88 targeted run completed 15/15 mechanically with zero deterministic audit flags, but showed
    generic etiology overfill for `axillary_mass` and `breast_calcification_cluster`.
  - v89/v90 prompt rechecks fixed those generic etiology overfills and restored
    `radiolucent_urinary_calculus` to `urinary tract` only.
  - v91 had timeout failures under concurrent recheck; v92 serial retry completed the failed
    records with zero audit flags and the expected qualitative outputs.
  - v93 confirmed `tunneled_catheter` keeps `iatrogenic:device` and `anterior chest wall`. Generic
    pericardial effusion still includes broad common etiologies including a post-operative bucket;
    this is being treated as broad-cause behavior rather than a per-case prompt target unless review
    establishes a narrower etiology standard.
- 2026-05-08: Current-wheel v94 15-record serial smoke completed with 14 dry-run successes and one
  validator failure on `axillary_mass`. The failure was not a metadata semantic conflict; the final
  consistency validator raised after the agents completed because low-confidence anatomy was paired
  with a localized body region. Assembly already drops low-confidence anatomic selections with a
  warning, so the hard failure was removed. Focused package verification still passed after the
  change: `task test:findingmodel-ai -- -q -x` reported 205 passed and 85 callouts deselected. v95
  rechecked `axillary_mass` successfully with null etiologies, `upper_extremity` body region,
  exact mass index codes, and zero deterministic audit flags.
- 2026-05-08: Split the remaining broad workflow/applicability component into two focused agents:
  Patient Applicability assigns only `age_profile` and `sex_specificity`; Imaging Workflow assigns
  only `subspecialties` and `applicable_modalities`. Anatomy now owns `body_regions` alongside
  anatomic candidate selection, so the orchestrator continues to assemble outputs and run structural
  validation rather than deciding field semantics. Focused verification passed with
  `uv run ruff check packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py` and
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q` with 35 tests.
  Package-local verification also passed with `task test:findingmodel-ai -- -q -x`: 207 passed, 85
  callouts deselected, and one Logfire no-config warning.
- 2026-05-08: Prompt-compression pass started after v96 smoke showed clean mechanics but workflow
  overreach. Goal: replace accumulated micro-rules with shorter reusable decision tests, especially
  for imaging workflow and unlocalized body-region assignment.
- 2026-05-08: Imaging Workflow prompt now uses an explicit evidence standard instead of more
  example-specific cautions. Direct support is limited to source text/tags for the modeled finding,
  selected canonical ontology that itself implies workflow/modality, and selected anatomy only when
  the modeled finding is normally read in that anatomy-specific workflow. Existing metadata,
  attributes, rejected/review candidates, adjacent anatomy, possible complications, and downstream
  workup are context only.
- 2026-05-08: Current next step is a fresh mixed qualitative dry-run after the evidence-standard and
  assembly-contract changes. The pass should inspect successful records by field quality, especially
  weak-evidence modalities, weak-evidence subspecialties, existing-metadata copying, attribute
  leakage, etiology overfill, time-course preservation, and anatomy/body-region scope.
- 2026-05-08: v104 bounded 25-record quality run completed in
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v104-quality-eval-bounded/run`:
  18 successes, 7 failures, 0 deterministic audit flags. Failures were five 90-second timeouts and
  two low-confidence required `body_regions` validator failures. Qualitative review of the successes
  found repeated output-quality issues, so the tool is not ready for a larger corpus pass:
  etiology overfill on broad findings/diagnoses, modality/workflow overfill on procedure and cardiac
  records, unstable time-course preservation, and occasional age/applicability overfill.
- 2026-05-08: Logfire diagnostic runs v105/v106 showed that slow single-record runtime was driven
  by oversized candidate-list prompts, not BioPortal latency. One anatomy decision call for
  `breast_density` had been asked to judge 100 gathered candidates and used about 73k focused-phase
  input tokens before candidate capping. The enrichment config now has
  `metadata_candidate_decision_limit` with default 15 and minimum 5, and all metadata candidate-list
  LLM prompts are bounded through that setting: BioOntology categorization, focused ontology
  selection, focused anatomy selection, and the downstream identity, etiology, patient
  applicability, and imaging workflow prompts. Added explicit Logfire spans for each focused agent
  so future runs show the slow component directly. Focused verification passed with
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py
  packages/findingmodel-ai/tests/test_ontology_search.py -q`: 59 passed, one Logfire no-config
  warning.
- 2026-05-08: v107 reran the same three-record Logfire diagnostic after applying the candidate cap
  globally:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-right-sized-v107-logfire-global-candidate-cap/run`.
  The run completed 3/3 with zero audit flags. Logfire showed focused ontology prompts with 5-9
  candidates and focused anatomy prompts with 15 candidates, while gathered anatomy candidates still
  ranged from 49-100. Downstream identity, etiology, patient-applicability, and imaging-workflow
  prompts dropped to roughly 2.8k-3.9k input tokens because they now receive bounded candidate
  context. Remaining runtime is concentrated in anatomy decisions over the 15-candidate set, not in
  uncapped candidate lists.
- 2026-05-08: Next cleanup target from Logfire: focused agents are retrying because optional
  `field_confidence` bookkeeping is too strict. Agents naturally emitted numeric scores or
  task-local confidence keys such as `canonical`, `breast`, or `selected_anatomy`; Pydantic AI
  retried even though the actual metadata/candidate decisions were usable. The next implementation
  step is to make confidence numeric/tolerant with low/medium/high computed by orchestration, and to
  audit focused output models so each prompt returns only fields consumed downstream.
- 2026-05-08: Implementing the lean-output cleanup by extracting confidence parsing and focused
  output models out of `assignment.py`. Numeric 0-1 confidence is the primary review shape; legacy
  labels may be accepted on input and converted. Ontology/anatomy focused outputs should not carry
  rationale or confidence because orchestration does not need them for assembly.
- 2026-05-08: Lean-output cleanup implemented and focused verification passed. Confidence parsing
  now lives in `metadata/confidence.py` and output models live in `metadata/decisions.py`, keeping
  `assignment.py` focused on orchestration. `field_confidence` is numeric 0-1 internally, accepts
  legacy labels/0-100 values, and ignores invalid optional confidence keys instead of retrying.
  Focused ontology/anatomy outputs no longer carry rationale/confidence fields, focused prompts no
  longer request rationale, and missing confidence no longer creates warnings or retry pressure.
  Verification:
  `uv run ruff check packages/findingmodel-ai/src/findingmodel_ai/metadata/confidence.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/decisions.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/types.py
  packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py
  packages/findingmodel-ai/tests/test_metadata_types.py` passed, and
  `uv run pytest packages/findingmodel-ai/tests/test_metadata_types.py
  packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q` passed with 47 tests and one
  Logfire no-config warning. The narrower assignment-only run also passed:
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q` passed with 37 tests and one
  Logfire no-config warning.
- 2026-05-08: Fresh wheelhouse v108 Logfire-instrumented seven-record dry run showed the schema
  cleanup worked mechanically but did not fully solve real-output quality. Six records succeeded and
  one failed. Logfire showed candidate caps behaving as intended: focused ontology prompts used 6-12
  candidates and focused anatomy prompts used 15 candidates even when gathered anatomy candidates
  ranged from 36-106. Focused agents generally completed in about 1-4.5 seconds each; no optional
  confidence-key validation retries were observed. Qualitative output was mixed: etiology nulling
  improved for `axillary_mass` and `breast_calcification_cluster`, `tunneled_catheter` kept
  `anterior chest wall`, and `radiolucent_urinary_calculus` still selected `urinary tract` when the
  hard validator was removed. Remaining quality problems include `arterial_stent` selecting named
  artery children alongside `arterial system`, `breast_density` selecting `accessory breast`, and
  broad modality/time-course overfill on some records. The one v108 failure was caused by a brittle
  hard validator that interpreted any selected anatomy display ending in "system" as whole-body
  anatomy; this misread "calyx of renal collecting system" and forced retries. That validator was
  removed and covered with a focused regression test. v109 reran `radiolucent_urinary_calculus`
  successfully with anatomy `urinary tract` only, zero warnings, zero audit flags, and Logfire trace
  `019e09a3d3e6b62588ea349711790331`. Verification after the validator removal:
  `uv run ruff check packages/findingmodel-ai/src/findingmodel_ai/metadata/confidence.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/decisions.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/assignment.py
  packages/findingmodel-ai/src/findingmodel_ai/metadata/types.py
  packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py
  packages/findingmodel-ai/tests/test_metadata_types.py` passed, and
  `uv run pytest packages/findingmodel-ai/tests/test_metadata_types.py
  packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q` passed with 48 tests and one
  Logfire no-config warning.
- 2026-05-08: Inspecting actual Logfire anatomy-agent calls for the remaining anatomy quality
  failures before making further prompt changes. `arterial_stent` offered `arterial system` plus
  named artery option values, and the agent incorrectly selected both the parent and the named
  arteries. `breast_density` offered only weak `search_only` anatomy candidates, including narrower
  or qualified breast parts, and the agent incorrectly selected `accessory breast` as a proxy for the
  broader breast scope. The next prompt edit will replace case-shaped instructions with a compact
  label-support rule: every qualifier in a selected candidate label must be supported by the modeled
  finding scope; when no offered candidate preserves the scope, leave anatomy unselected and use
  `body_regions` for the broad region.
- 2026-05-08: Added a focused anatomy-decision replay suite at
  `packages/findingmodel-ai/evals/metadata_anatomy_decision.py` so prompt tuning can exercise only
  the anatomy sub-agent against saved example payloads, without running search, orchestration,
  assembly, audit, or the other focused agents. Rewrote the anatomy prompt as a shorter selection
  contract instead of a long case-shaped rule list. The new prompt emphasizes minimal selected
  anatomy, candidate-label support, parent/system coverage over attribute option children, null
  anatomy over proxy selections, and placement/course handling for devices and catheters. Included
  two non-fixture examples: unsupported breast qualifiers and generic arterial attribute options.
  Focused replay now passes all four anatomy cases: generic arterial stent selects only
  `arterial system` with `whole_body`, breast density selects no proxy anatomy with `breast`,
  radiolucent urinary calculus selects only `urinary tract` with `abdomen`, and tunneled catheter
  selects `anterior chest wall` with `chest`. The replayed anatomy prompt input dropped from about
  2.6k tokens per case to about 1.5k tokens per case. Verification:
  `uv run ruff check src/findingmodel_ai/metadata/assignment.py
  evals/metadata_anatomy_decision.py` passed from `packages/findingmodel-ai`;
  `uv run --env-file ../../.env python -m evals.metadata_anatomy_decision` passed with
  SelectedCandidateEvaluator 1.00 and BodyRegionEvaluator 1.00; and
  `uv run pytest packages/findingmodel-ai/tests/test_assign_metadata.py
  packages/findingmodel-ai/tests/test_assign_metadata_modes.py -q` passed with 38 tests and one
  Logfire no-config warning.
- 2026-05-08: Adopt the focused anatomy-decision replay pattern for the other metadata sub-agents
  before another broad corpus run. For each sub-agent, use Logfire traces and reviewed outputs to
  build small replay payloads that call only that focused agent, then tune the prompt against those
  examples without running the full assignment pipeline. Prioritize sub-agents where qualitative
  review still shows field-quality problems: etiology overfill, modality/subspecialty overfill,
  identity/time-course preservation, and patient applicability overfill. The pattern should be:
  inspect actual calls, create a small focused replay suite, simplify the prompt around the real
  decision contract, verify the focused suite, then run a small mixed end-to-end dry run to confirm
  the focused improvement survives orchestration.
- 2026-05-08: Etiology focused pass started. Scope is intentionally narrow: inspect actual
  etiology-agent calls and reviewed output examples, build a replay eval that calls only the
  etiology sub-agent, simplify the etiology prompt around supported causal mechanism decisions, and
  verify the focused replay before any broader dry run. This pass must not add keyword-driven
  validators or finding-specific special cases.
- 2026-05-08: Added the focused etiology replay suite at
  `packages/findingmodel-ai/evals/metadata_etiology_decision.py` and tightened the etiology prompt.
  The prompt now states the decision boundary as concept identity versus possible causes; treats
  tags as weak support rather than stand-alone evidence; keeps generic masses, calcification
  patterns, fluid collections, soft-tissue abnormalities, and broad signs null unless the modeled
  concept itself names a causal class; prefers the most specific supported bucket instead of
  child-plus-parent duplicates; and avoids `neoplastic:potential` when benign/malignant/metastatic
  buckets already describe the concept. Focused replay passes all nine current cases: generic
  axillary mass null, differential-language rib-spacing finding null, nonspecific pericardial
  effusion null, tag-only mandibular notch null, urinary calculus metabolic, pulmonary embolism
  vascular thrombotic, aortic stent iatrogenic device, primary brain tumor benign/malignant
  neoplastic without metastatic or potential, and pneumonia infectious without the broad
  inflammatory parent. Verification so far: `uv run ruff check
  src/findingmodel_ai/metadata/assignment.py evals/metadata_etiology_decision.py` passed and
  `uv run --env-file ../../.env python -m evals.metadata_etiology_decision` passed with
  EtiologyEvaluator 1.00. Next step is targeted assignment tests and a small mixed end-to-end dry
  run to confirm the prompt change survives orchestration.
- 2026-05-08: Etiology focused pass follow-up completed after Logfire showed that one apparent
  etiology miss was actually caused upstream by ontology canonical selection. In v115/v116,
  `primary_brain_tumor` inherited `neoplastic:malignant` when the ontology agent accepted
  `Primary malignant neoplasm of brain` as canonical for the broader unqualified tumor concept.
  The ontology prompt now has a general qualifier-preservation rule: for unqualified tumor/neoplasm
  concepts, reject candidates that add benign, malignant, metastatic, premalignant,
  histologic-subtype, or grade qualifiers unless the modeled source itself includes that qualifier.
  Added focused ontology replay `packages/findingmodel-ai/evals/metadata_ontology_decision.py`;
  it calls only the ontology sub-agent and verifies this semantic rule without hard-coded
  assignment logic.
- 2026-05-08: The etiology prompt now also separates urinary/biliary calculus or stone from
  calcification and microcalcification imaging appearances. `calculus`/`stone` can support
  `metabolic`; calcification, mineralization, microcalcification, or calcification cluster does not
  support `metabolic` by itself. Added a focused breast-calcification replay case based on the
  actual Logfire payload.
- 2026-05-08: Verification after the etiology/ontology focused pass:
  `uv run ruff check src/findingmodel_ai/metadata/assignment.py
  evals/metadata_etiology_decision.py evals/metadata_ontology_decision.py` passed;
  `uv run --env-file ../../.env python -m evals.metadata_etiology_decision` passed with
  EtiologyEvaluator 1.00 across ten cases; `uv run --env-file ../../.env python -m
  evals.metadata_ontology_decision` passed with OntologyEvaluator 1.00; and
  `uv run --env-file ../../.env pytest tests/test_config.py tests/test_assign_metadata.py
  tests/test_assign_metadata_modes.py -q` passed with 94 tests and one Logfire no-config warning.
  Rebuilt the wheelhouse used by `/Users/talkasab/repos/findingmodels-metadata/scripts/
  metadata_assign_batch.py`.
- 2026-05-08: v117 mixed end-to-end dry run used the fallback-capable project `.env`, completed
  12/12 dry-run records with zero deterministic audit flags, and showed the intended etiology
  behavior on the reviewed cases: `axillary_mass`, `breast_calcification_cluster`,
  `pericardial_effusion`, `widening_of_rib_interspaces`, and `antegonial_notching_of_the_mandible`
  had null etiologies; `radiolucent_urinary_calculus` was `metabolic`; `pulmonary_embolism` was
  `vascular:thrombotic`; `aortic_stent` and `tunneled_catheter` were `iatrogenic:device`;
  `pneumonia` was `inflammatory:infectious`; `malignant_primary_bone_neoplasm` was
  `neoplastic:malignant`; and `primary_brain_tumor` was `neoplastic:benign` plus
  `neoplastic:malignant` with only the broad source code retained. Remaining quality issues in this
  smoke are outside the etiology pass: anatomy/body-region over-selection for `axillary_mass`,
  workflow/subspecialty overreach on examples such as mandibular notching and pulmonary embolism,
  and time-course overfill on some chronic-looking findings. Next focused-agent target should be
  imaging workflow/subspecialty and modality behavior, using the same Logfire-to-focused-replay
  pattern.
- 2026-05-08: Checked enum alignment after workflow tuning. The active structured-output schemas
  already used the canonical enum values for body regions, subspecialties, etiologies, entity type,
  modalities, patient sex, age stages, and time-course values. The gap was clarity, not mismatched
  values: most focused output fields had no schema descriptions. Added concise field-level schema
  descriptions in `packages/findingmodel-ai/src/findingmodel_ai/metadata/decisions.py` so the
  sub-agents see both allowed enum values and short meanings without repeating long enum glossaries
  in every prompt. Also confirmed the `metadata_assign` fallback chain is
  `openai:gpt-5.4-mini` with no reasoning, `google-gla:gemini-3.1-flash-lite-preview`, then
  `anthropic:claude-sonnet-4-6`.
- 2026-05-08: Imaging workflow focused pass completed for the two quality problems seen in v117:
  ordinary presence/change attributes should not create `SQ`, and vascular embolus/thrombus models
  should not inherit regional workflow labels or indirect/source-workup modalities. Added
  `packages/findingmodel-ai/evals/metadata_imaging_workflow_decision.py`, which calls only the
  imaging workflow sub-agent. The workflow prompt now defines applicable modalities as the smallest
  default routine direct set, distinguishes diagnostic use from mere detectability, treats
  vessel-centered models as vascular workflow without automatic regional labels, and makes `SQ`
  available only for acquisition/artifact/quality/safety/dose/report-quality/technique issues.
  The production workflow payload now removes source tags before this sub-agent runs and passes only
  selected canonical ontology plus selected anatomy, instead of asking the model to ignore noisy
  tags and review-only candidates. Focused replay expectations are behavior-oriented: they require
  the supported workflow values and forbid the known overreach classes, rather than asserting one
  arbitrary exact modality list when a narrow variation is acceptable.
- 2026-05-08: Stabilized focused metadata-agent calls by setting `temperature=0` on the metadata
  assignment agents themselves. This is scoped to metadata classification agents and does not change
  unrelated generation agents. Verification after the enum/schema and workflow pass:
  `uv run ruff check src/findingmodel_ai/metadata/decisions.py
  src/findingmodel_ai/metadata/assignment.py evals/metadata_imaging_workflow_decision.py
  evals/metadata_etiology_decision.py evals/metadata_ontology_decision.py` passed;
  `uv run --env-file ../../.env python -m evals.metadata_imaging_workflow_decision` passed with
  ImagingWorkflowEvaluator 1.00; `uv run --env-file ../../.env python -m
  evals.metadata_etiology_decision` passed with EtiologyEvaluator 1.00 across ten cases;
  `uv run --env-file ../../.env python -m evals.metadata_ontology_decision` passed with
  OntologyEvaluator 1.00; and `uv run --env-file ../../.env pytest tests/test_config.py
  tests/test_assign_metadata.py tests/test_assign_metadata_modes.py -q` passed with 94 tests and
  one Logfire no-config warning. Next step is to rebuild the wheelhouse and run a small mixed
  end-to-end smoke focused on the v117 workflow failures before deciding whether to rerun a larger
  corpus dry run.
- 2026-05-08: v118/v119 smoke follow-up found that the workflow and etiology focused changes are
  mechanically holding, but one v119 `aortic_stent` record timed out during the second focused
  classifier stage. Logfire showed ontology and anatomy completed normally; then identity, patient,
  and imaging-workflow each opened an OpenAI Responses request with no HTTP status or model response
  before the outer record timeout cancelled the record. Focused local replays of the same aortic
  payload, both sequentially and with the same three-agent concurrency pattern, completed normally,
  so the evidence does not support a prompt/schema mismatch or `gpt-5.4-mini` availability failure.
  A public per-provider timeout setting was considered and removed as unnecessary surface area for
  the current evidence. Continue to use the batch record timeout as the outer safety guard unless
  repeated Logfire traces prove provider-request stalls are a recurring failure mode.
- 2026-05-09: Readiness path tightened to avoid another loop of tiny-smoke overfitting. The
  8-record smoke set is no longer the primary readiness signal; it is useful only for mechanical
  regressions around familiar examples. The next work cycle is: keep focused replay evals for
  anatomy, ontology, etiology, and imaging workflow green; rebuild the wheelhouse; run a
  representative larger dry run; and only make further prompt changes for repeated high-impact
  failure patterns seen in that larger run. Do not add brittle validators, candidate-ID exclusions,
  finding-name keyword rules, or public transport knobs without repeated evidence.
- 2026-05-09: v122 representative 200-record dry run completed with 169 first-pass successes, 31
  first-pass failures, and zero deterministic audit flags on successful records. Failure triage
  showed 29 DNS/provider connection errors, one DuckDB WAL "too many open files" cache error, and
  one repeated semantic failure (`hyperphosphatasemia`) where the tool could not assign required
  `applicable_modalities` for a lab/clinical concept. v123 reran the 31 failed records with
  concurrency 1 and retries enabled: 30 succeeded with zero deterministic audit flags, leaving only
  `hyperphosphatasemia` failing again for the same required-modality reason. This confirms the
  transport/resource failures were operational noise, while `hyperphosphatasemia` is the only
  repeated product-shape failure from this pass.
- 2026-05-10: The 400-record non-GMTS plus filtered-GMTS review pass exposed five remaining
  assessment/measurement failures from the ontology sanity check that required every canonical
  ontology decision on assessment/measurement models to be labeled `exact_match`. This check caught
  real overreach in several CDE-style measurement modules, but it also failed valid cases such as
  `pediatric_bone_age` where `RADLEX:RID39030` and `SNOMEDCT:123980006` are appropriate matches.
  Implementation plan: keep the sanity behavior, but make it corrective instead of fatal. For
  assessment/measurement outputs, downgrade non-exact canonical ontology decisions to review
  candidates with warnings; preserve exact source codes and selected exact/substitutable candidates;
  and let the record complete unless required fields are still missing.
- 2026-05-10: Added the supervised writeback workflow to this plan. Larger dry runs should now be
  treated as proposed metadata batches with subagent queue labels: `proposed_accept`,
  `proposed_skip`, `needs_attention`, or `suspected_tool_problem`. Human reviewers choose only
  `approved`, `skipped`, or `feedback`. Only human-approved artifacts should be promoted to source
  files; skipped and feedback records remain untouched and feed either human review or the next
  focused prompt-improvement cycle.
- 2026-05-10: Added the standalone supervised-review prompt/context document at
  `docs/plans/metadata-enrichment-supervised-review-prompt-2026-05-10.md`.
- 2026-05-10: Implemented the supervised review handoff for the 400-record
  non-GMTS plus filtered-GMTS batch. Eight subagent review batches were written as JSON artifacts
  under `/tmp/metadata-review-subagent-outputs/`, collated into
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/review-decisions.json`,
  and rendered into the human review app at
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/review/index.html`.
  The collated triage is 252 `proposed_accept`, 100 `needs_attention`, 38 `proposed_skip`, and
  10 `suspected_tool_problem`. No source metadata files should be promoted until the exported
  human review decisions mark records as `approved`; `skipped` and `feedback` records remain
  writeback-ineligible.
