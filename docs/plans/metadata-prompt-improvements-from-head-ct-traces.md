# Plan: Metadata Prompt Improvements From Head CT Trace Review

Status: Proposed (2026-04-03, updated after completed rerun review)

## Goal

Use the Logfire trace review from the head CT findings export to tighten the generic metadata-assignment prompts so they are:

- more conservative when evidence is weak
- less likely to drift outside the intended anatomic scope
- less likely to over-assert etiologies, chronicity, or modality coverage
- more reliable in structured output formatting

This plan is about improving the reusable prompt layer in the metadata pipeline, not just patching one CSV export.

## Why This Plan Exists

The head CT rerun showed that the pipeline is instrumented correctly and the span structure is healthy, but the prompt behavior is still inconsistent for curated source lists.

Observed trace patterns:

- some rows were strong or close to strong, such as `intracranial aneurysm coil`
- some rows recovered to acceptable outputs despite weak intermediate evidence, such as `missing tooth`
- some rows were clearly over-assertive, such as `tonsillolith` getting `degenerative`
- some rows drifted toward the wrong body region because the anatomic candidate path forced a weak choice, such as `subcutaneous emphysema`
- the classifier still frequently returns invalid numeric `field_confidence` values on the first pass and succeeds only after validator retry

Observed completed-rerun output patterns:

- `10` rows have no `body_regions`
- `34` rows have no `applicable_modalities`
- `25` rows are explicitly negative findings such as `No acute intracranial abnormality`
- `4` rows were classified as `technique_issue`
- `1` row was classified as `assessment`
- `1` row was classified as `grouping`
- warning-bearing rows include:
  - `Arterial infundibulum` drifting to `chest`, `CA|CH`, and `CT|MR|US|XR`
  - `Post-surgical intracranial change` becoming `grouping`
  - `No postoperative complication` producing a `clear_fields` warning

## Scope

In scope:

- prompt improvements in the generic metadata classifier
- prompt improvements in the anatomic query and anatomic selection agents
- payload improvements that provide bounded source context to the classifier
- validation focused on representative failure cases seen in Logfire
- documentation updates reflecting the new prompt behavior and known limits

Out of scope:

- redesigning the ontology search pipeline
- changing the underlying enum schema
- final cleanup of the head CT output CSV
- hierarchy reconstruction for the head CT finding list

## Proposed Prompt Improvements

### 1. Add Explicit Source Context To The Classifier Prompt

Current issue:

- the classifier sees the finding model and candidate evidence, but not enough bounded context about the source collection
- this makes it easier to treat a curated source list as if it were a generic whole-library assignment task

Proposed change:

- extend the classifier payload with a small `source_context` block when available
- include only stable, bounded hints, such as:
  - source collection name
  - known anatomic scope
  - known modality bias
  - whether rows come from a curated hierarchical list

Expected benefit:

- reduces context loss when a finding label is understandable only relative to the source collection
- helps the classifier avoid unnecessary whole-body reasoning for scoped datasets

### 2. Make The Anatomic Query Agent Prefer Null Over Generic Whole-Body Fallbacks

Current issue:

- the anatomic query agent sometimes returns `Body` when the location is unclear
- this creates weak or misleading intermediate evidence that downstream steps still try to use

Proposed change:

- revise the anatomic query prompt to say:
  - if no specific region is well supported, return `null`
  - do not use `Body` as a generic fallback
  - prefer scoped source context over generic whole-body reasoning when provided

Expected benefit:

- fewer bad candidate searches
- less anatomy drift from weak intermediate context

### 3. Let The Anatomic Selection Agent Reject Weak Candidate Sets

Current issue:

- the selection prompt currently encourages choosing the “best” candidate even when the result set is poor
- this produced false precision in traces like `subcutaneous emphysema`

Proposed change:

- revise the anatomic selection prompt so it can explicitly decline to select a primary location when:
  - all candidates are weak
  - candidates are cross-region and none is clearly primary
  - the available candidates conflict with source context

Expected benefit:

- reduces false anatomic specificity
- prevents a weak candidate set from forcing a misleading body-region assignment

### 4. Tighten Etiology And Time-Course Instructions

Current issue:

- the classifier sometimes converts plausibility into metadata
- examples from the trace review include:
  - `tonsillolith` -> `degenerative`
  - `subcutaneous emphysema` -> `days`, `stable`

Proposed change:

- revise the classifier prompt to require direct support before setting:
  - `etiologies`
  - `expected_time_course`
  - `age_profile`
- explicitly instruct:
  - do not set these fields just because a value is common
  - leave the field blank if the support is indirect, weak, or generic

Expected benefit:

- more conservative metadata
- fewer fabricated chronicity and etiology assignments

### 5. Tighten Modality Guidance

Current issue:

- the classifier sometimes expands to every technically plausible modality rather than the reasonable clinical set

Proposed change:

- revise modality guidance to emphasize:
  - choose modalities that meaningfully evaluate the finding in routine practice
  - do not include modalities merely because the finding could theoretically be seen there
  - use source-context modality bias when available

Expected benefit:

- less modality inflation
- outputs that match how the library is intended to be used

### 6. Make Structured Output Constraints More Explicit

Current issue:

- the classifier often first returns numeric values for `field_confidence`
- it also treats some non-field concepts as eligible keys

Proposed change:

- add explicit instructions that:
  - `field_confidence` values must be exactly `high`, `medium`, or `low`
  - numeric scores are invalid
  - only actual metadata fields belong in `field_confidence`
  - explanatory fields like `classification_rationale` should not appear there unless intentionally supported by schema

Expected benefit:

- fewer validator retries
- cleaner and more stable structured outputs

### 7. Add Negative Examples, Not Just Positive Guidance

Current issue:

- the prompt explains what the fields mean, but does not show enough examples of overreach to avoid

Proposed change:

- add short negative examples to the classifier prompt, such as:
  - `tonsillolith` does not automatically imply `degenerative`
  - `missing tooth` does not automatically imply `congenital`
  - `subcutaneous emphysema` does not automatically imply `chest`
  - implanted device findings should not be collapsed into procedure concepts

Expected benefit:

- better calibration on edge cases
- lower rate of plausible-but-unsupported metadata

### 8. Add Explicit Guidance For Negative Findings

Current issue:

- the completed rerun contains a meaningful bucket of negative statements such as:
  - `No acute stroke`
  - `No intracranial mass`
  - `No postoperative complication`
- the generic prompt does not clearly distinguish negative findings from positive findings, diagnoses, or assessments

Proposed change:

- add negative-finding guidance to the classifier prompt:
  - negative statements should usually stay `finding` unless they are clearly a higher-level interpretive judgment
  - avoid inventing body regions, etiologies, or chronicity when the statement is only an absence claim
  - do not convert negative findings into `assessment` unless the text is truly report-level judgment rather than finding-library content

Expected benefit:

- more consistent handling of `No ...` rows
- fewer empty-or-awkward outputs for negative statements

### 9. Add Explicit Guidance For Technique / Artifact Rows

Current issue:

- artifact and technical-quality rows are currently mixed:
  - some get reasonable `technique_issue`
  - some get empty body-region/subspecialty combinations
  - some get very broad modality coverage without enough explanation

Proposed change:

- extend classifier guidance for artifact/technique rows:
  - `technique_issue` should be preferred for acquisition or reconstruction quality problems
  - body region may intentionally remain blank when artifact is not anatomy-specific
  - modalities should be limited to the modalities actually implicated by the artifact or technique issue

Expected benefit:

- cleaner behavior on non-anatomic rows
- less over-broad modality output for artifact cases

### 10. Add Guidance For Variant / Device / Postoperative-State Rows

Current issue:

- device and postoperative rows are often among the stronger outputs, but there is still confusion between:
  - the device or postoperative state
  - the underlying treated disease
  - the procedure concept
  - broad grouping labels

Proposed change:

- extend classifier guidance to separate:
  - implanted device finding
  - postoperative state or expected change
  - procedure concept
  - broad grouping bucket
- add examples drawn from the rerun:
  - `intracranial aneurysm coil` should remain a device finding, not a procedure
  - `post-surgical intracranial change` should not become `grouping` unless the authored model is truly a grouping concept

Expected benefit:

- fewer device/procedure conflations
- better handling of postoperative expected-state findings

## Concrete Workstreams

### Workstream A: Structured Output Reliability

Target problems:

- numeric `field_confidence` retries
- non-field keys appearing in `field_confidence`

Changes:

- tighten classifier instructions
- add explicit bad-output examples
- consider validator messages that point directly to the allowed keys and enum literals

Validation set:

- `missing tooth`
- `intracranial aneurysm coil`

### Workstream B: Anatomy Drift Control

Target problems:

- `Body` fallback in the anatomic query path
- forced selection from weak candidate sets
- wrong-region results such as `Arterial infundibulum -> chest`

Changes:

- update anatomic query prompt
- update anatomic selection prompt
- add source-context support

Validation set:

- `subcutaneous emphysema`
- `arterial infundibulum`
- `ventricular abnormality`

### Workstream C: Conservative Metadata Gating

Target problems:

- unsupported etiologies
- unsupported time-course metadata
- overly broad modality lists

Changes:

- tighten classifier instructions for `etiologies`, `expected_time_course`, `age_profile`, and `applicable_modalities`
- add negative examples showing when blanks are preferred

Validation set:

- `tonsillolith`
- `normal soft tissues`
- `no vascular abnormality`

### Workstream D: Special-Case Content Classes

Target problems:

- negative statements
- artifact / technique rows
- postoperative / device / expected-state rows

Changes:

- add prompt guidance and examples for:
  - negative findings
  - technique issues
  - device findings
  - postoperative expected states

Validation set:

- `no postoperative complication`
- `motion artifact`
- `metallic artifact`
- `post-surgical intracranial change`
- `intracranial aneurysm coil`

## Implementation Plan

1. Update this plan as prompt changes are made so the repository reflects the intended direction.
2. Implement Workstream A in the classifier prompt and validator guidance.
3. Implement Workstream B in the anatomic query and selection prompts.
4. Add any bounded `source_context` plumbing needed so export jobs can pass scoped hints without rewriting the generic pipeline.
5. Implement Workstream C in the classifier prompt.
6. Implement Workstream D using small targeted examples rather than broad prompt expansion.
7. Re-run a representative validation set that includes:
   - `intracranial aneurysm coil`
   - `missing tooth`
   - `tonsillolith`
   - `subcutaneous emphysema`
   - `arterial infundibulum`
   - `no postoperative complication`
   - `motion artifact`
8. Use Logfire traces to review:
   - prompt payloads
   - intermediate ontology/anatomic outputs
   - classifier decisions
   - validator retry frequency
9. Compare the before/after CSV behavior specifically for the completed-rerun failure clusters:
   - missing body region
   - missing modality
   - warning-bearing rows
   - negative rows
10. Update this plan with the observed before/after behavior.
11. Review whether related docs or changelog entries need updating once the prompt changes are complete.

## Validation Criteria

The prompt work should count as successful only if:

- clearly weak evidence is more often left blank instead of being converted into metadata
- anatomic drift decreases on representative edge cases
- modality lists become narrower and more defensible where appropriate
- first-pass structured outputs produce fewer validator retries
- the improved behavior is visible in Logfire traces, not just inferred from final CSV rows
- the known bad rerun cases move in the intended direction:
  - `Arterial infundibulum` no longer drifts to chest/cardiac metadata
  - `tonsillolith` no longer gets unsupported chronicity or etiology
  - `No postoperative complication` no longer emits the current warning path
  - negative findings are handled more consistently as negative findings rather than awkward empty shells

## Documentation Follow-Up

When implementation starts or lands:

- update this plan with concrete edits and validation notes
- update `docs/plans/head-ct-findings-metadata-csv.md` if the prompt changes materially alter how that export should be interpreted
- review whether user-facing `CHANGELOG.md` entries are warranted based on externally visible behavior changes
