# Metadata Enrichment Supervised Review Prompt

> Superseded for active execution by `docs/plans/metadata-enrichment-current-plan.md`. Keep this file as historical evidence only; pull any still-useful decisions into the active plan or a stable reference doc before acting on them.



Status: Historical prompt/context artifact; superseded for active execution
Date: 2026-05-10

## Purpose

This document defines the subagent review task for metadata-enrichment dry runs. It is the source of
truth for the prompt used to pre-triage generated metadata before human review and source writeback.

The review task is not metadata assignment. Reviewers decide whether generated dry-run artifacts are
safe to promote, should be skipped, or need feedback before writeback.

## Current Batch Context

- Source repo: `/Users/talkasab/repos/findingmodels-metadata`
- Tool repo: `/Users/talkasab/repos/findingmodel-metadata`
- Current dry run:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/run`
- Current review data:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/review-data/review-data.json`
- Current review page:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/review/index.html`

For each assigned record, the reviewer should inspect:

- `defs/<stem>.fm.json`
- `<run>/before-after/<stem>.before.json`
- `<run>/before-after/<stem>.after.json`
- `<run>/reviews/<stem>.metadata-review.json`
- `<run>/audits/<stem>.audit.json`, when present

## Triage Labels

Subagent triage labels organize the human review queue. They are not human button labels.

- `proposed_accept`: the source concept is in scope and the enriched metadata appears safe to
  write back.
- `proposed_skip`: the source concept should not be enriched in this pass.
- `needs_attention`: the concept is probably in scope, but one or more fields need human feedback
  before writeback.
- `suspected_tool_problem`: the output suggests a repeated prompt/tool behavior that should be
  considered before promoting similar records.

Human review actions remain intentionally simple:

- `approved`: eligible for source writeback.
- `skipped`: excluded from this enrichment pass.
- `feedback`: not applied yet; needs correction, domain input, or prompt/tool follow-up.

## Review Standards

In-scope records are single imaging-observable finding models, diagnoses, measurements,
assessments, devices, technique issues, or recommendations that radiologists reasonably see,
measure, assess, or describe in imaging workflows.

Skip records that are not suitable for this enrichment pass:

- clinical scales without direct imaging-finding semantics, such as Glasgow Coma Scale;
- lab-only or clinical-only concepts;
- templates, protocols, report sections, or CDE-style modules rather than one modeled finding;
- broad workflow tasks rather than an image-observable finding, diagnosis, measurement, assessment,
  device, technique issue, or recommendation.

Field standards:

- `index_codes`: must represent the modeled concept, not merely anatomy, target site, procedure,
  imaging method, view, component, or a related differential. Qualifiers in the source concept must
  be preserved.
- `anatomic_locations`: must be supported by the modeled concept. Do not accept unsupported
  narrowing to a specific lobe, rib, vessel segment, lymph node, organ subpart, or nearby landmark.
  If a parent anatomy cleanly covers all involved parts, choose the parent.
- `body_regions`: must follow selected anatomy. Use `whole_body` only for true system-level or
  nonlocalized findings, not for named organs or named vessels.
- `etiologies`: should describe intrinsic or common causal mechanism of the modeled concept. Do not
  assign etiology from differential diagnosis, downstream concern, source tags alone, possible
  complications, or weak clinical association.
- `applicable_modalities`: should be routine direct ways to evaluate the modeled finding, not every
  modality where it might be visible.
- `subspecialties`: should reflect reading workflow, not organ membership alone.
- `entity_type`: should distinguish finding, diagnosis, measurement, assessment, device,
  technique_issue, recommendation, and grouping.
- `age_profile`, `sex_specificity`, and `expected_time_course`: should be filled only when supported
  by the modeled concept.

Do not flag a record for a mere preference difference. Flag it only when the source or output gives
concrete evidence that a field is unsupported, too broad, too narrow, or otherwise unsafe to write.

## Examples From Current Review

Known skip examples:

- `glasgow_coma_scale__gcs_`: clinical neurologic score, not an imaging finding.
- `mr_rectal_tumor_imaging`: template/protocol-like record, not a single finding.
- `flow_in_the_ascending_aorta`: CDE-style measurement module, not a clean finding model for this
  pass.
- `low_hairline`: clinical/cosmetic observation, not imaging-observable metadata for this pass.

Known carry-forward-with-feedback examples:

- `pulmonary_mass`: valid finding, but neoplastic etiologies need review because source language
  raises concern/differential rather than proving intrinsic etiology.
- `annular_calcifications`: valid concept, but code/anatomy mismatch needs review.
- `colonic_diverticulosis`: valid concept, but `normal-variant` etiology is not supported.
- `pulmonary_nodule`: valid concept, but unsupported anatomy narrowing should be flagged.
- `rib_destruction`: valid concept, but unsupported selection of specific ribs should be flagged.
- `epidermal_inclusion_cyst`: valid concept, but `normal-variant` etiology is not supported.

Known good behavior examples:

- `radiolucent_urinary_calculus`: anatomy should be `urinary tract`, not separate child parts.
- `tunneled_catheter`: anterior chest wall can be appropriate for device course/entry.
- `arterial_stent`: generic arterial system can be appropriate when source does not name a specific
  artery.

## Subagent Prompt

Use this prompt for each subagent. Replace `<RUN_DIR>` and `<ASSIGNED_RECORDS>` before dispatch.

```text
You are reviewing metadata-enriched finding models before source writeback.

Your task is to review only the assigned records and decide whether each generated dry-run artifact
is safe to promote, should be skipped, or needs human feedback. Do not edit files. Do not rewrite the
models. Do not tune prompts. Do not propose architecture changes. Inspect the actual artifacts and
cite concrete evidence.

Context:
- Source repo: /Users/talkasab/repos/findingmodels-metadata
- Tool repo: /Users/talkasab/repos/findingmodel-metadata
- Dry-run directory: <RUN_DIR>
- Assigned records: <ASSIGNED_RECORDS>

For each assigned record, inspect:
- defs/<stem>.fm.json
- <RUN_DIR>/before-after/<stem>.before.json
- <RUN_DIR>/before-after/<stem>.after.json
- <RUN_DIR>/reviews/<stem>.metadata-review.json
- <RUN_DIR>/audits/<stem>.audit.json, if present

Classify each record into exactly one triage category:
- proposed_accept: the source concept is in scope and the enriched metadata appears safe to write
  back.
- proposed_skip: the source concept should not be enriched in this pass, such as a clinical scale,
  lab-only concept, template, protocol, module, report section, or non-single finding.
- needs_attention: the concept is probably in scope, but one or more fields need human feedback
  before writeback.
- suspected_tool_problem: the output suggests a repeated prompt/tool behavior that may affect other
  records and should be considered before promoting similar records.

Review standards:
- index_codes must represent the modeled concept, not just anatomy, target site, procedure,
  measurement method, imaging view, component, or related differential.
- anatomy must be supported by the source concept. Do not accept unsupported narrowing to a specific
  lobe, rib, vessel segment, lymph node, organ subpart, or nearby landmark.
- if a parent anatomy cleanly covers the modeled scope, prefer the parent over separate child parts.
- body_regions must follow selected anatomy.
- etiologies should describe intrinsic/common cause of the modeled finding, not differential
  diagnosis, downstream concern, source tags alone, possible complications, or weak association.
- modalities should be routine direct ways to evaluate the modeled finding, not every modality where
  it might be visible.
- subspecialties should reflect reading workflow, not organ membership alone.
- entity_type should distinguish finding, diagnosis, measurement, assessment, device,
  technique_issue, recommendation, and grouping.
- age, sex, and time course should be filled only when supported by the modeled concept.

Known skip examples:
- glasgow_coma_scale__gcs_: clinical neurologic score, not an imaging finding.
- mr_rectal_tumor_imaging: template/protocol-like record, not a single finding.
- flow_in_the_ascending_aorta: CDE-style measurement module, not a clean finding model for this pass.
- low_hairline: clinical/cosmetic observation, not imaging-observable metadata for this pass.

Known feedback examples:
- pulmonary_mass: valid finding, but neoplastic etiologies need review if source language only
  raises concern/differential.
- annular_calcifications: valid concept, but code/anatomy mismatch needs review.
- colonic_diverticulosis: valid concept, but normal-variant etiology is not supported.
- pulmonary_nodule: valid concept, but unsupported anatomy narrowing should be flagged.
- rib_destruction: valid concept, but unsupported selection of specific ribs should be flagged.

Do not mark a record bad for mere preference differences. Mark it only when you can explain what
field is unsupported or why the source concept is out of scope.

Output JSON only:

{
  "records": [
    {
      "id": "<record stem>",
      "path": "defs/<record>.fm.json",
      "triage_category": "proposed_accept | proposed_skip | needs_attention | suspected_tool_problem",
      "recommended_human_action": "approve | skip | provide_feedback",
      "field_flags": ["index_codes", "anatomic_locations", "etiologies"],
      "reason": "<one or two plain-English sentences>",
      "evidence": [
        "<specific source/output evidence>",
        "<specific source/output evidence>"
      ],
      "reviewer_notes": "<optional concise note>"
    }
  ],
  "patterns": [
    {
      "pattern": "<recurring issue, if any>",
      "record_ids": ["<record>", "<record>"],
      "suggested_next_action": "human_review | source_fix | prompt_tuning | no_action",
      "why": "<plain-English explanation>"
    }
  ]
}
```

## Collation Rules

- Subagent outputs are collated into one `review-decisions.json`.
- Duplicate record IDs are rejected rather than merged silently.
- Missing `recommended_human_action` may be inferred from triage category:
  - `proposed_accept` -> `approve`
  - `proposed_skip` -> `skip`
  - `needs_attention` -> `provide_feedback`
  - `suspected_tool_problem` -> `provide_feedback`
- Human approval, skip, and feedback decisions override subagent recommendations for promotion.
- Only human `approved` records are eligible for source writeback.

## Current Batch Artifacts

For the 400-record non-GMTS plus filtered-GMTS review batch:

- subagent batch lists: `/tmp/metadata-review-subagent-batches/batch-01.txt` through
  `/tmp/metadata-review-subagent-batches/batch-08.txt`
- subagent output JSON: `/tmp/metadata-review-subagent-outputs/batch-01.json` through
  `/tmp/metadata-review-subagent-outputs/batch-08.json`
- collated decision file:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/review-decisions.json`
- human review app:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/review/index.html`
- review data:
  `/Users/talkasab/repos/findingmodels-metadata/.metadata-runs/phase6-nongmts-gmts-review-v1/review-data/review-data.json`

Current triage counts: 252 `proposed_accept`, 100 `needs_attention`, 38 `proposed_skip`, and
10 `suspected_tool_problem`.
