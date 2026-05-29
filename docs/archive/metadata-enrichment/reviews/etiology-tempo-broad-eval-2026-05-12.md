# Etiology Tempo Broad Eval Review

## Prompt-Master Distillation

- Target tool: Pydantic AI structured-output agent.
- Source material: current etiology prompt plus the observable-duration rules embedded in the
  identity prompt.
- Output contract: `etiologies`, `expected_time_course`, optional `field_confidence`.
- Prompt constraint: concise rules, no chain-of-thought, no `clear_fields`.

## Policy

- Etiology means common cause or disease class radiologists would reasonably carry for the finding,
  not all theoretically possible causes.
- Expected time course means whether a finding seen on prior imaging should still be expected on a
  later study.
- For expected time course, choose the long end of common observable persistence, not median
  clinical recovery and not rare outliers.
- Etiology rules should be general and ordered by importance. Case-specific gold alignment belongs
  in eval expectations and review notes, not as inline prompt exceptions.
- Existing gold fixtures are source material, but focused review can revise expected results where
  this policy is clearer.

## Pilot Surface

- Prompt: `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/etiology_tempo.md`
- Decision model: `EtiologyTempoDecision`
- Agent factory: `create_etiology_tempo_agent`
- Eval suite: `packages/findingmodel-ai/evals/metadata_etiology_tempo_decision.py`

## Eval Coverage

The pilot eval covers generic-null cases, common-cause fluid/nodule/lymph-node findings, vascular
events, neoplasm buckets, infection/inflammation, trauma, degeneration, stones, devices, artifacts,
assessments, and follow-up persistence expectations from days through permanent.

## Results

- Earlier checkpoint: `metadata_etiology_tempo_decision.py` reached 20/20 exact matches before
  expanding gold-backed cases.
- Current checkpoint: expanded gold-backed eval coverage is in place, but exact expected outputs
  need review against the long-end-of-common-time-course policy before this should be treated as a
  green suite.
- `breast_density_has_tempo_but_no_etiology`: 5/5 repeat samples after tightening the
  normal-variant guardrail.
- Prompt length after removing case-spam exceptions: 63 lines, 563 words.
- Neighboring pilot checks after implementation:
  - `metadata_subspecialty_domain_decision.py`: 46/46.
  - `metadata_modality_applicability_decision.py`: 29/29 after tightening aortic dissection to
    exclude RF unless a fluoroscopic procedure is named.
