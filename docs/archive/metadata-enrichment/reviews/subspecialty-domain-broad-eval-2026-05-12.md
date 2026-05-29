# Subspecialty Domain Broad Eval Review

## Scope

Pilot prompt: `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/subspecialty_domain.md`

Focused eval: `packages/findingmodel-ai/evals/metadata_subspecialty_domain_decision.py`

## Current Prompt Size

- 49 lines
- 441 words
- 757 `gpt-4o` tokens

## Eval Set

The broad pilot suite now has 46 curated cases covering:

- all subspecialty codes;
- anatomic/body-region traps including coronary/chest, breast/chest, cervical/head-neck, rib/chest/MSK, and abdomen/GI-vs-vascular;
- horizontal overlays including VA, OI, SQ, ER, IR, MI, NM, and PD;
- cases where body region alone should not drive the domain;
- cases where a mass, calcification, or lesion can reasonably carry an oncologic overlay.

Required-label coverage:

| Code | Required Cases |
|---|---:|
| BR | 1 |
| CA | 2 |
| CH | 7 |
| ER | 4 |
| GI | 7 |
| GU | 5 |
| HN | 3 |
| IR | 3 |
| MI | 1 |
| MK | 4 |
| NM | 2 |
| NR | 4 |
| OB | 4 |
| OI | 3 |
| PD | 2 |
| SQ | 2 |
| VA | 7 |

OI is now exercised both as an allowed overlay and as a required domain for metastatic disease,
known malignancy staging, and tumor treatment response.

## Human-Reviewed Policy Adjustments

- Cardiac findings commonly assessed on routine chest imaging may be both CA and CH.
- Mass, lesion, or calcification findings may carry OI when clinically reasonable, even when malignancy is not explicit.
- Horizontal domains can overlay anatomic domains.
- Abdominal or pelvic vessels should remain VA unless organ disease is also modeled.
- `clear_fields` is not part of the focused subspecialty-domain output contract.

## Prompt Changes From Broad Eval Failures

- Added `IR` positive mapping for image-guided procedures, biopsy, ablation, embolization, drains, catheters, and stents.
- Added abdominal/pelvic vessel guardrail to prevent abdomen -> GI or pelvis -> GU by body region alone.
- Added rib-fracture mapping to preserve both CH and MK.
- Added torsion to ER mapping.
- Added central venous catheter mapping to preserve both IR and VA.
- Tightened PD wording to require explicit pediatric context or pediatric-specific concepts, while
  allowing PD for testicular torsion as a clinically reasonable optional overlay.
- Added OI-required fixtures for metastatic liver lesions, lung cancer staging, and tumor treatment
  response.
- Added dissection to ER mapping after acute aortic dissection undercalled ER.
- Added ER exclusions for aneurysm/mass/lesion without acute complication, including a specific
  scrotal/testicular mass rule.

## Repeat Sampling

Command:

```bash
uv run --env-file ../../.env python -m evals.metadata_subspecialty_domain_decision \
  --repeats 8 \
  --sample-case cervical_lymphoid_tissue_is_head_neck \
  --sample-case central_venous_catheter_malposition_is_ir_vascular_chest_possible \
  --sample-case testicular_torsion_is_gu_emergency_vascular_possible
```

Result:

- `cervical_lymphoid_tissue_is_head_neck`: 8/8 passed; observed `HN, OI`.
- `central_venous_catheter_malposition_is_ir_vascular_chest_possible`: 8/8 passed; observed `CH, IR, VA`.
- `testicular_torsion_is_gu_emergency_vascular_possible`: 8/8 passed; observed `ER, GU` and `ER, GU, PD`.

Additional targeted samples:

- `metastatic_liver_lesions_are_oncologic_gi`: 5/5 passed; observed `GI, OI`.
- `lung_cancer_staging_is_oncologic_chest`: 5/5 passed; observed `CH, OI`.
- `tumor_treatment_response_is_oncologic`: 5/5 passed; observed `MI, OI`.
- `aortic_dissection_is_vascular_emergency_chest_possible`: 8/8 passed after adding dissection to
  ER mapping; observed `CH, ER, VA`.
- `aortic_aneurysm_is_vascular`: 8/8 passed after adding the non-acute aneurysm ER exclusion;
  observed `VA`.
- `scrotal_mass_is_gu_oncologic_possible_not_ob`: 8/8 passed after adding the scrotal/testicular
  mass ER exclusion; observed `GU, OI`.

## Verification

Command:

```bash
uv run --env-file ../../.env python -m evals.metadata_subspecialty_domain_decision
```

Result after OI expansion and sampling adjustments: 46/46 deterministic fixture cases passed.

Command:

```bash
uv run --package findingmodel-ai pytest packages/findingmodel-ai/tests/test_metadata_prompt_repair_pilot.py -q
```

Result: 3/3 tests passed.
