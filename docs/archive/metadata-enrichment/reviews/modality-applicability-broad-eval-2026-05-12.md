# Modality Applicability Broad Eval Review

## Scope

Pilot prompt: `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/modality_applicability.md`

Focused eval: `packages/findingmodel-ai/evals/metadata_modality_applicability_decision.py`

## Current Prompt Size

- 40 lines
- 344 words
- 602 `gpt-4o` tokens

## Eval Set

The modality pilot suite has 29 curated cases covering:

- all modality codes: XR, CT, MR, US, PET, NM, MG, RF, and DSA;
- direct modality-specific language such as T2/MR, PET-avid, mammographic, scintigraphy, fluoroscopy, and catheter angiography;
- modality traps including radiolucent calculus not XR, torsion not XR, tumor response not every modality, and generic artifact not every modality;
- direct vascular/procedural imaging such as aortic dissection, pulmonary embolism, uterine artery embolization, and aneurysm coiling;
- common cross-sectional and ultrasound cases including hydronephrosis, thyroid nodule, adnexal mass, and placental findings.

Required-label coverage:

| Code | Required Cases |
|---|---:|
| CT | 8 |
| DSA | 2 |
| MG | 2 |
| MR | 4 |
| NM | 3 |
| PET | 1 |
| RF | 2 |
| US | 6 |
| XR | 2 |

## Prompt Changes From Eval Failures

- Added specific exclusions for radiolucent calculus and torsion to prevent XR overcalls.
- Added T2 adnexal mass guidance to prevent CT overcalls.
- Added hydronephrosis guidance to prevent XR/RF/NM overcalls.
- Added tumor-response guidance to prevent outputting every modality.
- Added generic-artifact guidance to return no specific modality instead of every modality.

## Repeat Sampling

Command:

```bash
uv run --env-file ../../.env python -m evals.metadata_modality_applicability_decision \
  --repeats 5 \
  --sample-case hydronephrosis_supports_us_and_ct \
  --sample-case radiolucent_urinary_calculus_excludes_xr \
  --sample-case testicular_torsion_supports_us \
  --sample-case tumor_treatment_response_supports_pet_and_ct \
  --sample-case uterine_artery_embolization_supports_dsa \
  --sample-case cerebral_aneurysm_coiling_supports_dsa \
  --sample-case motion_artifact_has_no_specific_modality
```

Result: all sampled cases passed 5/5 after prompt and fixture adjustments.

Additional targeted sample:

- `adnexal_t2_mass_supports_mr_and_us`: 8/8 passed after the T2 adnexal mass guidance; observed
  `MR` and `MR, US`.
- `acute_aortic_dissection_supports_ct`: 8/8 passed; observed `CT, DSA, MR`.

## Verification

Command:

```bash
uv run --env-file ../../.env python -m evals.metadata_modality_applicability_decision
```

Result: 29/29 deterministic fixture cases passed.

Command:

```bash
uv run --package findingmodel-ai pytest packages/findingmodel-ai/tests/test_metadata_prompt_repair_pilot.py -q
```

Result: 5/5 tests passed.
