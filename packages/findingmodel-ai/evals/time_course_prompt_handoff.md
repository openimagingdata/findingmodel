# Handoff: tune the etiology/tempo agent prompt for `expected_time_course`

You are revising the prompt that assigns `expected_time_course` (and `etiologies`) during metadata enrichment. Below is measured evidence of where the agent's time-course output diverges from human-curated answers on 54 development findings, plus the current prompt and the rules you must follow. Propose concrete prompt edits — do not change code or schema.

## The problem (measured on 54 dev findings)

- `expected_time_course` averages **0.69** against a **0.75** floor — the field is failing.
- **19 exact**, **19 pure omissions** (agent returned null where a curator committed a value), **16 commit-but-wrong** (agent committed but duration/modifier disagreed).

- The omissions span every duration band (permanent, years, months, days, hours) and every modifier — this is **systematic over-abstention**, not a single blind spot. The agent declines to commit in cases where a human curator confidently assigns persistence.

- Secondary issue: when it does commit, duration/modifier is often off (the 16 disagreements). Both under-committing and imprecise-when-committing need addressing.

## Current prompt (verbatim)

File: `packages/findingmodel-ai/src/findingmodel_ai/metadata/prompts/etiology_tempo.md`

```
Consider whether this radiology finding implies one or more etiologies and/or an expected time
course, and if so, assign the corresponding labels from the allowed values below.

Goal:
- `etiologies` are common process types that produce the finding or diagnosis. They help group
  findings that may be related.
- `expected_time_course` is how long the finding commonly remains visible on imaging.

Return only `etiologies`, `expected_time_course`, and optional `field_confidence`.

Allowed etiology codes: inflammatory; inflammatory:infectious; neoplastic:benign;
neoplastic:malignant; neoplastic:metastatic; neoplastic:potential; traumatic:acute;
traumatic:sequela; vascular:ischemic; vascular:hemorrhagic; vascular:thrombotic;
vascular:aneurysmal; vascular; cardiac; degenerative; metabolic; congenital; developmental;
autoimmune; toxic; mechanical; iatrogenic:post-operative; iatrogenic:post-radiation;
iatrogenic:device; iatrogenic:medication-related; idiopathic; normal-variant.
Allowed time course: duration = hours, days, weeks, months, years, permanent; modifier =
progressive, stable, evolving, resolving, intermittent, fluctuating, recurrent.

Etiology guidelines:
- Use `etiologies` for common underlying processes that the finding name or description clearly
  implies. Otherwise assign null.
- Generic descriptive abnormalities usually have null etiologies. Examples: generic mass, nodule,
  lesion, cystic lesion, lucency, opacity, filling defect, thickening, or enhancement pattern do
  not by themselves imply neoplastic, inflammatory, or other process labels.
- Appearance, distribution, signal, density, size, and enhancement patterns do not by themselves
  imply a cause. Do not turn these descriptions into a differential diagnosis.
- Artifacts, measurements, indices, classifications, assessment scales, recommendations, and
  technique-only concepts do not carry etiologies.
- Some descriptive findings support broad grouping labels only when the name or description clearly
  implies them. Examples: metastatic lymphadenopathy supports malignancy; abscess or empyema
  supports infection; inflammatory fluid collections support inflammation.
- Do not assign an etiology from the organ or structure alone. A finding in the heart is not
  automatically `cardiac`, and a finding in a vessel is not automatically `vascular`; use those
  labels when the finding itself implies cardiac dysfunction or a vascular mechanism.
- Lymphadenopathy implies broad inflammatory and malignant processes. Calcified lymph nodes are a
  separate descriptive finding and do not automatically inherit those etiologies.
- Generic cysts and descriptive fluid findings such as effusions usually have null etiologies unless
  the name or description states a specific process. Do not infer cardiac, inflammatory,
  iatrogenic, neoplastic, or vascular causes from fluid location or character alone.
- FDG avidity and other metabolic activity do not mean `neoplastic:potential`; use inflammatory or
  malignant labels only when the finding name or description supports them.
- Prefer the most specific etiology that is clearly justified. If no specific child code is
  justified but a broader family clearly is, use the parent code. Do not output parent plus child
  unless they are separate supported causes. Usually use no more than three etiology codes.

Etiology heuristics:
- For an unspecified tumor or neoplasm, include both `neoplastic:benign` and
  `neoplastic:malignant`. For explicitly benign, malignant, or metastatic diagnoses, use only the
  stated category. Use `neoplastic:potential` only for risk, suspicion, premalignant change, or
  malignant potential.
- Use `neoplastic:metastatic` when metastasis is named or clearly implied; do not also output
  `neoplastic:malignant` unless a separate primary malignancy process is represented.
- For vascular findings, use `vascular:thrombotic` for clot or embolus, `vascular:aneurysmal` for
  aneurysm or dilation, and parent `vascular` for other vascular wall injury.
- Use `inflammatory:infectious` only when infection, pathogen, abscess, or pus is named; otherwise
  use `inflammatory`.
- Use iatrogenic labels only when the finding itself is a device or treatment effect.
- Age context alone, including fetal, newborn, infant, child, or pediatric wording, does not imply
  `congenital` or `developmental`; use those labels only when the finding name or description
  supports them.
- Use `mechanical`, `cardiac`, `normal-variant`, `autoimmune`, `toxic`, and `idiopathic` only when
  directly supported; do not infer them from anatomy or generic clinical association.

Time-course guidelines:
- Use `expected_time_course` only when the finding itself has a typical imaging persistence.
  Otherwise assign null.
- Use null for artifacts, measurements, indices, classifications, assessment scales,
  recommendations, technique-only concepts, and other concepts that are not persistent imaging
  findings.
- Use null for descriptive fluid findings such as effusions unless the finding name or description
  states a specific process with typical persistence.
- Use null when persistence depends mainly on the underlying cause rather than on the finding class
  itself.
- Choose the long end of common imaging persistence. This is how long the finding may remain
  visible on imaging, not how long symptoms last or how long treatment takes.
- Do not assign years merely because a generic lesion could persist; assign a time course only when
  the named finding class has characteristic imaging persistence.
- Null etiology does not require null time course. Nodules, calcifications, focal asymmetry, chronic
  enlargement, and other specific persistent finding classes may still have a time course.

Time-course heuristics:
- Hours/days: fleeting physiology and active acute processes that usually change quickly.
- Weeks: acute injuries or postoperative/treatment effects expected to resolve promptly.
- Months: common long-end persistence for treated infection, thrombus/embolus, hemorrhage, infarct,
  traumatic soft-tissue injury, and other resolving/evolving abnormalities.
- Years: stable focal lesions, chronic enlargement, benign hyperplasia, slow tumors, and chronic
  degenerative or malignant disease.
- Permanent: durable implants, fixed deformity, chronic tear, calcification, chronic vascular
  structural abnormality, infarct scar, or stable benign vascular lesion.
- Use at most one modifier: `resolving` for expected improvement, `evolving` for changing blood/
  infarct/dissection/fluid, `progressive` for growth or degeneration, `stable` for durable
  findings.
- If more than one modifier seems plausible, choose the dominant future behavior: `progressive` for
  expected worsening/enlargement; `evolving` for composition or phase changes; `stable` only when
  meaningful change is not expected.
```

## Evidence A — pure omissions (agent said null; curator committed)

These are the highest-value cases: the curator saw a typical imaging persistence the agent missed.

| finding | curator's expected_time_course | finding description |
| --- | --- | --- |
| acute lung injury and ards in children | days / progressive, evolving | Injury to the lungs in children leading to acute respiratory distress. |
| air in esophagus | hours / intermittent | Presence of air within the esophagus |
| arterial tortuosity | permanent / stable | Marked twisting or winding of an artery. |
| brain ischemia secondary to extracranial lesion | days / evolving, resolving | Reduced blood flow to the brain due to an external lesion. |
| breast calcification cluster | permanent / stable | Breast calcification clusters are typically a sign of benign changes in breast tissue but can sometimes indicate malignancy. |
| breast soft tissue lesion | years / progressive | A breast soft tissue lesion refers to any abnormal growth or mass within the soft tissue of the breast, which may include various entities such as cysts, fibromas, or malignant tumors, typically identified through imaging modalities like ma… |
| cardiac valve thickening | years / progressive | Thickening of the cardiac valves, often associated with stenosis or valvular disease. |
| early intrauterine pregnancy | months / progressive, evolving | Initial stages of a pregnancy inside the uterus |
| fetal chest mass | months / progressive | An abnormal growth located in the fetal thoracic region. |
| focal shadowing pancreatic lesion | months / stable, progressive | Local area of shadowing in the pancreas, suggestive of a mass. |
| increased resistance index of renal transplant | days / progressive, stable | Elevation of the resistance index in a transplanted kidney, suggesting possible complications. |
| infratentorial intracranial tumor in a child | months / progressive | Tumor located below the tentorium cerebelli, common in pediatric brain tumors. |
| large orbit | permanent / stable | An orbit that is larger than normal. |
| large vascular grooves of skull | years / stable | Prominent grooves on the inner table of the skull, indicate expanded venous channels. |
| omega sella | permanent / stable | An omega-shaped enlarged sella turcica. |
| renal ischemia | days / evolving | Decreased blood flow to the kidney causing functional impairment. |
| t2-hyperintense renal mass | months / progressive | Renal mass with high T2 signal on MRI. |
| vertebral compression fracture | years / stable, progressive | Loss of vertebral body height due to axial loading, appearing as wedging or endplate depression on radiograph. |
| vertebral coronal cleft | permanent / stable | Presence of a cleft or split in the coronal plane of a vertebra. |

## Evidence B — committed but disagreed (duration/modifier off)

| finding | agent said | curator said | finding description |
| --- | --- | --- | --- |
| Acute Clavicle Fracture | weeks / resolving | months / evolving, resolving | Acute Clavicle Fracture Detection |
| aortic stent | years / stable | permanent / stable | Endovascular stent graft within the aorta, visible as metallic mesh on radiograph. |
| arterial rupture | days / evolving | hours / progressive | A tear through the full thickness of the wall of an artery. |
| congenital premature craniosynostosis | permanent / stable | permanent / progressive | Premature fusion of one or more cranial sutures, affecting skull shape. |
| fracture | months / evolving | months / evolving, resolving, stable | Disruption of cortical bone continuity identified on radiograph. |
| gastrostomy tube | years / stable | permanent / stable | Feeding tube placed directly through the abdominal wall into the stomach. |
| hypoplastic fibula | years / stable | permanent / stable | Underdevelopment or incomplete formation of the fibula bone. |
| leadless pacemaker | years / stable | permanent / stable | Self-contained pacemaker implanted directly in the cardiac chamber without transvenous leads, visible as a small capsule. |
| Mastectomy Breast Implant | years / stable | permanent / stable | Finding related to mastectomy and breast implant |
| Pneumonia | weeks / resolving | weeks / evolving, resolving | This module describes the Common Data elements and Macros for Pneumonia |
| pulmonary artery catheterization | permanent / stable | weeks / stable | Insertion of a catheter into the pulmonary artery for diagnostic purposes. |
| pulmonary contusion | months / resolving | weeks / resolving | Hemorrhage and edema within the lung parenchyma resulting from blunt thoracic trauma, appearing as non-segmental airspace opacity. |
| renal parenchymal gas | weeks / evolving | days / evolving, progressive | Presence of gas within the renal parenchyma usually indicating infection. |
| sternal fixation | years / stable | permanent / stable | Orthopedic hardware securing a sternal fracture or osteotomy. |
| Traumatic Brain Injury | months / evolving | days / evolving, resolving | A subset of data elements from the National Institute for Neurological Disorders and Stroke (NINDS). The data elements are related to the reporting template for imaging of traumatic brain injury. |
| traumatic pneumatocele | months / evolving | weeks / resolving | A thin-walled, air-filled cystic space within the lung parenchyma resulting from traumatic disruption, typically seen adjacent to or within areas of pulmonary contusion. |

## What to produce

Proposed edits to `etiology_tempo.md` (show them as a diff or before/after of specific lines) that reduce over-abstention on time-course **without** inflating false commitments, and tighten duration/modifier selection. Explain the reasoning behind each edit, tied to the evidence above.

## Hard constraints (project rules)

- **Do NOT paste these eval findings, names, or answers into the prompt.** These are evaluation cases; encoding them in the prompt is overfitting and is forbidden. Generalize the *guidance*, not the *cases*.

- **Do not restate the schema/enum values as new rules** beyond what's already there; the structured-output schema is the spec.

- Keep edits minimal and principle-based (YAGNI). Prefer sharpening existing rules over adding bulk.

- The agent must still abstain (null) for genuine non-persistent concepts (artifacts, measurements, indices, classifications, pure technique). The goal is correct commitment, not blanket commitment.

