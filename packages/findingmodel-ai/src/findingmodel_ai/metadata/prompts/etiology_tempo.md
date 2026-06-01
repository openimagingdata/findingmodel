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
