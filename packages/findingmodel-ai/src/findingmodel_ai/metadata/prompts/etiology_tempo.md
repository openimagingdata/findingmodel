Consider whether this radiology finding implies one or more etiologies and/or an expected time
course, then return only `etiologies`, `expected_time_course`, and optional `field_confidence`.

Goal:
- `etiologies` are common process types that produce the finding or diagnosis. They are not a
  differential diagnosis.
- `expected_time_course` is the usual imaging-observable persistence of the named finding class.
  It is not symptom duration, treatment duration, or the persistence of an unresolved cause.

Core rules:
- The two fields have opposite defaults. **Etiology defaults to null** — assign only with direct
  support from the finding name or description. **Time course defaults to a committed value** —
  assign for any named finding class with characteristic imaging persistence, and use null only for
  the closed set of non-persistent concepts listed below.
- Decide etiology first, then use the etiology you assigned when choosing the time course.
- For etiologies, do not infer cause from anatomy, modality, appearance, age context, or generic
  clinical association.
- Null etiology does not require null time course. A finding may have no clear cause but still have
  characteristic persistence.

Allowed etiology codes: inflammatory; inflammatory:infectious; neoplastic:benign;
neoplastic:malignant; neoplastic:metastatic; neoplastic:potential; traumatic:acute;
traumatic:sequela; vascular:ischemic; vascular:hemorrhagic; vascular:thrombotic;
vascular:aneurysmal; vascular; cardiac; degenerative; metabolic; congenital; developmental;
autoimmune; toxic; mechanical; iatrogenic:post-operative; iatrogenic:post-radiation;
iatrogenic:device; iatrogenic:medication-related; idiopathic; normal-variant.
Allowed time course: duration = hours, days, weeks, months, years, permanent; modifier =
progressive, stable, evolving, resolving, intermittent, fluctuating, recurrent.

Etiology rules (default null; commit only on direct support):
- Use the most specific etiology that is clearly justified. If only a broad family is justified,
  use the parent code. Do not output parent plus child unless separate supported processes are
  represented. Usually use no more than three etiology codes.
- Generic descriptive abnormalities have null etiologies. Appearance, distribution, signal,
  density, size, and enhancement patterns do not by themselves imply a cause. Do not attach a
  benign/malignant/inflammatory differential to a nonspecific descriptive finding such as an
  asymmetric enlargement or a density/signal pattern.
- A named mass, tumor, or neoplasm still commits neoplastic labels per the rules below; the guard
  above is for descriptive appearance findings, not for named neoplasms.
- Generic cysts and descriptive fluid findings usually have null etiologies unless the name or
  description states a specific process.
- FDG avidity or other metabolic activity does not by itself imply neoplasm. Use inflammatory or
  malignant labels only when the finding name or description supports them.
- Lymphadenopathy supports broad inflammatory and malignant processes; calcified lymph nodes are a
  separate descriptive finding and do not automatically inherit those etiologies.
- For an unspecified or indeterminate mass, lesion, or tumor, use `neoplastic:potential`. Do NOT
  assert `neoplastic:malignant` (or `:benign`) without explicit support in the name or description —
  over-calling malignancy is the most costly error. For named benign, malignant, or metastatic
  entities, use only the stated category.
- Use `neoplastic:metastatic` when metastasis is named or clearly implied. Do not also output
  `neoplastic:malignant` unless a separate primary malignant process is represented.
- Use `vascular:thrombotic` for clot or embolus, `vascular:aneurysmal` for aneurysm or dilation,
  and parent `vascular` for other vascular wall injury.
- Use `inflammatory:infectious` only when infection, pathogen, abscess, or pus is named; otherwise
  use `inflammatory`.
- Use iatrogenic labels only when the finding itself is a device or treatment effect.
- Formation trio: a finding that is itself a structural malformation or anomaly of formation takes
  `developmental` and/or `congenital`; a fixed benign anatomic variant takes `normal-variant`.
  These commonly co-occur (congenital + developmental, or developmental + normal-variant). The
  trigger is the finding being a malformation or fixed variant — not the patient's age. Age context
  alone (fetal, newborn, infant, child, pediatric wording) does not imply these labels.
- Use `mechanical`, `cardiac`, `autoimmune`, `toxic`, and `idiopathic` only when directly supported.

Time-course rules (commit by default; null only for the closed set below):
- Null-trigger override (check first): if the finding is fundamentally a measurement, score, index,
  classification, assessment scale, recommendation, technique-only or administrative concept, or
  artifact, return null — even when it names a body part or sounds persistent. These override
  commit-by-default. (For example, a density measurement, an injury classification, or a severity
  score is null regardless of how durable the underlying anatomy is.)
- Otherwise, default to assigning a time course. Most named finding classes have a characteristic
  imaging persistence — commit to it rather than abstaining.
- Use the etiology you assigned, together with the finding class, to choose duration and modifiers:
  treated infection → months; acute trauma → weeks to months; hemorrhage, infarct, or thrombus →
  months; congenital or developmental malformation, calcification, fixed deformity, or implanted
  hardware → permanent or years.
- Durable structural or morphologic findings — deformity, dysplasia or hypoplasia, calcification,
  chronic enlargement, fixed anatomic variant, implanted hardware or device — are persistent by
  nature: commit to permanent or years.
- A measurement or index that names a persistent abnormal state takes that state's time course;
  only a bare number, score, or technique result is null.
- Assign null ONLY for this closed set: pure measurement values, scores, or indices; classifications
  and assessment scales; recommendations; technique-only or administrative concepts; artifacts; or a
  finding whose persistence depends entirely on an unspecified underlying cause (for example
  effusions and other descriptive fluid collections whose persistence tracks the cause).
- Choose the long end of common imaging persistence, not rare outliers. This is how long the finding
  remains visible on imaging, not symptom or treatment duration.

Duration anchors:
- Hours/days: fleeting physiology and active acute processes that usually change quickly.
- Weeks: acute injuries, temporary devices/tubes, or prompt postoperative/treatment effects.
- Months: long-end persistence for treated infection, thrombus/embolus, hemorrhage, infarct,
  traumatic soft-tissue injury, and other resolving/evolving abnormalities.
- Years: stable focal lesions, chronic enlargement, benign hyperplasia, slow tumors, and chronic
  degenerative or malignant disease.
- Permanent: durable implants or hardware, fixed deformity, chronic tear, calcification, chronic
  vascular structural abnormality, infarct scar, or stable benign vascular lesion.

Modifier anchors:
- Use the minimal set of characteristic modifiers.
- `progressive`: expected growth, worsening, pressure effect, or degeneration.
- `stable`: durable findings whose appearance should not materially change.
- `evolving`: changing phase, composition, or appearance.
- `resolving`: expected clearing, shrinkage, or improvement.
- More than one modifier is appropriate only when multiple phases are characteristic of the finding
  class.

Before returning:
- Do not output etiology labels as a differential diagnosis; nonspecific descriptive appearance
  findings (asymmetric enlargement, density/signal patterns) take null etiologies.
- Never assert malignancy for an indeterminate mass — use `neoplastic:potential`.
- Commit a time course for any named persistent finding class; reserve null for the closed trigger
  list above. Measurements, scores, indices, classifications, and assessment scales are always null
  even if they sound persistent.
- Do not let null etiology force null time course.
- Prefer the long end of common imaging persistence.
- Use only the minimal justified etiologies and modifiers.
