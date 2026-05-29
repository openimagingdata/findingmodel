# Finding Model Metadata Fields

This is the canonical reference for structured metadata fields on finding models. It records both
field definitions and review-derived decision standards.

## Identity Fields

### `oifm_id`

`FindingModelFull` only. Globally unique registry identifier matching
`OIFM_[A-Z]{3,4}_[0-9]{6}`. The 3-4 letter prefix is the contributing organization code.

### `name`

Canonical English name for the finding, diagnosis, measurement, assessment, recommendation, or
technique issue. Used as the primary display name and search target.

### `description`

Textbook-style definition. It should describe what the entity is, not how to diagnose it. Used as
context for metadata assignment and search indexing.

### `synonyms`

Alternate names a radiologist might use in a report.

### `tags`

Free-text browsing/search tags. These are not constrained to a controlled enum.

## Structured Metadata Fields

These fields are optional on the model. The enrichment tool should assign them only when the finding
supports the value. Only `entity_type` is required in current enrichment output.

### `entity_type`

Semantic category of what the model represents.

| Value | Meaning |
| --- | --- |
| `finding` | Imaging observation that requires further characterization to reach a diagnosis. |
| `diagnosis` | Specific pathologic entity with defined diagnostic criteria. |
| `grouping` | Collection of related findings described together. |
| `measurement` | Quantified imaging parameter. |
| `assessment` | Standardized scoring/classification system. |
| `recommendation` | Suggested follow-up action. |
| `technique_issue` | Image quality or acquisition problem. |

Decision standards:

- Finding versus diagnosis is the most important distinction: a finding is what is seen on the
  image; a diagnosis is what is concluded from what is seen.
- Pure assessment scales, report templates, technique-only concepts, recommendation-only concepts,
  acquisition-quality concepts, and non-imaging concepts may be skipped or treated as source-model
  issues rather than enriched.
- Examples to skip or treat as source-model issues: Glasgow coma scale, MR rectal tumor
  template-like content, thrombocytosis, and standalone non-finding flow/measurement concepts.
- BI-RADS and similar classification systems are `assessment`, not ordinary persistent imaging
  findings.

### `body_regions`

Gross anatomic region or regions where the finding is located.

| Value | Meaning |
| --- | --- |
| `head` | Intracranial and extracranial head |
| `neck` | Cervical soft tissues, thyroid, etc. |
| `chest` | Thorax including lungs, mediastinum, heart |
| `breast` | Mammary/breast tissue |
| `abdomen` | Abdominal cavity organs |
| `pelvis` | Pelvic organs and structures |
| `spine` | Vertebral column |
| `upper_extremity` | Shoulder, arm, forearm, wrist, hand |
| `lower_extremity` | Hip, thigh, knee, leg, ankle, foot |
| `whole_body` | Not anatomically localized or broadly systemic |

Decision standards:

- Assign the broadest accurate region when specificity is unsupported.
- Multi-region findings should carry all directly supported gross regions.
- Do not add regions for all possible causes, complications, or workup.
- Aorta may span chest and abdomen when the source does not justify thoracic versus abdominal
  narrowing.
- Cervical vertebral-column concepts should not be broadened to all spine when cervical specificity
  is explicit.
- Breast density and breast clips stay in breast anatomy; do not select proxy anatomy beyond breast.

### `subspecialties`

Radiology practice domains whose reports are directly concerned with the finding. These are
RSNA-aligned reader-domain codes retained for finding-model metadata, not the full RSNA specialty
content catalog. See `docs/metadata/subspecialties.md`.

Decision standards:

- Subspecialty labels describe radiologists' use of findings in reports. They are not ontology
  maintenance labels.
- Assign clear body-region domains.
- Allow horizontal domains to overlay body-region domains when justified.
- Do not infer all possible domains from all possible causes, complications, or workup.
- Body region alone is not enough for horizontal domains.

### `applicable_modalities`

Routine direct imaging methods used to demonstrate, evaluate, quantify, or follow the finding.

| Code | Modality |
| --- | --- |
| `XR` | Radiography |
| `CT` | Computed Tomography |
| `MR` | Magnetic Resonance Imaging |
| `US` | Ultrasound |
| `PET` | Positron Emission Tomography |
| `NM` | Nuclear Medicine, non-PET scintigraphy |
| `MG` | Mammography |
| `RF` | Fluoroscopy |
| `DSA` | Digital Subtraction Angiography |

Decision standards:

- Direct modality language is strong evidence.
- Modality-specific wording outranks broad anatomy.
- Body region alone is not enough.
- Do not add modalities for all theoretical detection methods, incidental visibility, downstream
  workup, source-clot workup, or rare problem-solving alternatives.
- Existing modality tags are context, not proof.
- T1/T2/MR wording supports MR.
- PET, FDG, and tracer-avid PET wording support PET.
- Scintigraphy, SPECT, bone scan, V/Q, HIDA, renogram, and thyroid uptake support NM.
- Mammographic and breast-screening density language supports MG.
- Fluoroscopy, swallow study, esophagram, upper GI series, enema, arthrogram, and tube/catheter
  position checks support RF.
- Catheter angiography, embolization, endovascular intervention, and angiographic run support DSA.
- Radiolucent stones should not be assigned XR merely because stones can be imaged with XR.
- Torsion should not be assigned XR.
- Hydronephrosis supports US and CT; do not add XR/RF/NM without named-study support.
- Generic treatment response should not output every modality.
- Generic artifact without a named modality should not output every modality.

### `etiologies`

Common process types that produce the finding or diagnosis. Etiologies are not a full differential
diagnosis and should not list all theoretically possible causes.

| Code | Meaning |
| --- | --- |
| `inflammatory` | General inflammation |
| `inflammatory:infectious` | Infection |
| `neoplastic:benign` | Benign tumor |
| `neoplastic:malignant` | Primary malignancy |
| `neoplastic:metastatic` | Metastatic disease |
| `neoplastic:potential` | Premalignant or uncertain malignant potential |
| `traumatic:acute` | Acute traumatic injury |
| `traumatic:sequela` | Post-traumatic chronic change |
| `vascular:ischemic` | Ischemic process |
| `vascular:hemorrhagic` | Hemorrhagic process |
| `vascular:thrombotic` | Thrombotic or embolic process |
| `vascular:aneurysmal` | Aneurysmal dilation |
| `vascular` | Vascular process when a more specific vascular subtype is not justified |
| `cardiac` | Cardiac mechanism or disease process |
| `degenerative` | Age-related wear or degeneration |
| `metabolic` | Metabolic or biochemical cause |
| `congenital` | Present from birth |
| `developmental` | Develops during growth |
| `autoimmune` | Autoimmune mechanism |
| `toxic` | Toxic exposure |
| `mechanical` | Mechanical cause |
| `iatrogenic:post-operative` | Post-surgical change |
| `iatrogenic:post-radiation` | Radiation-induced change |
| `iatrogenic:device` | Device-related |
| `iatrogenic:medication-related` | Drug-induced |
| `idiopathic` | Unknown cause |
| `normal-variant` | Anatomic variant, not pathologic |

Decision standards:

- Assign etiologies only when the finding name or description clearly implies the process type.
- If no underlying process is implied, or if the cause is context-dependent, assign null.
- Generic cysts, nodules, masses, lesions, lucencies, opacities, filling defects, thickening,
  density/signal/enhancement patterns, and nonspecific fluid collections usually have null etiology.
- Do not infer etiology from anatomy, organ system, modality, body region, or possible workup alone.
- Pediatric words such as fetal, newborn, infant, child, or pediatric do not by themselves imply
  congenital or developmental etiology.
- Prefer the most specific etiology when justified. Use the parent label when only the broader
  process is justified, such as `vascular` for a vascular process that is not specifically
  ischemic, hemorrhagic, thrombotic/embolic, or aneurysmal.
- Do not output a parent and child etiology unless they represent separate supported processes.
- Lymphadenopathy generally implies inflammatory and neoplastic/malignant processes unless the
  finding is a narrower descriptive entity such as calcified lymph node.
- Calcified lymph node is a separate descriptive finding and should not automatically inherit broad
  lymphadenopathy etiologies.
- Explicit cardiac mechanism findings can use `cardiac`.
- Explicit vascular mechanism findings can use `vascular`.
- Do not infer cardiac or vascular etiology from heart or vessel anatomy alone.
- FDG avidity does not mean `neoplastic:potential`; use inflammatory or malignant labels only when
  supported.
- Hemangioma does not need a special vascular-malformation rule. When it is a benign tumor, use
  `neoplastic:benign`.

Specific adjudications:

- Generic cysts: null etiology.
- Posterior fossa cystic lesion: do not assign congenital solely from cystic posterior fossa
  wording.
- Generic pleural, pericardial, and transudative effusions: null etiology unless the name or
  description states a specific process.
- Breast density: null etiology.
- Aortic dissection: vascular process.
- Prolonged cerebral vascular transit time: vascular process.
- Cardiomegaly: cardiac process.
- Neonatal heart failure: cardiac process.
- FDG-avid pulmonary nodule: inflammatory and neoplastic/malignant are reasonable.
- Hepatic hemangioma: `neoplastic:benign`.
- Pulmonary vascular engorgement: vascular/cardiac mechanism is appropriate; do not force
  inflammatory.

### `expected_time_course`

How long the finding commonly remains observable on imaging, and how it usually changes. Use the
long end of common persistence, not rare outliers.

| Duration | Meaning |
| --- | --- |
| `hours` | Resolves or changes within hours |
| `days` | Resolves or changes within days |
| `weeks` | Resolves or changes within weeks |
| `months` | Resolves or changes within months |
| `years` | Slowly changing over years |
| `permanent` | Does not resolve; persists indefinitely |

| Modifier | Meaning |
| --- | --- |
| `progressive` | Gets worse or grows over time |
| `stable` | Remains unchanged |
| `evolving` | Changes in character or appearance |
| `resolving` | Gets better or shrinks |
| `intermittent` | Comes and goes |
| `fluctuating` | Changes unpredictably |
| `recurrent` | Resolves but reappears |

Decision standards:

- Assign time course only when the finding class itself supports a common observable persistence.
- Use null when persistence mainly depends on an unresolved cause.
- Null etiology does not require null time course.
- Generic lesions should not get years-long persistence merely because they could persist.
- Measurements, classifications, assessment scales, technique-only concepts, and recommendations
  usually have null time course.
- When a process often clears in weeks but commonly remains visible for months, choose months.
- Use one dominant modifier when possible.

Common duration anchors:

- Hours/days: fleeting physiology or active acute processes that change quickly.
- Weeks: acute injuries or prompt postoperative/treatment effects.
- Months: treated infection, thrombus/embolus, hemorrhage, infarct, traumatic soft-tissue injury,
  and other resolving/evolving abnormalities at the long end of common persistence.
- Years: stable focal lesions, chronic enlargement, benign hyperplasia, slow tumors, and chronic
  degenerative or malignant disease.
- Permanent: durable implants, fixed deformity, chronic tear, calcification, infarct scar, and
  fixed chronic vascular structural abnormality.

### `age_profile`

Two-part age characterization. `applicability` defines the age window where the finding can
reasonably occur. `more_common_in` highlights where incidence peaks.

Decision standards:

- If a finding can apply across ages, use all ages even if it is more common in one group.
- Use `more_common_in` only when commonness is actually part of the reviewed expectation.
- Fetal/newborn/pediatric wording can define age applicability, but does not by itself imply a
  congenital/developmental etiology.
- Do not use pediatric subspecialty or pediatric etiology merely because a disease can occur in
  children.

Age stages:

| Value | Approximate Age Range |
| --- | --- |
| `newborn` | Birth to 28 days |
| `infant` | 29 days to 1 year |
| `preschool_child` | 2-5 years |
| `child` | 6-12 years |
| `adolescent` | 13-17 years |
| `young_adult` | 18-24 years |
| `adult` | 25-44 years |
| `middle_aged` | 45-64 years |
| `aged` | 65+ years |

### `sex_specificity`

Whether the finding is anatomically restricted to one sex. This is about whether the finding can
occur in both sexes, not prevalence differences.

| Value | Meaning |
| --- | --- |
| `male-specific` | Only occurs in male anatomy |
| `female-specific` | Only occurs in female anatomy |
| `sex-neutral` | Occurs in both sexes |

Decision standards:

- Sex-neutral should be assigned when the finding is not sex-specific.
- Do not convert prevalence differences into sex specificity.
- Do not convert organ/sex associations into sex specificity unless the finding itself is sex
  limited.

## Ontology And Anatomy Fields

### `index_codes`

Canonical ontology codes that are exact matches or clinically substitutable equivalents for the
finding model. They are not broader, narrower, or merely related concepts.

Decision standards:

- Existing index codes are likely, but not definitely, correct; ties and uncertainty lean toward
  keeping them.
- New unsupported codes should be penalized more strongly than existing extras.
- Missing display strings on existing index codes may be repaired without treating the record as a
  newly approved metadata decision.
- Non-exact ontology candidates belong in review artifacts, not canonical `index_codes`.

### `anatomic_locations`

RadLex-derived anatomic location codes identifying where the finding is located.

Decision standards:

- Existing anatomic codes are likely, but not definitely, correct; ties and uncertainty lean toward
  keeping them.
- Do not localize to an over-specific bone, organ part, or structure when the finding is broader.
- Use the broadest accurate anatomic location when specificity is unsupported.
- Source modality tags and descriptions can provide context, but should not override finding
  meaning.

## Source Context And Weak Evidence

Some source lists have bounded context, such as a known anatomic scope, modality bias, or curated
hierarchical structure. That context can prevent over-broad assignments, but it should not override
the finding's meaning.

General standards:

- Use bounded source context as supporting evidence, not as a substitute for the finding name and
  description.
- Prefer null over weak whole-body fallback when no specific region is well supported.
- Do not force a primary anatomic location from a weak candidate set.
- Reject candidate codes or locations when all candidates are weak, cross-region without a clear
  primary location, or inconsistent with source context.
- Negative findings such as "no acute intracranial abnormality" are usually not ordinary positive
  imaging findings and may need skip/source-model handling.
- Device, postoperative, and technique concepts should be classified according to the concept
  represented; do not convert every postoperative or device-related mention into a disease finding.

## Contributor And Attribute Fields

Contributor metadata and clinical attributes are authored content. The metadata enrichment pipeline
does not modify attributes. Attributes define the structured data elements a radiologist would fill
out when characterizing the finding in a report, such as severity, size, or laterality.
