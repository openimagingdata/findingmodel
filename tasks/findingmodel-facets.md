# Radiology Finding Model Facets

This document defines the faceted classification system for radiology findings in the repository. Each finding should be tagged with appropriate values from these facets to enable effective search, filtering, and semantic retrieval.

---

## 1. Body Region

**Type**: Single-select (with optional sub-region)  
**Structure**: Two-level hierarchy

### Primary Regions

- **Head**
  - Sub-regions: brain, skull, face, orbit, paranasal sinuses, temporal bone
- **Neck**
  - Sub-regions: pharynx, larynx, thyroid
- **Chest/Thorax**
  - Sub-regions: lungs, mediastinum, heart, pleura, chest wall
- **Breast**
  - Sub-regions: breast tissue, axilla
- **Abdomen**
  - Sub-regions: liver, spleen, pancreas, kidneys, bowel, peritoneum, abdominal wall
- **Pelvis**
  - Sub-regions: bladder, pelvic floor
- **Spine**
  - Sub-regions: cervical spine, thoracic spine, lumbar spine, sacrum/coccyx, spinal cord
- **Upper Extremity**
  - Sub-regions: shoulder, arm, elbow, forearm, wrist, hand
- **Lower Extremity**
  - Sub-regions: hip, thigh, knee, leg, ankle, foot
- **Whole Body**
  - For findings affecting multiple regions or systemic findings

**Implementation Notes**:

- Use the primary region for indexing
- Sub-regions are optional but recommended for specificity
- Can extend sub-regions as needed (e.g., "Chest", "Lungs")

---

## 2. Subspecialty

**Type**: Multi-select  
**Source**: RSNA standard subspecialty categories

- AB: Abdominal Radiology
- BR: Breast Imaging
- CA: Cardiac Imaging
- CH: Chest/Thoracic Imaging
- ER: Emergency Radiology
- GI: Gastrointestinal Radiology
- GU: Genitourinary Radiology
- HN: Head & Neck Imaging
- IR: Interventional Radiology
- MI: Molecular Imaging/Nuclear Medicine
- MK: Musculoskeletal Radiology
- NR: Neuroradiology
- OB: OB/Gyn Radiology
- OI: Oncologic Imaging
- PD: Pediatric Radiology
- VI: Vascular Imaging

**Implementation Notes**:

- Findings may be relevant to multiple subspecialties
- Example: Pulmonary embolism → Chest/Thoracic Imaging + Vascular Imaging + Emergency Radiology

---

## 3. Etiology

**Type**: Multi-select  
**Description**: Underlying cause or pathophysiologic mechanism

### Categories

- `inflammatory:infectious`
- `inflammatory`
- `neoplastic:benign`
- `neoplastic:malignant`
- `neoplastic:metastatic`
- `neoplastic:potential` (for indeterminate lesions, incidentalomas)
- `traumatic-acute` (acute injury)
- `post-traumatic` (sequelae of prior injury)
- `iatrogenic:post-operative`
- `iatrogenic:post-radiation`
- `iatrogenic:device`
- `iatrogenic:medication-related`
- `vascular:ischemic`
- `vascular:hemorrhagic`
- `vascular:thrombotic`
- `degenerative`
- `congenital`
- `metabolic`
- `toxic`
- `mechanical` (e.g., obstruction, herniation, torsion)
- `idiopathic`
- `normal-variant`

**Implementation Notes**:

- Multiple (or no) standard etiologies may apply (e.g., post-radiation inflammatory changes)

---

## 4. Entity Type

**Type**: Single-select  
**Description**: Fundamental category of what is being described

- **Finding**: Direct imaging observation (e.g., "consolidation", "pleural effusion", "mass", "lymphadenopathy")
- **Diagnosis**: Clinical interpretation or disease state (e.g., "pneumonia", "heart failure", "lymphoma")
- **Grouping**: Umbrella or organizational terms (e.g., "lung parenchymal abnormalities", "pleural abnormalities", "pancreatic pathology")

**Implementation Notes**:

- Most entries will be "Finding" or "Diagnosis"
- Groupings are useful for hierarchical organization

---

## 5. Applicable Modalities

**Type**: Multi-select  
**Source**: DICOM standard modality codes

- **XR** - Radiography (plain X-rays)
- **CT** - Computed Tomography
- **MR** - Magnetic Resonance Imaging
- **US** - Ultrasound (including Doppler)
- **PET** - Positron Emission Tomography
- **NM** - Nuclear Medicine (single-photon/SPECT)
- **MG** - Mammography
- **RF** - Fluoroscopy (real-time X-ray)
- **DSA** - Digital Subtraction Angiography

**Implementation Notes**:

- Tag with all modalities where the finding can be detected
- Example: "Pulmonary nodule" → XR, CT, MR, PET
- Example: "Pneumothorax" → CR, XR, CT, US, MR
- Some findings are modality-specific (e.g., "restricted diffusion" → MR only)

---

## 6. Expected Time Course

**Type**: Multi-select (duration + behavioral modifiers)  
**Description**: Expected timeframe for evolution, resolution, or persistence of the finding

### Duration Categories (select one)

- **Rapid** (hours to days)  
  Examples: Pulmonary edema responding to diuretics, simple pneumothorax after treatment
- **Short-term** (1-3 weeks)  
  Examples: Uncomplicated pneumonia, simple hematoma, acute inflammation
- **Intermediate** (weeks to months)  
  Examples: Healing fracture, organizing pneumonia, subacute infarct evolution
- **Long-term** (months to years)  
  Examples: Chronic PE changes, post-radiation fibrosis evolution, large hematoma resorption
- **Permanent** (no expected change)  
  Examples: Remote infarct, surgical absence of organ, chronic sequelae, old fracture

### Behavioral Modifiers (multi-select, optional)

- **Resolving**: Possibly/likely absent on future exams
- **Evolving**: Will change over time to its final form
- **Progressive**: Continuously worsening over time  
  Examples: Degenerative disease, growing neoplasm, progressive fibrosis
- **Stable**: Not expected to change  
  Examples: Stable nodule, chronic finding under observation
- **Intermittent**: Comes and goes, on-and-off pattern  
  Examples: Intermittent bowel obstruction, transient ischemia
- **Waxing-waning**: Fluctuates in severity without complete resolution  
  Examples: Inflammatory conditions, some autoimmune diseases
- **Relapsing**: Recurrent episodes with intervals of remission  
  Examples: Relapsing-remitting MS, recurrent infections

**Implementation Notes**:

- May select ONE duration category (or not if not applicable)
- Add behavioral modifiers as appropriate (can be multiple)
- Examples:
  - Uncomplicated pneumonia: "Short-term", "Resolving"
  - Malignant mass: "Long-term, Progressive"
  - Acute stroke: "Rapid, Evolving"
  - Remote stroke: "Permanent, Stable"
  - Recurrent obstruction: "Rapid, Intermittent"

---

## 7. Age Association

**Type**: Multi-select  
**Source**: FDA and standard pediatric/geriatric classifications  
**Description**: Age groups where this finding is particularly relevant or commonly seen

- **Neonate** (0-28 days)
- **Infant** (29 days to <2 years)
- **Young child** (2-5 years)
- **Child** (6-12 years)
- **Adolescent** (13-18 years)
- **Young adult** (19-40 years)
- **Middle-aged adult** (41-65 years)
- **Older adult** (66+ years)

**Implementation Notes**:

- Select all age groups where the finding is clinically relevant (leave blank if not age-specific)
- Some findings have strong age associations (e.g., "Hyaline membrane disease" → Neonate; "Degenerative disc disease" → Middle-aged adult, Older adult)

---

## 8. Sex-Specificity

**Type**: Single-select  
**Description**: Whether the finding is specific to biological sex

- **Male-specific**: Findings only in biological males  
  Examples: Prostatic pathology, testicular lesions, seminal vesicle abnormalities
- **Female-specific**: Findings only in biological females  
  Examples: Ovarian lesions, uterine pathology, pregnancy-related findings

**Implementation Notes**:

- Use select the appropriate value for findings that are anatomically or physiologically specific to one sex phenotype

---

## Usage Examples

### Example 1: Pulmonary Consolidation

```yaml
finding_name: "Pulmonary consolidation"
body_region: "Chest", "Lung"
subspecialty: ["Chest/Thoracic Imaging", "Emergency Radiology"]
etiology:
  [
    "inflammatory",
    "inflammatory:infectious",
    "neoplastic:malignant",
    "vascular:hemorrhagic",
  ]
entity_type: "Finding"
applicable_modalities: ["XR", "MR", "CT"]
expected_time_course:
  duration: "Short-term"
  modifiers: ["Resolving"]
```

### Example 2: Degenerative Disc Disease

```yaml
finding_name: "Degenerative disc disease"
body_region: "Spine → Lumbar spine"
subspecialty: ["Musculoskeletal Radiology", "Neuroradiology"]
etiology: ["degenerative"]
entity_type: "Diagnosis"
applicable_modalities: ["CR", "DX", "CT", "MR"]
expected_time_course:
  duration: "Permanent"
  modifiers: ["Progressive"]
age_association: ["Middle-aged adult", "Older adult"]
sex_specificity: "Sex-neutral"
```

### Example 3: Placenta Previa

```yaml
finding_name: "Placenta previa"
body_region: "Pelvis → Uterus"
subspecialty: ["Women's Imaging (OB/GYN)"]
etiology: ["normal-variant"]
entity_type: "Finding"
applicable_modalities: ["US", "MR"]
expected_time_course:
  duration: "Intermediate"
  modifiers: ["Stable"]
age_association: ["Young adult", "Middle-aged adult"]
sex_specificity: "Female-specific"
```

### Example 4: Acute Subdural Hematoma

```yaml
finding_name: "Acute subdural hematoma"
body_region: "Head → Brain"
subspecialty: ["Neuroradiology", "Emergency Radiology"]
etiology: ["traumatic", "vascular:hemorrhagic"]
entity_type: "Finding"
applicable_modalities: ["CT", "MR"]
expected_time_course:
  duration: "Intermediate"
  modifiers: ["Self-resolving", "Treatment-dependent"]
age_association: ["Any age"]
sex_specificity: "Sex-neutral"
```

### Example 5: Pulmonary Embolism

```yaml
finding_name: "Pulmonary embolism"
body_region: "Chest → Lungs"
subspecialty:
  ["Chest/Thoracic Imaging", "Vascular Imaging", "Emergency Radiology"]
etiology: ["vascular:thrombotic"]
entity_type: "Diagnosis"
applicable_modalities: ["CT", "NM", "MR"]
expected_time_course:
  duration: "Intermediate"
  modifiers: ["Treatment-dependent"]
age_association: ["Any age"]
sex_specificity: "Sex-neutral"
```

---

## Implementation Considerations

### Database Schema

Consider storing facets as:

- **Single-select facets**: String or enum fields
- **Multi-select facets**: JSON arrays or separate junction tables
- **Hierarchical facets** (Body Region): Separate fields for primary and sub-region

### Search and Filtering

These facets enable:

- Faceted search interfaces (filter by modality, subspecialty, time course, etc.)
- Semantic clustering (group by etiology, body region)
- Clinical decision support (filter by age group, expected evolution)
- RAG retrieval (embedding + facet filtering)

### Extensibility

- Body region sub-regions can be expanded as needed
- Behavioral modifiers can be added to Expected Time Course
- Additional facets can be added as requirements evolve

### Validation Rules

- Body Region: Required, single-select main region, may also have a RELEVANT sub-region
- Subspecialty: Required, at least one
- Etiology: Recommended, at least one
- Entity Type: Required, single-select
- Applicable Modalities: Required, at least one
- Expected Time Course: Recommended duration, optional modifiers
- Age Association: Recommended, at least one
- Sex-Specificity: Recommended, single-select

### Mapping to Standards

Where possible, map facet values to standard terminologies:

- Body regions → anatomiclocations.org
- Subspecialties → RSNA codes
- Modalities → DICOM codes
- Etiologies → SNOMED CT concepts

---

## Version History

- **v1.0** (2025-11-05): Initial facet definitions

---

## References

- [Anatomic Locations](https://anatomiclocations.org)
- [RSNA RadLex](https://www.rsna.org/practice-tools/data-tools-and-standards/radlex-radiology-lexicon)
- [DICOM Standard](https://www.dicomstandard.org/)
- FDA Age Group Classifications for Medical Devices
- Clinical terminology research for disease progression and temporal patterns
