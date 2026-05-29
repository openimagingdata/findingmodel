# RSNA Subspecialty Codes For Finding Metadata

This document records the RSNA specialty content codes considered for the `subspecialties` field,
which subset we keep for finding-model metadata, and the review-derived rules for applying them.

The `subspecialties` field is intentionally narrower than the full RSNA list. It captures
reader-domain content specialties that meaningfully describe which radiology service would
read/report a finding. It does not encode modalities, meeting tracks, research themes, management
topics, or other non-reader-domain buckets.

All kept codes are non-exclusive. Apply every supported code rather than choosing one primary
subspecialty.

## Kept Codes

| Code | Meaning |
| --- | --- |
| `BR` | Breast Imaging and Intervention |
| `CA` | Cardiac Radiology |
| `CH` | Chest/Thoracic Radiology |
| `ER` | Emergency Radiology |
| `GI` | Gastrointestinal Radiology |
| `GU` | Genitourinary Radiology |
| `HN` | Head and Neck Radiology |
| `IR` | Interventional Radiology |
| `MI` | Molecular Imaging |
| `MK` | Musculoskeletal Radiology |
| `NM` | Nuclear Medicine |
| `NR` | Neuroradiology |
| `OB` | OB/GYN Radiology |
| `OI` | Oncologic Imaging |
| `PD` | Pediatric Radiology |
| `SQ` | Quality Assurance / Quality Improvement / Safety |
| `VA` | Vascular |

## Not Kept

| Code | Meaning | Why Not In This Field |
| --- | --- | --- |
| `BQ` | Biomarkers/Quantitative Imaging | Methodology or science area, not reader domain |
| `CT` | Computed Tomography | Modality; represented by `applicable_modalities` |
| `DM` | Digital Mammography | Modality or technology bucket, not content domain |
| `ED` | Education | Education track |
| `HP` | Health Policy | Policy/operations topic |
| `IN` | Informatics | Informatics topic |
| `LM` | Leadership and Management | Management topic |
| `MR` | Magnetic Resonance Imaging | Modality; represented by `applicable_modalities` |
| `OT` | Other | Too vague |
| `PH` | Physics and Basic Science | Science topic |
| `PR` | Professionalism including Ethics | Professional topic |
| `RO` | Radiation Oncology | Treatment specialty, not intended reader-domain field |
| `RS` | Research and Statistical Methods | Research methods topic |
| `US` | Ultrasound | Modality; represented by `applicable_modalities` |

## Schema Corrections

| Old Code | Status | Resolution |
| --- | --- | --- |
| `AB` | Remove | Not an official RSNA specialty content code in the current list |
| `VI` | Remove | Replace with `VA` |
| `NM` | Add | Needed to distinguish Nuclear Medicine from Molecular Imaging |
| `SQ` | Add | Needed for artifacts, quality, QA/QI, and safety-type findings |

## General Rules

- Subspecialties describe radiology practice domains whose reports are directly concerned with the
  finding.
- Body-region domains and horizontal domains can coexist.
- Horizontal domains include vascular, oncologic, quality/safety/technique, emergency/acute,
  interventional, molecular/PET, nuclear medicine, and pediatric.
- Do not infer all possible domains from all possible causes, complications, or workup.
- Body region alone is not enough for horizontal domains.

## Body-Region Domains

- `BR`: breast findings, breast procedures/devices, breast density, breast calcifications, breast
  masses, and breast implants.
- `CA`: cardiac structures, cardiac disease, cardiac devices, and cardiac findings. Cardiac
  findings seen on routine chest imaging may also be `CH`.
- `CH`: thoracic findings including lungs, pleura, mediastinum, chest wall, and many cardiac
  findings seen on chest imaging.
- `GI`: gastrointestinal tract, liver, biliary system, pancreas, spleen, and abdominal visceral
  findings.
- `GU`: kidneys, ureters, bladder, prostate, testes, scrotum, uterus, ovaries, and other
  genitourinary or reproductive findings.
- `HN`: head and neck soft tissue, face, orbit, teeth/jaw, pharynx/larynx, thyroid, and cervical
  lymph nodes.
- `MK`: bones, joints, muscles, tendons, ligaments, spine degenerative/traumatic findings, and
  extremity soft tissues when musculoskeletal.
- `NR`: brain, intracranial structures, spine/neural axis when neuroradiology-centered, and
  neurovascular findings when the report domain is neuroradiology.
- `OB`: pregnancy, fetal, placenta, uterus/ovaries in obstetric or gynecologic imaging contexts.

## Horizontal Domains

### Vascular: `VA`

Use `VA` for aorta, named vessels, flow, perfusion, vascular malformation, stenosis, thrombosis,
embolus, aneurysm, dissection, vascular injury, vascular access, and vascular devices when they
represent a finding.

Examples:

- Aortic aneurysm: `VA`; not `ER` unless acute complication is explicit.
- Aortic dissection: `VA` and `ER`; `CH` may also apply when thoracic/chest context is present.
- Abdominal/pelvic vessels remain `VA` unless organ disease is also modeled.

### Oncologic Imaging: `OI`

Use `OI` when the finding is part of malignancy staging, surveillance, tumor burden, metastatic
disease, or a clinically oncologic mass/lesion context.

Mass, lesion, or calcification can carry `OI` when clinically reasonable, but normal tissue and
normal anatomy labels do not.

### Emergency Radiology: `ER`

Use `ER` for acute emergency conditions, critical trauma, torsion, acute vascular catastrophe, and
similar report domains. Do not add `ER` to chronic conditions only because they could have an acute
complication.

### Interventional Radiology: `IR`

Use `IR` for image-guided procedures, biopsy, ablation, embolization, drains, catheters, stents,
access devices, and procedure/device-related findings. Intravascular access devices can be both
`IR` and `VA`.

### Molecular Imaging And Nuclear Medicine: `MI` / `NM`

Use `MI` when the finding is fundamentally molecular/functional or PET-centered:

- FDG-avid or PET-defined abnormalities;
- molecular, receptor, or tracer-based biologic characterization;
- oncologic PET contexts where PET interpretation is central.

Use `NM` when the finding is fundamentally conventional nuclear medicine:

- planar scintigraphy;
- SPECT or SPECT-CT;
- classic radionuclide functional studies.

Use both when both domains are genuinely central.

Default tie-breaker when a finding could read either way:

- PET-centered oncologic/molecular characterization → usually `MI`;
- conventional scintigraphic / SPECT nuclear-medicine workflow → usually `NM`;
- both core to routine interpretation → `MI` and `NM`.

Worked examples (these read as `NM`, not `MI`): thyroid uptake / thyroid scan, hepatobiliary
scintigraphy (HIDA), gastric-emptying studies, and bone-scintigraphy-first findings. FDG-avid
oncologic lesions and PET-defined abnormalities read as `MI`.

### Pediatric: `PD`

Use `PD` when pediatric context is explicit or the finding is pediatric-specific. Do not use `PD`
merely because a disease can occur in children.

### Quality/Safety/Technique: `SQ`

Use `SQ` for artifact, acquisition, protocol, image-quality, dose, safety, report-quality, and
technique issues. `SQ` may stand alone or co-occur with a content-area specialty when the issue is
strongly tied to a specific domain.

## Review-Derived Examples

- Aorta should not miss `VA`.
- Cervical lymphoid tissue or cervical lymph nodes should carry `HN`.
- Cardiac findings commonly assessed on routine chest imaging may be both `CA` and `CH`.
- Rib or chest-wall bony trauma may be both `CH` and `MK`.
- Scrotal/testicular mass is `GU` and may be `OI`; it is not `OB`.
- Tumor response can be `MI` and/or `OI` depending on modality and context.
- Central venous catheter findings can be both `IR` and `VA`.
- Artifact and acquisition-quality concepts can carry `SQ`, but ordinary presence/change
  attributes should not.
