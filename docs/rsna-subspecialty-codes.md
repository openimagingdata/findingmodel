# RSNA Subspecialty Codes For Finding Metadata

This document records the full RSNA specialty content code list considered for the
`subspecialties` field, which subset we keep for finding-model metadata, and why.

The `subspecialties` field is intentionally narrower than the full RSNA list. It is meant to
capture reader-domain content specialties that meaningfully describe which radiology service would
 typically read/report a finding. It is not meant to encode modalities, meeting tracks, research
themes, management topics, or other non-reader-domain buckets.

All kept codes are non-exclusive. This field is fully multi-label: apply any and all codes that
are genuinely relevant to the finding. There is no single primary subspecialty slot.

## Kept In `Subspecialty`

| Code | Meaning |
|------|---------|
| `BR` | Breast (Imaging and Interventional) |
| `CA` | Cardiac Radiology |
| `CH` | Chest Radiology |
| `ER` | Emergency Radiology |
| `GI` | Gastrointestinal Radiology |
| `GU` | Genitourinary Radiology |
| `HN` | Head and Neck |
| `IR` | Interventional |
| `MI` | Molecular Imaging |
| `MK` | Musculoskeletal Radiology |
| `NM` | Nuclear Medicine |
| `NR` | Neuroradiology |
| `OB` | Obstetric/Gynecologic Radiology |
| `OI` | Oncologic Imaging |
| `PD` | Pediatric Radiology |
| `SQ` | Quality Assurance/Quality Improvement (including radiation and general safety issues) |
| `VA` | Vascular |

## Not Kept In `Subspecialty`

| Code | Meaning | Why Not In This Field |
|------|---------|------------------------|
| `BQ` | Biomarkers/Quantitative Imaging | Methodology / science area rather than a reader-domain specialty |
| `CT` | Computed Tomography | Modality; already represented by `applicable_modalities` |
| `DM` | Digital Mammography | Modality / technology bucket rather than a content-domain specialty |
| `ED` | Education | Education track, not finding metadata |
| `HP` | Health Policy | Policy/operations topic, not finding metadata |
| `IN` | Informatics | Informatics topic, not finding metadata |
| `LM` | Leadership & Management | Management topic, not finding metadata |
| `MR` | Magnetic Resonance Imaging | Modality; already represented by `applicable_modalities` |
| `OT` | Other | Too vague for controlled finding metadata |
| `PH` | Physics and Basic Science | Science topic, not finding metadata |
| `PR` | Professionalism (including Ethics) | Professional topic, not finding metadata |
| `RO` | Radiation Oncology | Separate treatment specialty rather than the intended radiology reader-domain field |
| `RS` | Research and Statistical Methods | Research methods topic, not finding metadata |
| `US` | Ultrasound | Modality; already represented by `applicable_modalities` |

## Explicit Corrections From The Previous Schema

| Old Code | Status | Replacement / Resolution |
|----------|--------|---------------------------|
| `AB` | Remove | Not an official RSNA specialty content code in the current list |
| `VI` | Remove | Replace with `VA` |
| `NM` | Add | Needed to distinguish Nuclear Medicine from Molecular Imaging |
| `SQ` | Add | Needed for artifacts, quality, QA/QI, and safety-type findings |

## `MI` vs `NM`

Use `MI` when the finding is fundamentally molecular/functional or PET-centered:

- FDG-avid / PET-defined abnormalities
- molecular / receptor / tracer-based biologic characterization
- oncologic PET workflows where PET interpretation is the core reader-domain issue

Use `NM` when the finding is fundamentally conventional nuclear medicine:

- planar scintigraphy
- SPECT / SPECT-CT
- classic radionuclide functional studies where the core interpretive domain is nuclear medicine

Use both when both domains are genuinely central.

## `SQ`

Use `SQ` for image-quality, technique, QA/QI, and safety findings, including:

- generalized artifacts
- technique failures
- protocol/quality problems
- dose/safety quality issues

`SQ` may stand alone or co-occur with a content-area specialty when the issue is strongly tied to a
specific domain.
