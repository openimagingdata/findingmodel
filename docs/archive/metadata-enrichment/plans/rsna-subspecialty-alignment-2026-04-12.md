# Plan: RSNA Subspecialty Alignment

Status: In Progress (2026-04-12)

## Goal

Replace the current ad hoc `Subspecialty` enum with a subset aligned to official RSNA specialty
content codes for this field's intended use: radiology content domains that meaningfully describe
which service would typically read/report a finding.

This plan also defines:

- which RSNA codes we keep for `subspecialties`
- which RSNA codes we exclude from this field
- what each RSNA code means
- when to use `MI` vs `NM`
- when to use `SQ`

## Official RSNA Specialty Content Codes

Source basis:

- user-provided current RSNA code list for this correction pass
- RSNA specialty content code references and RSNA education pages reviewed during this pass

### Full RSNA List With Field Decision

| Code | Meaning | Keep For `subspecialties`? | Rationale |
|------|---------|-----------------------------|-----------|
| `BR` | Breast (Imaging and Interventional) | Yes | True reader-domain specialty |
| `BQ` | Biomarkers/Quantitative Imaging | No | Methodology / science area, not routine reader-domain service |
| `CA` | Cardiac Radiology | Yes | True reader-domain specialty |
| `CH` | Chest Radiology | Yes | True reader-domain specialty |
| `CT` | Computed Tomography | No | Modality, already represented by `applicable_modalities` |
| `DM` | Digital Mammography | No | Modality-specific technology bucket, not a content-domain reader specialty |
| `ED` | Education | No | Educational track, not finding-domain specialty metadata |
| `ER` | Emergency Radiology | Yes | True reader-domain specialty |
| `GI` | Gastrointestinal Radiology | Yes | True reader-domain specialty |
| `GU` | Genitourinary Radiology | Yes | True reader-domain specialty |
| `HN` | Head and Neck | Yes | True reader-domain specialty |
| `HP` | Health Policy | No | Policy/operations topic, not finding-domain specialty metadata |
| `IN` | Informatics | No | Informatics topic, not finding-domain specialty metadata |
| `IR` | Interventional | Yes | True reader-domain specialty |
| `LM` | Leadership & Management | No | Management topic, not finding-domain specialty metadata |
| `MI` | Molecular Imaging | Yes | True reader-domain specialty for molecular / functional / PET-centered interpretation |
| `MK` | Musculoskeletal Radiology | Yes | True reader-domain specialty |
| `MR` | Magnetic Resonance Imaging | No | Modality, already represented by `applicable_modalities` |
| `NM` | Nuclear Medicine | Yes | True reader-domain specialty for conventional radionuclide / scintigraphic interpretation |
| `NR` | Neuroradiology | Yes | True reader-domain specialty |
| `OB` | Obstetric/Gynecologic Radiology | Yes | True reader-domain specialty |
| `OI` | Oncologic Imaging | Yes | True reader-domain specialty |
| `OT` | Other | No | Catch-all bucket; too vague for controlled finding metadata |
| `PD` | Pediatric Radiology | Yes | True reader-domain specialty |
| `PH` | Physics and Basic Science | No | Science/physics topic, not finding-domain specialty metadata |
| `PR` | Professionalism (including Ethics) | No | Professional topic, not finding-domain specialty metadata |
| `SQ` | Quality Assurance/Quality Improvement (including radiation and general safety issues) | Yes | Appropriate for artifacts, technique/quality issues, and imaging safety / QA-type findings |
| `RO` | Radiation Oncology | No | Separate treatment specialty; generally not the reader-domain for radiologic findings in this schema |
| `RS` | Research and Statistical Methods | No | Research methods topic, not finding-domain specialty metadata |
| `US` | Ultrasound | No | Modality, already represented by `applicable_modalities` |
| `VA` | Vascular | Yes | True reader-domain specialty for vascular findings |

## Keep Set For `Subspecialty`

The intended enum for this field should be:

- `BR`
- `CA`
- `CH`
- `ER`
- `GI`
- `GU`
- `HN`
- `IR`
- `MI`
- `MK`
- `NM`
- `NR`
- `OB`
- `OI`
- `PD`
- `SQ`
- `VA`

## Codes To Remove / Replace

- remove `AB`
  - not an official RSNA specialty content code in the provided current list
- remove `VI`
  - replace with `VA`
- add `NM`
- add `SQ`

## Policy: `MI` vs `NM`

### `MI`

Use `MI` when the finding is fundamentally interpreted in a molecular / functional imaging frame,
especially when one or more of these are true:

- PET/FDG uptake is central to the finding definition
- receptor / metabolic / physiologic imaging is central to interpretation
- the finding is routinely characterized in oncologic PET or other molecular-imaging workflows
- the imaging question is not just “is there uptake” but a broader molecular/functional biologic
  interpretation

Examples of intended use:

- FDG-avid lesion findings
- PET-driven oncologic characterization
- tracer-based functional characterization where molecular behavior is the main interpretive point

### `NM`

Use `NM` when the finding belongs to conventional nuclear medicine interpretation more than the
broader molecular-imaging frame, especially when one or more of these are true:

- the finding is routinely interpreted on planar scintigraphy
- the finding is routinely interpreted on SPECT/SPECT-CT
- the core workflow is classic radionuclide imaging rather than PET/molecular-oncology framing

Examples of intended use:

- thyroid uptake / thyroid scan patterns
- hepatobiliary scintigraphy findings
- gastric emptying or other classic nuclear medicine functional studies
- bone scintigraphy-first findings when the nuclear medicine workflow is the core reader domain

### When Both Apply

Use both `MI` and `NM` when both domains are genuinely central.

Default rule:

- PET-centered oncologic / molecular characterization -> usually `MI`
- conventional scintigraphic / SPECT nuclear medicine workflow -> usually `NM`
- if both are core to the finding's routine interpretation -> `MI` and `NM`

## Policy: `SQ`

Use `SQ` for findings or metadata objects whose primary content is imaging quality, safety, or QA/QI
rather than patient pathology.

Appropriate `SQ` uses include:

- generalized artifacts
- technique issues
- protocol adequacy / quality failures
- imaging safety or dose-related quality issues
- QA/QI-style imaging problems

Guidance:

- generalized artifact / technique findings may use `SQ` alone
- if a quality issue is strongly tied to a specific content area, `SQ` may co-occur with that
  specialty
- do not use `SQ` for ordinary clinical pathology findings

## Planned Implementation

1. Update the `Subspecialty` enum to the new keep-set.
2. Add human-readable meanings to the enum definition used for this field.
3. Update `docs/finding-model-metadata-fields.md` with the corrected keep-set and meanings.
4. Update the metadata-assignment prompt to remove `AB` / `VI`, add `VA` / `NM` / `SQ`, and align
   examples/rules.
5. Update gold fixtures, tests, and seed cases to the corrected schema.
6. Run targeted tests and metadata-assignment evals.
7. Update this plan with results and any remaining policy questions.
