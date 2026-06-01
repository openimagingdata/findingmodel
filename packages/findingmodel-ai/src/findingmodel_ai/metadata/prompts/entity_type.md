Assign only `entity_type` for one radiology finding model.

Output only `entity_type` and optional `field_confidence`.

Rules:
- Preserve the authored finding identity; do not rewrite or infer a different model.
- Use selected canonical ontology as identity support only. Rejected, broader, narrower, related,
  and review-only candidates do not support `entity_type`.
- `diagnosis` is for named diseases, disorders, injuries, complications, syndromes, and disease
  entities such as aneurysm, thrombosis, dissection, carcinoma, tumor, or infection.
- `finding` is for broad observations, descriptive lesions, morphology, enhancement patterns,
  effusions, uptake abnormalities, and umbrella abnormality labels unless the modeled concept is
  itself a disease entity.
- `measurement` is for one quantitative metric. `assessment` is for a score, grading scale,
  classification, reporting category, or grouped measurement/interpretation package.
- Use `recommendation`, `technique_issue`, or `grouping` only when the finding itself directly
  supports that type.
- Source tags are weak context. Do not let tags alone determine `entity_type`.
- Generic vascular connections, shunts, fistulas, and malformations are usually findings unless the
  authored model is a named disease or diagnosis.
- Negative findings, artifacts, assessments, and recommendations should not be converted into
  diseases merely because they imply clinical consequences.
- If `field_confidence` is present, keys must be real metadata fields and values must be numeric
  scores from 0 to 1.
