Decide ontology-code applicability for one radiology finding model.

Output ontology candidate decisions only. Use only offered candidate IDs.

Rules:
- Mark a candidate canonical only when it is an exact match or clinically substitutable for the
  modeled finding itself.
- Multiple candidates from different ontology systems may be canonical when they each match the
  modeled finding. Do not prefer an ontology system by name.
- Reject candidates that add unsupported detail such as material, device subtype, location, named
  vessel, pattern, disease context, benign/malignant status, histology, grade, or stage.
- Reject candidates that drop meaningful modeled qualifiers such as patient group, fetal/neonatal
  status, pregnancy, timing, acuity, chronicity, severity, location, modality, laterality, focality,
  pattern, or composition.
- A narrower subtype is not canonical for a generic lesion, mass, tumor, abnormality, or device
  unless the model source includes that subtype.
- An unqualified tumor or neoplasm can include benign and malignant forms. Do not select
  malignant-only, benign-only, metastatic, premalignant, histologic-subtype, or grade-specific
  candidates unless the model says so.
- If an authored exact broad code exists, keep narrower plausible examples as review evidence rather
  than canonical codes.
- Anatomy, body part, vessel group, imaging view, measurement target, disease target, procedure, and
  exam concepts are not canonical for an abnormality, measurement, assessment, device state, or
  postoperative-state model unless they represent the modeled concept itself.
- A measurement, score, classification, or assessment used to evaluate a disease is not the disease.
- `clinically_substitutable` is only for concepts that can stand in for the modeled finding itself,
  not downstream diagnoses, clinical correlates, possible results, or conditions the model may help
  evaluate.
- Preserve useful non-canonical concepts as review evidence with a relationship and rejection reason
  when supported.
- If no candidate is good enough for canonical selection, return no canonical candidate decisions.
