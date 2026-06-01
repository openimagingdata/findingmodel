Decide the top-level anatomic scope for one radiology finding model.

Output only `anatomic_decisions` and `body_regions`. Use only offered candidate IDs.

Rules:
- Select the smallest candidate set that covers the modeled anatomic scope. Do not select every
  possible site for one instance of the finding.
- The model name, description, exact ontology labels, and attributes define scope. Candidate
  `support_level`, `current_metadata`, and `default_selected` are evidence, not commands.
- Reject labels that add unsupported sex, variant/accessory anatomy, quadrant, segment, named
  vessel/branch, side, lobe, endpoint, or other locality specificity.
- Do not narrow a generic regional mass, lesion, swelling, or soft-tissue abnormality to one tissue
  or structure unless the modeled finding commits to that structure.
- Location attribute values are choices within the model. If one parent, tract, organ set, or system
  candidate covers those values, select that parent alone.
- If no offered candidate covers the supported scope, select no anatomic candidate rather than using
  a narrower child, variant, landmark, or search hit as a proxy. Still assign `body_regions`.
- Device, tube, and catheter models are about placement or course anatomy when that is the modeled
  finding. Prefer supported placement/course anatomy over broader context or endpoint organs.
- For placement/course models, candidate evidence can support normal course locality. A containing
  region or endpoint organ is context; choose the supported wall, tract, or course candidate when it
  describes where the device normally lies or travels.
- Select both a parent and child only when they are distinct modeled scopes.
- Assign `body_regions` from selected anatomy and modeled scope, not from every possible site.
- Breast maps to `breast`; shoulder to `upper_extremity`; ovary, uterus, adnexa, and prostate to
  `pelvis`; ribs and chest wall to `chest`; orbit, eye, and lacrimal anatomy to `head`.
- Kidney, ureter, renal pelvis, collecting-system, and upper urinary calculus models map to
  `abdomen` unless the modeled scope is lower pelvic anatomy.
- True system-level or nonlocalized anatomy, including arterial or vascular system findings, maps to
  `whole_body`.
- If no anatomy is selected because the finding is generic, unlocalized, or site-variable, use
  `whole_body` unless the source clearly supports a narrower body region.
- Use multiple body regions only when the modeled finding itself spans multiple primary regions.
