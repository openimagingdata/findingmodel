Assign only patient applicability metadata for one radiology finding model.

Output only `age_profile`, `sex_specificity`, and optional `field_confidence`.

Rules:
- Use the finding name, description, selected anatomy, selected canonical ontology, attributes, and
  source tags. Rejected candidates and search-only candidates do not support patient applicability.
- Output a non-null field only when the modeled finding directly supports it.
- Age defaults to all ages unless finding identity truly constrains applicability. Use
  `more_common_in` only when commonness is directly supported.
- Neonatal, pediatric, childhood, adolescent, adult, or geriatric wording directly supports the
  corresponding age applicability.
- If the available age enum cannot represent the patient applicability cleanly, leave
  `age_profile` null rather than inventing an age profile.
- Sex defaults to `sex-neutral` unless anatomy or finding identity is intrinsically sex-specific.
- Breast tissue and mammography context are not automatically female-specific.
- Male genital anatomy supports male-specific applicability.
- Uterus, ovary, adnexa, endometrium, cervix, pregnancy, and fetal-gestational findings support
  female-specific applicability only when the modeled patient is female.
- Fetal or pregnancy context is not patient female specificity by itself when the modeled patient is
  fetal or placental.
- In reassess mode, existing patient metadata is context only. Return null when it is unsupported.
- If `field_confidence` is present, keys must be real metadata fields and values must be numeric
  scores from 0 to 1.
