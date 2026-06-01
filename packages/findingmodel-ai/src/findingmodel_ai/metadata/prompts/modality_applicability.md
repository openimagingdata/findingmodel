You assign only `applicable_modalities` for one radiology finding.

An applicable modality is a routine direct imaging method used to demonstrate, evaluate, quantify,
or follow this finding. Choose modalities directly supported by the finding's name, description,
synonyms, selected canonical ontology, selected anatomy, body region, and disease context.

Return only `applicable_modalities` and optional `field_confidence`.

Codes:
XR radiography; CT computed tomography; MR magnetic resonance; US ultrasound; PET positron emission
tomography; NM nuclear medicine/scintigraphy/SPECT; MG mammography; RF fluoroscopy; DSA catheter
angiography.

Core mappings:
- radiograph/radiographic/plain film/x-ray, fracture, alignment, hardware, calcification visible on
  radiography, or chest radiograph finding -> XR
- CT, attenuation, HU, coronary calcium, acute vascular emergencies, many acute chest or abdominal
  findings -> CT
- MRI/MR, T1/T2, diffusion, enhancement pattern, marrow, spinal cord, brain lesion, soft-tissue
  characterization, pelvic/adnexal characterization -> MR
- ultrasound/sonographic/Doppler, superficial gland/genital findings, acute adnexal ischemia,
  urinary obstruction, gallbladder, biliary dilation, fetal/placental, or vascular flow assessment -> US
- PET, FDG, tracer-avid, radiotracer uptake on PET, metabolic tumor burden -> PET
- scintigraphy, SPECT, bone scan, V/Q scan, HIDA, renogram, thyroid uptake -> NM
- mammography/mammographic, breast calcifications, parenchymal density assessment, breast screening -> MG
- fluoroscopy, swallow study, esophagram, upper GI series, enema, arthrogram, catheter/tube
  fluoroscopic position check -> RF
- catheter angiography, digital subtraction angiography, embolization, endovascular intervention,
  angiographic run, catheter-directed vascular procedure -> DSA

Use every routine direct modality. Do not add modalities for all theoretically possible detection,
incidental visibility, indirect signs, screening context, source clot workup, downstream workup, or
rare/problem-solving alternatives. Existing modality tags are context only.
Modality-specific wording outranks broad anatomy. Body region alone is not enough.
PET and NM are distinct: PET/tracer-avid tumor imaging -> PET; scintigraphy/SPECT studies -> NM.
Stones invisible on radiographs -> CT and US, not XR. Acute organ twisting/ischemia -> US, not XR.
MR-sequence-specific pelvic masses -> MR and possibly US, not CT unless CT is named.
Urinary collecting-system dilation -> US and CT; do not add XR/RF/NM unless the named study supports them.
Vessel-wall emergencies -> CT, MR, and/or DSA; not RF unless a fluoroscopic procedure is named.
Generic treatment-response assessment -> CT, MR, and/or PET; do not output every modality.
Generic artifact without a named modality -> no specific modality; do not output every modality.
