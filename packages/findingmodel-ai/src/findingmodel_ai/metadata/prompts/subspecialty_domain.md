You assign only `subspecialties` for one radiology finding.

A subspecialty is a radiology practice domain whose reports are directly concerned with this finding.
Choose every domain directly supported by the finding's anatomy, organ system, imaging context, or
disease context. Do not leave `subspecialties` blank when the domain is clear.

Use the finding name, description, synonyms, selected canonical ontology, selected anatomy, body
region, and disease class as evidence.

Return only `subspecialties` and optional `field_confidence`.

Codes:
BR breast; CA cardiac; CH chest; ER emergency/acute; GI gastrointestinal/hepatobiliary; GU
genitourinary; HN head/neck; IR interventional; MI molecular/PET; MK musculoskeletal; NM nuclear
medicine; NR neuroradiology; OB obstetric/gynecologic; OI oncologic; PD pediatric; SQ
quality/safety/technique; VA vascular.

Core mappings:
- named vessels, vascular devices, aneurysm, dissection, stenosis, thrombosis, embolus, flow,
  perfusion, vascular malformation, or vascular injury -> VA
- abdominal/pelvic vessels are VA, not GI/GU unless organ disease is also modeled
- brain, spinal cord, intracranial structures, cranial nerves, meninges, ventricles, or sella -> NR
- neck, cervical lymph nodes, thyroid, salivary glands, larynx/pharynx, orbit, maxillofacial, or
  non-brain skull-base structures -> HN
- lung, pleura, mediastinum, thoracic airways, ribs, chest wall, or diaphragm -> CH
- heart, coronary arteries, pericardium, cardiac valves, or cardiac chambers -> CA
- cardiac findings commonly assessed on routine chest imaging may also be CH
- liver, biliary tree, pancreas, bowel, stomach, spleen, peritoneum, or mesentery -> GI
- kidney, ureter, bladder, prostate, scrotum/testis, or urinary tract -> GU
- gonadal/genital masses -> GU; OI possible; not ER unless acute pain, ischemia, or trauma is explicit
- uterus, ovary, adnexa, placenta, fetus, or pregnancy -> OB
- bones, joints, muscles, tendons, ligaments, extremities, or traumatic/degenerative spine -> MK
- traumatic bony chest-wall injury -> CH and MK
- breast tissue or mammographic findings -> BR
- malignancy staging, surveillance, tumor burden, or metastatic disease -> OI
- normal tissue/anatomy labels do not imply OI
- image-guided procedure, biopsy, ablation, embolization, drain, catheter, or stent -> IR
- intravascular access devices -> IR and VA
- PET/tracer-avid or molecular-imaging-centered findings -> MI
- conventional scintigraphy or SPECT-centered findings -> NM
- explicit pediatric context or pediatric-specific concepts -> PD
- acute trauma, dissection, torsion, acute life-threatening findings, or emergency presentations -> ER
- aneurysm, mass, or lesion alone is not ER unless acute complication is explicit
- artifact, acquisition, dose, quality, safety, report-quality, or technique issue -> SQ

Use every directly supported domain. Do not add domains for all theoretically possible causes,
complications, workup, or unrelated possible sites.
Horizontal domains such as VA, OI, SQ, ER, IR, MI, NM, and PD can overlay anatomic domains.
Specific anatomy/ontology outranks broad body region; body region alone is not enough.
