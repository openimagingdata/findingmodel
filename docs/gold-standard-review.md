# Gold Standard Fixture Review

Review each case. Mark issues inline. Delete this header when done.

---

## abdominal_aortic_aneurysm.fm

- **name**: abdominal aortic aneurysm
- **description**: An abdominal aortic aneurysm (AAA) is a localized dilation of the abdominal aorta, typically defined as a diameter greater than 3 cm, which can lead t...
- **synonyms**: ['AAA']
- **entity_type**: diagnosis
- **body_regions**: ['abdomen']
- **subspecialties**: ['VA', 'ER']
- **applicable_modalities**: ['CT', 'US', 'MR']
- **etiologies**: ['vascular:aneurysmal', 'degenerative']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=permanent, modifiers=['progressive']
- **index_code**: SNOMEDCT:233985008 "Abdominal aortic aneurysm"
- **anatomic_location**: ANATOMICLOCATIONS:RID905 "abdominal aorta"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate, unknown)
  - change from prior (choice: unchanged, stable, increased, decreased, new)

---

## acl_tear.fm

- **name**: anterior cruciate ligament tear
- **description**: An anterior cruciate ligament (ACL) tear is a partial or complete disruption of the ACL of the knee, typically resulting from a pivoting or hyperexten...
- **synonyms**: ['ACL tear', 'ACL rupture', 'torn ACL']
- **entity_type**: diagnosis
- **body_regions**: ['lower_extremity']
- **subspecialties**: ['MK']
- **applicable_modalities**: ['MR']
- **etiologies**: ['traumatic:acute']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adolescent', 'young_adult', 'adult', 'middle_aged'], more_common_in=['adolescent', 'young_adult', 'adult']
- **expected_time_course**: duration=months, modifiers=['evolving']
- **index_code**: SNOMEDCT:239725005 "Rupture of anterior cruciate ligament"
- **anatomic_location**: ANATOMICLOCATIONS:RID2781 "anterior cruciate ligament"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate)
  - completeness (choice: partial, complete)

---

## acute_appendicitis.fm

- **name**: acute appendicitis
- **description**: Acute appendicitis is inflammation of the vermiform appendix, typically presenting with right lower quadrant pain, and diagnosed on imaging by an enla...
- **synonyms**: ['appendicitis', 'inflamed appendix']
- **entity_type**: diagnosis
- **body_regions**: ['abdomen', 'pelvis']
- **subspecialties**: ['GI', 'ER']
- **applicable_modalities**: ['CT', 'US', 'MR']
- **etiologies**: ['inflammatory']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=['adolescent', 'young_adult']
- **expected_time_course**: duration=days, modifiers=['progressive']
- **index_code**: SNOMEDCT:85189001 "Acute appendicitis"
- **anatomic_location**: ANATOMICLOCATIONS:RID168 "appendix"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - diameter (numeric 0-30 mm)
  - perforation (choice: no, yes)

---

## aortic_dissection.fm

- **name**: aortic dissection
- **description**: Aortic dissection is a critical condition characterized by the separation of the aortic wall layers due to an intimal tear, resulting in an intramural...
- **synonyms**: ['aortic rupture', 'dissecting aneurysm']
- **entity_type**: diagnosis
- **body_regions**: ['chest']
- **subspecialties**: ['CA', 'CH', 'VA', 'ER']
- **applicable_modalities**: ['CT', 'MR', 'XR']
- **etiologies**: ['vascular:hemorrhagic']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=permanent, modifiers=['evolving']
- **index_code**: SNOMEDCT:308546005 "Dissection of aorta"
- **anatomic_location**: ANATOMICLOCATIONS:RID480 "aorta"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate, unknown)
  - change from prior (choice: unchanged, stable, increased, decreased, new)

---

## benign_prostatic_hyperplasia.fm

- **name**: benign prostatic hyperplasia
- **description**: Benign prostatic hyperplasia (BPH) is the non-neoplastic enlargement of the prostate gland due to stromal and glandular hyperplasia, commonly seen in ...
- **synonyms**: ['BPH', 'prostate enlargement', 'enlarged prostate']
- **entity_type**: diagnosis
- **body_regions**: ['pelvis']
- **subspecialties**: ['GU']
- **applicable_modalities**: ['US', 'MR', 'CT']
- **etiologies**: ['degenerative']
- **sex_specificity**: male-specific
- **age_profile**: applicability=['middle_aged', 'aged'], more_common_in=['aged']
- **expected_time_course**: duration=years, modifiers=['progressive']
- **index_code**: SNOMEDCT:266569009 "Benign prostatic hyperplasia"
- **anatomic_location**: ANATOMICLOCATIONS:RID343 "prostate"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate)
  - prostate volume (numeric 0-300 mL)

---

## birads_assessment.fm

- **name**: BI-RADS assessment
- **description**: The Breast Imaging Reporting and Data System (BI-RADS) assessment is a standardized classification system for mammographic, ultrasound, and MRI breast...
- **synonyms**: ['BIRADS', 'breast imaging assessment', 'ACR BI-RADS']
- **entity_type**: assessment
- **body_regions**: ['breast']
- **subspecialties**: ['BR']
- **applicable_modalities**: ['MG', 'US', 'MR']
- **etiologies**: None
- **sex_specificity**: female-specific
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=None
- **expected_time_course**: None
- **index_code**: SNOMEDCT:1348266008 "Breast Imaging and Reporting and Data System"
- **anatomic_location**: ANATOMICLOCATIONS:RID29895 "female breast"
- **attributes** (1):
  - category (choice: 0 - Incomplete, 1 - Negative, 2 - Benign, 3 - Probably benign, 4 - Suspicious, 5 - Highly suggestive, 6 - Known biopsy-proven)

---

## breast_density.fm

- **name**: Breast density
- **description**: Breast density refers to the proportion of fatty tissue to fibroglandular tissue in the breast as seen on a mammogram.
- **synonyms**: ['Mammographic density', 'Breast tissue density']
- **entity_type**: measurement
- **body_regions**: ['breast']
- **subspecialties**: ['BR']
- **applicable_modalities**: ['MG']
- **etiologies**: None
- **sex_specificity**: female-specific
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=years, modifiers=['evolving']
- **index_code**: SNOMEDCT:129793001 "Mammographic breast density"
- **anatomic_location**: ANATOMICLOCATIONS:RID29895 "female breast"
- **attributes** (2):
  - density score (numeric 1.0-4.99 )
  - density category (choice: a: The breasts are almost entirely fatty., b: There are scattered areas of fibroglandular density., c: The breasts are heterogeneously dense, which may obscure small masses., d: The breasts are extremely dense, which lowers the sensitivity of mammography.)

---

## cardiomegaly.fm

- **name**: cardiomegaly
- **description**: Cardiomegaly is the enlargement of the cardiac silhouette on imaging, typically defined as a cardiothoracic ratio greater than 0.5 on a PA chest radio...
- **synonyms**: ['enlarged heart', 'cardiac enlargement']
- **entity_type**: finding
- **body_regions**: ['chest']
- **subspecialties**: ['CA', 'CH']
- **applicable_modalities**: ['XR', 'CT']
- **etiologies**: ['degenerative']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['aged']
- **expected_time_course**: duration=months, modifiers=['progressive']
- **index_code**: SNOMEDCT:8186001 "Cardiomegaly"
- **anatomic_location**: ANATOMICLOCATIONS:RID1385 "heart"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate)
  - cardiothoracic ratio (numeric 0-1.0 )

---

## cerebral_infarction.fm

- **name**: cerebral infarction
- **description**: Cerebral infarction is the irreversible death of brain tissue due to ischemia, typically from arterial occlusion, visible on imaging as a region of re...
- **synonyms**: ['ischemic stroke', 'brain infarct', 'CVA']
- **entity_type**: diagnosis
- **body_regions**: ['head']
- **subspecialties**: ['NR', 'ER']
- **applicable_modalities**: ['CT', 'MR']
- **etiologies**: ['vascular:ischemic']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['aged']
- **expected_time_course**: duration=permanent, modifiers=['stable']
- **index_code**: SNOMEDCT:432504007 "Cerebral infarction"
- **anatomic_location**: ANATOMICLOCATIONS:RID6434 "brain"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - vascular territory (choice: ACA, MCA, PCA, posterior fossa, watershed)
  - acuity (choice: acute, subacute, chronic)

---

## cervical_lymphadenopathy.fm

- **name**: cervical lymphadenopathy
- **description**: Cervical lymphadenopathy is the enlargement of lymph nodes in the neck region, which may be reactive, infectious, or neoplastic in origin, and is eval...
- **synonyms**: ['enlarged cervical lymph nodes', 'neck lymphadenopathy']
- **entity_type**: finding
- **body_regions**: ['neck']
- **subspecialties**: ['HN', 'OI']
- **applicable_modalities**: ['CT', 'MR', 'US']
- **etiologies**: ['inflammatory:infectious', 'neoplastic:malignant', 'neoplastic:metastatic']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=None
- **expected_time_course**: duration=weeks, modifiers=['evolving']
- **index_code**: SNOMEDCT:127086001 "Cervical lymphadenopathy"
- **anatomic_location**: ANATOMICLOCATIONS:RID28848 "cervical lymph node group"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - laterality (choice: right, left, bilateral)
  - short axis diameter (numeric 0-100 mm)

---

## coronary_artery_calcification.fm

- **name**: coronary artery calcification
- **description**: Coronary artery calcification is the deposition of calcium within the walls of the coronary arteries, a marker of atherosclerotic disease, commonly qu...
- **synonyms**: ['coronary calcification', 'CAC', 'coronary artery calcium']
- **entity_type**: finding
- **body_regions**: ['chest']
- **subspecialties**: ['CA']
- **applicable_modalities**: ['CT']
- **etiologies**: ['degenerative']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['middle_aged', 'aged'], more_common_in=['aged']
- **expected_time_course**: duration=permanent, modifiers=['progressive']
- **index_code**: SNOMEDCT:445512009 "Calcification of coronary artery"
- **anatomic_location**: ANATOMICLOCATIONS:RID28727 "set of coronary arteries"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate)
  - Agatston score (numeric 0-10000 )

---

## distal_radius_fracture.fm

- **name**: distal radius fracture
- **description**: A distal radius fracture is a break in the distal end of the radius bone near the wrist joint, one of the most common fractures in clinical practice, ...
- **synonyms**: ['wrist fracture', 'Colles fracture', 'broken wrist']
- **entity_type**: diagnosis
- **body_regions**: ['upper_extremity']
- **subspecialties**: ['MK', 'ER']
- **applicable_modalities**: ['XR', 'CT']
- **etiologies**: ['traumatic:acute']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=['aged']
- **expected_time_course**: duration=weeks, modifiers=['resolving']
- **index_code**: SNOMEDCT:263199001 "Fracture of distal end of radius"
- **anatomic_location**: ANATOMICLOCATIONS:RID2109 "radius"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - displacement (choice: nondisplaced, displaced)
  - comminution (choice: simple, comminuted)

---

## fdg_avid_pulmonary_nodule.fm

- **name**: FDG-avid pulmonary nodule
- **description**: An FDG-avid pulmonary nodule is a lung nodule demonstrating elevated fluorodeoxyglucose uptake on PET imaging, raising concern for malignancy and requ...
- **synonyms**: ['hypermetabolic lung nodule', 'PET-positive lung nodule', 'FDG-avid lung lesion']
- **entity_type**: finding
- **body_regions**: ['chest']
- **subspecialties**: ['MI', 'OI', 'CH']
- **applicable_modalities**: ['PET', 'CT']
- **etiologies**: ['neoplastic:malignant', 'neoplastic:metastatic', 'inflammatory:infectious']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=months, modifiers=['progressive']
- **index_code**: SNOMEDCT:427359005 "Solitary nodule of lung"
- **anatomic_location**: ANATOMICLOCATIONS:RID1301 "lung"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - SUVmax (numeric 0-50 )
  - size (numeric 0-100 mm)

---

## hepatocellular_carcinoma.fm

- **name**: hepatocellular carcinoma
- **description**: Hepatocellular carcinoma (HCC) is the most common primary malignant tumor of the liver, typically arising in the setting of chronic liver disease and ...
- **synonyms**: ['HCC', 'liver cancer', 'hepatoma']
- **entity_type**: diagnosis
- **body_regions**: ['abdomen']
- **subspecialties**: ['GI', 'OI']
- **applicable_modalities**: ['CT', 'MR']
- **etiologies**: ['neoplastic:malignant']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=months, modifiers=['progressive']
- **index_code**: SNOMEDCT:25370001 "Hepatocellular carcinoma"
- **anatomic_location**: ANATOMICLOCATIONS:RID58 "liver"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - size (numeric 0-300 mm)
  - LI-RADS category (choice: LR-1, LR-2, LR-3, LR-4, LR-5, LR-M)

---

## hip_osteoarthritis.fm

- **name**: hip osteoarthritis
- **description**: Hip osteoarthritis is a degenerative joint disease characterized by progressive cartilage loss, osteophyte formation, subchondral sclerosis, and joint...
- **synonyms**: ['hip OA', 'degenerative joint disease of hip', 'hip arthritis']
- **entity_type**: diagnosis
- **body_regions**: ['lower_extremity']
- **subspecialties**: ['MK']
- **applicable_modalities**: ['XR', 'CT', 'MR']
- **etiologies**: ['degenerative']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['middle_aged', 'aged'], more_common_in=['aged']
- **expected_time_course**: duration=years, modifiers=['progressive']
- **index_code**: SNOMEDCT:239872002 "Osteoarthritis of hip"
- **anatomic_location**: ANATOMICLOCATIONS:RID2640 "hip joint"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - laterality (choice: right, left, bilateral)
  - severity (choice: mild, moderate, severe)

---

## intracranial_hemorrhage.fm

- **name**: intracranial hemorrhage
- **description**: Intracranial hemorrhage is the presence of blood within the cranial vault, which may occur in the epidural, subdural, subarachnoid, intraparenchymal, ...
- **synonyms**: ['ICH', 'intracranial bleed', 'brain hemorrhage']
- **entity_type**: finding
- **body_regions**: ['head']
- **subspecialties**: ['NR', 'ER']
- **applicable_modalities**: ['CT', 'MR']
- **etiologies**: ['vascular:hemorrhagic', 'traumatic:acute']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=['aged']
- **expected_time_course**: duration=weeks, modifiers=['evolving']
- **index_code**: SNOMEDCT:1386000 "Intracranial hemorrhage"
- **anatomic_location**: ANATOMICLOCATIONS:RID6434 "brain"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate)
  - compartment (choice: epidural, subdural, subarachnoid, intraparenchymal, intraventricular)

---

## kidney_stone.fm

- **name**: kidney stone
- **description**: A kidney stone (renal calculus) is a solid concretion formed from crystallized minerals and salts within the renal collecting system, visible on CT as...
- **synonyms**: ['renal calculus', 'nephrolithiasis', 'renal stone', 'urolithiasis']
- **entity_type**: diagnosis
- **body_regions**: ['abdomen']
- **subspecialties**: ['GU', 'ER']
- **applicable_modalities**: ['CT', 'US', 'XR']
- **etiologies**: ['metabolic']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['young_adult', 'adult', 'middle_aged', 'aged'], more_common_in=['adult', 'middle_aged']
- **expected_time_course**: duration=weeks, modifiers=['stable']
- **index_code**: SNOMEDCT:95570007 "Kidney stone"
- **anatomic_location**: ANATOMICLOCATIONS:RID205 "kidney"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - size (numeric 0-50 mm)
  - location (choice: renal pelvis, upper calyx, mid calyx, lower calyx, proximal ureter, distal ureter, UVJ)

---

## liver_hemangioma.fm

- **name**: hepatic hemangioma
- **description**: Hepatic hemangioma is the most common benign tumor of the liver, composed of vascular channels lined by endothelium, characteristically demonstrating ...
- **synonyms**: ['liver hemangioma', 'cavernous hemangioma of liver']
- **entity_type**: diagnosis
- **body_regions**: ['abdomen']
- **subspecialties**: ['GI']
- **applicable_modalities**: ['CT', 'MR', 'US']
- **etiologies**: ['congenital']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=None
- **expected_time_course**: duration=permanent, modifiers=['stable']
- **index_code**: SNOMEDCT:93469006 "Hemangioma of liver"
- **anatomic_location**: ANATOMICLOCATIONS:RID58 "liver"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - size (numeric 0-200 mm)
  - number (choice: solitary, multiple)

---

## lumbar_disc_herniation.fm

- **name**: lumbar disc herniation
- **description**: Lumbar disc herniation is the displacement of disc material beyond the intervertebral disc space in the lumbar spine, which may result in compression ...
- **synonyms**: ['herniated lumbar disc', 'lumbar disc protrusion', 'lumbar HNP', 'slipped disc']
- **entity_type**: diagnosis
- **body_regions**: ['spine']
- **subspecialties**: ['NR', 'MK']
- **applicable_modalities**: ['MR', 'CT']
- **etiologies**: ['degenerative', 'mechanical']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['young_adult', 'adult', 'middle_aged', 'aged'], more_common_in=['adult', 'middle_aged']
- **expected_time_course**: duration=months, modifiers=['evolving']
- **index_code**: SNOMEDCT:448591000124106 "Herniation of nucleus pulposus of lumbar intervertebral disc"
- **anatomic_location**: ANATOMICLOCATIONS:RID34573 "lumbar vertebral column"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - level (choice: L1-L2, L2-L3, L3-L4, L4-L5, L5-S1)
  - type (choice: protrusion, extrusion, sequestration)

---

## meningioma.fm

- **name**: meningioma
- **description**: A meningioma is a typically benign, slow-growing tumor arising from the meningeal coverings of the brain or spinal cord, commonly appearing as an extr...
- **synonyms**: ['meningeal tumor', 'intracranial meningioma']
- **entity_type**: diagnosis
- **body_regions**: ['head']
- **subspecialties**: ['NR']
- **applicable_modalities**: ['MR', 'CT']
- **etiologies**: ['neoplastic:benign']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=years, modifiers=['progressive']
- **index_code**: SNOMEDCT:302820008 "Intracranial meningioma"
- **anatomic_location**: ANATOMICLOCATIONS:RID7091 "meningeal cluster"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate)
  - size (numeric 0-150 mm)

---

## motion_artifact.fm

- **name**: motion artifact
- **description**: Motion artifact is image degradation caused by patient movement during acquisition, resulting in blurring, ghosting, or misregistration that may limit...
- **synonyms**: ['motion blur', 'patient motion', 'movement artifact']
- **entity_type**: technique_issue
- **body_regions**: ['whole_body']
- **subspecialties**: ['SQ']
- **applicable_modalities**: ['CT', 'MR']
- **etiologies**: None
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=None
- **expected_time_course**: None
- **index_code**: SNOMEDCT:47973001 "Artifact"
- **anatomic_locations**: None
- **attributes** (2):
  - severity (choice: mild, moderate, severe)
  - diagnostic quality (choice: diagnostic, limited, nondiagnostic)

---

## ovarian_cyst.fm

- **name**: ovarian cyst
- **description**: An ovarian cyst is a fluid-filled sac within or on the surface of an ovary, which may be physiologic (follicular or corpus luteum) or pathologic, comm...
- **synonyms**: ['ovarian follicular cyst', 'adnexal cyst']
- **entity_type**: finding
- **body_regions**: ['pelvis']
- **subspecialties**: ['GU', 'OB']
- **applicable_modalities**: ['US', 'CT', 'MR']
- **etiologies**: None
- **sex_specificity**: female-specific
- **age_profile**: applicability=['adolescent', 'young_adult', 'adult', 'middle_aged'], more_common_in=['young_adult', 'adult']
- **expected_time_course**: duration=weeks, modifiers=['resolving', 'recurrent']
- **index_code**: SNOMEDCT:79883001 "Cyst of ovary"
- **anatomic_location**: ANATOMICLOCATIONS:RID290 "ovary"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - laterality (choice: right, left, bilateral)
  - size (numeric 0-300 mm)

---

## pericardial_effusion.fm

- **name**: pericardial effusion
- **description**: Pericardial effusion is the accumulation of fluid within the pericardial sac surrounding the heart, which may be serous, hemorrhagic, or exudative, an...
- **synonyms**: ['fluid around heart', 'pericardial fluid collection']
- **entity_type**: finding
- **body_regions**: ['chest']
- **subspecialties**: ['CA']
- **applicable_modalities**: ['CT', 'US', 'MR']
- **etiologies**: ['inflammatory', 'neoplastic:malignant']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=['aged']
- **expected_time_course**: duration=weeks, modifiers=['evolving']
- **index_code**: SNOMEDCT:373945007 "Pericardial effusion"
- **anatomic_location**: ANATOMICLOCATIONS:RID1407 "pericardium"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate)
  - size (choice: trace, small, moderate, large)

---

## pleural_effusion.fm

- **name**: pleural effusion
- **description**: Pleural effusion is the accumulation of fluid in the pleural space, visible on imaging as dependent opacity with a meniscus sign on upright radiograph...
- **synonyms**: ['fluid in pleural space', 'hydrothorax']
- **entity_type**: finding
- **body_regions**: ['chest']
- **subspecialties**: ['CH']
- **applicable_modalities**: ['XR', 'CT', 'US']
- **etiologies**: ['inflammatory', 'neoplastic:malignant']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=['aged']
- **expected_time_course**: duration=weeks, modifiers=['evolving']
- **index_code**: SNOMEDCT:60046008 "Pleural effusion"
- **anatomic_location**: ANATOMICLOCATIONS:RID1362 "pleura"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - laterality (choice: right, left, bilateral)
  - size (choice: trace, small, moderate, large)

---

## pneumothorax.fm

- **name**: pneumothorax
- **description**: Pneumothorax is the presence of air in the pleural space, which may be spontaneous, traumatic, or iatrogenic, and is identified on imaging as a viscer...
- **synonyms**: ['collapsed lung', 'PTX']
- **entity_type**: diagnosis
- **body_regions**: ['chest']
- **subspecialties**: ['CH', 'ER']
- **applicable_modalities**: ['XR', 'CT']
- **etiologies**: ['traumatic:acute', 'mechanical']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['young_adult', 'adult', 'middle_aged'], more_common_in=['young_adult']
- **expected_time_course**: duration=days, modifiers=['resolving']
- **index_code**: SNOMEDCT:36118008 "Pneumothorax"
- **anatomic_location**: ANATOMICLOCATIONS:RID1363 "pleural space"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - laterality (choice: right, left, bilateral)
  - tension (choice: simple, tension)

---

## primary_lung_malignancy.fm

- **name**: primary lung malignancy
- **description**: Primary lung malignancy is a malignant neoplasm arising from the lung parenchyma or bronchial tree, most commonly non-small cell carcinoma, presenting...
- **synonyms**: ['lung cancer', 'bronchogenic carcinoma', 'primary lung cancer']
- **entity_type**: diagnosis
- **body_regions**: ['chest']
- **subspecialties**: ['CH', 'OI', 'MI']
- **applicable_modalities**: ['CT', 'PET', 'XR']
- **etiologies**: ['neoplastic:malignant']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=months, modifiers=['progressive']
- **index_code**: SNOMEDCT:93880001 "Primary malignant neoplasm of lung"
- **anatomic_location**: ANATOMICLOCATIONS:RID1301 "lung"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - size (numeric 0-200 mm)
  - location (choice: right upper lobe, right middle lobe, right lower lobe, left upper lobe, left lower lobe, central)

---

## pulmonary_embolism.fm

- **name**: pulmonary embolism
- **description**: Pulmonary embolism is the occlusion of a pulmonary artery or one of its branches, typically caused by thrombi that originate from the deep veins of th...
- **synonyms**: ['PE']
- **entity_type**: diagnosis
- **body_regions**: ['chest']
- **subspecialties**: ['CH', 'ER']
- **applicable_modalities**: ['CT', 'XR']
- **etiologies**: ['vascular:thrombotic']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['young_adult', 'adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=weeks, modifiers=['resolving']
- **index_code**: SNOMEDCT:59282003 "Pulmonary embolism"
- **anatomic_location**: ANATOMICLOCATIONS:RID974 "pulmonary artery"
- **attributes** (2):
  - presence (choice: absent, present, indeterminate, unknown)
  - change from prior (choice: unchanged, stable, increased, decreased, new)

---

## pyloric_stenosis.fm

- **name**: hypertrophic pyloric stenosis
- **description**: Hypertrophic pyloric stenosis is a condition of infancy characterized by hypertrophy of the pyloric muscle, resulting in gastric outlet obstruction an...
- **synonyms**: ['pyloric stenosis', 'infantile hypertrophic pyloric stenosis', 'IHPS']
- **entity_type**: diagnosis
- **body_regions**: ['abdomen']
- **subspecialties**: ['PD', 'GI']
- **applicable_modalities**: ['US']
- **etiologies**: ['congenital']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['newborn', 'infant'], more_common_in=['infant']
- **expected_time_course**: duration=weeks, modifiers=['progressive']
- **index_code**: SNOMEDCT:48644003 "Congenital hypertrophic pyloric stenosis"
- **anatomic_location**: ANATOMICLOCATIONS:RID122 "pylorus"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - pyloric muscle thickness (numeric 0-10 mm)
  - pyloric channel length (numeric 0-30 mm)

---

## rib_fracture.fm

- **name**: rib fracture
- **description**: A rib fracture is a break in the continuity of one or more ribs, typically resulting from trauma, visible on imaging as a cortical disruption or lucen...
- **synonyms**: ['broken rib', 'fractured rib']
- **entity_type**: diagnosis
- **body_regions**: ['chest']
- **subspecialties**: ['CH', 'ER', 'MK']
- **applicable_modalities**: ['XR', 'CT']
- **etiologies**: ['traumatic:acute']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=None
- **expected_time_course**: duration=weeks, modifiers=['resolving']
- **index_code**: SNOMEDCT:33737001 "Fracture of rib"
- **anatomic_location**: ANATOMICLOCATIONS:RID28591 "set of ribs"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - number of fractures (numeric 1-24 )
  - displacement (choice: nondisplaced, displaced)

---

## rotator_cuff_tear.fm

- **name**: rotator cuff tear
- **description**: A rotator cuff tear is a partial or complete disruption of one or more tendons of the rotator cuff muscles, most commonly the supraspinatus, resulting...
- **synonyms**: ['torn rotator cuff', 'rotator cuff rupture', 'supraspinatus tear']
- **entity_type**: diagnosis
- **body_regions**: ['upper_extremity']
- **subspecialties**: ['MK']
- **applicable_modalities**: ['MR', 'US']
- **etiologies**: ['degenerative', 'traumatic:acute']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=permanent, modifiers=['progressive']
- **index_code**: SNOMEDCT:926335004 "Rupture of rotator cuff of shoulder"
- **anatomic_location**: ANATOMICLOCATIONS:RID39518 "shoulder"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - completeness (choice: partial, full-thickness)
  - tendon (choice: supraspinatus, infraspinatus, subscapularis, teres minor)

---

## spinal_stenosis.fm

- **name**: spinal stenosis
- **description**: Spinal stenosis is the narrowing of the spinal canal, lateral recesses, or neural foramina, typically from degenerative changes including disc bulging...
- **synonyms**: ['lumbar stenosis', 'central canal stenosis', 'spinal canal narrowing']
- **entity_type**: diagnosis
- **body_regions**: ['spine']
- **subspecialties**: ['NR', 'MK']
- **applicable_modalities**: ['MR', 'CT']
- **etiologies**: ['degenerative']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=years, modifiers=['progressive']
- **index_code**: SNOMEDCT:76107001 "Spinal stenosis"
- **anatomic_location**: ANATOMICLOCATIONS:RID34573 "lumbar vertebral column"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - severity (choice: mild, moderate, severe)
  - level (choice: L1-L2, L2-L3, L3-L4, L4-L5, L5-S1)

---

## subdural_hematoma.fm

- **name**: subdural hematoma
- **description**: A subdural hematoma is a collection of blood in the subdural space between the dura mater and arachnoid membrane, typically resulting from tearing of ...
- **synonyms**: ['SDH', 'subdural hemorrhage']
- **entity_type**: diagnosis
- **body_regions**: ['head']
- **subspecialties**: ['NR', 'ER']
- **applicable_modalities**: ['CT', 'MR']
- **etiologies**: ['traumatic:acute', 'vascular:hemorrhagic']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=all_ages, more_common_in=['aged']
- **expected_time_course**: duration=weeks, modifiers=['evolving']
- **index_code**: SNOMEDCT:95453001 "Subdural intracranial hematoma"
- **anatomic_location**: ANATOMICLOCATIONS:RID6383_RID5824 "intracranial head"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - acuity (choice: acute, subacute, chronic, mixed)
  - laterality (choice: right, left, bilateral)

---

## thyroid_nodule.fm

- **name**: thyroid nodule
- **description**: A thyroid nodule is a discrete lesion within the thyroid gland that is radiologically distinct from the surrounding thyroid parenchyma, commonly detec...
- **synonyms**: ['thyroid mass', 'thyroid lesion']
- **entity_type**: finding
- **body_regions**: ['neck']
- **subspecialties**: ['HN']
- **applicable_modalities**: ['US', 'CT']
- **etiologies**: ['neoplastic:benign', 'neoplastic:potential']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['middle_aged', 'aged']
- **expected_time_course**: duration=years, modifiers=['stable']
- **index_code**: SNOMEDCT:237495005 "Thyroid nodule"
- **anatomic_location**: ANATOMICLOCATIONS:RID7578 "thyroid gland"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - size (numeric 0-80 mm)
  - TI-RADS category (choice: TR1, TR2, TR3, TR4, TR5)

---

## uterine_leiomyoma.fm

- **name**: uterine leiomyoma
- **description**: Uterine leiomyoma (fibroid) is a benign smooth muscle tumor of the uterus, the most common pelvic tumor in women, appearing as a well-circumscribed ma...
- **synonyms**: ['uterine fibroid', 'fibroid', 'myoma']
- **entity_type**: diagnosis
- **body_regions**: ['pelvis']
- **subspecialties**: ['GU', 'OB']
- **applicable_modalities**: ['US', 'MR']
- **etiologies**: ['neoplastic:benign']
- **sex_specificity**: female-specific
- **age_profile**: applicability=['adult', 'middle_aged'], more_common_in=['adult', 'middle_aged']
- **expected_time_course**: duration=years, modifiers=['progressive']
- **index_code**: SNOMEDCT:95315005 "Uterine leiomyoma"
- **anatomic_location**: ANATOMICLOCATIONS:RID302 "uterus"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - number (choice: solitary, multiple)
  - largest diameter (numeric 0-200 mm)

---

## vertebral_compression_fracture.fm

- **name**: vertebral compression fracture
- **description**: A vertebral compression fracture is a collapse of a vertebral body, most commonly in the thoracic or lumbar spine, typically resulting from osteoporos...
- **synonyms**: ['VCF', 'compression fracture', 'vertebral fracture']
- **entity_type**: diagnosis
- **body_regions**: ['spine']
- **subspecialties**: ['MK', 'NR']
- **applicable_modalities**: ['XR', 'CT', 'MR']
- **etiologies**: ['traumatic:acute', 'degenerative']
- **sex_specificity**: sex-neutral
- **age_profile**: applicability=['adult', 'middle_aged', 'aged'], more_common_in=['aged']
- **expected_time_course**: duration=permanent, modifiers=['stable']
- **index_code**: SNOMEDCT:42942008 "Compression fracture of vertebral column"
- **anatomic_location**: ANATOMICLOCATIONS:RID34572 "thoracic vertebral column"
- **attributes** (3):
  - presence (choice: absent, present, indeterminate)
  - acuity (choice: acute, chronic, indeterminate)
  - height loss percentage (numeric 0-100 )
