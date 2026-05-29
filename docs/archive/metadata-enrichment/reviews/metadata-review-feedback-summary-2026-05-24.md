# Metadata Review Feedback Summary

Status: Generated review aid
Date: 2026-05-24

This table summarizes latest human-feedback records from the review evidence register.

## Counts

- Latest feedback records: 83

Disposition counts:

- code/anatomy-review: 11
- expected-code-or-location-extraction: 9
- expected-metadata-extraction: 61
- source-model-issue: 2

Affected-field counts:

- age_profile: 15
- anatomic_locations: 28
- applicable_modalities: 7
- body_regions: 1
- entity_type: 2
- etiologies: 9
- expected_time_course: 46
- index_codes: 9
- sex_specificity: 11
- subspecialties: 4

## Feedback Records

| Finding | Fields | Labels | Disposition queue | Comment |
| --- | --- | --- | --- | --- |
| `acquired_posterior_neural_arch_defect` | age_profile, expected_time_course, sex_specificity | age_or_sex_applicability, missing_assignment, time_course | expected-metadata-extraction | Why no sex-neutral, all-ages, and why no timecourse? |
| `amelia` | anatomic_locations, index_codes | anatomic_location, code_mapping_issue, incorrect_assignment, index_code, over_specific_or_bad_code | code/anatomy-review | Anatomic location code is wrong--can't localize to the arm, should be upper extremity, lower extremity. |
| `arthritis_with_swan_neck_deformity` | age_profile, expected_time_course | age_or_sex_applicability, time_course | expected-metadata-extraction | age profile is all, more common in elderly.<br>Time course is permanent progressive. |
| `automated_insall_salvatti_index` | anatomic_locations, index_codes | anatomic_location, code_mapping_issue, index_code | code/anatomy-review | anatomic location here is just "knee joint", which encompasses the other codes given. |
| `axillary_mass` | expected_time_course | time_course | expected-metadata-extraction | Mass is months/years |
| `axillary_nodal_dissection` | index_codes | code_mapping_issue, index_code, missing_assignment | code/anatomy-review | Why don't we have the RadElement codes here? I'm sure they were in the original--did they get dropped? |
| `bilateral_adrenal_enlargement` | age_profile, expected_time_course, sex_specificity, subspecialties | age_or_sex_applicability, domain_or_region, time_course | expected-metadata-extraction | At least GU imaging. Sex specificity is neutral Age profile is any age.<br>Time course is years progressive, usually. |
| `breast_focal_asymmetry` | expected_time_course | time_course | expected-metadata-extraction | time couirse is months/years, progressive |
| `breast_skin_thickening` | index_codes | code_mapping_issue, index_code | code/anatomy-review | This almost certainly came with a RadElement index code--what happened to it? |
| `calcified_axillary_lymph_nodes` | anatomic_locations, expected_time_course | anatomic_location, time_course | expected-metadata-extraction | Time course is permanent. Anatomic location should be axilla.  |
| `chondromatosis` | anatomic_locations, body_regions | anatomic_location, domain_or_region | expected-metadata-extraction | Should be upper extremity too for body region, and the locations should include the upper extremity joints a well. |
| `composite_metrics_hippocampal_occupancy_score` | anatomic_locations, index_codes | anatomic_location, code_mapping_issue, index_code, missing_assignment | code/anatomy-review | Why don't we have the actual names for these index codes? And why do we not have the hippocampus for the anatomic location? |
| `congenital_absence_of_phalanx__digit__hand__or_foot` | anatomic_locations, applicable_modalities | anatomic_location, modality | expected-metadata-extraction | Could also be CT or MR. Anatomic locations needs to be the hand as well. |
| `congenital_fused_vertebrae` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | Location can be at least spine, or cervical, thoracic, and lumbar spines. |
| `cystic_abdominal_mass_in_a_fetus_or_newborn` | applicable_modalities | modality | expected-metadata-extraction | Modality is also CT |
| `disrupted_epiphyseal_metaphyseal_junction` | - | incorrect_assignment, over_specific_or_bad_code | code/anatomy-review | Totally inappropriate to localize this to the fibula. |
| `double_balloon_esophageal_catheter` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | anatomic location should be esophagus |
| `duodenal_narrowing_or_obstruction` | applicable_modalities | modality | expected-metadata-extraction | Can also be ultrasound in a newborn for modality |
| `duplication_of_great_toe` | applicable_modalities, age_profile | age_or_sex_applicability, modality | expected-metadata-extraction | Modality also CT, MR. Not "more common" in any age. |
| `enhancing_mediastinal_mass` | age_profile, expected_time_course, sex_specificity | age_or_sex_applicability, time_course | expected-metadata-extraction | Sex neutral, all ages, months/years + progressive. |
| `enlarged_nerve_roots` | expected_time_course | time_course | expected-metadata-extraction | Time course is months/years |
| `esophageal_plaques` | expected_time_course | time_course | expected-metadata-extraction | Time course is months |
| `focal_defect_in_nephrogram` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | Anatomic location is cortex of kidney.  |
| `focal_leptomeningeal_enhancement` | expected_time_course, sex_specificity | age_or_sex_applicability, time_course | expected-metadata-extraction | sex-neutral.<br>time course is months |
| `hair_on_end_skull` | sex_specificity | age_or_sex_applicability | expected-metadata-extraction | sex-neutral |
| `heart_failure_in_the_first_week_of_life` | index_codes | code_mapping_issue, index_code, over_specific_or_bad_code | code/anatomy-review | the SNOMED heart-failure is too general for this. |
| `hemivertebra` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | anatomic location is spine |
| `hemorrhagic_liver_metastasis` | expected_time_course | time_course | expected-metadata-extraction | time course is years, progresive |
| `hypodense_supratentorial_lesion` | expected_time_course | time_course | expected-metadata-extraction | time course is months/years progressive. |
| `hypoechoic_liver_lesion` | expected_time_course, sex_specificity | age_or_sex_applicability, time_course | expected-metadata-extraction | sex-neutral. time course is years, progressive |
| `implantable_cardioverter_defibrillator_leads` | anatomic_locations | anatomic_location, over_specific_or_bad_code | code/anatomy-review | Anatomic locations is chest, not right atrium. |
| `incidental_pulmonary_nodules` | age_profile, expected_time_course | age_or_sex_applicability, time_course | expected-metadata-extraction | age profile is all ages.<br>time course is years progressive. |
| `infrapatellar_mass_in_hoffa_fat_pad` | anatomic_locations, expected_time_course | anatomic_location, time_course | expected-metadata-extraction | time course is months, progressive. Anatomic locations is knee joint. |
| `intrauterine_growth_retardation` | age_profile | age_or_sex_applicability | expected-metadata-extraction | Adolescent, adult, middle-aged only |
| `intravascular_line` | expected_time_course, index_codes, subspecialties | code_mapping_issue, domain_or_region, index_code, time_course | code/anatomy-review | Is "SQ" subspeciality really indicated? Duration is months. Are you sure there's no SNOMED code for this? |
| `localized_sunburst_pattern_in_skull` | age_profile, expected_time_course | age_or_sex_applicability, time_course | expected-metadata-extraction | all ages.<br>months/years, progressive |
| `malignant_primary_bone_neoplasm` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | anatomic location musculoskeletal system, surely? |
| `mucocele_of_gallbladder` | expected_time_course | time_course | expected-metadata-extraction | duration is months/years |
| `multiple_duodenal_filling_defects` | age_profile | age_or_sex_applicability | expected-metadata-extraction | all ages<br> |
| `multiple_intracranial_enhancing_lesions` | expected_time_course | time_course | expected-metadata-extraction | time course is months years |
| `multiple_sclerotic_foci_in_an_infant_or_child` | expected_time_course | time_course | expected-metadata-extraction | Time course is permanent |
| `osseous_lucent_lesion` | expected_time_course | time_course | expected-metadata-extraction | Time course is months/years |
| `pancreatic_lesion_characterized_by_blood` | age_profile, expected_time_course | age_or_sex_applicability, time_course | expected-metadata-extraction | all ages<br>months/years progressive |
| `parapelvic_renal_cyst` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | anatomic location should be kidney at worst, maybe renal collecting system if we have that. |
| `peribronchovascular_interstitial_thickening` | anatomic_locations, etiologies, expected_time_course | anatomic_location, etiology, time_course | expected-metadata-extraction | etiology could be degenerative. Time course could be years. Anatomic location is lung. |
| `pericardial_effusion` | expected_time_course | time_course | expected-metadata-extraction | time course is months years |
| `pericoronal_mixed_radiopacity_jaw_lesion` | expected_time_course | time_course | expected-metadata-extraction | time course is years |
| `periosteal_new_bone_formation_in_a_child` | index_codes | code_mapping_issue, incorrect_assignment, index_code, over_specific_or_bad_code | code/anatomy-review | "tibia" is way off--musculoskeletal system. "Periosteal reaction" is NOT the same for the Radlex index code. |
| `peripheral_rim_enhancement_of_kidney` | expected_time_course | time_course | expected-metadata-extraction | Time course is months |
| `pleural_calcification` | applicable_modalities, etiologies | etiology, missing_assignment, modality | expected-metadata-extraction | Hey, where are my etiologies? Could be post-infectious, post-exposure, post-treatment. |
| `pleural_effusion_with_disease_in_abdomen` | expected_time_course | time_course | expected-metadata-extraction | Time course is months |
| `polyostotic_bone_lesions_in_children` | expected_time_course | time_course | expected-metadata-extraction | Time course is years |
| `posterior_fossa_cystic_lesion` | etiologies | etiology, incorrect_assignment | expected-metadata-extraction | Etiologies should NOT be congenital. |
| `primary_brain_tumor` | expected_time_course | time_course | expected-metadata-extraction | time course is years progressive |
| `prolonged_cerebral_vascular_transit_time` | applicable_modalities, age_profile, entity_type, etiologies, sex_specificity | age_or_sex_applicability, entity_type, etiology, modality | expected-metadata-extraction | I think etiology is just vascular.<br>sex-neutral. age profile is any age.<br>This is also a CT (CTA) finding--I don't think US applies. |
| `pulmonary_vascular_engorgement` | etiologies | etiology | expected-metadata-extraction | Most likely cause is cardiovascular, heart failure, not inflammatory--that should be part of etiologies. |
| `renal_pseudotumor` | age_profile, expected_time_course, sex_specificity | age_or_sex_applicability, time_course | expected-metadata-extraction | sex neutral, all ages, permanent |
| `rib_fracture` | age_profile | age_or_sex_applicability | expected-metadata-extraction | Doesn't really resolve. Not really more common in adult/aged. |
| `sacroiliac_joint_disease` | expected_time_course | time_course | expected-metadata-extraction | Time course is years |
| `short_thumb` | anatomic_locations | anatomic_location, missing_assignment | expected-code-or-location-extraction | Don't we have first digit? Hand is OK. |
| `sinus_keros_classification_on_ct` | age_profile, sex_specificity | age_or_sex_applicability | expected-metadata-extraction | sex-neutral, all agents, |
| `small_anterior_fontanelle` | anatomic_locations, expected_time_course | anatomic_location, time_course | expected-metadata-extraction | anatomic location is at least skull<br>duration is years |
| `soft_tissue_abnormality` | anatomic_locations, applicable_modalities, etiologies, subspecialties | anatomic_location, domain_or_region, etiology, modality, source_model_issue | source-model-issue | This is OK because the description says "chest radiograph", but this should be fixed so the description is updated to be more general. Modalities are XR, CT, MR, US. Subspecialties are MK mostly. Region is whole body. No anatomic location. |
| `soft_tissue_tumor_with_prominent_vascularity` | expected_time_course | time_course | expected-metadata-extraction | time course is years.  |
| `striated_nephrogram` | expected_time_course | time_course | expected-metadata-extraction | Time course is weeks. |
| `subsegmental_liver_perfusion_abnormality` | etiologies, expected_time_course | etiology, time_course | expected-metadata-extraction | Duration is months<br>It's just vascular, not ischemic |
| `supraglottic_mass` | anatomic_locations, expected_time_course | anatomic_location, time_course | expected-metadata-extraction | Expected time course is months/years<br>Anatomic location s is larynx at worst, at least neck. |
| `t1_hyperintense_adnexal_mass` | expected_time_course | time_course | expected-metadata-extraction | Time course ins months to years |
| `t2_intermediate_endometrial_uterine_mass` | expected_time_course | time_course | expected-metadata-extraction | Time course is months/years |
| `t2_isointense_intracranial_lesion` | expected_time_course | time_course | expected-metadata-extraction | time course is months/years |
| `thin_walled_lung_cavity` | etiologies | etiology | expected-metadata-extraction | Not neoplastic:benign |
| `thoracolumbar_spine_ao_injury_classification_on_ct__tlics_` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | Location thoracicx and lumbar spines |
| `thrombocytosis` | anatomic_locations, entity_type | anatomic_location, entity_type, source_model_issue | source-model-issue | anatomic location is whole body<br>We should actually skip this one--not even an imaging finding. |
| `thymic_enlargement` | expected_time_course | time_course | expected-metadata-extraction | Expected time course is months/years |
| `thyroid_bed_clips` | anatomic_locations, index_codes | anatomic_location, code_mapping_issue, index_code, missing_assignment | code/anatomy-review | Aren't there index codes for thyroid surgery? And the anatomic location is neck at least. |
| `tibiotalar_tilt` | anatomic_locations, expected_time_course | anatomic_location, time_course | expected-metadata-extraction | Time course is weeks/months<br>Location is ankle joint |
| `tracheostomy_tube` | anatomic_locations, expected_time_course, subspecialties | anatomic_location, domain_or_region, time_course | expected-metadata-extraction | subspecialty is also chest<br>duration is months/years<br>Anatomic locations is neck |
| `transudative_pleural_effusion` | etiologies, expected_time_course | etiology, missing_assignment, time_course | expected-metadata-extraction | WHy no etiologies? Inflammation is the usual, but also vascular.<br>time course is weeks |
| `unilateral_sacroiliac_joint_disease` | anatomic_locations, sex_specificity | age_or_sex_applicability, anatomic_location | expected-metadata-extraction | Anatomic location is SI joints.<br>Sex-neutral |
| `upper_abdominal_mass_in_a_neonate_or_child` | expected_time_course | time_course | expected-metadata-extraction | Time course in months/years. |
| `upper_cervical_spine_ao_injury_classification_in_ct` | anatomic_locations, age_profile, expected_time_course, sex_specificity | age_or_sex_applicability, anatomic_location, time_course | expected-metadata-extraction | Anatomic location is just cervical vertebral column.<br>sex-neutral<br>any age<br>weeks to months for time course |
| `vertical_trabeculation_of_vertebral_body` | anatomic_locations | anatomic_location | expected-code-or-location-extraction | Anatomic location spine |
| `wedging_of_vertebral_body` | anatomic_locations, expected_time_course | anatomic_location, time_course | expected-metadata-extraction | time course in permanent<br>anatomic locaiton should be spine<br> |
