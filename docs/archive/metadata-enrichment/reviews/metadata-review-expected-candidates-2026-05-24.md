# Metadata Review Expected Candidates

Status: Generated promotion review aid
Date: 2026-05-24

These are conservative metadata hints extracted from latest human feedback.
They are candidates, not gold, until explicitly promoted.

## Counts

- Candidate records: 57

Extracted fields:

- age_profile: 14
- etiology_hints: 6
- expected_time_course: 45
- forbidden_etiology_hints: 4
- sex_specificity: 11

## Candidate Records

| Finding | Extracted expected metadata | Human comment | Promotion status |
| --- | --- | --- | --- |
| `acquired_posterior_neural_arch_defect` | `{"age_profile": {"applicability": "all_ages"}, "sex_specificity": "sex-neutral"}` | Why no sex-neutral, all-ages, and why no timecourse? | candidate |
| `arthritis_with_swan_neck_deformity` | `{"age_profile": {"applicability": "all_ages", "more_common_in": ["aged"]}, "expected_time_course": {"duration": "permanent", "modifiers": ["progressive"], "source_span": "permanent"}}` | age profile is all, more common in elderly.<br>Time course is permanent progressive. | candidate |
| `axillary_mass` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | Mass is months/years | candidate |
| `bilateral_adrenal_enlargement` | `{"age_profile": {"applicability": "all_ages"}, "expected_time_course": {"duration": "years", "modifiers": ["progressive"], "source_span": "years"}, "sex_specificity": "sex-neutral"}` | At least GU imaging. Sex specificity is neutral Age profile is any age.<br>Time course is years progressive, usually. | candidate |
| `breast_focal_asymmetry` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | time couirse is months/years, progressive | candidate |
| `calcified_axillary_lymph_nodes` | `{"expected_time_course": {"duration": "permanent", "modifiers": [], "source_span": "permanent"}}` | Time course is permanent. Anatomic location should be axilla.  | candidate |
| `duplication_of_great_toe` | `{"age_profile": {"applicability": "all_ages"}}` | Modality also CT, MR. Not "more common" in any age. | candidate |
| `enhancing_mediastinal_mass` | `{"age_profile": {"applicability": "all_ages"}, "expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}, "sex_specificity": "sex-neutral"}` | Sex neutral, all ages, months/years + progressive. | candidate |
| `enlarged_nerve_roots` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | Time course is months/years | candidate |
| `esophageal_plaques` | `{"expected_time_course": {"duration": "months", "modifiers": [], "source_span": "months"}}` | Time course is months | candidate |
| `focal_leptomeningeal_enhancement` | `{"expected_time_course": {"duration": "months", "modifiers": [], "source_span": "months"}, "sex_specificity": "sex-neutral"}` | sex-neutral.<br>time course is months | candidate |
| `hair_on_end_skull` | `{"sex_specificity": "sex-neutral"}` | sex-neutral | candidate |
| `hemorrhagic_liver_metastasis` | `{"expected_time_course": {"duration": "years", "modifiers": [], "source_span": "years"}}` | time course is years, progresive | candidate |
| `hypodense_supratentorial_lesion` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | time course is months/years progressive. | candidate |
| `hypoechoic_liver_lesion` | `{"expected_time_course": {"duration": "years", "modifiers": ["progressive"], "source_span": "years"}, "sex_specificity": "sex-neutral"}` | sex-neutral. time course is years, progressive | candidate |
| `incidental_pulmonary_nodules` | `{"age_profile": {"applicability": "all_ages"}, "expected_time_course": {"duration": "years", "modifiers": ["progressive"], "source_span": "years"}}` | age profile is all ages.<br>time course is years progressive. | candidate |
| `infrapatellar_mass_in_hoffa_fat_pad` | `{"expected_time_course": {"duration": "months", "modifiers": ["progressive"], "source_span": "months"}}` | time course is months, progressive. Anatomic locations is knee joint. | candidate |
| `intrauterine_growth_retardation` | `{"age_profile": {"applicability": ["adolescent", "adult", "middle_aged", "aged"]}}` | Adolescent, adult, middle-aged only | candidate |
| `intravascular_line` | `{"expected_time_course": {"duration": "months", "modifiers": [], "source_span": "months"}}` | Is "SQ" subspeciality really indicated? Duration is months. Are you sure there's no SNOMED code for this? | candidate |
| `localized_sunburst_pattern_in_skull` | `{"age_profile": {"applicability": "all_ages"}, "expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | all ages.<br>months/years, progressive | candidate |
| `mucocele_of_gallbladder` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | duration is months/years | candidate |
| `multiple_duodenal_filling_defects` | `{"age_profile": {"applicability": "all_ages"}}` | all ages<br> | candidate |
| `multiple_intracranial_enhancing_lesions` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months years"}}` | time course is months years | candidate |
| `multiple_sclerotic_foci_in_an_infant_or_child` | `{"expected_time_course": {"duration": "permanent", "modifiers": [], "source_span": "permanent"}}` | Time course is permanent | candidate |
| `osseous_lucent_lesion` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | Time course is months/years | candidate |
| `pancreatic_lesion_characterized_by_blood` | `{"age_profile": {"applicability": "all_ages"}, "expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | all ages<br>months/years progressive | candidate |
| `peribronchovascular_interstitial_thickening` | `{"etiology_hints": ["degenerative"], "expected_time_course": {"duration": "years", "modifiers": [], "source_span": "years"}}` | etiology could be degenerative. Time course could be years. Anatomic location is lung. | candidate |
| `pericardial_effusion` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months years"}}` | time course is months years | candidate |
| `pericoronal_mixed_radiopacity_jaw_lesion` | `{"expected_time_course": {"duration": "years", "modifiers": [], "source_span": "years"}}` | time course is years | candidate |
| `peripheral_rim_enhancement_of_kidney` | `{"expected_time_course": {"duration": "months", "modifiers": [], "source_span": "months"}}` | Time course is months | candidate |
| `pleural_calcification` | `{"etiology_hints": ["post-infectious", "post-exposure", "post-treatment"]}` | Hey, where are my etiologies? Could be post-infectious, post-exposure, post-treatment. | candidate |
| `pleural_effusion_with_disease_in_abdomen` | `{"expected_time_course": {"duration": "months", "modifiers": [], "source_span": "months"}}` | Time course is months | candidate |
| `polyostotic_bone_lesions_in_children` | `{"expected_time_course": {"duration": "years", "modifiers": [], "source_span": "years"}}` | Time course is years | candidate |
| `posterior_fossa_cystic_lesion` | `{"forbidden_etiology_hints": ["congenital"]}` | Etiologies should NOT be congenital. | candidate |
| `primary_brain_tumor` | `{"expected_time_course": {"duration": "years", "modifiers": ["progressive"], "source_span": "years"}}` | time course is years progressive | candidate |
| `prolonged_cerebral_vascular_transit_time` | `{"age_profile": {"applicability": "all_ages"}, "etiology_hints": ["vascular"], "sex_specificity": "sex-neutral"}` | I think etiology is just vascular.<br>sex-neutral. age profile is any age.<br>This is also a CT (CTA) finding--I don't think US applies. | candidate |
| `pulmonary_vascular_engorgement` | `{"etiology_hints": ["vascular", "heart failure", "cardiovascular"], "forbidden_etiology_hints": ["inflammatory"]}` | Most likely cause is cardiovascular, heart failure, not inflammatory--that should be part of etiologies. | candidate |
| `renal_pseudotumor` | `{"age_profile": {"applicability": "all_ages"}, "expected_time_course": {"duration": "permanent", "modifiers": [], "source_span": "permanent"}, "sex_specificity": "sex-neutral"}` | sex neutral, all ages, permanent | candidate |
| `sacroiliac_joint_disease` | `{"expected_time_course": {"duration": "years", "modifiers": [], "source_span": "years"}}` | Time course is years | candidate |
| `sinus_keros_classification_on_ct` | `{"age_profile": {"applicability": "all_ages"}, "sex_specificity": "sex-neutral"}` | sex-neutral, all agents, | candidate |
| `small_anterior_fontanelle` | `{"expected_time_course": {"duration": "years", "modifiers": [], "source_span": "years"}}` | anatomic location is at least skull<br>duration is years | candidate |
| `soft_tissue_tumor_with_prominent_vascularity` | `{"expected_time_course": {"duration": "years", "modifiers": [], "source_span": "years"}}` | time course is years.  | candidate |
| `striated_nephrogram` | `{"expected_time_course": {"duration": "weeks", "modifiers": [], "source_span": "weeks"}}` | Time course is weeks. | candidate |
| `subsegmental_liver_perfusion_abnormality` | `{"etiology_hints": ["vascular"], "expected_time_course": {"duration": "months", "modifiers": [], "source_span": "months"}, "forbidden_etiology_hints": ["ischemic"]}` | Duration is months<br>It's just vascular, not ischemic | candidate |
| `supraglottic_mass` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | Expected time course is months/years<br>Anatomic location s is larynx at worst, at least neck. | candidate |
| `t1_hyperintense_adnexal_mass` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months to years"}}` | Time course ins months to years | candidate |
| `t2_intermediate_endometrial_uterine_mass` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | Time course is months/years | candidate |
| `t2_isointense_intracranial_lesion` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | time course is months/years | candidate |
| `thin_walled_lung_cavity` | `{"forbidden_etiology_hints": ["neoplastic:benign"]}` | Not neoplastic:benign | candidate |
| `thymic_enlargement` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | Expected time course is months/years | candidate |
| `tibiotalar_tilt` | `{"expected_time_course": {"duration_candidates": ["weeks", "months"], "source_span": "weeks/months"}}` | Time course is weeks/months<br>Location is ankle joint | candidate |
| `tracheostomy_tube` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | subspecialty is also chest<br>duration is months/years<br>Anatomic locations is neck | candidate |
| `transudative_pleural_effusion` | `{"etiology_hints": ["vascular", "inflammatory"], "expected_time_course": {"duration": "weeks", "modifiers": [], "source_span": "weeks"}}` | WHy no etiologies? Inflammation is the usual, but also vascular.<br>time course is weeks | candidate |
| `unilateral_sacroiliac_joint_disease` | `{"sex_specificity": "sex-neutral"}` | Anatomic location is SI joints.<br>Sex-neutral | candidate |
| `upper_abdominal_mass_in_a_neonate_or_child` | `{"expected_time_course": {"duration_candidates": ["months", "years"], "source_span": "months/years"}}` | Time course in months/years. | candidate |
| `upper_cervical_spine_ao_injury_classification_in_ct` | `{"age_profile": {"applicability": "all_ages"}, "expected_time_course": {"duration_candidates": ["weeks", "months"], "source_span": "weeks to months"}, "sex_specificity": "sex-neutral"}` | Anatomic location is just cervical vertebral column.<br>sex-neutral<br>any age<br>weeks to months for time course | candidate |
| `wedging_of_vertebral_body` | `{"expected_time_course": {"duration": "permanent", "modifiers": [], "source_span": "permanent"}}` | time course in permanent<br>anatomic locaiton should be spine<br> | candidate |
