# Proposed Prompt Examples for Review

Based on analysis of actual results in `scripts/enrichment_comparison/comparison_results.json` and ground truth from `scripts/ipl_finding_models.json`.

## Quality Issues Identified

| Finding | Issue | Current Output | Correct Output |
|---------|-------|----------------|----------------|
| calcification in a tendon | Body region | ALL | Arm, Leg |
| thoracic spine degenerative change | Body region | Chest | Chest, Abdomen |
| enostosis | Etiology | neoplastic:benign, neoplastic:malignant | normal-variant |
| enostosis | Modalities | (empty) | XR, CT |

---

## Proposed Examples for Final Enrichment Agent

### Example 1: pneumonia (Baseline - Standard Chest Finding)

**Purpose:** Establish basic pattern for single-region infectious finding

```xml
<example>
<finding>pneumonia</finding>
<context>
SNOMED: 233604007 (Pneumonia)
RadLex: RID5350 (pneumonia)
Anatomic locations: lung, lung parenchyma
</context>
<classification>
body_regions: ["Chest"]
etiologies: ["inflammatory:infectious", "inflammatory"]
modalities: ["XR", "CT"]
subspecialties: ["CH", "ER", "PD"]
reasoning: Pneumonia is lung parenchymal infection/inflammation - clearly Chest
region only. Primary etiology is infectious, with general inflammatory also
applicable. XR is first-line, CT for complications. Chest radiology primary;
ER sees acute cases; Pediatrics sees it commonly in children.
</classification>
</example>
```

**Teaching points:**
- Single clear body region
- Infectious → inflammatory:infectious
- Limited modality selection (not every possible modality)
- Focused subspecialty selection

---

### Example 2: calcification in a tendon (CORRECTIVE - NOT "ALL")

**Purpose:** Correct the observed error where extremity findings get "ALL"

```xml
<example>
<finding>calcification in a tendon</finding>
<context>
RadLex: RID5196 (calcification)
Anatomic locations: supraspinatus tendon, common extensor tendon, Achilles tendon
</context>
<classification>
body_regions: ["Arm", "Leg"]
etiologies: ["degenerative", "metabolic"]
modalities: ["XR", "US", "MR"]
subspecialties: ["MK"]
reasoning: Tendon calcifications occur in specific locations - shoulder/elbow
tendons (Arm) and ankle/knee tendons (Leg). This is NOT "ALL" because the
finding is localized to extremity tendons, not truly systemic. Degenerative
calcific tendinitis is most common; metabolic (calcium pyrophosphate) also
occurs. XR shows calcium; US/MR characterize tendon. MSK radiology primary.
</classification>
</example>
```

**Teaching points:**
- "Arm" and "Leg" are CORRECT, NOT "ALL"
- Findings in multiple extremity locations ≠ systemic disease
- "ALL" reserved for truly systemic conditions

---

### Example 3: enostosis (CRITICAL CORRECTIVE - Normal Variant)

**Purpose:** Correct serious misclassification of normal variant as neoplastic

```xml
<example>
<finding>enostosis</finding>
<context>
SNOMED: 782164008 (Enostosis)
RadLex: RID4003 (enostosis)
Also known as: bone island - focus of compact bone within cancellous bone
</context>
<classification>
body_regions: ["ALL"]
etiologies: ["normal-variant"]
modalities: ["XR", "CT"]
subspecialties: ["MK"]
reasoning: Enostosis (bone island) is a NORMAL VARIANT, not a neoplasm. It is
a benign developmental focus of compact bone - completely benign with no
malignant potential. Can occur in any bone throughout the skeleton, so "ALL"
is appropriate here. Typically incidental on XR or CT. Important to recognize
as normal to avoid unnecessary oncologic workup.
</classification>
</example>
```

**Teaching points:**
- "normal-variant" etiology EXISTS and should be used
- Bone islands are NOT neoplastic (benign or malignant)
- "ALL" appropriate when finding can occur in any bone
- Always include modalities (XR, CT for bone lesions)

---

### Example 4: thoracic spine degenerative change (Multi-Region Anatomy)

**Purpose:** Show that spine findings span multiple body regions

```xml
<example>
<finding>thoracic spine degenerative change</finding>
<context>
SNOMED: 387802007 (Thoracic spondylosis)
Anatomic locations: thoracic vertebral column, thoracic vertebrae
</context>
<classification>
body_regions: ["Chest", "Abdomen"]
etiologies: ["degenerative"]
modalities: ["XR", "CT", "MR"]
subspecialties: ["MK", "NR"]
reasoning: The thoracic spine anatomically spans TWO body regions - upper
thoracic vertebrae are within the Chest, while lower thoracic vertebrae
extend into the Abdomen at the thoracolumbar junction. Include BOTH regions
for thoracic spine findings. Etiology is purely degenerative. Standard spine
imaging: XR for alignment, CT for bony detail, MR for discs/nerves. MSK
primary; Neuroradiology when neural elements affected.
</classification>
</example>
```

**Teaching points:**
- Spine findings often span multiple regions
- Thoracic = Chest + Abdomen
- Cervical = Head + Neck
- Lumbar = Abdomen
- Don't default to single region for spine

---

### Example 5: atherosclerosis (When "ALL" IS Correct)

**Purpose:** Demonstrate when "ALL" is the appropriate choice

```xml
<example>
<finding>atherosclerosis</finding>
<context>
SNOMED: 38716007 (Atherosclerosis)
Anatomic locations: aorta, coronary arteries, carotid arteries, iliac arteries
</context>
<classification>
body_regions: ["ALL"]
etiologies: ["degenerative", "metabolic", "vascular:ischemic"]
modalities: ["CT", "US", "MR"]
subspecialties: ["CA", "VI", "NR", "AB"]
reasoning: Atherosclerosis is a TRUE SYSTEMIC disease affecting arteries
throughout the entire body - coronary (Chest), carotid (Head/Neck), aorta
(Chest/Abdomen), iliac/femoral (Leg). "ALL" is correct because no single
region captures this disease. Multiple subspecialties involved: Cardiac for
coronary disease, Vascular for peripheral vessels, Neuro for carotid,
Abdominal for aortic/mesenteric involvement.
</classification>
</example>
```

**Teaching points:**
- "ALL" for TRUE systemic diseases (vascular, metabolic, metastatic)
- Atherosclerosis affects vessels in every body region
- Multiple subspecialties appropriate for systemic disease

---

## Proposed Guidance Text for Body Regions

Add this explicit guidance BEFORE the examples:

```xml
<body_region_guidance>
SELECT SPECIFIC REGIONS when the finding is anatomically localized:
- Lung findings → Chest
- Brain findings → Head
- Liver findings → Abdomen
- Shoulder/elbow/wrist findings → Arm
- Hip/knee/ankle findings → Leg

SELECT MULTIPLE REGIONS when anatomy spans regions:
- Thoracic spine → Chest AND Abdomen
- Cervical spine → Head AND Neck
- Aortic aneurysm → Chest AND Abdomen (depending on location)

SELECT "ALL" ONLY for truly systemic conditions:
- Atherosclerosis (affects all vessels)
- Osteoporosis (affects all bones)
- Metastatic disease (can occur anywhere)
- Soft tissue mass (can occur in any region)

DO NOT use "ALL" for:
- Extremity findings (use Arm and/or Leg)
- Findings that happen to occur in multiple specific locations
- When you're uncertain (pick the most likely regions instead)
</body_region_guidance>
```

---

## Proposed Guidance Text for Etiologies

Add this guidance with grouped categories:

```xml
<etiology_guidance>
Select 2-5 MOST COMMON etiologies. Quality over quantity.

INFLAMMATORY/INFECTIOUS:
- inflammatory:infectious → bacterial, viral, fungal infections
- inflammatory → non-infectious inflammation (autoimmune, reactive)

VASCULAR:
- vascular:ischemic → reduced blood flow, infarction
- vascular:hemorrhagic → bleeding
- vascular:thrombotic → clot formation
- vascular:aneurysmal → vessel dilation

NEOPLASTIC (use carefully):
- neoplastic:benign → true benign tumors (NOT normal variants!)
- neoplastic:malignant → primary malignancy
- neoplastic:metastatic → spread from distant primary

DEGENERATIVE/METABOLIC:
- degenerative → wear and tear, aging
- metabolic → metabolic/endocrine disorders

IMPORTANT - NORMAL VARIANTS:
- normal-variant → anatomic variants, developmental variations
- Examples: bone island (enostosis), Schmorl's node, limbus vertebra
- These are NOT neoplastic even if they look like "lesions"

TRAUMATIC/OTHER:
- traumatic → injury, fracture
- congenital → present from birth
- iatrogenic → caused by medical intervention
</etiology_guidance>
```

---

## Questions for Review

1. **Are these 5 examples sufficient?** They cover: single region, multi-region, extremity, systemic, and normal variant cases.

2. **Is the enostosis example medically accurate?** I classified it as normal-variant with XR/CT modalities and MK subspecialty.

3. **Should thoracic spine include Neck as well as Chest/Abdomen?** I only included Chest and Abdomen based on the ground truth.

4. **For atherosclerosis, are 4 subspecialties (CA, VI, NR, AB) appropriate?** Or is this over-selection?

5. **Are there other edge cases we should include?** Possibilities:
   - Fracture (traumatic etiology)
   - Metastatic disease (systemic but neoplastic)
   - Congenital finding

---

## Next Steps

Once you approve/modify these examples, I will:
1. Integrate them into the actual prompt code
2. Test on the same 5 findings to measure improvement
3. Iterate based on results
