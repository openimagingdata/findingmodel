# Finding Enrichment Pipeline Optimization Proposal

## Executive Summary

This document analyzes the current finding enrichment pipeline, identifies performance bottlenecks and prompt quality issues, and proposes specific improvements based on current prompting best practices.

**Key Findings:**
- Pipeline makes 5 sequential/parallel LLM calls totaling ~25-30s
- Prompts lack structure, examples, and domain-specific guidance
- Quality issues observed: "ALL" body region overuse, inconsistent subspecialties, etiology over-selection

**Recommendations:**
1. **Adopt Option D: Unified Classifier** - Skip intermediate LLM calls, pass raw search results to final agent
2. Restructure prompts using XML tags and few-shot examples
3. Add domain-specific guidance for medical imaging classifications

---

## Current Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENRICHMENT PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INDEX LOOKUP (~1s)                                          │
│     └── DuckDB query for existing model                         │
│                                                                 │
│  2. PARALLEL SEARCHES (~15-25s)                                 │
│     ┌─────────────────────┬─────────────────────┐               │
│     │  ONTOLOGY SEARCH    │  ANATOMIC SEARCH    │               │
│     │                     │                     │               │
│     │  A1: Query Gen      │  B1: Query Gen      │  ← LLM (small)│
│     │      (~5-10s)       │      (~5-10s)       │               │
│     │         ↓           │         ↓           │               │
│     │  A2: DB Search      │  B2: DB Search      │  ← DuckDB     │
│     │      (~2s)          │      (~2s)          │               │
│     │         ↓           │         ↓           │               │
│     │  A3: Categorize     │  B3: Selection      │  ← LLM (small)│
│     │      (~8-15s)       │      (~8-12s)       │               │
│     └─────────────────────┴─────────────────────┘               │
│                          ↓                                      │
│  3. FINAL CLASSIFICATION (~10-15s)                              │
│     └── Enrichment Agent                          ← LLM (base)  │
│                                                                 │
│  TOTAL: 5 LLM calls, ~25-30s with Anthropic                     │
└─────────────────────────────────────────────────────────────────┘
```

### Current LLM Calls

| Step | Purpose | Model Tier | Time | Prompt Location |
|------|---------|------------|------|-----------------|
| A1 | Ontology query generation | small | 5-10s | `ontology_concept_match.py:345-358` |
| A3 | Ontology categorization | small | 8-15s | `ontology_concept_match.py:221-255` |
| B1 | Anatomic query generation | small | 5-10s | `anatomic_location_search.py:60-86` |
| B3 | Anatomic location selection | small | 8-12s | `anatomic_location_search.py:171-189` |
| Final | Enrichment classification | base | 10-15s | `finding_enrichment.py:370-439` |

---

## Pipeline Optimization Options

### Option A: Optimize Prompts Only (Recommended First)
- Keep 5 LLM calls, improve each prompt
- Estimated savings: 0-5s
- Risk: Low
- Quality impact: Positive

### Option B: Merge Query Generators into Heuristics
- Replace A1 and B1 with code-based term expansion
- Reduce to 3 LLM calls
- Estimated savings: 8-12s
- Risk: Medium (may miss creative variations)
- Implementation:
  ```python
  def expand_finding_terms(finding_name: str) -> list[str]:
      """Generate search terms without LLM."""
      terms = [finding_name]
      # Add common variations
      terms.append(finding_name.replace(" ", "-"))  # hyphenated
      terms.append(finding_name + "s")  # plural
      # Add medical synonyms from lookup table
      terms.extend(MEDICAL_SYNONYMS.get(finding_name.lower(), []))
      return terms[:5]
  ```

### Option C: Single Super-Agent (Not Recommended)
- Merge all into one agent (like Agentic version)
- Reduce to 1-2 LLM calls
- Risk: High (lost robustness on edge cases)
- Evidence: "enotosis" test showed Original handled typo, Agentic failed

### Option D: Unified Classifier with Raw Results (RECOMMENDED)
- Keep LLM query generation (handles synonyms, typos, medical terminology)
- Skip intermediate categorization/selection LLM calls
- Pass raw search results directly to final classifier agent
- Reduce from 5 to 3 LLM calls

**Proposed Pipeline:**
```
┌─────────────────────────────────────────────────────────────────┐
│                 OPTIMIZED ENRICHMENT PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INDEX LOOKUP (~1s)                                          │
│     └── DuckDB query for existing model                         │
│                                                                 │
│  2. PARALLEL SEARCHES (~10-15s)                                 │
│     ┌─────────────────────┬─────────────────────┐               │
│     │  ONTOLOGY SEARCH    │  ANATOMIC SEARCH    │               │
│     │                     │                     │               │
│     │  A1: Query Gen      │  B1: Query Gen      │  ← LLM (small)│
│     │      (~5-8s)        │      (~5-8s)        │               │
│     │         ↓           │         ↓           │               │
│     │  A2: DB Search      │  B2: DB Search      │  ← DuckDB     │
│     │      (~1s)          │      (~1s)          │               │
│     │         ↓           │         ↓           │               │
│     │  (raw results)      │  (raw results)      │  ← NO LLM!    │
│     └─────────────────────┴─────────────────────┘               │
│                          ↓                                      │
│  3. UNIFIED CLASSIFICATION (~10-15s)                            │
│     └── Single agent: categorizes codes, selects      ← LLM    │
│         anatomy, classifies body/etiology/modality/subspec      │
│                                                                 │
│  TOTAL: 3 LLM calls, ~15-20s estimated                          │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Works:**
- Raw results are ~60 items (30 ontology + 30 anatomic) = ~2-3K tokens, manageable
- Final agent uses "base" tier (Sonnet 4.5), capable of filtering/selection
- Single model makes coherent decisions rather than fragmented multi-step
- Simpler pipeline, easier to debug and optimize
- Preserves LLM query generation (handles edge cases like "enotosis" typo)

---

## Prompt Improvement Proposals

### Priority 1: Unified Classifier Agent (Option D)

**Current Issues:**
- 5 LLM calls with intermediate filtering wastes time
- 29 etiologies in flat list - overwhelming
- No examples of expected output
- No guidance on subspecialty codes or anatomy selection
- Generic instructions ("be precise")

**Proposed Solution:** Single unified classifier that receives raw search results and performs:
1. Ontology code categorization (exact matches, related, marginal)
2. Anatomic location selection (primary + alternates)
3. Body region, etiology, modality, subspecialty classification

**Prompt Template Location:** `src/findingmodel/tools/prompt_templates/unified_enrichment_classifier.md.jinja`

**Proposed Prompt:**

```markdown
# SYSTEM

<role>Medical imaging finding enrichment specialist</role>

<task>
Analyze raw ontology search results and classify a radiologic finding across multiple dimensions.

You will receive:
1. A finding name and optional description
2. Raw SNOMED/RadLex search results from ontology databases
3. Raw anatomic location search results from RadLex
4. Any existing model data to preserve

You must:
1. Select ontology codes - Identify exact matches, related codes, and marginal codes
2. Select anatomic locations - Pick the best primary location and 2-3 alternates
3. Classify the finding - Body regions, etiologies, modalities, subspecialties
</task>

<ontology_code_selection>
From the raw search results, categorize codes into:

<category name="exact_matches" max="5">
Concepts whose meaning is IDENTICAL to the finding name.
CRITICAL: Never miss an exact text match! Prioritize SNOMED CT over RadLex.
</category>

<category name="should_include" max="10">
Closely related concepts - subtypes, variants, strong associations.
Example: "bacterial pneumonia" for finding "pneumonia"
</category>

<category name="marginal" max="10">
Peripherally related - broader categories, distinct but related.
Example: "respiratory infection" for finding "pneumonia" (too broad)
</category>

<exclusions>
EXCLUDE from all categories:
- Drug/medication names
- Procedures or interventions
- Pure anatomical structures (use anatomic locations instead)
- Concepts with same words but unrelated meaning
</exclusions>
</ontology_code_selection>

<anatomic_location_selection>
Select the single highest-level anatomic container that encompasses all locations
where this finding can occur.

<criteria>
- Find the "sweet spot": specific enough to be accurate, general enough to cover all occurrences
- If body region is "ALL" (truly systemic), select "whole body" as primary location
- AVOID: Overly specific locations (e.g., "left anterior descending artery" for general coronary finding)
</criteria>

<location_examples>
- "pneumonia" → primary: "lung" (not "lower lobe" - too specific)
- "medial meniscal tear" → primary: "medial meniscus" (exact structure)
- "atherosclerosis" → primary: "whole body" (systemic condition)
- "liver lesion" → primary: "liver"
</location_examples>
</anatomic_location_selection>

<body_regions>
Select specific regions where this finding occurs.

<options>Head, Neck, Chest, Breast, Abdomen, Arm, Leg, ALL</options>

<guidance>
- Use "ALL" ONLY for truly systemic conditions (atherosclerosis, osteoporosis, metastatic disease)
- Findings in multiple specific locations ≠ "ALL" (e.g., tendon calcification → Arm, Leg - NOT ALL)
- Spine findings span regions: Cervical=Head+Neck, Thoracic=Chest+Abdomen, Lumbar=Abdomen
</guidance>
</body_regions>

<etiologies>
Select 2-5 upstream causes that PRODUCE this finding. Focus on what causes the finding
to exist, NOT downstream effects or consequences.

<group name="inflammatory">
- inflammatory - general non-infectious inflammation (autoimmune, reactive)
- inflammatory:infectious - bacterial, viral, fungal infections
</group>

<group name="vascular">
- vascular:ischemic - reduced blood flow, infarction
- vascular:hemorrhagic - bleeding
- vascular:thrombotic - clot formation
- vascular:aneurysmal - vessel wall weakening and dilation
</group>

<group name="neoplastic">
Use carefully - NOT for normal variants!
- neoplastic:benign - true benign tumors
- neoplastic:malignant - primary malignancy
- neoplastic:metastatic - spread from distant primary
- neoplastic:potential - could become malignant (premalignant)
</group>

<group name="degenerative_metabolic">
- degenerative - wear and tear, aging changes
- metabolic - metabolic/endocrine disorders (calcium, iron, etc.)
</group>

<group name="traumatic">
- traumatic - acute injury, fracture
- post-traumatic - chronic sequelae of prior trauma (scarring, deformity,
  malunion, chronic instability, post-traumatic arthritis)
</group>

<group name="mechanical">
- mechanical - physical forces, compression, obstruction
</group>

<group name="iatrogenic">
- iatrogenic:post-operative - surgical changes
- iatrogenic:post-radiation - radiation therapy effects
- iatrogenic:medication-related - drug-induced changes
- iatrogenic:device-related - implant/device-related
</group>

<group name="other">
- congenital - present from birth
- developmental - abnormal development
- autoimmune - immune system attacking self
- toxic - toxin/poison exposure
- idiopathic - unknown cause
- normal-variant - anatomic variants (bone islands, limbus vertebra, etc.)
</group>
</etiologies>

<modalities>
Select modalities that ROUTINELY visualize this finding (not every possible modality).

<options>XR, CT, MR, US, PET, NM, MG, RF, DSA</options>

<domain_guidance>
- Bone/calcified lesions: Always CT, usually XR
- Soft tissue/muscle: US, MR
- Lung parenchyma: XR, CT
- Brain: Usually both CT and MR; CT for structural changes, MR for tissue changes
- Vascular: US, CT, MR, DSA
- Breast: MG, US, MR
- Tumors/staging: PET, CT, MR
- GI tract: RF (fluoroscopy), CT
</domain_guidance>
</modalities>

<subspecialties>
Select subspecialties that would consider this finding part of their domain.

This means: If a GENERAL radiologist saw this finding, which subspecialist(s) would
they consult? These are findings the subspecialist would expect to be asked about
and would expect their subspecialty colleagues to know about.

<codes>
- AB - Abdominal Radiology (liver, spleen, pancreas, kidneys, GI tract)
- BR - Breast Imaging
- CA - Cardiac Radiology (heart, coronary arteries, pericardium)
- CH - Chest/Thoracic Radiology (lungs, airways, mediastinum)
- ER - Emergency Radiology (acute trauma, stroke, acute abdomen)
- GI - Gastrointestinal Radiology (esophagus, stomach, intestines)
- GU - Genitourinary Radiology (kidneys, bladder, reproductive organs)
- HN - Head and Neck Radiology (sinuses, orbits, neck soft tissue, thyroid)
- IR - Interventional Radiology (procedures, vascular access)
- MI - Molecular Imaging (PET, nuclear medicine)
- MK - Musculoskeletal Radiology (bones, joints, muscles, tendons)
- NR - Neuroradiology (brain, spine, peripheral nerves)
- OB - OB/GYN Radiology (obstetric, gynecologic imaging)
- OI - Oncologic Imaging (tumor staging, treatment response)
- PD - Pediatric Radiology (findings common in children)
- VI - Vascular/Interventional (vessels, aneurysms, stenoses)
</codes>
</subspecialties>

<whole_body_reference>
For findings that can occur anywhere in the body (body_region = "ALL"), use this as the primary anatomic location:
  RID39569: "whole body"
</whole_body_reference>

<examples>

<example>
<finding>pneumonia</finding>
<raw_ontology_results>
- SNOMEDCT 233604007: "Pneumonia"
- SNOMEDCT 441590008: "Bacterial pneumonia"
- RADLEX RID5350: "pneumonia"
- SNOMEDCT 50417007: "Lower respiratory tract infection"
</raw_ontology_results>
<raw_anatomic_results>
- RID1301: "lung"
- RID1303: "upper lobe of right lung"
- RID1327: "upper lobe of left lung"
- RID13437: "lung parenchyma"
</raw_anatomic_results>
<output>
ontology_codes:
  exact_matches: [SNOMEDCT 233604007 "Pneumonia", RADLEX RID5350 "pneumonia"]
  should_include: [SNOMEDCT 441590008 "Bacterial pneumonia"]
  marginal: [SNOMEDCT 50417007 "Lower respiratory tract infection"]
anatomic_locations:
  primary: RID1301 "lung"
  alternates: [RID13437 "lung parenchyma"]
body_regions: ["Chest"]
etiologies: ["inflammatory:infectious", "inflammatory"]
modalities: ["XR", "CT"]
subspecialties: ["CH", "ER", "PD"]
reasoning: "Pneumonia is primarily infectious lung inflammation. Exact SNOMED and RadLex
matches found. 'lung' is the appropriate highest-level container. XR is first-line, CT for
complications. Chest radiology primary; ER for acute presentations; Pediatrics common."
</output>
</example>

<example>
<finding>enostosis</finding>
<raw_ontology_results>
- SNOMEDCT 782164008: "Enostosis"
- RADLEX RID4003: "enostosis"
- SNOMEDCT 3723001: "Bone island"
</raw_ontology_results>
<raw_anatomic_results>
- RID9082: "mandible"
- RID9361: "temporal bone"
- RID2507: "femur"
- RID39569: "whole body"
</raw_anatomic_results>
<output>
ontology_codes:
  exact_matches: [SNOMEDCT 782164008 "Enostosis", RADLEX RID4003 "enostosis", SNOMEDCT 3723001 "Bone island"]
  should_include: []
  marginal: []
anatomic_locations:
  primary: RID39569 "whole body"
  alternates: []
body_regions: ["ALL"]
etiologies: ["normal-variant"]
modalities: ["CT", "XR"]
subspecialties: ["MK"]
reasoning: "Enostosis (bone island) is a NORMAL VARIANT - a benign developmental focus
of compact bone, NOT a neoplasm. Can occur in any bone throughout the skeleton, so 'ALL'
body region and 'whole body' as highest-level container. CT best shows characteristic
features; XR often incidental. MSK radiology primary for bone lesion characterization."
</output>
</example>

<example>
<finding>calcification in a tendon</finding>
<raw_ontology_results>
- RADLEX RID5196: "calcification"
- SNOMEDCT 129755006: "Punctate calcification"
- SNOMEDCT 60963005: "Calcific tendinitis"
</raw_ontology_results>
<raw_anatomic_results>
- RID1953: "supraspinatus tendon"
- RID2068: "common extensor tendon"
- RID2654: "Achilles tendon"
- RID39569: "whole body"
</raw_anatomic_results>
<output>
ontology_codes:
  exact_matches: [RADLEX RID5196 "calcification"]
  should_include: [SNOMEDCT 60963005 "Calcific tendinitis", SNOMEDCT 129755006 "Punctate calcification"]
  marginal: []
anatomic_locations:
  primary: RID39569 "whole body"
  alternates: []
body_regions: ["Arm", "Leg"]
etiologies: ["degenerative", "metabolic"]
modalities: ["XR", "CT", "US", "MR"]
subspecialties: ["MK"]
reasoning: "Tendon calcification can occur in tendons throughout the body (shoulder, elbow,
wrist, hip, knee, ankle), so 'whole body' is the appropriate highest-level container.
Body regions are Arm and Leg because tendons are in extremities, NOT 'ALL' (not truly
systemic like atherosclerosis). Degenerative calcific tendinitis most common; metabolic
(calcium pyrophosphate) also occurs. CT/XR for calcium; US/MR for tendon detail. MSK primary."
</output>
</example>

<example>
<finding>atherosclerosis</finding>
<raw_ontology_results>
- SNOMEDCT 38716007: "Atherosclerosis"
- SNOMEDCT 443502000: "Atherosclerotic plaque"
- RADLEX RID4698: "atherosclerosis"
</raw_ontology_results>
<raw_anatomic_results>
- RID480: "artery"
- RID1283: "aorta"
- RID1385: "heart"
- RID6062: "carotid artery"
- RID39569: "whole body"
</raw_anatomic_results>
<output>
ontology_codes:
  exact_matches: [SNOMEDCT 38716007 "Atherosclerosis", RADLEX RID4698 "atherosclerosis"]
  should_include: [SNOMEDCT 443502000 "Atherosclerotic plaque"]
  marginal: []
anatomic_locations:
  primary: RID39569 "whole body"
  alternates: []
body_regions: ["ALL"]
etiologies: ["degenerative", "metabolic", "vascular:ischemic"]
modalities: ["CT", "US", "MR"]
subspecialties: ["CA", "VI", "NR", "AB"]
reasoning: "Atherosclerosis is a TRUE SYSTEMIC disease affecting arteries throughout the
body - coronary, carotid, aorta, peripheral vessels. 'ALL' body region and 'whole body'
anatomic location are both appropriate. Caused by degenerative/metabolic processes; leads
to ischemia. CT angiography, carotid US, MRA all used. Cardiac (coronary), Vascular
(peripheral), Neuro (carotid/intracranial), Abdominal (aortic/mesenteric) all routinely encounter."
</output>
</example>

# USER

<finding>{{ finding_name }}</finding>
{% if description %}
<description>{{ description }}</description>
{% endif %}

<raw_ontology_results>
{{ ontology_results }}
</raw_ontology_results>

<raw_anatomic_results>
{{ anatomic_results }}
</raw_anatomic_results>

{% if existing_model %}
<existing_model>
{{ existing_model }}
</existing_model>
{% endif %}

Provide your complete classification following the output format shown in the examples.
```

---

### (Fallback) Standalone Ontology Categorization

> **Note:** If using Option D (unified classifier), this prompt is NOT needed.
> The unified classifier handles ontology categorization internally.

**Proposed Prompt (for standalone use):**

```python
system_prompt="""<role>Medical ontology categorization expert</role>

<task>
Categorize ontology search results by relevance to the finding.
Return concept IDs in appropriate categories.
</task>

<categories>

<category name="exact_matches" max="5" priority="critical">
Concepts whose meaning is IDENTICAL to the finding.
CRITICAL: Never miss an exact text match!
Prioritize SNOMED CT over other ontologies.
</category>

<category name="should_include" max="10">
Closely related: subtypes, variants, strong associations.
Example: "bacterial pneumonia" for finding "pneumonia"
</category>

<category name="marginal" max="10">
Peripherally related: broader categories, distinct but related.
Example: "respiratory infection" for finding "pneumonia" (too broad)
</category>

</categories>

<exclusions>
NEVER include these even if they appear in results:
- Drug/medication names
- Procedures or interventions
- Pure anatomical structures
- Concepts with same words but unrelated meaning
</exclusions>

<example>
<finding>liver lesion</finding>
<results>
- 300332007: "Liver lesion" (SNOMED)
- RID3789: "hepatic mass" (RadLex)
- 126851005: "Liver disease" (SNOMED)
- 39732002: "Hepatomegaly" (SNOMED)
</results>
<categorization>
exact_matches: ["300332007"]
should_include: ["RID3789"]
marginal: ["126851005"]
excluded: ["39732002"]
rationale: "300332007 exact match. RID3789 'hepatic mass' is synonym. 126851005 too broad. 39732002 is different finding."
</categorization>
</example>

<output>
Return concept IDs only. Each concept in at most one category.
Rationale: 1-2 sentences.
</output>"""
```

---

### (Fallback) Standalone Anatomic Location Selection

> **Note:** If using Option D (unified classifier), this prompt is NOT needed.
> The unified classifier handles anatomic location selection internally.

**Proposed Prompt (for standalone use):**

```python
system_prompt="""<role>Anatomic location specialist for medical imaging</role>

<task>
Select the best anatomic location(s) from search results.
Find the "sweet spot": specific enough to be accurate, general enough to cover
all locations where the finding occurs.
</task>

<selection_criteria>
1. PRIMARY: Single highest-level anatomic container that encompasses the finding
2. ALTERNATES: 2-3 other valid locations at similar specificity level
3. If the finding is truly systemic (body region "ALL"), select "whole body"
4. AVOID: Overly narrow locations (e.g., "left anterior descending artery" for general coronary finding)
</selection_criteria>

<examples>

<example>
<finding>pneumonia</finding>
<best_choice>RID1301: lung</best_choice>
<reasoning>Lung is specific to respiratory system but general enough for all pneumonia types (lobar, broncho-, interstitial).</reasoning>
</example>

<example>
<finding>medial meniscal tear</finding>
<best_choice>RID2772: medial meniscus</best_choice>
<reasoning>Exact anatomic structure. "Knee" would be too broad.</reasoning>
</example>

<example>
<finding>atherosclerosis</finding>
<best_choice>whole body</best_choice>
<reasoning>Systemic condition affecting vessels throughout the entire body. "Whole body" is the appropriate highest-level container.</reasoning>
</example>

</examples>

<output>
primary_location: Single best location
alternate_locations: 2-3 valid alternatives
reasoning: Brief explanation of selection logic
</output>"""
```

---

### Query Generators (Keep as-is)

**Ontology Query Generator - Proposed:**

```python
system_prompt="""<task>
Generate alternative medical terms for ontology matching.
</task>

<context>
Medical ontologies use formal terminology that may differ from common names.
Generate variations to maximize match chances.
</context>

<include>
- Formal medical synonyms (e.g., "rupture" ↔ "tear")
- Standard abbreviations (e.g., "PE" for "pulmonary embolism")
- Broader/narrower terms
- Spelling variants (British/American)
</include>

<example>
<finding>quadriceps tendon rupture</finding>
<terms>
["quadriceps tendon tear", "quad tendon rupture", "quadriceps tendon disruption", "extensor mechanism injury"]
</terms>
</example>

<example>
<finding>pneumonia</finding>
<terms>
["pneumonia", "lung infection", "pulmonary infection", "pneumonitis"]
</terms>
</example>

<output>
Return 3-5 terms. Include the original finding name.
Use formal medical terminology suitable for SNOMED/RadLex ontologies.
</output>"""
```

---

## Quality Issue Fixes

### Issue 1: Body Region "ALL" Overuse

**Problem:** Model defaults to ALL when uncertain.

**Fix:** Add explicit guidance in prompt:
```
Use "ALL" ONLY for truly systemic conditions:
- Atherosclerosis, osteoporosis, metastatic disease
- NOT for findings in multiple but specific regions
For "calcification in tendon" → use [Arm, Leg], NOT ALL
```

### Issue 2: Modality Selection Gaps

**Problem:** Some findings get no modalities (e.g., enostosis).

**Fix:** Add domain guidance:
```
<modality_hints>
Bone/calcified lesions: Always CT, usually XR
Soft tissue masses: US, MR
Lung findings: XR, CT
Brain: Usually both CT and MR; CT for structural changes, MR for tissue changes
</modality_hints>
```

### Issue 3: Etiology Over-Selection

**Problem:** Some findings get 8-9 etiologies.

**Fix:** Add quantity guidance:
```
Select 3-5 MOST COMMON etiologies.
Rare or theoretical causes can be omitted.
Quality over quantity.
```

### Issue 4: Subspecialty Inconsistency

**Problem:** Same finding gets different subspecialties across runs.

**Fix:** Add explicit criteria:
```
Select subspecialties that ROUTINELY encounter this finding.
"Routinely" = multiple times per week for a typical radiologist in that subspecialty.
Do not include subspecialties that only occasionally see this finding.
```

---

## Implementation Plan

### Phase 1: Unified Classifier Implementation
1. Create `src/findingmodel/tools/prompt_templates/unified_enrichment_classifier.md.jinja`
2. Create new `enrich_finding_unified()` function that:
   - Runs ontology query generation (LLM, small tier)
   - Runs anatomic query generation (LLM, small tier) - in parallel
   - Executes DuckDB searches for both
   - Passes raw results to unified classifier (LLM, base tier)
3. Update output model to include:
   - `ontology_codes` (categorized: exact/should_include/marginal)
   - `anatomic_locations` (primary + alternates)
   - `body_regions`, `etiologies`, `modalities`, `subspecialties`
4. Test on same 5 findings from comparison
5. Compare: timing, quality, consistency

### Phase 2: Validation & Rollout
1. Run comprehensive tests with IPL ground truth
2. Compare accuracy against current 5-call pipeline
3. If quality matches or improves, replace `enrich_finding()` implementation
4. Keep old implementation available via flag for rollback

### Phase 3: Evaluation Framework
1. Create eval suite with ground truth classifications
2. Implement LLM-as-judge for quality scoring
3. Track metrics: accuracy, consistency, speed
4. Iterate on prompts based on eval results

---

## References

- [Claude 4.x Prompting Best Practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices)
- [Chain-of-Thought Prompting Guide](https://www.promptingguide.ai/techniques/cot)
- [OntoGPT: LLM-based ontological extraction](https://github.com/monarch-initiative/ontogpt)
- [CLOZE Framework for Medical Ontology Extension](https://arxiv.org/html/2511.16548)
