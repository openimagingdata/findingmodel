# Enrichment Approach Comparison

Comparing the **Original** (parallel pre-fetch) vs **Agentic** (tool-calling) approaches.

## Summary Table

| Finding | Orig Time | Agentic Time | Winner |
|---------|-----------|--------------|--------|
| pulmonary segmental consolidation | 24.4s | 26.9s | Original |
| calcification in a tendon | 26.3s | 26.9s | Tie |
| abnormal intracranial enhancement | 19.7s | 28.4s | Original |
| thoracic spine degenerative change | 28.0s | 26.3s | Agentic |
| enotosis | 28.9s | 31.8s | Original |

**Average:** Original 25.5s, Agentic 28.1s

---

## Detailed Comparison

### 1. pulmonary segmental consolidation

| Aspect | Original | Agentic | Notes |
|--------|----------|---------|-------|
| Time | 24.4s | 26.9s | |
| SNOMED | 3 codes | 3 codes | Same codes |
| RadLex | 1 code | 1 code | Same code |
| Locations | 4 | 4 | Same locations |
| Body Regions | Chest | Chest | Match |
| Etiologies | 6 | 7 | Agentic adds `iatrogenic:post-radiation` |
| Modalities | XR, CT, US | XR, CT, US, MR | Agentic adds MR |
| Subspecialties | CH, ER, OI, PD | CH, ER, PD | Original has OI |

**Verdict:** Similar quality. Original slightly faster.

---

### 2. calcification in a tendon

| Aspect | Original | Agentic | Notes |
|--------|----------|---------|-------|
| Time | 26.3s | 26.9s | |
| SNOMED | 1 code | 2 codes | Different codes |
| RadLex | 1 code | 1 code | Different codes |
| Locations | 4 | 4 | Overlap but some differences |
| Body Regions | ALL | Arm, Leg | **Agentic more specific, matches ground truth** |
| Etiologies | 3 | 2 | Original adds `metabolic` |
| Modalities | XR, CT, US, MR | XR, CT, US, MR | Match |
| Subspecialties | MK, ER | MK | Original adds ER |

**Verdict:** Agentic's body regions match the ground truth (Arm, Leg) better than ALL.

---

### 3. abnormal intracranial enhancement

| Aspect | Original | Agentic | Notes |
|--------|----------|---------|-------|
| Time | 19.7s | 28.4s | Original much faster |
| SNOMED | 0 | 0 | Neither found SNOMED |
| RadLex | 1 code | 1 code | Same code |
| Locations | 4 | 4 | Mostly same |
| Body Regions | Head | Head | Match |
| Etiologies | 8 | 9 | Agentic adds `neoplastic:potential` |
| Modalities | CT, MR | CT, MR | Match |
| Subspecialties | NR, ER, OI, PD | NR, ER | Original has more |

**Verdict:** Similar quality. Original notably faster.

---

### 4. thoracic spine degenerative change

| Aspect | Original | Agentic | Notes |
|--------|----------|---------|-------|
| Time | 28.0s | 26.3s | Agentic faster |
| SNOMED | 2 codes | 1 code | Original has more |
| RadLex | 0 | 0 | Neither found RadLex |
| Locations | 4 | 4 | Mostly same |
| Body Regions | Chest | Chest | **Ground truth: Chest, Abdomen** |
| Etiologies | degenerative | degenerative | Match |
| Modalities | XR, CT, MR | XR, CT, MR | Match |
| Subspecialties | MK, CH | MK, NR | Different but both reasonable |

**Verdict:** Similar. Neither got body regions perfectly (should include Abdomen).

---

### 5. enotosis (bone island)

| Aspect | Original | Agentic | Notes |
|--------|----------|---------|-------|
| Time | 28.9s | 31.8s | |
| SNOMED | 1 code | 0 | **Original found it** |
| RadLex | 1 code | 0 | **Original found it** |
| Locations | 4 | 0 | **Original found them** |
| Body Regions | ALL | (empty) | Original matches ground truth |
| Etiologies | 3 | (empty) | **Agentic failed completely** |
| Modalities | (empty) | (empty) | Both empty |
| Subspecialties | OI, MI | (empty) | **Agentic failed** |

**Verdict:** Agentic completely failed on this misspelled term ("enotosis" vs "enostosis"). Original handled it gracefully.

---

## Overall Assessment

### Performance
- **Original is ~10% faster** on average (25.5s vs 28.1s)
- Speed difference varies by finding (from tie to 8.7s difference)

### Quality
- **Original more robust**: Handles edge cases better (e.g., misspellings)
- **Agentic sometimes more precise**: Body regions for "calcification in a tendon"
- **Both comparable** on typical findings

### Display Quality Issue
- Agentic version only captures code IDs, not display names
- This is a fixable implementation issue, not architectural

### Recommendations
1. **Keep Original as primary** - more robust, faster
2. **Fix Agentic's display name capture** for fair comparison
3. **Consider Agentic for cases** where tool orchestration order matters
4. **Add retry/fallback logic** to Agentic for robustness
