# Unified Enrichment Test Results

## Summary

Tested the unified enrichment approach (`enrich_finding_unified()`) on the same 5 findings used in the original comparison.

### Test Results

| Metric | Value |
|--------|-------|
| Total tests | 5 |
| Successful | 4 (80%) |
| Failed | 1 (20%) |
| Body region accuracy | 4/4 (100% on successful tests) |
| Average time | 59.7s per finding |
| Total time | 298.5s |

### Individual Findings

1. **pulmonary segmental consolidation**: ✓ Success (77.8s)
   - Body regions: Chest ✓ (matches ground truth)
   - SNOMED: 6 codes, RadLex: 0 codes
   - Anatomic locations: 4 locations

2. **calcification in a tendon**: ✓ Success (60.1s)
   - Body regions: Arm, Leg ✓ (matches ground truth)
   - SNOMED: 4 codes, RadLex: 1 code
   - Anatomic locations: 4 locations

3. **abnormal intracranial enhancement**: ✓ Success (59.3s)
   - Body regions: Head ✓ (matches ground truth)
   - SNOMED: 2 codes, RadLex: 0 codes
   - Anatomic locations: 4 locations

4. **thoracic spine degenerative change**: ✓ Success (72.3s)
   - Body regions: Chest, Abdomen ✓ (matches ground truth)
   - SNOMED: 2 codes, RadLex: 0 codes
   - Anatomic locations: 3 locations

5. **enotosis**: ✗ Failed (24.0s)
   - Error: IndexCode validation error (display value "Or" too short)
   - Body regions: [] (ground truth: ALL)

### Comparison to Original Pipeline

From `comparison_results.json`:
- Original pipeline average time: ~25.5s per finding (range: 19.7-28.9s)
- Agentic pipeline average time: ~28.1s per finding (range: 26.3-31.8s)
- **Unified pipeline average time: ~59.7s per finding** (range: 24.0-77.8s)

Note: The unified pipeline is significantly slower than expected. This may be due to:
- Cold start effects (first run after implementation)
- Network latency
- Additional processing in the unified classifier
- Need for further optimization

### Body Region Accuracy

All successful tests (4/4) matched ground truth body regions exactly:
- pulmonary segmental consolidation: Chest ✓
- calcification in a tendon: Arm, Leg ✓
- abnormal intracranial enhancement: Head ✓
- thoracic spine degenerative change: Chest, Abdomen ✓

### Issues Identified

1. **Validation Error**: The LLM returned "Or" as a display value for an IndexCode, which violates the minimum length constraint (3 characters). This caused the "enotosis" test to fail.

2. **Performance**: The unified pipeline took 59.7s on average, much slower than the original (25.5s) or agentic (28.1s) pipelines. This needs investigation.

3. **Success Rate**: 80% success rate (4/5) vs the original pipeline which appears to have completed all 5 findings.

## Recommendations

1. Investigate the performance issue - the unified approach should be faster, not slower
2. Add validation/sanitization for IndexCode display values to handle edge cases like very short strings
3. Rerun tests to verify if timing was an anomaly
4. Consider adding retry logic for validation errors

## Test Data Location

- Test script: `scripts/test_unified_enrichment.py`
- Results: `scripts/enrichment_comparison/unified_results.json`
- Ground truth: `scripts/ipl_finding_models.json`
