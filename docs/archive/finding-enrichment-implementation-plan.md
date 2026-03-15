# Finding Enrichment System - Implementation Plan

## Implementation Status

**Last Updated**: 2025-11-25

### Phase Completion Summary

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Core Data Models | ✅ Complete | FindingEnrichmentResult, type aliases, ETIOLOGIES constant |
| Phase 2: Index Lookup | ✅ Complete | Using DuckDBIndex directly (no wrapper needed) |
| Phase 3: Ontology Search | ✅ Complete | search_ontology_codes_for_finding() with exclude_anatomical |
| Phase 4: Enrichment Agent | ✅ Complete | Pydantic AI agent with structured output, 2 tools |
| Phase 5: Main Function | ✅ Complete | 5-step orchestration with parallel execution |
| Phase 6: Script Interface | ✅ Complete | scripts/enrich_finding.py with logging enabled |
| Phase 7: Unit Tests | ✅ Complete | 57 tests covering models, helpers, agent, orchestration |
| Phase 8: Integration Tests | ✅ Complete | 8 tests with @pytest.mark.callout for real API calls |
| Phase 9: Manual Validation | 🚧 Not Started | Planned |
| Phase 10: Documentation | 🚧 In Progress | PRD and plan updated, memory updates pending |

### Key Implementation Decisions Made

1. **No wrapper functions**: Used DuckDBIndex, match_ontology_concepts, find_anatomic_locations directly
2. **Two-phase design**: Agent classifies → Main function assembles (cleaner separation of concerns)
3. **Parallel execution**: Ontology and anatomic searches run concurrently in Step 2
4. **Best-effort errors**: asyncio.gather with return_exceptions=True allows partial results
5. **Loguru logging**: Must explicitly enable with logger.enable("findingmodel")
6. **JSON-only output**: Simplified script interface, no pretty-print mode
7. **Canonical names**: When finding exists in Index, uses canonical name (e.g., "Pneumonia" not "pneumonia")

---

## Overview

This document outlines the technical implementation plan for the Finding Enrichment System, building on existing codebase patterns and tools.

## Implementation Phases

### Phase 1: Core Data Models & Result Structure ✅

**Status**: COMPLETE

**Files created/modified:**
- `src/findingmodel/tools/finding_enrichment.py` (lines 1-193)

**Completed Tasks:**
1. ✅ Defined `FindingEnrichmentResult` Pydantic model with all required fields
2. ✅ Defined `EnrichmentContext` and `EnrichmentClassification` for agent I/O
3. ✅ Added type aliases: BodyRegion, Modality, Subspecialty
4. ✅ Created ETIOLOGIES constant list with 22 categories
5. ✅ Comprehensive docstrings and field descriptions
6. ✅ Validation for enumerations via Literal types and field validators

**Implementation Notes:**
- Used Literal types for compile-time validation of enums
- ETIOLOGIES validated via field_validator checking against constant list
- Separate models for agent context (input) vs classification (output) vs final result

### Phase 2: Index Lookup Function ✅

**Status**: COMPLETE

**Files modified:**
- `src/findingmodel/tools/finding_enrichment.py` (enrich_finding function, lines 579-660)

**Completed Tasks:**
1. ✅ Implemented index lookup using DuckDBIndex directly (no wrapper needed)
   - Uses `index.get(identifier)` which handles both OIFM ID and name lookups
   - Uses `index.get_full(oifm_id)` to retrieve complete FindingModelFull
   - Proper async context manager: `async with DuckDBIndex(read_only=True) as index:`
2. ✅ Extracts relevant context: name, description, existing codes
3. ✅ Error handling with try/except and proper cleanup via context manager

**Implementation Notes:**
- Did NOT create wrapper function - used public DuckDBIndex API directly
- Context manager handles connection lifecycle automatically
- Gracefully handles non-existent findings (returns None, continues processing)

### Phase 3: Ontology Code Search Wrapper ✅

**Status**: COMPLETE

**Files modified:**
- `src/findingmodel/tools/finding_enrichment.py` (lines 197-253)

**Completed Tasks:**
1. ✅ Created `search_ontology_codes_for_finding()` function
   - Calls `match_ontology_concepts()` with `exclude_anatomical=True` parameter
   - NO additional filtering needed (upstream function already handles it)
   - Separates SNOMED vs RadLex codes from categorized results
   - Returns tuple: (list[IndexCode], list[IndexCode]) for SNOMED and RadLex
2. ✅ Filtering handled by match_ontology_concepts (exclude_anatomical=True)
   - Uses existing SNOMED hierarchy filtering
   - No need for duplicate filtering logic
3. ✅ Handles empty results gracefully (returns empty lists)

**Implementation Notes:**
- Initially implemented redundant filtering, removed after evaluator feedback
- Trusts upstream exclude_anatomical parameter rather than duplicating logic
- Collects both exact_matches and should_include categorizations
- Simplified to ~50 lines vs originally planned complex filtering

### Phase 4: Enrichment Agent Implementation ✅

**Status**: COMPLETE

**Files modified:**
- `src/findingmodel/tools/finding_enrichment.py` (lines 256-570)

**Completed Tasks:**
1. ✅ Defined `EnrichmentContext` dependency class (lines 322-368)
   - finding_name, finding_description
   - existing_codes, existing_model
   - All fields properly documented
2. ✅ Created Pydantic AI agent with structured output (lines 481-570)
   - Input: EnrichmentContext (deps_type)
   - Output: EnrichmentClassification (NOT full result - two-phase design)
   - Two tools implemented: search_ontology_codes, find_anatomic_location
3. ✅ Comprehensive system prompt (lines 407-479, _create_enrichment_system_prompt)
   - Role: Medical imaging finding enrichment specialist
   - Tasks: Classify body regions, etiologies, modalities, subspecialties
   - Instructions: Precise, multiple values allowed, leave blank if uncertain
   - Full taxonomy definitions embedded (22 etiologies, 9 modalities, 16 subspecialties)
4. ✅ Implemented tool wrappers with @agent.tool decorators (lines 490-566)
   - search_ontology_codes: calls search_ontology_codes_for_finding(), formats as string
   - find_anatomic_location: calls find_anatomic_locations() with small tier, returns JSON
   - Proper error handling and logging throughout
5. ✅ Agent configuration via create_enrichment_agent() (lines 481-570)
   - Tier-based model selection: defaults to "base" tier
   - Multi-provider support: optional provider parameter
   - Logfire instrumentation: automatic via get_model()

**Implementation Notes:**
- Two-phase design: Agent outputs EnrichmentClassification, main function assembles full result
- Agent has tools available but may not need them (parallel results provided in context)
- System prompt is ~250 lines with complete taxonomy definitions
- Model tier changed from originally planned 'main' to 'base' after review

### Phase 5: Main Enrichment Function ✅

**Status**: COMPLETE

**Files modified:**
- `src/findingmodel/tools/finding_enrichment.py` (lines 579-755)

**Completed Tasks:**
1. ✅ Implemented `enrich_finding()` main entry point
   - 5-step orchestration workflow (lines 579-755)
   - Step 1: Index lookup with DuckDBIndex
   - Step 2: Parallel execution (ontology + anatomic)
   - Step 3: Create EnrichmentContext
   - Step 4: Run agent classification
   - Step 5: Assemble complete FindingEnrichmentResult
2. ✅ Parallel execution implemented (lines 662-669)
   - asyncio.gather() with return_exceptions=True
   - Ontology search and anatomic location run concurrently
   - Defense-in-depth: handles individual failures gracefully
3. ✅ Comprehensive error handling throughout
   - try/except blocks at each step
   - Graceful degradation: empty results on failures
   - Detailed logging at decision points (15+ log statements)
   - Clear error messages with context
4. ✅ Metadata fields added to result (lines 736-751)
   - enrichment_timestamp: datetime.now(timezone.utc)
   - model_provider: from settings or parameter
   - model_tier: hardcoded to "base"

**Implementation Notes:**
- Provider parameter properly propagated to create_enrichment_agent() (fixed in revision)
- Best-effort approach: continues even if individual tools fail
- Timezone-aware timestamps using timezone.utc
- ~175 lines of orchestration logic with extensive error handling

### Phase 6: Script Interface ✅

**Status**: COMPLETE

**Files created:**
- `scripts/enrich_finding.py` (44 lines)

**Completed Tasks:**
1. ✅ Created standalone script with argparse (minimal implementation)
   - Positional argument: finding identifier (name or OIFM ID)
   - Optional flag: `--provider` (choices: openai, anthropic)
   - NOT implemented: --format, --output, --verbose (simplified scope)
2. ✅ JSON output formatting only
   - Uses `result.model_dump_json(exclude_none=True, indent=2)`
   - Clean output to stdout
   - No pretty-print mode (not needed for MVP)
3. ✅ Usage examples in docstring (lines 2-8)
4. ✅ Error handling with try/except (lines 32-39)
   - Prints error message with ❌ prefix
   - Re-raises for proper exit codes

**Implementation Notes:**
- Loguru logging enabled: `logger.enable("findingmodel")` (CRITICAL for visibility)
- Logger disabled by default in __init__.py, must explicitly enable
- Simplified from original plan: JSON-only output, no file saving, no verbose flag
- Three iterations required to get logging right (initially used wrong logging module)

**Actual usage:**
```bash
# Basic usage with finding name
python scripts/enrich_finding.py "pneumonia"

# With Anthropic provider
python scripts/enrich_finding.py "pneumonia" --provider anthropic

# With OIFM ID
python scripts/enrich_finding.py OIFM_AI_000001
```

### Phase 7: Testing - Unit Tests ✅

**Status**: COMPLETE

**Files created:**
- `test/test_finding_enrichment.py` (1,666 lines)
- `test/data/test_enrichment_samples.json` (test fixtures)

**Completed Tasks:**
1. ✅ Set up test module with `ALLOW_MODEL_REQUESTS = False` at module level
2. ✅ Created fixtures:
   - `mock_finding_model`: FindingModelFull instance (Pulmonary Nodule)
   - `mock_snomed_codes`, `mock_radlex_codes`: Lists of IndexCode objects
   - `mock_anatomic_locations`: List of OntologySearchResult objects
   - `mock_enrichment_context`: Complete EnrichmentContext for agent testing
3. ✅ Unit tests for data models (28 tests):
   - `TestFindingEnrichmentResult`: 7 tests for validation, serialization
   - `TestEnumTypes`: 6 tests for BodyRegion, Modality, Subspecialty Literals
   - `TestEtiologiesConstant`: 5 tests for taxonomy validation
   - `TestEnrichmentContext`: 3 tests for context model
   - `TestEnrichmentClassification`: 4 tests for agent output model
   - `TestModelCompatibility`: 3 tests for integration between models
4. ✅ Unit tests for helper functions (9 tests):
   - `TestSearchOntologyCodesForFinding`: 5 tests for code separation, filtering, errors
   - `TestCreateEnrichmentSystemPrompt`: 4 tests for prompt content
5. ✅ Unit tests for agent (14 tests):
   - `TestCreateEnrichmentAgent`: 6 tests for configuration
   - `TestEnrichmentAgentBehavior`: 8 tests with TestModel
6. ✅ Unit tests for orchestration (6 tests):
   - `TestEnrichFindingOrchestration`: 6 tests for main function with mocked dependencies

**Implementation Notes:**
- Total: 57 unit tests, all passing
- Uses `@patch` for external dependencies (DuckDBIndex, ontology search, anatomic search)
- Uses `TestModel` from pydantic_ai for agent behavior tests
- Comprehensive error handling and edge case coverage
- Run with: `task test -- test/test_finding_enrichment.py`

### Phase 8: Testing - Integration Tests ✅

**Status**: COMPLETE

**Files modified:**
- `test/test_finding_enrichment.py` (added TestFindingEnrichmentIntegration class)

**Completed Tasks:**
1. ✅ Added 8 integration tests marked with `@pytest.mark.callout`
2. ✅ Tested representative findings:
   - Simple: `test_enrich_pneumonia`, `test_enrich_fracture`
   - Moderate: `test_enrich_pulmonary_nodule`, `test_enrich_liver_lesion`
   - Complex: `test_enrich_ground_glass_opacity`
   - Edge: `test_enrich_mass_ambiguous`, `test_enrich_unknown_finding`
3. ✅ Proper try/finally blocks to restore ALLOW_MODEL_REQUESTS
4. ✅ Verified:
   - Result structure (all fields present and typed correctly)
   - Classifications contain expected values (flexible assertions for AI variability)
   - Multiple values supported correctly (all lists validated)
   - Empty results handled gracefully
5. ✅ Performance check: `test_enrich_performance_under_30s`

**Implementation Notes:**
- Total: 8 integration tests, require real API keys to run
- Uses case-insensitive assertions for finding_name (Index returns canonical form)
- Flexible assertions accommodate AI variability (e.g., "Chest" OR len > 0)
- Run with: `task test-full -- test/test_finding_enrichment.py -m callout`
- All tests properly restore ALLOW_MODEL_REQUESTS in finally blocks

### Phase 9: Manual Validation & Iteration

**Tasks:**
1. Create validation dataset:
   - Select 20-30 diverse findings from Index
   - Include various body regions, etiologies, modalities
   - Document expected/ideal results (manual expert review)
2. Run enrichment on validation set
3. Document discrepancies and error patterns:
   - Incorrect classifications
   - Missing values
   - Over-inclusive results
   - Agent reasoning issues
4. Iterate on prompts and logic:
   - Refine system prompt for better classifications
   - Add examples to prompt if needed
   - Adjust filtering logic if ontology codes are wrong
   - Fine-tune tool descriptions
5. Re-run and measure improvement

**Estimated complexity:** Medium (iterative)
**Dependencies:** Phases 6-8
**Note:** This is inherently iterative and requires medical expertise

### Phase 10: Documentation & Memory Updates

**Files to create/modify:**
- `docs/finding-enrichment-prd.md` (already created)
- `docs/finding-enrichment-implementation-plan.md` (this file)
- Serena memory: Create new memory for finding enrichment system

**Tasks:**
1. Write comprehensive docstrings in code
2. Create usage examples in README or docs
3. Document common issues and solutions
4. Update Serena memory with:
   - Architecture overview
   - How to use the enrichment system
   - Integration points with existing tools
   - Testing patterns
   - Known limitations and future work
5. Add entry to `suggested_commands` memory for script usage

**Estimated complexity:** Low
**Dependencies:** Phases 6-9

## Technical Decisions & Rationale

### 1. Single Agent vs. Multi-Agent

**Decision:** Single agent with tools

**Rationale:**
- Not doing iterative refinement like anatomic location search
- Classifications are straightforward given context
- Simpler architecture, easier to debug
- Can add second agent later if needed

### 2. Parallel vs. Sequential Tool Execution

**Decision:** Parallel where possible

**Rationale:**
- Ontology search and anatomic location are independent
- Significant time savings (2 API calls instead of sequential)
- Agent can analyze all context together

### 3. Filtering Anatomic Codes

**Decision:** Filter at wrapper layer, not in match_ontology_concepts

**Rationale:**
- Preserves existing function behavior for other uses
- Enrichment-specific filtering logic
- Can be refined independently
- Alternative: Add optional parameter to existing function (may do later)

### 4. Output Structure Design

**Decision:** Flat structure with all metadata fields

**Rationale:**
- Easy to review and validate
- Straightforward JSON serialization
- Clear separation from FindingModel
- Simple to convert to FindingModel fields later

### 5. Error Handling Strategy

**Decision:** Best-effort with empty results

**Rationale:**
- Human review process can handle missing data
- Strict failures would require manual intervention anyway
- Allows partial results to be useful
- Clear distinction between "couldn't find" vs. "system error"

## Risk Assessment

### High Risk
1. **Ontology code filtering accuracy**
   - Mitigation: Manual review of first 20-30 results, iterate on filtering logic
   - Fallback: Allow human to filter in review process

2. **Agent classification quality**
   - Mitigation: Comprehensive prompt engineering, validation dataset
   - Fallback: Iterate on prompts based on error patterns

### Medium Risk
1. **Anatomic location tool at scale**
   - First scaled application of multi-location support
   - Mitigation: Extra testing, monitoring for edge cases
   - Fallback: Limit to single best location if issues arise

2. **Performance**
   - Multiple tool calls + LLM inference could be slow
   - Mitigation: Parallel execution, use base tier (faster than full)
   - Fallback: Add caching for repeated lookups

### Low Risk
1. **Index lookup**
   - Well-established DuckDB patterns
   - Simple queries

2. **Script interface**
   - Straightforward CLI implementation

## Success Metrics

### Code Quality
- [ ] All tests pass with >85% coverage
- [ ] No linting errors (`task check`)
- [ ] Type annotations complete (`mypy` passes)
- [ ] Comprehensive docstrings

### Functional Quality
- [ ] Processes common findings without errors (90%+ success rate)
- [ ] Ontology codes are clinically appropriate (manual review)
- [ ] Classifications ≥80% accurate on validation set
- [ ] Handles edge cases gracefully (ambiguous findings, unknowns)

### Performance
- [ ] <30 seconds per finding (acceptable for manual workflow)
- [ ] Memory usage reasonable (no leaks with repeated calls)

### Documentation
- [ ] PRD and implementation plan complete
- [ ] Code docstrings comprehensive
- [ ] Usage examples clear
- [ ] Serena memory updated

## Future Enhancements (Post-MVP)

See PRD section "Future Enhancements" for full list. Key items:

1. Batch processing mode
2. Database update capability
3. CLI integration
4. Evaluation suite with ground truth
5. Caching layer for performance
6. Confidence scores
7. Interactive correction mode

## Development Timeline

**Note:** Per user instructions, no time estimates provided. Implementation will proceed phase-by-phase with validation at each step.

**Phase dependencies:**
- Phases 1-3: Can be done in sequence (data models → lookups)
- Phase 4: Requires 1-3 complete
- Phase 5: Requires 1-4 complete
- Phase 6: Requires 5 complete (can start in parallel with 5)
- Phases 7-8: Can partially overlap with 6
- Phase 9: Requires working end-to-end system (6-8)
- Phase 10: Final documentation pass

**Critical path:** 1 → 2 → 3 → 4 → 5 → 6 → 9 (manual validation)

## Open Questions

1. **SNOMED hierarchy filtering:** Need to verify exact hierarchy codes for anatomic structures to exclude. May need to research SNOMED CT structure or test empirically.

2. **RadLex filtering:** Less hierarchical than SNOMED. May need regex or keyword-based filtering (e.g., exclude results with "anatomy", "region", "location" in preferred terms).

3. **Confidence thresholds:** Should we have minimum confidence thresholds for ontology matches, or accept all results? (Lean toward: accept all, let human review decide)

4. **Prompt engineering:** Classification prompts will require iteration. Should we include examples in the prompt? (Lean toward: start simple, add examples if needed)

5. **Anatomic location count:** Should we limit to top N locations (e.g., 3-5) or allow any number? (Lean toward: allow any, let agent decide)

## Notes for Implementation

- Follow code style conventions in Serena memory `code_style_conventions`
- Use tier-based model selection via `get_model()` from common.py
- Add Logfire instrumentation for observability
- Ensure proper cleanup of database connections (try/finally)
- Write tests before or alongside implementation (TDD-friendly)
- Commit after each phase completion with descriptive messages
- Update Serena memory incrementally as design decisions are made
