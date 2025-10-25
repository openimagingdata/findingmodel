# Expand Agent Evaluation Coverage

## Status: Planning

**Created:** 2025-10-18

**Priority:** Medium

**Prerequisites:** `tasks/refactor_model_editor_evals.md` Phases 1-2 must be complete

**Related Documents:** `docs/evaluation_guide.md`, memory: `agent_evaluation_best_practices_2025`

## Overview

Create comprehensive evaluation suites for all AI agents in `src/findingmodel/tools/`. This plan assumes the base evaluation framework from the refactoring work is already in place and can be reused.

## Scope

Create comprehensive evaluation suites for all AI agents in `src/findingmodel/tools/`. Each eval implementation task includes:

1. Creating comprehensive eval suite in `evals/`
2. Reducing existing callout tests in `test/` to single sanity check
3. Moving all behavioral testing from tests to evals

**Agents to implement:**

1. `similar_finding_models` - Similarity search and duplicate detection
2. `ontology_concept_match` - Multi-backend ontology concept matching
3. `anatomic_location_search` - Two-agent architecture for anatomic location lookup
4. `markdown_in` - Markdown to finding model parsing
5. `finding_description` - LLM-generated clinical descriptions

**Test Reduction Opportunity:** ~15 callout tests across these agents will be replaced with 5 sanity checks + comprehensive evals.

## Test Reduction Strategy

### Principle: One Sanity Check Per Agent

Each agent tool should have **exactly ONE** callout-marked integration test in `test/` that verifies:
1. Basic wiring is correct (agent can be instantiated and called)
2. API key availability (skip gracefully if not configured)
3. Returns expected data structure type

**All comprehensive behavioral testing moves to evals.**

### Pattern for Sanity Check Tests

```python
@pytest.mark.callout
@pytest.mark.asyncio
async def test_tool_name_basic_wiring() -> None:
    """Sanity check: Verify basic wiring with real API.

    All comprehensive behavioral testing is in evals/tool_name.py.
    This test only verifies the tool can be called successfully.
    """
    # Skip if API key not configured
    if not settings.required_api_key:
        pytest.skip("API key not configured")

    # Call with simplest valid input
    result = await tool_function(basic_input)

    # Assert only on structure, not behavior
    assert isinstance(result, ExpectedType)
    assert result.required_field is not None

    # NO behavioral assertions - those belong in evals
```

### Current Test Coverage by Agent

#### Similar Finding Models
**Current:** `test/test_tools.py` lines 677-719 (2 callout tests)
- `test_find_similar_models_integration` - Full behavior test
- `test_find_similar_models_edge_cases` - Edge case testing

**Action:** Replace with single sanity check, move both to evals

#### Ontology Concept Match
**Current:** `test/test_ontology_search.py` lines 657-779 (5 callout tests)
- `test_bioontology_search_pneumonia` - Full search test
- `test_bioontology_search_all_pages` - Pagination test
- `test_bioontology_search_as_ontology_results` - Format conversion
- `test_bioontology_semantic_type_filter` - Filter behavior
- `test_bioontology_integration` - End-to-end test

**Action:** Replace with single sanity check, move all 5 to evals

#### Anatomic Location Search
**Current:** `test/test_anatomic_locations.py` lines 1078-1139 (2 callout tests)
- `test_create_anatomic_database_with_real_embeddings` - Database creation
- `test_build_with_real_openai` - CLI integration

**Action:** Replace with single search sanity check, move behavior to evals

#### Markdown Input (create_model_from_markdown)
**Current:** `test/test_tools.py` (2 callout tests)
- `test_create_model_from_markdown_integration` (lines 503-560)
- `test_create_model_from_markdown_file_integration` (lines 564-601)

**Action:** Replace with single sanity check, move parsing tests to evals

#### Finding Description (create_info_from_name, add_details_to_info)
**Current:** `test/test_tools.py` (6 callout tests)
- `test_create_info_from_name_integration` (lines 427-449)
- `test_create_info_from_name_edge_cases` (lines 454-467)
- `test_add_details_to_info_integration` (lines 472-498)
- `test_create_info_from_name_integration_normalizes_output` (lines 605-621)
- `test_ai_tools_error_handling` (lines 627-642)

**Action:** Analyze tool dependencies, create eval suite, reduce to sanity check(s)

### Reduction Checklist Per Agent

When creating an eval suite, also complete:

- [ ] Identify all callout tests for this agent in `test/`
- [ ] Create comprehensive eval cases based on those tests
- [ ] Replace multiple callout tests with single sanity check
- [ ] Verify sanity check skips gracefully without API keys
- [ ] Document what moved where (commit message references eval suite)
- [ ] Run `task test` to ensure sanity check works
- [ ] Run `task evals:agent_name` to verify eval suite comprehensive
- [ ] Update this task document with completion status

## Agent Evaluation Plans

### 1. Similar Finding Models (Priority 1)

**File:** `evals/similar_models.py`

**Current Test Coverage:**
- 2 callout tests in `test/test_tools.py` (lines 677-719)
- Tests cover: basic integration, edge cases
- **Action:** Reduce to 1 sanity check, move all behavior to evals

**Agent Details:**

- Finds similar finding models in the index
- Uses semantic similarity and text matching
- Helps identify duplicates and related models
- Located at: `src/findingmodel/tools/similar_finding_models.py`

#### Test Case Categories

**Success Cases:**

- Finding exact duplicates
- Finding near-duplicates (minor variations)
- Finding semantically similar models
- Finding models in same domain/category
- Ranking by similarity

**Negative Cases:**

- Dissimilar findings that shouldn't match
- Models with similar names but different meanings
- Edge cases that look similar but aren't

**Ranking Cases:**

- Exact match should rank highest
- Near-duplicates should rank above distant matches
- Irrelevant results should have low scores
- Consistent ordering across runs

**Edge Cases:**

- Query finding not in index
- Empty or minimal finding information
- Very common finding names (many matches)
- Newly created findings with no history

**Performance Cases:**

- Large index (100+ models)
- Query time within reasonable bounds
- Memory usage reasonable

#### Evaluators to Create

- **DuplicateDetectionEvaluator**: Binary - found known duplicate or not
- **RankingQualityEvaluator**: MRR or NDCG for ranking
- **PrecisionAtKEvaluator**: Precision at top-K results (K=5, 10)
- **SemanticSimilarityEvaluator**: Checks semantic similarity scores
- **PerformanceEvaluator**: Tracks query time and resource usage

#### Test Data Considerations

- Need curated pairs of similar/dissimilar findings
- Ground truth rankings for evaluation
- Mix of clear cases and ambiguous cases

#### Acceptance Criteria

- [ ] Minimum 15 test cases covering all categories
- [ ] Duplicate detection accuracy ≥95%
- [ ] MRR ≥0.8 for ranking quality
- [ ] Precision@5 ≥0.9 for top results
- [ ] Mock tests with controlled index
- [ ] Performance benchmarks documented
- [ ] Documentation with similarity threshold guidance
- [ ] Callout tests in `test/test_tools.py` reduced to single sanity check
- [ ] Test reduction documented in commit/PR

---

### 2. Ontology Concept Match (Priority 2)

**File:** `evals/ontology_match.py`

**Current Test Coverage:**
- 5 callout tests in `test/test_ontology_search.py` (lines 657-779)
- Tests cover: basic search, pagination, format conversion, filters, integration
- **Action:** Reduce to 1 sanity check, move all behavior to evals

**Agent Details:**

- Multi-backend: BioOntology API and DuckDB
- Handles concept matching and ranking
- Supports multiple ontologies (RADLEX, SNOMEDCT, etc.)
- Located at: `src/findingmodel/tools/ontology_concept_match.py`

#### Test Case Categories

**Success Cases:**

- Common radiological concepts (e.g., "pneumothorax", "fracture")
- Specific pathologies (e.g., "Type A aortic dissection")
- Anatomy + pathology combinations
- Modality-specific terms (e.g., "FLAIR signal")
- Different ontologies (RADLEX vs SNOMEDCT)

**Rejection/Error Cases:**

- Non-medical terms
- Extremely rare or obsolete terms
- Ambiguous concepts without context
- Invalid ontology names

**Edge Cases:**

- Synonyms and abbreviations (e.g., "MI" vs "myocardial infarction")
- Similar but distinct concepts
- Case variations
- Terms with special characters
- Very long or compound concept names

**Ranking Cases:**

- Multiple valid matches (best match should rank first)
- Partial matches vs exact matches
- Context-dependent rankings

**Backend-Specific Cases:**

- BioOntology API availability
- Fallback to DuckDB when API unavailable
- Consistency between backends
- Rate limiting handling

#### Evaluators to Create

- **ConceptMatchAccuracyEvaluator**: Checks if correct concept found
- **RankingQualityEvaluator**: Uses NDCG or MRR for ranking quality
- **BackendConsistencyEvaluator**: Compares results across backends
- **SynonymHandlingEvaluator**: Tests synonym and abbreviation handling
- **LLMJudgeEvaluator** (optional): Uses LLM to judge match quality

#### Acceptance Criteria

- [ ] Minimum 20 test cases covering all categories
- [ ] Tests all supported backends (BioOntology, DuckDB)
- [ ] Ranking quality metrics (NDCG ≥0.8)
- [ ] Backend consistency checks (>90% agreement)
- [ ] Mock and full API test modes
- [ ] Documentation with examples from different ontologies
- [ ] Callout tests in `test/test_ontology_search.py` reduced to single sanity check
- [ ] Test reduction documented in commit/PR

---

### 3. Anatomic Location Search (Priority 3)

**File:** `evals/anatomic_search.py`

**Current Test Coverage:**
- 2 callout tests in `test/test_anatomic_locations.py` (lines 1078-1139)
- Tests cover: database creation with embeddings, CLI integration
- **Action:** Create search-focused sanity check, move database/CLI behavior to evals

**Agent Details:**

- Two-agent architecture: search agent + matching agent
- Backend support: MongoDB and DuckDB
- Handles anatomic hierarchy and relationships
- Located at: `src/findingmodel/tools/anatomic_location_search.py`

#### Test Case Categories

**Success Cases:**

- Common anatomic terms (e.g., "heart", "lung", "liver")
- Specific locations (e.g., "left ventricle", "right upper lobe")
- Hierarchical relationships (e.g., "mitral valve" should relate to "heart")
- Synonyms and variations (e.g., "pulmonary artery" vs "PA")
- Anatomic systems (e.g., "cardiovascular system")

**Rejection/Error Cases:**

- Non-anatomic terms (e.g., "diabetes", "protocol")
- Ambiguous terms (e.g., "trunk" - could be body or vascular)
- Misspellings with no close matches
- Empty or very short queries
- Special characters and invalid input

**Edge Cases:**

- Very long anatomic names
- Terms with multiple valid interpretations
- Case sensitivity variations
- Terms with accents or special characters
- Composite anatomic locations (e.g., "left anterior descending artery")

**Backend-Specific Cases:**

- MongoDB backend available vs unavailable
- Fallback to DuckDB when MongoDB fails
- Performance with large result sets
- Handling of missing data in either backend

#### Evaluators to Create

- **SearchAccuracyEvaluator**: Checks if correct anatomic location found
- **HierarchyEvaluator**: Verifies hierarchical relationships are preserved
- **BackendFallbackEvaluator**: Tests MongoDB → DuckDB fallback behavior
- **RankingQualityEvaluator**: Assesses result ranking (most relevant first)

#### Acceptance Criteria

- [ ] Minimum 20 test cases covering all categories
- [ ] Uses base evaluators from `evals/base.py`
- [ ] Tests both MongoDB and DuckDB backends
- [ ] Mock tests using `TestModel` for quick validation
- [ ] Documentation with example cases
- [ ] Success rate threshold: ≥85% for common terms
- [ ] Callout tests in `test/test_anatomic_locations.py` reduced to single sanity check
- [ ] Test reduction documented in commit/PR

---

### 4. Markdown Input (Priority 4)

**File:** `evals/markdown_in.py`

**Current Test Coverage:**
- 2 callout tests in `test/test_tools.py` (lines 503-601)
- Tests cover: markdown parsing integration, file input handling
- **Action:** Reduce to 1 sanity check, move parsing behavior to evals

**Agent Details:**

- Parses markdown text into finding model structure
- Handles attributes, descriptions, metadata
- Validates structure and completeness
- Located at: `src/findingmodel/tools/markdown_in.py`

#### Test Case Categories

**Success Cases:**

- Well-formed markdown with all sections
- Simple attributes (choice, text)
- Complex attributes (numeric with units, hierarchical)
- Multiple attributes
- Nested structures

**Error Handling Cases:**

- Malformed markdown (syntax errors)
- Missing required sections
- Invalid attribute types
- Inconsistent structure
- Empty sections

**Edge Cases:**

- Very long descriptions
- Special characters in attribute names
- Unusual formatting (extra whitespace, mixed case)
- Comments and metadata
- Markdown variations (different header levels)

**Complex Structure Cases:**

- Hierarchical attributes
- Attributes with constraints
- Conditional attributes
- Attributes with units and ranges

#### Evaluators to Create

- **StructuralValidityEvaluator**: Checks parsed model has correct structure
- **AttributePreservationEvaluator**: Verifies all attributes parsed correctly
- **TypeCorrectnessEvaluator**: Checks attribute types match markdown specification
- **ErrorMessageQualityEvaluator**: Assesses clarity of error messages
- **RoundTripEvaluator**: Model → markdown → model should be equivalent

#### Acceptance Criteria

- [ ] Minimum 12 test cases covering all categories
- [ ] Structural validity for all success cases
- [ ] Clear error messages for all error cases
- [ ] Round-trip preservation (≥99% accuracy)
- [ ] Mock tests with predefined markdown
- [ ] Documentation with markdown format specification
- [ ] Handles all attribute types from `finding_model.py`
- [ ] Callout tests in `test/test_tools.py` reduced to single sanity check
- [ ] Test reduction documented in commit/PR

---

### 5. Finding Description (Priority 5)

**File:** `evals/finding_description.py`

**Current Test Coverage:**
- 6 callout tests in `test/test_tools.py` (lines 427-642)
- Tests cover: `create_info_from_name` (3 tests), `add_details_to_info` (2 tests), error handling (1 test)
- **Action:** Analyze tool dependencies, create dedicated eval suite, reduce to sanity check(s)

**Agent Details:**

- Generates clinical descriptions for finding models
- Uses LLM to create human-readable text
- Requires clinical accuracy validation
- May be embedded in `create_info_from_name` and `add_details_to_info` tools
- Located at: `src/findingmodel/tools/finding_description.py` (if standalone)

#### Test Case Categories

**Success Cases:**

- Common findings with clear descriptions
- Complex findings requiring detail
- Findings with multiple attributes
- Edge findings requiring caveats
- Standardized vs custom descriptions

**Quality Assessment Cases:**

- Clinical accuracy (medical correctness)
- Appropriate level of detail
- Clarity and readability
- Consistency across similar findings
- Proper medical terminology usage

**Error Handling Cases:**

- Unknown or rare findings
- Incomplete finding information
- Ambiguous finding names
- Missing critical attributes

**Edge Cases:**

- Very simple findings (minimal attributes)
- Very complex findings (many attributes)
- Findings with contradictory information
- Findings requiring disclaimers

#### Evaluators to Create

- **LengthAppropriatenessEvaluator**: Checks description isn't too short/long
- **TerminologyEvaluator**: Verifies proper medical terminology
- **ConsistencyEvaluator**: Compares similar findings for consistency
- **LLMJudgeQualityEvaluator**: Uses LLM to assess clinical accuracy
- **ReadabilityEvaluator**: Checks Flesch reading ease or similar metric

#### Special Considerations

- **Medical Validation**: Some cases require expert review
- **Ground Truth**: Need curated reference descriptions
- **Subjectivity**: Quality assessment has inherent subjectivity
- **Cost**: LLM-as-judge can be expensive; limit to key cases
- **Tool Dependencies**: May need to extract description generation from other tools

#### Acceptance Criteria

- [ ] Minimum 15 test cases covering all categories
- [ ] LLM-as-judge evaluator implemented
- [ ] At least 5 cases with medical expert validation
- [ ] Consistency evaluator for similar findings
- [ ] Mock tests with predefined good descriptions
- [ ] Documentation including validation methodology
- [ ] Quality threshold: ≥80% on LLM judge score
- [ ] Callout tests in `test/test_tools.py` reduced to sanity check(s)
- [ ] Test reduction documented in commit/PR

---

## Implementation Strategy

### Recommended Order (Prioritized by Test Reduction Impact)

**Priority 1 - Highest Test Reduction Opportunity:**

1. **Similar Finding Models** (2 callout tests → 1 sanity check)
   - Simplest agent with clear metrics
   - Good learning experience for eval creation + test reduction
   - Immediate runtime/cost savings

**Priority 2 - High Test Reduction Opportunity:**

2. **Ontology Concept Match** (5 callout tests → 1 sanity check)
   - Largest test reduction opportunity (5 tests eliminated)
   - Multi-backend testing patterns useful for other agents
   - Significant runtime/cost savings

**Priority 3 - Moderate Test Reduction + Architecture Learning:**

3. **Anatomic Location Search** (2 callout tests → 1 sanity check)
   - Two-agent system architecture test
   - Good pattern for complex agent evaluation
   - Database/CLI testing moves to evals

**Priority 4 - Moderate Test Reduction:**

4. **Markdown Input** (2 callout tests → 1 sanity check)
   - Parsing accuracy evaluation patterns
   - Round-trip testing strategies

**Priority 5 - Highest Complexity:**

5. **Finding Description** (6 callout tests → 1-2 sanity checks)
   - Largest number of tests but most complex to untangle
   - May require extracting description generation from other tools
   - Medical validation requirements
   - LLM-as-judge patterns useful for future work

### Task Completion Pattern

Each agent implementation is **one complete task** that includes:

1. **Create eval suite** in `evals/agent_name.py`
   - Define data models
   - Create focused evaluators
   - Build comprehensive test cases from existing callout tests
   - Add agent-specific behavioral tests
   - Document evaluation methodology

2. **Reduce unit tests** in `test/test_*.py`
   - Replace multiple callout tests with single sanity check
   - Ensure graceful skipping when API keys unavailable
   - Add comment referencing eval suite location

3. **Verify quality maintained**
   - Run `task test` (sanity check passes)
   - Run `task evals:agent_name` (comprehensive coverage)
   - Compare eval coverage to original test coverage

4. **Document changes**
   - Update this task document with completion status
   - Commit message explains what moved where
   - PR description highlights test reduction impact

### Shared Patterns

All evaluation suites should follow these patterns (see `evals/model_editor.py` for complete example):

```python
# 1. Import base components (absolute imports)
from evals.base import (
    ExactMatchEvaluator,
    KeywordMatchEvaluator,
    StructuralValidityEvaluator,
)
from evals.utils import load_fm_json
from pydantic_evals import Case, Dataset, Evaluator, EvaluatorContext

# NO Logfire imports needed - automatic via evals/__init__.py

# 2. Define data models
class AgentInput(BaseModel): ...
class AgentExpectedOutput(BaseModel): ...
class AgentActualOutput(BaseModel): ...

# 3. Create agent-specific evaluators (focused, hybrid scoring)
class SpecificEvaluator(Evaluator[AgentInput, AgentActualOutput, AgentExpectedOutput]):
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Return 0.0-1.0 score (strict or partial credit as appropriate)
        ...

# 4. Build dataset at module level
evaluators = [SpecificEvaluator(), KeywordMatchEvaluator(...), ...]
dataset = Dataset(
    cases=[
        Case(input=..., expected_output=..., metadata=...),
        # More cases...
    ],
    evaluators=evaluators,
)

# 5. Task function (NO manual Logfire spans)
async def run_agent_name_task(input_data: AgentInput) -> AgentActualOutput:
    """Execute task - automatic instrumentation captures everything."""
    result = await agent_function(input_data)
    return AgentActualOutput(result=result)

# 6. Main eval function (returns Report)
async def run_agent_name_evals() -> Report:
    """Run evaluation suite."""
    report = await dataset.evaluate(run_agent_name_task)
    return report

# 7. Standalone execution
if __name__ == "__main__":
    import asyncio

    async def main():
        print("\nRunning agent_name evaluation suite...")
        print("=" * 80)

        report = await run_agent_name_evals()

        print("\n" + "=" * 80)
        print("AGENT_NAME EVALUATION RESULTS")
        print("=" * 80 + "\n")
        report.print(include_input=False, include_output=True)

        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\nOVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\n")

    asyncio.run(main())
```

**Key Points:**
- NO Logfire code needed (automatic via `evals/__init__.py`)
- Use absolute imports: `from evals.base` NOT `from .base`
- Main function named `run_agent_name_evals()` NOT `test_run_agent_name_evals()`
- Returns Report object for programmatic use
- Standalone execution via `python -m evals.agent_name`

## Success Criteria

### Overall Success When

- [ ] All 5 agents have evaluation suites
- [ ] All agents reach minimum quality thresholds
- [ ] Base evaluators reused across multiple agents
- [ ] Documentation exists for each suite
- [ ] Both mock and API tests for each agent
- [ ] Total coverage: 80+ test cases across all agents

### Per-Agent Success When

- [ ] Minimum test cases met (12-20 depending on agent)
- [ ] All test categories covered (success, failure, edge)
- [ ] Agent-specific evaluators implemented
- [ ] Quality threshold met (typically 80-90%)
- [ ] Documentation includes examples and guidance
- [ ] Can run standalone via `python -m evals.agent_name` and `task evals:agent_name`
- [ ] Callout tests in `test/` reduced to single sanity check
- [ ] Sanity check skips gracefully when API keys unavailable
- [ ] Test reduction documented in commit/PR
- [ ] Original test coverage maintained or improved in eval suite

## Open Questions

### General

1. **Evaluation frequency**: How often to run full eval suites? (PR, nightly, weekly?)
2. **Cost management**: How to manage API costs for LLM-as-judge?
3. **Ground truth maintenance**: Who maintains reference data for evaluations?
4. **Failure triage**: What's the process when eval quality drops?

### Agent-Specific

1. **Anatomic Location Search**: Which anatomic taxonomy to use as ground truth?
2. **Ontology Concept Match**: How to handle ontology version updates?
3. **Finding Description**: Who provides medical expert validation?
4. **Similar Finding Models**: What similarity threshold is "good enough"?
5. **Markdown Input**: Should we support multiple markdown flavors?

## Implementation Tracking

### Completed
- [ ] None yet

### In Progress
- [ ] None

### Planned
- [ ] Similar Finding Models (Priority 1)
- [ ] Ontology Concept Match (Priority 2)
- [ ] Anatomic Location Search (Priority 3)
- [ ] Markdown Input (Priority 4)
- [ ] Finding Description (Priority 5)

### Test Reduction Impact Summary

**Before:** ~17 callout tests across 5 agents (in `test/`)
**After:** ~5 sanity checks + comprehensive evals (in `evals/`)

**Benefits:**
- Faster `task test` runs (unit tests only need minimal API verification)
- More comprehensive behavioral coverage (evals test quality, not just correctness)
- Clearer separation of concerns (tests verify wiring, evals assess behavior)
- Better observability (Logfire traces for eval runs)
- Easier maintenance (eval suites can evolve independently)

## References

- **Prerequisites:** `tasks/refactor_model_editor_evals.md`
- **Full guide:** `docs/evaluation_guide.md`
- **Serena memory:** `agent_evaluation_best_practices_2025`
- **Agent source:** `src/findingmodel/tools/`
- **Test data:** `test/data/`
- **Related memories:**
  - `pydantic_ai_testing_best_practices`
  - `test_suite_improvements_2025`
  - `anatomic_location_search_implementation`
  - `bioontology_integration_2025`
