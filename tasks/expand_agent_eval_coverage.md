# Expand Agent Evaluation Coverage

## Status: Planning

**Created:** 2025-10-18

**Priority:** Medium

**Prerequisites:** `tasks/refactor_model_editor_evals.md` Phases 1-2 must be complete

**Related Documents:** `docs/evaluation_guide.md`, memory: `agent_evaluation_best_practices_2025`

## Overview

Create comprehensive evaluation suites for all AI agents in `src/findingmodel/tools/`. This plan assumes the base evaluation framework from the refactoring work is already in place and can be reused.

## Scope

Create evaluation suites for these agents:

1. `anatomic_location_search` - Two-agent architecture for anatomic location lookup
2. `ontology_concept_match` - Multi-backend ontology concept matching
3. `finding_description` - LLM-generated clinical descriptions
4. `similar_finding_models` - Similarity search and duplicate detection
5. `markdown_in` - Markdown to finding model parsing

## Agent Evaluation Plans

### 1. Anatomic Location Search

**File:** `test/evals/test_anatomic_search_evals.py`

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
- [ ] Uses base evaluators from `test/evals/base.py`
- [ ] Tests both MongoDB and DuckDB backends
- [ ] Mock tests using `TestModel` for quick validation
- [ ] Full API tests marked with `@pytest.mark.callout`
- [ ] Documentation with example cases
- [ ] Success rate threshold: ≥85% for common terms

---

### 2. Ontology Concept Match

**File:** `test/evals/test_ontology_match_evals.py`

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

---

### 3. Finding Description

**File:** `test/evals/test_finding_description_evals.py`

**Agent Details:**

- Generates clinical descriptions for finding models
- Uses LLM to create human-readable text
- Requires clinical accuracy validation
- Located at: `src/findingmodel/tools/finding_description.py`

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

#### Acceptance Criteria

- [ ] Minimum 15 test cases covering all categories
- [ ] LLM-as-judge evaluator implemented
- [ ] At least 5 cases with medical expert validation
- [ ] Consistency evaluator for similar findings
- [ ] Mock tests with predefined good descriptions
- [ ] Documentation including validation methodology
- [ ] Quality threshold: ≥80% on LLM judge score

---

### 4. Similar Finding Models

**File:** `test/evals/test_similar_models_evals.py`

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

---

### 5. Markdown Input

**File:** `test/evals/test_markdown_in_evals.py`

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

---

## Implementation Strategy

### Recommended Order

1. **Markdown Input** (simplest, good learning experience)
2. **Similar Finding Models** (moderate complexity, clear metrics)
3. **Anatomic Location Search** (two-agent system, good architecture test)
4. **Ontology Concept Match** (similar to anatomic search but more complex)
5. **Finding Description** (most complex, requires medical validation)

### Shared Patterns

All evaluation suites should follow these patterns:

```python
# 1. Import base components
from test.evals.base import (
    AgentEvaluationSuite,
    KeywordMatchEvaluator,
    StructuralValidityEvaluator,
    ErrorHandlingEvaluator
)
from test.evals.utils import load_fm_json

# 2. Define data models
class MyAgentInput(BaseModel): ...
class MyAgentExpectedOutput(BaseModel): ...
class MyAgentActualOutput(BaseModel): ...

# 3. Create agent-specific evaluators
class MySpecificEvaluator(Evaluator[MyAgentInput, MyAgentActualOutput]):
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Return 0.0-1.0 score
        ...

# 4. Create evaluation suite
class MyAgentEvalSuite(AgentEvaluationSuite[...]):
    def create_successful_cases(self) -> list[Case]: ...
    def create_failure_cases(self) -> list[Case]: ...
    def create_edge_cases(self) -> list[Case]: ...
    async def execute_agent(self, input_data: MyAgentInput) -> MyAgentActualOutput: ...

# 5. Build dataset and run
suite = MyAgentEvalSuite()
evaluators = [MySpecificEvaluator(), KeywordMatchEvaluator(...), ...]
dataset = suite.build_dataset(evaluators)

@pytest.mark.callout
@pytest.mark.asyncio
async def test_my_agent_evals():
    report = await dataset.evaluate_async(suite.execute_agent)
    report.print()
    assert report.overall_score() >= 0.85

@pytest.mark.asyncio
async def test_single_case_mock():
    # Quick validation without API
    ...
```

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
- [ ] Can run in CI/CD pipeline

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

## Next Actions

- [ ] Complete refactoring work (prerequisite)
- [ ] Choose first agent to implement (recommend: Markdown Input)
- [ ] Gather test data for chosen agent
- [ ] Create evaluation suite using base components
- [ ] Document learnings and patterns
- [ ] Iterate on remaining agents

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
