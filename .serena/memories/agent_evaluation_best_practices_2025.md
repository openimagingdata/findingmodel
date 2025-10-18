# Agent Evaluation Best Practices - 2025

## Framework: Pydantic AI Evals

Use [Pydantic AI Evals](https://ai.pydantic.dev/evals/) for all agent evaluation suites. It provides type-safe, observable evaluation of non-deterministic functions.

## Core Pattern

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

# 1. Define data models
class AgentInput(BaseModel): ...
class AgentExpectedOutput(BaseModel): ...
class AgentActualOutput(BaseModel): ...

# 2. Create focused evaluators (return 0.0-1.0)
class MyEvaluator(Evaluator[AgentInput, AgentActualOutput, AgentExpectedOutput]):
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Return continuous score
        return score

# 3. Build dataset AT MODULE LEVEL with evaluators
evaluators = [MyEvaluator(), OtherEvaluator()]
dataset = Dataset(cases=[...], evaluators=evaluators)

# 4. Run evaluation (evaluators already in Dataset)
report = await dataset.evaluate(task_function)
report.print()
assert report.overall_score() >= threshold
```

**IMPORTANT API Details:**
- Method is `evaluate()` NOT `evaluate_async()`
- Evaluators passed to `Dataset()` constructor, NOT to `evaluate()` method
- Use relative imports in test files: `from .utils import ...` NOT `from test.evals.utils import ...`

## Key Principles

### 1. Use Multiple Focused Evaluators
**Don't:** One big evaluator doing everything  
**Do:** Separate evaluators for each aspect (IDs, structure, keywords, success/failure)

Each evaluator tests ONE thing and returns 0.0-1.0 score.

### 2. Hybrid Scoring Approach ⭐️ NEW
Combine strict and partial credit scoring based on requirement type:

**Strict (0.0 or 1.0):** Non-negotiables that must always pass
- ID preservation (IDs must never change)
- Content preservation on rejection (model unchanged when edit rejected)
- Error recording (rejections/changes must be recorded)

**Partial Credit (0.0-1.0):** Quality measures with proportional scoring
- Attribute addition (3/5 attributes added = 0.6)
- Keyword matching (2/3 keywords found = 0.67)

**Hybrid:** Strict check FIRST, then partial credit if passed
```python
def evaluate(self, ctx: EvaluatorContext[...]) -> float:
    # STRICT: Must have changes recorded (non-negotiable)
    if len(ctx.output.changes) == 0:
        return 0.0
    
    # PARTIAL: Quality of change descriptions (proportional)
    if ctx.metadata.keywords:
        matches = sum(1 for kw in keywords if kw.lower() in text)
        return matches / len(keywords)
    return 1.0
```

### 3. Return Continuous Scores
**Don't:** Binary pass/fail only  
**Do:** Partial credit (e.g., 3/5 keywords matched = 0.6)

### 4. Handle Non-Determinism
**Don't:** Exact string matching  
**Do:** Keyword/concept matching, structured outputs, LLM-as-judge

### 5. Organize Cases by Category
- **Success cases**: Agent handles request correctly
- **Rejection cases**: Agent properly rejects unsafe requests  
- **Edge cases**: Boundary conditions, empty/large inputs
- **Error handling**: Network failures, malformed data

### 6. Use Real Test Data
Load actual finding models from `test/data/defs/*.fm.json` rather than synthetic data.

### 7. Follow Project Conventions
- Use `@pytest.mark.callout` for tests requiring API access
- Provide mock tests using `TestModel` for quick validation without API
- Enable Logfire integration for observability (Phase 3)

## Reusable Components

### Base Evaluator Library (`test/evals/base.py`)

Generic reusable evaluators for any agent:
- `ExactMatchEvaluator[InputT, str]`: Exact string match
- `ContainsEvaluator[InputT, str]`: Substring match with case sensitivity option
- `KeywordMatchEvaluator[InputT, OutputT]`: Multiple keyword matching with partial credit
- `StructuralValidityEvaluator[InputT, BaseModel]`: Check Pydantic model has required fields
- `ErrorHandlingEvaluator[InputT, OutputT]`: Verify errors handled as expected

All base evaluators are unit tested in `test/evals/test_base_evaluators.py` (25 tests, 0.09s).

### Agent-Specific Evaluators

Create specialized evaluators in the agent's test file for domain-specific checks.

**Example: model_editor evaluators** (`test/evals/test_model_editor_evals.py`):
```python
# 1. IDPreservationEvaluator (strict) - Model IDs must never change
# 2. AttributeAdditionEvaluator (partial) - Proportional score for attributes added
# 3. ChangeTrackingEvaluator (hybrid) - Changes must be recorded + keyword quality
# 4. RejectionAccuracyEvaluator (hybrid) - Rejections must be recorded + keyword quality
# 5. ContentPreservationEvaluator (strict) - Model unchanged when edits rejected
```

### Base Suite Class

```python
class AgentEvaluationSuite(ABC, Generic[InputT, ExpectedT, ActualT]):
    @abstractmethod
    def create_successful_cases(self) -> list[Case]: ...
    
    @abstractmethod
    def create_failure_cases(self) -> list[Case]: ...
    
    @abstractmethod
    def create_edge_cases(self) -> list[Case]: ...
    
    @abstractmethod
    async def execute_agent(self, input_data: InputT) -> ActualT: ...
```

## Running Evaluations

```bash
# Quick mock test (no API)
pytest test/evals/test_model_editor_evals.py::test_single_successful_case -v

# Full evaluation suite (requires API)
pytest test/evals/test_model_editor_evals.py::test_run_model_editor_evals -v -s
task test-full test/evals/

# Run as standalone
python test/evals/test_model_editor_evals.py

# List available cases
python test/evals/list_cases.py
```

## Evaluation Suite Structure

```
test/evals/
├── base.py                          # Reusable evaluators & base classes
├── test_base_evaluators.py          # Unit tests for base evaluators
├── test_model_editor_evals.py       # ✅ model_editor evaluation (Phase 2 COMPLETE)
├── test_anatomic_search_evals.py    # TODO: anatomic_location_search
├── test_ontology_match_evals.py     # TODO: ontology_concept_match
├── utils.py                         # Shared helpers
├── README.md                        # Documentation
├── add_case_example.py              # Template examples
└── list_cases.py                    # Utility to list all cases
```

## Agents Needing Evaluation Suites

Priority order for creating eval suites:
1. **model_editor** ✅ **COMPLETE (Phase 2)** - 12 cases, 5 evaluators, hybrid scoring
2. **anatomic_location_search** - Two-agent architecture
3. **ontology_concept_match** - Multi-backend
4. **finding_description** - LLM-generated content
5. **similar_finding_models** - Similarity/ranking
6. **markdown_in** - Parsing accuracy

## Current Status (Updated 2025-10-18)

### ✅ Phase 1 Complete
- Base evaluator library (`test/evals/base.py`) - 5 generic evaluators
- Base suite class (`AgentEvaluationSuite`)
- Shared utilities (`test/evals/utils.py`)
- Unit tests for evaluators (`test/evals/test_base_evaluators.py`)

### ✅ Phase 2 Complete
- Refactored model_editor evals to use Evaluator classes
- 5 focused evaluators with hybrid scoring approach
- Using `Dataset.evaluate()` pattern correctly
- All 12 test cases passing (11 original + 1 new)
- Documentation updated (README.md, add_case_example.py)
- Critical import bug fixed (relative imports for test modules)

### 🔲 Phase 3 Optional
- Logfire integration for observability
- Metrics collection and reporting
- Trace-based debugging

## Anti-Patterns to Avoid

❌ Setting `models.ALLOW_MODEL_REQUESTS = False` in eval tests (this blocks TestModel)  
❌ Custom evaluation functions returning dicts instead of Evaluator classes  
❌ Single monolithic evaluator instead of focused evaluators  
❌ Binary pass/fail only (no partial credit for quality measures)  
❌ Exact string matching for LLM outputs  
❌ Testing library functionality instead of your agent logic  
❌ Using `evaluate_async()` method (doesn't exist, use `evaluate()`)  
❌ Passing evaluators to `evaluate()` method (pass to Dataset constructor)  
❌ Absolute imports in test files (use relative: `from .utils import ...`)

## Lessons Learned (Phase 2)

### 1. Hybrid Scoring is Key
Non-negotiables (ID preservation, error recording) must be strict (0.0 or 1.0). Quality measures (keyword matching, completeness) should use partial credit.

### 2. Always Run the Tests
Linting passes doesn't mean code works. Critical import bug prevented tests from running at all until senior review actually executed them.

### 3. Relative Imports in Test Modules
Use `from .utils import ...` NOT `from test.evals.utils import ...` to avoid `ModuleNotFoundError`.

### 4. Evaluators in Dataset Constructor
Evaluators passed to `Dataset(cases=..., evaluators=...)` constructor, NOT to `evaluate()` method.

### 5. API is `evaluate()` Not `evaluate_async()`
The method name is `evaluate()` despite being async. There is no `evaluate_async()`.

## Resources

- **Full guide:** `docs/evaluation_guide.md`
- **Phase 2 implementation:** `test/evals/test_model_editor_evals.py`
- **Phase 2 review:** `tasks/PHASE_2_SENIOR_REVIEW.md`
- **Refactoring plan:** `tasks/refactor_model_editor_evals.md`
- **Pydantic AI Evals:** https://ai.pydantic.dev/evals/
- **Related memories:** `pydantic_ai_testing_best_practices`, `test_suite_improvements_2025`
