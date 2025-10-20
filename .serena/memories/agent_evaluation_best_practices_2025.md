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
- Use absolute imports in eval files: `from evals.base import ...` NOT `from .base import ...`

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
- Eval suites run standalone via `task evals` or `python -m evals.tool_name`
- For integration tests requiring APIs, use `@pytest.mark.callout` (in test/ not evals/)
- Provide mock tests using `TestModel` for quick validation without API

## Reusable Components

### Base Evaluator Library (`evals/base.py`)

Generic reusable evaluators for any agent:
- `ExactMatchEvaluator[InputT, str]`: Exact string match
- `ContainsEvaluator[InputT, str]`: Substring match with case sensitivity option
- `KeywordMatchEvaluator[InputT, OutputT]`: Multiple keyword matching with partial credit
- `StructuralValidityEvaluator[InputT, BaseModel]`: Check Pydantic model has required fields
- `ErrorHandlingEvaluator[InputT, OutputT]`: Verify errors handled as expected

All base evaluators are unit tested in `test/test_base_evaluators.py` (25 tests).

### Agent-Specific Evaluators

Create specialized evaluators in the agent's eval file for domain-specific checks.

**Example: model_editor evaluators** (`evals/model_editor.py`):
```python
# 1. IDPreservationEvaluator (strict) - Model IDs must never change
# 2. AttributeAdditionEvaluator (partial) - Proportional score for attributes added
# 3. ChangeTrackingEvaluator (hybrid) - Changes must be recorded + keyword quality
# 4. RejectionAccuracyEvaluator (hybrid) - Rejections must be recorded + keyword quality
# 5. ContentPreservationEvaluator (strict) - Model unchanged when edits rejected
```

## Running Evaluations

```bash
# Run all eval suites
task evals

# Run specific suite
task evals:model_editor
python -m evals.model_editor

# From Python
from evals.model_editor import run_model_editor_evals
report = await run_model_editor_evals()
```

## Evaluation Suite Structure

```
evals/                               # Root-level directory (NOT in test/)
├── base.py                          # Reusable evaluators & base classes
├── model_editor.py                  # ✅ model_editor evaluation (COMPLETE)
├── utils.py                         # Shared helpers
├── README.md                        # Quick-start guide
├── evals_guide.md                   # Comprehensive how-to-write guide
└── CLAUDE.md                        # AI agent reference for eval development

test/                                # Test directory (pytest discovers here)
├── test_base_evaluators.py          # Unit tests for evaluator library
└── test_*.py                        # Unit and integration tests
```

**Key distinction:** Evals assess behavioral quality (0.0-1.0 scores), tests verify correctness (pass/fail).

## Creating New Eval Suites

When creating new eval suites, follow this pattern:

### File Structure
- **Filename**: `evals/tool_name.py` (NOT `test_tool_name.py`)
- **Main function**: `run_tool_name_evals()` (NOT `test_run_tool_name_evals()`)
- **Imports**: Absolute (`from evals.base import ...`)

### Required Components

1. **Define data models** (input, expected output, actual output)
2. **Create focused evaluators** inheriting from `Evaluator` base class
3. **Build dataset** with cases and evaluators at module level
4. **Main eval function** that runs `dataset.evaluate()` and returns Report
5. **`__main__` block** for standalone execution

### Template

```python
"""Evaluation suite for {tool_name} agent."""

from evals.base import ExactMatchEvaluator, KeywordMatchEvaluator
from evals.utils import load_fm_json
from pydantic_evals import Case, Dataset

# 1. Data models
class ToolInput(BaseModel): ...
class ToolExpected(BaseModel): ...
class ToolOutput(BaseModel): ...

# 2. Evaluators (focused, hybrid scoring)
class CustomEvaluator(Evaluator[ToolInput, ToolOutput, ToolExpected]):
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Return 0.0-1.0 score
        return score

# 3. Dataset (at module level)
evaluators = [CustomEvaluator(), KeywordMatchEvaluator(...)]
dataset = Dataset(
    cases=[
        Case(input=..., expected_output=..., metadata=...),
        # More cases...
    ],
    evaluators=evaluators,
)

# 4. Main eval function
async def run_tool_name_evals() -> Report:
    """Run evaluation suite for {tool_name}."""
    async def task_fn(input_data: ToolInput) -> ToolOutput:
        # Execute agent and return output
        return result
    
    report = await dataset.evaluate(task_fn)
    return report

# 5. Standalone execution
if __name__ == "__main__":
    import asyncio

    async def main():
        print("\nRunning {tool_name} evaluation suite...")
        print("=" * 80)
        
        report = await run_tool_name_evals()
        
        print("\n" + "=" * 80)
        print("{TOOL_NAME} EVALUATION RESULTS")
        print("=" * 80 + "\n")
        report.print(include_input=False, include_output=True)
        print(f"\nOVERALL SCORE: {report.overall_score():.2f}")
        print("=" * 80 + "\n")
    
    asyncio.run(main())
```

### Converting from Pytest-Based Evals

If migrating from pytest-based eval tests, apply these transformations:

1. **Remove pytest decorators**: Delete `@pytest.mark.callout`, `@pytest.mark.asyncio`
2. **Rename function**: `test_run_X_evals()` → `run_X_evals()`
3. **Change imports**: Relative (`from .base`) → Absolute (`from evals.base`)
4. **Return Report**: Function returns Report object instead of using assertions
5. **Add `__main__` block**: Enable standalone execution via `python -m evals.X`

See `evals/model_editor.py` for complete working example.

## Agents Needing Evaluation Suites

Priority order for creating eval suites:
1. **model_editor** ✅ **COMPLETE** - 12 cases, 5 evaluators, hybrid scoring
2. **anatomic_location_search** - Two-agent architecture
3. **ontology_concept_match** - Multi-backend
4. **finding_description** - LLM-generated content
5. **similar_finding_models** - Similarity/ranking
6. **markdown_in** - Parsing accuracy

## Current Status (Updated 2025-10-21)

### ✅ Eval/Test Separation Complete (October 2025)
- Moved eval suites from `test/evals/` to root `evals/` directory
- Pytest no longer discovers eval suites during normal test runs
- Clear three-tier structure: unit tests, integration tests, evals
- Added `task evals` and `task evals:model_editor` commands
- Documentation: `evals/README.md`, `evals/evals_guide.md`, `evals/CLAUDE.md`

### ✅ Phase 1 Complete
- Base evaluator library (`evals/base.py`) - 5 generic evaluators
- Shared utilities (`evals/utils.py`)
- Unit tests for evaluators (`test/test_base_evaluators.py`)

### ✅ Phase 2 Complete
- Refactored model_editor evals to use Evaluator classes
- 5 focused evaluators with hybrid scoring approach
- Using `Dataset.evaluate()` pattern correctly
- All 12 test cases passing
- Standalone execution via `python -m evals.model_editor`

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
❌ Using pytest decorators in eval suites (evals run standalone, not via pytest)  
❌ Relative imports in eval files (use `from evals.base` not `from .base`)  
❌ Naming eval files with `test_` prefix (use `tool_name.py`)

## Lessons Learned

### 1. Hybrid Scoring is Key
Non-negotiables (ID preservation, error recording) must be strict (0.0 or 1.0). Quality measures (keyword matching, completeness) should use partial credit.

### 2. Always Run the Tests
Linting passes doesn't mean code works. Critical import bug prevented tests from running at all until senior review actually executed them.

### 3. Absolute Imports in Eval Files
Use `from evals.base import ...` NOT `from .base import ...` since eval files are modules, not test files.

### 4. Evaluators in Dataset Constructor
Evaluators passed to `Dataset(cases=..., evaluators=...)` constructor, NOT to `evaluate()` method.

### 5. API is `evaluate()` Not `evaluate_async()`
The method name is `evaluate()` despite being async. There is no `evaluate_async()`.

### 6. Evals vs Tests Separation Matters
- **Tests** (in `test/`): Verify correctness with pass/fail, discovered by pytest
- **Evals** (in `evals/`): Assess quality with 0.0-1.0 scores, run standalone
- Keeping them separate prevents slow expensive evals from running during normal testing

## Resources

- **Quick start:** `evals/README.md`
- **Full guide:** `evals/evals_guide.md`
- **AI reference:** `evals/CLAUDE.md`
- **Example implementation:** `evals/model_editor.py`
- **Refactoring plan:** `tasks/refactor_model_editor_evals.md`
- **Pydantic AI Evals:** https://ai.pydantic.dev/evals/
- **Related memories:** `pydantic_ai_testing_best_practices`, `test_suite_improvements_2025`
