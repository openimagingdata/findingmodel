# Agent Evaluation Best Practices - 2025

**Last Updated:** 2025-10-29 (Updated for evaluator architecture refactoring)

## Framework: Pydantic AI Evals

Use [Pydantic AI Evals](https://ai.pydantic.dev/evals/) for all agent evaluation suites. It provides type-safe, observable evaluation of non-deterministic functions.

## Evaluator Architecture

See `.serena/memories/evaluator_architecture_2025.md` for complete guidance on where evaluators live and when to use built-in vs. custom evaluators.

**Quick Summary:**
1. **Prefer Pydantic Evals built-ins** (EqualsExpected, Contains, IsInstance, LLMJudge)
2. **Keep evaluators inline** in eval scripts (agent-specific, < 20 lines, single-use)
3. **Extract to `src/findingmodel/tools/evaluators.py`** ONLY if truly reusable (used 2+ times AND complex)

## Core Pattern

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EqualsExpected, Contains
from findingmodel.tools.evaluators import PerformanceEvaluator  # If needed

# 1. Define data models
class AgentInput(BaseModel): ...
class AgentExpectedOutput(BaseModel): ...
class AgentActualOutput(BaseModel): ...

# 2. Create focused evaluators (return 0.0-1.0)
@dataclass
class MyEvaluator(Evaluator[AgentInput, AgentActualOutput, AgentExpectedOutput]):
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Return continuous score
        return score

# 3. Build dataset AT MODULE LEVEL with evaluators
evaluators = [
    EqualsExpected(),  # Built-in from Pydantic Evals
    PerformanceEvaluator(time_limit=5.0),  # From src/findingmodel/tools/evaluators.py
    MyEvaluator(),  # Inline evaluator
]
dataset = Dataset(cases=[...], evaluators=evaluators)

# 4. Run evaluation (evaluators already in Dataset)
report = await dataset.evaluate(task_function)
report.print()
assert report.overall_score() >= threshold
```

**IMPORTANT API Details:**
- Method is `evaluate()` NOT `evaluate_async()`
- Evaluators passed to `Dataset()` constructor, NOT to `evaluate()` method
- Use absolute imports in eval files: `from evals.utils import ...` NOT `from .utils import ...`

## Logfire Integration (Automatic)

Logfire observability is configured via `ensure_instrumented()` called in each eval script's `__main__` block.
Individual eval modules require minimal Logfire code.

### What Happens When `ensure_instrumented()` Is Called

When `ensure_instrumented()` is called in `__main__`:
1. logfire.configure() is called with settings from .env (idempotent)
2. logfire.instrument_pydantic_ai() enables automatic agent tracing (idempotent)

When Dataset.evaluate() is called:
1. Root span created for evaluation suite (automatic)
2. Per-case spans created for each case (automatic)
3. All Pydantic AI spans captured (agent runs, model calls, tools)
4. Inputs, outputs, and scores recorded (automatic)

### For New Eval Suites

Minimal Logfire code required - just call `ensure_instrumented()` in `__main__`:

```python
if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()  # Explicit instrumentation for eval run

    async def main() -> None:
        report = await run_X_evals()
        # Print report...

    asyncio.run(main())
```

Then define:
- Data models
- Evaluators (prefer built-ins → inline → reusable)
- Cases
- Dataset
- Main function calling dataset.evaluate()

Observability happens automatically via instrumentation.

### Package-Level Configuration Pattern

```python
# evals/__init__.py
"""Evaluation suites for findingmodel agents.

This module provides Logfire instrumentation configuration for eval suites.
"""

import logfire
from logfire import ConsoleOptions
from findingmodel.config import settings

_instrumented = False

def ensure_instrumented() -> None:
    """Ensure Logfire is configured and Pydantic AI is instrumented.
    
    This function is idempotent - safe to call multiple times.
    Call this explicitly in eval suite __main__ blocks before running evals.
    """
    global _instrumented
    
    if _instrumented:
        return
    
    logfire.configure(
        token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
        send_to_logfire=False if settings.disable_send_to_logfire else "if-token-present",
        console=ConsoleOptions(colors="auto", min_log_level="debug")
        if settings.logfire_verbose
        else False,
    )
    
    logfire.instrument_pydantic_ai()
    _instrumented = True

__all__ = ["ensure_instrumented"]
```

**Benefits:**
- Idempotent configuration (safe to call multiple times)
- Prevents instrumentation at import time (doesn't affect unit tests)
- Minimal Logfire code per eval suite (~3 lines in __main__)

## Key Principles

### 1. Use Multiple Focused Evaluators
**Don't:** One big evaluator doing everything  
**Do:** Separate evaluators for each aspect (IDs, structure, keywords, success/failure)

Each evaluator tests ONE thing and returns 0.0-1.0 score.

### 2. Hybrid Scoring Approach ⭐️
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

## Reusable Evaluators

### From Pydantic Evals (Prefer These First)
- `EqualsExpected` - Exact match comparison
- `Contains` - Substring/membership checks
- `IsInstance` - Type validation
- `LLMJudge` - AI-assisted evaluation

See: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals/evaluators/built-in.md

### From `src/findingmodel/tools/evaluators.py`
- `PerformanceEvaluator` - Validates execution time against configurable threshold (used by 5 eval suites)

**When to add here:** ONLY if used 2+ times AND complex logic AND non-trivial
**See:** `.serena/memories/evaluator_architecture_2025.md` for decision framework

### Inline in Eval Scripts
Most evaluators should remain inline for clarity and context. Examples:
- `IDPreservationEvaluator` - Model IDs must never change (model_editor.py)
- `AttributeAdditionEvaluator` - Proportional score for attributes added (model_editor.py)
- `ChangeTrackingEvaluator` - Changes recorded + keyword quality (model_editor.py)
- `DuplicateDetectionEvaluator` - Duplicate detection scoring (similar_models.py)
- `RankingQualityEvaluator` - Ranking quality assessment (similar_models.py)
- And 30+ more agent-specific evaluators across eval suites

## Running Evaluations

```bash
# Run all eval suites
task evals

# Run specific suite
task evals:model_editor
task evals:similar_models
python -m evals.model_editor

# From Python
from evals.model_editor import run_model_editor_evals
report = await run_model_editor_evals()
```

## Evaluation Suite Structure

```
evals/                               # Root-level directory (NOT in test/)
├── __init__.py                      # Package-level Logfire configuration
├── utils.py                         # Shared helpers
├── model_editor.py                  # ✅ model_editor evaluation (COMPLETE)
├── similar_models.py                # ✅ similar_models evaluation (COMPLETE)
├── ontology_match.py                # ✅ ontology_match evaluation (COMPLETE)
├── anatomic_search.py               # ✅ anatomic_search evaluation (COMPLETE)
├── markdown_in.py                   # ✅ markdown_in evaluation (COMPLETE)
├── finding_description.py           # ✅ finding_description evaluation (COMPLETE)
├── README.md                        # Quick-start guide for humans
├── evals_guide.md                   # Comprehensive how-to-write guide
└── CLAUDE.md                        # Lightweight pointer to this memory

src/findingmodel/tools/
└── evaluators.py                    # ONLY truly reusable evaluators

test/                                # Test directory (pytest discovers here)
├── tools/
│   └── test_evaluators.py          # Unit tests for src/findingmodel/tools/evaluators.py
├── data/defs/                       # Test data (finding models, etc.)
└── test_*.py                        # Unit and integration tests
```

**Important:** Unit tests NEVER import from `evals/` - only from `src/findingmodel/`

**Key distinction:** Evals assess behavioral quality (0.0-1.0 scores), tests verify correctness (pass/fail).

## Creating New Eval Suites

### When to Create New Eval Suites

Create a new eval suite when:
1. **New agent added**: Any new AI agent or tool with non-deterministic behavior
2. **Complex functionality**: Tools that require comprehensive behavioral assessment
3. **Regression prevention**: Critical functionality that needs quality tracking

Do NOT create evals for:
- Deterministic utility functions (use unit tests)
- Simple data transformations (use unit tests)
- One-off scripts or experiments

### File Naming Conventions

**Eval Suite Files:**
- **Naming**: `tool_name.py` NOT `test_tool_name.py`
  - ✅ `model_editor.py`
  - ❌ `test_model_editor.py` (old test convention)

- **Main function**: `run_tool_name_evals()` NOT `test_run_tool_name_evals()`
  - ✅ `async def run_model_editor_evals():`
  - ❌ `async def test_run_model_editor_evals():` (old test convention)

- **Return value**: Main function should return the Report object
  ```python
  async def run_model_editor_evals():
      report = await dataset.evaluate(task_function)
      return report
  ```

### File Structure
- **Filename**: `evals/tool_name.py` (NOT `test_tool_name.py`)
- **Main function**: `run_tool_name_evals()` (NOT `test_run_tool_name_evals()`)
- **Imports**: Absolute (`from evals.utils import ...`)
- **Minimal Logfire**: Just call `ensure_instrumented()` in `__main__`

### Required Components

1. **Define data models** (input, expected output, actual output)
2. **Create focused evaluators** (prefer built-ins → inline → reusable)
3. **Build dataset** with cases and evaluators at module level
4. **Main eval function** that runs `dataset.evaluate()` and returns Report
5. **`__main__` block** with `ensure_instrumented()` call for standalone execution

### Template

```python
"""Evaluation suite for {tool_name} agent.

Minimal Logfire configuration - just call ensure_instrumented() in __main__.
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EqualsExpected, Contains
from evals.utils import load_fm_json
from findingmodel.tools.evaluators import PerformanceEvaluator

# 1. Data models
class ToolInput(BaseModel): ...
class ToolExpected(BaseModel): ...
class ToolOutput(BaseModel): ...

# 2. Inline evaluators (agent-specific, focused)
@dataclass
class CustomEvaluator(Evaluator[ToolInput, ToolOutput, ToolExpected]):
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Return 0.0-1.0 score
        return score

# 3. Dataset (at module level)
evaluators = [
    EqualsExpected(),  # Built-in
    PerformanceEvaluator(time_limit=5.0),  # Reusable
    CustomEvaluator(),  # Inline
]
dataset = Dataset(
    cases=[
        Case(input=..., expected_output=..., metadata=...),
        # More cases...
    ],
    evaluators=evaluators,
)

# 4. Task function (automatic instrumentation)
async def run_tool_name_task(input_data: ToolInput) -> ToolOutput:
    """Execute task - automatic instrumentation captures everything."""
    result = await tool.process(input_data)
    return ToolOutput(result=result)

# 5. Main eval function (automatic instrumentation)
async def run_tool_name_evals() -> Report:
    """Run evaluation suite.
    
    Dataset.evaluate() automatically creates spans and captures results.
    """
    report = await dataset.evaluate(run_tool_name_task)
    return report

# 6. Standalone execution
if __name__ == "__main__":
    import asyncio

    from evals import ensure_instrumented

    ensure_instrumented()  # Explicit instrumentation for eval run

    async def main():
        print("\\nRunning {tool_name} evaluation suite...")
        print("=" * 80)
        
        report = await run_tool_name_evals()
        
        print("\\n" + "=" * 80)
        print("{TOOL_NAME} EVALUATION RESULTS")
        print("=" * 80 + "\\n")
        report.print(include_input=False, include_output=True)
        
        # Calculate overall score
        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\\nOVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\\n")
    
    asyncio.run(main())
```

### Import Patterns

Always use absolute imports, never relative:

```python
# ✅ Correct
from pydantic_evals.evaluators import EqualsExpected, Contains, IsInstance
from findingmodel.tools.evaluators import PerformanceEvaluator
from evals.utils import load_fm_json, compare_models
from findingmodel.tools import model_editor

# ❌ Wrong (old test convention)
from .utils import load_fm_json
```

**From other modules:**
```python
# Importing eval utilities
from evals.utils import load_fm_json

# Running eval suite
from evals.model_editor import run_model_editor_evals
report = await run_model_editor_evals()
```

### Case Organization Patterns

Organize cases into clear categories:

```python
def create_successful_cases() -> list[Case]:
    """Cases where agent should succeed."""
    ...

def create_rejection_cases() -> list[Case]:
    """Cases where agent should reject gracefully."""
    ...

def create_edge_cases() -> list[Case]:
    """Boundary conditions and unusual inputs."""
    ...

# Combine at module level
all_cases = create_successful_cases() + create_rejection_cases() + create_edge_cases()
```

### Dataset and Evaluator Setup

Define dataset at module level with evaluators:

```python
# Define evaluators (prefer built-ins → inline → reusable)
evaluators = [
    # Built-ins from Pydantic Evals
    EqualsExpected(),
    Contains(expected="keyword"),
    
    # Reusable from src/
    PerformanceEvaluator(time_limit=5.0),
    
    # Inline evaluators
    IDPreservationEvaluator(),
    AttributeAdditionEvaluator(),
]

# Create dataset with cases and evaluators
tool_name_dataset = Dataset(cases=all_cases, evaluators=evaluators)

# In main function, just call evaluate()
async def run_tool_name_evals():
    report = await tool_name_dataset.evaluate(task_function)
    return report
```

### Test Data Patterns

Eval suites access test data from `test/data/defs/`:

```python
from pathlib import Path

def load_fm_json(filename: str) -> str:
    """Load finding model JSON from test data."""
    test_data_dir = Path(__file__).parent.parent / "test" / "data" / "defs"
    return (test_data_dir / filename).read_text()
```

## Current Status (Updated 2025-10-29)

### ✅ Evaluator Architecture Refactoring Complete (October 2025)
- Deleted unused `evals/base.py` (~350 lines) - evaluators were never used by eval scripts
- Deleted `test/test_base_evaluators.py` (~600 lines) - testing unused code
- Created `src/findingmodel/tools/evaluators.py` with ONLY PerformanceEvaluator
- Created `test/tools/test_evaluators.py` with 20 unit tests
- Updated 5 eval scripts to use centralized PerformanceEvaluator (~190 lines duplication removed)
- Net result: ~1,010 lines deleted, ~130 lines added
- Clean architecture: unit tests never import from `evals/`
- See: `.serena/memories/evaluator_architecture_2025.md` for complete details

### ✅ Eval/Test Separation Complete (October 2025)
- Moved eval suites from `test/evals/` to root `evals/` directory
- Pytest no longer discovers eval suites during normal test runs
- Clear three-tier structure: unit tests, integration tests, evals
- Added `task evals` and `task evals:X` commands
- Documentation: `evals/README.md`, `evals/evals_guide.md`, `evals/CLAUDE.md`

### ✅ Six Eval Suites Complete
1. **model_editor** ✅ - 12 cases, 5 inline evaluators, hybrid scoring
2. **similar_models** ✅ - Uses PerformanceEvaluator from src/
3. **ontology_match** ✅ - Uses PerformanceEvaluator from src/
4. **anatomic_search** ✅ - Uses PerformanceEvaluator from src/
5. **markdown_in** ✅ - Uses PerformanceEvaluator from src/
6. **finding_description** ✅ - Uses PerformanceEvaluator from src/

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
❌ Relative imports in eval files (use `from evals.utils` not `from .utils`)
❌ Naming eval files with `test_` prefix (use `tool_name.py`)
❌ Creating manual spans in eval modules (Dataset.evaluate() does this automatically)
❌ Manual logging in task functions (automatic instrumentation captures everything)
❌ Passing Pydantic models to TestModel custom_output_args (must use `.model_dump()`)
❌ Missing __main__ block in eval suites (prevents standalone execution)
❌ Missing ensure_instrumented() call in __main__ (Logfire won't instrument)
❌ Extracting evaluators to src/ prematurely (prefer inline, extract only if used 2+ times AND complex)
❌ Duplicating Pydantic Evals built-ins (use EqualsExpected, Contains, IsInstance first)

## Lessons Learned

### 1. Hybrid Scoring is Key
Non-negotiables (ID preservation, error recording) must be strict (0.0 or 1.0). Quality measures (keyword matching, completeness) should use partial credit.

### 2. Always Run the Tests
Linting passes doesn't mean code works. Critical import bug prevented tests from running at all until senior review actually executed them.

### 3. Absolute Imports in Eval Files
Use `from evals.utils import ...` NOT `from .utils import ...` since eval files are modules, not test files.

### 4. Evaluators in Dataset Constructor
Evaluators passed to `Dataset(cases=..., evaluators=...)` constructor, NOT to `evaluate()` method.

### 5. API is `evaluate()` Not `evaluate_async()`
The method name is `evaluate()` despite being async. There is no `evaluate_async()`.

### 6. Evals vs Tests Separation Matters
- **Tests** (in `test/`): Verify correctness with pass/fail, discovered by pytest
- **Evals** (in `evals/`): Assess quality with 0.0-1.0 scores, run standalone
- Keeping them separate prevents slow expensive evals from running during normal testing

### 7. Idempotent Instrumentation Prevents Duplication
Using `ensure_instrumented()` eliminates duplication while keeping instrumentation explicit. Automatic instrumentation from Pydantic Evals + Pydantic AI captures everything manual spans were capturing.

### 8. TestModel Requires Dicts Not Pydantic Models
When using TestModel with `custom_output_args`, you MUST pass a dict, not a Pydantic model.

**Why:** `ToolCallPart.args` is typed as `str | dict[str, Any] | None` and does NOT support Pydantic models directly. When Logfire instrumentation is enabled via `logfire.instrument_pydantic_ai()`, it tries to serialize tool call arguments for tracing and calls `.items()` on them, which Pydantic models don't have.

**The Bug:** This issue only manifests when instrumentation is active. Without instrumentation, passing a Pydantic model works (silently violating the type contract). With instrumentation, tests fail with `AttributeError: 'ModelName' object has no attribute 'items'`.

**The Fix:**
```python
# ❌ Wrong - fails with Logfire instrumentation
mock_output = EditResult(model=..., changes=[...])
TestModel(custom_output_args=mock_output)

# ✅ Correct - always works
mock_output = EditResult(model=..., changes=[...])
TestModel(custom_output_args=mock_output.model_dump())
```

### 9. Most Evaluators Should Stay Inline
Phase 0 research revealed that Pydantic Evals philosophy is: built-ins → inline → reusable (in that order). Most evaluators belong inline in eval scripts for clarity and context. Only extract to src/ if truly reusable (used 2+ times) AND complex AND non-trivial.

### 10. Wheel Reinvention Wastes Effort
Before creating custom evaluators, check Pydantic Evals built-ins (EqualsExpected, Contains, IsInstance, LLMJudge). Don't duplicate what already exists.

## Resources

- **Architecture reference:** `.serena/memories/evaluator_architecture_2025.md`
- **Quick start:** `evals/README.md`
- **Full guide:** `evals/evals_guide.md`
- **AI reference:** `evals/CLAUDE.md`
- **Example implementations:** `evals/model_editor.py`, `evals/similar_models.py`
- **Pydantic AI Evals:** https://ai.pydantic.dev/evals/
- **Built-in Evaluators:** https://github.com/pydantic/pydantic-ai/blob/main/docs/evals/evaluators/built-in.md
- **Related memories:** `pydantic_ai_testing_best_practices`, `test_suite_improvements_2025`
