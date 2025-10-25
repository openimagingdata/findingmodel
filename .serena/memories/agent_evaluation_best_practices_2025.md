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

## Logfire Integration (Automatic)

Logfire observability is configured ONCE at package level in `evals/__init__.py`.
Individual eval modules require ZERO Logfire code.

### What Happens Automatically

When `evals/__init__.py` is imported (automatically on first eval module import):
1. logfire.configure() is called with settings from .env
2. logfire.instrument_pydantic_ai() enables automatic agent tracing

When Dataset.evaluate() is called:
1. Root span created for evaluation suite (automatic)
2. Per-case spans created for each case (automatic)
3. All Pydantic AI spans captured (agent runs, model calls, tools)
4. Inputs, outputs, and scores recorded (automatic)

### For New Eval Suites

NO Logfire code required. Just define:
- Data models
- Evaluators
- Cases
- Dataset
- Main function calling dataset.evaluate()

Observability happens automatically via package-level configuration.

### Package-Level Configuration Pattern

```python
# evals/__init__.py
"""Evaluation suites for findingmodel agents.

Logfire observability configured automatically for entire package.
"""

import logfire
from logfire import ConsoleOptions
from findingmodel.config import settings

# Configure Logfire once for entire evals package
logfire.configure(
    token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
    send_to_logfire=False if settings.disable_send_to_logfire else "if-token-present",
    console=ConsoleOptions(
        colors="auto",
        min_log_level="debug",
    )
    if settings.logfire_verbose
    else False,
)

# Instrument Pydantic AI once for automatic agent/model/tool tracing
logfire.instrument_pydantic_ai()

__all__ = []  # No public exports - configuration only
```

**Benefits:**
- DRY principle: Configuration in ONE place
- Zero Logfire code per eval suite
- Prevents ~80 lines of duplication per new suite
- Follows Python logging best practices
- Automatic instrumentation leveraged fully

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
├── __init__.py                      # Package-level Logfire configuration
├── base.py                          # Reusable evaluators & base classes
├── utils.py                         # Shared helpers
├── model_editor.py                  # ✅ model_editor evaluation (COMPLETE)
├── README.md                        # Quick-start guide for humans
├── evals_guide.md                   # Comprehensive how-to-write guide
└── CLAUDE.md                        # Lightweight pointer to this memory

Future eval suites (flat structure for now):
├── anatomic_search.py               # Anatomic location search evals
├── ontology_match.py                # Ontology concept match evals
└── ... (add more as needed)

test/                                # Test directory (pytest discovers here)
├── test_base_evaluators.py          # Unit tests for evaluator library
├── data/defs/                       # Test data (finding models, etc.)
└── test_*.py                        # Unit and integration tests
```

**Important:** Directory structure is currently flat. Nest into subdirectories only when complexity demands it (e.g., >10 eval suites).

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
- **Imports**: Absolute (`from evals.base import ...`)
- **NO Logfire imports needed** (automatic via evals/__init__.py)

### Required Components

1. **Define data models** (input, expected output, actual output)
2. **Create focused evaluators** inheriting from `Evaluator` base class
3. **Build dataset** with cases and evaluators at module level
4. **Main eval function** that runs `dataset.evaluate()` and returns Report
5. **`__main__` block** for standalone execution

### Template (NO Logfire Code Needed)

```python
"""Evaluation suite for {tool_name} agent.

NO Logfire configuration needed - automatic instrumentation via evals/__init__.py.
"""

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

# 4. Task function (NO manual Logfire spans needed)
async def run_tool_name_task(input_data: ToolInput) -> ToolOutput:
    """Execute task - automatic instrumentation captures everything."""
    result = await tool.process(input_data)
    return ToolOutput(result=result)

# 5. Main eval function (NO manual Logfire spans needed)
async def run_tool_name_evals() -> Report:
    """Run evaluation suite.
    
    Dataset.evaluate() automatically creates spans and captures results.
    """
    report = await dataset.evaluate(run_tool_name_task)
    return report

# 6. Standalone execution
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
        
        # Calculate overall score
        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\nOVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\n")
    
    asyncio.run(main())
```

### Import Patterns

Always use absolute imports, never relative:

```python
# ✅ Correct
from evals.base import ExactMatchEvaluator, KeywordMatchEvaluator
from evals.utils import load_fm_json, compare_models
from findingmodel.tools import model_editor

# ❌ Wrong (old test convention)
from .base import ExactMatchEvaluator
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
# Define evaluators
evaluators = [
    IDPreservationEvaluator(),
    AttributeAdditionEvaluator(),
    ChangeTrackingEvaluator(),
    RejectionAccuracyEvaluator(),
    ContentPreservationEvaluator(),
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

### Converting from Pytest-Based Evals

If migrating from pytest-based eval tests, apply these transformations:

1. **Remove pytest decorators**: Delete `@pytest.mark.callout`, `@pytest.mark.asyncio`
2. **Rename function**: `test_run_X_evals()` → `run_X_evals()`
3. **Change imports**: Relative (`from .base`) → Absolute (`from evals.base`)
4. **Remove Logfire code**: Delete all logfire imports, configure(), spans
5. **Return Report**: Function returns Report object instead of using assertions
6. **Add `__main__` block**: Enable standalone execution via `python -m evals.X`

See `evals/model_editor.py` for complete working example.

## Agents Needing Evaluation Suites

Priority order for creating eval suites:
1. **model_editor** ✅ **COMPLETE** - 12 cases, 5 evaluators, hybrid scoring, automatic Logfire
2. **anatomic_location_search** - Two-agent architecture
3. **ontology_concept_match** - Multi-backend
4. **finding_description** - LLM-generated content
5. **similar_finding_models** - Similarity/ranking
6. **markdown_in** - Parsing accuracy

## Current Status (Updated 2025-10-24)

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

### ✅ Phase 0 Complete (October 2025)
- Package-level Logfire configuration in `evals/__init__.py`
- Zero Logfire code required in individual eval modules
- Automatic instrumentation via Pydantic Evals + Pydantic AI
- Reduced model_editor.py by ~160 lines
- Fixed TestModel bug in tests (convert Pydantic models to dicts)
- All 325 tests passing with Logfire instrumentation enabled
- Documentation consolidated (CLAUDE.md → memory, added TestModel warning to evals_guide.md)
- DRY principle maintained, scalable architecture

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
❌ Adding logfire.configure() to individual eval modules (configure once in __init__.py)
❌ Creating manual spans in eval modules (Dataset.evaluate() does this automatically)
❌ Manual logging in task functions (automatic instrumentation captures everything)
❌ Passing Pydantic models to TestModel custom_output_args (must use `.model_dump()`)
❌ Missing __main__ block in eval suites (prevents standalone execution)

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

### 7. Package-Level Configuration Prevents Duplication
Configuring Logfire in `evals/__init__.py` eliminates ~80 lines per eval suite and follows Python logging best practices. Automatic instrumentation from Pydantic Evals + Pydantic AI captures everything manual spans were capturing.

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

**Discovered:** October 2025 during Phase 0 implementation. Two tests failed when Logfire instrumentation was added to `evals/__init__.py`, exposing this latent bug that existed in the test code.

## Resources

- **Quick start:** `evals/README.md`
- **Full guide:** `evals/evals_guide.md`
- **AI reference:** `evals/CLAUDE.md`
- **Example implementation:** `evals/model_editor.py`
- **Logfire observability:** `docs/logfire_observability_guide.md`
- **Pydantic AI Evals:** https://ai.pydantic.dev/evals/
- **Related memories:** `pydantic_ai_testing_best_practices`, `test_suite_improvements_2025`
