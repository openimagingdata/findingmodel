# CLAUDE.md - Eval Development Reference for AI Agents

This file provides project-specific conventions for Claude Code and GitHub Copilot when working with evaluation suites.

## Directory Structure

```
evals/                               # Root-level directory (NOT in test/)
├── __init__.py                      # Package marker
├── base.py                          # Reusable evaluators & base classes
├── utils.py                         # Shared helpers
├── model_editor.py                  # model_editor evaluation suite
├── README.md                        # Quick-start guide for humans
├── evals_guide.md                   # Comprehensive how-to-write guide
└── CLAUDE.md                        # This file (AI agent reference)

Future eval suites (flat structure for now):
├── anatomic_search.py               # Anatomic location search evals
├── ontology_match.py                # Ontology concept match evals
└── ... (add more as needed)
```

**Important**: Directory structure is currently flat. Nest into subdirectories only when complexity demands it (e.g., >10 eval suites).

## File Naming Conventions

### Eval Suite Files

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

### Required __main__ Block

Every eval suite must include a __main__ block for standalone execution:

```python
if __name__ == "__main__":
    import asyncio

    async def main():
        print("\nRunning {tool_name} evaluation suite...")
        print("=" * 80)

        report = await run_{tool_name}_evals()

        print("\n" + "=" * 80)
        print("{TOOL NAME} EVALUATION RESULTS")
        print("=" * 80 + "\n")
        report.print(include_input=False, include_output=True)

        # Calculate overall score (average of all evaluator scores across all cases)
        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\nOVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\n")

        # Future Phase 3: Logfire integration via pydantic-evals
        # Future: Save report to file, compare to baseline, etc.

    asyncio.run(main())
```

## When to Create New Eval Suites

Create a new eval suite when:

1. **New agent added**: Any new AI agent or tool with non-deterministic behavior
2. **Complex functionality**: Tools that require comprehensive behavioral assessment
3. **Regression prevention**: Critical functionality that needs quality tracking

Do NOT create evals for:

- Deterministic utility functions (use unit tests)
- Simple data transformations (use unit tests)
- One-off scripts or experiments

## Import Patterns

### Within Eval Suites

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

### From Other Modules

```python
# Importing eval utilities
from evals.utils import load_fm_json

# Running eval suite
from evals.model_editor import run_model_editor_evals
report = await run_model_editor_evals()
```

## Evaluator Patterns

Follow the hybrid scoring approach documented in `evals_guide.md`:

1. **Strict evaluators** (0.0 or 1.0): Non-negotiable requirements
   - ID preservation
   - Content preservation on rejection
   - Error handling (success/failure as expected)

2. **Partial credit evaluators** (0.0-1.0): Quality measures
   - Keyword matching (2/3 keywords = 0.67)
   - Attribute addition (3/5 attributes = 0.60)

3. **Hybrid evaluators**: Strict check first, then partial credit
   ```python
   def evaluate(self, ctx: EvaluatorContext[...]) -> float:
       # STRICT: Must have changes recorded
       if len(ctx.output.changes) == 0:
           return 0.0
       
       # PARTIAL: Quality of descriptions
       matches = sum(1 for kw in keywords if kw.lower() in text)
       return matches / len(keywords)
   ```

## Logfire Integration (Phase 3 - Planned)

For detailed Logfire patterns and instrumentation, see `../docs/logfire_observability_guide.md`.

### Quick Reference

When implementing Phase 3 Logfire integration:

- Wrap eval execution in `logfire.span('eval_suite', tool_name='model_editor')`
- Include case metadata as span attributes (case name, edit type, model ID)
- Log evaluation results as structured events
- Configure with `send_to_logfire='if-token-present'` for graceful degradation
- Use `logfire.instrument_pydantic()` for automatic model tracing

### Current Status

Phase 3 is planned but not yet implemented. Evals currently output to console only via `report.print()`.

## Running Evals

### Via Taskfile

```bash
# All evals
task evals

# Specific suite
task evals:model_editor
```

### Directly via Python

```bash
python -m evals.model_editor
```

### From Python Code

```python
from evals.model_editor import run_model_editor_evals

report = await run_model_editor_evals()

# Calculate overall score (average of all evaluator scores across all cases)
all_scores = [score.value for case in report.cases for score in case.scores.values()]
overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
print(f"Overall score: {overall_score:.2f}")
```

## Case Organization

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

## Dataset and Evaluator Setup

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
{tool_name}_dataset = Dataset(cases=all_cases, evaluators=evaluators)

# In main function, just call evaluate()
async def run_{tool_name}_evals():
    report = await {tool_name}_dataset.evaluate(task_function)
    return report
```

## Test Data

Eval suites access test data from `test/data/defs/`:

```python
from pathlib import Path

def load_fm_json(filename: str) -> str:
    """Load finding model JSON from test data."""
    test_data_dir = Path(__file__).parent.parent / "test" / "data" / "defs"
    return (test_data_dir / filename).read_text()
```

## Documentation Links

- **Quick start**: `evals/README.md` - How to run evals
- **Comprehensive guide**: `evals/evals_guide.md` - How to write evals
- **Serena memory**: `.serena/memories/agent_evaluation_best_practices_2025.md`
- **Pydantic AI Evals**: https://ai.pydantic.dev/evals/

## Common Pitfalls

1. **Using pytest decorators**: Remove `@pytest.mark.callout`, `@pytest.mark.asyncio`
2. **Relative imports**: Use `from evals.` not `from .`
3. **Test naming**: Use `run_tool_name_evals()` not `test_run_tool_name_evals()`
4. **Missing __main__ block**: Always include for standalone execution
5. **Wrong paths**: Test data is at `test/data/defs/` not `data/defs/`

## Example Template

See `evals/model_editor.py` for a complete working example following all conventions.

