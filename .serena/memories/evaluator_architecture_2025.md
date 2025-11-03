# Evaluator Architecture (2025)

**Last Updated:** 2025-10-29  
**Context:** Architectural refactoring completed via `tasks/refactor-evaluator-architecture.md`

## Overview

This document describes where evaluators live in the findingmodel project and when to use built-in vs. custom evaluators.

## Evaluator Hierarchy (Priority Order)

1. **Pydantic Evals Built-ins** (ALWAYS prefer these first)
   - `EqualsExpected` - Exact match comparison
   - `Contains` - Substring/membership checks
   - `IsInstance` - Type validation
   - `LLMJudge` - AI-assisted evaluation
   - See: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals/evaluators/built-in.md

2. **Inline Evaluators in Eval Scripts** (for most custom cases)
   - Agent-specific logic
   - Simple evaluators (< 20 lines)
   - Single-use evaluators
   - Keeps domain knowledge close to usage
   - Examples: IDPreservationEvaluator, AttributeAdditionEvaluator, RankingQualityEvaluator

3. **Reusable Custom Evaluators** (only if truly needed)
   - Location: `src/findingmodel/tools/evaluators.py`
   - Criteria: Used 2+ times AND complex logic AND non-trivial
   - Currently contains: `PerformanceEvaluator` (used by 5 eval suites)

## Where Evaluators Live

### `src/findingmodel/tools/evaluators.py`

**Purpose:** ONLY truly reusable evaluators used across multiple eval suites

**Current Evaluators:**
- `PerformanceEvaluator` - Validates execution time against configurable threshold

**Philosophy:**
- This module should remain SMALL
- Most evaluators belong inline in eval scripts
- Extract to here ONLY if:
  - Used by 2+ eval suites
  - Complex logic worth unit testing
  - Non-trivial calculation or state

**Module Docstring Guidance:**
```python
"""Reusable evaluators for Pydantic AI evaluation suites.

EVALUATOR PHILOSOPHY:
    This module contains ONLY truly reusable evaluators - those used across multiple eval suites
    with complex, non-trivial logic. Most evaluators should remain inline in eval scripts for
    clarity and context.

    Prefer this hierarchy:
    1. Pydantic Evals built-in evaluators (if available)
    2. Inline evaluators in eval scripts (for most cases)
    3. Evaluators in this module (only if used 2+ times and complex)
"""
```

### Inline in Eval Scripts (`evals/*.py`)

**35+ evaluators remain inline** - this is correct architecture per Pydantic Evals patterns

**Examples:**
- `evals/model_editor.py`: IDPreservationEvaluator, AttributeAdditionEvaluator, ChangeTrackingEvaluator, RejectionAccuracyEvaluator, ContentPreservationEvaluator
- `evals/similar_models.py`: DuplicateDetectionEvaluator, RankingQualityEvaluator, PrecisionAtKEvaluator, SemanticSimilarityEvaluator, ExclusionEvaluator
- `evals/ontology_match.py`: (various match-specific evaluators)
- `evals/anatomic_search.py`: (various search-specific evaluators)
- `evals/markdown_in.py`: (various markdown-specific evaluators)
- `evals/finding_description.py`: (various description-specific evaluators)

**Why Inline:**
- Agent-specific domain knowledge
- Clear context for maintainers
- Follows Pydantic Evals examples
- Easier to modify per eval suite needs

## Import Patterns

### For Eval Scripts

```python
# Built-in evaluators from Pydantic Evals
from pydantic_evals.evaluators import EqualsExpected, Contains, IsInstance, LLMJudge

# Custom reusable evaluators from our library
from findingmodel.tools.evaluators import PerformanceEvaluator

# Simple inline evaluators (defined in same file)
@dataclass
class IDPreservationEvaluator(Evaluator[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        # Evaluator implementation
        ...
```

### For Unit Tests

```python
# Test custom evaluators - imports from src/, NOT evals/
from findingmodel.tools.evaluators import PerformanceEvaluator

# Unit tests for evaluators live in test/tools/
# test/tools/test_evaluators.py
```

**CRITICAL:** Unit tests NEVER import from `evals/` package - only from `src/findingmodel/`

## Creating New Custom Evaluators

### Decision Framework

**Keep Inline if:**
- Used by only one eval suite
- Simple logic (< 20 lines)
- Agent-specific or case-specific
- Example: Checking if specific keywords appear in output

**Extract to src/ if:**
- Used by 2+ eval suites
- Complex logic worth unit testing
- Non-trivial calculation or state
- Example: PerformanceEvaluator with time source precedence logic

**Use Pydantic Evals Built-in if:**
- Checking equality: `EqualsExpected`
- Checking membership: `Contains`
- Type validation: `IsInstance`
- LLM-based assessment: `LLMJudge`

### Pattern for Reusable Evaluators

```python
from dataclasses import dataclass
from typing import TypeVar

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")

@dataclass
class MyEvaluator(Evaluator[InputT, OutputT, MetadataT]):
    """Brief description of what this evaluator does.
    
    USE CASES:
        - When to use this evaluator
        - Example scenarios
        
    REQUIREMENTS:
        - What the output object must have
        - Any assumptions made
        
    SCORING LOGIC:
        - Returns X.X if condition Y
        - Returns Z.Z if condition W
        - N/A cases and how handled
        
    Example usage:
        >>> from findingmodel.tools.evaluators import MyEvaluator
        >>> evaluator = MyEvaluator(param=value)
        >>> # Use in dataset evaluators list
        
    Attributes:
        param: Description of parameter
    """
    
    param: type = default
    
    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT, MetadataT]) -> float:
        """Evaluate the output.
        
        Args:
            ctx: Evaluation context
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Implementation
        ...
```

## Unit Testing Evaluators

### Location

`test/tools/test_evaluators.py` - Unit tests for `src/findingmodel/tools/evaluators.py`

### Required Guard

```python
from pydantic_ai import models

# Prevent accidental API calls during unit tests
models.ALLOW_MODEL_REQUESTS = False
```

### Test Helper Pattern

```python
def _create_ctx(
    output: Any,
    duration: float | None = None,
    metadata: Any = None,
) -> EvaluatorContext:
    """Helper to create minimal EvaluatorContext for testing."""
    return EvaluatorContext(
        name="test",
        inputs={},
        metadata=metadata,
        expected_output=None,
        output=output,
        duration=duration,
        _span_tree=None,  # type: ignore[arg-type]
        attributes={},
        metrics={},
    )
```

### Test Coverage

For each evaluator, test:
- Normal cases (pass and fail)
- Boundary conditions
- Edge cases and N/A scenarios
- Various input values
- Configuration options

## Logfire Configuration

Each eval script configures Logfire directly in `__main__` via `ensure_instrumented()`:

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

**Why `ensure_instrumented()`:**
- Idempotent (safe to call multiple times)
- Prevents instrumentation at import time
- Ensures Logfire only active when running evals, not unit tests

## Historical Context

### Before Refactoring (Oct 2025)

**Problems:**
- `evals/base.py` contained 5 evaluators NEVER used by eval scripts (only by unit tests)
- PerformanceEvaluator duplicated 5x across eval scripts (~190 lines)
- Unit tests importing from `evals/` package (architectural violation)
- ExactMatchEvaluator, ContainsEvaluator duplicated Pydantic Evals built-ins
- StructuralValidityEvaluator was redundant (Pydantic AI guarantees valid models)

### After Refactoring (Oct 2025)

**Results:**
- Deleted `evals/base.py` (~350 lines of unused code)
- Deleted `test/test_base_evaluators.py` (~600 lines testing unused code)
- Created `src/findingmodel/tools/evaluators.py` with ONLY PerformanceEvaluator
- Updated 5 eval scripts to use centralized PerformanceEvaluator
- Created `test/tools/test_evaluators.py` with proper unit tests
- Net result: ~1,010 lines deleted, ~130 lines added

**Key Insight:** Research in Phase 0 revealed that most evaluators SHOULD stay inline per Pydantic Evals philosophy. Only truly reusable complex evaluators belong in src/.

## Related Documentation

- `.serena/memories/agent_evaluation_best_practices_2025.md` - Eval patterns and best practices
- `.serena/memories/pydantic_ai_testing_best_practices.md` - AI agent testing guidance
- `evals/CLAUDE.md` - Quick reference for eval development
- `evals/evals_guide.md` - Comprehensive tutorial for humans
- `tasks/refactor-evaluator-architecture.md` - Complete refactoring plan and history
- `tasks/phase-0-research-findings.md` - Research on Pydantic Evals patterns

## External References

- Pydantic Evals: https://ai.pydantic.dev/evals/
- Built-in Evaluators: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals/evaluators/built-in.md
- Evaluators Overview: https://github.com/pydantic/pydantic-ai/blob/main/docs/evals/evaluators/overview.md
