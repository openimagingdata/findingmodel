# Evaluation Guide for FindingModel AI Agents

## Overview

This guide describes best practices for evaluating AI agents in the findingmodel project. It covers the Pydantic AI Evals framework, evaluation patterns, and how to create robust, reusable evaluation suites.

## Table of Contents

1. [Why Evaluate AI Agents](#why-evaluate-ai-agents)
2. [Evaluation Framework](#evaluation-framework)
3. [Pydantic Evals Architecture](#pydantic-evals-architecture)
4. [Creating Evaluation Suites](#creating-evaluation-suites)
5. [Best Practices](#best-practices)
6. [Reusable Patterns](#reusable-patterns)
7. [Running Evaluations](#running-evaluations)
8. [Interpreting Results](#interpreting-results)

## Why Evaluate AI Agents

AI agents are non-deterministic systems that require systematic testing beyond traditional unit tests. Evaluation suites help:

- **Ensure reliability**: Verify agents produce correct outputs across diverse inputs
- **Catch regressions**: Detect when changes degrade agent performance
- **Guide development**: Identify edge cases and failure modes
- **Measure quality**: Track improvements over time with metrics
- **Enable iteration**: Test prompt changes, model upgrades, and architectural modifications

### Two-Level Evaluation Strategy

Modern agent evaluation requires testing at multiple levels:

1. **Component-level evaluation**: Test individual agents in isolation (e.g., `model_editor`, `anatomic_location_search`)
2. **End-to-end evaluation**: Test complete workflows involving multiple agents and tools

This guide focuses primarily on component-level evaluation.

## Evaluation Framework

### Pydantic AI Evals

We use [Pydantic AI Evals](https://ai.pydantic.dev/evals/), a framework designed for evaluating non-deterministic functions including LLM agents. Key features:

- **Type-safe**: Leverages Pydantic models for inputs, outputs, and expected results
- **Flexible**: Works with any agent framework (not just Pydantic AI)
- **Observable**: Integrates with OpenTelemetry and Logfire for trace analysis
- **Pythonic**: Uses familiar Python patterns rather than DSLs

### Core Concepts

**Case**: A single test scenario with inputs and expected outputs
```python
case = Case(
    name='add_severity_attribute',
    inputs=ModelEditorInput(model_json=..., command="Add severity attribute"),
    expected_output=ModelEditorExpectedOutput(should_succeed=True, ...)
)
```

**Dataset**: Collection of cases with shared evaluators
```python
dataset = Dataset(
    cases=[case1, case2, case3],
    evaluators=[IDPreservationEvaluator(), AttributeAdditionEvaluator()]
)
```

**Evaluator**: Assesses one aspect of agent output, returning a score (0.0-1.0)
```python
class IDPreservationEvaluator(Evaluator[InputT, OutputT]):
    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        # Return 1.0 if IDs preserved, 0.0 otherwise
        return 1.0 if ctx.output.model.oifm_id == original_id else 0.0
```

**Report**: Results from running a dataset, with aggregate metrics and per-case details

## Pydantic Evals Architecture

### The Standard Pattern

```python
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

# 1. Define input/output models
class MyInput(BaseModel):
    query: str
    context: dict

class MyExpectedOutput(BaseModel):
    should_succeed: bool
    expected_keywords: list[str]

class MyActualOutput(BaseModel):
    result: str
    error: str | None = None

# 2. Create evaluator(s)
class KeywordMatchEvaluator(Evaluator[MyInput, MyActualOutput]):
    def evaluate(self, ctx: EvaluatorContext[MyInput, MyActualOutput]) -> float:
        keywords = ctx.expected_output.expected_keywords
        text = ctx.output.result.lower()
        matches = sum(1 for kw in keywords if kw in text)
        return matches / len(keywords) if keywords else 1.0

# 3. Define cases
cases = [
    Case(
        name="basic_query",
        inputs=MyInput(query="test", context={}),
        expected_output=MyExpectedOutput(should_succeed=True, expected_keywords=["result"])
    )
]

# 4. Create dataset
dataset = Dataset(cases=cases, evaluators=[KeywordMatchEvaluator()])

# 5. Define task function (wraps agent execution)
async def my_task(input_data: MyInput) -> MyActualOutput:
    try:
        result = await my_agent.run(input_data.query, context=input_data.context)
        return MyActualOutput(result=result.output)
    except Exception as e:
        return MyActualOutput(result="", error=str(e))

# 6. Run evaluation
report = await dataset.evaluate_async(my_task)
report.print()
```

### Key Design Principles

1. **Use Evaluator classes, not functions**: Enables composition, reuse, and proper scoring
2. **Return continuous scores (0.0-1.0)**: Allows nuanced evaluation and partial credit
3. **Multiple focused evaluators**: Each evaluator tests one aspect (separation of concerns)
4. **Type safety throughout**: Pydantic models for all data structures
5. **Handle errors gracefully**: Capture exceptions in output models, evaluate error handling

## Creating Evaluation Suites

### Step 1: Define Your Data Models

```python
from pydantic import BaseModel

class AgentInput(BaseModel):
    """Input to the agent being tested."""
    # Fields specific to your agent
    query: str
    options: dict = {}

class AgentExpectedOutput(BaseModel):
    """What we expect from the agent."""
    should_succeed: bool
    expected_attributes: list[str] = []
    expected_keywords: list[str] = []
    minimum_quality_score: float = 0.8

class AgentActualOutput(BaseModel):
    """Actual output from the agent."""
    # Fields from agent's actual response
    result: YourAgentResult
    error: str | None = None
```

### Step 2: Create Focused Evaluators

Each evaluator should test ONE aspect of agent behavior:

```python
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

class SuccessEvaluator(Evaluator[AgentInput, AgentActualOutput]):
    """Verify agent succeeded when expected (or failed when expected)."""

    def evaluate(self, ctx: EvaluatorContext[AgentInput, AgentActualOutput]) -> float:
        expected = ctx.expected_output.should_succeed
        actual_success = ctx.output.error is None
        return 1.0 if actual_success == expected else 0.0

class AttributePresenceEvaluator(Evaluator[AgentInput, AgentActualOutput]):
    """Check that expected attributes are present in result."""

    def evaluate(self, ctx: EvaluatorContext[AgentInput, AgentActualOutput]) -> float:
        expected_attrs = ctx.expected_output.expected_attributes
        if not expected_attrs:
            return 1.0  # N/A - no attributes expected

        actual_attrs = {attr.name for attr in ctx.output.result.attributes}
        matches = sum(1 for attr in expected_attrs if attr in actual_attrs)
        return matches / len(expected_attrs)  # Partial credit

class KeywordEvaluator(Evaluator[AgentInput, AgentActualOutput]):
    """Check for expected keywords in text output."""

    def evaluate(self, ctx: EvaluatorContext[AgentInput, AgentActualOutput]) -> float:
        keywords = ctx.expected_output.expected_keywords
        if not keywords:
            return 1.0

        text = str(ctx.output.result).lower()
        matches = sum(1 for kw in keywords if kw.lower() in text)
        return matches / len(keywords)
```

### Step 3: Create Test Cases

Organize cases into categories:

```python
def create_successful_cases() -> list[Case[AgentInput, AgentExpectedOutput]]:
    """Cases where agent should succeed."""
    cases = []

    cases.append(
        Case(
            name="simple_query",
            inputs=AgentInput(query="Add severity attribute"),
            expected_output=AgentExpectedOutput(
                should_succeed=True,
                expected_attributes=["severity"],
                expected_keywords=["added", "severity"]
            )
        )
    )

    # Add more successful cases...
    return cases

def create_rejection_cases() -> list[Case[AgentInput, AgentExpectedOutput]]:
    """Cases where agent should reject the request."""
    cases = []

    cases.append(
        Case(
            name="reject_delete",
            inputs=AgentInput(query="Delete the presence attribute"),
            expected_output=AgentExpectedOutput(
                should_succeed=False,
                expected_keywords=["reject", "delete", "not allowed"]
            )
        )
    )

    return cases

def create_edge_cases() -> list[Case[AgentInput, AgentExpectedOutput]]:
    """Edge cases and boundary conditions."""
    cases = []

    cases.append(
        Case(
            name="empty_query",
            inputs=AgentInput(query=""),
            expected_output=AgentExpectedOutput(should_succeed=False)
        )
    )

    # Very long inputs, special characters, etc.
    return cases
```

### Step 4: Build Dataset and Task Function

```python
from pydantic_evals import Dataset

# Combine all cases
all_cases = (
    create_successful_cases() +
    create_rejection_cases() +
    create_edge_cases()
)

# Create evaluators
evaluators = [
    SuccessEvaluator(),
    AttributePresenceEvaluator(),
    KeywordEvaluator(),
]

# Build dataset
dataset = Dataset(cases=all_cases, evaluators=evaluators)

# Define task function (executes the agent)
async def run_agent_task(input_data: AgentInput) -> AgentActualOutput:
    """Execute the agent being evaluated."""
    try:
        result = await my_agent.run(input_data.query, **input_data.options)
        return AgentActualOutput(result=result)
    except Exception as e:
        # Capture errors for evaluation
        return AgentActualOutput(result=None, error=str(e))
```

### Step 5: Create Pytest Test

```python
import pytest

@pytest.mark.callout  # Requires API access
@pytest.mark.asyncio
async def test_agent_evals():
    """Run full evaluation suite."""
    report = await dataset.evaluate_async(run_agent_task)

    # Print detailed report
    report.print(include_input=True, include_output=True)

    # Assert minimum quality threshold
    assert report.overall_score() >= 0.9, f"Only {report.overall_score():.1%} passing"

    # Can also check specific evaluators
    for evaluator_name, score in report.evaluator_scores().items():
        print(f"{evaluator_name}: {score:.2%}")
```

### Step 6: Add Mock Test for Quick Validation

```python
from pydantic_ai.models.test import TestModel

@pytest.mark.asyncio
async def test_single_case_mock():
    """Quick test without API calls using mock."""
    # Create expected output
    mock_result = MyAgentResult(attributes=[...])

    # Use TestModel to mock agent
    with my_agent.override(model=TestModel(custom_output_args=mock_result)):
        result = await run_agent_task(AgentInput(query="test"))

        assert result.error is None
        assert len(result.result.attributes) > 0
```

## Best Practices

### 1. Test at Multiple Levels

**Component Tests** (what we're building now):
- Test individual agents in isolation
- Use mocks for dependencies
- Fast, deterministic, focused

**Integration Tests** (future):
- Test multiple agents working together
- Test real backend integrations (MongoDB, BioOntology)
- Slower, more comprehensive

### 2. Design Test Cases by Category

Organize cases into clear categories:
- **Success cases**: Agent handles request correctly
- **Rejection cases**: Agent properly rejects unsafe/invalid requests
- **Edge cases**: Boundary conditions, empty inputs, very large inputs
- **Error handling**: Network failures, malformed data, timeouts
- **Regression cases**: Specific bugs that were previously fixed

### 3. Use Real Test Data

Prefer real data from your project over synthetic data:
- Tests realistic scenarios
- Catches edge cases from production
- Maintains consistency with actual usage

For findingmodel:
```python
def load_fm_json(filename: str) -> str:
    """Load real finding model from test data."""
    test_data_dir = Path(__file__).parent.parent / "data" / "defs"
    return (test_data_dir / filename).read_text()
```

### 4. Handle Non-Determinism

LLM outputs are non-deterministic. Design evaluators accordingly:

**❌ Don't use exact string matching:**
```python
return 1.0 if output == "Expected exact text" else 0.0
```

**✅ Use keyword/concept matching:**
```python
keywords = ["severity", "added", "attribute"]
matches = sum(1 for kw in keywords if kw in output.lower())
return matches / len(keywords)
```

**✅ Use structured outputs:**
```python
# Agent returns Pydantic model, not free text
return 1.0 if output.severity in ["mild", "moderate", "severe"] else 0.0
```

**✅ Use LLM-as-judge (for quality assessment):**
```python
judge_prompt = f"Rate this description for clinical accuracy (0-10): {output}"
score = await judge_agent.run(judge_prompt)
return score / 10.0
```

### 5. Prefer Multiple Small Evaluators

**❌ One big evaluator doing everything:**
```python
class BigEvaluator(Evaluator):
    def evaluate(self, ctx):
        score = 0
        if check_ids(): score += 0.25
        if check_attrs(): score += 0.25
        if check_keywords(): score += 0.25
        if check_structure(): score += 0.25
        return score
```

**✅ Multiple focused evaluators:**
```python
evaluators = [
    IDPreservationEvaluator(),     # 0.0 or 1.0
    AttributeAdditionEvaluator(),  # 0.0 to 1.0 (partial credit)
    KeywordMatchEvaluator(),       # 0.0 to 1.0 (partial credit)
    StructuralValidityEvaluator(), # 0.0 or 1.0
]
```

Benefits:
- Clear separation of concerns
- Independent scores for each aspect
- Easier to debug failures
- Reusable across different agents

### 6. Use @pytest.mark.callout for API Tests

Follow project convention:

```python
# Mock test - runs by default (fast, no API)
@pytest.mark.asyncio
async def test_quick_validation():
    with agent.override(model=TestModel(...)):
        result = await agent.run("test")
        assert result.is_valid()

# Full evaluation - requires API
@pytest.mark.callout
@pytest.mark.asyncio
async def test_full_evals():
    report = await dataset.evaluate_async(task_function)
    assert report.overall_score() >= 0.9
```

### 7. Add Observability

Enable Logfire for deep debugging:

```python
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    environment='test',
    service_name='findingmodel-evals',
)

# All evaluation traces automatically sent to Logfire
# View execution paths, token usage, durations
```

## Reusable Patterns

### Base Evaluator Library

Create `test/evals/base.py` with reusable evaluators:

```python
"""Reusable evaluators for findingmodel agents."""

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic import BaseModel
from typing import TypeVar, Generic, Callable

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class ExactMatchEvaluator(Evaluator[InputT, str], Generic[InputT]):
    """Check if output exactly matches expected output."""

    def evaluate(self, ctx: EvaluatorContext[InputT, str]) -> float:
        return 1.0 if ctx.output == ctx.expected_output else 0.0


class ContainsEvaluator(Evaluator[InputT, str], Generic[InputT]):
    """Check if output contains expected substring."""

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def evaluate(self, ctx: EvaluatorContext[InputT, str]) -> float:
        output = ctx.output if self.case_sensitive else ctx.output.lower()
        expected = ctx.expected_output if self.case_sensitive else ctx.expected_output.lower()
        return 1.0 if expected in output else 0.0


class KeywordMatchEvaluator(Evaluator[InputT, OutputT], Generic[InputT, OutputT]):
    """Check for keywords in extracted text."""

    def __init__(
        self,
        keyword_field: str,  # Field in expected_output with keywords list
        text_extractor: Callable[[OutputT], str],  # Extract text from output
        partial_credit: bool = True
    ):
        self.keyword_field = keyword_field
        self.text_extractor = text_extractor
        self.partial_credit = partial_credit

    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        keywords = getattr(ctx.expected_output, self.keyword_field, [])
        if not keywords:
            return 1.0

        text = self.text_extractor(ctx.output).lower()
        matches = sum(1 for kw in keywords if kw.lower() in text)

        if self.partial_credit:
            return matches / len(keywords)
        else:
            return 1.0 if matches == len(keywords) else 0.0


class StructuralValidityEvaluator(Evaluator[InputT, BaseModel], Generic[InputT]):
    """Check if output has required fields."""

    def __init__(self, required_fields: list[str] | None = None):
        self.required_fields = required_fields or []

    def evaluate(self, ctx: EvaluatorContext[InputT, BaseModel]) -> float:
        if not isinstance(ctx.output, BaseModel):
            return 0.0

        if not self.required_fields:
            return 1.0

        model_dict = ctx.output.model_dump()
        present = sum(1 for field in self.required_fields if field in model_dict)
        return present / len(self.required_fields)


class ErrorHandlingEvaluator(Evaluator[InputT, OutputT], Generic[InputT, OutputT]):
    """Check that errors are handled as expected."""

    def __init__(self, error_field: str = "error"):
        self.error_field = error_field

    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        should_succeed = getattr(ctx.expected_output, "should_succeed", True)
        has_error = getattr(ctx.output, self.error_field, None) is not None

        if should_succeed and not has_error:
            return 1.0  # Should succeed and did
        elif not should_succeed and has_error:
            return 1.0  # Should fail and did
        else:
            return 0.0  # Mismatch
```

### Agent Evaluation Suite Base Class

Create abstract base for consistency:

```python
"""Base class for agent evaluation suites."""

from abc import ABC, abstractmethod
from pydantic_evals import Case, Dataset
from typing import Generic, TypeVar

InputT = TypeVar('InputT')
ExpectedT = TypeVar('ExpectedT')
ActualT = TypeVar('ActualT')

class AgentEvaluationSuite(ABC, Generic[InputT, ExpectedT, ActualT]):
    """Base class for creating agent evaluation suites.

    Subclass this to create evaluation suites for any agent in tools/.
    """

    @abstractmethod
    def create_successful_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Create cases where agent should succeed."""
        pass

    @abstractmethod
    def create_failure_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Create cases where agent should fail gracefully."""
        pass

    @abstractmethod
    def create_edge_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Create edge cases and boundary conditions."""
        pass

    @abstractmethod
    async def execute_agent(self, input_data: InputT) -> ActualT:
        """Execute the agent being tested."""
        pass

    def get_all_cases(self) -> list[Case[InputT, ExpectedT]]:
        """Get all cases combined."""
        return (
            self.create_successful_cases() +
            self.create_failure_cases() +
            self.create_edge_cases()
        )

    def build_dataset(self, evaluators: list) -> Dataset:
        """Build dataset with all cases and provided evaluators."""
        return Dataset(cases=self.get_all_cases(), evaluators=evaluators)
```

## Running Evaluations

### Quick Mock Test (No API)

```bash
# Run single mock test
pytest test/evals/test_model_editor_evals.py::test_single_case_mock -v
```

### Full Evaluation Suite (Requires API)

```bash
# Run all eval tests marked with @pytest.mark.callout
pytest test/evals/test_model_editor_evals.py::test_full_evals -v -s

# Run all eval tests across all agents
task test-full test/evals/

# View detailed output
pytest test/evals/ -m callout -v -s
```

### Standalone Execution

```bash
# Many eval files can be run directly
cd test/evals
python test_model_editor_evals.py
```

### CI/CD Integration

Evals marked with `@pytest.mark.callout` are:
- Skipped in regular `task test` runs
- Run in `task test-full` with API credentials
- Should be part of release validation

## Interpreting Results

### Reading Reports

```
                           Evaluation Summary: my_agent
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID ┃ Inputs ┃ Outputs   ┃ Scores     ┃ Duration ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ case_1  │ ...    │ ...       │ Eval1: 1.0 │    250ms │
│         │        │           │ Eval2: 0.8 │          │
├─────────┼────────┼───────────┼────────────┼──────────┤
│ case_2  │ ...    │ ...       │ Eval1: 0.0 │    180ms │
│         │        │           │ Eval2: 1.0 │          │
├─────────┼────────┼───────────┼────────────┼──────────┤
│ Avg     │        │           │ Eval1: 0.5 │    215ms │
│         │        │           │ Eval2: 0.9 │          │
└─────────┴────────┴───────────┴────────────┴──────────┘
```

**Key metrics:**
- **Per-case scores**: How each case performed on each evaluator
- **Average scores**: Overall performance per evaluator
- **Duration**: Execution time per case
- **Overall score**: `report.overall_score()` - average across all evaluators

### Investigating Failures

1. **Look at per-case details**: Which specific cases failed?
2. **Check evaluator scores**: Which aspect failed (IDs, keywords, structure)?
3. **Examine outputs**: Print inputs/outputs for failed cases
4. **Use Logfire**: View full trace to see agent's reasoning

```python
# Print detailed failure info
report = await dataset.evaluate_async(task)
for case_result in report.case_results:
    if case_result.score < 1.0:
        print(f"Failed: {case_result.case_name}")
        print(f"  Input: {case_result.input}")
        print(f"  Output: {case_result.output}")
        print(f"  Scores: {case_result.evaluator_scores}")
```

### Setting Quality Thresholds

```python
# Require 90% overall
assert report.overall_score() >= 0.9

# Require specific evaluators to pass
for name, score in report.evaluator_scores().items():
    if name == "IDPreservationEvaluator":
        assert score == 1.0, "All IDs must be preserved"
    elif name == "AttributeAdditionEvaluator":
        assert score >= 0.8, "Must add 80%+ of expected attributes"
```

## Next Steps

1. **Create base evaluator library** (`test/evals/base.py`)
2. **Refactor existing model_editor evals** to use Pydantic Evals standard pattern
3. **Create eval suites for other agents**:
   - `anatomic_location_search`
   - `ontology_concept_match`
   - `finding_description`
   - `similar_finding_models`
4. **Add Logfire integration** for observability
5. **Document agent-specific evaluation criteria** in each eval file
6. **Set up regression tracking** to compare eval runs over time

## Resources

- [Pydantic AI Evals Documentation](https://ai.pydantic.dev/evals/)
- [Pydantic AI Testing Best Practices](https://ai.pydantic.dev/testing/)
- [Logfire Evals Integration](https://ai.pydantic.dev/evals/#logfire-integration)
- Project memory: `pydantic_ai_testing_best_practices`
- Project memory: `test_suite_improvements_2025`
