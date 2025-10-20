# Eval Suites for FindingModel AI Agents

Quick-start guide for running and understanding evaluation suites.

## What are Evals?

FindingModel uses a three-tier testing structure:

1. **Unit Tests** (`test/test_*.py`) - Verify logic correctness with mocked dependencies
   - Fast, no API calls
   - Run with `task test`

2. **Integration Tests** (`test/test_*.py` with `@pytest.mark.callout`) - Verify wiring with real APIs
   - Real API calls, specific scenarios
   - Run with `task test-full`

3. **Evals** (`evals/*.py`) - Assess behavioral quality comprehensively
   - Dataset.evaluate() with focused evaluators
   - Run with `task evals` or `task evals:model_editor`
   - Manual execution, not part of CI (initially)

**Key Distinction**: Tests verify correctness (pass/fail), evals assess quality (0.0-1.0 scores with partial credit).

## How to Run Evals

### Run all eval suites

```bash
task evals
```

### Run specific eval suite

```bash
task evals:model_editor
# Or directly:
python -m evals.model_editor
```

### From Python

```python
from evals.model_editor import run_model_editor_evals

report = await run_model_editor_evals()

# Calculate overall score (average of all evaluator scores across all cases)
all_scores = [score.value for case in report.cases for score in case.scores.values()]
overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
print(f"Overall score: {overall_score:.2f}")
```

## How to Read Eval Output

Eval reports show:

1. **Per-case results**: Each test case with evaluator scores
2. **Per-evaluator metrics**: How each evaluator performed across all cases
3. **Overall score**: Aggregate score (0.0-1.0) across all evaluators and cases

Example output:

```
================================================================================
MODEL EDITOR EVALUATION RESULTS
================================================================================

Case: add_severity_attribute
  IDPreservationEvaluator: 1.00
  AttributeAdditionEvaluator: 1.00
  ChangeTrackingEvaluator: 1.00
  Overall: 1.00

Case: reject_rename_attribute
  IDPreservationEvaluator: 1.00
  RejectionAccuracyEvaluator: 0.67  # Keywords found: 2/3
  ContentPreservationEvaluator: 1.00
  Overall: 0.89

OVERALL SCORE: 0.95
```

### Understanding Scores

- **1.00**: Perfect (all criteria met)
- **0.67**: Partial credit (e.g., 2/3 keywords found)
- **0.00**: Failed (criteria not met)

## When to Run Evals

Run evals when:

- **Before major changes**: Establish baseline
- **After agent modifications**: Detect regressions
- **Testing new prompts**: Compare performance
- **Release validation**: Ensure quality threshold met

## Current Eval Suites

- **model_editor** - AI-powered model editing (natural language and markdown)
  - 12 cases (successful edits, rejections, markdown edits)
  - 5 evaluators (ID preservation, attribute addition, change tracking, rejection accuracy, content preservation)

## Learn More

- **Writing evals**: See `evals/evals_guide.md` for comprehensive guide
- **AI agent reference**: See `evals/CLAUDE.md` for development conventions
- **Best practices**: See `.serena/memories/agent_evaluation_best_practices_2025.md`

## Quick Links

- [Pydantic AI Evals Documentation](https://ai.pydantic.dev/evals/)
- [Model Editor Eval Suite](model_editor.py)
- [Base Evaluators Library](base.py)
- [Evaluation Utilities](utils.py)

