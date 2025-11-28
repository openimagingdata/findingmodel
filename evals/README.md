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
task evals:similar_models
task evals:ontology_match
task evals:markdown_in

# Or directly:
python -m evals.model_editor
python -m evals.ontology_match
python -m evals.markdown_in
```

### Compare AI Providers (OpenAI vs Anthropic)

To compare performance between providers on any eval suite:

```bash
# Run with default model (OpenAI)
task evals:finding_description

# Run with Anthropic
DEFAULT_MODEL=anthropic:claude-sonnet-4-5 task evals:finding_description

# Or directly:
DEFAULT_MODEL=anthropic:claude-sonnet-4-5 python -m evals.finding_description
```

The eval suite will use whichever model you specify via `DEFAULT_MODEL` (Pydantic AI format: `provider:model`). Compare the overall scores and individual evaluator results to see which provider performs better for your use case.

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

## Observability with Logfire

Logfire observability is **configured automatically** - no setup needed per eval suite.

### How It Works

- **Automatic configuration** in `evals/__init__.py`
- **Zero Logfire code** required in individual eval modules
- **Automatic instrumentation** via Pydantic Evals + Pydantic AI

When you run an eval suite, Logfire automatically captures:
- Evaluation root span and per-case execution spans
- Agent run spans with prompts/completions
- Model call spans and tool execution spans
- Evaluation scores and results

### Setup (Optional - Cloud Tracing)

Logfire works in local-only mode by default. For cloud tracing:

```bash
# 1. Create account at https://logfire.pydantic.dev/
# 2. Get write token from dashboard
# 3. Add to .env file:
echo "LOGFIRE_TOKEN=your_token_here" >> .env

# 4. Run evaluations - traces automatically appear in Logfire UI
python -m evals.model_editor
```

### Environment Variables

- `LOGFIRE_TOKEN` - Write token (optional, enables cloud tracing)
- `DISABLE_SEND_TO_LOGFIRE` - Force local-only mode (default: false)
- `LOGFIRE_VERBOSE` - Enable console logging (default: false)

**Note:** By default, Logfire console output is disabled to keep eval runs clean. Traces are still sent to the cloud when a token is present.

### For New Eval Suites

**No Logfire code needed.** Observability happens automatically via package-level configuration.

See `docs/logfire_observability_guide.md` for comprehensive documentation.

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

- **similar_models** - Finding similar models via DuckDB vector search and AI analysis
  - 23 cases (exact duplicates, semantic similarity, edge cases, performance)
  - 6 evaluators (duplicate detection, ranking quality, precision@K, semantic similarity, exclusion, performance)

- **ontology_match** - Matching findings to medical ontology concepts
  - 22 cases (success, synonyms, edge cases, ranking, rejection, performance)
  - 6 evaluators (concept match accuracy, ranking quality, backend consistency, synonym handling, error handling, performance)
  - Requires BioOntology API key

- **markdown_in** - Parsing markdown text into finding model structure
  - 18 cases (success, edge cases, error handling, complex structures)
  - 6 evaluators (structural validity, attribute preservation, type correctness, error message quality, round-trip preservation, performance)
  - Requires OpenAI API key

## Learn More

- **Writing evals**: See `evals/evals_guide.md` for comprehensive guide
- **AI agent reference**: See `evals/CLAUDE.md` for development conventions
- **Best practices**: See `.serena/memories/agent_evaluation_best_practices_2025.md`

## Quick Links

- [Pydantic AI Evals Documentation](https://ai.pydantic.dev/evals/)
- [Model Editor Eval Suite](model_editor.py)
- [Base Evaluators Library](base.py)
- [Evaluation Utilities](utils.py)

