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
   - Dataset.evaluate() with component-specific evaluators
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
task evals:metadata
task evals:metadata:smoke
task evals:metadata:full
PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_assignment --fixture-sample 2 --seed 20260515 --details-output /tmp/metadata-assignment-bounded-details.csv
PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_patient_applicability_decision
PYTHONPATH=packages/findingmodel-ai uv run python -m evals.metadata_etiology_tempo_decision --case-set expanded --details-output /tmp/etiology-tempo-details.csv

# Run with a specific model for the describe_finding agent
AGENT_MODEL_OVERRIDES__describe_finding=anthropic:claude-sonnet-4-6 task evals:finding_description

# Or directly:
AGENT_MODEL_OVERRIDES__describe_finding=anthropic:claude-sonnet-4-6 python -m evals.finding_description
```

Override specific agents via `AGENT_MODEL_OVERRIDES__<tag>=provider:model`. Compare overall scores and individual evaluator results to see which provider performs better for your use case.

The metadata eval task includes the end-to-end assignment suite plus component suites for
ontology, anatomy, entity type, patient applicability, subspecialty, modality, and etiology/tempo.
Submaximal scores are expected when the model makes lower-cost clinical mistakes; execution and
schema failures are reported as gates instead.

The etiology/time-course component eval supports `--case-set pilot`, `gold`, `reviewed`,
`expanded`, and `all`. Use `--details-output` to write per-case expected/actual values and miss
labels for prompt-tuning review.

The end-to-end metadata assignment eval also supports `--details-output`; bounded runs should write
that CSV so gate failures and lower-scoring fields are reviewable case-by-case.

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
3. **Overall score or gate status**: Report shape is suite-specific; metadata evals separate
   pass/fail gates from quality scores.

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

Metadata enrichment evals separate gates from quality. Gates are pass/fail checks for whether a
case produced an interpretable result. The headline score is metadata quality only; ordinary
submaximal metadata judgment scores are review signals, not proof that the prompt should memorize
the case.

For metadata assignment, `entity_type` is the only required metadata field. Optional blanks receive
conservative partial credit when gold has a value, and existing index/anatomic code extras carried
forward from the input are not penalized as new hallucinated additions. Use `--fixture`,
`--fixture-sample`, `--seed`, `--scenario`, `--case`, and `--limit` on
`python -m evals.metadata_assignment` for filtered runs.

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

**Note:** `ensure_instrumented()` currently configures Logfire with `send_to_logfire="if-token-present"` and `console=False`. If you need different behavior, update `packages/findingmodel-ai/evals/__init__.py`.

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
