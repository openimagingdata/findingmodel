# Logfire Observability for FindingModel - 2025

## Overview

Pydantic Logfire is used for observability in the FindingModel project, particularly in AI agent evaluation suites. It provides OpenTelemetry-based tracing, metrics, and logs with Python-centric insights.

**Full Documentation:** `docs/logfire_observability_guide.md`

## Key Design Principles - FindingModel Implementation

### 1. Settings-Based Configuration (NOT os.getenv)

**IMPORTANT:** FindingModel uses its centralized settings system, NOT direct os.getenv calls:

```python
import logfire
from logfire import ConsoleOptions
from findingmodel.config import settings

# Configure Logfire using project settings
logfire.configure(
    token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
    send_to_logfire=False if settings.disable_send_to_logfire else 'if-token-present',
    console=ConsoleOptions(
        colors="auto",
        min_log_level="debug" if settings.logfire_verbose else "info",
    ),
)

# Instrument Pydantic AI agents (PRIMARY instrumentation)
logfire.instrument_pydantic_ai()
```

**Benefits:**
- Consistent with project's API key management patterns (SecretStr)
- Automatic .env loading via pydantic-settings
- Type safety and validation
- No hard-coded service names or environment metadata

### 2. Environment Variables (Managed by Settings)

Configuration via .env file (automatically loaded by FindingModelConfig):

- `LOGFIRE_TOKEN` - Write token from logfire.pydantic.dev (optional)
- `DISABLE_SEND_TO_LOGFIRE` - Set to `true` to force local-only mode (default: false)
- `LOGFIRE_VERBOSE` - Set to `true` for verbose console logging (default: false)

### 3. ConsoleOptions for Console Configuration

Use ConsoleOptions object instead of direct console_* parameters:

```python
from logfire import ConsoleOptions

console=ConsoleOptions(
    colors="auto",
    min_log_level="debug" if settings.logfire_verbose else "info",
)
```

This matches the actual Logfire 1.0+ API.

## Structured Logging Patterns

### Use Named Parameters

```python
# ✅ Good: Structured
logfire.info('Evaluated {count} cases in {duration}s', count=10, duration=5.2)

# ❌ Bad: Unstructured
logfire.info(f'Evaluated 10 cases in 5.2s')
```

### Appropriate Span Granularity

```python
# ✅ Good: Meaningful operations
with logfire.span('eval_case {name}', name=case.name, should_succeed=True):
    result = await run_case(case)

# ❌ Too granular
with logfire.span('validate_input'):
    validate(input)  # Unless expensive
```

### Rich Context in Spans

```python
# ✅ Good: Rich context
with logfire.span(
    'eval_case {name}',
    name=case_name,
    case_name=case_name,
    edit_type='natural_language',
    should_succeed=True,
):
    result = await execute_case()
```

## Evaluation Suite Integration - Actual Implementation

### Module-Level Setup (evals/model_editor.py pattern)

```python
import logfire
from logfire import ConsoleOptions
from findingmodel.config import settings

# Configure Logfire
logfire.configure(
    token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
    send_to_logfire=False if settings.disable_send_to_logfire else "if-token-present",
    console=ConsoleOptions(
        colors="auto",
        min_log_level="debug" if settings.logfire_verbose else "info",
    ),
)

# Instrument Pydantic AI agents (PRIMARY instrumentation)
logfire.instrument_pydantic_ai()
```

### Suite-Level Instrumentation

```python
async def run_model_editor_evals():
    """Run evaluation suite with Logfire tracing."""
    with logfire.span(
        'model_editor_eval_suite',
        total_cases=len(all_cases),
        evaluator_count=len(evaluators),
    ):
        logfire.info(
            'Starting model_editor evaluation suite',
            cases_total=len(all_cases),
            evaluators=[e.__class__.__name__ for e in evaluators],
        )

        report = await model_editor_dataset.evaluate(run_model_editor_task)

        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        logfire.info(
            'Evaluation suite completed',
            overall_score=overall_score,
            cases_total=len(report.cases),
        )

        return report
```

### Case-Level Instrumentation with Metadata Lookup

Since pydantic-evals Dataset.evaluate() only passes InputT to task functions, not full Case objects, use a metadata lookup pattern:

```python
import hashlib

def _make_metadata_lookup_key(input_data: ModelEditorInput) -> str:
    """Create stable lookup key for case metadata mapping."""
    content = f"{input_data.model_json}|{input_data.command}|{input_data.edit_type}"
    return hashlib.sha256(content.encode()).hexdigest()

# Module-level mapping
_case_metadata_map: dict[str, tuple[str, bool]] = {
    _make_metadata_lookup_key(case.inputs): (case.name, case.metadata.should_succeed)
    for case in all_cases
}

async def run_model_editor_task(input_data: ModelEditorInput) -> ModelEditorActualOutput:
    """Execute case with Logfire tracing."""
    lookup_key = _make_metadata_lookup_key(input_data)
    case_name, should_succeed = _case_metadata_map.get(lookup_key, ("unknown", False))

    with logfire.span(
        'eval_case {name}',
        name=case_name,
        case_name=case_name,
        edit_type=input_data.edit_type,
        should_succeed=should_succeed,
    ):
        if settings.logfire_verbose:
            logfire.debug(
                'Starting evaluation case',
                case_name=case_name,
                edit_type=input_data.edit_type,
            )

        with logfire.span('model_editor_execution', operation=input_data.edit_type):
            # Execute model editor
            if input_data.edit_type == 'natural_language':
                result = await model_editor.edit_model_natural_language(model, input_data.command)
            else:
                result = await model_editor.edit_model_markdown(model, input_data.command)

        logfire.info(
            'Case completed',
            case_name=case_name,
            success=len(result.changes) > 0,
            changes_count=len(result.changes),
            rejections_count=len(result.rejections),
        )

        return ModelEditorActualOutput(
            model=result.model,
            changes=result.changes,
            rejections=result.rejections,
        )
```

## Best Practices

### DO

- ✅ Configure once at module level
- ✅ Use `send_to_logfire='if-token-present'` for graceful degradation
- ✅ Use FindingModelConfig settings (NOT os.getenv)
- ✅ Use ConsoleOptions for console configuration
- ✅ Use structured logging with named parameters
- ✅ Add rich context to spans (IDs, types, metadata)
- ✅ Call `logfire.instrument_pydantic_ai()` after configure
- ✅ Use hash-based lookup keys for metadata mapping
- ✅ Handle exceptions in spans (auto-captured)

### DON'T

- ❌ Call `configure()` multiple times
- ❌ Use os.getenv directly (use settings.logfire_token, etc.)
- ❌ Use direct console_* parameters (use ConsoleOptions)
- ❌ Hard-code service names or environment metadata
- ❌ Use f-strings instead of named parameters
- ❌ Log sensitive data (API keys, tokens, secrets)
- ❌ Create spans for trivial operations
- ❌ Require Logfire authentication for local development

## Setup for Developers

### Local Development (No Cloud)

```bash
# Works immediately - local-only logging
python -m evals.model_editor
```

### Cloud Tracing (Optional)

```bash
# 1. Create account at https://logfire.pydantic.dev/
# 2. Get write token from dashboard
# 3. Add to .env file:
echo "LOGFIRE_TOKEN=pfp_your_token_here" >> .env

# 4. Run evals - traces appear in Logfire UI
python -m evals.model_editor
```

### Disabling Logfire

```bash
# In .env
DISABLE_SEND_TO_LOGFIRE=true
```

## Current Status in Project

### Implemented (Phase 3 Complete)
- ✅ Comprehensive documentation: `docs/logfire_observability_guide.md`
- ✅ Phase 3 tasks completed: `tasks/refactor_model_editor_evals.md:287-729`
- ✅ Configuration fields added to FindingModelConfig (src/findingmodel/config.py:54-66)
- ✅ Logfire dependency added (pyproject.toml:56)
- ✅ Module-level configuration in evals/model_editor.py (lines 59-71)
- ✅ Suite-level instrumentation (lines 898-930)
- ✅ Case-level instrumentation with metadata lookup (lines 714-797)
- ✅ Documentation updated (evals/README.md:89-128)
- ✅ .env.sample updated (lines 17-21)

### Future Work
- 🔲 Apply Logfire pattern to other agent evaluation suites
- 🔲 Optional: Instrument base evaluators with opt-in tracing
- 🔲 Set up CI/CD integration for production monitoring

## Integration Points

### Pydantic AI Evals
Logfire works seamlessly with pydantic-evals framework. Primary instrumentation via `logfire.instrument_pydantic_ai()` automatically traces agent runs.

### Metadata Access Pattern
Since Dataset.evaluate() only passes InputT to task functions, use hash-based metadata lookup to access case name and expected outcomes for span attributes.

## Resources

- **Full guide:** `docs/logfire_observability_guide.md`
- **Task plan:** `tasks/refactor_model_editor_evals.md:287-729` (Phase 3)
- **Official docs:** https://logfire.pydantic.dev/docs/
- **Configuration:** https://logfire.pydantic.dev/docs/reference/configuration/
- **Testing:** https://logfire.pydantic.dev/docs/reference/advanced/testing/

## Key Takeaway

For FindingModel evaluation suites, use this configuration pattern:

```python
import logfire
from logfire import ConsoleOptions
from findingmodel.config import settings

logfire.configure(
    token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
    send_to_logfire=False if settings.disable_send_to_logfire else 'if-token-present',
    console=ConsoleOptions(
        colors="auto",
        min_log_level="debug" if settings.logfire_verbose else "info",
    ),
)

logfire.instrument_pydantic_ai()
```

This provides:
- Integration with project settings system
- Local observability without setup
- Cloud tracing when authenticated
- Graceful degradation
- Automatic Pydantic AI instrumentation
- Zero friction for developers