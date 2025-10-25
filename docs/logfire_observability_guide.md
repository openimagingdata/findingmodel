# Logfire Observability Guide for FindingModel

## Overview

This guide describes how to use [Pydantic Logfire](https://logfire.pydantic.dev/) for observability in the FindingModel project, with a focus on AI agent evaluation suites and testing.

**Logfire** is an observability platform built by the Pydantic team that provides:

- OpenTelemetry-based tracing, metrics, and logs
- Python-centric insights with rich object display
- SQL-based querying of telemetry data
- Seamless integration with Pydantic AI and pytest

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Configuration Patterns](#configuration-patterns)
3. [Using Logfire in Evaluation Suites](#using-logfire-in-evaluation-suites)
4. [Testing with Logfire](#testing-with-logfire)
5. [Environment Variables](#environment-variables)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Installation

Logfire is included in the project's development dependencies:

```bash
# Install all dev dependencies including logfire
uv pip install -e ".[dev]"

# Or install logfire specifically
uv pip install logfire
```

### Authentication

For full Logfire platform features (sending traces to the cloud):

```bash
# Authenticate with Logfire (interactive)
logfire auth

# This creates credentials in ~/.logfire/
```

**Note:** Authentication is **optional** for local development and testing. Logfire will gracefully degrade if no token is present.

## Configuration Patterns

### Pattern 1: Package-Level Configuration (Recommended for Evals)

**PREFERRED APPROACH**: Configure Logfire once at package level in `__init__.py`.

This follows Python logging best practices and eliminates code duplication.

```python
# evals/__init__.py
"""Evaluation suites for findingmodel agents.

Logfire observability configured automatically for entire package.
Individual eval modules require NO Logfire code.
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
- **DRY principle**: Configuration in ONE place
- **Zero code per eval**: Individual modules need no Logfire imports
- **Automatic instrumentation**: Pydantic Evals + Pydantic AI handle everything
- **Scalable**: Add new eval suites without duplicating configuration

**How it works:**
- Python guarantees `__init__.py` executes before any child module imports
- Configuration happens automatically on first import of any eval module
- Dataset.evaluate() automatically creates spans
- Pydantic AI instrumentation captures agent/model/tool calls

### Pattern 2: Module-Level Configuration (Legacy)

**NOTE:** This pattern is shown for reference only. Use Package-Level Configuration instead.

```python
import logfire
from typing import TYPE_CHECKING

# Configure at module level (NOT RECOMMENDED - use package-level instead)
logfire.configure(
    send_to_logfire='if-token-present',  # Only send if LOGFIRE_TOKEN exists
    service_name='findingmodel-evals',
    environment='test',
    console=True,  # Always show console output
    console_min_log_level='info',  # Only show important logs in console
)

if TYPE_CHECKING:
    # For type checking only, doesn't run at runtime
    from logfire import Logfire
```

**Key points:**

- `send_to_logfire='if-token-present'` enables automatic token detection
- Gracefully degrades to local-only logging if no token
- Console output still works without authentication
- No warnings or errors if token is missing

### Pattern 2: Explicit Control via Environment Variable

For more control, use an environment variable:

```python
import os
import logfire

# Check both LOGFIRE_TOKEN and a custom disable flag
enable_logfire = (
    os.getenv('LOGFIRE_TOKEN') is not None
    and os.getenv('LOGFIRE_DISABLE', 'false').lower() != 'true'
)

logfire.configure(
    send_to_logfire=enable_logfire,
    service_name='findingmodel-evals',
    environment='test',
    console=True,
)

if enable_logfire:
    logfire.info('Logfire enabled - traces will be sent to platform')
else:
    logfire.info('Logfire local-only mode - traces stay on this machine')
```

### Pattern 3: Testing Configuration

When writing tests for code that uses Logfire:

```python
import pytest
import logfire
from logfire.testing import capfire, CaptureLogfire

@pytest.fixture(autouse=True)
def configure_logfire_for_tests():
    """Configure Logfire for test environment."""
    logfire.configure(
        send_to_logfire=False,  # Never send during tests
        console=False,  # Quiet during tests
    )

def test_my_instrumented_code(capfire: CaptureLogfire):
    """Test code with Logfire instrumentation."""
    # Your code that emits spans/logs
    with logfire.span('processing'):
        result = process_data()

    # Assert on captured spans
    assert len(capfire.exporter.exported_spans) == 2  # pending + completed
    assert capfire.exporter.exported_spans[-1].name == 'processing'
```

**Note:** Pytest automatically sets `send_to_logfire=False` when running tests, so explicit configuration is often unnecessary.

## Using Logfire in Evaluation Suites

### Automatic Instrumentation (Recommended)

**PREFERRED APPROACH**: Use package-level configuration in `evals/__init__.py`.

Individual eval modules require **ZERO Logfire code**. Automatic instrumentation provides:

```python
# evals/my_agent.py - NO Logfire imports needed!
from pydantic_evals import Dataset, Case
from findingmodel.tools import my_agent

# 1. Define data models
class MyInput(BaseModel): ...
class MyOutput(BaseModel): ...

# 2. Create evaluators
evaluators = [MyEvaluator(), ...]

# 3. Create dataset
dataset = Dataset(cases=all_cases, evaluators=evaluators)

# 4. Task function - NO manual spans needed
async def run_my_agent_task(input_data: MyInput) -> MyOutput:
    """Execute task - automatic instrumentation captures everything."""
    result = await my_agent.process(input_data)
    return MyOutput(result=result)

# 5. Main eval function - NO manual spans needed
async def run_my_agent_evals():
    """Dataset.evaluate() automatically creates spans."""
    report = await dataset.evaluate(run_my_agent_task)
    return report
```

**What gets traced automatically:**
- Evaluation root span (from Dataset.evaluate())
- Per-case execution spans (from Dataset.evaluate())
- Agent run spans (from logfire.instrument_pydantic_ai())
- Model call spans with prompts/completions
- Tool execution spans
- Evaluation scores and results

**Benefits:**
- Zero Logfire code per eval module
- No risk of forgetting to add spans
- Consistent tracing across all evals
- Easier to maintain and refactor

### Manual Instrumentation (Legacy)

**NOTE:** Manual spans are shown for reference only. Use automatic instrumentation instead.

```python
import logfire
from pydantic_evals import Dataset, Case

# NOT RECOMMENDED - use automatic instrumentation instead
async def run_evaluation_case(case: Case) -> dict:
    """Run a single evaluation case with manual tracing."""
    with logfire.span(
        'eval_case {name}',
        name=case.name,
        case_id=case.name,
        case_category=case.metadata.get('category', 'unknown')
    ):
        # Execute the case
        result = await execute_agent(case.inputs)

        # Log important events
        logfire.info(
            'Case completed',
            success=result.success,
            score=result.score,
        )

        return result

# Run evaluation
dataset = Dataset(cases=all_cases, evaluators=evaluators)

with logfire.span('eval_suite model_editor', total_cases=len(all_cases)):
    report = await dataset.evaluate(run_evaluation_case)

    logfire.info(
        'Evaluation complete',
        overall_score=report.overall_score(),
        cases_passed=sum(1 for r in report.results if r.score >= 0.8),
        cases_total=len(report.results),
    )
```

### Instrumentation Best Practices

**DO:**

```python
# ✅ Use structured logging with named parameters
logfire.info('Agent processed {count} items', count=len(items))

# ✅ Add context to spans
with logfire.span('model_edit', model_id=model.oifm_id, operation='add_attribute'):
    result = editor.edit(model, command)

# ✅ Log at appropriate levels
logfire.debug('Intermediate calculation', value=x)  # Verbose details
logfire.info('Case completed', success=True)        # Important events
logfire.warning('Unexpected value', value=x)        # Warnings
logfire.error('Operation failed', error=str(e))     # Errors

# ✅ Use spans for operations that take time
with logfire.span('database_query'):
    results = await db.query(...)
```

**DON'T:**

```python
# ❌ Don't use string formatting (f-strings are OK with inspect_arguments=True)
logfire.info(f'Processed {count} items')  # Loses structure

# ❌ Don't log sensitive data
logfire.info('API key', key=api_key)  # Security risk!

# ❌ Don't create spans for trivial operations
with logfire.span('add_numbers'):  # Too granular
    result = a + b

# ❌ Don't log in tight loops without sampling
for item in million_items:
    logfire.debug('Processing', item=item)  # Will overwhelm system
```

### Instrumenting Evaluators

Add tracing to custom evaluators:

```python
import logfire
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

class ModelEditorEvaluator(Evaluator[InputT, OutputT]):
    """Custom evaluator with Logfire tracing."""

    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        with logfire.span(
            'evaluate {evaluator}',
            evaluator=self.__class__.__name__,
            case_name=ctx.case_name,
        ):
            # Evaluation logic
            score = self._calculate_score(ctx)

            logfire.info(
                'Evaluation result',
                evaluator=self.__class__.__name__,
                score=score,
                passed=score >= 0.8,
            )

            return score
```

## Testing with Logfire

### Using the `capfire` Fixture

Logfire provides excellent testing utilities:

```python
import pytest
import logfire
from logfire.testing import CaptureLogfire

def test_agent_logging(capfire: CaptureLogfire):
    """Test that agent emits expected spans."""
    # Run code that uses logfire
    with logfire.span('agent_task'):
        logfire.info('Processing started')
        result = my_agent.process()
        logfire.info('Processing completed', success=True)

    # Access captured spans
    exporter = capfire.exporter

    # Check span count
    assert len(exporter.exported_spans) == 3  # pending, 2 logs, completed span

    # Check span names
    span_names = [span.name for span in exporter.exported_spans]
    assert 'agent_task' in span_names
    assert 'Processing started' in span_names

    # Get structured representation for detailed assertions
    spans_dict = exporter.exported_spans_as_dict()
    assert spans_dict[0]['attributes']['logfire.msg'] == 'Processing started'

    # Clear for next assertion
    exporter.clear()
```

### Snapshot Testing with Logfire

For deterministic testing of complex instrumentation:

```python
import pytest
import logfire
from logfire.testing import CaptureLogfire
from inline_snapshot import snapshot

def test_agent_instrumentation_snapshot(capfire: CaptureLogfire):
    """Test agent instrumentation with snapshot testing."""
    # Run instrumented code
    with logfire.span('process_model'):
        logfire.info('Model validated', model_id='OIFM_TEST_000001')

    # Assert against snapshot (updates with --inline-snapshot=fix)
    assert capfire.exporter.exported_spans_as_dict() == snapshot([
        {
            'name': 'Model validated',
            'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'attributes': {
                'logfire.span_type': 'log',
                'logfire.level_num': 9,
                'logfire.msg': 'Model validated',
                'model_id': 'OIFM_TEST_000001',
            },
        },
        {
            'name': 'process_model',
            'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'attributes': {
                'logfire.msg': 'process_model',
            },
        },
    ])
```

### Test Configuration

Create a `conftest.py` fixture for consistent test configuration:

```python
# test/conftest.py
import pytest
import logfire

@pytest.fixture(scope='session', autouse=True)
def configure_logfire():
    """Configure Logfire for all tests."""
    logfire.configure(
        send_to_logfire=False,  # Never send during tests
        console=False,  # Quiet during tests
    )
```

## Environment Variables

### Core Configuration Variables

| Variable                        | Description                               | Default | Example               |
| ------------------------------- | ----------------------------------------- | ------- | --------------------- |
| `LOGFIRE_TOKEN`                 | Authentication token for Logfire platform | None    | `pfp_abc123...`       |
| `LOGFIRE_SEND_TO_LOGFIRE`       | Whether to send spans to platform         | `true`  | `false`               |
| `LOGFIRE_SERVICE_NAME`          | Name of the service                       | None    | `findingmodel-evals`  |
| `LOGFIRE_ENVIRONMENT`           | Environment name                          | None    | `test`, `dev`, `prod` |
| `LOGFIRE_CONSOLE`               | Enable console output                     | `true`  | `false`               |
| `LOGFIRE_CONSOLE_MIN_LOG_LEVEL` | Minimum level for console                 | `info`  | `debug`, `warning`    |

### Custom Variables for This Project

We use these custom environment variables:

| Variable               | Description                | Default | Example |
| ---------------------- | -------------------------- | ------- | ------- |
| `LOGFIRE_DISABLE`      | Completely disable Logfire | `false` | `true`  |
| `LOGFIRE_EVAL_VERBOSE` | Verbose eval logging       | `false` | `true`  |

### Setting Variables

**For local development (.env):**

```bash
# .env file
LOGFIRE_TOKEN=pfp_your_token_here
LOGFIRE_SERVICE_NAME=findingmodel-evals
LOGFIRE_ENVIRONMENT=development
LOGFIRE_CONSOLE_MIN_LOG_LEVEL=info
```

**For CI/CD:**

```bash
# GitHub Actions, GitLab CI, etc.
export LOGFIRE_TOKEN=${{ secrets.LOGFIRE_TOKEN }}
export LOGFIRE_SERVICE_NAME=findingmodel-evals
export LOGFIRE_ENVIRONMENT=ci
export LOGFIRE_SEND_TO_LOGFIRE=true
```

**For one-off testing:**

```bash
# Disable Logfire entirely
LOGFIRE_DISABLE=true pytest test/evals/

# Enable verbose logging
LOGFIRE_EVAL_VERBOSE=true pytest test/evals/test_model_editor_evals.py
```

## Best Practices

### 1. Configure Once at Module Level

```python
# At the top of your evaluation module
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    service_name='findingmodel-evals',
    environment='test',
)
```

**Don't:** Call `configure()` multiple times or inside functions.

### 2. Use Structured Logging

```python
# ✅ Good: Structured with named parameters
logfire.info('Evaluated {count} cases in {duration}s', count=10, duration=5.2)

# ❌ Bad: Unstructured string
logfire.info(f'Evaluated 10 cases in 5.2s')
```

### 3. Choose Appropriate Span Granularity

```python
# ✅ Good: Meaningful operations
with logfire.span('evaluate_model_editor_case', case_name=case.name):
    result = await run_case(case)

# ❌ Too granular: Single function calls
with logfire.span('validate_input'):
    validate(input)  # Unless this is expensive
```

### 4. Add Context to Spans

```python
# ✅ Good: Rich context
with logfire.span(
    'model_edit',
    model_id=model.oifm_id,
    operation=command.operation,
    user='agent',
):
    result = editor.edit(model, command)

# ❌ Bad: No context
with logfire.span('edit'):
    result = editor.edit(model, command)
```

### 5. Handle Exceptions in Spans

```python
# ✅ Good: Exceptions are automatically captured
with logfire.span('risky_operation'):
    result = might_fail()  # Exception will be recorded in span

# If you catch exceptions, log them
try:
    result = might_fail()
except Exception as e:
    logfire.error('Operation failed', error=str(e), exc_info=True)
    raise
```

### 6. Use Conditional Verbose Logging

```python
import os

VERBOSE = os.getenv('LOGFIRE_EVAL_VERBOSE', 'false').lower() == 'true'

if VERBOSE:
    logfire.debug('Detailed evaluation step', intermediate_value=value)
```

### 7. Avoid Logging Secrets

```python
# ✅ Good: Sanitize sensitive data
logfire.info('API call', endpoint=url, headers={'Authorization': '***'})

# ❌ Bad: Logging secrets
logfire.info('API call', headers=headers)  # May contain auth tokens
```

### 8. Sample High-Volume Events

```python
import random

# Only log 1% of high-frequency events
if random.random() < 0.01:
    logfire.debug('High frequency event', data=data)
```

## Troubleshooting

### Issue: No traces appearing in Logfire platform

**Possible causes:**

1. No `LOGFIRE_TOKEN` set
2. `send_to_logfire=False` in configuration
3. Running under pytest (automatically sets `send_to_logfire=False`)

**Solution:**

```bash
# Check token exists
echo $LOGFIRE_TOKEN

# Explicitly enable sending
logfire.configure(send_to_logfire=True)

# Or use 'if-token-present' mode
logfire.configure(send_to_logfire='if-token-present')
```

### Issue: Too many logs/spans, overwhelming output

**Solution:**

```python
# Increase minimum log level
logfire.configure(console_min_log_level='warning')

# Or disable console output
logfire.configure(console=False)

# Use LOGFIRE_CONSOLE_MIN_LOG_LEVEL environment variable
export LOGFIRE_CONSOLE_MIN_LOG_LEVEL=warning
```

### Issue: Tests failing due to Logfire configuration

**Solution:**

```python
# In conftest.py, ensure clean configuration
@pytest.fixture(scope='session', autouse=True)
def configure_logfire():
    logfire.configure(send_to_logfire=False, console=False)
```

### Issue: Wanting to see traces locally without sending to cloud

**Solution:**

```python
logfire.configure(
    send_to_logfire=False,  # Keep local
    console=True,           # See in terminal
    console_verbose=True,   # More details
)
```

### Issue: Token authentication errors

**Possible causes:**

1. Invalid or expired token
2. Token for wrong project
3. Network connectivity issues

**Solution:**

```bash
# Re-authenticate
logfire auth

# Check credentials file
cat ~/.logfire/credentials.json

# Test with explicit token
export LOGFIRE_TOKEN=pfp_your_token_here
```

## Integration with Pydantic AI Evals

Logfire integrates seamlessly with Pydantic AI Evals. Example configuration for `test/evals/test_model_editor_evals.py`:

```python
"""
Model Editor evaluation suite with Logfire observability.

This module uses Logfire for tracing and observability. Traces are sent
to the Logfire platform if a token is present, otherwise they are logged
locally only.

To enable Logfire platform integration:
    1. Run: logfire auth
    2. Or set: LOGFIRE_TOKEN environment variable

To disable Logfire entirely:
    export LOGFIRE_DISABLE=true
"""

import os
import logfire
from pydantic_evals import Dataset, Case

# Configure Logfire at module level
if os.getenv('LOGFIRE_DISABLE', 'false').lower() != 'true':
    logfire.configure(
        send_to_logfire='if-token-present',
        service_name='findingmodel-model-editor-evals',
        environment='test',
        console=True,
        console_min_log_level='info',
    )
    logfire.info('Logfire configured for model_editor evaluations')
else:
    logfire.configure(send_to_logfire=False, console=False)
    # No-op configuration when disabled

# Rest of your evaluation suite code...
```

## References

- **Official Documentation:** https://logfire.pydantic.dev/docs/
- **Configuration Reference:** https://logfire.pydantic.dev/docs/reference/configuration/
- **Testing Guide:** https://logfire.pydantic.dev/docs/reference/advanced/testing/
- **GitHub Repository:** https://github.com/pydantic/logfire
- **Integrations:** https://logfire.pydantic.dev/docs/integrations/

## Summary

**Key Takeaways:**

1. Use `send_to_logfire='if-token-present'` for graceful degradation
2. Configure once at module level
3. Use structured logging with named parameters
4. Add rich context to spans
5. Test instrumentation with `capfire` fixture
6. Never log secrets or sensitive data
7. Use environment variables for configuration
8. Logfire works seamlessly with Pydantic AI Evals

**For most evaluation suites, this is all you need:**

```python
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    service_name='findingmodel-evals',
    environment='test',
    console=True,
)
```

This configuration will:

- Send traces to Logfire if authenticated
- Work locally without authentication
- Show console output for debugging
- Integrate automatically with Pydantic AI
