# Logfire Observability for FindingModel - 2025

## Overview

Pydantic Logfire is used for observability in the FindingModel project, particularly in AI agent evaluation suites. It provides OpenTelemetry-based tracing, metrics, and logs with Python-centric insights.

**Full Documentation:** `docs/logfire_observability_guide.md`

## Key Design Principles

### 1. Graceful Degradation (On by Default)

Logfire should be **enabled by default** but work seamlessly without authentication:

```python
import logfire

logfire.configure(
    send_to_logfire='if-token-present',  # Auto-detect token
    service_name='findingmodel-evals',
    environment='test',
    console=True,  # Always show console output
)
```

**Benefits:**
- Works immediately without setup
- Developers get local observability for free
- Production users can opt-in to cloud tracing
- No errors or warnings when token absent

### 2. Environment Variable Control

Support these environment variables for configuration:

- `LOGFIRE_TOKEN` - Authentication token (from `logfire auth`)
- `LOGFIRE_DISABLE` - Set to `'true'` to completely disable Logfire
- `LOGFIRE_EVAL_VERBOSE` - Set to `'true'` for verbose evaluation logging
- `LOGFIRE_SEND_TO_LOGFIRE` - Override sending behavior
- `LOGFIRE_SERVICE_NAME` - Service name for tracing
- `LOGFIRE_ENVIRONMENT` - Environment (test, dev, prod)

### 3. Module-Level Configuration

Configure Logfire once at the top of each module:

```python
"""Module with Logfire observability."""

import os
import logfire

# Check if disabled
LOGFIRE_DISABLED = os.getenv('LOGFIRE_DISABLE', 'false').lower() == 'true'

if not LOGFIRE_DISABLED:
    logfire.configure(
        send_to_logfire='if-token-present',
        service_name='findingmodel-model-editor-evals',
        environment='test',
        console=True,
    )
    
    # Log configuration status
    if os.getenv('LOGFIRE_TOKEN'):
        logfire.info('Logfire enabled - traces will be sent to platform')
    else:
        logfire.info('Logfire local-only mode')
else:
    logfire.configure(send_to_logfire=False, console=False)
```

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
with logfire.span('eval_case {name}', name=case.name, case_type='success'):
    result = await run_case(case)

# ❌ Too granular
with logfire.span('validate_input'):
    validate(input)  # Unless expensive
```

### Rich Context in Spans

```python
# ✅ Good: Rich context
with logfire.span(
    'model_edit',
    model_id=model.oifm_id,
    operation='add_attribute',
    edit_type='natural_language',
):
    result = editor.edit(model, command)
```

## Evaluation Suite Integration

### Module-Level Setup

At the top of evaluation test files:

```python
import os
import logfire

LOGFIRE_DISABLED = os.getenv('LOGFIRE_DISABLE', 'false').lower() == 'true'
VERBOSE_EVALS = os.getenv('LOGFIRE_EVAL_VERBOSE', 'false').lower() == 'true'

if not LOGFIRE_DISABLED:
    logfire.configure(
        send_to_logfire='if-token-present',
        service_name='findingmodel-<agent>-evals',
        environment='test',
        console=True,
        console_min_log_level='debug' if VERBOSE_EVALS else 'info',
    )
```

### Suite-Level Instrumentation

Wrap entire evaluation suite:

```python
@pytest.mark.callout
@pytest.mark.asyncio
async def test_run_agent_evals():
    with logfire.span('agent_eval_suite', total_cases=len(cases)):
        logfire.info('Starting evaluation suite', cases_total=len(cases))
        
        report = await dataset.evaluate(task_function)
        
        logfire.info(
            'Evaluation complete',
            overall_score=report.overall_score(),
            cases_passed=sum(1 for r in report.results if r.score >= 0.8),
        )
```

### Case-Level Instrumentation

Instrument individual case execution:

```python
async def run_agent_task(case: AgentCase) -> AgentOutput:
    with logfire.span(
        'eval_case {name}',
        name=case.name,
        case_type=case.metadata.get('category'),
        should_succeed=case.expected.should_succeed,
    ):
        # Execute case
        result = await agent.process(case.input)
        
        # Log result
        logfire.info(
            'Case completed',
            case_name=case.name,
            success=result.success,
        )
        
        return result
```

## Testing with Logfire

### Using capfire Fixture

```python
from logfire.testing import CaptureLogfire

def test_agent_instrumentation(capfire: CaptureLogfire):
    """Test that agent emits expected spans."""
    with logfire.span('agent_task'):
        logfire.info('Processing')
        result = agent.process()
    
    # Assert on captured spans
    assert len(capfire.exporter.exported_spans) == 3
    assert 'agent_task' in [s.name for s in capfire.exporter.exported_spans]
```

### Test Configuration

Pytest automatically sets `send_to_logfire=False`. For explicit control:

```python
# In conftest.py
import pytest
import logfire

@pytest.fixture(scope='session', autouse=True)
def configure_logfire():
    logfire.configure(send_to_logfire=False, console=False)
```

## Best Practices

### DO

- ✅ Configure once at module level
- ✅ Use `send_to_logfire='if-token-present'` for graceful degradation
- ✅ Use structured logging with named parameters
- ✅ Add rich context to spans (IDs, types, metadata)
- ✅ Log at appropriate levels (debug, info, warning, error)
- ✅ Wrap meaningful operations in spans
- ✅ Handle exceptions in spans (auto-captured)
- ✅ Use environment variables for configuration
- ✅ Test instrumentation with `capfire` fixture

### DON'T

- ❌ Call `configure()` multiple times
- ❌ Use f-strings instead of named parameters
- ❌ Log sensitive data (API keys, tokens, secrets)
- ❌ Create spans for trivial operations
- ❌ Log in tight loops without sampling
- ❌ Require Logfire authentication for local development
- ❌ Fail loudly when no token present

## Setup for Developers

### Local Development (No Cloud)

```bash
# Works immediately - local-only logging
pytest test/evals/test_model_editor_evals.py -v
```

### Cloud Tracing (Optional)

```bash
# One-time setup
logfire auth

# Run evals - traces appear in Logfire UI
pytest test/evals/test_model_editor_evals.py -v
```

### Disabling Logfire

```bash
# Completely disable
LOGFIRE_DISABLE=true pytest test/evals/

# Or in .env
LOGFIRE_DISABLE=true
```

## Common Environment Configurations

### Local Development
```bash
# .env (or no config - uses defaults)
LOGFIRE_CONSOLE=true
LOGFIRE_CONSOLE_MIN_LOG_LEVEL=info
```

### CI/CD
```bash
# GitHub Actions secrets
LOGFIRE_TOKEN=${{ secrets.LOGFIRE_TOKEN }}
LOGFIRE_SERVICE_NAME=findingmodel-evals
LOGFIRE_ENVIRONMENT=ci
LOGFIRE_SEND_TO_LOGFIRE=true
```

### Production
```bash
LOGFIRE_TOKEN=<production-token>
LOGFIRE_SERVICE_NAME=findingmodel-production
LOGFIRE_ENVIRONMENT=prod
LOGFIRE_SEND_TO_LOGFIRE=true
LOGFIRE_CONSOLE=false
```

## Troubleshooting

### No traces in Logfire platform

**Cause:** No token or `send_to_logfire=False`

**Solution:**
```bash
logfire auth
# Or set explicitly:
export LOGFIRE_TOKEN=pfp_your_token
```

### Too many logs

**Solution:**
```python
logfire.configure(console_min_log_level='warning')
# Or environment variable:
export LOGFIRE_CONSOLE_MIN_LOG_LEVEL=warning
```

### Tests failing

**Cause:** Logfire configuration issues

**Solution:** Pytest auto-disables sending, but ensure clean configuration in conftest.py

## Current Status in Project

### Implemented
- ✅ Comprehensive documentation: `docs/logfire_observability_guide.md`
- ✅ Phase 3 plan detailed in `tasks/refactor_model_editor_evals.md`
- ✅ Patterns established for evaluation suites

### Pending Implementation
- 🔲 Add `logfire>=1.0.0` to pyproject.toml dev dependencies
- 🔲 Instrument `test/evals/test_model_editor_evals.py`
- 🔲 Update `test/evals/README.md` with Logfire section
- 🔲 Verify integration with all existing tests

### Future Work
- 🔲 Apply Logfire pattern to other agent evaluation suites
- 🔲 Optional: Instrument base evaluators with opt-in tracing
- 🔲 Set up CI/CD integration for production monitoring

## Integration Points

### Pydantic AI Evals
Logfire works seamlessly with Pydantic Evals framework - just add spans and logging in evaluation functions.

### Pytest
Automatically disables platform sending during test runs. Use `capfire` fixture to test instrumentation.

### Environment Detection
Logfire auto-detects pytest environment and adjusts configuration accordingly.

## Resources

- **Full guide:** `docs/logfire_observability_guide.md`
- **Task plan:** `tasks/refactor_model_editor_evals.md` (Phase 3)
- **Official docs:** https://logfire.pydantic.dev/docs/
- **Configuration:** https://logfire.pydantic.dev/docs/reference/configuration/
- **Testing:** https://logfire.pydantic.dev/docs/reference/advanced/testing/

## Key Takeaway

For most evaluation suites, this simple configuration is all you need:

```python
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    service_name='findingmodel-<agent>-evals',
    environment='test',
    console=True,
)
```

This provides:
- Local observability without setup
- Cloud tracing when authenticated
- Graceful degradation
- Zero friction for developers
