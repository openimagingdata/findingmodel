# Logfire Observability Guide

This guide documents the current Logfire integration used by the evaluation suites under `packages/findingmodel-ai/evals/`.

## Current Behavior

Eval suites use `ensure_instrumented()` from `packages/findingmodel-ai/evals/__init__.py`.

That function is idempotent and currently does exactly two things:

```python
logfire.configure(send_to_logfire="if-token-present", console=False)
logfire.instrument_pydantic_ai()
```

As a result:

- eval runs stay quiet in the terminal by default
- traces are sent to Logfire only when `LOGFIRE_TOKEN` is present
- Pydantic AI agent/model/tool calls are instrumented automatically

## How To Use It

When running an eval module directly, call `ensure_instrumented()` before starting the evaluation run:

```python
from evals import ensure_instrumented

ensure_instrumented()
```

Most eval modules in `packages/findingmodel-ai/evals/` already follow this pattern in their `__main__` blocks.

## Environment Variables

### Supported in current eval instrumentation

- `LOGFIRE_TOKEN`
  - Optional.
  - When set, Logfire sends traces to the configured Logfire project.

### Not implemented as project-level eval controls

The current eval instrumentation does not provide repo-specific support for extra knobs like:

- `DISABLE_SEND_TO_LOGFIRE`
- `LOGFIRE_VERBOSE`
- custom service/environment naming via project code

If those controls are needed, add them explicitly in `packages/findingmodel-ai/evals/__init__.py` and document them there.

## Where To Look

- `packages/findingmodel-ai/evals/__init__.py`
- `packages/findingmodel-ai/evals/README.md`
- <https://ai.pydantic.dev/evals/#integration-with-logfire>

## Historical Note

An older, much longer Logfire guide existed here and described configuration patterns that are not implemented in the current codebase. It has been replaced with this shorter guide so the documentation matches the actual repository behavior.
