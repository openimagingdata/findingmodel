# CLAUDE.md - Eval Development Quick Reference

This is a lightweight pointer for AI agents. For complete guidance, see the authoritative sources below.

## Purpose

Quick reference for AI agents developing evaluation suites. All detailed patterns, templates, and conventions are in `.serena/memories/agent_evaluation_best_practices_2025.md`.

## Documentation Map

- **`.serena/memories/agent_evaluation_best_practices_2025.md`** - Authoritative reference for AI agents (complete patterns, templates, conventions)
- **`evals/evals_guide.md`** - Comprehensive tutorial for humans learning to write evals
- **`evals/README.md`** - Quick-start guide for humans running evals
- **`evals/model_editor.py`** - Working example implementation

## Essential Quick Facts

### File Naming
- Eval file: `tool_name.py` NOT `test_tool_name.py`
- Main function: `run_tool_name_evals()` returns Report object
- Imports: Absolute (`from evals.base`) NOT relative (`from .base`)

### No Logfire Code Needed
Individual eval modules require ZERO Logfire code. Configuration is automatic via `evals/__init__.py`.

### Standard Pattern
```python
# 1. Data models (input, expected output, actual output)
# 2. Evaluators inheriting from Evaluator base class
# 3. Dataset at module level with cases + evaluators
# 4. Main function: async def run_X_evals() -> Report
# 5. __main__ block for standalone execution
```

### Critical Warning: TestModel Requires Dicts
TestModel `custom_output_args` must be a dict, not a Pydantic model:

```python
# ✅ Correct
TestModel(custom_output_args=my_model.model_dump())

# ❌ Wrong - fails with Logfire instrumentation
TestModel(custom_output_args=my_model)
```

**Why:** When Logfire instrumentation is active (automatic in evals/), it calls `.items()` on tool args for serialization. Pydantic models don't have `.items()`, causing `AttributeError`.

## Complete Reference

For detailed patterns, templates, file structure, conventions, anti-patterns, and lessons learned, see:

**`.serena/memories/agent_evaluation_best_practices_2025.md`**
