---
name: test-review
description: Review rubric for pytest tests — coverage, markers, fixtures, AI test patterns
---

# Test Review

Review tests for quality, coverage, and correct use of project testing patterns.

## Scope

- Unit tests (pytest)
- Integration tests (`@pytest.mark.callout`)
- AI agent tests (TestModel/FunctionModel patterns)
- Fixtures and test data
- Eval quality scores

**Out of scope**: Production code (use `/python-review` or `/pydantic-ai-review`)

## Criteria

### 1. Completeness
- All public methods tested?
- Error cases covered?
- Edge cases (empty, None, boundaries)?
- Integration points tested?
- For AI tests: agent config, tools, workflow, validators all tested?

### 2. Focus
- Tests verify behavior, not implementation details?
- Not testing library features (pytest, Pydantic, Pydantic AI)?
- Each test has a clear purpose?
- Tests are independent?

### 3. Conciseness
- Fixtures reused via `conftest.py`?
- No duplicate setup code?
- Parametrized where appropriate?
- One assertion focus per test?

### 4. Appropriateness

**General pytest**:
- `@pytest.mark.callout` on external API tests
- Async fixtures for async code
- Test data in `packages/<pkg>/tests/data/`
- Tests mirror source structure

**AI agent tests (critical checks)**:
- `models.ALLOW_MODEL_REQUESTS = False` at module level
- `TestModel` for deterministic outputs
- `FunctionModel` for controlled behavior
- `Agent.override()` pattern (not patching)
- Integration tests restore `ALLOW_MODEL_REQUESTS` in `finally`

**Three-tier structure**:
- Unit tests: fast, mocked, no API calls
- Integration tests: `@pytest.mark.callout`, real APIs
- Evals: quality scores 0.0–1.0

## Standards Reference

Read Serena `pydantic_ai_testing_best_practices` for AI test patterns and `pytest_fixtures_reference_2025` for fixture conventions.

## Severity Guide

- **Critical**: Missing `ALLOW_MODEL_REQUESTS = False`, real API calls without marker, broken test isolation
- **Important**: Missing coverage for core paths, wrong focus (testing library), poor fixture organization
- **Minor**: Could use parametrize, naming improvement, missing optional integration test
