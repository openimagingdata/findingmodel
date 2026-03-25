---
name: python-review
description: Review rubric for core Python implementations — data models, async patterns, database code, utilities
---

# Python Core Review

Review core Python code (non-AI) for quality, correctness, and project alignment.

## Scope

- Data structures and Pydantic models
- Sync/async patterns
- Database interactions (DuckDB)
- File I/O, parsing, CLI commands
- Configuration and validation
- Utility functions

**Out of scope**: AI agents (use `/pydantic-ai-review`), tests (use `/test-review`)

## Criteria

### 1. Completeness
- All required functions/methods present?
- All fields in data models?
- Error cases handled?
- Edge cases addressed?

### 2. Focus
- Only implements the assigned task?
- No unrelated features or refactoring?
- No over-anticipation of future needs?

### 3. Conciseness
- Code is straightforward and readable?
- No unnecessary abstraction layers?
- Reuses existing utilities (check with Serena)?
- YAGNI and DRY followed?

### 4. Appropriateness
- **Style**: 120 char lines, snake_case functions, PascalCase classes
- **Typing**: Strict type hints, `Annotated`/`Field` for constraints
- **Async**: Used for IO-bound code; removed if no awaits (RUF029)
- **Errors**: Project exceptions (e.g., `ConfigurationError`), cleanup with `try/finally`
- **Models**: Pydantic `BaseModel` for data, not raw dicts
- **Layout**: Code in correct package under `packages/`

## Standards Reference

Read Serena `code_style_conventions` for the full set. Key points:
- Ruff formatting with preview mode
- `Annotated` + `Field` over bare type hints for constrained fields
- Side-effect-free validators
- `importlib.resources.files()` for package data

## Severity Guide

- **Critical**: Broken functionality, data corruption risk, security issue
- **Important**: Standards violation, missing error handling, wrong async pattern
- **Minor**: Style preference, naming suggestion, optional optimization
