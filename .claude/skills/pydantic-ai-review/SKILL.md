---
name: pydantic-ai-review
description: Review rubric for Pydantic AI agent implementations — agents, tools, workflows, model config
---

# Pydantic AI Review

Review AI agent implementations for correctness, best practices, and project alignment.

## Scope

- Agent definitions and configuration
- Agent tools and their docstrings
- Output validators
- Multi-agent workflows
- Model selection and reasoning config

**Out of scope**: Core Python (use `/python-review`), tests (use `/test-review`)

## Criteria

### 1. Completeness
- All agent components present (output type, deps, tools, system prompt)?
- Tools have clear docstrings for the LLM?
- Output validators handle errors?
- Multi-agent handoff complete?

### 2. Focus
- Only AI logic implemented?
- Data structures delegated to core code?
- No test code mixed in?

### 3. Conciseness
- Agents focused on single responsibility?
- Tools do one thing well?
- No unnecessary abstraction layers?
- YAGNI followed?

### 4. Appropriateness

**Agent structure (Pydantic AI v1.70+)**:
- Explicit `output_type` with Pydantic models (not `result_type` — pre-v1 name)
- Typed dependencies: `Agent[DepsType, OutputType]`
- `instructions=` preferred over `system_prompt=` (strips on history handoff)
- `description=` for observability (v1.69+)
- Tool Output mode (default) unless model requires Native/Text

**Tools and toolsets**:
- Clear docstrings (LLM reads these — griffe parses google/numpy/sphinx styles)
- `RunContext` for dependency injection; `@agent.tool_plain` when no context needed
- Toolsets (`FunctionToolset`, etc.) for reusable tool collections
- `args_validator=` for pre-execution validation (v1.63+)

**Output validators**:
- Async for IO operations
- `ModelRetry` for recoverable errors
- Prefer adjusting instructions over post-processing

**Workflows**:
- Two-agent pattern for complex tasks (search → analyze)
- Delegation via tool with `ctx.usage` for combined tracking
- Structured outputs at each step

**Project-specific**:
- Model config via `settings.get_agent_model("tag")` and `supported_models.toml`
- No schema/enum duplication in prompts (the structured output IS the spec)
- All config via pydantic-settings, never `os.environ`
- Logfire observability via `logfire.instrument_pydantic_ai()`

## Standards Reference

Read Serena `pydantic_ai_best_practices` for current patterns. Always verify against latest Pydantic AI docs with Ref/Context7 for API changes.

## Severity Guide

- **Critical**: Wrong model usage, missing output validation, security issue
- **Important**: Untyped agent, poor tool docstrings, sync validator with async ops
- **Minor**: Could use two-agent pattern, tool naming suggestion
