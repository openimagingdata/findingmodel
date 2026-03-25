# Multi-Agent Orchestration System (Updated 2026-03-23)

## Overview

A system for implementing multi-phase technical plans using specialized subagents and review skills. Modernized in March 2026 to replace the legacy evaluator matrix with a single reviewer + portable review skills.

## Architecture

### Orchestrator Layer
- **plan-orchestrator** (skill in `.claude/skills/plan-orchestrator/`) — Coordinates implementation
  - Reads plan documents, delegates to implementers, validates via reviewer
  - Includes guardrails: assesses orchestrator vs agent teams before starting, confirms with user
  - Handles escalations (max 3 cycles before escalating to user)

### Implementation Agents (`.claude/agents/`)
1. **python-core-implementer** — Core Python code (data structures, sync/async, DuckDB, CLI)
2. **pydantic-ai-implementer** — Pydantic AI agents, tools, workflows (v1.70+ patterns)
3. **python-test-implementer** — Pytest tests for core Python code
4. **ai-test-eval-implementer** — AI agent tests (TestModel/FunctionModel) AND Pydantic Evals suites

### Review Layer
- **reviewer** (`.claude/agents/reviewer.md`) — Single read-only subagent (sonnet model)
  - Applies appropriate review skill(s) based on domain
  - Returns APPROVED | NEEDS_REVISION | BLOCKED with file:line findings

### Review Skills (`.claude/skills/`)
- **python-review** — Core Python rubric (completeness, focus, conciseness, appropriateness)
- **pydantic-ai-review** — AI agent rubric (v1.70+ patterns, model config, toolsets)
- **test-review** — Test rubric (pytest + Pydantic Evals patterns)
- **docs-review** — Documentation rubric (accuracy, discoverability, alignment)

All review skills use portable frontmatter (name + description only) for OpenCode/Codex compatibility.

## Workflow

For each phase:
```
1. Orchestrator reads phase description
2. Delegates to appropriate implementer(s) — parallel when independent
3. Implementer completes work
4. Reviewer validates using relevant review skill(s)
5. Review returns: APPROVED | NEEDS_REVISION | BLOCKED
6. If NEEDS_REVISION: fix cycle (max 3)
7. If BLOCKED: check Serena → escalate to user
8. Run tests (task test)
9. Suggest commit for user confirmation
```

## Orchestrator vs Agent Teams

The plan-orchestrator assesses each plan before starting and recommends the right approach:

**Orchestrator is better for:**
- Phases with dependencies or ordering constraints
- Quality gates (implement → review → test → commit)
- Concrete plans with specific deliverables
- Token-cost-sensitive work

**Agent teams are better for:**
- 3+ truly independent phases touching separate files/packages
- Exploratory research or investigation
- Workers that need to discuss with each other
- Work without quality gates (spikes, prototyping)

The orchestrator always confirms with the user before proceeding.

## Key Design Changes (March 2026)

### What changed
- **10 agents → 5**: Removed 4 evaluators, refactoring analyzer, documentation updater
- **Evaluators → review skills**: Rubric content extracted into 4 portable skills
- **Single reviewer**: One read-only subagent replaces 4 domain-specific evaluators
- **ai-test-implementer → ai-test-eval-implementer**: Now covers Pydantic Evals suites too
- **All implementers updated**: Current monorepo layout, Pydantic AI v1.70+, per-agent model config, Logfire, loguru
- **Agent teams guardrails**: Orchestrator recommends approach before starting

### Why
- Legacy agents had stale references (MongoDB, wrong file paths, nonexistent Serena memories)
- Evaluator matrix was over-specialized — a single reviewer with skill-based rubrics is simpler and more maintainable
- Review skills are portable across Claude Code, OpenCode, and Codex

## Related Memories

- `code_style_conventions` — Coding standards all agents follow
- `pydantic_ai_best_practices` — AI patterns for implementers
- `pydantic_ai_testing_best_practices` — Testing patterns
- `agent_evaluation_best_practices_2025` — Eval suite patterns
- `evaluator_architecture_2025` — Where evaluators live
- `suggested_commands` — Dev workflow
