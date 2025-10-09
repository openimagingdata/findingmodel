# Multi-Agent Orchestration System (2025)

## Overview

A system for implementing multi-phase technical plans using specialized sub-agents. Designed for conciseness and clarity - all agent definitions condensed to focus on essential patterns.

## Architecture

### Orchestrator Layer
- **plan-orchestrator** (78 lines) - Coordinates entire implementation process
  - Reads plan documents
  - Delegates phases to implementation agents (parallel when possible)
  - Coordinates evaluation cycles
  - Handles escalations
  - Manages git commits and Serena updates

### Implementation Agents
1. **python-core-implementer** (166 lines) - Core Python code (data structures, sync/async, non-AI logic)
2. **pydantic-ai-implementer** (81 lines) - Pydantic AI agents, tools, workflows
3. **python-test-implementer** (102 lines) - Pytest tests for core Python code
4. **ai-test-implementer** (127 lines) - Tests for Pydantic AI agents (TestModel/FunctionModel)

### Evaluation Agents
1. **python-core-evaluator** (242 lines) - Evaluates core Python implementations
2. **pydantic-ai-evaluator** (82 lines) - Evaluates Pydantic AI implementations
3. **python-test-evaluator** (95 lines) - Evaluates core Python tests
4. **ai-test-evaluator** (135 lines) - Evaluates AI agent tests

**Total: 1108 lines** (down from ~2384 original - 54% reduction)

## Workflow

For each phase:
```
1. Orchestrator reads phase description
2. Delegates to appropriate implementation agent(s)
   - Parallel delegation when phases independent
   - Sequential when dependencies exist
3. Implementation agent completes work
4. Corresponding evaluation agent assesses work
5. Evaluation returns: APPROVED | NEEDS_REVISION | BLOCKED
6. If NEEDS_REVISION:
   a. Orchestrator reformulates issues for implementer
   b. Implementer fixes
   c. Re-evaluate (max 3 cycles)
7. If BLOCKED:
   a. Check Serena for guidance
   b. Provide guidance if clear from patterns
   c. Escalate to user if architectural decision needed
8. Run tests (task test or task test-full)
9. If tests fail, iterate to fix (max 3 cycles)
10. Create git commit for phase
11. Update Serena memories with learnings
```

## Evaluation Criteria

All evaluators check:
- **Completeness**: Task fully done?
- **Focus**: Stayed within boundaries?
- **Conciseness**: YAGNI/DRY adherence? Not over-engineered?
- **Appropriateness**: Follows local standards? Reuses existing code? Fits tech stack?

Status responses:
- **APPROVED**: Ready for tests/next phase
- **NEEDS_REVISION**: Fixable issues with specific guidance
- **BLOCKED**: Requires architectural decision or user guidance

## Orchestration Patterns

### Parallel Processing
When phases are independent:
```python
# Single message with multiple Task calls
Task(subagent_type="python-core-implementer", ...)
Task(subagent_type="python-test-implementer", ...)
```

### Sequential Processing
When phases have dependencies:
```python
# Wait for each to complete before next
Task(subagent_type="python-core-implementer", ...)
# Wait for result...
Task(subagent_type="python-core-evaluator", ...)
```

### Escalation Handling
Agents report to orchestrator (not user directly):
```
Agent → Orchestrator → (check Serena) → User (if needed)
```

## Usage

### Via Slash Command
```bash
/implement-plan tasks/my-plan.md
```

### Via Review Command
```bash
/review-plan tasks/my-plan.md
```

## Plan File Format

Markdown file with clear phases:
```markdown
# Plan Title

## Phase 1: Description
- Specific task
- Another task
- Success criteria

## Phase 2: Description
- Tasks for this phase
```

## Agent Specialization

### Implementation Agents
- Have Read, Write, Edit, Bash, Serena tools
- Access project context via Serena memories
- Follow project coding standards
- Escalate to orchestrator when unclear

### Evaluation Agents
- Have Read, Grep, Glob, Serena read-only tools
- Cannot edit code (only evaluate)
- Provide specific, actionable feedback with file:line references
- Use APPROVED/NEEDS_REVISION/BLOCKED status

## Project Context Integration

All agents read relevant Serena memories:
- `project_overview` - Architecture
- `code_style_conventions` - Standards
- `pydantic_ai_best_practices_2025_09` - AI patterns
- `pydantic_ai_testing_best_practices` - Testing patterns
- `suggested_commands` - Dev workflow

## Git Commits

Orchestrator creates commits with:
- Descriptive message explaining phase
- Reference to plan document
- Claude Code co-author footer

## Condensing Improvements (2025)

Reduced verbosity by:
1. **Removing code examples** - Reference existing code instead of 100+ lines of inline examples
2. **Shortening "Before You Finish"** - From 8-9 items to 5 max checklist items
3. **Eliminating redundancy** - Removed "Communication Style" sections (obvious)
4. **Consolidating examples** - Keep 1 brief example max instead of multiple
5. **Referencing Serena** - Point to memories instead of listing all project context

Result: 54% reduction (2384→1108 lines) while maintaining clarity and completeness.

## Key Design Principles

1. **Separation of concerns**: Implementation vs evaluation
2. **Specialization**: Each agent has focused expertise
3. **Iterative refinement**: Fix cycles until quality criteria met (max 3)
4. **Project alignment**: All agents follow local standards
5. **Context awareness**: Agents read Serena memories for consistency
6. **Concise definitions**: Technical language, essential patterns only
7. **Escalation chain**: Agents → Orchestrator → User (when needed)

## Agent Files

Located in `.claude/agents/`:
- `plan-orchestrator.md` (78 lines)
- `python-core-implementer.md` (166 lines)
- `pydantic-ai-implementer.md` (81 lines)
- `python-test-implementer.md` (102 lines)
- `ai-test-implementer.md` (127 lines)
- `python-core-evaluator.md` (242 lines)
- `pydantic-ai-evaluator.md` (82 lines)
- `python-test-evaluator.md` (95 lines)
- `ai-test-evaluator.md` (135 lines)

Commands in `.claude/commands/`:
- `implement-plan.md` - Orchestrates implementation
- `review-plan.md` - Assesses plan quality

## Example Plans

- `tasks/test-multi-agent-system.md` - Minimal test plan
- `tasks/index-duckdb-migration.md` - Complex real-world example

## Benefits

- **Consistency**: All implementations follow project standards
- **Quality**: Evaluation cycles ensure completeness and appropriateness
- **Efficiency**: Specialized agents + parallel processing
- **Maintainability**: Clear separation, concise definitions
- **Learning**: Serena memories capture patterns and decisions
- **Focus**: Technical language, no fluff

## Related Memories

- `code_style_conventions` - Coding standards all agents follow
- `pydantic_ai_best_practices_2025_09` - AI patterns for implementers
- `pydantic_ai_testing_best_practices` - Testing patterns for test agents
- `ai_assistant_usage_2025` - How AI assistants should work together
- `instruction_files_plan_2025` - Guidance file maintenance
