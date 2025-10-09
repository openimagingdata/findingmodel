---
argument-hint: <plan-file-path>
description: Orchestrates implementation of a multi-phase plan using specialized sub-agents
---

You are implementing a technical plan using the multi-agent orchestration system.

## Task

Read the plan file at: `$1`

Then delegate to the **plan-orchestrator** sub-agent to coordinate implementation.

The orchestrator will:
1. Break the plan into phases
2. Delegate each phase to appropriate implementation agents
3. Coordinate evaluation and fix cycles
4. Run tests and make git commits
5. Update documentation

## Usage

```bash
/implement-plan tasks/my-plan.md
```

## Plan File Format

The plan should be a markdown file with clear phases. Example structure:

```markdown
# My Implementation Plan

## Phase 1: Create Data Models
- Implement X model
- Implement Y model
- Add validation

## Phase 2: Implement Core Logic
- Add method A
- Add method B
- Error handling

## Phase 3: Add Tests
- Unit tests for models
- Integration tests for logic
```

## Orchestrator Handoff

Pass the plan content and path to the plan-orchestrator agent:

```
I'm handing off implementation of the plan at $1 to you.

Please:
1. Read the plan file
2. Identify all phases
3. For each phase:
   - Delegate to appropriate implementation agent
   - Have evaluation agent assess the work
   - Coordinate fixes if needed
   - Run tests
   - Create git commit
   - Update Serena memories

The plan content is:

[include plan file content here]
```

Delegate using the Task tool with `subagent_type="plan-orchestrator"`.
