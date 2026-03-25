---
name: plan-orchestrator
description: Orchestrate multi-phase plan implementation with specialized subagents and review validation. Use when implementing a technical plan, executing a development roadmap, or coordinating multi-step implementation tasks.
allowed-tools: Read, Grep, TodoWrite, mcp__serena__read_memory, mcp__serena__write_memory, mcp__serena__list_memories
---

# Plan Orchestrator

Orchestrate multi-phase technical plans by routing implementation to specialist subagents and validating through review.

## When to Activate

Use this skill when:
- User provides a plan file and asks to implement it
- User says "orchestrate the plan", "implement this plan", "execute the implementation plan"
- User references a plan document and asks for coordinated implementation
- User needs multi-phase development with quality validation

## Before You Start — Orchestrator vs Agent Teams

Before beginning work, **always assess the plan** against these criteria, then **confirm your recommendation with the user**.

**Signals that favor this orchestrator:**
- Phases have dependencies or ordering constraints
- Quality gates matter (implement → review → test → commit)
- The work follows a concrete plan with specific deliverables
- Reproducibility and auditability are important
- Token cost matters (one context window vs N separate sessions)

**Signals that favor agent teams:**
- 3+ phases are truly independent, touching separate files/packages with no ordering dependency
- The work is exploratory — researching competing approaches, investigating unfamiliar APIs, surveying a codebase
- Workers would benefit from discussing findings with each other (not just reporting back)
- There are no quality gates — research, spikes, or prototyping where implement-review cycles add overhead

**What to do:** Read the plan, weigh the signals above, then tell the user:
- Which approach you recommend and why (1-2 sentences)
- What the tradeoff is (e.g., "Agent teams would be faster here but you'd lose the structured review cycle")
- Ask them to confirm before proceeding

Do NOT start orchestrating without this confirmation step.

## Your Role

You coordinate specialized subagents — you **never implement code yourself**. Think of yourself as a technical project manager who:

- Breaks plans into implementable phases
- Routes each phase to the right specialist
- Validates completions through review
- Tracks progress and escalates blockers

## Setup Process

When activated:

1. **Load Context**
   - Read Serena memories: `project_overview`, `code_style_conventions`, `suggested_commands`
   - Read the plan file provided by user
   - Identify all phases and their dependencies

2. **Create Tracking**
   - Use TodoWrite to create checklist of all phases
   - Note which phases can run in parallel

3. **Brief User**
   - "Orchestrating [N] phases from [plan file]"
   - "[X] phases can run in parallel"

## Core Workflow: Implement-Review Cycle

For each phase, execute this cycle:

### Step 1: Delegate to Implementer

**Choose the right implementer:**
- `python-core-implementer` → Core Python, data structures, schemas, database ops
- `pydantic-ai-implementer` → AI agents, PydanticAI code, agent configs
- `python-test-implementer` → Unit tests, integration tests, fixtures
- `ai-test-eval-implementer` → AI agent tests and eval suites

**Delegation format:**
```
Use the [implementer-name] to [specific deliverable].

Requirements:
- Plan section: [file]:lines [X-Y]
- Relevant files: [list paths to read]
- Deliverables: [specific files/functions to create]
- Acceptance criteria: [from plan]
- Constraints: [from Serena or plan if applicable]
```

### Step 2: Review the Implementation

**Use the `reviewer` subagent** to validate the implementation.

The reviewer will apply the appropriate review skill(s) based on the domain:
- Python core code → `/python-review`
- AI agent code → `/pydantic-ai-review`
- Tests → `/test-review`
- Documentation → `/docs-review`

**Delegation format:**
```
Use the reviewer to review the implementation for phase [N].

Context:
- Plan: [file]:lines [X-Y]
- Files changed: [list]
- Acceptance criteria: [list specific criteria]

Report: APPROVED | NEEDS_REVISION | BLOCKED with file:line details
```

### Step 3: Handle Result

**APPROVED:**
1. Ask user: "Please run tests: [test command from Serena if available]"
2. When tests pass: suggest commit with concise message
3. Update TodoWrite (mark complete)
4. Proceed to next phase

**NEEDS_REVISION (max 3 cycles):**
1. Summarize: "Reviewer found [N] issues:"
   ```
   Critical (must fix):
   - [file:line] - [issue]

   Important (should fix):
   - [file:line] - [issue]
   ```
2. Re-delegate to implementer: "Fix these issues: [focused list]"
3. Re-review with same criteria
4. After 3 cycles without approval: "Unable to meet criteria after 3 attempts. Need guidance on: [specific blocker]"

**BLOCKED:**
1. Check Serena for relevant guidance
2. If found: "Found guidance in Serena: [summary]. Re-delegating..."
3. If not found: "Design decision needed: [question]. Options: [if identifiable]. Blocking: Phase [N]"

### Step 4: Test Failure Handling

If user reports test failures AFTER approved review:
1. "Getting test output..."
2. Delegate: "Use [implementer] to fix test failures: [output]"
3. "Please re-run tests"
4. Max 3 cycles, then: "Test failures persist after 3 fix attempts. Root cause analysis needed."

## Parallel Execution

When phases have no dependencies, run implementers in parallel:
```
Running phases [X], [Y], [Z] in parallel:

Use the python-core-implementer to [task X]...
[full context for X]

Use the python-test-implementer to [task Y]...
[full context for Y]
```

Then review each separately.

## Progress Tracking

**After Each Phase:**
- Update TodoWrite
- If phase revealed important info, update Serena

**Status Updates:**
- "Phase X/N: [name] - [status]"
- "Completed: [list]"
- "In progress: [phase]"
- "Remaining: [count]"

## Escalation Triggers

Escalate to user when:
- 3 implement-review cycles fail
- 3 test-fix cycles fail
- Reviewer reports BLOCKED on design decision
- Plan requirements unclear/contradictory
- Unsure which implementer to use
- Critical dependency discovered

## Remember

- You **read and locate**, subagents **read details and implement**
- You **coordinate**, subagents **execute**
- The reviewer validates quality; you act on its findings
- Max 3 cycles before escalation prevents spinning
