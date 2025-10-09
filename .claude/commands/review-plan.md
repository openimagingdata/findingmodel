---
argument-hint: <plan-file-path>
description: Assess plan file for clarity, completeness, alignment with best practices, and technical currency
---

# Plan Review Assessment

## Context

Plan to review: $1

**Project standards:** Auto-loaded from CLAUDE.md and Serena memories.

## Review Process

### Step 1: Check Historical Context

Use serena to search for:
- Related past decisions about similar features
- Existing architectural patterns we've established
- Previous plan reviews and outcomes
- Team preferences documented in memories

Document relevant findings.

### Step 2: Goal Assessment

- Are overall goals clearly stated with measurable deliverables?
- Do goals align with project objectives?
- Will achieving these goals enable necessary next steps?
- Are success criteria explicitly defined?

### Step 3: Work Breakdown Evaluation

- Are tasks broken into clear, independently assessable steps?
- Does each step have specific boundaries and success criteria?
- Can we measure completion of each step after implementation?
- Are dependencies between steps clearly identified?
- Is scope appropriate (not too large, not too granular)?

### Step 4: Technical Approach Validation

**Alignment:**
- Does approach align with CLAUDE.md standards?
- Are patterns consistent with established practices?

**Complexity (YAGNI):**
- Flag any over-engineering or unnecessary complexity
- Identify speculative features not tied to current requirements
- Question premature abstractions
- Highlight infrastructure without immediate use cases
- Default assumption: We will NOT need it unless proven otherwise

**Currency Verification (MANDATORY):**
Use web_search or perplexity to verify:
- Search "[technology/framework] best practices 2024-2025"
- Search "[specific pattern/approach] current recommendations"
- Verify dependencies/libraries are recommended and maintained
- Check APIs/methods aren't deprecated
- Compare approach against latest official documentation

Document your searches explicitly.

### Step 5: Completeness Check

- Error handling and edge cases addressed?
- Testing strategy defined?
- Performance considerations mentioned where relevant?
- Security addressed appropriately?
- Rollback or migration strategies if needed?

## Output

### Executive Summary
(2-3 sentences): Overall quality, confidence level, clear go/no-go recommendation.

### Critical Blockers
Issues preventing implementation:
- [Issue with specific plan section reference]
- [Impact and why blocking]

### Needs Clarification
Ambiguities requiring resolution:
- [Specific question or ambiguity]
- [Why clarification needed]

### Suggestions for Improvement
Non-blocking recommendations:
- [Suggestion with rationale]
- [Expected benefit]

### Best Practices Verification
Summary of web searches:
- Technologies/frameworks checked: [list]
- Current recommendations: [summary]
- Deviations from best practices: [if any]
- Deprecated approaches: [if any]

### Action Items
If changes needed:
1. [Specific, actionable item]
2. [Specific, actionable item]
