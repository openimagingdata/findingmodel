# Claude Subagents Modernization and Skill Portability Plan

**Date:** 2026-03-19

**Status:** Complete (2026-03-23)

## Goal

Modernize the legacy `.claude/agents/` setup so it reflects current March 2026 Claude Code practices, while also designing the new review skills so they can be reused by other agent environments where practical.

This plan focuses on:

- replacing the legacy matrix of stale specialist subagents with a much smaller current setup
- introducing reusable review skills
- deciding how much of those skills can be shared across Claude Code, OpenCode, and Codex

## Research Summary

### Claude Code

- Custom subagents are still a supported and recommended feature for focused, isolated work.
- Skills are now the preferred mechanism for reusable workflow instructions and can be combined with subagents.
- Skills can also fork into built-in agents such as `general-purpose`, `Explore`, or `Plan`.
- Agent teams exist, but they are a separate experimental feature intended for sustained multi-agent collaboration rather than simple local review delegation.

### OpenCode

- OpenCode explicitly supports skill discovery from `.claude/skills/<name>/SKILL.md`.
- OpenCode documents a `SKILL.md` frontmatter schema of:
  - `name`
  - `description`
  - `license`
  - `compatibility`
  - `metadata`
- Unknown frontmatter fields are ignored by OpenCode.

### Codex

- The local Codex environment in use here also uses `SKILL.md` skills.
- Its local validator currently accepts a narrower frontmatter schema:
  - `name`
  - `description`
  - `license`
  - `allowed-tools`
  - `metadata`
- This means the overlap between OpenCode and the current local Codex skill schema is:
  - `name`
  - `description`
  - `license`
  - `metadata`

## Key Decisions

### 1. Retire the worst legacy agent immediately

Retire `.claude/agents/code-refactoring-analyzer.md`.

Reason:

- It encodes stale repo architecture and outdated tool assumptions.
- It is the farthest from current Claude Code expectations.
- It would cost more to salvage than to replace.

### 2. Stop using a matrix of narrow evaluator/implementer subagents

Do not keep the current pattern of many narrowly scoped implementer/evaluator subagents.

Replace it with:

- one read-only `reviewer` subagent
- a set of reusable review skills
- the existing orchestration logic updated to call the smaller modern surface

### 3. Put rubrics and workflow guidance into skills

Treat the following domains as skills rather than dedicated subagents:

- Python/core review
- Pydantic AI review
- test review
- documentation review

These are fundamentally reusable review rubrics and procedures, not distinct execution environments.

### 4. Keep the subagent simple and read-only

The new reviewer subagent should:

- be read-only in both intent and tool access
- focus on findings, risks, gaps, and validation suggestions
- avoid implementation work
- preload review skills instead of embedding long project-specific checklists directly into the subagent file

## Target State

### New Claude surface

#### Subagent

Create:

- `.claude/agents/reviewer.md`

Characteristics:

- read-only
- minimal tool set such as `Read`, `Grep`, `Glob`, `Bash`
- focused description for automatic delegation
- findings-first output contract
- preload review skills if Claude subagent frontmatter continues to support that cleanly

#### Skills

Create:

- `.claude/skills/python-review/SKILL.md`
- `.claude/skills/pydantic-ai-review/SKILL.md`
- `.claude/skills/test-review/SKILL.md`
- `.claude/skills/docs-review/SKILL.md`

Skill bodies should contain:

- review scope
- severity rubric
- review checklist
- known repo-specific standards
- reporting format

### Updated orchestration

Update `.claude/skills/plan-orchestrator/SKILL.md` so it no longer depends on the old implementer/evaluator matrix for review steps.

At minimum it should be able to:

- route review work to the new `reviewer` subagent
- reference the appropriate review skills
- avoid assuming the presence of the retired legacy agents

## Portability Strategy

## Principle

Write **portable core review skill content** first, then add platform-specific glue only where necessary.

### What can be shared directly

The review rubric content itself should be portable across Claude Code, OpenCode, and Codex if we keep the shared skill files conservative.

To maximize direct portability, shared skill frontmatter should use only the known common subset:

- `name`
- `description`
- optional `license`
- optional `metadata`

### What should stay platform-specific

Do **not** put platform-specific execution behavior into the canonical portable skill content unless we are willing to maintain wrappers or mirrors.

Examples of behavior that may need platform-specific handling:

- Claude-only skill routing such as forked execution into built-in agents
- Codex-specific `allowed-tools`
- OpenCode-specific `compatibility`

### Recommended portability implementation

Phase 1:

- Author the canonical review skill content in `.claude/skills/` using only the shared frontmatter subset.
- Confirm Claude Code still gets enough value from the body content even without richer skill frontmatter.
- Rely on the `reviewer` subagent definition for tool restrictions and review execution behavior.

Phase 2:

- Verify OpenCode loads the same `.claude/skills/*/SKILL.md` files without modification.
- If needed, add OpenCode-only metadata only after verifying it does not create portability cost we care about.

Phase 3:

- If Codex reuse is desired in practice, either:
  - mirror the same portable skills into the Codex skill location, or
  - generate Codex-compatible wrappers from the same canonical source
- Do not assume literal drop-in compatibility for richer Claude skill frontmatter.

## Execution Plan

### Phase 1: Inventory and decisions

1. Record the final disposition of each file in `.claude/agents/`:
   - retire
   - replace with skill
   - replace with reviewer subagent
2. Remove the clearly obsolete agent first:
   - `code-refactoring-analyzer.md`

### Phase 2: Define the new reviewer subagent

1. Create a single new read-only `reviewer` subagent.
2. Give it a concise description that makes delegation obvious:
   - recent changes
   - regressions
   - missing tests
   - stale docs
   - maintainability risks
3. Keep the prompt findings-first and review-only.
4. Ensure the tool set stays read-only.

### Phase 3: Extract review rubrics into skills

1. Create the four review skills:
   - `python-review`
   - `pydantic-ai-review`
   - `test-review`
   - `docs-review`
2. Move rubric content out of the legacy evaluator files into those skills.
3. Keep each skill short, focused, and discoverable.
4. Move lengthy standards/examples into referenced files only if needed.

### Phase 4: Update orchestration

1. Rewrite `plan-orchestrator` to use the new reviewer path.
2. Remove dependencies on retired evaluator/implementer agent names.
3. Decide whether review invocation should happen by:
   - calling the `reviewer` subagent directly, or
   - invoking a forked review skill from the main context
4. Prefer the simpler option unless a real workflow gap appears.

### Phase 5: Portability verification

1. Verify OpenCode discovery from `.claude/skills/`.
2. Validate the portable skill frontmatter against the current local Codex validator.
3. If Codex portability is still wanted, create a follow-up wrapper/mirroring step rather than polluting the canonical skill files with platform-specific keys.

### Phase 6: Cleanup and docs

1. Delete or archive replaced legacy agent files.
2. Update any internal docs that still mention the legacy subagent matrix.
3. Mark this plan complete and note any unresolved portability caveats.

## Acceptance Criteria

- The repo no longer depends on the legacy matrix of evaluator/implementer subagents.
- A single read-only reviewer subagent exists and is clearly scoped.
- Review knowledge is encoded as reusable skills rather than long monolithic subagent prompts.
- The canonical review skills are written with a frontmatter subset that is portable to OpenCode and compatible with local Codex validation.
- The plan orchestrator no longer references retired agent names.
- The old refactoring analyzer is removed.

## Deliverables

- new `.claude/agents/reviewer.md`
- four review skills under `.claude/skills/`
- updated orchestration skill
- retired legacy review/evaluator agent files
- portability notes captured in repo docs or completion notes

## Follow-Up Questions — Resolved

1. **Reviewer model:** Uses `model: sonnet`. Confirmed during implementation.
2. **Fork behavior:** The orchestrator invokes the reviewer subagent directly; review skills provide the rubric content, not execution behavior.
3. **Codex reuse:** Clean path only for now. Review skill frontmatter uses the portable subset (`name`, `description`). No Codex-specific wrappers created.

## Completion Notes (2026-03-23)

### What was delivered (original scope)

- **Deleted 6 agents**: `code-refactoring-analyzer`, `python-core-evaluator`, `pydantic-ai-evaluator`, `ai-test-evaluator`, `python-test-evaluator`, `documentation-updater`
- **Created `reviewer.md`**: Single read-only reviewer subagent (sonnet model), references review skills
- **Created 4 review skills**: `python-review`, `pydantic-ai-review`, `test-review`, `docs-review` — all with portable frontmatter (name + description only)
- **Rewrote `plan-orchestrator`**: Uses reviewer instead of evaluator matrix

### Additional work beyond original scope

- **Renamed `ai-test-implementer` → `ai-test-eval-implementer`**: Expanded to cover both pytest-based AI tests AND Pydantic Evals suites (evaluator hierarchy, hybrid scoring, 6 existing suites documented)
- **Updated all 4 implementer agents** with current project knowledge:
  - Correct monorepo layout (`packages/<pkg>/...`)
  - Current Serena memory names (removed stale references)
  - MongoDB references removed
  - loguru logging, pydantic-settings config patterns
  - Per-agent model config via `supported_models.toml`, FallbackModel chains
  - Logfire observability
- **Updated `pydantic-ai-implementer`** with Pydantic AI v1.70+ patterns: `instructions` vs `system_prompt`, toolsets, `description=`, `args_validator=`, output modes
- **Updated `pydantic-ai-review` skill** with v1.70+ API patterns
- **Researched Claude Code agent teams**: Assessed overlap with plan-orchestrator; concluded they serve complementary purposes
- **Added orchestrator guardrails**: "Before You Start" section that assesses orchestrator vs agent teams, recommends an approach, and confirms with user before proceeding

### Acceptance criteria status

- [x] No dependency on legacy evaluator/implementer matrix
- [x] Single read-only reviewer subagent exists
- [x] Review knowledge encoded as reusable skills
- [x] Portable frontmatter on all review skills
- [x] Plan orchestrator references only current agent names
- [x] Old refactoring analyzer removed

### Final inventory

**Agents (5):** `reviewer`, `pydantic-ai-implementer`, `python-core-implementer`, `python-test-implementer`, `ai-test-eval-implementer`

**Skills (6):** `python-review`, `pydantic-ai-review`, `test-review`, `docs-review`, `plan-orchestrator`, `manage-anatomic-locations`
