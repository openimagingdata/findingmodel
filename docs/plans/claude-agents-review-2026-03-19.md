# Plan: Review Legacy Claude Sub-Agents Against Current Claude Code Guidance

## Date

2026-03-19

## Status

Completed

## Goal

Review the legacy sub-agent definitions under `.claude/agents/` against current March 2026 Claude Code guidance and recommend whether to:

- keep them as-is,
- simplify or consolidate them,
- replace them with a newer Claude Code pattern, or
- retire them.

## Scope

- `.claude/agents/*.md`
- related local Claude config that affects their use
- current Anthropic / Claude Code documentation relevant to sub-agents, commands, skills, hooks, and task delegation

## Execution Plan

1. Inventory the existing `.claude/agents` definitions and classify what each is trying to achieve.
2. Research current March 2026 Claude Code best practices from primary sources.
3. Compare the local agent patterns to the current recommended mechanisms in Claude Code.
4. Identify overlaps, obsolete patterns, and gaps.
5. Produce a recommendation matrix:
   - keep
   - merge/simplify
   - replace with newer feature
   - retire
6. Update this plan with completion notes and any recommended follow-up doc changes.

## Deliverables

- a written assessment of each local sub-agent definition
- a recommendation for whether `.claude/agents` should remain part of the workflow
- concrete next steps for cleanup or migration if warranted

## Completion Notes

Completed on 2026-03-19 after:

- reviewing all current `.claude/agents/*.md` definitions and related `.claude` config
- comparing them against current Claude Code docs for subagents, skills, hooks, plugins, agent teams, settings, and tools
- confirming that custom subagents are still supported and still useful for focused, isolated work with constrained tools
- identifying that this repo's current subagent set is stale in both repo assumptions and Claude Code patterns

### Recommendation Summary

- Keep the concept of custom subagents, but not this exact inventory.
- Prefer a smaller set of focused subagents only where isolated context, tool restrictions, or persistent subagent memory adds clear value.
- Prefer skills for reusable workflows and procedural guidance, especially when the task should stay in the main conversation or should fork into built-in `Explore`/`Plan` agents.
- Prefer plugins only if the resulting components need to be shared across multiple repos or teammates beyond this project.
- Do not treat agent teams as the default replacement: they are useful for sustained parallel collaboration but remain experimental and add coordination overhead.

### Specific Follow-Up

1. Retire or rewrite `code-refactoring-analyzer.md` first; it contains the most outdated project and tool assumptions.
2. Consolidate the implementer/evaluator matrix into a much smaller set of current, repo-accurate agents if the team still finds subagent delegation valuable.
3. Move workflow-style behavior such as documentation and orchestration guidance toward skills, using `context: fork` with built-in agents where appropriate.
4. If the surviving components should be reused across repos, package them as a plugin after simplifying them.
