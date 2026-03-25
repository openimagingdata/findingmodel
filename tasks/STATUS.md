# Task Status Overview

**Last Updated:** 2026-03-23

This file tracks the current state of active planning documents. Historical plans belong under `tasks/done/`.

## Project Health

- Monorepo restructuring is complete and historical monorepo planning docs are archived under `tasks/done/`.
- The branch currently contains completed per-agent model configuration work reflected in code and changelog.
- Active documentation is being refreshed via [docs/plans/documentation-audit-2026-03-17.md](../docs/plans/documentation-audit-2026-03-17.md).

## Current Focus

### Documentation audit and cleanup

- **Status**: Completed
- **Plan**: [docs/plans/documentation-audit-2026-03-17.md](../docs/plans/documentation-audit-2026-03-17.md)
- **Goal**: Update active docs to match the current codebase and archive material that is historical but still looks active.

### Claude subagents modernization and skill portability

- **Status**: Complete (2026-03-23)
- **Plan**: [claude-subagents-modernization-and-skill-portability-plan.md](claude-subagents-modernization-and-skill-portability-plan.md)
- **Result**: Replaced 10-agent evaluator/implementer matrix with 5 agents + 4 portable review skills. Added orchestrator-vs-agent-teams guardrails. See plan completion notes for details.

## Active Plans Requiring Revalidation

These plans predate major package and API changes or contain assumptions that should be rechecked before execution.

### [refactoring/02-api-cleanup.md](refactoring/02-api-cleanup.md)

- **Status**: Needs revalidation
- **Reason**: The proposed canonical renames do not match the current public API surface.

### [refactoring/01-index-decomposition.md](refactoring/01-index-decomposition.md)

- **Status**: Needs revalidation
- **Reason**: The plan still references pre-monorepo and pre-current-package layouts.

### [facets-implementation-plan.md](facets-implementation-plan.md)

- **Status**: Blocked pending design decisions
- **Spec**: [findingmodel-facets.md](findingmodel-facets.md)

## Active Plans Still Relevant

### [bioontology-oidm-common-plan.md](bioontology-oidm-common-plan.md)

- **Status**: Planned
- **Notes**: Current package-path references appear aligned with the monorepo layout.

### [pending_ty_upgrade.md](pending_ty_upgrade.md)

- **Status**: Blocked on external toolchain gaps

### [schema-versioning-future.md](schema-versioning-future.md)

- **Status**: Deferred

## Notes

- If an active plan is found to be historical during implementation, move it to `tasks/done/` or clearly mark it as superseded.
- Do not treat older status or punchlist docs as current without checking the codebase first.
