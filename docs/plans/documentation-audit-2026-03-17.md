# Plan: Documentation Audit and Staleness Cleanup

## Date

2026-03-17

## Status

Completed

## Goal

Audit the repository's active documentation against the current codebase and either:

- update documents that should remain active, or
- archive documents that are materially stale and no longer represent the current system.

## Scope

Active documentation and planning/status documents that influence current development or user behavior:

- `README.md`
- `docs/**/*.md`
- `packages/*/README.md`
- `tasks/STATUS.md`
- active `tasks/*.md` plans that are referenced as current work

Historical material already under `docs/archive/` or `tasks/done/` is out of scope unless it is incorrectly linked as current.

## Audit Criteria

A document is stale if any of the following are true:

- it documents commands, imports, config, or behavior that no longer exist
- it presents an implementation plan as current when the work is already complete or abandoned
- it conflicts with the current package boundaries or public API
- it names the wrong next steps or project status

## Execution Plan

1. Inventory active docs and compare them to current code, CLI surfaces, config models, and recent completed work.
2. Classify each stale document:
   - update in place if the topic is still active and useful
   - archive or clearly mark as historical if it no longer reflects the active system
3. Update cross-references so active docs do not point readers at stale material.
4. Refresh planning/status docs so they reflect the true current state after the audit.
5. Review `CHANGELOG.md` and related documentation trackers for any necessary follow-up note.
6. Verify representative commands, imports, and env var names referenced by the updated docs.

## Deliverables

- corrected active documentation
- archived or explicitly historical stale docs
- refreshed status/plan docs describing the real current state
- a short audit summary with any remaining follow-up items

## Completion Notes

Completed on 2026-03-17 after:

- updating active user-facing docs to match the current CLI/config/runtime behavior
- archiving historical enrichment and punchlist documents that were still sitting in active locations
- marking older active planning docs as requiring revalidation before use
