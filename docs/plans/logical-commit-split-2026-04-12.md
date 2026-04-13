# Logical Commit Split Plan

Date: 2026-04-12
Status: Complete

## Goal

Split the current worktree into a small number of reviewable commits that reflect real workstreams rather than a single branch-wide snapshot.

## Planned commits

1. `findingmodel` metadata type refactor
   - New `findingmodel.types` module structure
   - Removal of legacy `facets` / `finding_model` modules
   - Export and CLI/stub/index adjustments required by the refactor
   - Refactor-aligned `packages/findingmodel` tests
2. Gold-backed metadata assignment eval expansion
   - Eval harness, gold fixtures, task wiring, and assignment-mode coverage
3. Metadata prompt and gold-standard alignment
   - Prompt/example revisions, RSNA subspecialty alignment, ontology display handling, docs, and fixture refreshes
4. Optional follow-on authoring/editor cleanup
   - Only if it stands on its own after the first three commits are staged

## Execution notes

- Stage each commit and verify the staged diff is internally coherent.
- Keep cross-cutting docs and task wiring out of commit 1 unless they are required for the refactor to build or test.
- After staging each commit, review `git diff --cached --stat` and the staged patch before any commit is created.

## Current split state

- Commit 1 completed:
  - `70ae1db` `Refactor findingmodel types and align core metadata enums`
- Commit 2 completed:
  - `2fc5db3` `Expand metadata assignment evals and reassessment behavior`
- Commit 3 target:
  - prompt/example iteration
  - gold-review follow-up docs
  - changelog, benchmark updates, and remaining metadata assignment refinements

## Final staging update

- Commit 3 is the remaining bucket and consists mostly of:
  - prompt-review/reference docs
  - benchmark script changes
  - changelog and remaining rewrite-doc updates

## Documentation follow-up

- Update this plan as each commit bucket is staged.
- Update `CHANGELOG.md` only in the commit where user-visible behavior actually changes.
- After the final split is complete, review plan docs and related docs for any completion or archival updates.
