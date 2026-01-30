# Issues to Submit for Immediate Fix Post-Merge

Date: 2026-01-29
Branch: `refactor-multiple-packages`
Sources: Claude Code review, Codex automated review, Gemini automated review

These items should be tracked as issues and addressed promptly after merge. None are merge-blockers, but all represent real gaps that should be closed before the first release from the new structure.

---

## Issue 1: Implement missing anatomic-locations CLI commands

**Priority: Highest** — documented features don't exist

README and docs reference `search`, `hierarchy`, `children` subcommands for the `anatomic-locations` CLI. The actual CLI only exposes:
- `stats`
- `query ancestors`
- `query descendants`
- `query laterality`
- `query code`

**Fix:** Implement the missing commands to match documented behavior, or update documentation to match the actual CLI if the scope has changed.

---

## Issue 2: `findingmodel-ai` unit test can trigger network download

**Priority: High** — violates test isolation, will break clean CI

`packages/findingmodel-ai/tests/test_tools.py:37` calls `findingmodel.tools.add_ids_to_model()` which instantiates `Index()` without a db path. This triggers `ensure_index_db()` which can fetch a remote manifest and download the database.

**Fix:** Mock `Index()` or inject a local test db path so `test_add_ids_to_model` never hits the network. Confirm the test is excluded from `task test` (no-callout) runs.

---

## Issue 3: Phase 4.9 refactor cleanup items

**Priority: Medium** — code quality debt documented in `tasks/monorepo_plan_phases/phase-4.9-refactor-cleanup.md`

Tracked items:
1. **Embedding helper duplication** — `_internal/common.py` has embedding logic that overlaps with `oidm_common`
2. **`strip_quotes` duplication** — utility exists in multiple packages
3. **`evals/__init__.py` LOGFIRE_CONSOLE** — can be simplified
4. **`finding_description.py` OPENAI_API_KEY workaround** — needs documentation or proper fix

---

## Issue 4: Remove unused duplicate test data file

**Priority: Low** — cleanup

`packages/findingmodel/tests/data/test_enrichment_samples.json` is identical to `packages/findingmodel-ai/tests/data/test_enrichment_samples.json` and is not referenced by any test in the `findingmodel` package (verified via grep).

**Fix:** Delete `packages/findingmodel/tests/data/test_enrichment_samples.json`.

---

## Issue 5: Cross-package conftest collision

**Priority: Low** — workaround exists, but should be documented

Running `pytest packages/*/tests` in a single invocation fails with `ImportPathMismatchError` because multiple `tests/conftest.py` files exist. The Taskfile correctly runs each package separately, but:

1. The plan doc's example test command (`tasks/monorepo-plan-overview.md`) should be updated to show per-package invocation
2. Consider adding a top-level `conftest.py` comment or a note in CONTRIBUTING docs explaining why single-invocation doesn't work

---

## Issue 6: Plan phase status alignment

**Priority: Low** — internal documentation consistency

`tasks/monorepo-plan-overview.md` shows Phase 6 as pending, but the Phase 6 file marks it as complete and the repo includes `.claude/rules/*` and updated `CLAUDE.md`. Align the plan status to avoid confusion.
