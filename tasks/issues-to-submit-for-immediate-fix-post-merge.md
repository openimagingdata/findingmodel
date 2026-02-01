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

## ~~Issue 2: `findingmodel-ai` unit test can trigger network download~~ ✓ RESOLVED

Resolved: tests that created bare `Index()` (triggering network downloads) were removed from `findingmodel-ai` and relocated to `findingmodel` with local test database fixtures. Three `test_model_editor.py` tests updated to use `index_with_test_db` fixture. No bare `Index()` calls remain in `findingmodel-ai/tests/`.

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

---

## Issue 7: Stop exporting `DuckDBIndex` from `findingmodel`

**Priority: Medium** — leaks implementation detail in public API

`findingmodel/__init__.py` exports both `Index` and `DuckDBIndex`. The public API should only expose `Index`; `DuckDBIndex` is an implementation detail. Remove the `DuckDBIndex` export and the `__all__` entry. Deprecate first if backward compatibility is a concern.

---

## Issue 8: Database builds don't use the embedding cache

**Priority: Medium** — wasteful and slow rebuilds

Both the finding model and anatomic location database build pipelines regenerate all embeddings from scratch on every rebuild, making expensive OpenAI API calls for unchanged data. `oidm-common` has an `EmbeddingCache` facility, but it is not wired into the build pipelines.

This subsumes the related issue in `tasks/pending-fixes.md` ("Anatomic Location DuckDB Rebuild - Preserve Embeddings"). The fix should cover both build pipelines:

1. Wire `EmbeddingCache` into `oidm-maintenance` build commands so previously computed embeddings are reused
2. Use hash-based detection to identify changed entries and only regenerate embeddings for new/modified data
3. Apply to both finding model index builds and anatomic location builds

---

## Issue 9: Update dependency tree

**Priority: High** — stale dependencies accumulate security and compatibility risk

The project's dependency tree needs a thorough update. Run `uv lock --upgrade`, review the changes, fix any breakage, and verify with `task test` and `task check`. Pay particular attention to major version bumps in core dependencies (pydantic, duckdb, httpx, pydantic-ai, etc.).
