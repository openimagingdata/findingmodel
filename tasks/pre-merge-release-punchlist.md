# Pre-Merge Release Punchlist

Date: 2026-01-29
Branch: `refactor-multiple-packages` → `dev`
Sources: Claude Code review, Codex automated review, Gemini automated review

## Status: Ready to merge after completing this punchlist

The monorepo restructuring is **structurally complete** — package layout, dependency graph, build system, CLI entry points, and test infrastructure all match the plan. These are the remaining items to address before merge.

---

## 1. Update README examples to use new module paths

**Severity: High** — current examples will raise `ModuleNotFoundError`

All references to `findingmodel_ai.tools` must be updated to the new module layout (`findingmodel_ai.authoring`, `findingmodel_ai.search`, `findingmodel_ai.enrichment`).

**Files:**
- `README.md:58` — `from findingmodel_ai.tools import create_info_from_name`
- `packages/findingmodel-ai/README.md:59` — `create_info_from_name, add_details_to_info`
- `packages/findingmodel-ai/README.md:80` — `create_model_from_markdown, create_info_from_name`
- `packages/findingmodel-ai/README.md:103` — `edit_model_natural_language, edit_model_markdown`
- `packages/findingmodel-ai/README.md:128` — `find_anatomic_locations`
- `packages/findingmodel-ai/README.md:149` — `match_ontology_concepts`
- `packages/findingmodel-ai/README.md:168` — `find_similar_models`

---

## 2. Update `docs/` to reflect new package boundaries

**Severity: Medium** — documentation misleads users about CLIs and imports

### `docs/anatomic-locations.md`
- Lines 10, 54, 101: `from findingmodel import AnatomicLocationIndex` → should use `anatomic_locations` package
- Lines 24-46: `python -m findingmodel anatomic ...` → should use `anatomic-locations` CLI

### `docs/database-management.md`
- Lines 67-116: `python -m findingmodel index build/update/publish` → now `oidm-maintain` CLI
- Lines 146-183: `python -m findingmodel anatomic build/validate/stats` → now `oidm-maintain` or `anatomic-locations` CLI

### `docs/duckdb-development.md`
- Line 94: `from findingmodel.tools.duckdb_utils import setup_duckdb_connection` → now `oidm_common.duckdb`

---

## 3. Fix stale docstrings in `index.py`

**Severity: Low** — confusing but not breaking

Three docstring examples reference `await index.setup()` which no longer exists on the read-only `DuckDBIndex`:
- `packages/findingmodel/src/findingmodel/index.py:150`
- `packages/findingmodel/src/findingmodel/index.py:758`
- `packages/findingmodel/src/findingmodel/index.py:810`

Replace with `async with Index() as index:` pattern or simply remove the `setup()` line.

---

## 4. Update planning documents to reflect current state

**Severity: Medium** — internal docs should be accurate before we freeze the branch

- `tasks/monorepo-plan-overview.md` — Phase 6 status shows pending but is complete; update to match
- `tasks/monorepo-plan-overview.md` — example test command uses single-invocation pattern that doesn't work; update to show per-package `task test:*` pattern
- Ensure all phase docs reflect final decisions (e.g., YAGNI on `OIDM_MAINTAIN_` prefix)

---

## 5. Clean up tasks directory

**Severity: Low** — housekeeping before merge

- Delete `tasks/codex-merge-readiness-analysis.md` and `tasks/gemini-merge-readiness-analysis.md` (findings are consolidated into this punchlist and the post-merge issues doc)
- Move `tasks/monorepo-plan-overview.md` and `tasks/monorepo_plan_phases/` to `tasks/done/`

---

## 6. Resolve missing `scripts/release.py`

**Severity: Low** — `task release*` commands will error, but not user-facing

The Taskfile defines four `release` targets that invoke `scripts/release.py`, which does not exist. Either:
- (a) Create a stub `release.py` that prints "not yet implemented", or
- (b) Remove/comment the four release tasks from `Taskfile.yml` until the script is ready

---

## Verification

After completing all items, run:
```bash
task test        # all unit tests pass (594 expected)
task check       # ruff format + lint + mypy clean
```

---

## Appendix: Detailed PyPI Release Plan

### Current State

| Package | PyPI Status | Current PyPI Version | Branch Version | Notes |
|---------|------------|---------------------|----------------|-------|
| `findingmodel` | **Exists** | 0.6.0 | 1.0.0 | Major version bump — breaking: monorepo split, read-only index, AI tools removed |
| `oidm-common` | **New** | — | 0.2.0 | First publish; shared infrastructure extracted from findingmodel |
| `anatomic-locations` | **New** | — | 0.2.0 | First publish; anatomic location queries extracted from findingmodel |
| `findingmodel-ai` | **New** | — | 0.2.0 | First publish; AI tools extracted from findingmodel |
| `oidm-maintenance` | **Not published** | — | 0.2.0 | Internal only; not in build/publish pipeline |

### Breaking Changes for `findingmodel` 0.6.0 → 1.0.0

Users upgrading from `findingmodel` 0.6.0 will encounter:
1. **Removed modules**: `findingmodel.tools` AI functions are gone — users must `pip install findingmodel-ai` and import from `findingmodel_ai.authoring`, `.search`, `.enrichment`
2. **Removed CLI commands**: `findingmodel index build/update/publish` and `findingmodel anatomic *` are gone — build/publish is now `oidm-maintain` (internal only)
3. **Read-only Index**: `DuckDBIndex` no longer has `setup()`, `build()`, or write methods
4. **New dependency**: `findingmodel` now depends on `oidm-common` (installed automatically)

### Publish Order (strict — each step depends on the previous)

The dependency graph requires publishing in this exact order:

```
Step 1: oidm-common        (no internal deps)
Step 2: findingmodel        (depends on oidm-common)
        anatomic-locations  (depends on oidm-common)  — can publish in parallel with findingmodel
Step 3: findingmodel-ai     (depends on findingmodel + anatomic-locations)
```

This matches the existing `task publish:pypi` target in the Taskfile.

### Pre-Publish Checklist

1. **Merge to `dev`** — complete the punchlist above first
2. **Merge `dev` → `main`** — release from main branch
3. **Run verification script**:
   ```bash
   task verify:install
   ```
   This builds all packages with `--no-sources`, installs each in isolation via `uv run --no-project --find-links dist/`, and validates imports, CLI entry points, and database access.
4. **Confirm version numbers** — all `pyproject.toml` versions match intended release:
   - `oidm-common` == 0.2.0
   - `findingmodel` == 1.0.0
   - `anatomic-locations` == 0.2.0
   - `findingmodel-ai` == 0.2.0
5. **Confirm dependency version pins** — each package's `pyproject.toml` must pin its internal dependencies to the exact versions being published (not workspace references). Check that `[tool.uv.sources]` workspace overrides are stripped during build (uv does this automatically with `--no-sources`).
### Publish Execution

```bash
# Build all packages (strips workspace references)
task build:packages

# Publish in dependency order
task publish:pypi
```

This runs (from `Taskfile.yml`):
```
uv publish dist/oidm_common-*.tar.gz dist/oidm_common-*.whl
uv publish dist/findingmodel-[0-9]*.tar.gz dist/findingmodel-[0-9]*.whl
uv publish dist/anatomic_locations-*.tar.gz dist/anatomic_locations-*.whl
uv publish dist/findingmodel_ai-*.tar.gz dist/findingmodel_ai-*.whl
```

Requires `UV_PUBLISH_TOKEN` or `UV_PUBLISH_USERNAME`/`UV_PUBLISH_PASSWORD` environment variables set for PyPI authentication.

### Post-Publish Verification

```bash
# Verify from real PyPI (in a clean venv or with --no-project)
uv run --no-project --with findingmodel==1.0.0 -- findingmodel --help
uv run --no-project --with anatomic-locations==0.2.0 -- anatomic-locations stats
uv run --no-project --with findingmodel-ai==0.2.0 -- findingmodel-ai --help
```

### Post-Publish Tasks

1. **Git tag** the release: `v1.0.0` (use findingmodel's version as the monorepo tag)
2. **GitHub release** with changelog noting the monorepo split and migration guide
3. **Update README** install instructions: `pip install findingmodel` for core, `pip install findingmodel-ai` for AI tools
4. **Notify users** of breaking changes and new package names
