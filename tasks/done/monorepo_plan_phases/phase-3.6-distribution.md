# Phase 3.6: Distribution Code Cleanup

**Status:** ✅ COMPLETE

**Goal:** Remove duplicate distribution code from findingmodel, create anatomic-locations config

## Completed Work

- Created `anatomic_locations/config.py` with:
  - `AnatomicLocationSettings` class (pydantic-settings, ANATOMIC_ prefix)
  - `get_settings()` singleton function
  - `ensure_anatomic_db()` using `oidm_common.distribution.ensure_db_file()`
- Updated anatomic-locations imports:
  - Removed ALL findingmodel imports from anatomic-locations package
  - Direct imports from `anatomic_locations.config` (no fallbacks)
  - Changed logger to use loguru directly
  - Changed OpenAI API key to use `os.getenv("OPENAI_API_KEY")` directly
- Cleaned findingmodel/config.py (~275 lines removed):
  - Removed duplicate distribution functions
  - Removed `ensure_anatomic_db()` (now in anatomic-locations)
  - Removed anatomic-related settings fields
  - Updated `ensure_index_db()` to use oidm-common
- Updated dependent code:
  - `findingmodel/tools/duckdb_search.py` → imports from anatomic_locations.config
  - `findingmodel/evals/anatomic_search.py` → imports from anatomic_locations.config
  - `findingmodel/tests/test_manifest_integration.py` → imports from oidm_common.distribution
- Created tests for `anatomic_locations/config.py` (10 tests)

## Key Achievement

**anatomic-locations now has ZERO imports from findingmodel - truly independent.**

## Verification

```bash
uv run --package oidm-common pytest         # ✅ 76 passed
uv run --package anatomic-locations pytest  # ✅ 90 passed (80 + 10 new)
uv run --package findingmodel pytest        # ✅ 460 passed
anatomic --help                             # ✅ Works without findingmodel fallback
ruff check packages/                        # ✅ All checks passed
```
