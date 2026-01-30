# Phase 4.7: AI Separation Cleanup

**Status:** ✅ COMPLETE (2026-01-17)

**Goal:** Complete the AI separation by moving misplaced tests and removing config cruft

## Problem Statement

Phase 4 (AI Separation) was marked complete but audit reveals:
1. **5 test files** in `packages/findingmodel/tests/` import from `findingmodel_ai` - they belong in `findingmodel-ai/tests/`
2. **8 config settings** in `findingmodel/config.py` are defined but never used in findingmodel core
3. **findingmodel-ai/tests/** is essentially empty (only `__init__.py`)

---

## Sub-Phase 4.7.1: Move Test Files to findingmodel-ai

### Files to Move

| Source | Destination | Test Count |
|--------|-------------|------------|
| `packages/findingmodel/tests/test_finding_enrichment.py` | `packages/findingmodel-ai/tests/` | 43 tests |
| `packages/findingmodel/tests/test_model_editor.py` | `packages/findingmodel-ai/tests/` | 8 tests |
| `packages/findingmodel/tests/test_ontology_search.py` | `packages/findingmodel-ai/tests/` | 15+ tests |
| `packages/findingmodel/tests/test_tools.py` | `packages/findingmodel-ai/tests/` | 20+ tests |
| `packages/findingmodel/tests/tools/test_evaluators.py` | `packages/findingmodel-ai/tests/tools/` | 20 tests |

### Create conftest.py for findingmodel-ai

**File:** `packages/findingmodel-ai/tests/conftest.py`

Must include these fixtures (from findingmodel conftest.py):
- `configure_test_logging` - session-scoped logfire setup
- `base_model` - FindingModelBase instance
- `full_model` - FindingModelFull instance
- `real_model` - loads `data/pulmonary_embolism.fm.json`
- `real_model_markdown` - loads `data/pulmonary_embolism.md`
- `finding_info` - FindingInfo instance
- Model constants: `TEST_OPENAI_MODEL`, `TEST_ANTHROPIC_MODEL`, `TEST_GOOGLE_MODEL`

Fixtures NOT needed (findingmodel-specific):
- `tmp_defs_path` - for index building tests
- `prebuilt_db_path` - for DuckDB index tests
- `pe_fm_json`, `tn_fm_json`, `tn_markdown` - only used by findingmodel tests

### Copy Test Data Files

**Create:** `packages/findingmodel-ai/tests/data/`

**Required files:**
- `test_enrichment_samples.json` - used by test_finding_enrichment.py
- `pulmonary_embolism.fm.json` - used by real_model fixture
- `pulmonary_embolism.md` - used by real_model_markdown fixture

### Update Test File Imports

After moving, verify imports work correctly:
- `from conftest import ...` should resolve to new conftest
- Test data paths may need adjustment (use `Path(__file__).parent / "data"`)

### Create tools/ Subdirectory

**Create:** `packages/findingmodel-ai/tests/tools/__init__.py`

Move `test_evaluators.py` to this subdirectory.

---

## Sub-Phase 4.7.2: Clean findingmodel Config

### Settings to REMOVE (never used in findingmodel core)

| Setting | Reason for Removal |
|---------|-------------------|
| `anthropic_api_key` | Never referenced in findingmodel source |
| `google_api_key` | Never referenced in findingmodel source |
| `tavily_api_key` | Only used by `check_ready_for_tavily()` which is called from findingmodel-ai |
| `tavily_search_depth` | Never referenced in findingmodel source |
| `bioontology_api_key` | Never referenced in findingmodel source |
| `logfire_token` | Never referenced anywhere |
| `disable_send_to_logfire` | Never referenced anywhere |
| `logfire_verbose` | Never referenced anywhere |

### Method to REMOVE

- `check_ready_for_tavily()` - Already exists in findingmodel-ai/config.py, just delete from findingmodel

### Settings to KEEP (actively used)

| Setting | Used In |
|---------|---------|
| `openai_api_key` | `tools/duckdb_utils.py` for embeddings |
| `openai_embedding_model` | `tools/duckdb_utils.py` |
| `openai_embedding_dimensions` | `tools/duckdb_utils.py`, `index.py` |
| `duckdb_index_path` | `index.py` |
| `remote_index_db_url` | `ensure_index_db()` |
| `remote_index_db_hash` | `ensure_index_db()` |
| `remote_manifest_url` | `ensure_index_db()` |

### Backwards Compatibility

`ConfigurationError` is imported by findingmodel-ai:
```python
from findingmodel.config import ConfigurationError
```

This import must continue to work. Keep `ConfigurationError` in findingmodel.

---

## Sub-Phase 4.7.3: Delete Original Test Files

After verifying tests pass in new location, delete from findingmodel:
- `packages/findingmodel/tests/test_finding_enrichment.py`
- `packages/findingmodel/tests/test_model_editor.py`
- `packages/findingmodel/tests/test_ontology_search.py`
- `packages/findingmodel/tests/test_tools.py`
- `packages/findingmodel/tests/tools/test_evaluators.py`
- `packages/findingmodel/tests/tools/__init__.py` (if empty after move)

---

## Verification

### Step 1: Run findingmodel-ai tests only
```bash
uv run --package findingmodel-ai pytest packages/findingmodel-ai/tests/ -v
```
Expected: All moved tests pass (~100 tests)

### Step 2: Run findingmodel tests only
```bash
uv run --package findingmodel pytest packages/findingmodel/tests/ -v
```
Expected: Remaining tests pass (~230 tests, down from 337)

### Step 3: Run all package tests
```bash
task test
```
Expected: All tests pass (total count should remain ~498)

### Step 4: Verify no findingmodel_ai imports in findingmodel
```bash
grep -r "from findingmodel_ai" packages/findingmodel/src/
grep -r "from findingmodel_ai" packages/findingmodel/tests/
# Both should return nothing
```

### Step 5: Verify config is clean
```bash
grep -E "anthropic_api_key|google_api_key|tavily_|logfire_|bioontology" packages/findingmodel/src/findingmodel/config.py
# Should return nothing
```

### Step 6: Run checks
```bash
task check
```
Expected: All checks pass

---

## Acceptance Criteria

- [x] 5 test files moved to findingmodel-ai/tests/
- [x] findingmodel-ai/tests/conftest.py created with required fixtures
- [x] findingmodel-ai/tests/data/ created with required test files
- [x] findingmodel-ai/tests/tools/ created for test_evaluators.py
- [x] Original test files deleted from findingmodel/tests/
- [x] 8 unused config settings removed from findingmodel/config.py
- [x] `check_ready_for_tavily()` removed from findingmodel/config.py
- [x] All tests pass in both packages (333 passed, 4 skipped)
- [x] All checks pass
- [x] No findingmodel_ai imports remain in findingmodel source or tests

---

## Files Summary

### Create:
- `packages/findingmodel-ai/tests/conftest.py`
- `packages/findingmodel-ai/tests/data/test_enrichment_samples.json` (copy)
- `packages/findingmodel-ai/tests/data/pulmonary_embolism.fm.json` (copy)
- `packages/findingmodel-ai/tests/data/pulmonary_embolism.md` (copy)
- `packages/findingmodel-ai/tests/tools/__init__.py`

### Move:
- `test_finding_enrichment.py` → findingmodel-ai/tests/
- `test_model_editor.py` → findingmodel-ai/tests/
- `test_ontology_search.py` → findingmodel-ai/tests/
- `test_tools.py` → findingmodel-ai/tests/
- `tools/test_evaluators.py` → findingmodel-ai/tests/tools/

### Modify:
- `packages/findingmodel/src/findingmodel/config.py` - remove 8 settings + method

### Delete:
- Original test files from findingmodel/tests/ (after verification)
- `packages/findingmodel/tests/tools/__init__.py` (if empty)
