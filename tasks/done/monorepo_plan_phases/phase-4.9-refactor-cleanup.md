# Phase 4.9: Refactor Cleanup - Config and Duplication Issues

## Status: PLANNED

## Problem Statement

During the Phase 4.8 audit, we discovered several config misuse patterns and code duplication issues that need cleanup before the monorepo architecture is sound.

## Issues to Resolve

### 1. Remove `_internal/common.py` Embedding Duplication

**Location:** `packages/findingmodel-ai/src/findingmodel_ai/_internal/common.py`

**Problem:**
- Duplicates `get_embedding` and `get_embeddings_batch` that already exist in `oidm-common`
- Cross-imports `fm_settings` from `findingmodel.config` for `openai_embedding_model`
- Cross-imports `settings` from `findingmodel_ai.config` for `openai_api_key`

**Fix:**
1. Delete `get_embedding`, `get_embeddings_batch`, `_get_embedding_cache`, `_lookup_cached_embeddings`, `_fetch_and_store_embeddings` from `_internal/common.py`
2. Update callers to use `oidm_common.embeddings.get_embedding()` with explicit params
3. Remove the `from findingmodel.config import settings as fm_settings` import
4. Keep only `get_async_tavily_client` and `get_markdown_text_from_path_or_text` in `_internal/common.py`

**Callers to update:** Use `find_referencing_symbols` on `get_embedding` in `_internal/common.py`

### 2. Move `strip_quotes` to oidm-common

**Locations:**
- `packages/findingmodel/src/findingmodel/config.py` (lines 14-22)
- `packages/findingmodel-ai/src/findingmodel_ai/config.py` (lines 62-70)

**Problem:** Identical utility functions duplicated in two configs.

**Fix:**
1. Add `strip_quotes` and `strip_quotes_secret` to `oidm-common` (new module: `oidm_common/utils.py` or add to existing)
2. Update both config files to import from `oidm-common`
3. Verify with `find_symbol("strip_quotes")` that no other duplicates exist

### 3. Simplify `evals/__init__.py`

**Location:** `packages/findingmodel-ai/evals/__init__.py`

**Problem:** Directly reads `os.getenv("LOGFIRE_CONSOLE")` - invented env var, bypasses config.

**Fix:**
1. Remove `LOGFIRE_CONSOLE` feature entirely
2. Simplify to just:
   ```python
   logfire.configure(send_to_logfire="if-token-present", console=False)
   logfire.instrument_pydantic_ai()
   ```
3. Remove `import os`
4. Update docstring to remove mention of LOGFIRE_CONSOLE

### 4. Document `evals/finding_description.py` Workaround

**Location:** `packages/findingmodel-ai/evals/finding_description.py` (lines 69-70)

**Problem:** Sets `os.environ["OPENAI_API_KEY"]` from config as workaround for LLMJudge.

**Decision:** Keep as-is with better documentation. This is a necessary bridge between our config system and pydantic_evals' LLMJudge which reads directly from environment.

**Fix:**
1. Add comment explaining why this workaround exists
2. Consider opening issue with pydantic_evals about config-based API key support

### 5. Config Field Duplication - Intentional Design

**Not a problem to fix.**

Fields that appear in multiple configs:
- `openai_api_key` (4 places)
- `openai_embedding_model` (3 places)
- `openai_embedding_dimensions` (3 places)

**This is intentional.** In the future, `findingmodel` and `anatomic-locations` should be able to independently specify their embedding model/dimensions. Having separate config fields (even if currently named the same and reading from the same env vars) makes this easier to implement later.

Each package maintains its own config for independence - this is correct architecture.

## Implementation Order

1. **Step 1:** Move `strip_quotes` to oidm-common (low risk, isolated)
2. **Step 2:** Simplify `evals/__init__.py` (remove LOGFIRE_CONSOLE)
3. **Step 3:** Remove embedding duplication from `_internal/common.py` (higher risk, has callers)
4. **Step 4:** Document the finding_description.py workaround

## Verification

After each step:
```bash
task test                    # All tests pass
task check                   # No lint/type errors
```

After Step 3, also verify:
```bash
task evals:anatomic_search   # Evals still work (uses embeddings)
```

## Files to Modify

**oidm-common:**
- `src/oidm_common/utils.py` (new file for strip_quotes)
- `src/oidm_common/__init__.py` (export strip_quotes)

**findingmodel:**
- `src/findingmodel/config.py` (import strip_quotes from oidm-common)

**findingmodel-ai:**
- `src/findingmodel_ai/config.py` (import strip_quotes from oidm-common)
- `src/findingmodel_ai/_internal/common.py` (delete embedding functions)
- `evals/__init__.py` (remove LOGFIRE_CONSOLE)
- `evals/finding_description.py` (add documentation comment)

**Callers of _internal/common.py embedding functions:** TBD via `find_referencing_symbols`

## Risk Assessment

- **Low risk:** Steps 1, 2, 4 (isolated changes)
- **Medium risk:** Step 3 (need to update all callers, verify embedding behavior unchanged)
