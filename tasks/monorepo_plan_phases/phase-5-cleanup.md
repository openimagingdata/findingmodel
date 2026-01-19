# Phase 5: Clean up findingmodel

**Status:** ✅ COMPLETE

**Goal:** Remove extracted code, verify clean split

## Tasks

1. ✅ Remove AI tools from tools/ directory (non-AI utilities like `add_ids_to_model`, `add_standard_codes_to_model` remain)
2. ✅ Remove evals/ (moved to findingmodel-ai)
3. ✅ Update config.py to use oidm-common distribution (`ensure_db_file`, `strip_quotes`)
4. ✅ Remove AI-related dependencies from pyproject.toml (no pydantic-ai, tavily, anthropic)
5. ✅ Run full test suite (516 passed)

## Verification

```bash
uv run --package findingmodel pytest
# pip install from built wheel works without AI deps
```

## Completion Notes (2026-01-19)

- `tools/` directory retained for non-AI utilities; `__init__.py` documents that AI tools moved to findingmodel_ai
- Config uses `oidm_common.distribution.ensure_db_file` and `strip_quotes` from oidm-common
- Dependencies are minimal: oidm-common[openai] for embeddings only, no AI agent dependencies
