# Phase 5: Clean up findingmodel

**Status:** ⏳ PENDING

**Goal:** Remove extracted code, verify clean split

## Tasks

1. Remove tools/ directory (moved to findingmodel-ai)
2. Remove evals/ (moved to findingmodel-ai)
3. Update config.py to use oidm-common distribution
4. Remove AI-related dependencies from pyproject.toml
5. Run full test suite

## Verification

```bash
uv run --package findingmodel pytest
# pip install from built wheel works without AI deps
```
