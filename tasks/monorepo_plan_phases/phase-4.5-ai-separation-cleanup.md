# Phase 4.5: AI Separation Cleanup

**Status:** ✅ COMPLETE

**Goal:** Complete clean separation between findingmodel (core) and findingmodel-ai (AI tools)

## Architectural Principles

1. **findingmodel-ai is self-contained** - has its own settings, logger, CLI
2. **findingmodel is AI-agent-free** - no pydantic_ai Agent usage, only optional openai for embeddings
3. **No circular dependency** - findingmodel CANNOT depend on findingmodel-ai
4. **One-way dependency** - findingmodel-ai depends on findingmodel (for core models)

---

## Sub-Phase 4.5.1: Create findingmodel-ai Infrastructure

**Status:** ⏳ PENDING

Create settings and logger for findingmodel-ai:

1. **Create `packages/findingmodel-ai/src/findingmodel_ai/config.py`**
   - Copy relevant settings from findingmodel.config
   - AI model configuration (tiers, providers, agent tags)
   - API keys (OpenAI, Anthropic, Google, Tavily)
   - Remove non-AI settings (DuckDB paths, etc.)

2. **Create `packages/findingmodel-ai/src/findingmodel_ai/__init__.py`**
   - Set up logger for findingmodel_ai
   - Export settings, logger

3. **Update all findingmodel-ai tools to use local config/logger**
   - Change `from findingmodel.config import settings` → `from findingmodel_ai.config import settings`
   - Change `from findingmodel import logger` → `from findingmodel_ai import logger`

---

## Sub-Phase 4.5.2: Move Enrichment Code to findingmodel-ai

**Status:** ⏳ PENDING

Move the AI orchestration code:

1. **Move files:**
   - `findingmodel/tools/finding_enrichment.py` → `findingmodel_ai/tools/finding_enrichment.py`
   - `findingmodel/tools/finding_enrichment_agentic.py` → `findingmodel_ai/tools/finding_enrichment_agentic.py`

2. **Update imports in moved files:**
   - Use findingmodel_ai config/logger
   - Keep imports of core models from findingmodel

3. **Delete original files from findingmodel**

---

## Sub-Phase 4.5.3: Move AI CLI Commands to findingmodel-ai

**Status:** ⏳ PENDING

Create CLI for findingmodel-ai:

1. **Create `packages/findingmodel-ai/src/findingmodel_ai/cli.py`**
   - Move these commands from findingmodel CLI:
     - `make-stub-model` (uses create_stub)
     - `markdown-to-fm` (uses create_finding_model_from_markdown)
     - `describe` / `detail` (uses finding_description tools)
   - Keep findingmodel CLI for non-AI commands (stats, index queries)

2. **Add CLI entry point to findingmodel-ai pyproject.toml:**
   ```toml
   [project.scripts]
   fm-ai = "findingmodel_ai.cli:cli"
   ```

3. **Update findingmodel CLI to remove AI commands**

---

## Sub-Phase 4.5.4: Strip AI Dependencies from findingmodel

**Status:** ⏳ PENDING

Make findingmodel AI-agent-free:

1. **Update findingmodel/pyproject.toml dependencies:**
   - REMOVE: `pydantic-ai-slim`
   - KEEP (as optional): `openai` for embedding generation
   - KEEP: `pydantic`, `pydantic-settings` (core config)

2. **Update findingmodel/config.py:**
   - Remove pydantic_ai model configuration
   - Remove agent tier/tag logic
   - Keep basic settings (paths, API keys for optional embedding)

3. **Verify no pydantic_ai imports remain in findingmodel:**
   ```bash
   grep -r "pydantic_ai" packages/findingmodel/src/
   # Should return nothing
   ```

---

## Sub-Phase 4.5.5: Update Cross-Package Imports

**Status:** ⏳ PENDING

Fix any remaining import issues:

1. **Update tests that import enrichment tools:**
   - `test_finding_enrichment.py` → import from findingmodel_ai
   - Move enrichment tests to findingmodel-ai/tests/

2. **Update any other cross-package references**

3. **Verify no findingmodel code imports from findingmodel_ai:**
   ```bash
   grep -r "from findingmodel_ai" packages/findingmodel/src/
   # Should return nothing
   ```

---

## Sub-Phase 4.5.6: Verify and Test

**Status:** ⏳ PENDING

1. Run all tests: `task test`
2. Run all checks: `task check`
3. Verify import isolation:
   - findingmodel imports work without findingmodel-ai installed
   - findingmodel-ai imports work with findingmodel installed
4. Test both CLIs work independently

---

## Acceptance Criteria

- [ ] findingmodel-ai has its own config.py and logger
- [ ] findingmodel-ai has its own CLI (fm-ai)
- [ ] finding_enrichment*.py files are in findingmodel-ai
- [ ] findingmodel has NO pydantic_ai imports
- [ ] findingmodel has NO imports from findingmodel_ai
- [ ] openai is optional dependency in findingmodel (for embeddings only)
- [ ] All tests pass
- [ ] All checks pass

---

## Files Summary

### Moving FROM findingmodel TO findingmodel-ai:
- `tools/finding_enrichment.py`
- `tools/finding_enrichment_agentic.py`
- CLI commands: make-stub-model, markdown-to-fm, describe, detail

### Creating in findingmodel-ai:
- `config.py` (settings)
- `cli.py` (AI commands)
- Updated `__init__.py` (logger)

### Modifying in findingmodel:
- `pyproject.toml` (remove pydantic-ai dep)
- `config.py` (strip AI config)
- `cli.py` (remove AI commands)
- `tools/__init__.py` (remove enrichment exports)

---

## Completion Summary (2025-01-16)

All sub-phases completed successfully:

1. **4.5.1-4.5.3**: findingmodel-ai now has its own `config.py`, logger, and CLI (`fm-ai`)
2. **4.5.4**: findingmodel stripped of pydantic-ai - depends on `oidm-common[openai]` for embeddings
3. **4.5.5**: All cross-package imports fixed
4. **4.5.6**: All 337 tests pass, all checks pass

### Additional Work: Embedding Consolidation

During this phase, we also consolidated embedding functionality:

- **oidm-common/embeddings/generation.py** now provides high-level API:
  - `get_embedding(text, api_key, model, dimensions)` - handles client creation internally
  - `get_embeddings_batch(texts, api_key, model, dimensions)` - batch version
  - Graceful degradation when openai not installed

- **findingmodel** no longer imports `openai` directly
  - Changed dependency: `oidm-common` → `oidm-common[openai]`
  - `tools/duckdb_utils.py` uses oidm-common's high-level API
  - `index.py` simplified (no more `_openai_client` management)

### Remaining Work → Phase 4.6

One architectural issue remains: `findingmodel/tools/duckdb_search.py` contains `DuckDBOntologySearchClient` which searches the anatomic-locations database but lives in the wrong package. This is addressed in Phase 4.6: DuckDB Search Consolidation.
