# Unified Test Suite Remediation Plan (findingmodel monorepo)

## Goals
- Ensure tests validate **our code paths**, not just third‑party behavior.
- Eliminate unintended network/API calls from unit tests.
- Fill coverage gaps in critical areas (indexing/search, builds, embeddings, caches).
- Reduce redundancy and brittle tests that lock in known bugs.

## Constraints
- Reuse existing fixtures wherever possible.
- **No new shared fixtures without approval.**
- Keep callout tests opt‑in only (marked and skipped when keys missing).

---

## Top Issues to Fix First
1) **Accidental network calls in unit tests**
   - `findingmodel-ai` tests that instantiate `Index()` or `findingmodel.tools.add_ids_to_model()` can trigger manifest downloads.
2) **Tests that don’t exercise production code**
   - `findingmodel-ai/tests/test_ontology_search.py::test_query_terms_deduplication` re‑implements logic instead of calling our functions.
3) **Tests that lock broken behavior**
   - `findingmodel/tests/test_embedding_cache.py` asserts the known `older_than_days` bug.
4) **“Test” script collected by pytest**
   - `packages/findingmodel-ai/scripts/test_unified_enrichment.py` can be collected and will hit APIs.

---

## Phase 0 — Guardrails (Prevent External Calls / Bad Collection)
1) **Stop pytest from collecting API‑hitting scripts**
   - Rename `packages/findingmodel-ai/scripts/test_unified_enrichment.py` to a non‑test name (e.g., `unified_enrichment_smoke.py`) **or** add `__test__ = False` at top.
   - Expected: `pytest` won’t collect it by default.

2) **Make `findingmodel-ai` unit tests network‑safe**
   - Any test that calls `Index()` or `findingmodel.tools.add_ids_to_model()` must supply a local DB path or patch `findingmodel.index.DuckDBIndex` to avoid manifest download.
   - Expected: unit tests run offline without fetching manifests or DBs.

---

## Phase 1 — Fix Misaligned / Bug‑Locking Tests

### findingmodel (core)
1) **Remove `tag_logic` arg from `test_duckdb_index.py`**
   - Update `test_search_with_multiple_tags_and_logic` to call `search()` without `tag_logic`.
   - Expected: API matches `DuckDBIndex.search`.

2) **Empty query behavior**
   - `index.search("")` and `index.search("   ")` should raise `ValueError`.
   - `search_batch`: skip blank entries; if all blank, raise `ValueError`.

3) **Embedding cache `older_than_days`**
   - Stop asserting the bug (“returns 0”). Update tests to verify actual deletion after fixing implementation.

### findingmodel‑ai
4) **Replace “deduplication” test that doesn’t call prod code**
   - Move logic into a real function (if not already) and test that function, or remove the test.

---

## Phase 2 — Coverage Additions by Package

### A) oidm‑common
**DuckDB utilities**
- FTS index functionality: confirm `fts_main_{table}.match_bm25` returns results.
- HNSW index presence: verify `duckdb_indexes()` includes created index.
- `drop_search_indexes`: verify indexes are actually removed.
- `setup_duckdb_connection` with invalid extension raises error.

**Embeddings generation wrappers**
- `get_embedding` returns `None` when API key missing.
- `_get_or_create_client` returning `None` (missing OpenAI) yields `None` / list of `None`.
- `get_embeddings_batch` returns list of `None` on errors.

**EmbeddingCache**
- `clear_cache(older_than_days=...)` works (delete only old entries).
- `clear_cache(model=..., older_than_days=...)` works for combined filters.
- Error handling: DB connection exceptions do not propagate.
- Batch retrieval respects model + dimension filters.

**Models**
- `WebReference.from_tavily_result` handles optional `description` if present.
- URL validation: decide whether no‑scheme URLs are allowed; test expected behavior.

---

### B) findingmodel (core + oidm-maintenance)
**DuckDB Index**
- `search()` exact match honors tag filter.
- `get` / `contains` resolve by slug and synonym.
- `get_full` missing ID raises `KeyError`.
- `get_full_batch` returns only found IDs.
- `search_batch` calls batch embeddings once (patch `batch_embeddings_for_duckdb`).
- `IndexEntry.match` matches ID/name/synonym (case‑insensitive).
- `all(order_by="created_at"/"updated_at")` sorted correctly.
- Tag filtering in FTS/semantic paths with controlled tags.
- Empty/whitespace query raises `ValueError` (see Phase 1).
- `search_batch` blank‑skipping behavior (see Phase 1).
- Mark latency benchmarks as `@pytest.mark.slow`.

**findingmodel build (oidm-maintenance)**
- Duplicate `name` / `slug_name` rejection.
- `slug_name == normalize_name(name)` in DB.
- `file_hash_sha256` matches actual file hash.
- `search_text` includes name/description/synonyms/tags/attribute names.
- Synonym/tag de‑duplication produces single rows.
- Contributor ordering and `role` in `model_people` / `model_organizations`.
- Embedding error paths: missing key or None embeddings raise `RuntimeError`.

**CLI**
- `--no-embeddings` produces zero vectors in DB.
- Invalid source path errors clearly.

---

### C) findingmodel‑ai
- Patch Index usage so tests do **not** download DBs (use prebuilt DB or stub `DuckDBIndex`).
- Make `test_query_terms_deduplication` call actual code.
- Add a unit test that ensures `add_ids_to_model` uses the provided `source` and does not touch network (via patched `DuckDBIndex`).
- Keep `@pytest.mark.callout` tests strictly opt‑in and structure‑only.

---

### D) anatomic‑locations
- Current tests look solid; no changes required unless gaps are identified later.

---

## Phase 3 — Reduce Redundancy
1) **Drop redundant oidm‑common tests in findingmodel**
   - `packages/findingmodel/tests/test_duckdb_utils.py` duplicates oidm‑common coverage.
   - Options: remove, or reduce to a minimal smoke test validating re‑exports only.

2) **Unused fixtures cleanup**
   - Remove unused fixtures in oidm‑common tests if they remain unused.

---

## Fixture Usage Summary (Existing)
- **findingmodel**: `base_model`, `full_model`, `real_model`, `tmp_defs_path`, `prebuilt_db_path`.
- **oidm-maintenance**: `built_test_db`, `source_data_dir`, `temp_db_path`.
- **findingmodel-ai**: `base_model`, `full_model`, `real_model`, `finding_info`.
- **anatomic-locations**: `prebuilt_db_path`, `anatomic_query_embeddings`, `anatomic_sample_data`.

**No new shared fixtures will be added without your approval.**

---

## Potential New Fixtures (Approval Required)
- `tagged_test_db`: small DB with controlled tags/synonyms for search filter tests.
- `mini_fm_source_dir`: temp directory with a few curated `.fm.json` models for build tests.

---

## Expected Confidence After Completion
- **High** confidence in DuckDB index/search behavior (findingmodel + anatomic-locations).
- **High** confidence in build integrity (oidm-maintenance).
- **Moderate‑high** confidence in findingmodel‑ai wiring without network dependence.
- **High** confidence in oidm‑common utilities, embeddings wrappers, and caches.
