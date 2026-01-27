# Test Suite Upgrade Plan (DuckDB Index + findingmodel build)

**Status: COMPLETED** (2025-01-27)

## Summary

All planned tests implemented except tag filtering tests (punted - tags being redesigned).
Empty/whitespace query handling updated to raise `ValueError` (implementation + tests).

---

## DuckDB Index (packages/findingmodel/tests)

### Fix/Align Existing Tests
1) **Align `tag_logic` usage with current API** - ✅ DONE

### Add/Expand Tests (concrete behaviors)
2) **Exact match short‑circuit honors tag filter** - ❌ REMOVED (punted on tags)
3) **Exact match returns immediately when tags match** - ❌ REMOVED (punted on tags)
4) **`get`/`contains` resolve by slug** - ✅ `test_get_contains_resolve_by_slug`
5) **`get`/`contains` resolve by synonym** - ✅ `test_get_contains_resolve_by_synonym`
6) **`get_full` missing ID raises KeyError** - ✅ `test_get_full_missing_id_raises_keyerror`
7) **`get_full_batch` returns only found IDs** - ✅ `test_get_full_batch_returns_only_found_ids`
8) **`search_batch` calls embeddings once** - ✅ `test_search_batch_calls_embeddings_once`
9) **`IndexEntry.match` handles synonyms (case‑insensitive)** - ✅ `test_index_entry_match_handles_synonyms`
10) **`all()` sorting by timestamps** - ✅ `test_all_sorting_by_timestamps`
11) **Tag filtering in FTS/semantic paths** - ❌ REMOVED (punted on tags)
12) **Empty/whitespace query behavior (explicit)** - ✅ DONE
    - `test_search_empty_query_raises_valueerror`
    - `test_search_batch_all_blank_queries_raises_valueerror`
    - Implementation updated: `search()` and `search_batch()` now raise `ValueError`

### Performance/Flakiness
13) **Mark latency benchmarks as slow** - ❌ REMOVED
    - Marker wasn't configured to filter anything
    - Test still exists: `test_search_latency_benchmark`

---

## findingmodel Build (packages/oidm-maintenance/tests)

### Add/Expand Tests (concrete behaviors)
1) **Duplicate name / slug rejection** - ✅ `test_duplicate_name_slug_rejection`
2) **`slug_name` equals `normalize_name(name)`** - ✅ `test_slug_name_equals_normalize_name`
3) **`file_hash_sha256` correctness** - ✅ `test_file_hash_sha256_correctness`
4) **`search_text` contains expected components** - ✅ `test_search_text_contains_expected_components`
5) **Tags/synonyms de‑duplication** - ✅ `test_tags_synonyms_deduplication`
6) **Contributor ordering + role** - ✅ `test_contributor_ordering_and_role`
7) **Embeddings error path (missing API key)** - ✅ `test_embeddings_error_missing_api_key`
8) **Embeddings error path (None embeddings)** - ✅ `test_embeddings_error_none_embeddings`

---

## CLI Build Command (packages/oidm-maintenance/tests/test_cli.py)

1) **`--no-embeddings` produces zero vectors** - ✅ `test_cli_build_no_embeddings_produces_zero_vectors`
2) **Error output on invalid source** - ✅ `test_cli_build_error_on_invalid_source`

---

## Notes

- **Tag tests removed**: Items 2, 3, 11 were removed because tag filtering is being redesigned.
- **Slow marker removed**: Item 13 was removed because the marker wasn't actually filtering tests.
- **No new fixtures created**: All tests use existing fixtures.

## Follow-up Issues

- Refactor hybrid search pattern into oidm-common (see GitHub issue template in conversation)
- Revisit tag filtering tests after tags redesign
