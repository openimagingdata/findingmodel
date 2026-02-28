---
paths: "packages/anatomic-locations/**"
---

# anatomic-locations Package Rules

## Purpose
Anatomic location ontology navigation: hierarchy traversal, laterality variants, semantic search.

## Constraints
- **No AI dependencies**: Query-only package. AI-assisted discovery lives in findingmodel-ai.
- **Database auto-download**: Uses oidm-common's pooch-based download on first use.
- **Async-first**: Search operations are async; lookup/hierarchy methods are sync.
- **Inherits `ReadOnlyDuckDBIndex`**: Base class in `oidm_common.duckdb.base` provides connection lifecycle and auto-open.

## Key Patterns
- `AnatomicLocationIndex` as main entry point (inherits `ReadOnlyDuckDBIndex`)
- Auto-open: `_ensure_connection()` opens the connection on first use; no explicit `open()` required
- Hybrid search (FTS + vector) with RRF fusion
- Laterality variant generation (left, right, bilateral)
- **Hydration pattern**: All location queries go through `_fetch_locations(conn, suffix_sql, params)` — a single entry point that appends WHERE/ORDER to `_LOCATION_SELECT` and returns hydrated objects. `_LOCATION_SELECT` uses DuckDB correlated subqueries to pull codes, synonyms, and references inline in one query (DuckDB optimizes these into a join plan). `_build_location(row)` is a pure transform using `AnatomicLocation.model_validate(data)` — Pydantic handles enum coercion, nested-model construction, and extra-key ignoring automatically. `_get_locations_by_ids(conn, ids)` wraps `_fetch_locations` and re-sorts to preserve input order.
- **Schema evolution safety**: `_LOCATION_SELECT` uses `SELECT al.* EXCLUDE (search_text, vector)`. Only exclude columns present in ALL schema versions. `synonyms_text` (added v0.2.3) is NOT excluded — old schemas lack it and `EXCLUDE` raises a `Binder Error` for missing columns. Extra columns from `SELECT *` are silently ignored by `model_validate`. Correlated subqueries alias codes/synonyms/refs inline — also safe on old schemas since all 3 related tables exist in every version.

## CLI (`anatomic-locations`)

Top-level commands (accept location ID like `RID56` or name/synonym):
- `anatomic-locations search <query>` – Hybrid FTS + semantic search
- `anatomic-locations hierarchy <id|name>` – Show ancestors, location, and descendants tree
- `anatomic-locations children <id|name>` – List direct children
- `anatomic-locations ancestors <id|name>` – Show containment ancestors
- `anatomic-locations descendants <id|name>` – Show containment descendants
- `anatomic-locations laterality <id|name>` – Show laterality variants
- `anatomic-locations code <system> <code>` – Find by external code (SNOMED, FMA)
- `anatomic-locations stats` – Database statistics

## Python API

Key methods:
- `index.get(identifier)` – sync lookup by RID, description, or synonym (case-insensitive; resolution order: ID → description → synonym)
- `index.search(query, limit=10)` – async hybrid search (FTS + semantic + RRF)
- `index.search_batch(queries, limit=10)` – async batch search; batches all embedding API calls into one `get_embeddings_batch()` call; returns `dict[str, list[AnatomicLocation]]`
- `index.get_children_of(parent_id)` – sync get direct children
- `index.find_by_code(system, code)` – sync lookup by external code
- `location.get_containment_ancestors()` – get ancestors
- `location.get_containment_descendants()` – get descendants
- `location.get_laterality_variants()` – get laterality variants (dict)

## Configuration (`AnatomicLocationSettings`)
- Uses `env_prefix="ANATOMIC_"` and `env_file=".env"` (reads from project `.env`)
- `ANATOMIC_DB_PATH` – database file path
- `ANATOMIC_REMOTE_DB_URL` / `ANATOMIC_REMOTE_DB_HASH` – custom download
- `OPENAI_API_KEY` or `ANATOMIC_OPENAI_API_KEY` – enables semantic search (AliasChoices fallback)
- Without OpenAI key, search degrades to FTS-only (keyword matching)

## Serena References
- `anatomic_location_search_implementation` – search implementation details
- `duckdb_hybrid_search_research_2025` – hybrid search design decisions
