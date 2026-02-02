---
paths: "packages/anatomic-locations/**"
---

# anatomic-locations Package Rules

## Purpose
Anatomic location ontology navigation: hierarchy traversal, laterality variants, semantic search.

## Constraints
- **No AI dependencies**: Query-only package. AI-assisted discovery lives in findingmodel-ai.
- **Database auto-download**: Uses oidm-common's pooch-based download on first use.
- **Async-first**: All database operations are async via DuckDB.

## Key Patterns
- `AnatomicLocationIndex` as main entry point
- Hybrid search (FTS + vector) with configurable weights
- Laterality variant generation (left, right, bilateral)

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
- `index.search(query, limit=10)` – async hybrid search
- `index.get(location_id)` – sync get by ID
- `index.get_children_of(parent_id)` – sync get direct children
- `index.find_by_code(system, code)` – sync lookup by external code
- `location.get_containment_ancestors()` – get ancestors
- `location.get_containment_descendants()` – get descendants
- `location.get_laterality_variants()` – get laterality variants (dict)

## Serena References
- `anatomic_location_search_implementation` – search implementation details
- `duckdb_hybrid_search_research_2025` – hybrid search design decisions
