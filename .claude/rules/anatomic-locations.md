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

## CLI
- `anatomic search <query>` – Search for anatomic locations
- `anatomic hierarchy <id>` – Show hierarchy for a location

## Serena References
- `anatomic_location_search_implementation` – search implementation details
- `duckdb_hybrid_search_research_2025` – hybrid search design decisions
