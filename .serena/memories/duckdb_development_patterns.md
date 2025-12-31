DuckDB development guide: docs/duckdb-development.md

Key patterns:
- Distribution: ensure_*_db() functions, manifest.json
- Connection: setup_duckdb_connection() with extensions
- Bulk loading: read_json() for FLOAT[]/STRUCT[] (1000x faster than executemany)
- Embeddings: get_embedding_for_duckdb() for float32 conversion
- Search: weighted_fusion() for hybrid FTS + semantic

Common pitfalls: unquoted column types in read_json, missing float32 conversion, HNSW on read-only connections.
