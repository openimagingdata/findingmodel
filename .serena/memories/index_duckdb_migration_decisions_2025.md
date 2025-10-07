2025-10-07: Implemented DuckDBIndex Step 2 core CRUD.
- Added `duckdb_index.py` with schema creation (finding_models + normalized/denormalized tables), read-only by default connections, OpenAI embedding generation, and CRUD helpers.
- Normalized contributors stored in `people`/`organizations` with link tables (`model_people`, `model_organizations`) preserving contributor order.
- Denormalized tables (`synonyms`, `tags`, `attributes`) refreshed on update; `search_text` built from name/description/synonyms/tags/attributes.
- Embeddings generated via `get_embedding_for_duckdb` (float32, 512 dims) and required for inserts; read-only mode blocks writes.
- Config gained `duckdb_index_path`; top-level package now exports `DuckDBIndex`.
- Shared utilities file `duckdb_utils.py` now installs/loads FTS+VSS extensions and provides embedding/score helpers.
Next: implement exact match + FTS/HNSW search, validation, and batch ops per remaining plan steps.