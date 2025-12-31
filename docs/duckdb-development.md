# DuckDB Development Guide

Developer reference for working with DuckDB databases in findingmodel.

## Overview

FindingModel uses DuckDB for two databases:

| Database | Purpose | Key Features |
|----------|---------|--------------|
| `finding_models.duckdb` | Finding model index | FTS + semantic search, model metadata |
| `anatomic_locations.duckdb` | Anatomic location ontology | Hierarchy navigation, laterality, semantic search |

Both use the same patterns: remote distribution via manifest, local caching, and shared utilities in `tools/duckdb_utils.py`.

---

## Database Distribution

### How It Works

1. **manifest.json** on remote storage lists available databases with URLs and hashes
2. **ensure_*_db()** functions check local cache, download if needed, verify hash
3. **Pooch** handles download, caching, and hash verification
4. Local files cached in platform-native directories (`~/.local/share/findingmodel/` on Linux)

### Key Functions

```python
from findingmodel.config import ensure_index_db, ensure_anatomic_db, ensure_db_file

# Get database path (downloads if needed)
index_path = ensure_index_db()
anatomic_path = ensure_anatomic_db()

# Low-level: explicit control
db_path = ensure_db_file(
    file_path=None,           # None = managed download, or explicit path
    remote_url=None,          # Override manifest URL
    remote_hash=None,         # Override manifest hash
    manifest_key="finding_models"
)
```

### Configuration

```bash
# .env - Override defaults
DUCKDB_INDEX_PATH=/mnt/data/finding_models.duckdb      # Skip download, use this file
DUCKDB_ANATOMIC_PATH=/mnt/data/anatomic_locations.duckdb

# Or specify explicit remote (overrides manifest)
REMOTE_INDEX_DB_URL=https://example.com/db.duckdb
REMOTE_INDEX_DB_HASH=sha256:abc123...
```

### Adding a New Database

1. Add entry to `manifest.json`:
   ```json
   "new_database": {
     "version": "2025-01-01",
     "url": "https://...",
     "hash": "sha256:...",
     "size_bytes": 12345
   }
   ```

2. Add settings in `config.py`:
   ```python
   duckdb_new_path: str | None = None
   remote_new_db_url: str | None = None
   remote_new_db_hash: str | None = None
   ```

3. Add ensure function:
   ```python
   def ensure_new_db() -> Path:
       return ensure_db_file(
           settings.duckdb_new_path,
           settings.remote_new_db_url,
           settings.remote_new_db_hash,
           manifest_key="new_database",
       )
   ```

---

## Connection Patterns

### Standard Setup

```python
from findingmodel.tools.duckdb_utils import setup_duckdb_connection

# Read-only (queries)
conn = setup_duckdb_connection(db_path, read_only=True)

# Read-write (migrations, index builds)
conn = setup_duckdb_connection(db_path, read_only=False)
# Automatically sets: hnsw_enable_experimental_persistence = true
```

### Extensions

The following extensions are loaded automatically:
- **fts** - Full-text search with BM25 scoring
- **vss** - Vector similarity search (HNSW indexes)

### Context Manager Pattern

```python
class MyIndex:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or ensure_my_db()
        self.conn: duckdb.DuckDBPyConnection | None = None

    def __enter__(self):
        self.conn = setup_duckdb_connection(self.db_path, read_only=True)
        return self

    def __exit__(self, *args):
        if self.conn:
            self.conn.close()
            self.conn = None
```

---

## Bulk Loading

### The Problem

DuckDB's `executemany` is extremely slow with complex column types due to Python-to-DuckDB parameter conversion:

| Method | Speed | Use Case |
|--------|-------|----------|
| `executemany` | ~52ms/row | Simple types only |
| `read_json()` | ~0.05ms/row | **Complex types (1000x faster)** |

This affects: `FLOAT[N]` vectors, `STRUCT(...)[]` arrays, nested types.

### The Solution

Write data to temporary NDJSON file, bulk load with `read_json()`:

```python
import json
import tempfile
from pathlib import Path

def bulk_load_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    data: list[dict],
    column_types: dict[str, str],
) -> int:
    """Bulk load data into a table via read_json().

    Args:
        conn: DuckDB connection
        table_name: Target table
        data: List of row dictionaries
        column_types: Column name -> DuckDB type mapping
    """
    if not data:
        return 0

    # Write NDJSON (one JSON object per line)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
        temp_path = Path(f.name)

    try:
        # IMPORTANT: Quote complex types in column spec
        columns_spec = ", ".join(
            f"{name}: '{dtype}'" for name, dtype in column_types.items()
        )
        conn.execute(f"""
            INSERT INTO {table_name}
            SELECT * FROM read_json('{temp_path}', columns={{{columns_spec}}})
        """)
        return len(data)
    finally:
        temp_path.unlink()
```

### Key Details

1. **Quote column types** - Types like `FLOAT[512]` and `STRUCT(id VARCHAR, display VARCHAR)[]` must be quoted in the columns spec
2. **NDJSON format** - One JSON object per line, not a JSON array
3. **STRUCT arrays work natively** - `[{"id": "x", "display": "y"}]` maps directly
4. **Float precision** - JSON floats round-trip correctly to FLOAT columns

### When to Use

- Inserting 100+ rows with vector embeddings
- Any table with STRUCT[] columns
- Batch migrations or database rebuilds

### Reference Implementation

See `src/findingmodel/anatomic_migration.py:_bulk_load_table()`

---

## Embeddings

### Float32 Conversion

DuckDB FLOAT columns are 32-bit. OpenAI returns 64-bit floats. Always convert:

```python
from findingmodel.tools.duckdb_utils import get_embedding_for_duckdb, batch_embeddings_for_duckdb

# Single embedding (returns float32 list)
embedding = await get_embedding_for_duckdb("kidney tumor")

# Batch embeddings (single API call, all float32)
embeddings = await batch_embeddings_for_duckdb(["term1", "term2", "term3"], client=openai_client)
```

### Embedding Dimensions

Configured via `settings.openai_embedding_dimensions` (default: 512). Both databases use the same dimension.

---

## Search Patterns

### Hybrid Search

Combine full-text (BM25) and semantic (vector) search:

```python
from findingmodel.tools.duckdb_utils import normalize_scores, weighted_fusion

# 1. Full-text search
fts_results = conn.execute("""
    SELECT id, fts_main_table.match_bm25(id, ?) as score
    FROM table WHERE score IS NOT NULL
    ORDER BY score DESC LIMIT ?
""", [query, limit]).fetchall()

# 2. Semantic search
vector_results = conn.execute("""
    SELECT id, array_cosine_similarity(vector, ?::FLOAT[512]) as score
    FROM table
    ORDER BY score DESC LIMIT ?
""", [query_embedding, limit]).fetchall()

# 3. Combine with weighted fusion
combined = weighted_fusion(
    fts_results, vector_results,
    weight_a=0.3,  # FTS weight
    weight_b=0.7,  # Semantic weight
)
```

### Score Normalization

BM25 scores are unbounded. Normalize before fusion:

```python
from findingmodel.tools.duckdb_utils import normalize_scores

normalized = normalize_scores([12.5, 8.3, 5.1])  # -> [1.0, 0.43, 0.0]
```

---

## Index Management

### Schema Changes

DuckDB doesn't support `ALTER TABLE` for adding columns to tables with indexes. Pattern:

1. Drop all indexes
2. Make schema changes
3. Rebuild indexes

### HNSW Vector Indexes

```sql
-- Create (only on write connection)
CREATE INDEX idx_vectors ON table USING HNSW (vector);

-- Requires this setting (set automatically by setup_duckdb_connection)
SET hnsw_enable_experimental_persistence = true;
```

### FTS Indexes

```sql
-- Create full-text index
PRAGMA create_fts_index('table', 'id', 'name', 'description', 'synonyms');

-- Search
SELECT *, fts_main_table.match_bm25(id, 'search query') as score
FROM table
WHERE score IS NOT NULL;
```

### Rebuild Pattern

When updating data:

```python
# 1. Drop indexes
conn.execute("DROP INDEX IF EXISTS idx_hnsw")
conn.execute("PRAGMA drop_fts_index('table')")

# 2. Clear and reload data
conn.execute("DELETE FROM table")
# ... insert new data ...

# 3. Rebuild indexes
conn.execute("CREATE INDEX idx_hnsw ON table USING HNSW (vector)")
conn.execute("PRAGMA create_fts_index('table', 'id', 'name', 'description')")
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/findingmodel/config.py` | Settings, ensure_*_db() functions |
| `src/findingmodel/tools/duckdb_utils.py` | Connection, embeddings, fusion utilities |
| `src/findingmodel/index.py` | Finding model index implementation |
| `src/findingmodel/anatomic_index.py` | Anatomic location index |
| `src/findingmodel/anatomic_migration.py` | Database build with bulk loading |
| `docs/manifest_schema.md` | Manifest.json specification |

---

## Common Pitfalls

1. **Slow inserts** - Use `read_json()` bulk loading for complex types
2. **Unquoted column types** - `FLOAT[512]` must be quoted in read_json columns spec
3. **Float precision** - Always use `get_embedding_for_duckdb()` for float32 conversion
4. **Index on read-only** - Can't create HNSW indexes on read-only connections
5. **Missing extensions** - Use `setup_duckdb_connection()` to ensure FTS/VSS loaded
