# Index API Enhancements and Database Self-Containment Plan

**Target Release**: v0.5.0
**Status**: Approved - Ready for Implementation
**Created**: 2025-01-24
**Last Reviewed**: 2025-01-25
**Priority**: High

**Review Notes**: Plan reviewed and approved. Key clarifications documented in "Confirmed Decisions" section. All features address real operational needs with sound architecture.

## Overview

Enhance the DuckDB Index implementation with four major improvements:
1. **✅ Enhanced API Methods** - Add all/search/count methods requested by FindingModelForge
2. **✅ Manifest-Based Downloads** - Enable database updates without library releases
3. **✅ Self-Contained Databases** - Store full JSON models in DuckDB (separate table)
4. **✅ Local ID Generation** - Move OIFM ID generation from GitHub to Index

**Current Status**: Phases 1, 2, and 3 completed and tested. These changes address real-world production needs from FindingModelForge and improve the overall developer experience.

## Key Architecture Decisions

### JSON Storage
- **Separate `finding_model_json` table** - keeps main index compact
- **TEXT column** (not JSON type) - we don't need DuckDB to parse JSON
- **Automatic compression** - DuckDB handles this, no manual compression needed
- **Lookup by primary key only** - no JOINs needed, simple key-based fetch

### Manifest Schema
- **Documented format** with versioning
- **Multiple databases** in single manifest
- **Metadata** for validation (hash, size, record count)

### Code Quality
- **Extracted common patterns** - no duplication between list/search/count
- **Index on slug_name** - for efficient LIKE queries
- **Proper separation** of WHERE params vs ORDER BY params

## Background

### Current Problems

1. **Incomplete API** - Applications must break abstraction to access DuckDB connection directly for basic operations like pagination and simple search
2. **Data/Code Coupling** - Database updates require new library releases
3. **Split Storage** - Metadata in DuckDB, full JSON on disk separately
4. **External ID Generation** - OIFM IDs generated via GitHub-based counter (network dependency, concurrency issues)

### Real-World Impact

FindingModelForge (production application) currently does:
```python
# ❌ Breaking abstraction - accessing internal connection
conn = self.index._ensure_connection()
rows = conn.execute("""
    SELECT oifm_id, name, slug_name FROM finding_models
    WHERE slug_name LIKE ? ORDER BY LOWER(name) LIMIT ? OFFSET ?
""", [pattern, limit, offset]).fetchall()
```

This is a code smell indicating our API is incomplete for real-world UI needs.

## Goals

1. **Complete API Surface** - Enable apps to use Index without breaking abstraction
2. **Self-Contained Databases** - Single .duckdb file with everything (metadata + embeddings + JSON)
3. **Independent Updates** - Update databases without releasing new library versions
4. **Better DX** - Local, thread-safe ID generation

## Implementation Plan

### Phase 1: Foundation - Manifest Pattern & JSON Storage

**Status**: ✅ COMPLETED
**Priority**: Critical (blocks other phases)

#### Task 1.1: Add httpx Dependency

**File**: `pyproject.toml`

Add httpx for manifest fetching:

```toml
dependencies = [
    "httpx>=0.27.0",  # For manifest fetching
    # ... existing deps
]
```

**Why httpx?**
- Modern, actively maintained
- Built-in timeout support
- Better async support (future-proofing)
- Similar API to requests

**Acceptance Criteria**:
- [x] httpx added to dependencies
- [x] `uv sync` installs cleanly

---

#### Task 1.2: Document Manifest Schema

**File**: `docs/manifest_schema.md` (new file)

Create comprehensive manifest.json schema documentation:

```markdown
# Manifest.json Schema

## Purpose
The manifest.json file enables database updates without requiring new library releases. The library fetches this manifest at runtime to discover the latest database URLs and versions.

## Schema (v1.0)

```json
{
  "manifest_version": "1.0",
  "generated_at": "2025-01-24T10:30:00Z",
  "databases": {
    "finding_models": {
      "version": "2025-01-24",
      "url": "https://findingmodelsdata.t3.storage.dev/finding_models_2025-01-24.duckdb",
      "hash": "sha256:abc123...",
      "size_bytes": 52428800,
      "record_count": 1234,
      "description": "Finding model index with embeddings and full JSON"
    },
    "anatomic_locations": {
      "version": "2025-01-20",
      "url": "https://findingmodelsdata.t3.storage.dev/anatomic_locations_2025-01-20.duckdb",
      "hash": "sha256:def456...",
      "size_bytes": 47185920,
      "record_count": 5678,
      "description": "Anatomic location ontologies with embeddings"
    }
  }
}
```

## Field Definitions

### Top Level
- `manifest_version` (string, required): Schema version for backward compatibility
- `generated_at` (ISO 8601 datetime, required): When manifest was generated
- `databases` (object, required): Dictionary of available databases

### Database Entry
- `version` (string, required): Database version identifier (typically date)
- `url` (string, required): Full HTTPS URL to .duckdb file
- `hash` (string, required): SHA256 hash in format "sha256:hexdigest"
- `size_bytes` (integer, required): File size in bytes
- `record_count` (integer, optional): Number of records in database
- `description` (string, optional): Human-readable description

## Validation Rules

1. Hash format must be `sha256:` followed by 64 hex characters
2. URL must use HTTPS protocol
3. Database keys should match config field names (e.g., "finding_models" matches `duckdb_index_path`)
4. Version should be sortable (recommend ISO date format: YYYY-MM-DD)

## Update Process

1. Build new database file
2. Compute SHA256: `shasum -a 256 finding_models.duckdb`
3. Upload to hosting
4. Update manifest.json with new URL, hash, version
5. Users automatically get new version on next library use

## Backward Compatibility

- manifest_version allows future schema changes
- Library gracefully falls back to direct URL/hash if manifest fetch fails
- Missing optional fields don't break parsing
```

---

#### Task 1.3: Implement Manifest Fetching

**File**: `src/findingmodel/config.py`

Add manifest infrastructure:

```python
from typing import Any
import httpx

# Module-level cache for manifest (cleared on process restart)
_manifest_cache: dict[str, Any] | None = None

class Config(BaseSettings):
    # ... existing fields ...

    remote_manifest_url: str | None = Field(
        default="https://findingmodelsdata.t3.storage.dev/manifest.json",
        description="URL to JSON manifest for database versions"
    )

def fetch_manifest() -> dict[str, Any]:
    """Fetch and parse the remote manifest JSON with session caching.

    Returns:
        Parsed manifest with database version info

    Raises:
        ConfigurationError: If manifest URL not configured
        httpx.HTTPError: If fetch fails

    Example:
        manifest = fetch_manifest()
        db_info = manifest["finding_models"]
        # {"version": "2025-01-24", "url": "...", "hash": "sha256:..."}
    """
    global _manifest_cache

    # Return cached manifest if available
    if _manifest_cache is not None:
        logger.debug("Using cached manifest")
        return _manifest_cache

    settings = Config()
    if not settings.remote_manifest_url:
        raise ConfigurationError("Manifest URL not configured")

    logger.info(f"Fetching manifest from {settings.remote_manifest_url}")
    response = httpx.get(settings.remote_manifest_url, timeout=10.0)
    response.raise_for_status()

    _manifest_cache = response.json()
    logger.debug(f"Manifest cached with keys: {list(_manifest_cache.keys())}")
    return _manifest_cache

def clear_manifest_cache() -> None:
    """Clear the manifest cache (for testing)."""
    global _manifest_cache
    _manifest_cache = None
```

**Acceptance Criteria**:
- [x] fetch_manifest() successfully fetches and parses JSON
- [x] Session caching prevents repeated network calls
- [x] clear_manifest_cache() works for testing
- [x] Proper error messages for network failures
- [x] Type hints throughout

---

#### Task 1.3: Update ensure_db_file() for Manifest Support

**File**: `src/findingmodel/config.py`

Enhance ensure_db_file() to support manifest lookups:

```python
def ensure_db_file(
    filename: str,
    remote_url: str | None,
    remote_hash: str | None,
    manifest_key: str | None = None,
) -> Path:
    """Download database file with manifest support.

    Args:
        filename: Database filename (e.g., "finding_models.duckdb")
        remote_url: Direct URL (fallback/backward compat)
        remote_hash: Direct hash (fallback/backward compat)
        manifest_key: Key in manifest JSON (e.g., "finding_models")

    Priority:
        1. Use existing local file if present and hash matches
        2. Try manifest fetch if key provided
        3. Fall back to direct URL/hash
        4. Error if all methods fail

    Returns:
        Path to downloaded/verified database file

    Example:
        # Prefer manifest, fall back to direct URL
        db_path = ensure_db_file(
            "finding_models.duckdb",
            remote_url="https://example.com/db.duckdb",
            remote_hash="sha256:abc123...",
            manifest_key="finding_models"
        )
    """
    # Check if local file exists and is valid
    local_path = get_data_directory() / filename
    if local_path.exists():
        # TODO: Add hash verification here
        logger.info(f"Using existing local file: {local_path}")
        return local_path

    # Try manifest first
    url_to_use = remote_url
    hash_to_use = remote_hash

    if manifest_key:
        try:
            manifest = fetch_manifest()
            db_info = manifest.get(manifest_key)
            if db_info:
                url_to_use = db_info["url"]
                hash_to_use = db_info["hash"]
                version = db_info.get("version", "unknown")
                logger.info(f"Using manifest version {version} for {manifest_key}")
        except Exception as e:
            logger.warning(f"Manifest fetch failed, trying direct URL: {e}")
            # Fall through to use direct URL/hash

    # Download using Pooch (existing logic)
    if not url_to_use or not hash_to_use:
        raise ConfigurationError(
            f"No URL/hash available for {filename}. "
            "Manifest fetch failed and no direct URL configured."
        )

    # ... existing Pooch download logic ...
    return downloaded_path
```

**Acceptance Criteria**:
- [x] Manifest lookup attempted first when manifest_key provided
- [x] Graceful fallback to direct URL/hash
- [x] Clear logging shows which method succeeded
- [x] Error message helpful when all methods fail
- [x] Manifest key now required (not optional) - manifest is primary source

---

#### Task 1.4: Add Separate JSON Storage Table

**File**: `src/findingmodel/duckdb_index.py`

Create separate table for JSON blobs to keep main index compact:

```python
def _create_schema(self) -> None:
    """Create DuckDB schema with separate JSON storage table."""

    # Main finding_models table (unchanged - stays compact)
    self.conn.execute("""
        CREATE TABLE IF NOT EXISTS finding_models (
            -- Core identifiers
            oifm_id VARCHAR PRIMARY KEY,
            slug_name VARCHAR UNIQUE NOT NULL,
            name VARCHAR UNIQUE NOT NULL,

            -- File tracking
            filename VARCHAR UNIQUE NOT NULL,
            file_hash_sha256 VARCHAR NOT NULL,

            -- Content
            description TEXT,
            full_description TEXT,

            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Embeddings (required for HNSW)
            embedding FLOAT[1536] NOT NULL
        )
    """)

    # Separate table for full JSON blobs (NEW)
    self.conn.execute("""
        CREATE TABLE IF NOT EXISTS finding_model_json (
            oifm_id VARCHAR PRIMARY KEY,
            model_json TEXT NOT NULL  -- DuckDB auto-compresses TEXT
        )
    """)

    # Create index on slug_name for efficient LIKE queries (NEW)
    self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_slug_name
        ON finding_models(slug_name)
    """)

    # ... rest of schema (people, organizations, denormalized tables) unchanged ...
```

**Rationale**:
- **Separate table keeps main index compact** for list/search operations
- **JSON only loaded when needed** via get_full() method
- **Better cache locality** - frequently accessed metadata not mixed with large JSON blobs
- **TEXT column with automatic compression** - DuckDB handles compression, no manual setup needed
- **No JOIN needed** - simple primary key lookup

**Acceptance Criteria**:
- [x] Schema creates both tables successfully
- [x] finding_model_json table accepts TEXT
- [x] Index on slug_name created
- [x] Can store/retrieve full JSON from separate table
- [x] Doesn't break existing tests

---

#### Task 1.5: Update Write Operations for JSON Storage

**File**: `src/findingmodel/duckdb_index.py`

Update all insert/update operations:

```python
def add_or_update_entry_from_file(self, json_file: Path) -> None:
    """Add or update finding model from JSON file."""
    # Read and parse JSON
    json_text = json_file.read_text(encoding="utf-8")
    model = FindingModelFull.model_validate_json(json_text)

    # ... existing metadata extraction ...

    # Insert/update main metadata (NO JSON column)
    self.conn.execute("""
        INSERT INTO finding_models (
            oifm_id, slug_name, name, filename, file_hash_sha256,
            description, full_description, embedding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (oifm_id) DO UPDATE SET
            name = EXCLUDED.name,
            slug_name = EXCLUDED.slug_name,
            description = EXCLUDED.description,
            full_description = EXCLUDED.full_description,
            embedding = EXCLUDED.embedding,
            updated_at = CURRENT_TIMESTAMP
    """, [
        model.oifm_id,
        model.slug_name,
        model.name,
        json_file.name,
        file_hash,
        model.description,
        model.full_description,
        embedding_array
    ])

    # Insert/update JSON in separate table
    self.conn.execute("""
        INSERT INTO finding_model_json (oifm_id, model_json)
        VALUES (?, ?)
        ON CONFLICT (oifm_id) DO UPDATE SET
            model_json = EXCLUDED.model_json
    """, [model.oifm_id, json_text])

    # ... rest of denormalized table updates ...
```

**Acceptance Criteria**:
- [x] JSON stored in separate table on insert
- [x] JSON updated in separate table on model update
- [x] Main table operations don't load JSON
- [x] All existing tests pass
- [x] No performance regression

---

#### Task 1.6: Add get_full() Method

**File**: `src/findingmodel/duckdb_index.py`

Add method to retrieve full model:

```python
def get_full(self, oifm_id: str) -> FindingModelFull:
    """Get full FindingModelFull object by ID.

    Args:
        oifm_id: The OIFM ID to retrieve

    Returns:
        Full FindingModelFull object parsed from stored JSON

    Raises:
        KeyError: If model not found

    Example:
        model = index.get_full("OIFM_RADLEX_000001")
        # Returns complete FindingModelFull with all attributes
    """
    result = self.conn.execute(
        "SELECT model_json FROM finding_model_json WHERE oifm_id = ?",
        [oifm_id]
    ).fetchone()

    if not result:
        raise KeyError(f"Model not found: {oifm_id}")

    json_text = result[0]
    return FindingModelFull.model_validate_json(json_text)

def get_full_batch(self, oifm_ids: list[str]) -> dict[str, FindingModelFull]:
    """Get multiple full models efficiently.

    Args:
        oifm_ids: List of OIFM IDs to retrieve

    Returns:
        Dict mapping OIFM ID to FindingModelFull object

    Example:
        models = index.get_full_batch(["OIFM_RADLEX_000001", "OIFM_CUSTOM_000042"])
        # {oifm_id: FindingModelFull, ...}
    """
    if not oifm_ids:
        return {}

    placeholders = ", ".join(["?"] * len(oifm_ids))
    results = self.conn.execute(
        f"SELECT oifm_id, model_json FROM finding_model_json WHERE oifm_id IN ({placeholders})",
        oifm_ids
    ).fetchall()

    return {
        oifm_id: FindingModelFull.model_validate_json(json_text)
        for oifm_id, json_text in results
    }
```

**Acceptance Criteria**:
- [x] get_full() returns correct FindingModelFull
- [x] Raises KeyError for missing models
- [x] get_full_batch() handles empty list
- [x] Batch method is more efficient than loop
- [x] Tests verify JSON roundtrip integrity

---

### Phase 2: Enhanced API Methods

**Status**: ✅ COMPLETED
**Priority**: High (unblocks FindingModelForge)

**Note**: Method `list()` renamed to `all()` to avoid shadowing Python's built-in `list` type.

#### Task 2.0: Implement Shared Helper Methods

**File**: `src/findingmodel/duckdb_index.py`

Extract common patterns to eliminate duplication:

```python
def _build_slug_search_clause(
    self,
    pattern: str,
    match_type: Literal["exact", "prefix", "contains"]
) -> tuple[str, str, str]:
    """Build WHERE clause and patterns for slug matching.

    Args:
        pattern: Search pattern (will be normalized)
        match_type: How to match the pattern

    Returns:
        (where_clause, sql_pattern, normalized_pattern) tuple

    Example:
        where, sql_pat, norm = self._build_slug_search_clause("abscess", "contains")
        # ("slug_name LIKE ?", "%abscess%", "abscess")
    """
    from findingmodel.finding_model import normalize_name

    normalized = normalize_name(pattern)

    if match_type == "exact":
        return ("slug_name = ?", normalized, normalized)
    elif match_type == "prefix":
        return ("slug_name LIKE ?", f"{normalized}%", normalized)
    else:  # contains
        return ("slug_name LIKE ?", f"%{normalized}%", normalized)

def _execute_paginated_query(
    self,
    where_clause: str = "",
    where_params: list = None,
    order_clause: str = "LOWER(name)",
    order_params: list = None,
    limit: int = 100,
    offset: int = 0
) -> tuple[list[IndexEntry], int]:
    """Execute paginated query with count and result fetching.

    Shared by list() and search_by_slug() to eliminate duplication.

    Args:
        where_clause: SQL WHERE clause (without WHERE keyword)
        where_params: Parameters for WHERE clause
        order_clause: SQL ORDER BY clause (without ORDER BY keyword)
        order_params: Parameters for ORDER BY clause (e.g., for CASE expressions)
        limit: Maximum results to return
        offset: Number of results to skip

    Returns:
        (list of IndexEntry objects, total count) tuple
    """
    where_params = where_params or []
    order_params = order_params or []
    where_sql = f"WHERE {where_clause}" if where_clause else ""

    # Get total count (only uses WHERE params)
    total = self.conn.execute(
        f"SELECT COUNT(*) FROM finding_models {where_sql}",
        where_params
    ).fetchone()[0]

    # Get paginated results (uses WHERE + ORDER + pagination params)
    results = self.conn.execute(f"""
        SELECT oifm_id, name, slug_name, description, created_at, updated_at
        FROM finding_models
        {where_sql}
        ORDER BY {order_clause}
        LIMIT ? OFFSET ?
    """, where_params + order_params + [limit, offset]).fetchall()

    # Build IndexEntry objects
    entries = [
        IndexEntry(
            oifm_id=row[0],
            name=row[1],
            slug_name=row[2],
            description=row[3],
            created_at=row[4],
            updated_at=row[5]
        )
        for row in results
    ]

    return entries, total
```

**Rationale**:
- `_build_slug_search_clause()` - eliminates duplicate pattern building
- `_execute_paginated_query()` - eliminates duplicate COUNT + SELECT + IndexEntry building
- Proper separation of WHERE params vs ORDER BY params
- Used by all(), search_by_slug(), and indirectly by count_search()

**Acceptance Criteria**:
- [x] Both helper methods implemented
- [x] Proper parameter separation (where vs order)
- [x] No duplication in subsequent methods
- [x] Clear documentation of return types

---

#### Task 2.1: Implement all() Method

**File**: `src/findingmodel/duckdb_index.py`

Add paginated listing using `_execute_paginated_query()` helper:

```python
def all(
    self,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "name",
    order_dir: Literal["asc", "desc"] = "asc"
) -> tuple[list[IndexEntry], int]:
    """Get all finding models with pagination.

    Args:
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        order_by: Field to sort by ("name", "oifm_id", "created_at", "updated_at")
        order_dir: Sort direction ("asc" or "desc")

    Returns:
        Tuple of (list of IndexEntry objects, total count)

    Raises:
        ValueError: If order_by field is invalid

    Example:
        # Get page 3 (items 41-60) sorted by name
        models, total = index.all(limit=20, offset=40, order_by="name")
        print(f"Showing {len(models)} of {total} total models")
    """
    # Validate order_by field
    valid_fields = {"name", "oifm_id", "created_at", "updated_at", "slug_name"}
    if order_by not in valid_fields:
        raise ValueError(f"Invalid order_by field: {order_by}")

    # Validate order_dir
    if order_dir not in {"asc", "desc"}:
        raise ValueError(f"Invalid order_dir: {order_dir}")

    # Build order clause (use LOWER() for case-insensitive sorting on text fields)
    order_clause = f"LOWER({order_by})" if order_by in {"name", "slug_name"} else order_by
    order_clause = f"{order_clause} {order_dir.upper()}"

    # Use helper to execute query (no WHERE clause for all)
    return self._execute_paginated_query(
        order_clause=order_clause,
        limit=limit,
        offset=offset
    )
```

**Security Note**: order_by is validated against whitelist, not directly interpolated, to prevent SQL injection.

**Acceptance Criteria**:
- [x] Returns correct paginated results
- [x] Total count accurate
- [x] Sorting works for all valid fields
- [x] Case-insensitive sorting for name/slug
- [x] Raises ValueError for invalid parameters
- [x] Works with empty database
- [x] Works with single page of results

---

#### Task 2.2: Implement search_by_slug() Method

**File**: `src/findingmodel/duckdb_index.py`

Add pattern-based search using helper methods:

```python
def search_by_slug(
    self,
    pattern: str,
    limit: int = 100,
    offset: int = 0,
    match_type: Literal["exact", "prefix", "contains"] = "contains"
) -> tuple[list[IndexEntry], int]:
    """Search finding models by slug name pattern.

    Args:
        pattern: Search pattern (will be normalized via normalize_name)
        limit: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        match_type: How to match the pattern:
            - "exact": Exact match on slug_name
            - "prefix": slug_name starts with pattern
            - "contains": slug_name contains pattern (default)

    Returns:
        Tuple of (list of matching IndexEntry objects, total count)

    Example:
        # User searches for "abscess" - find all models with "abscess" in slug
        models, total = index.search_by_slug("abscess", limit=20, offset=0)
        # Internally: WHERE slug_name LIKE '%abscess%' LIMIT 20 OFFSET 0
    """
    # Build WHERE clause using helper
    where_clause, sql_pattern, normalized = self._build_slug_search_clause(pattern, match_type)

    # Build ORDER BY clause for relevance ranking
    order_clause = """
        CASE
            WHEN slug_name = ? THEN 0
            WHEN slug_name LIKE ? THEN 1
            ELSE 2
        END,
        LOWER(name)
    """

    # Use helper to execute query
    return self._execute_paginated_query(
        where_clause=where_clause,
        where_params=[sql_pattern],
        order_clause=order_clause,
        order_params=[normalized, f"{normalized}%"],
        limit=limit,
        offset=offset
    )
```

**Acceptance Criteria**:
- [x] Exact match works correctly
- [x] Prefix match works correctly
- [x] Contains match works correctly
- [x] Results ranked by relevance (exact > prefix > contains)
- [x] Pattern normalized before matching
- [x] Pagination works correctly
- [x] Returns empty list for no matches

---

#### Task 2.3: Implement count() and count_search() Methods

**File**: `src/findingmodel/duckdb_index.py`

Add count helpers using `_build_slug_search_clause()` helper:

```python
def count(self) -> int:
    """Get total count of finding models in index.

    Returns:
        Total number of finding models

    Example:
        total = index.count()
        print(f"Index contains {total} finding models")
    """
    result = self.conn.execute(
        "SELECT COUNT(*) FROM finding_models"
    ).fetchone()
    return result[0]

def count_search(
    self,
    pattern: str,
    match_type: Literal["exact", "prefix", "contains"] = "contains"
) -> int:
    """Get count of finding models matching search pattern.

    Args:
        pattern: Search pattern (will be normalized)
        match_type: How to match the pattern

    Returns:
        Number of matching finding models

    Example:
        count = index.count_search("abscess", match_type="contains")
        print(f"Found {count} models matching 'abscess'")
    """
    # Build WHERE clause using helper
    where_clause, sql_pattern, _ = self._build_slug_search_clause(pattern, match_type)

    result = self.conn.execute(
        f"SELECT COUNT(*) FROM finding_models WHERE {where_clause}",
        [sql_pattern]
    ).fetchone()

    return result[0]
```

**Acceptance Criteria**:
- [x] count() returns correct total (already existed, async)
- [x] count_search() works with all match types
- [x] Returns 0 for empty database
- [x] Efficient (doesn't fetch all rows)

---

### Phase 3: ID Generation

**Status**: ✅ COMPLETED
**Priority**: High

**Purpose**: Replace GitHub-based ID registry with Index database queries. The DuckDB Index already contains all existing models and attributes, so it IS the authoritative registry of used IDs.

**Current system** (to be replaced):
- `IdManager` fetches `ids.json` from GitHub with used OIFM and OIFMA IDs
- Generates random IDs and checks collisions against GitHub registry
- Requires network access

**New system**:
- Index database IS the registry (contains all models/attributes)
- Query database for existing IDs with given source
- Generate random ID and check collision in memory
- No network access needed

#### Task 3.1: Implement generate_model_id() Method

**File**: `src/findingmodel/duckdb_index.py`

Generate unique OIFM IDs (model IDs) by querying the Index with caching:

```python
class DuckDBIndex:
    def __init__(self, db_path: Path | None = None):
        # ... existing init ...
        # Cache for existing IDs (loaded once per source, updated as we generate)
        self._oifm_id_cache: dict[str, set[str]] = {}  # {source: {id, ...}}
        self._oifma_id_cache: dict[str, set[str]] = {}  # {source: {id, ...}}

    def _load_oifm_ids_for_source(self, source: str) -> set[str]:
        """Load all existing OIFM IDs for a source from database (cached).

        Args:
            source: The source code (already validated)

        Returns:
            Set of existing OIFM IDs for this source
        """
        if source in self._oifm_id_cache:
            return self._oifm_id_cache[source]

        # Query database once
        existing_ids = self.conn.execute(
            "SELECT oifm_id FROM finding_models WHERE oifm_id LIKE ?",
            [f"OIFM_{source}_%"]
        ).fetchall()

        # Cache the result
        self._oifm_id_cache[source] = {row[0] for row in existing_ids}
        return self._oifm_id_cache[source]

    def generate_model_id(self, source: str = "OIDM", max_attempts: int = 100) -> str:
        """Generate unique OIFM ID by querying Index database.

        Replaces GitHub-based ID registry. The Index database already contains
        all existing models, so we query it to get used IDs and check collisions
        in memory. The ID set is cached per source and updated as we generate
        new IDs to avoid stepping on our own feet.

        Args:
            source: 3-4 uppercase letter code for originating organization
                    (default: "OIDM" for Open Imaging Data Model)
            max_attempts: Maximum collision retry attempts (default: 100)

        Returns:
            Unique OIFM ID in format: OIFM_{SOURCE}_{6_DIGITS}

        Raises:
            ValueError: If source is invalid (not 3-4 uppercase letters)
            RuntimeError: If unable to generate unique ID after max_attempts

        Example:
            # Generate new model ID with default source
            id1 = index.generate_model_id()  # "OIFM_OIDM_472951"

            # Generate with custom source
            id2 = index.generate_model_id("GMTS")  # "OIFM_GMTS_038572"

            # Generate multiple IDs - cache prevents collisions with ourselves
            id3 = index.generate_model_id("GMTS")  # Won't collide with id2

        Note:
            With 1,000,000 possible IDs per source and typical usage of a few
            thousand models, collision probability is <1%. One database query
            fetches all existing IDs for the source (cached per Index instance),
            then collision checking happens in memory for efficiency.
        """
        from findingmodel.finding_model import _random_digits

        # Validate and normalize source
        source_upper = source.strip().upper()
        if not (3 <= len(source_upper) <= 4 and source_upper.isalpha()):
            raise ValueError(f"Source must be 3-4 uppercase letters, got: {source}")

        # Load existing IDs for this source (cached)
        existing_set = self._load_oifm_ids_for_source(source_upper)

        # Generate random ID until we find one not in use
        for attempt in range(max_attempts):
            candidate_id = f"OIFM_{source_upper}_{_random_digits(6)}"

            if candidate_id not in existing_set:
                # Add to cache so we don't reuse it ourselves
                existing_set.add(candidate_id)
                return candidate_id

            logger.debug(f"ID collision on attempt {attempt + 1}: {candidate_id}")

        # Exhausted max attempts
        raise RuntimeError(
            f"Failed to generate unique ID after {max_attempts} attempts. "
            f"ID space for {source_upper} may be nearly exhausted."
        )

    def _load_oifma_ids_for_source(self, source: str) -> set[str]:
        """Load all existing OIFMA IDs for a source from database (cached).
        
        Args:
            source: The source code (already validated)
            
        Returns:
            Set of existing OIFMA IDs for this source
        """
        if source in self._oifma_id_cache:
            return self._oifma_id_cache[source]
        
        # Query database once
        existing_ids = self.conn.execute(
            "SELECT attribute_id FROM attributes WHERE attribute_id LIKE ?",
            [f"OIFMA_{source}_%"]
        ).fetchall()
        
        # Cache the result
        self._oifma_id_cache[source] = {row[0] for row in existing_ids}
        return self._oifma_id_cache[source]

    def generate_attribute_id(
        self,
        model_oifm_id: str | None = None,
        source: str | None = None,
        max_attempts: int = 100
    ) -> str:
        """Generate unique OIFMA ID by querying Index database.

        Replaces GitHub-based ID registry. Attribute IDs (OIFMA) identify
        individual attributes within finding models. Source can be inferred
        from the parent model's OIFM ID or provided explicitly.
        
        The ID set is cached per source and updated as we generate new IDs
        to avoid stepping on our own feet when generating multiple IDs.

        Args:
            model_oifm_id: Parent model's OIFM ID (source will be inferred)
            source: Explicit 3-4 uppercase letter source code (overrides inference)
            max_attempts: Maximum collision retry attempts (default: 100)

        Returns:
            Unique OIFMA ID in format: OIFMA_{SOURCE}_{6_DIGITS}

        Raises:
            ValueError: If source is invalid or cannot be inferred
            RuntimeError: If unable to generate unique ID after max_attempts

        Example:
            # Infer source from parent model
            attr_id = index.generate_attribute_id(model_oifm_id="OIFM_GMTS_123456")
            # Returns: "OIFMA_GMTS_472951" (same source as parent)

            # Explicit source for new model
            attr_id = index.generate_attribute_id(source="OIDM")
            # Returns: "OIFMA_OIDM_382746"
            
            # Generate multiple attributes - cache prevents self-collision
            attr_id2 = index.generate_attribute_id(source="OIDM")
            # Won't collide with attr_id

        Note:
            Value codes (OIFMA_XXX_NNNNNN.0, OIFMA_XXX_NNNNNN.1, etc.) are
            automatically generated from attribute IDs by the model editor.
            This method only generates the base attribute ID.
        """
        from findingmodel.finding_model import _random_digits

        # Determine source
        if source:
            resolved_source = source.strip().upper()
        elif model_oifm_id:
            # Infer from parent model: OIFM_GMTS_123456 → GMTS
            parts = model_oifm_id.split("_")
            if len(parts) != 3 or parts[0] != "OIFM":
                raise ValueError(f"Cannot infer source from invalid model ID: {model_oifm_id}")
            resolved_source = parts[1]
        else:
            resolved_source = "OIDM"

        # Validate source
        if not (3 <= len(resolved_source) <= 4 and resolved_source.isalpha()):
            raise ValueError(f"Source must be 3-4 uppercase letters, got: {resolved_source}")

        # Load existing IDs for this source (cached)
        existing_set = self._load_oifma_ids_for_source(resolved_source)

        # Generate random ID until we find one not in use
        for attempt in range(max_attempts):
            candidate_id = f"OIFMA_{resolved_source}_{_random_digits(6)}"

            if candidate_id not in existing_set:
                # Add to cache so we don't reuse it ourselves
                existing_set.add(candidate_id)
                return candidate_id

            logger.debug(f"Attribute ID collision on attempt {attempt + 1}: {candidate_id}")

        raise RuntimeError(
            f"Failed to generate unique attribute ID after {max_attempts} attempts. "
            f"ID space for {resolved_source} may be nearly exhausted."
        )
```

**Acceptance Criteria**:
- [x] Generates random IDs with collision checking
- [x] Different sources have independent ID spaces
- [x] Thread-safe (DuckDB transaction isolation)
- [x] Collision retry works (tested with pre-populated IDs)
- [x] RuntimeError raised when max_attempts exhausted
- [x] Attribute ID generation works independently
- [x] No collisions between different users' independently created IDs

---

### Phase 4: Testing

**Status**: ✅ COMPLETED for Phases 1, 2 & 3 (18 + 19 = 37 new tests added)
**Priority**: Critical (quality gate)

**Test Strategy**: Focus on testing our logic and integration sanity checks. We don't need to test DuckDB's internals, just ensure our code is wired up correctly. Local DB files make this much easier than MongoDB testing.

#### Test Coverage Requirements

**Manifest Fetching** (5-7 tests - focus on our logic):
- [ ] Successful fetch and parse
- [ ] Session caching prevents duplicate fetches
- [ ] Malformed JSON handled gracefully
- [ ] Network timeout handled
- [ ] Missing manifest URL raises ConfigurationError
- [ ] Fallback to direct URL works
- [ ] clear_manifest_cache() resets state

**JSON Storage** (6-8 tests - our logic + sanity checks):
- [ ] Roundtrip: JSON → DuckDB → FindingModelFull
- [ ] get_full() returns correct object
- [ ] get_full() raises KeyError for missing model
- [ ] get_full_batch() handles multiple IDs
- [ ] get_full_batch() handles empty list
- [ ] JSON updated when model updated
- [ ] Large JSON (complex attributes) works
- [ ] Unicode in JSON preserved

**list() Method** (6-8 tests - our API logic):
- [ ] Basic pagination works
- [ ] Returns correct total count
- [ ] Sorting by name works (case-insensitive)
- [ ] Sorting by oifm_id works
- [ ] Sorting by created_at works
- [ ] DESC sorting works
- [ ] Empty database returns ([], 0)
- [ ] Invalid order_by raises ValueError

**search_by_slug()** (7-9 tests - our search logic):
- [ ] Exact match works
- [ ] Prefix match works
- [ ] Contains match works
- [ ] Results ranked by relevance
- [ ] Pattern normalized before search
- [ ] Pagination works
- [ ] No matches returns ([], 0)
- [ ] Case-insensitive matching
- [ ] Special characters in pattern handled

**count() Methods** (4-5 tests - our counting logic):
- [ ] count() returns correct total
- [ ] count() returns 0 for empty database
- [ ] count_search() works with exact match
- [ ] count_search() works with contains match
- [ ] count_search() returns 0 for no matches

**ID Generation** (19 tests - our collision logic):
- [x] Generates random 6-digit IDs (confirmed: human-readable, ~20K total namespace)
- [x] IDs are unique (no collisions in large batch)
- [x] Different sources have independent ID spaces
- [x] Collision detection works (test with pre-populated IDs)
- [x] Retry logic works on collision
- [x] RuntimeError raised when max_attempts exhausted
- [x] Invalid source raises ValueError
- [x] Attribute ID generation works independently
- [x] Cache prevents self-collision when generating multiple IDs

**Integration Tests** (@pytest.mark.callout, 3-4 tests - end-to-end sanity checks):
- [ ] Full workflow: manifest fetch → DB download → list/search
- [ ] Real manifest URL fetch works
- [ ] Fallback to direct URL when manifest fails
- [ ] get_full() returns valid FindingModelFull from real data

**Total new tests**: ~35-45 tests (focusing on our logic, not DuckDB internals)

---

### Phase 5: Documentation

**Status**: ✅ COMPLETED for Phases 1, 2 & 3 (README.md updated with ID generation examples)
**Priority**: High (required for adoption)

**Approach**: Propose documentation with code snippets during implementation. Flesh out comprehensive docs after features are validated and working.

**Note**: No migration path needed - databases will be completely rebuilt with new schema. Build code must handle new architecture from scratch.

#### Task 5.1: Update README.md

Add sections with working code examples:

**Database Auto-Updates**:
```markdown
### Database Auto-Updates

The library automatically downloads the latest database indexes from a remote manifest. This happens transparently on first use.

To check current versions:
```bash
uv run python -m findingmodel db-info
```

To pin to specific versions (offline scenarios):
```bash
export REMOTE_INDEX_DB_URL="https://example.com/finding_models.duckdb"
export REMOTE_INDEX_DB_HASH="sha256:abc123..."
```

**New Index API Methods**:
```markdown
### Index API

#### Browsing Models

```python
from findingmodel import Index

index = Index()

# List all models with pagination
models, total = index.list(limit=20, offset=0, order_by="name")
print(f"Showing {len(models)} of {total} models")

# Search by name pattern
models, total = index.search_by_slug("abscess", match_type="contains")

# Get total count
total = index.count()

# Get full model with all attributes
full_model = index.get_full("OIFM_RADLEX_000001")
```

#### Creating Models

```python
# Generate next OIFM ID
new_id = index.generate_model_id("CUSTOM")  # "OIFM_CUSTOM_000001"
```

---

#### Task 5.2: API Usage Guide for Applications

**File**: `docs/index_api_guide.md`

Create guide for applications to use new Index API methods (especially FindingModelForge):

```markdown
# Index API Usage Guide

## Overview

The Index API provides complete methods for browsing, searching, and retrieving finding models without needing to access internal DuckDB connections.

## Usage Examples

### Before: Direct Connection Access

```python
# ❌ Old way - breaks abstraction
conn = index._ensure_connection()
rows = conn.execute("""
    SELECT oifm_id, name, slug_name FROM finding_models
    ORDER BY LOWER(name) LIMIT ? OFFSET ?
""", [limit, offset]).fetchall()

total = conn.execute("SELECT COUNT(*) FROM finding_models").fetchone()[0]
```

### After: Index API Methods

```python
# ✅ New way - uses Index API
models, total = index.list(limit=limit, offset=offset, order_by="name")
```

### Before: Pattern Search

```python
# ❌ Old way
pattern = f"%{normalize_name(search_term)}%"
conn = index._ensure_connection()
rows = conn.execute("""
    SELECT oifm_id, name FROM finding_models
    WHERE slug_name LIKE ? LIMIT ? OFFSET ?
""", [pattern, limit, offset]).fetchall()
```

### After: search_by_slug()

```python
# ✅ New way
models, total = index.search_by_slug(
    search_term,
    limit=limit,
    offset=offset,
    match_type="contains"
)
```

## Benefits

1. **Abstraction** - Don't depend on DuckDB internals
2. **Type Safety** - Get IndexEntry objects, not raw tuples
3. **Future-Proof** - API stable even if backend changes
4. **Better Performance** - Index can optimize internally

## Index Connection Patterns

DuckDBIndex supports two usage patterns for connection management:

### Pattern 1: Async Context Manager (Explicit Cleanup)

```python
async with Index() as index:
    # Get models
    result = await index.get("abdominal abscess")
    full = await index.get_full(result.oifm_id)

    # Get count
    count = await index.count()
# Connection automatically closed when exiting the async with block
```

**When to use:**
- When you want explicit control over connection lifecycle
- In long-running applications where you want to free resources
- When performing a discrete set of operations

### Pattern 2: Direct Instantiation (Lazy Cleanup)

```python
index = Index()

# Each method call ensures connection is open
result = await index.get("abdominal abscess")
full = await index.get_full(result.oifm_id)
count = await index.count()

# Connection stays open until index object is garbage collected
```

**When to use:**
- In scripts or notebooks where connection cleanup isn't critical
- When you'll be making many calls over time
- Default pattern for simplicity

**How it works:**
- Each method (get, get_full, count, etc.) calls `_ensure_connection()` internally
- Connections are read-only by default (lightweight, safe to keep open)
- DuckDB connections are single-file, no server processes
- Garbage collection will eventually close connections

**Both patterns are valid** - choose based on your needs. The context manager gives explicit control, while direct instantiation is simpler for most use cases.

## Deprecation Timeline

- v0.4.1: New API methods available, old patterns still work
- v0.4.2: Deprecation warnings for `_ensure_connection()` access
- v0.5.0: Make `_ensure_connection()` private (breaking change)
```

---

#### Task 5.3: Add db-info CLI Command

**File**: `src/findingmodel/cli.py`

Add command to check database status:

```python
@app.command()
def db_info() -> None:
    """Show database version information and status.

    Example:
        uv run python -m findingmodel db-info
    """
    from findingmodel.config import fetch_manifest, ensure_db_file, Config
    from rich.console import Console
    from rich.table import Table

    console = Console()
    settings = Config()

    # Finding models database
    try:
        local_index = ensure_db_file(
            settings.duckdb_index_path,
            settings.remote_index_db_url,
            settings.remote_index_db_hash,
            manifest_key="finding_models",
        )
        index_exists = local_index.exists()
        index_size_mb = local_index.stat().st_size / (1024 * 1024) if index_exists else 0
    except Exception as e:
        console.print(f"[red]Error checking finding models database: {e}[/red]")
        local_index = None
        index_exists = False
        index_size_mb = 0

    # Anatomic locations database
    try:
        local_anatomic = ensure_db_file(
            settings.duckdb_anatomic_path,
            settings.remote_anatomic_db_url,
            settings.remote_anatomic_db_hash,
            manifest_key="anatomic_locations",
        )
        anatomic_exists = local_anatomic.exists()
        anatomic_size_mb = local_anatomic.stat().st_size / (1024 * 1024) if anatomic_exists else 0
    except Exception as e:
        console.print(f"[red]Error checking anatomic locations database: {e}[/red]")
        local_anatomic = None
        anatomic_exists = False
        anatomic_size_mb = 0

    # Try to fetch manifest for version info
    manifest = None
    try:
        manifest = fetch_manifest()
    except Exception as e:
        console.print(f"[yellow]Could not fetch manifest: {e}[/yellow]")

    # Display results in table
    table = Table(title="Database Status")
    table.add_column("Database", style="cyan")
    table.add_column("Local Path", style="blue")
    table.add_column("Exists", style="green")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Remote Version", style="magenta")

    # Finding models row
    fm_version = "N/A"
    if manifest and "finding_models" in manifest:
        fm_version = manifest["finding_models"].get("version", "unknown")

    table.add_row(
        "Finding Models",
        str(local_index) if local_index else "N/A",
        "✓" if index_exists else "✗",
        f"{index_size_mb:.1f}" if index_exists else "-",
        fm_version
    )

    # Anatomic locations row
    al_version = "N/A"
    if manifest and "anatomic_locations" in manifest:
        al_version = manifest["anatomic_locations"].get("version", "unknown")

    table.add_row(
        "Anatomic Locations",
        str(local_anatomic) if local_anatomic else "N/A",
        "✓" if anatomic_exists else "✗",
        f"{anatomic_size_mb:.1f}" if anatomic_exists else "-",
        al_version
    )

    console.print(table)
```

**Acceptance Criteria**:
- [ ] Shows local database paths
- [ ] Shows existence and size
- [ ] Shows remote version from manifest
- [ ] Graceful when manifest unavailable
- [ ] Clear, readable output with Rich

---

#### Task 5.4: Update Configuration Documentation

**File**: `.env.sample`

Add new configuration options:

```bash
# Database Manifest Configuration
# Optional: Override manifest URL (default uses official hosting)
# REMOTE_MANIFEST_URL=https://findingmodelsdata.t3.storage.dev/manifest.json

# Optional: Pin database versions (bypasses manifest)
# Useful for offline scenarios or version pinning
# REMOTE_INDEX_DB_URL=https://example.com/finding_models.duckdb
# REMOTE_INDEX_DB_HASH=sha256:abc123...
# REMOTE_ANATOMIC_DB_URL=https://example.com/anatomic_locations.duckdb
# REMOTE_ANATOMIC_DB_HASH=sha256:def456...
```

---

#### Task 5.5: Create Serena Memory

**Memory Name**: `index_api_enhancements_2025`

Document the new patterns:

```markdown
# Index API Enhancements (January 2025)

## Overview
Major enhancements to DuckDB Index API to support real-world application needs.

## New Features

### 1. Manifest-Based Downloads
- Databases auto-update from remote manifest.json
- No library release needed for database updates
- Fallback to direct URL/hash for offline scenarios

### 2. Self-Contained Databases
- Full FindingModelFull JSON stored in DuckDB
- Single .duckdb file contains everything
- `get_full()` and `get_full_batch()` methods

### 3. Enhanced API Methods
- `list(limit, offset, order_by, order_dir)` - Paginated browsing
- `search_by_slug(pattern, match_type)` - Simple pattern search
- `count()` and `count_search()` - Efficient counts
- All return (results, total) for pagination UI

### 4. Local ID Generation
- `generate_model_id(source)` - Random generation with collision checking
- `generate_attribute_id(source)` - For attribute IDs
- No GitHub dependency, thread-safe via DuckDB locking
- **Random, not sequential** - prevents ID collisions when multiple users independently create models

## Migration from MongoDB

Applications using `_ensure_connection()` should migrate to new API methods.
See docs/migration_guide_index_api.md for examples.

## Configuration

```python
# Manifest URL (optional override)
remote_manifest_url: str | None

# Direct URL/hash (fallback/pinning)
remote_index_db_url: str | None
remote_index_db_hash: str | None
```

## CLI Commands

```bash
# Check database status
uv run python -m findingmodel db-info
```

## Testing Patterns

- Mock httpx for manifest tests
- Use pre-computed data for JSON storage tests
- Test pagination edge cases (empty, single page, last page)
- Verify thread safety of ID generation

## Performance Notes

- Manifest cached for process lifetime
- JSON storage adds ~10-50MB to database size
- list/search methods use indexed columns
- get_full_batch() more efficient than loop
```

---

## Success Criteria

### Must Have (Blocking) - v0.5.0 Release
- [ ] All existing tests pass
- [ ] 35+ new tests pass
- [ ] FindingModelForge can remove all `_ensure_connection()` calls
- [ ] Manifest fetch works with fallback
- [ ] JSON storage roundtrip verified
- [ ] Database build code handles new schema (finding_model_json table)
- [ ] API usage guide with working examples

### Nice to Have
- [ ] db-info command looks great
- [ ] Comprehensive code examples in documentation
- [ ] Serena memory comprehensive

## Risks & Mitigations

### Risk: Schema Changes Breaking Existing Databases
**Mitigation**:
- Complete database rebuild with new schema (no migration needed)
- Build code creates finding_model_json table from scratch
- New databases distributed via manifest.json auto-update
- Users get complete, consistent databases automatically

### Risk: Manifest Fetch Failures
**Mitigation**:
- Robust error handling
- Clear fallback path
- Session caching to minimize network calls
- Timeout configuration

### Risk: Breaking Changes for Existing Apps
**Mitigation**:
- New methods additive only
- Deprecate (don't remove) old patterns
- Comprehensive migration guide
- Version timeline for breaking changes

### Risk: Performance Regression
**Mitigation**:
- Benchmark before/after
- Optimize queries with EXPLAIN
- Use indexes on commonly queried fields
- Test with large datasets (1000+ models)

## Post-Implementation

### v0.5.0 Release Checklist
- [ ] All tests passing (`task test` and `task test-full`)
- [ ] Code formatted (`task check`)
- [ ] Documentation updated (README.md, API usage guide)
- [ ] CHANGELOG.md entry added
- [ ] Version bumped to 0.5.0 in pyproject.toml
- [ ] Database build code updated for new schema
- [ ] New databases built and uploaded to hosting
- [ ] Manifest.json updated and uploaded to hosting
- [ ] Git tag created: `git tag v0.5.0`
- [ ] GitHub release published

### Follow-up Tasks (Post v0.5.0)
- [ ] Monitor manifest fetch success rates
- [ ] Gather user feedback on new API
- [ ] Consider cursor-based pagination if needed for very large result sets
- [ ] Evaluate additional search methods if requested

## Key Design Decisions

### 1. Random ID Generation (not Sequential)
**Decision**: Use `random.randint(0, 999999)` with collision checking instead of sequential IDs.

**Rationale**:
- Multiple users independently creating models would generate conflicting sequential IDs
- When indexes are merged or models shared, sequential IDs collide
- Random generation with 1M possible IDs per source provides ample space
- Collision probability is <1% even with thousands of models
- Database check ensures uniqueness even in unlikely collision cases

**Implementation**:
- Generate random 6-digit number
- Check if ID exists in database
- Retry up to 100 times if collision
- Raise RuntimeError if space exhausted

### 2. Separate Methods (not Unified Search)
**Decision**: Distinct `list()`, `search_by_slug()`, `count()` methods instead of one polymorphic `search()`.

**Rationale**:
- Each method has single, clear purpose
- Easier to discover and document
- Different use cases: browse all vs. simple search vs. semantic search
- Explicit is better than implicit (Pythonic)

### 3. IndexEntry Return Type (not FindingModelFull)
**Decision**: `list()` and `search_by_slug()` return lightweight IndexEntry objects.

**Rationale**:
- Most UI needs just metadata (ID, name, description)
- Loading full JSON for every result is wasteful
- Users can call `get_full()` for specific models they want
- Keeps API performant for large result sets

### 4. JSON in DuckDB (not Separate Files)
**Decision**: Store full FindingModelFull JSON in model_json column.

**Rationale**:
- Self-contained database (single .duckdb file)
- Simpler deployment and distribution
- Atomic updates (metadata + JSON together)
- Only ~10-50MB overhead for thousands of models
- DuckDB has excellent JSON support

### 5. Manifest-Based Downloads (not Hardcoded URLs)
**Decision**: Fetch manifest.json at runtime to get latest database URLs.

**Rationale**:
- Decouple data updates from code releases
- Users automatically get latest databases
- Graceful fallback to direct URL for offline/pinning scenarios
- Industry standard pattern (npm, pip, apt all use manifests)

## Confirmed Decisions

1. **ID Generation**: 6-digit random IDs (human-readable, ~20K total namespace max, well under collision threshold with retry logic)

2. **Schema Migration**: Complete database rebuild (no migration path needed, build code handles new schema from scratch)

3. **Test Strategy**: Focus on our logic and integration sanity checks, not DuckDB internals (local DB makes testing easier)

4. **Documentation Timing**: Propose with code snippets during implementation, flesh out comprehensive docs after validation

5. **Release Approach**: Single v0.5.0 release (no phased rollout)

6. **Usage Telemetry**: Not implementing

7. **List/Search Return Type**: IndexEntry (lighter weight, use get_full() when full object needed)

8. **Manifest Versioning**: Yes, include "manifest_version": "1.0" field for backward compatibility

9. **JSON Schema Evolution**: Store raw JSON as-is, Pydantic handles version differences

10. **Pagination**: Offset-based (simpler), consider cursor-based later if performance issues arise

---

### Phase 6: Deprecate IdManager

**Status**: ⏸️ NOT STARTED
**Priority**: High (cleanup technical debt)

**Purpose**: Replace all usage of GitHub-based IdManager with Index-based ID generation. Index now provides all IdManager functionality using database queries instead of GitHub API calls.

**Current State Analysis**:

IdManager (`src/findingmodel/tools/add_ids.py`) currently:
- Fetches `ids.json` from GitHub with used IDs
- Provides `add_ids_to_model()` and `finalize_placeholder_attribute_ids()` methods
- Exported as singleton `id_manager` from tools module
- Used in `model_editor.py` for placeholder finalization
- Extensively tested in `test_tools.py`

Index (`src/findingmodel/index.py`) now provides:
- `generate_model_id(source)` - database-based ID generation with collision checking
- `generate_attribute_id(model_oifm_id, source)` - database-based attribute ID generation
- `add_ids_to_model(finding_model, source)` - high-level orchestration
- `finalize_placeholder_attribute_ids(finding_model, source)` - placeholder replacement

**Migration Strategy**:

1. **Backward Compatibility**: Keep IdManager working with deprecation warnings
2. **Internal Migration**: Update all internal code to use Index
3. **Test Updates**: Remove GitHub-specific tests, add Index-based tests
4. **Documentation**: Provide clear migration guide

**Key Technical Decisions**:

**Q: How to handle async/sync in convenience functions?**
**A**: Index ID generation methods are synchronous. Index() constructor is sync (connection is lazy via `_ensure_connection()`). We can create Index instances synchronously without calling `setup()` for ID generation use cases.

**Q: Should IdManager delegate to Index or be completely replaced?**
**A**: Keep IdManager as deprecated thin wrapper for transition period, but update all internal usage to Index directly.

**Q: What about the singleton pattern?**
**A**: Remove singleton export. Tools functions will create Index instances as needed (lightweight operation).

---

#### Task 6.1: Update model_editor.py to Use Index

**File**: `src/findingmodel/tools/model_editor.py`

**Current code** (lines 349-356):
```python
def _finalize_placeholder_ids(
    model: FindingModelFull,
    *,
    source: str | None = None,
    manager: IdManager | None = None,
) -> FindingModelFull:
    """Replace placeholder attribute IDs using the configured ID manager."""
    mgr = manager if manager is not None else id_manager
    return mgr.finalize_placeholder_attribute_ids(model, source=source)
```

**Changes needed**:

1. Change parameter from `manager: IdManager | None` to `index: Index | None`
2. Create module-level Index instance (lazy initialization)
3. Update function to use Index

**New implementation**:
```python
# At module level (after imports)
_index: Index | None = None

def _get_index() -> Index:
    """Get or create module-level Index instance."""
    global _index
    if _index is None:
        _index = Index()
    return _index

def _finalize_placeholder_ids(
    model: FindingModelFull,
    *,
    source: str | None = None,
    index: Index | None = None,
) -> FindingModelFull:
    """Replace placeholder attribute IDs using Index database queries.

    Args:
        model: Model with potential placeholder IDs
        source: Source code (inferred from model if None)
        index: Index instance (module default if None)

    Returns:
        Model with placeholders replaced by real IDs
    """
    idx = index if index is not None else _get_index()
    return idx.finalize_placeholder_attribute_ids(model, source=source)
```

**Imports to add**:
```python
from findingmodel import Index
```

**Imports to remove/deprecate**:
```python
from findingmodel.tools.add_ids import IdManager, id_manager  # Remove
```

**Acceptance Criteria**:
- [x] Module-level Index instance with lazy initialization
- [x] _finalize_placeholder_ids() uses Index parameter
- [x] No references to IdManager in model_editor.py
- [x] All callers updated (search for _finalize_placeholder_ids calls)

---

#### Task 6.2: Update tools/__init__.py Convenience Functions

**File**: `src/findingmodel/tools/__init__.py`

**Current exports** (lines 1, 17-18, 37):
```python
from .add_ids import id_manager
add_ids_to_model = id_manager.add_ids_to_model
add_ids_to_finding_model = id_manager.add_ids_to_finding_model
__all__ = [..., "id_manager", ...]
```

**Changes needed**:

1. Remove `id_manager` singleton export
2. Reimplement `add_ids_to_model` using Index
3. Keep `add_ids_to_finding_model` as deprecated alias

**New implementation**:
```python
# Remove: from .add_ids import id_manager

from findingmodel import Index
from findingmodel.finding_model import FindingModelBase, FindingModelFull

def add_ids_to_model(
    finding_model: FindingModelBase | FindingModelFull,
    source: str,
) -> FindingModelFull:
    """Generate and add IDs to a finding model using database-based ID generation.

    Replaces GitHub-based IdManager with Index database queries.

    Args:
        finding_model: Model to add IDs to (base or full)
        source: 3-4 uppercase letter source code

    Returns:
        FindingModelFull with all IDs generated

    Example:
        >>> from findingmodel.tools import add_ids_to_model
        >>> model = add_ids_to_model(base_model, "GMTS")
        >>> print(model.oifm_id)  # "OIFM_GMTS_472951"
    """
    index = Index()
    try:
        return index.add_ids_to_model(finding_model, source)
    finally:
        index.close()

def add_ids_to_finding_model(
    finding_model: FindingModelBase | FindingModelFull,
    source: str,
) -> FindingModelFull:
    """DEPRECATED: Use add_ids_to_model instead."""
    import warnings
    warnings.warn(
        "add_ids_to_finding_model is deprecated, use add_ids_to_model instead",
        DeprecationWarning,
        stacklevel=2
    )
    return add_ids_to_model(finding_model, source)
```

**__all__ updates**:
- Remove `"id_manager"`
- Keep `"add_ids_to_model"`, `"add_ids_to_finding_model"`

**Acceptance Criteria**:
- [x] add_ids_to_model() uses Index internally
- [x] Properly manages Index lifecycle (create, use, close)
- [x] No id_manager singleton export
- [x] Deprecated alias still works with warning
- [x] No breaking changes to public API

---

#### Task 6.3: Add Deprecation Warning to IdManager

**File**: `src/findingmodel/tools/add_ids.py`

**Changes needed**:

1. Add deprecation warning to `IdManager.__init__()`
2. Keep class functional for backward compatibility
3. Point users to Index in deprecation message

**New implementation**:
```python
class IdManager:
    """DEPRECATED: Use Index class for ID generation.

    This class is maintained for backward compatibility but will be removed
    in a future release. Please migrate to using Index:

    Old way:
        from findingmodel.tools import id_manager
        model = id_manager.add_ids_to_model(finding, "GMTS")

    New way:
        from findingmodel import Index
        index = Index()
        try:
            model = index.add_ids_to_model(finding, "GMTS")
        finally:
            index.close()

    Or use convenience function:
        from findingmodel.tools import add_ids_to_model
        model = add_ids_to_model(finding, "GMTS")
    """

    def __init__(self, url: str | None = None) -> None:
        import warnings
        warnings.warn(
            "IdManager is deprecated and will be removed in v0.6.0. "
            "Use Index.add_ids_to_model() and Index.finalize_placeholder_attribute_ids() instead. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        self.url = url or GITHUB_IDS_URL
        self.oifm_ids: dict[str, str] = {}
        self.attribute_ids: dict[str, tuple[str, str]] = {}
```

**Add to module docstring**:
```python
"""ID management utilities.

DEPRECATED: This module is deprecated in favor of Index-based ID generation.
Please use Index class from findingmodel.index for all ID generation needs.

Migration guide: See docs/migration_id_manager.md
"""
```

**Acceptance Criteria**:
- [x] DeprecationWarning raised on IdManager() construction
- [x] Clear migration message pointing to Index
- [x] Class remains functional during deprecation period
- [x] Module docstring documents deprecation

---

#### Task 6.4: Update Tests

**Files**: `test/test_tools.py`, `test/test_model_editor.py`

**Changes for test_tools.py**:

1. **Remove GitHub-specific tests**:
   - `test_add_ids_from_github` (lines 68-83)
   - `test_add_ids_multiple_attributes` (lines 100-114)
   - `test_finalize_placeholder_attribute_ids_from_github` (lines 122-136)
   - `test_id_manager_load_from_github` (lines 262-282)
   - `test_id_manager_load_refresh` (lines 285-306)
   - `test_id_manager_custom_url` (lines 309-322)

   **Rationale**: These test GitHub API integration which is no longer used.

2. **Update add_ids_to_model tests** to use Index:
   - `test_add_ids_to_finding` - update to use Index directly
   - `test_add_ids_with_value_codes` - update to use Index directly
   - Keep testing the convenience function from tools/__init__.py

3. **Add deprecation warning tests**:
   ```python
   def test_id_manager_deprecation_warning():
       """Test that IdManager raises deprecation warning."""
       with pytest.warns(DeprecationWarning, match="IdManager is deprecated"):
           from findingmodel.tools.add_ids import IdManager
           _ = IdManager()
   ```

4. **Remove singleton tests**:
   - `test_id_manager_singleton` (line 808) - no longer a singleton

**Changes for test_model_editor.py**:

1. **Update _finalize_placeholder_ids tests**:
   - Lines 147-163: Update mock to use Index instead of IdManager
   - Lines 188-210: Update to pass Index parameter
   - Lines 222-227: Update to use Index

2. **Add Index parameter tests**:
   ```python
   def test_finalize_placeholder_with_custom_index():
       """Test _finalize_placeholder_ids accepts Index parameter."""
       from findingmodel import Index
       index = Index()
       # ... test with custom index
   ```

**Acceptance Criteria**:
- [x] GitHub-specific tests removed (6 tests)
- [x] add_ids_to_model tests use Index
- [x] Deprecation warning tests added
- [x] model_editor tests updated for Index parameter
- [x] All updated tests pass
- [x] Test count reduced by ~6, add ~2 new tests

---

#### Task 6.5: Update Documentation

**Files to Update**:

**1. README.md** - Add migration section:

```markdown
### Migrating from IdManager (Deprecated)

The `IdManager` class is deprecated in favor of `Index`-based ID generation.

**Old pattern** (deprecated):
```python
from findingmodel.tools import id_manager

# This now raises DeprecationWarning
model = id_manager.add_ids_to_model(finding, "GMTS")
```

**New pattern** (recommended):
```python
from findingmodel import Index

index = Index()
try:
    model = index.add_ids_to_model(finding, "GMTS")
finally:
    index.close()
```

**Convenience function** (simplest):
```python
from findingmodel.tools import add_ids_to_model

# Automatically manages Index lifecycle
model = add_ids_to_model(finding, "GMTS")
```

**Benefits of new approach**:
- No GitHub API dependency (works offline)
- Database-based collision detection
- Better performance (local DB queries vs network)
- Same API surface, better implementation
```

**2. CHANGELOG.md** - Add deprecation notice:

```markdown
### Deprecated

- `IdManager` class - Use `Index.add_ids_to_model()` and `Index.finalize_placeholder_attribute_ids()` instead
- `id_manager` singleton export from `findingmodel.tools` - Use convenience functions or create Index instances
- `IdManager.load_used_ids_from_github()` - No longer needed, Index queries database directly

### Migration Guide

IdManager functionality has been moved to the Index class:

| Old (IdManager) | New (Index) |
|----------------|-------------|
| `id_manager.add_ids_to_model(model, source)` | `index.add_ids_to_model(model, source)` |
| `id_manager.finalize_placeholder_attribute_ids(model, source)` | `index.finalize_placeholder_attribute_ids(model, source)` |
| `IdManager.load_used_ids_from_github()` | Not needed - Index uses database |

See README.md for complete migration examples.
```

**3. Create docs/migration_id_manager.md**:

Comprehensive migration guide with:
- Why the change was made
- Step-by-step migration examples
- Troubleshooting common issues
- Timeline for removal (target v0.6.0)

**Acceptance Criteria**:
- [x] README.md has migration section
- [x] CHANGELOG.md documents deprecation
- [x] Migration guide created
- [x] Examples are tested and working

---

**Phase 6 Success Criteria**:

- [x] All internal code uses Index (no IdManager usage)
- [x] IdManager deprecated with clear warnings
- [x] Convenience functions work with Index
- [x] GitHub-specific tests removed
- [x] All tests pass
- [x] Documentation updated
- [x] Backward compatibility maintained during deprecation period

**Estimated Effort**: 4-6 hours
- Task 6.1: 30 min (update model_editor)
- Task 6.2: 45 min (update tools/__init__.py)
- Task 6.3: 15 min (add deprecation warning)
- Task 6.4: 2 hours (update extensive tests)
- Task 6.5: 1 hour (documentation)

**Removal Timeline**:
- v0.5.0: Deprecation warnings added
- v0.5.x: Deprecation period (6+ months) - **ACCELERATED**: User confirms only add_ids_to_model() is used
- v0.6.0: IdManager removed completely - **ACCELERATED TO v0.5.0**

---

## Phase 7: Complete IdManager Removal (Accelerated)

**Status**: 📋 Ready to Execute
**Goal**: Remove deprecated IdManager code entirely since main application only uses add_ids_to_model()
**Timeline**: Immediate (v0.5.0) - user confirmed safe to remove

### Rationale for Accelerated Removal

**User Confirmation**: Main calling application only uses `add_ids_to_model()`, which we've already reimplemented in tools/__init__.py using Index. No code uses IdManager directly.

**Risk Assessment**: **LOW**
- Internal code already migrated (Phase 6)
- Tests already updated (Phase 6)
- Only 1 external dependency (notebook import)
- Breaking changes acceptable (deprecated code removal)
- Users get immediate import errors → clear signal to use migration guide

### Analysis

**Current State After Phase 6**:
- ✅ Internal code migrated to Index
- ✅ Tests updated (removed 8 GitHub-specific tests)
- ✅ Documentation has migration guide
- ✅ Deprecation warnings in place
- ⚠️ add_ids.py still exists (~240 lines of deprecated code)

**External Dependencies Found**:
1. `scripts/edit_finding_model.py` line 26: imports PLACEHOLDER_ATTRIBUTE_ID from add_ids
2. httpx dependency: Used by other modules (ontology_search, config, anatomic_migration), so **KEEP IT**

**What Gets Removed**:
- IdManager class (GitHub-based ID generation)
- id_manager singleton
- PLACEHOLDER_ATTRIBUTE_ID duplicate (canonical version in index.py)
- GitHub API integration (~240 lines total)

---

### Task 7.1: Update Notebook Import

**File**: `scripts/edit_finding_model.py`

**Change** (line 26):
```python
# OLD
from findingmodel.tools.add_ids import PLACEHOLDER_ATTRIBUTE_ID

# NEW
from findingmodel.index import PLACEHOLDER_ATTRIBUTE_ID
```

**Rationale**: PLACEHOLDER_ATTRIBUTE_ID is defined in index.py (line 36), the canonical location.

**Acceptance Criteria**:
- [ ] Import updated
- [ ] Notebook runs without errors

---

### Task 7.2: Delete add_ids.py

**File**: `src/findingmodel/tools/add_ids.py`

**Action**: Delete entire file (~240 lines)

**Contents Being Removed**:
- `IdManager` class with GitHub API integration
- `id_manager` singleton instance
- `PLACEHOLDER_ATTRIBUTE_ID` constant (duplicate)
- `GITHUB_IDS_URL` constant
- Helper methods: `_generate_unique_oifm()`, `_generate_unique_oifma()`, `_validate_source()`

**Replacement**: All functionality available via Index class:
- `id_manager.add_ids_to_model()` → `index.add_ids_to_model()` or `tools.add_ids_to_model()`
- `id_manager.finalize_placeholder_attribute_ids()` → `index.finalize_placeholder_attribute_ids()`
- GitHub ID checking → Database-backed collision detection

**Acceptance Criteria**:
- [ ] File deleted
- [ ] No import errors when importing findingmodel
- [ ] All tests pass

---

### Task 7.3: Update Documentation

**Files**: `README.md`, `CHANGELOG.md`

**README.md Changes**:

Remove migration section (lines 239-271) and replace with brief note:

**DELETE**:
```markdown
### Migrating from IdManager (Deprecated)

The `IdManager` class is deprecated in favor of `Index`-based ID generation and will be removed in v0.6.0.
...
[entire migration section]
```

**ADD** (brief note in same location):
```markdown
> **Note**: Prior to v0.5.0, ID generation used a GitHub-based `IdManager`. This has been replaced with database-backed ID generation via Index. Use `add_ids_to_model()` from `findingmodel.tools` for the simplest API.
```

**CHANGELOG.md Changes**:

Update to reflect immediate removal:

**Change** (lines 41-43):
```markdown
### Deprecated

- **`IdManager` class** - Use `Index.add_ids_to_model()` and `Index.finalize_placeholder_attribute_ids()` instead (will be removed in v0.6.0)
- **`id_manager` singleton** export from `findingmodel.tools` - Use convenience functions or create Index instances
- **`IdManager.load_used_ids_from_github()`** - No longer needed, Index queries database directly
```

**TO**:
```markdown
### Removed

- **`IdManager` class** - Removed in v0.5.0. Use `Index.add_ids_to_model()` and `Index.finalize_placeholder_attribute_ids()` instead
- **`id_manager` singleton** - Removed in v0.5.0. Use `findingmodel.tools.add_ids_to_model()` convenience function
- **GitHub-based ID generation** - Replaced with database-backed ID generation via Index
- **`src/findingmodel/tools/add_ids.py`** - Module removed entirely
```

Remove the "Migration Guide" table (lines 45-61) since IdManager no longer exists.

**Acceptance Criteria**:
- [ ] README updated with brief note
- [ ] CHANGELOG shows "Removed" not "Deprecated"
- [ ] Migration guide removed (no longer needed)

---

### Task 7.4: Verify httpx Dependency

**Action**: Confirm httpx is used by other modules and should remain

**Other httpx Usage** (verified):
- `src/findingmodel/tools/ontology_search.py` - BioPortal API calls
- `src/findingmodel/config.py` - Manifest downloads
- `src/findingmodel/anatomic_migration.py` - HTTP operations

**Decision**: **KEEP httpx** in dependencies.

**Acceptance Criteria**:
- [x] Verified httpx is used elsewhere (confirmed above)
- [x] No changes to pyproject.toml needed

---

### Task 7.5: Run Full Test Suite

**Commands**:
```bash
uv run pytest test/ -m "not callout" -q  # 412 fast tests
task check                                # Format + lint + mypy
```

**Expected Results**:
- ✅ 412/412 tests pass
- ✅ No import errors
- ✅ No linting/formatting errors
- ✅ No type checking errors

**Acceptance Criteria**:
- [ ] All tests pass
- [ ] No import errors
- [ ] Linting passes
- [ ] Type checking passes

---

### Phase 7 Success Criteria

**Complete Removal**:
- [ ] `add_ids.py` deleted (~240 lines removed)
- [ ] Notebook import updated (1 line changed)
- [ ] No broken imports anywhere
- [ ] All 412 tests pass
- [ ] Documentation updated (removed migration guide)
- [ ] httpx dependency retained (used elsewhere)

**Benefits**:
- ✅ ~240 lines of deprecated code removed
- ✅ No GitHub API dependency for ID generation
- ✅ Cleaner codebase
- ✅ No confusing dual implementations
- ✅ Clear error messages for anyone still importing IdManager

**Timeline**: Single task, ~20 minutes implementation

**Deliverable**: Complete removal with no backward compatibility burden

---

## References

- **Request**: tasks/findingmodel_index_api_requests.md
- **Manifest Plan**: tasks/drive_to_041.md
- **Current Index**: src/findingmodel/duckdb_index.py
- **Tests**: test/test_duckdb_index.py
- **FindingModelForge**: External application requesting these features
