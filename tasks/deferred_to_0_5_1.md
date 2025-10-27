# Deferred to v0.5.1

Features and enhancements deferred from v0.5.0 to the next release.

## 1. Manifest and Database Distribution Enhancements

**Priority**: High
**Complexity**: Medium
**Deferred from**: Index API Enhancements Plan (v0.5.0)

### Overview

Comprehensive improvements to database distribution workflow including automated manifest generation, S3 upload integration, and version management. These enhancements will streamline database releases and enable reproducible deployments.

### 1.1 CLI Command for Manifest Generation

**Complexity**: Low

#### Description

Add CLI command to automatically generate manifest.json from built database files.

#### Current Gap

The `index build` command creates the database but doesn't generate a manifest.json file. Currently, manifest.json must be created manually.

#### Proposed Solution

Add new CLI command:

```bash
python -m findingmodel index generate-manifest \
  --database finding_models.duckdb \
  --url https://findingmodelsdata.t3.storage.dev/finding_models.duckdb \
  --output manifest.json
```

**Implementation:**
- Read database file to compute SHA256 hash
- Query database for record count
- Get file size from stat()
- Use current date for version
- Generate manifest.json following schema in docs/manifest_schema.md
- Support multiple databases (finding_models + anatomic_locations)

#### Workaround for v0.5.0

Generate manifest.json manually or with ad-hoc script when uploading databases to hosting.

### 1.2 S3 Upload and Deployment

**Complexity**: Medium

#### Description

Extend manifest generation to support direct S3 upload of database files and manifest, eliminating manual upload steps and potentially avoiding local file writes entirely.

#### Current Gap

Current workflow requires multiple manual steps:
1. Build database locally with `index build`
2. Manually generate manifest.json
3. Manually upload database files to S3/storage
4. Manually upload manifest.json

This is error-prone and requires local disk space for full database files.

#### Proposed Solution

Add integrated S3 upload capability:

```bash
# Upload databases and generate manifest in one command
python -m findingmodel index upload-to-s3 \
  --database finding_models.duckdb \
  --database anatomic_locations.duckdb \
  --s3-bucket findingmodelsdata \
  --s3-prefix "" \
  --base-url https://findingmodelsdata.t3.storage.dev \
  --generate-manifest \
  --manifest-output manifest.json
```

**Implementation:**
- Add S3 client integration (boto3 or aioboto3)
- Stream database files to S3 without requiring local storage
- Compute SHA256 hash during streaming upload
- Query database for metadata (record count) before upload
- Generate manifest.json with S3 URLs
- Upload manifest.json to S3
- Optionally write manifest.json locally for reference

**Configuration:**
```python
# In FindingModelConfig or env vars
s3_access_key_id: SecretStr | None
s3_secret_access_key: SecretStr | None
s3_region: str = "auto"  # For Cloudflare R2, AWS, etc.
s3_endpoint_url: str | None  # For S3-compatible services
```

**Benefits:**
- One-command deployment workflow
- No manual upload steps
- Streaming reduces local disk space requirements
- Atomic manifest + database updates
- Less error-prone (hashes computed automatically)
- Works with any S3-compatible storage (AWS S3, Cloudflare R2, MinIO, etc.)

**Advanced Features:**
- Support for multipart uploads for large files
- Progress bars for upload status
- Dry-run mode to preview without uploading
- Ability to update just manifest without re-uploading databases
- Automatic versioning with date stamps

**Workflow:**
1. Build database locally or in temp storage
2. Stream to S3 while computing hash
3. Query for metadata
4. Generate manifest with actual URLs and hashes
5. Upload manifest to S3
6. Clean up temp files
7. Users get new version automatically

### 1.3 Version History and Pinning

**Complexity**: Low

#### Description

Extend manifest.json schema to include a catalog of older database versions, enabling version pinning, rollback, and reproducibility.

#### Current Gap

Current manifest.json only includes the latest version of each database. Users cannot:
- Pin to specific database versions for reproducibility
- Roll back to previous versions if issues arise
- Compare current version against historical versions
- Access specific dated snapshots

#### Proposed Solution

Extend manifest schema to include version history:

```json
{
  "manifest_version": "1.0",
  "generated_at": "2025-10-26T14:02:11Z",
  "databases": {
    "finding_models": {
      "latest": {
        "version": "2025-10-26",
        "url": "https://findingmodelsdata.t3.storage.dev/finding_models_20251026.duckdb",
        "hash": "sha256:0a75653e...",
        "size_bytes": 34353152,
        "record_count": 1955,
        "description": "Finding model index with embeddings and full JSON"
      },
      "versions": [
        {
          "version": "2025-10-26",
          "url": "https://findingmodelsdata.t3.storage.dev/finding_models_20251026.duckdb",
          "hash": "sha256:0a75653e...",
          "size_bytes": 34353152,
          "record_count": 1955,
          "released_at": "2025-10-26T14:02:11Z"
        },
        {
          "version": "2025-10-17",
          "url": "https://findingmodelsdata.t3.storage.dev/finding_models_20251017.duckdb",
          "hash": "sha256:86e52f7c...",
          "size_bytes": 32145600,
          "record_count": 1850,
          "released_at": "2025-10-17T10:30:00Z"
        }
      ]
    }
  }
}
```

**Implementation:**
- Add `latest` object for current version (backward compatible)
- Add `versions` array with full history
- Include `released_at` timestamp for each version
- Update upload command to append to history, not replace
- Add `--keep-versions N` flag to limit history size
- Support version pinning in config:
  ```python
  duckdb_index_version: str | None = None  # Pin to specific version
  ```

**Benefits:**
- **Reproducibility**: Pin exact database version in research/production
- **Rollback**: Revert to previous version if issues found
- **Testing**: Test against multiple database versions
- **Comparison**: Analyze changes between versions
- **Audit trail**: Track when databases were updated

**Configuration:**
```python
# In FindingModelConfig
duckdb_index_version: str | None = Field(
    default=None,
    description="Pin to specific database version (e.g., '2025-10-17'), or None for latest"
)
duckdb_anatomic_version: str | None = Field(
    default=None,
    description="Pin to specific anatomic locations version, or None for latest"
)
```

**Usage:**
```python
# Use latest (default)
index = DuckDBIndex()

# Pin to specific version via env
# DUCKDB_INDEX_VERSION=2025-10-17
index = DuckDBIndex()

# Or programmatically
config = FindingModelConfig(duckdb_index_version="2025-10-17")
```

**Backward Compatibility:**
- If no `versions` array exists, fall back to top-level fields (current schema)
- If `latest` object exists, use it unless version pinned
- CLI can detect old vs. new schema and handle both

### Related Files

- `docs/manifest_schema.md` - Schema specification (update for v1.1 with version history)
- `src/findingmodel/cli.py` - New commands (generate-manifest, upload-to-s3)
- `src/findingmodel/config.py` - S3 credentials, version pinning configuration
- `pyproject.toml` - Add boto3/aioboto3 as optional dependency
- `tasks/index_api_enhancements_plan.md` - Original plan

---

## 2. Search Type Parameter

**Priority**: Medium
**Complexity**: Low
**Deferred from**: v0.5.0 implementation

### Description

Add `search_type` parameter to search methods to support different search strategies and graceful degradation when OpenAI API key is unavailable.

### Current Gap

The `search()` method always uses hybrid search (BM25 + semantic embeddings), which requires an OpenAI API key. If the key is missing, search fails entirely even though FTS could still provide useful results.

### Proposed Solution

Add `search_type` parameter to search methods:

```python
async def search(
    self,
    query: str,
    limit: int = 10,
    search_type: Literal["hybrid", "semantic", "text"] = "hybrid"
) -> list[SearchResult]:
    """
    Search finding models.

    Args:
        query: Search query
        limit: Maximum results to return
        search_type:
            - "hybrid" (default): BM25 (0.3) + semantic (0.7) weighted search
            - "semantic": Vector-only search using embeddings
            - "text": FTS-only search using BM25
    """
```

**Implementation:**
- Default to "hybrid" for backward compatibility
- "text" mode bypasses embedding generation, works without API key
- "semantic" mode uses only vector search
- Update `_hybrid_search()` to support all three modes
- Add validation to check API key availability for semantic modes

**Benefits:**
- Graceful degradation when OpenAI API key unavailable
- Users can choose search strategy based on use case
- FTS-only mode is faster for simple text matching
- Vector-only mode for semantic similarity

### Related Files

- `src/findingmodel/index.py` - DuckDBIndex.search() and AnatomicIndex.search()
- `src/findingmodel/config.py` - API key availability check

---

## 3. Remote DuckDB Access via URL

**Priority**: Medium
**Complexity**: Medium
**Deferred from**: v0.5.0 implementation

### Description

Add option to open DuckDB files directly via URL without downloading, using DuckDB's httpfs extension for remote file access.

### Current Gap

The library always downloads database files to local storage before opening them. For read-only queries or temporary usage, this requires unnecessary disk space and download time.

### Proposed Solution

Add configuration option to use DuckDB's httpfs extension for direct URL access:

```python
# In FindingModelConfig
duckdb_use_remote_access: bool = Field(
    default=False,
    description="Open DuckDB files via URL without downloading (requires httpfs extension)"
)
```

**Implementation:**
- Install and load httpfs extension in DuckDB connection
- When `duckdb_use_remote_access=True`, skip Pooch download
- Open database directly: `duckdb.connect(database=url, read_only=True)`
- Validate hash via HEAD request metadata if available
- Fall back to download mode if httpfs not available or connection fails

**Benefits:**
- No local disk space required for databases
- Faster startup for one-off queries
- Always use latest database version from manifest
- Useful for serverless/containerized deployments

**Considerations:**
- Network latency on every query vs. one-time download
- Read-only access only (no writes)
- Requires stable internet connection
- May need caching strategy for frequently accessed data

**Trade-offs:**
- Default should remain download mode for performance
- Remote mode best for ephemeral environments or limited storage
- Could add hybrid mode: stream for reads, download for index building

### Related Files

- `src/findingmodel/index.py` - DuckDBIndex connection initialization
- `src/findingmodel/config.py` - Configuration option
- DuckDB httpfs extension documentation

---

## 4. Synchronous Index API Methods

**Priority**: Low
**Complexity**: Low
**Deferred from**: v0.5.0 Phase 2

### Description

Add synchronous versions of async Index methods for simpler usage patterns when async/await is not needed.

### Current Gap

All Index API methods are async (count, all, search_by_slug, count_search) even though DuckDB operations are synchronous. This forces callers to use async/await syntax unnecessarily.

### Proposed Solution

Add synchronous versions with `_sync` suffix:
```python
def all_sync(...) -> tuple[list[IndexEntry], int]:
    """Synchronous version of all()."""

def search_by_slug_sync(...) -> tuple[list[IndexEntry], int]:
    """Synchronous version of search_by_slug()."""

def count_sync() -> int:
    """Synchronous version of count()."""

def count_search_sync(...) -> int:
    """Synchronous version of count_search()."""
```

**Benefits**:
- Simpler API for synchronous contexts (CLI, notebooks, simple scripts)
- No async/await overhead when not needed
- Maintains backward compatibility with existing async methods

### Related Files
- src/findingmodel/index.py
