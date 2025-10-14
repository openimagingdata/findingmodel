# Drive to v0.4.1 Release

**Goal**: Add JSON manifest for runtime database version discovery

**Status**: Planning
**Created**: 2025-10-14

## Overview

v0.4.1 adds a lightweight JSON manifest pattern to enable database updates without requiring new library releases. Users will automatically get the latest database versions by fetching a manifest at runtime, while still supporting direct URL/hash configuration for offline or custom scenarios.

## Background

After v0.4.0 ships with direct URL/hash configuration (like anatomic locations currently use), we'll have some production experience with:
- How often database indexes need updates
- Whether users prefer auto-updates or pinned versions
- Network/reliability considerations for manifest fetching

This information will help refine the manifest implementation.

## Rationale

**Problem**: With v0.4.0's direct URL/hash config, updating databases requires:
1. Building new database
2. Computing SHA256 hash
3. Releasing new library version with updated config
4. Users installing updated library

**Solution**: Runtime manifest fetch:
1. Build new database
2. Compute SHA256 hash
3. Update manifest JSON on hosting (no code release)
4. Users automatically get latest on next session

**User benefit**: Always get current database indexes without waiting for library updates.

## Implementation Plan

### Phase 1: Manifest Fetch Infrastructure

**Goal**: Add lightweight runtime config fetching

#### Implementation Tasks

1. **Add httpx dependency** to `pyproject.toml`:
   ```toml
   dependencies = [
       "httpx>=0.27.0",  # For manifest fetching
       # ... existing deps
   ]
   ```

2. **Add config field** in `config.py`:
   ```python
   remote_manifest_url: str | None = Field(
       default="https://findingmodelsdata.t3.storage.dev/manifest.json",
       description="URL to JSON manifest for database versions"
   )
   ```

3. **Create `fetch_manifest()` function** in `config.py`:
   ```python
   def fetch_manifest() -> dict[str, Any]:
       """Fetch and parse the remote manifest JSON.

       Returns:
           Parsed manifest with database version info

       Raises:
           ConfigurationError: If manifest URL not configured
           httpx.HTTPError: If fetch fails
       """
       import httpx

       if not settings.remote_manifest_url:
           raise ConfigurationError("Manifest URL not configured")

       response = httpx.get(settings.remote_manifest_url, timeout=10)
       response.raise_for_status()
       return response.json()
   ```

4. **Update `ensure_db_file()`** to support manifest:
   ```python
   def ensure_db_file(
       filename: str,
       remote_url: str | None,
       remote_hash: str | None,
       manifest_key: str | None = None,
   ) -> Path:
       """Download DB file with manifest support.

       Args:
           filename: Database filename
           remote_url: Direct URL (backward compat)
           remote_hash: Direct hash (backward compat)
           manifest_key: Key in manifest JSON (e.g., "finding_models")

       Priority:
           1. Use existing local file if present
           2. Try manifest fetch if key provided
           3. Fall back to direct URL/hash
       """
       # ... existing local check ...

       # Try manifest first
       if manifest_key:
           try:
               manifest = fetch_manifest()
               db_info = manifest.get(manifest_key)
               if db_info:
                   url = db_info["url"]
                   hash_val = db_info["hash"]
                   logger.info(f"Using manifest version {db_info.get('version', 'unknown')}")
                   # ... Pooch download with manifest URL/hash ...
                   return downloaded_path
           except Exception as e:
               logger.warning(f"Manifest fetch failed, trying direct URL: {e}")

       # Fall back to direct URL/hash
       if remote_url and remote_hash:
           # ... existing direct download logic ...
   ```

5. **Add session-level manifest caching**:
   ```python
   _manifest_cache: dict[str, Any] | None = None

   def fetch_manifest() -> dict[str, Any]:
       """Fetch manifest with session caching."""
       global _manifest_cache

       if _manifest_cache is not None:
           return _manifest_cache

       # ... fetch logic ...
       _manifest_cache = response.json()
       return _manifest_cache
   ```

**Success Criteria**:
- Manifest fetch works with valid URL
- Graceful fallback to direct URL/hash when manifest unavailable
- No repeated fetches in same session
- Clear error messages for network/parse failures

### Phase 2: CLI Database Info Command

**Goal**: Let users check database versions

#### Implementation Tasks

1. **Add `db-info` command** in `cli.py`:
   ```python
   @app.command()
   def db_info() -> None:
       """Show database version information."""
       from findingmodel.config import fetch_manifest, ensure_db_file

       # Finding models database
       local_index = ensure_db_file(
           settings.duckdb_index_path,
           settings.remote_index_db_url,
           settings.remote_index_db_hash,
           manifest_key="finding_models",
       )

       # Anatomic locations database
       local_anatomic = ensure_db_file(
           settings.duckdb_anatomic_path,
           settings.remote_anatomic_db_url,
           settings.remote_anatomic_db_hash,
           manifest_key="anatomic_locations",
       )

       # Fetch manifest for version info
       try:
           manifest = fetch_manifest()
           show_db_status("Finding Models", local_index, manifest.get("finding_models"))
           show_db_status("Anatomic Locations", local_anatomic, manifest.get("anatomic_locations"))
       except Exception as e:
           logger.error(f"Could not fetch manifest: {e}")

   def show_db_status(name: str, local_path: Path, remote_info: dict | None) -> None:
       """Display status for one database."""
       print(f"\n{name}:")
       print(f"  Local: {local_path}")
       print(f"  Exists: {local_path.exists()}")

       if local_path.exists():
           size_mb = local_path.stat().st_size / (1024 * 1024)
           print(f"  Size: {size_mb:.1f} MB")

       if remote_info:
           print(f"  Remote version: {remote_info.get('version', 'unknown')}")
           print(f"  Remote size: {remote_info.get('size_mb', 'unknown')} MB")
   ```

**Success Criteria**:
- Shows local file paths and existence
- Shows remote version info when manifest available
- Graceful when manifest unavailable
- Clear, readable output

### Phase 3: Sample Manifest and Documentation

**Goal**: Provide reference implementation and update patterns

#### Implementation Tasks

1. **Create `data/manifest.json.sample`** in repo:
   ```json
   {
     "finding_models": {
       "version": "2025-10-14",
       "url": "https://findingmodelsdata.t3.storage.dev/finding_models_2025-10-14.duckdb",
       "hash": "sha256:abc123...",
       "size_mb": 12.5,
       "record_count": 1234,
       "description": "Finding model index with embeddings"
     },
     "anatomic_locations": {
       "version": "2025-10-10",
       "url": "https://findingmodelsdata.t3.storage.dev/anatomic_locations_2025-10-10.duckdb",
       "hash": "sha256:def456...",
       "size_mb": 45.2,
       "record_count": 5678,
       "description": "Anatomic location ontologies with embeddings"
     }
   }
   ```

2. **Document manifest pattern** in README.md:
   ```markdown
   ### Database Auto-Updates

   The library automatically downloads the latest database indexes from a remote
   manifest. This happens transparently on first use.

   To check current versions:
   ```bash
   uv run python -m findingmodel db-info
   ```

   To pin to specific versions (offline scenarios):
   ```bash
   export REMOTE_INDEX_DB_URL="https://example.com/finding_models.duckdb"
   export REMOTE_INDEX_DB_HASH="sha256:abc123..."
   ```

3. **Add troubleshooting section** for manifest fetch failures

4. **Update `.env.sample`**:
   ```
   # Optional: Override manifest URL (default uses official hosting)
   # REMOTE_MANIFEST_URL=https://example.com/manifest.json

   # Optional: Pin database versions (bypasses manifest)
   # REMOTE_INDEX_DB_URL=https://example.com/finding_models.duckdb
   # REMOTE_INDEX_DB_HASH=sha256:abc123...
   ```

**Success Criteria**:
- Sample manifest demonstrates schema
- README clearly explains auto-update behavior
- Users understand how to pin versions if needed
- Troubleshooting covers common issues

## Testing Strategy

### Unit Tests
- `fetch_manifest()` with mocked httpx responses
- `ensure_db_file()` priority logic (manifest → direct URL → error)
- Manifest caching behavior
- Error handling for network failures, invalid JSON, missing keys

### Integration Tests
- `db-info` command with real manifest
- Full download cycle using manifest
- Fallback to direct URL when manifest unavailable

### Edge Cases
- Malformed JSON in manifest
- Missing required fields (url, hash)
- Network timeout during fetch
- Manifest returns 404
- Hash mismatch after manifest download

## Release Checklist

Before tagging v0.4.1:

- [ ] All Phase 1 tasks complete (manifest infrastructure)
- [ ] All Phase 2 tasks complete (CLI command)
- [ ] All Phase 3 tasks complete (documentation)
- [ ] All tests passing (`task test`)
- [ ] Callout tests passing (`task test-full`)
- [ ] `db-info` command tested with real manifest
- [ ] Build succeeds (`task build`)
- [ ] Code formatted (`task check`)
- [ ] Production manifest.json uploaded to hosting
- [ ] Manifest URL validated (accessible, correct schema)
- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] Git tag created: `git tag v0.4.1`
- [ ] Release published on GitHub

## Post-Release

- [ ] Update Serena memory with manifest pattern details
- [ ] Monitor manifest fetch success/failure rates
- [ ] Gather user feedback on auto-update behavior
- [ ] Consider adding opt-out flag if users prefer manual updates

## Design Rationale

### Why httpx over requests?
- Modern, actively maintained
- Built-in timeout support
- Better async support (future-proofing)
- Similar API to requests (easy migration)

### Why session caching?
- Avoid repeated network calls in single session
- Manifest changes infrequently (daily/weekly at most)
- Simple `global` variable sufficient (no complex cache invalidation)

### Why manifest key parameter?
- Supports multiple databases from one manifest
- Backward compatible (optional parameter)
- Clear which database we're fetching

### Why fallback to direct URL?
- Offline scenarios (no network access)
- Corporate firewalls blocking manifest URL
- Users who prefer pinned versions
- Development/testing with custom databases

## Lessons from v0.4.0

After shipping v0.4.0, we'll know:
1. **Update frequency**: How often do we actually rebuild databases?
2. **User preferences**: Do users want auto-updates or control?
3. **Network reliability**: Any issues with Pooch downloads?
4. **Database sizes**: Are downloads too large for some users?

These learnings will refine the manifest implementation in v0.4.1.

## Notes

- **Lightweight scope**: Just JSON fetch, no versioning logic, update checks, or migrations
- **Backward compatible**: Direct URL/hash still works (takes priority for pinning)
- **Simple caching**: Session-level only, no persistent cache
- **Clear errors**: Network failures don't break existing functionality
- **User control**: Can opt out by setting direct URL/hash
