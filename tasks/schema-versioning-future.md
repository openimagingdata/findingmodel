# Plan 2: Schema Versioning (Future Work)

**Status**: Deferred until after initial release
**Depends on**: Plan 1 (Versioning, Publishing, and Release)

---

## Overview

This plan adds schema versioning to enable backward compatibility when database schemas change. The key insight is:

> **A package version requires exactly ONE schema version.**
> Backward compatibility comes from the manifest keeping old schema databases available.

---

## 1. Core Concepts

### Schema Version vs Content Version

- **Schema version**: Database structure (table columns, indexes) - changes when queries would break
- **Content version**: Data freshness (date-based) - changes when data is updated

### One Schema Per Package Version

```
findingmodel==0.7.0  →  requires schema "1.0"
findingmodel==0.8.0  →  requires schema "1.1"
```

Packages cannot "support multiple schemas" because:
- Code is compiled against specific query patterns
- Package cannot know about future schema changes

---

## 2. Database Metadata Table

Add `_metadata` table to all built databases:

```sql
CREATE TABLE _metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

INSERT INTO _metadata VALUES
    ('schema_version', '1.0'),
    ('content_version', '2025-01-22'),
    ('built_at', '2025-01-22T10:30:00Z'),
    ('built_by', 'oidm-maintenance 0.2.0'),
    ('record_count', '2149');
```

### Files to Modify

- `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/build.py`
- `packages/oidm-maintenance/src/oidm_maintenance/anatomic/build.py`

---

## 3. Manifest Schema v2.0

**Note**: Plan 1 separates storage so each package has its own T3 bucket. This simplifies schema versioning - each manifest only tracks one database.

### Separate Storage (from Plan 1)

| Package | Bucket | Manifest |
|---------|--------|----------|
| findingmodel | `findingmodelsdata.t3.storage.dev` | `manifest.json` |
| anatomic-locations | `anatomiclocationdata.t3.storage.dev` | `manifest.json` |

### Current (v1.0): Single entry

```json
{
  "manifest_version": "1.0",
  "databases": {
    "finding_models": {
      "version": "2025-01-22",
      "url": "...",
      "hash": "..."
    }
  }
}
```

### Proposed (v2.0): Array of schema entries

Each manifest has one database key with an array of schema versions:

```json
{
  "manifest_version": "2.0",
  "generated_at": "2025-01-22T10:30:00Z",
  "database": "finding_models",
  "entries": [
    {
      "schema_version": "1.1",
      "content_version": "2025-01-22",
      "url": "https://.../finding_models_1.1.duckdb",
      "hash": "sha256:...",
      "size_bytes": 52428800,
      "record_count": 2149
    },
    {
      "schema_version": "1.0",
      "content_version": "2025-01-22",
      "url": "https://.../finding_models_1.0.duckdb",
      "hash": "sha256:...",
      "size_bytes": 51000000,
      "record_count": 2149
    }
  ]
}
```

**Simplified**: No nested `databases` object needed since each manifest is package-specific.

---

## 4. Package Configuration

Each package declares its required schema:

```python
# findingmodel/config.py
REQUIRED_SCHEMA = "1.0"

# anatomic_locations/config.py
REQUIRED_SCHEMA = "1.0"
```

### Selection Logic

```python
def select_database_entry(
    manifest_entries: list[dict],
    required_schema: str
) -> dict:
    """Find database matching the required schema."""
    for entry in manifest_entries:
        if entry["schema_version"] == required_schema:
            return entry

    available = [e["schema_version"] for e in manifest_entries]
    raise DistributionError(
        f"No database for schema {required_schema}. "
        f"Available: {available}. Upgrade package?"
    )
```

### Files to Modify

- `packages/oidm-common/src/oidm_common/distribution/manifest.py` (read v2.0 format)
- `packages/findingmodel/src/findingmodel/config.py` (add REQUIRED_SCHEMA)
- `packages/anatomic-locations/src/anatomic_locations/config.py` (add REQUIRED_SCHEMA)

**Note**: Manifest URL separation is done in Plan 1, so each package already points to its own manifest.

---

## 5. Backward Compatibility Workflow

### When Schema Changes

1. **Release new package version** with updated `REQUIRED_SCHEMA`
2. **Build new database** with new schema
3. **Keep old schema database** in manifest
4. **Old packages** continue to find their schema

### Example Timeline

```
Jan 2025: findingmodel 0.7.0 uses schema 1.0
          manifest: [finding_models schema=1.0]

Feb 2025: Schema change needed
          - Release findingmodel 0.8.0 with REQUIRED_SCHEMA="1.1"
          - Build new schema 1.1 database
          - manifest: [schema=1.1, schema=1.0]

Mar 2025: Content update
          - Rebuild both 1.0 and 1.1 databases with new content
          - Or freeze 1.0 if content change needs new schema
```

---

## 6. Publishing Updates

### Build with Schema Version

```bash
# Build database (writes schema to _metadata table)
oidm-maintain findingmodel build --output finding_models.duckdb

# Publish (reads schema from _metadata, updates correct manifest entry)
oidm-maintain findingmodel publish finding_models.duckdb
```

### Manifest Management

- Publish reads `_metadata.schema_version` from database
- Updates/adds entry for that schema version
- Keeps other schema entries unchanged

### Files to Modify

- `packages/oidm-maintenance/src/oidm_maintenance/s3.py`
- `packages/oidm-maintenance/src/oidm_maintenance/findingmodel/publish.py`
- `packages/oidm-maintenance/src/oidm_maintenance/anatomic/publish.py`

---

## 7. Migration Path

### From v1.0 to v2.0 Manifest

1. First publish with new system creates v2.0 manifest
2. Old packages (pre-schema-versioning) need update to read new format
3. Option: Keep v1.0 manifest at old URL during transition

### Package Updates Required

Packages must update to:
1. Read array format (not object) for database entries
2. Filter by `REQUIRED_SCHEMA`
3. Handle `DistributionError` when schema not found

---

## 8. Implementation Phases

### Phase A: Add metadata table to databases
- Update build scripts to create `_metadata` table
- No breaking changes

### Phase B: Add REQUIRED_SCHEMA to packages
- Add constant to config files
- No behavior change yet (just preparation)

### Phase C: Update manifest format to v2.0
- Update oidm-common to read array format
- Update publishing to write array format
- Add `select_database_entry()` function

### Phase D: Enable schema selection
- Wire up REQUIRED_SCHEMA to database selection
- Test with multiple schema versions

---

## 9. Open Questions

1. **Schema version format**: Semver ("1.0") vs date-based ("2025-01")?
2. **Content freeze policy**: Rebuild old schemas with new content, or freeze?
3. **Deprecation**: Mark old schemas as deprecated in manifest?
4. **CLI flag**: Add `--schema` to build command, or auto-detect?

---

## Critical Files

| Purpose | File Path |
|---------|-----------|
| Metadata in DB | `packages/oidm-maintenance/src/oidm_maintenance/*/build.py` |
| Schema selection | `packages/oidm-common/src/oidm_common/distribution/manifest.py` |
| Required schema | `packages/findingmodel/src/findingmodel/config.py` |
| Required schema | `packages/anatomic-locations/src/anatomic_locations/config.py` |
| Manifest publishing | `packages/oidm-maintenance/src/oidm_maintenance/s3.py` |
| Documentation | `docs/manifest_schema.md` |
