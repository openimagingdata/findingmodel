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
3. Database keys should match config field names (e.g., "finding_models" matches `db_path`)
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
