# Database Publishing Automation

**Status**: Planning
**Priority**: High
**Target Version**: v0.7.0

## Overview

Automate the manual process of building, validating, and publishing FindingModel database files to Tigris S3 storage with automatic manifest.json updates.

## Current Manual Process (Problems)

1. Run `index build` CLI command on definitions directory
2. Manually inspect DuckDB file for sanity
3. Compute SHA256 hash with `shasum -a 256`
4. Download existing manifest.json from storage
5. Manually edit manifest.json with new version/URL/hash
6. Manually upload database file to Tigris S3
7. Rename old manifest.json as backup
8. Manually upload new manifest.json to Tigris S3

**Issues**: Error-prone, time-consuming, requires multiple manual steps with copy/paste operations.

## Solution: `index publish` CLI Command

New CLI command that automates the entire workflow with validation checkpoints.

### Command Modes

**Mode 1: Build and Publish** (builds from source definitions)
```bash
python -m findingmodel index publish \
  --defs-dir path/to/definitions \
  [--skip-checks]
```

**Mode 2: Publish Existing** (publishes pre-built database)
```bash
python -m findingmodel index publish \
  --database path/to/existing.duckdb \
  [--skip-checks]
```

### Workflow Steps

1. **Build Database** (Mode 1 only)
   - Build to temporary file using existing `DuckDBIndex.update_from_directory()`
   - Use same logic as existing `index build` command

2. **Sanity Check** (unless `--skip-checks`)
   - Query database statistics (record count, file size)
   - Retrieve first 3 OIFM IDs
   - Fetch and display first complete model (first 20 lines of JSON)
   - **User confirmation prompt**: "Does this look correct? [yes/no/cancel]"
     - `yes`: continue to upload
     - `no`: stop, don't upload
     - `cancel`: abort immediately

3. **Compute Hash**
   - Calculate SHA256 hash of database file
   - Format as `sha256:hexdigest`

4. **Upload Database**
   - Generate filename: `findingmodels_YYYYMMDD.duckdb` (compact date format)
   - Upload to S3 bucket root
   - Use Tigris virtual hosted addressing style

5. **Backup Current Manifest**
   - Download current `manifest.json` from S3
   - Upload backup to `manifests/archive/manifest_YYYYMMDD_HHMMSS.json`
   - Backup stored in S3, not locally

6. **Update Manifest**
   - Parse current manifest.json
   - Update `databases.finding_models` entry:
     - `version`: ISO date format `YYYY-MM-DD`
     - `url`: Full HTTPS URL to new database
     - `hash`: Computed SHA256 hash with `sha256:` prefix
     - `size_bytes`: File size in bytes
     - `record_count`: From database query
     - `description`: Keep existing description
   - Update top-level `generated_at` to current ISO 8601 timestamp

7. **Upload New Manifest**
   - Upload updated manifest.json to S3 bucket root
   - Replaces existing manifest.json

8. **Display Summary**
   - Show uploaded database URL
   - Show updated manifest URL
   - Show backup manifest location
   - Show record count and size

## Configuration

Extends `FindingModelConfig` with publishing-specific settings:

```python
class PublishConfig(FindingModelConfig):
    """Settings for database publishing."""

    # AWS/Tigris credentials (standard AWS env vars)
    aws_access_key_id: SecretStr | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)

    # Tigris S3 settings
    s3_endpoint_url: str = Field(default="https://t3.storage.dev")
    s3_bucket: str = Field(default="findingmodelsdata")
    manifest_backup_prefix: str = Field(default="manifests/archive/")
```

**Environment Variables**:
- `AWS_ACCESS_KEY_ID`: Tigris access key
- `AWS_SECRET_ACCESS_KEY`: Tigris secret key
- `S3_ENDPOINT_URL`: Override endpoint (default: `https://t3.storage.dev`)
- `S3_BUCKET`: Override bucket (default: `findingmodelsdata`)

## Technical Details

### Tigris S3 Configuration

```python
import boto3
from botocore.client import Config

s3_client = boto3.client(
    's3',
    endpoint_url='https://t3.storage.dev',
    aws_access_key_id=config.aws_access_key_id,
    aws_secret_access_key=config.aws_secret_access_key,
    config=Config(s3={'addressing_style': 'virtual'})  # Required for Tigris
)
```

**Key Requirements**:
- Virtual hosted addressing style is **required** for Tigris (buckets created after Feb 2025)
- No region configuration needed (Tigris handles globally)
- Standard boto3 S3 API compatibility

### Manifest Schema (v1.0)

```json
{
  "manifest_version": "1.0",
  "generated_at": "2025-11-17T14:30:00Z",
  "databases": {
    "finding_models": {
      "version": "2025-11-17",
      "url": "https://findingmodelsdata.t3.storage.dev/findingmodels_20251117.duckdb",
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

**Only update `finding_models` entry** - leave `anatomic_locations` unchanged.

### Naming Conventions

- **Database filename**: `findingmodels_YYYYMMDD.duckdb` (compact date, no hyphens)
- **Manifest version**: `YYYY-MM-DD` (ISO date with hyphens)
- **Manifest backup**: `manifests/archive/manifest_YYYYMMDD_HHMMSS.json` (includes timestamp)

### Error Handling

- Validate S3 credentials before starting
- Check database file exists and is readable
- Verify database can be queried before upload
- Atomic manifest updates (download → backup → update → upload)
- Roll back on upload failures (keep old manifest)
- Clear error messages for common issues:
  - Missing AWS credentials
  - Network/S3 connection failures
  - Invalid database file
  - Permission issues

## Implementation Files

### New Files
- `src/findingmodel/publishing.py` - Publishing logic and PublishConfig
- `test/test_publishing.py` - Unit tests for publishing workflow

### Modified Files
- `src/findingmodel/cli.py` - Add `index publish` command

## Future Extensions

Once finding_models publishing is stable, extend to support:

1. **Anatomic locations database** - Same workflow, different manifest key
2. **Multi-database publish** - Publish both databases in one command
3. **Rollback command** - `index rollback` to restore from backup manifest
4. **Dry-run mode** - `--dry-run` flag to show what would happen without uploading

## Testing Strategy

### Unit Tests
- Mock boto3 S3 operations
- Test manifest JSON generation
- Test hash computation
- Test filename/version formatting

### Integration Tests (Manual)
- Test with real Tigris S3 (test bucket)
- Verify uploaded files accessible via HTTPS
- Confirm manifest updates work with existing `ensure_index_db()`
- Test rollback from backup manifests

### Pre-Deployment Checklist
- [ ] Test build mode with sample definitions
- [ ] Test publish mode with existing database
- [ ] Verify sanity check displays correct info
- [ ] Confirm user prompts work as expected
- [ ] Test skip-checks flag
- [ ] Verify manifest backup to S3
- [ ] Confirm manifest updates correctly
- [ ] Test with missing credentials (proper error)
- [ ] Test with network failure (proper error)

## Dependencies

### New Dependencies
- `boto3` - AWS SDK for S3 operations
- `botocore` - For Config and error handling

### Existing Dependencies (reused)
- `httpx` - For downloading current manifest
- `pooch` - For hash computation (already used)
- `click` - For CLI interface
- `rich` - For console output

## Success Criteria

1. One-command database publishing (no manual steps)
2. Interactive validation with clear user prompts
3. Automatic manifest updates (no manual JSON editing)
4. Manifest backups stored in S3
5. Clear error messages for all failure modes
6. Works for both build-and-publish and publish-only workflows
7. Easy to extend for anatomic_locations in future
