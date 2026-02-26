# Database Management Guide

This guide is for maintainers who need to build, update, or manage the DuckDB database files used by the findingmodel and anatomic-locations packages.

## Overview

The packages use **DuckDB** as the default backend for indexing finding models and anatomic locations. DuckDB provides high-performance vector search (HNSW indexing) and full-text search capabilities in a single-file database.

For most users, databases **auto-download on first use** - no manual setup required. This guide is for maintainers who need to create or update database files.

> **Package note**: Build and publish operations live in the `oidm-maintenance` package (`oidm-maintain` CLI). The user-facing `findingmodel` and `anatomic-locations` packages only provide read-only database access.

## Database Architecture

### Finding Models Index

The finding models index stores:
- Finding model metadata (name, description, synonyms, tags)
- Contributor information (people and organizations)
- Full-text search indexes
- Base contributors (4 people, 7 organizations) automatically loaded

**Default location**: Platform-native data directory via `platformdirs`
- macOS: `~/Library/Application Support/findingmodel/`
- Linux: `~/.local/share/findingmodel/`
- Windows: `%LOCALAPPDATA%\findingmodel\`

**Default filename**: `finding_models.duckdb`

### Anatomic Locations Database

The anatomic locations database stores:
- Anatomic location concepts from multiple ontologies
- Vector embeddings for semantic search
- Text indexes for keyword search
- Data from anatomic_locations, RadLex, and SNOMED CT

**Default location**: Platform-native data directory via `platformdirs`

**Default filename**: `anatomic_locations.duckdb`

## Automatic Database Downloads

Configure automatic downloads by setting environment variables in `.env`:

```bash
# Finding models index
FINDINGMODEL_REMOTE_DB_URL=https://your-server.com/finding_models.duckdb
FINDINGMODEL_REMOTE_DB_HASH=sha256:your_hash_here

# Anatomic locations database
ANATOMIC_REMOTE_DB_URL=https://your-server.com/anatomic_locations.duckdb
ANATOMIC_REMOTE_DB_HASH=sha256:your_hash_here
```

Both URL and SHA256 hash must be provided. The package uses [Pooch](https://www.fatiando.org/pooch/) for:
- Automatic downloads on first use
- Hash verification (re-downloads if hash mismatches)
- Caching in platform-native directories

## CLI Commands

All build and publish commands use the `oidm-maintain` CLI from the `oidm-maintenance` package.

### Finding Models Index Management

#### Build a New Index

```bash
# Build from a directory of .fm.json files
oidm-maintain findingmodel build /path/to/defs/

# Build with custom output path
oidm-maintain findingmodel build /path/to/defs/ --index /custom/path/index.duckdb
```

Creates a new DuckDB index by scanning a directory tree for `*.fm.json` files.

#### Publish to Remote Storage

```bash
# Build from definitions and publish to S3
oidm-maintain findingmodel publish --defs-dir /path/to/defs/

# Publish an existing database file
oidm-maintain findingmodel publish --database /path/to/existing.duckdb
```

Automates database publishing workflow:
1. Builds database from definitions (if using `--defs-dir`) or uses existing database
2. Runs sanity checks: record count, sample OIFM IDs, model JSON validation
3. Computes SHA256 hash and file size
4. Uploads database to S3/Tigris storage with date-based filename
5. Updates and publishes `manifest.json` with new version info
6. Creates automatic manifest backup in `manifests/archive/`

**Requirements**: AWS credentials in `.env` file (Tigris S3-compatible storage):
```bash
OIDM_MAINTAIN_AWS_ACCESS_KEY_ID=your_access_key
OIDM_MAINTAIN_AWS_SECRET_ACCESS_KEY=your_secret_key
```

#### View Index Statistics (user CLI)

```bash
findingmodel stats
```

### Anatomic Location Database Management

#### Build Anatomic Database

```bash
# Build from default URL (configured in package)
oidm-maintain anatomic build

# Build from local file
oidm-maintain anatomic build --source /path/to/anatomic_locations.json

# Force overwrite existing database
oidm-maintain anatomic build --force
```

**What it does**:
1. Downloads/loads anatomic location data
2. Generates OpenAI embeddings for each concept
3. Creates DuckDB database with vector and text indexes
4. Stores in platform-native directory

#### Publish Anatomic Database

```bash
oidm-maintain anatomic publish
```

#### View Anatomic Database Statistics (user CLI)

```bash
anatomic-locations stats
```

## Base Contributors

Fresh databases automatically include base contributors:

**Organizations** (7):
- Microsoft (MSFT)
- MassGeneral Brigham (MGB)
- Radiology Gamuts Ontology (GMTS)
- Radiological Society of North America (RSNA)
- American College of Radiology (ACR)
- ACR/RSNA Common Data Elements Project (CDE)
- Open Imaging Data Model (OIDM)

**People** (4):
- talkasab (Tarik Alkasab, MD, PhD - MGB)
- HeatherChase (Heather Chase - MSFT)
- hoodcm (C. Michael Hood, MD - MGB)
- radngandhi (Namita Gandhi, MD - RSNA)

These are loaded from package data files during database setup.

> **Historical Note**: Prior to v0.5.0, a MongoDB-based Index implementation was available. This was replaced with DuckDB in v0.5.0. Users needing MongoDB should use findingmodel v0.4.x.

## Troubleshooting

### Database Not Found Errors

If you see "Database not found" errors:
1. Check if remote URLs are configured in `.env`
2. Verify the database path exists
3. Run `oidm-maintain findingmodel build` or `oidm-maintain anatomic build` to create the database

### Hash Mismatch on Download

If Pooch reports a hash mismatch:
1. The remote file has changed
2. Update `REMOTE_*_HASH` in `.env` with the new SHA256 hash
3. Pooch will automatically re-download with the new hash

### Regenerating Embeddings

To regenerate embeddings for anatomic locations:
```bash
oidm-maintain anatomic build --source /path/to/data.json --force
```

The `--force` flag overwrites the existing database.

## File Locations Reference

### Custom Paths

Override defaults via environment variables:
```bash
# Override database paths (still uses platform-native directory)
FINDINGMODEL_DB_PATH=my_custom_index.duckdb
ANATOMIC_DB_PATH=my_custom_anatomic.duckdb
```
