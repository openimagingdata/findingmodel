# Database Management Guide

This guide is for maintainers who need to build, update, or manage the DuckDB database files used by the `findingmodel` package.

## Overview

The package uses **DuckDB** as the default backend for indexing finding models and anatomic locations. DuckDB provides high-performance vector search (HNSW indexing) and full-text search capabilities in a single-file database.

For most users, databases **auto-download on first use** - no manual setup required. This guide is for maintainers who need to create or update database files.

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

**Default location**: Same as finding models (platform-native)

**Default filename**: `anatomic_locations.duckdb`

## Automatic Database Downloads

Configure automatic downloads by setting environment variables in `.env`:

```bash
# Finding models index
REMOTE_INDEX_DB_URL=https://your-server.com/finding_models.duckdb
REMOTE_INDEX_DB_HASH=sha256:your_hash_here

# Anatomic locations database
REMOTE_ANATOMIC_DB_URL=https://your-server.com/anatomic_locations.duckdb
REMOTE_ANATOMIC_DB_HASH=sha256:your_hash_here
```

Both URL and SHA256 hash must be provided. The package uses [Pooch](https://www.fatiando.org/pooch/) for:
- Automatic downloads on first use
- Hash verification (re-downloads if hash mismatches)
- Caching in platform-native directories

## CLI Commands

### Finding Models Index Management

#### Build a New Index

```bash
# Build from a directory of .fm.json files
python -m findingmodel index build /path/to/defs/

# Build with custom output path
python -m findingmodel index build /path/to/defs/ --index /custom/path/index.duckdb
```

Creates a new DuckDB index by scanning a directory tree for `*.fm.json` files.

#### Update an Existing Index

```bash
# Update default index from directory
python -m findingmodel index update /path/to/defs/

# Update custom index
python -m findingmodel index update /path/to/defs/ --index /custom/path/index.duckdb
```

Synchronizes the index with the directory:
- Adds new finding models
- Updates modified models
- Removes models whose files no longer exist

#### View Index Statistics

```bash
# Stats for default index
python -m findingmodel index stats

# Stats for custom index
python -m findingmodel index stats --index /custom/path/index.duckdb
```

Displays:
- Total finding models
- Number of people and organizations
- Database file size
- Index status (HNSW vector index, FTS text index)

#### Publish to Remote Storage

```bash
# Build from definitions and publish to S3
uv run python -m findingmodel index publish --defs-dir /path/to/defs/

# Publish an existing database file
uv run python -m findingmodel index publish --database /path/to/existing.duckdb

# Skip sanity checks and confirmation prompts
uv run python -m findingmodel index publish --database /path/to/db.duckdb --skip-checks
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
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

**When to use**:
- Use `--defs-dir` for regular updates from finding model definitions
- Use `--database` to republish an existing database file
- Use `--skip-checks` for automated deployments (bypasses interactive prompts)

The published database becomes available immediately via manifest-based auto-download for all users.

### Anatomic Location Database Management

#### Build Anatomic Database

```bash
# Build from default URL (configured in package)
python -m findingmodel anatomic build

# Build from local file
python -m findingmodel anatomic build --source /path/to/anatomic_locations.json

# Build from custom URL
python -m findingmodel anatomic build --source https://example.com/data.json

# Build with custom output path
python -m findingmodel anatomic build --output /custom/path/anatomic.duckdb

# Force overwrite existing database
python -m findingmodel anatomic build --force
```

**What it does**:
1. Downloads/loads anatomic location data
2. Generates OpenAI embeddings for each concept
3. Creates DuckDB database with vector and text indexes
4. Stores in platform-native directory

#### Validate Anatomic Data

```bash
# Validate without building database
python -m findingmodel anatomic validate --source /path/to/data.json
```

Checks that the data file has required fields without creating a database.

#### View Anatomic Database Statistics

```bash
# Stats for default database
python -m findingmodel anatomic stats

# Stats for custom database
python -m findingmodel anatomic stats --db-path /custom/path/anatomic.duckdb
```

Displays:
- Total anatomic location records
- Number of vectors (embeddings)
- Region distribution
- Database file size

## Python API for Database Management

### Adding/Updating Finding Models

```python
import asyncio
from findingmodel import Index, FindingModelFull

async def update_index():
    # Open index in write mode (read_only=False is default)
    async with Index() as index:
        # Ensure schema is set up
        await index.setup()

        # Update from a directory
        added, updated, removed = await index.update_from_directory("path/to/defs")
        print(f"Sync: {added} added, {updated} updated, {removed} removed")

        # Add or update a single file
        model = FindingModelFull.model_validate_json(open("model.fm.json").read())
        await index.add_or_update_entry_from_file("model.fm.json", model)

asyncio.run(update_index())
```

### Working with Contributors

```python
async def add_contributors():
    async with Index() as index:
        from findingmodel.contributor import Person, Organization

        # Add a person
        person = Person(
            github_username="johndoe",
            name="John Doe",
            email="john@example.com"
        )
        await index.add_person(person)

        # Add an organization
        org = Organization(
            code="ACME",
            name="ACME Corporation",
            url="https://acme.com"
        )
        await index.add_organization(org)
```

### Batch Operations

```python
async def batch_updates():
    async with Index() as index:
        # Add/update multiple models efficiently
        models_dict = {
            "file1.fm.json": model1,
            "file2.fm.json": model2,
            "file3.fm.json": model3,
        }
        results = await index.batch_add_or_update(models_dict)

        # Search with tag filtering
        results = await index.search_batch(
            ["pneumonia", "pneumothorax"],
            tags=["chest"]
        )
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

These are loaded from package data files (`src/findingmodel/data/base_*.jsonl`) during database setup.

> **Historical Note**: Prior to v0.5.0, a MongoDB-based Index implementation was available. This was replaced with DuckDB in v0.5.0. Users needing MongoDB should use findingmodel v0.4.x.

## Troubleshooting

### Database Not Found Errors

If you see "Database not found" errors:
1. Check if remote URLs are configured in `.env`
2. Verify the database path exists
3. Run `index build` or `anatomic build` to create the database

### Hash Mismatch on Download

If Pooch reports a hash mismatch:
1. The remote file has changed
2. Update `REMOTE_*_HASH` in `.env` with the new SHA256 hash
3. Pooch will automatically re-download with the new hash

### Regenerating Embeddings

To regenerate embeddings for anatomic locations:
```bash
python -m findingmodel anatomic build --source /path/to/data.json --force
```

The `--force` flag overwrites the existing database.

## File Locations Reference

### Default Paths

Use `python -m findingmodel config` to see current configuration including file paths.

### Custom Paths

Override defaults via environment variables:
```bash
# Override filenames (still uses platform-native directory)
DUCKDB_INDEX_PATH=my_custom_index.duckdb
DUCKDB_ANATOMIC_PATH=my_custom_anatomic.duckdb
```

Or specify custom paths in CLI:
```bash
python -m findingmodel index stats --index /completely/custom/path.duckdb
```
