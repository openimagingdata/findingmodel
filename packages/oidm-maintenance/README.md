# oidm-maintenance

Maintenance tools for OIDM packages. **This package is for OIDM maintainers only.**

## Purpose

Provides tools for building and publishing OIDM databases:

- Build finding model DuckDB index with embeddings
- Build anatomic location DuckDB database
- Publish databases to S3 for auto-download

## Installation

```bash
pip install oidm-maintenance
```

## Requirements

- AWS credentials configured for S3 access
- OpenAI API key for embedding generation

## CLI

```bash
# Build findingmodel database
oidm-maintain findingmodel build --source /path/to/fm-json/ --output /tmp/finding_models.duckdb

# Build metadata-aware findingmodel database with provenance
oidm-maintain findingmodel build --source /path/to/fm-json/ --output /tmp/finding_models_metadata.duckdb \
  --schema-name finding_models_metadata --schema-version 2.0.0 --source-commit abc123

# Build anatomic location database
oidm-maintain anatomic build --source /path/to/anatomic.json --output /tmp/anatomic_locations.duckdb

# Publish databases
oidm-maintain findingmodel publish /tmp/finding_models.duckdb --dry-run
oidm-maintain findingmodel publish /tmp/finding_models_metadata.duckdb --dry-run \
  --manifest-key finding_models_metadata --s3-prefix findingmodel-metadata --artifact-name findingmodels_metadata.duckdb
oidm-maintain anatomic publish /tmp/anatomic_locations.duckdb --dry-run

# Migrate embedding cache into current oidm-common namespace
oidm-maintain embeddings migrate

# Show current cache stats (total/new metadata keys and per-model counts)
oidm-maintain embeddings stats

# Import arbitrary DuckDB embedding cache file (upsert into current cache)
oidm-maintain embeddings import-duckdb /path/to/embeddings.duckdb

# Import entries from another diskcache directory
oidm-maintain embeddings import-cache /path/to/embeddings.cache
```

Embedding import commands report `written/new/updated/skipped/total` counts.
They exit non-zero if the current cache directory cannot be opened for writes.
`import-cache` expects a diskcache directory; copy the entire directory when transferring between machines.

Findingmodel database builds record artifact provenance in a `database_metadata` table, including
schema name/version, source commit, package versions, build timestamp, and embedding profile.
Publishing defaults to the current `finding_models` manifest key; use `--manifest-key` and related
artifact options when intentionally publishing a metadata-aware artifact.

## Documentation

See [Database Management Guide](../../docs/database-management.md) for complete instructions.

## Note

End users do not need this package. Databases are automatically downloaded by:

- [findingmodel](../findingmodel/README.md) - Finding model Index
- [anatomic-locations](../anatomic-locations/README.md) - Anatomic location queries
