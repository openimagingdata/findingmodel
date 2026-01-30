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
# Build finding model index
oidm-maintain build-index

# Build anatomic location database
oidm-maintain build-anatomic

# Publish to S3
oidm-maintain publish --bucket oidm-data
```

## Documentation

See [Database Management Guide](../../docs/database-management.md) for complete instructions.

## Note

End users do not need this package. Databases are automatically downloaded by:

- [findingmodel](../findingmodel/README.md) - Finding model Index
- [anatomic-locations](../anatomic-locations/README.md) - Anatomic location queries
