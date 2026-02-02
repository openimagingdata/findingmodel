# anatomic-locations

Python library for anatomic location ontology navigation: hierarchy traversal, laterality variants, and semantic search.

## Installation

```bash
pip install anatomic-locations
```

## Features

- **Hybrid Search**: Combined full-text and semantic vector search
- **Hierarchy Navigation**: Traverse parent/child relationships with tree visualization
- **Laterality Variants**: Generate left, right, bilateral variants
- **Auto-Download**: Database downloads automatically on first use
- **Flexible Lookup**: Use location IDs (RID*) or names/synonyms

## Quick Start

The database auto-downloads on first use. No setup required.

```python
from anatomic_locations import AnatomicLocationIndex

with AnatomicLocationIndex() as index:
    location = index.get("RID2772")  # Kidney
    print(location.description)       # "kidney"
    print(location.region.value)      # "Abdomen"
```

## Configuration

Override the default database path via environment variable or constructor:

```bash
# Environment variable
export ANATOMIC_DB_PATH=/path/to/custom.duckdb
```

```python
# Or specify directly
index = AnatomicLocationIndex(db_path="/path/to/custom.duckdb")
```

## CLI

The `anatomic-locations` CLI provides search, hierarchy navigation, and lookup commands. Use location IDs (e.g., `RID2772`) or names (e.g., "stomach"):

```bash
# Semantic search
anatomic-locations search "posterior cruciate ligament"

# Show full hierarchy tree
anatomic-locations hierarchy stomach
```

See the [full CLI reference](../../docs/anatomic-locations.md#cli-commands) for all commands.

## Related Packages

- **[findingmodel-ai](../findingmodel-ai/README.md)**: AI-assisted anatomic location discovery
- **[findingmodel](../findingmodel/README.md)**: Core finding model library

## Documentation

- [Anatomic Locations Guide](../../docs/anatomic-locations.md) - Full CLI reference, Python API, and configuration
