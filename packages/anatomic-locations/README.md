# anatomic-locations

Python library for anatomic location ontology navigation: hierarchy traversal, laterality variants, and semantic search.

## Installation

```bash
pip install anatomic-locations
```

## Features

- **Hybrid Search**: Combined full-text and semantic vector search
- **Hierarchy Navigation**: Traverse parent/child relationships
- **Laterality Variants**: Generate left, right, bilateral variants
- **Auto-Download**: Database downloads automatically on first use

## Configuration

The anatomic location database is automatically downloaded on first use. To use a custom path:

```bash
# Optional: Custom database path
ANATOMIC_DB_PATH=/mnt/data/anatomic_locations.duckdb
```

## CLI (`anatomic-locations`)

```bash
# Search for anatomic locations
anatomic-locations search "posterior cruciate ligament"

# Show hierarchy for a location
anatomic-locations hierarchy RID2905

# List children of a location
anatomic-locations children RID56
```

## Python API

```python
import asyncio
from anatomic_locations import AnatomicLocationIndex

async def main():
    async with AnatomicLocationIndex() as index:
        # Hybrid search (FTS + semantic)
        results = await index.search("knee joint", limit=10)
        for result in results:
            print(f"- {result.name} ({result.id})")

        # Get location by ID
        location = await index.get("RID2905")
        if location:
            print(f"Name: {location.name}")
            print(f"Parents: {location.parents}")

        # Get children
        children = await index.get_children("RID56")
        for child in children:
            print(f"  - {child.name}")

        # Generate laterality variants
        variants = await index.get_laterality_variants("RID2905")
        for variant in variants:
            print(f"  {variant.laterality}: {variant.name}")

asyncio.run(main())
```

## Related Packages

- **[findingmodel-ai](../findingmodel-ai/README.md)**: AI-assisted anatomic location discovery
- **[findingmodel](../findingmodel/README.md)**: Core finding model library

## Documentation

- [Anatomic Locations Guide](../../docs/anatomic-locations.md)
