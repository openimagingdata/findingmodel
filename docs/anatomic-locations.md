# Anatomic Locations

Query and navigate anatomic locations with hierarchy traversal, laterality variants, and semantic search.

## Quick Start

The database auto-downloads on first use. No setup required.

```python
from anatomic_locations import AnatomicLocationIndex

with AnatomicLocationIndex() as index:
    location = index.get("RID2772")  # Kidney
    print(location.description)       # "kidney"
    print(location.region.value)      # "Abdomen"
```

## CLI Commands

The `anatomic-locations` CLI accepts either a location ID (e.g., `RID56`) or a location name/synonym (e.g., "stomach").

**Note:** The database auto-downloads on first CLI use.

### search

Semantic search using hybrid full-text + vector search:

```bash
$ anatomic-locations search "posterior cruciate ligament"

               Search Results for "posterior cruciate ligament"
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ ID           ┃ Name                        ┃ Region          ┃ Laterality   ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ RID2784      │ posterior cruciate ligament │ Lower Extremity │ generic      │
└──────────────┴─────────────────────────────┴─────────────────┴──────────────┘

Total results: 1
```

### hierarchy

Show full hierarchy tree with ancestors above and descendants below the specified location:

```bash
$ anatomic-locations hierarchy stomach

Hierarchy Tree:

whole body - RID39569
    └── abdomen - RID56
        └── peritoneal cavity - RID397
            └── ▶ stomach - RID114 ◀
                ├── gastric fundus - RID116
                └── pylorus - RID122
```

### children

List direct children in a table:

```bash
$ anatomic-locations children stomach

Location: stomach (RID114)
Region: Abdomen, Type: structure

                             Direct Children
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ ID                   ┃ Name           ┃ Region          ┃ Laterality   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ RID116               │ gastric fundus │ Abdomen         │ nonlateral   │
│ RID122               │ pylorus        │ Abdomen         │ nonlateral   │
└──────────────────────┴────────────────┴─────────────────┴──────────────┘

Total children: 2
```

### ancestors

Show containment ancestors in a table:

```bash
$ anatomic-locations ancestors "stomach"

Location: stomach (RID114)
Region: Abdomen, Type: structure

            Containment Ancestors (nearest to root)
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃  Level ┃ ID                   ┃ Name            ┃ Region           ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│      1 │ RID397               │ peritoneal      │ Abdomen          │
│        │                      │ cavity          │                  │
│      2 │ RID56                │ abdomen         │ Body             │
│      3 │ RID39569             │ whole body      │ Body             │
└────────┴──────────────────────┴─────────────────┴──────────────────┘
```

### descendants

Show containment descendants in a table:

```bash
$ anatomic-locations descendants "abdominal cavity"
```

### laterality

Show laterality variants for a location:

```bash
$ anatomic-locations laterality "axillary lymph node"

Location: axillary lymph node (RID1517)
Laterality: generic, Region: Thorax

                        Laterality Variants
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Laterality   ┃ ID                   ┃ Name                      ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ left         │ RID1517_RID5824      │ left axillary lymph node  │
│ right        │ RID1517_RID5825      │ right axillary lymph node │
└──────────────┴──────────────────────┴───────────────────────────┘
```

### code

Find location by external code (SNOMED, FMA, RadLex):

```bash
$ anatomic-locations code SNOMED 64033007
```

### stats

Show database statistics:

```bash
$ anatomic-locations stats

Anatomic Location Database Statistics

Database: /home/user/.local/share/anatomic-locations/anatomic_locations.duckdb

          Database Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric                 ┃    Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total Records          │    4,847 │
│ Records with Vectors   │    4,847 │
│ Records with Hierarchy │    4,847 │
│ Records with Codes     │    4,847 │
│ Unique Regions         │       10 │
│ Total Synonyms         │    2,934 │
│ Total Codes            │    8,456 │
│ File Size              │  12.34 MB│
└────────────────────────┴──────────┘
```

## Python API

### Basic Queries

```python
import asyncio
from anatomic_locations import AnatomicLocationIndex

# Sync operations
with AnatomicLocationIndex() as index:
    # Get by ID (sync)
    location = index.get("RID2772")
    print(f"Name: {location.description}")
    print(f"Region: {location.region.value if location.region else 'N/A'}")

    # Find by external code (sync)
    locations = index.find_by_code("SNOMED", "64033007")
    for loc in locations:
        print(f"Found: {loc.description}")

    # Get direct children (sync)
    children = index.get_children_of("RID56")
    for child in children:
        print(f"  Child: {child.description}")

# Async operations
async def search_locations():
    async with AnatomicLocationIndex() as index:
        # Hybrid search (FTS + semantic) - async
        results = await index.search("knee joint", limit=10)
        for result in results:
            print(f"- {result.description} ({result.id})")

asyncio.run(search_locations())
```

### Hierarchy Navigation

```python
with AnatomicLocationIndex() as index:
    kidney = index.get("RID2772")

    # Navigate up (ancestors return from immediate parent to root)
    for ancestor in kidney.get_containment_ancestors():
        print(f"  Ancestor: {ancestor.description}")

    # Navigate down (all descendants)
    for descendant in kidney.get_containment_descendants():
        print(f"  Descendant: {descendant.description}")
```

### Laterality Variants

```python
from anatomic_locations import Laterality

with AnatomicLocationIndex() as index:
    # Get generic location (e.g., "axillary lymph node")
    lymph_node = index.get("RID1517")

    # Get variants (returns dict: Laterality -> AnatomicLocation)
    variants = lymph_node.get_laterality_variants()

    # Access specific variant
    left_variant = variants.get(Laterality.LEFT)
    right_variant = variants.get(Laterality.RIGHT)

    for laterality, variant in variants.items():
        print(f"{laterality.value}: {variant.description} ({variant.id})")
```

### FastAPI Integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from anatomic_locations import AnatomicLocationIndex

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.anatomic_index = AnatomicLocationIndex().open()
    yield
    app.state.anatomic_index.close()

app = FastAPI(lifespan=lifespan)

@app.get("/location/{location_id}")
def get_location(location_id: str):
    index = app.state.anatomic_index
    return index.get(location_id)
```

## Key Concepts

### Laterality

Locations have a `laterality` field:
- `generic` - Has left/right variants (e.g., "kidney")
- `left` / `right` - Sided variant (e.g., "left kidney")
- `nonlateral` - No laterality (e.g., "heart", "spine")

### Hierarchies

Two hierarchy types, both with materialized paths for fast queries:
- **Containment** - Spatial "contained by" (kidney → abdomen → body)
- **Part-of** - Structural "part of" (nephron → kidney)

### External Codes

Locations are linked to external ontologies:
- SNOMED CT
- FMA (Foundational Model of Anatomy)
- RadLex

```python
for code in location.codes:
    print(f"{code.system}: {code.code} ({code.display})")
```

## Configuration

### Environment Variable

Set the database path via environment variable:

```bash
export ANATOMIC_DB_PATH=/path/to/custom.duckdb
```

### Constructor Parameter

Override path directly in code:

```python
# Sync context manager
with AnatomicLocationIndex(db_path="/path/to/custom.duckdb") as index:
    location = index.get("RID2772")

# Or async
async with AnatomicLocationIndex(db_path="/path/to/custom.duckdb") as index:
    results = await index.search("kidney", limit=5)
```

### Auto-Download Behavior

On first use, if no database exists at the configured path, the package automatically downloads the latest version from the remote manifest. This requires internet access on first run but works offline thereafter.
