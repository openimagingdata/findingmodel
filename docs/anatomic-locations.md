# Anatomic Locations

Query and navigate anatomic locations with hierarchy traversal, laterality variants, and semantic search.

## Quick Start

The database auto-downloads on first use. No setup required.

```python
from findingmodel import AnatomicLocationIndex

with AnatomicLocationIndex() as index:
    location = index.get("RID2772")  # Kidney
    print(location.description)       # "kidney"
    print(location.region)            # AnatomicRegion.ABDOMEN
```

## CLI Commands

### Query by ID

```bash
# Show ancestors (hierarchy to root)
python -m findingmodel anatomic query ancestors RID2772

# Show descendants
python -m findingmodel anatomic query descendants RID39569

# Show laterality variants (left/right/generic)
python -m findingmodel anatomic query laterality RID2772

# Find by external code (SNOMED, FMA, RadLex)
python -m findingmodel anatomic query code snomed 64033007
```

### Database Management

```bash
# View statistics
python -m findingmodel anatomic stats

# Rebuild database (requires OPENAI_API_KEY for embeddings)
python -m findingmodel anatomic build --force

# Validate source data without building
python -m findingmodel anatomic validate --source /path/to/data.json
```

## Python API

### Basic Queries

```python
from findingmodel import AnatomicLocationIndex

with AnatomicLocationIndex() as index:
    # Get by ID
    location = index.get("RID2772")

    # Semantic search
    results = index.search("left kidney", limit=5)
    for loc, score in results:
        print(f"{loc.description}: {score:.3f}")

    # Find by external code
    locations = index.find_by_code("snomed", "64033007")
```

### Hierarchy Navigation

```python
with AnatomicLocationIndex() as index:
    kidney = index.get("RID2772")

    # Navigate up (to root)
    for ancestor in kidney.get_containment_ancestors():
        print(f"  {ancestor.description}")

    # Navigate down
    for descendant in kidney.get_containment_descendants():
        print(f"  {'  ' * descendant.containment_depth}{descendant.description}")
```

### Laterality Variants

```python
with AnatomicLocationIndex() as index:
    kidney = index.get("RID2772")  # Generic "kidney"

    variants = kidney.get_laterality_variants()
    # {Laterality.LEFT: <left kidney>, Laterality.RIGHT: <right kidney>}

    left_kidney = variants.get(Laterality.LEFT)
```

### FastAPI Integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from findingmodel import AnatomicLocationIndex

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

Override the default database path:

```bash
# In .env
DUCKDB_ANATOMIC_PATH=/path/to/custom.duckdb
```

Or specify directly:

```python
index = AnatomicLocationIndex(db_path="/path/to/custom.duckdb")
```
