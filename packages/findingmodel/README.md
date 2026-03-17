# findingmodel

Core Python library for Open Imaging Finding Models - structured data models for describing medical imaging findings in radiology reports.

## Installation

```bash
pip install findingmodel
```

## Features

- **Finding Model Management**: Create and manage structured medical finding models with attributes
- **Finding Model Index**: Fast lookup and search across finding model definitions with DuckDB
- **MCP Server**: Model Context Protocol server for AI agent integration
- **Medical Ontology Support**: Index codes from RadLex, SNOMED-CT, and other vocabularies

## Configuration

Create a `.env` file in your project root:

```bash
# Optional: OpenAI-backed semantic search
OPENAI_API_KEY=your_key_here

# Optional: Custom database path
FINDINGMODEL_DB_PATH=/mnt/data/finding_models.duckdb
```

The finding model database is automatically downloaded on first use. If no OpenAI key is configured, `findingmodel` can use local-profile database artifacts instead of OpenAI-backed search.

## CLI (`findingmodel`)

```bash
# View configuration
findingmodel config


# Show index statistics
findingmodel stats

# Search for finding models
findingmodel search "lung nodule"

# Validate one model JSON file
findingmodel validate model.fm.json

# Validate and reformat all models under a directory
findingmodel validate ./defs --reformat
# - validates all matching files and exits non-zero if any fail
# - --reformat updates valid files in place

# Convert finding model to Markdown
findingmodel fm-to-markdown model.json
```

Markdown export is a presentation view over canonical finding model JSON. It is intended for reading, review, and lightweight sharing, not as the source-of-truth storage format.

## Models

### FindingModelBase

Basic finding model structure with name, description, attributes, and optional structured metadata.

```python
from findingmodel import FindingModelBase, BodyRegion, EntityType, ExpectedDuration, ExpectedTimeCourse

model = FindingModelBase(
    name="pneumothorax",
    description="Presence of air in the pleural space",
    body_regions=[BodyRegion.CHEST],
    entity_type=EntityType.FINDING,
    expected_time_course=ExpectedTimeCourse(duration=ExpectedDuration.DAYS),
    attributes=[...],
)

# Export to Markdown for human-readable review
print(model.as_markdown())
```

### FindingModelFull

Extended model with OIFM IDs, index codes, contributor information, and structured metadata.

```python
from findingmodel import FindingModelFull

# Load from JSON — legacy values auto-normalize (e.g., "Chest" → "chest", "CR" → "XR")
model = FindingModelFull.model_validate_json(json_content)

print(f"Model ID: {model.oifm_id}")
print(f"Body regions: {model.body_regions}")
print(f"Entity type: {model.entity_type}")
for attr in model.attributes:
    print(f"  {attr.name}: {attr.oifma_id}")
```

### Structured Metadata Types

Optional canonical metadata fields available on both `FindingModelBase` and `FindingModelFull`:

| Field | Type | Description |
|-------|------|-------------|
| `body_regions` | `list[BodyRegion]` | Broad anatomic regions (head, chest, abdomen, etc.) |
| `subspecialties` | `list[Subspecialty]` | Radiology subspecialties (NR, CH, MK, etc.) |
| `etiologies` | `list[EtiologyCode]` | Etiology categories (inflammatory, neoplastic:malignant, etc.) |
| `entity_type` | `EntityType` | What the model represents (finding, diagnosis, measurement, etc.) |
| `applicable_modalities` | `list[Modality]` | Imaging modalities (CT, MR, XR, etc.) |
| `expected_time_course` | `ExpectedTimeCourse` | Duration and behavioral modifiers |
| `age_profile` | `AgeProfile` | Age applicability and prevalence |
| `sex_specificity` | `SexSpecificity` | Sex-specific or sex-neutral |

All fields are optional and default to `None`. Legacy values normalize automatically on load.

### FindingInfo

Metadata about a finding including description, synonyms, and optional citations.

```python
from findingmodel import FindingInfo

info = FindingInfo(
    name="pneumothorax", synonyms=["PTX", "collapsed lung"], description="Presence of air in the pleural space"
)
```

## Index API

The `Index` class provides async access to the finding model database.

```python
import asyncio
from findingmodel import Index


async def main():
    async with Index() as index:
        # Count indexed models
        count = await index.count()
        print(f"Total models: {count}")

        # Lookup by name or ID
        model = await index.get("pneumothorax")
        if model:
            print(f"Found: {model.name} ({model.oifm_id})")

        # Search for models
        results = await index.search("lung nodule", limit=5)
        for result in results:
            print(f"- {result.name}: {result.description}")

        # List all with pagination
        models, total = await index.all(limit=20, offset=0)


asyncio.run(main())
```

## MCP Server

The package includes an MCP server for AI agent integration.

### Tools Provided

- **search_finding_models**: Hybrid search (FTS + semantic) for finding models
- **get_finding_model**: Retrieve specific models by ID, name, or synonym
- **count_finding_models**: Get index statistics

### Running the Server

```bash
# Run directly
python -m findingmodel.mcp_server

# Or use the CLI entry point
findingmodel-mcp
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "finding-model-search": {
      "command": "python",
      "args": ["-m", "findingmodel.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "your-key-here"
      }
    }
  }
}
```

See [MCP Server Guide](../../docs/mcp_server.md) for complete documentation.

## Related Packages

- **[findingmodel-ai](../findingmodel-ai/README.md)**: AI-powered tools for model authoring
- **[anatomic-locations](../anatomic-locations/README.md)**: Anatomic location ontology queries

## Documentation

- [Configuration Guide](../../docs/configuration.md)
- [Database Management](../../docs/database-management.md)
- [MCP Server Guide](../../docs/mcp_server.md)
