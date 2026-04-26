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

# Optional: Select a manifest database key
FINDINGMODEL_DB_MANIFEST_KEY=finding_models
```

The finding model database is automatically downloaded on first use. An `OPENAI_API_KEY` is required for semantic search. Set `FINDINGMODEL_DB_MANIFEST_KEY=finding_models_metadata` only when intentionally opting into a metadata-aware artifact.

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

Optional structured metadata fields available on both `FindingModelBase` and `FindingModelFull`:

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

        # Browse by metadata filters (no text search)
        chest_findings, total = await index.browse(
            body_regions=[BodyRegion.CHEST],
            entity_type=EntityType.FINDING,
            limit=10,
        )

        # Search with metadata filters (text search + metadata pre-filtering)
        results = await index.search(
            "effusion",
            body_regions=[BodyRegion.CHEST],
            applicable_modalities=[Modality.CT, Modality.XR],
            limit=5,
        )

        # Find related models by deterministic metadata-overlap scoring
        related = await index.related_models("OIFM_RADLEX_000001", limit=5)
        for entry, score in related:
            print(f"  Related: {entry.name} (score={score:.1f})")


asyncio.run(main())
```

### browse()

Pure SQL metadata filtering with pagination. Filter semantics: OR-within-field, AND-across-fields, ALL-of for tags.

```python
entries, total = await index.browse(
    body_regions=[BodyRegion.CHEST, BodyRegion.ABDOMEN],  # OR: chest OR abdomen
    entity_type=EntityType.FINDING,  # AND: must be a finding
    tags=["established"],  # AND + ALL-of: must have this tag
    limit=20,
    offset=0,
)
```

### Metadata-Filtered Search

`search()` and `search_batch()` accept the same metadata filter parameters as `browse()`. Filters are pushed into SQL WHERE clauses before ranking.

Supported metadata filters:
- `body_regions`
- `subspecialties`
- `etiologies`
- `entity_type`
- `applicable_modalities`
- `age_applicability`
- `age_more_common_in`
- `sex_specificity`
- `time_course_durations`
- `time_course_modifiers`
- `tags`

### related_models()

Deterministic metadata-overlap scoring — no LLM calls. Finds models sharing structured metadata with a source model, scored by configurable weights.

```python
from findingmodel import RelatedModelWeights

# Use default weights
related = await index.related_models("OIFM_RADLEX_000001", limit=10, min_score=3.0)

# Custom weights (e.g., prioritize anatomic location overlap)
weights = RelatedModelWeights(anatomic_location_ids=10.0, body_regions=1.0)
related = await index.related_models("OIFM_RADLEX_000001", weights=weights)
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
