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
# Required for embedding-based search
OPENAI_API_KEY=your_key_here

# Optional: Custom database path
FINDINGMODEL_DB_PATH=/mnt/data/finding_models.duckdb
```

The finding model database is automatically downloaded on first use.

## CLI (`findingmodel`)

```bash
# View configuration
findingmodel config


# Show index statistics
findingmodel stats

# Search for finding models
findingmodel search "lung nodule"

# Convert finding model to Markdown
findingmodel fm-to-markdown model.json
```

## Models

### FindingModelBase

Basic finding model structure with name, description, and attributes.

```python
from findingmodel import FindingModelBase

model = FindingModelBase(
    name="pneumothorax",
    description="Presence of air in the pleural space",
    attributes=[...]
)

# Export to Markdown
print(model.as_markdown())
```

### FindingModelFull

Extended model with OIFM IDs, index codes, and contributor information.

```python
from findingmodel import FindingModelFull

# Load from JSON
model = FindingModelFull.model_validate_json(json_content)

print(f"Model ID: {model.oifm_id}")
for attr in model.attributes:
    print(f"  {attr.name}: {attr.oifma_id}")
```

### FindingInfo

Metadata about a finding including description, synonyms, and optional citations.

```python
from findingmodel import FindingInfo

info = FindingInfo(
    name="pneumothorax",
    synonyms=["PTX", "collapsed lung"],
    description="Presence of air in the pleural space"
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
- **list_finding_model_tags**: List all available tags
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
