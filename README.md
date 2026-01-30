# Open Imaging Finding Models

A Python workspace for managing Open Imaging Finding Models - structured data models used to describe medical imaging findings in radiology reports.

## Packages

| Package | Description | Install |
|---------|-------------|---------|
| **[findingmodel](packages/findingmodel/README.md)** | Core models, Index API, MCP server, `findingmodel` CLI | `pip install findingmodel` |
| **[findingmodel-ai](packages/findingmodel-ai/README.md)** | AI-powered tools, `findingmodel-ai` CLI | `pip install findingmodel-ai` |
| **[anatomic-locations](packages/anatomic-locations/README.md)** | Anatomic location queries, `anatomic-locations` CLI | `pip install anatomic-locations` |

Internal packages (not published):
- **[oidm-common](packages/oidm-common/README.md)** - Shared infrastructure
- **[oidm-maintenance](packages/oidm-maintenance/README.md)** - Database build/publish (maintainers only)

## Quick Start

```bash
# Install core package
pip install findingmodel

# Install AI tools (requires API key)
pip install findingmodel-ai
```

Create a `.env` file:

```bash
# Required for embedding-based search and AI tools
OPENAI_API_KEY=your_key_here

# Or use another provider
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### Using the Index

```python
import asyncio
from findingmodel import Index

async def main():
    async with Index() as index:
        # Search for finding models
        results = await index.search("pneumothorax", limit=5)
        for result in results:
            print(f"- {result.name}: {result.description}")

asyncio.run(main())
```

### Using AI Tools

```python
import asyncio
from findingmodel_ai.authoring import create_info_from_name

async def main():
    info = await create_info_from_name("pneumothorax")
    print(f"Name: {info.name}")
    print(f"Description: {info.description}")

asyncio.run(main())
```

### CLI Tools

```bash
# Core tools
findingmodel config              # View configuration
findingmodel index stats         # Index statistics

# AI tools
findingmodel-ai make-info "pneumothorax"     # Generate finding info
findingmodel-ai make-stub-model "finding"    # Create model template

# Anatomic locations
anatomic-locations search "knee joint"       # Search locations
```

## Configuration

See [Configuration Guide](docs/configuration.md) for:
- AI provider setup (OpenAI, Anthropic, Google, Ollama)
- Model tier configuration
- Per-agent model overrides
- Database path customization

## Documentation

- [Configuration Guide](docs/configuration.md) - API keys, providers, model selection
- [Anatomic Locations Guide](docs/anatomic-locations.md) - Anatomic location features
- [MCP Server Guide](docs/mcp_server.md) - Claude Desktop integration
- [Database Management](docs/database-management.md) - For maintainers

## Development

```bash
# Install all packages in development mode
uv sync

# Run tests
task test           # Unit tests (no API calls)
task test-full      # Integration tests

# Run checks
task check          # Format, lint, type check
```

## License

MIT
