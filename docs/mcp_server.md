# Finding Model MCP Server

The Finding Model MCP (Model Context Protocol) server provides AI agents with access to search and retrieve finding models from the DuckDB index.

## Overview

The MCP server exposes four tools that agents can use to interact with the Finding Model Index:

1. **search_finding_models**: Search for finding models using hybrid search (FTS + semantic)
2. **get_finding_model**: Retrieve a specific finding model by ID, name, or synonym
3. **list_finding_model_tags**: List all unique tags used across finding models
4. **count_finding_models**: Get statistics about the index (models, people, organizations)

## Installation

The MCP server is included in the `findingmodel` package. To use it, ensure you have installed the package with all dependencies:

```bash
pip install findingmodel
```

Or with `uv`:

```bash
uv add findingmodel
```

## Running the Server

### Standalone Execution

Run the server directly using Python:

```bash
python -m findingmodel.mcp_server
```

Or with `uv`:

```bash
uv run python -m findingmodel.mcp_server
```

### Claude Desktop Configuration

To use the MCP server with Claude Desktop, add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "finding-model-search": {
      "command": "python",
      "args": ["-m", "findingmodel.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key-here"
      }
    }
  }
}
```

If using `uv`:

```json
{
  "mcpServers": {
    "finding-model-search": {
      "command": "uv",
      "args": ["run", "python", "-m", "findingmodel.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key-here"
      }
    }
  }
}
```

**Note**: The server requires an OpenAI API key for semantic search functionality. Set the `OPENAI_API_KEY` environment variable or configure it in your `.env` file.

## Tools Reference

### search_finding_models

Search for finding models using hybrid search (FTS + semantic with Reciprocal Rank Fusion).

**Parameters:**
- `query` (string, required): Search query (e.g., "pneumothorax", "lung nodule")
- `limit` (integer, optional): Maximum results to return (default: 10, max: 100)
- `tags` (array of strings, optional): Filter by tags - models must have ALL specified tags

**Returns:** SearchResponse object containing:
- `query`: The search query used
- `limit`: The limit applied
- `tags`: Tags used for filtering (if any)
- `result_count`: Number of results returned
- `results`: Array of SearchResult objects, each containing:
  - `oifm_id`: Unique identifier
  - `name`: Display name
  - `slug_name`: URL-friendly name
  - `filename`: Source filename
  - `description`: Detailed description
  - `synonyms`: Alternative names
  - `tags`: Associated tags
  - `contributors`: List of contributors
  - `attributes`: Array of attributes with id, name, and type

**Example usage in Claude:**
```
Search for pneumothorax finding models with a limit of 5 results
```

### get_finding_model

Retrieve a specific finding model by its ID, name, or synonym.

**Parameters:**
- `identifier` (string, required): OIFM ID, name, slug name, or synonym to look up

**Returns:** SearchResult object or null if not found

**Example usage in Claude:**
```
Get the finding model with ID OIFM_RSNA_000001
```

### list_finding_model_tags

List all unique tags used across all finding models.

**Parameters:** None

**Returns:** Array of tag strings (sorted)

**Example usage in Claude:**
```
Show me all available tags for finding models
```

### count_finding_models

Get statistics about the finding model index.

**Parameters:** None

**Returns:** Object with counts:
- `finding_models`: Total number of finding models
- `people`: Total number of contributors (people)
- `organizations`: Total number of organizations

**Example usage in Claude:**
```
How many finding models are in the index?
```

## Database Configuration

The MCP server uses the DuckDB index for finding models. By default, it will:

1. Look for a local database file at `~/.local/share/findingmodel/finding_models.duckdb`
2. If not found, attempt to download the latest database from the remote URL configured in settings

You can override the database location by setting the `DUCKDB_INDEX_PATH` environment variable:

```bash
export DUCKDB_INDEX_PATH=/path/to/your/custom/database.duckdb
```

## Development

### Running Tests

```bash
uv run pytest test/test_mcp_server.py -v
```

### Type Checking

```bash
uv run mypy src/findingmodel/mcp_server.py
```

### Linting

```bash
uv run ruff check src/findingmodel/mcp_server.py
uv run ruff format src/findingmodel/mcp_server.py
```

## Troubleshooting

### OpenAI API Key Not Found

**Error**: `ConfigurationError: OpenAI API key not configured`

**Solution**: Set the `OPENAI_API_KEY` environment variable or add it to your `.env` file:

```bash
export OPENAI_API_KEY=your-api-key-here
```

### Database Download Fails

**Error**: `Failed to download/verify database file`

**Solution**:
1. Check your internet connection
2. Verify the remote database URL is accessible
3. Manually download the database and place it in `~/.local/share/findingmodel/finding_models.duckdb`

### Server Not Appearing in Claude Desktop

**Solution**:
1. Verify the configuration file path is correct for your OS
2. Restart Claude Desktop after updating the configuration
3. Check the Claude Desktop logs for errors

## API Design

The MCP server follows these design principles:

1. **Stateless**: Each tool call is independent
2. **Type-safe**: All inputs and outputs use Pydantic models for validation
3. **Comprehensive**: Provides both search and retrieval capabilities
4. **Efficient**: Uses hybrid search with RRF fusion for best results

## Contributing

To add new tools or modify existing ones:

1. Add the tool function in `src/findingmodel/mcp_server.py`
2. Decorate it with `@mcp.tool()`
3. Provide comprehensive docstrings (used by AI to understand the tool)
4. Add tests in `test/test_mcp_server.py`
5. Update this documentation

## License

MIT License - see LICENSE file for details.
