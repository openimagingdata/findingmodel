# Configuration Guide

This guide covers all configuration options for the `findingmodel` library.

## Overview

Configuration is managed through environment variables, typically set in a `.env` file in your project root. The library uses [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to load and validate configuration.

```bash
# Copy the sample and edit
cp .env.sample .env
```

## AI Model Providers

The library supports multiple AI providers through [Pydantic AI](https://ai.pydantic.dev/). Models are specified in the format `provider:model-name`.

### Provider Overview

| Provider | Prefix | API Key | Best For |
|----------|--------|---------|----------|
| OpenAI | `openai:` | `OPENAI_API_KEY` | Production default, best medical benchmarks |
| Anthropic | `anthropic:` | `ANTHROPIC_API_KEY` | Alternative cloud provider |
| Google Gemini | `google:` | `GOOGLE_API_KEY` | Cost-effective, fast |
| Ollama | `ollama:` | None (local) | Development, air-gapped environments |
| Gateway | `gateway/*:` | `PYDANTIC_AI_GATEWAY_API_KEY` | Enterprise proxy, centralized billing |

### OpenAI Setup

OpenAI is the default provider with the best performance on medical domain tasks.

1. Get an API key from [platform.openai.com](https://platform.openai.com/api-keys)
2. Add to your `.env`:

```bash
OPENAI_API_KEY=sk-...
```

**Recommended models:**
- `openai:gpt-5-mini` - Balanced cost/performance (default)
- `openai:gpt-5-nano` - Fastest, cheapest
- `openai:gpt-5.2` - Best reasoning capability

### Anthropic Setup

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)
2. Add to your `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=anthropic:claude-sonnet-4-5
```

**Recommended models:**
- `anthropic:claude-sonnet-4-5` - Balanced (recommended)
- `anthropic:claude-haiku-4-5` - Fast, cheap
- `anthropic:claude-opus-4-5` - Most capable

### Google Gemini Setup

Google Gemini offers competitive pricing and performance.

1. Get an API key from [AI Studio](https://aistudio.google.com/apikey)
2. Add to your `.env`:

```bash
GOOGLE_API_KEY=AI...
DEFAULT_MODEL=google:gemini-3-flash-preview
```

**Provider variants:**
- `google:` or `google-gla:` - Google Generative Language API (direct)
- `gateway/google:` or `gateway/google-vertex:` - Via Gateway using Vertex AI

**Recommended models:**
- `google:gemini-3-flash-preview` - Fast, cost-effective
- `google:gemini-2.0-flash` - Stable, production-ready

> **Note:** Models with `-preview` suffix are in preview and may change names when generally available.

### Ollama Setup (Local Models)

Ollama enables running models locally without API keys - ideal for development, testing, or air-gapped deployments.

#### Installation

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the server
ollama serve
```

#### Pulling Models

```bash
# Pull a model before using it
ollama pull llama3
ollama pull gpt-oss:20b
```

#### Configuration

```bash
# Use Ollama as default provider
DEFAULT_MODEL=ollama:llama3

# For remote Ollama server (e.g., GPU server)
OLLAMA_BASE_URL=http://gpu-server:11434/v1
```

#### Model Availability Validation

When you request an Ollama model, the library validates it's available on the server before proceeding. If the model isn't found or the server is unreachable, you'll get a clear error:

```
ConfigurationError: Ollama model 'mistral' not found.
Available: gpt-oss:20b, llama3:latest.
Pull with: ollama pull mistral
```

This fail-fast behavior prevents confusing errors later in your workflow.

### Pydantic AI Gateway

The Gateway provides a unified proxy for multiple providers, useful for:
- Centralized API key management
- Usage tracking and billing
- Enterprise security requirements

```bash
PYDANTIC_AI_GATEWAY_API_KEY=your-gateway-key
DEFAULT_MODEL=gateway/openai:gpt-5-mini
```

Gateway supports: `gateway/openai:`, `gateway/anthropic:`, `gateway/google:`

## Model Tier System

The library uses a three-tier model selection system to balance cost and capability.

### Tier Overview

| Tier | Environment Variable | Default | Use Case |
|------|---------------------|---------|----------|
| `small` | `DEFAULT_MODEL_SMALL` | `openai:gpt-5-nano` | Simple classification, query generation |
| `base` | `DEFAULT_MODEL` | `openai:gpt-5-mini` | Most agent workflows |
| `full` | `DEFAULT_MODEL_FULL` | `openai:gpt-5.2` | Complex reasoning, model editing |

### Programmatic Access

```python
from findingmodel_ai.config import settings

# Get model for a specific tier (requires findingmodel-ai package)
model = settings.get_model("base")   # Most workflows
model = settings.get_model("small")  # Fast/cheap tasks
model = settings.get_model("full")   # Complex reasoning
```

### Overriding Defaults

Override any tier in your `.env`:

```bash
# Use Gemini for base tier (cost savings)
DEFAULT_MODEL=google:gemini-3-flash-preview

# Use Claude for complex tasks
DEFAULT_MODEL_FULL=anthropic:claude-opus-4-5

# Use local model for simple tasks
DEFAULT_MODEL_SMALL=ollama:llama3
```

## Per-Agent Model Overrides

For advanced use cases, you can override the model used by specific agents without modifying code.

### Configuration

Override agents via environment variables:

```bash
AGENT_MODEL_OVERRIDES__enrich_classify=anthropic:claude-opus-4-5
AGENT_MODEL_OVERRIDES__edit_instructions=ollama:llama3
```

The agent checks for an override first, then falls back to its default tier.

### Agent Tags by Workflow

#### Finding Enrichment

Used by: `enrich_finding()`, `findingmodel-ai make-stub-model` CLI

| Tag | What It Does | Default |
|-----|--------------|---------|
| `enrich_classify` | Classifies finding category, modality, anatomy from search results | base |
| `enrich_unified` | Runs the full unified enrichment pipeline | base |
| `enrich_research` | Agentic enrichment that searches web before classifying | base |

#### Model Editing

Used by: `edit_model_natural_language()`, `edit_model_markdown()`

| Tag | What It Does | Default |
|-----|--------------|---------|
| `edit_instructions` | Applies natural language edit instructions to a finding model | base |
| `edit_markdown` | Applies edits from a modified markdown representation | base |

#### Finding Description

Used by: `findingmodel-ai make-info` CLI, `create_info_from_name()`, `add_details_to_info()`

| Tag | What It Does | Default |
|-----|--------------|---------|
| `describe_finding` | Generates initial FindingInfo from a finding name | caller-specified |
| `describe_details` | Adds citations and detailed descriptions via web search | small |

#### Similar Finding Search

Used by: `find_similar_models()`

| Tag | What It Does | Default |
|-----|--------------|---------|
| `similar_search` | Plans search strategy and generates search terms (2 agents share this tag) | base or small |
| `similar_assess` | Analyzes and ranks similar finding results | base |

#### Anatomic Location

Used by: anatomic location search during enrichment

| Tag | What It Does | Default |
|-----|--------------|---------|
| `anatomic_search` | Generates query terms for anatomic location lookup | small |
| `anatomic_select` | Selects best anatomic locations from search candidates | small |

#### Ontology Matching

Used by: ontology concept matching during enrichment

| Tag | What It Does | Default |
|-----|--------------|---------|
| `ontology_match` | Scores and categorizes ontology concept candidates | base |
| `ontology_search` | Generates query terms for ontology search | small |

#### Markdown Import

Used by: `findingmodel-ai markdown-to-fm` CLI, `create_model_from_markdown()`

| Tag | What It Does | Default |
|-----|--------------|---------|
| `import_markdown` | Converts markdown outline to structured FindingModel | base |

### When to Use

- **Cost optimization**: Use cheaper models for high-volume agents (e.g., `anatomic_search`)
- **Quality boost**: Use Claude Opus for complex editing tasks
- **Local development**: Override to Ollama for offline testing
- **A/B testing**: Compare model performance on specific workflows

### Startup Validation

For production applications, validate API keys are configured at startup:

```python
from findingmodel_ai.config import settings

# Raises ConfigurationError if keys missing for default models (requires findingmodel-ai)
settings.validate_default_model_keys()
```

This provides fail-fast behavior with clear error messages:
```
ConfigurationError: Missing API keys for default models:
default_model (openai:gpt-5-mini): OPENAI_API_KEY
```

## Database Configuration

The library uses DuckDB databases for the finding model index and anatomic location search. By default, databases are automatically downloaded on first use.

### Automatic Download (Default)

Databases are downloaded from a manifest to a platform-specific user data directory:
- macOS: `~/Library/Application Support/findingmodel/`
- Linux: `~/.local/share/findingmodel/`
- Windows: `C:\Users\<user>\AppData\Local\findingmodel\`

No configuration required - just use the library.

### Production/Docker Deployment

For production, pre-download or mount databases and specify paths:

```bash
# Absolute paths to pre-mounted databases
FINDINGMODEL_DB_PATH=/mnt/data/finding_models.duckdb
ANATOMIC_DB_PATH=/mnt/data/anatomic_locations.duckdb
```

### Custom Download URLs

Lock to specific database versions:

```bash
# Both URL and hash required
FINDINGMODEL_REMOTE_DB_URL=https://your-host/finding_models.duckdb
FINDINGMODEL_REMOTE_DB_HASH=sha256:abc123...

ANATOMIC_REMOTE_DB_URL=https://your-host/anatomic_locations.duckdb
ANATOMIC_REMOTE_DB_HASH=sha256:def456...
```

### Configuration Priority

1. If file exists and no URL/hash specified → uses file directly
2. If file exists with URL/hash → verifies hash, re-downloads if mismatch
3. If file doesn't exist with URL/hash → downloads from URL
4. If nothing specified → downloads from manifest.json (default)

For database maintenance and building, see [Database Management Guide](database-management.md).

## External Services

### Tavily Search

Required for `add_details_to_info()` which adds citations and detailed descriptions.

1. Get an API key from [tavily.com](https://tavily.com/)
2. Add to your `.env`:

```bash
TAVILY_API_KEY=tvly-...

# Optional: search depth (default: advanced)
TAVILY_SEARCH_DEPTH=advanced  # or "basic" for faster/cheaper
```

### BioOntology API

Required for searching 800+ medical ontologies (SNOMED-CT, ICD-10, LOINC, etc.) via the BioOntology backend.

1. Get an API key from [bioportal.bioontology.org](https://bioportal.bioontology.org/accounts/new)
2. Add to your `.env`:

```bash
BIOONTOLOGY_API_KEY=...
```

> **Note:** The DuckDB-based ontology search works without this key. BioOntology provides additional coverage.

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | One AI key required | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | One AI key required | - | Anthropic API key |
| `GOOGLE_API_KEY` | One AI key required | - | Google AI Studio API key |
| `PYDANTIC_AI_GATEWAY_API_KEY` | For gateway/* | - | Pydantic AI Gateway key |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama server URL |
| `DEFAULT_MODEL` | No | `openai:gpt-5-mini` | Base tier model |
| `DEFAULT_MODEL_FULL` | No | `openai:gpt-5.2` | Full tier model |
| `DEFAULT_MODEL_SMALL` | No | `openai:gpt-5-nano` | Small tier model |
| `AGENT_MODEL_OVERRIDES__<tag>` | No | - | Override model for specific agent tag (e.g., enrich_classify) |
| `TAVILY_API_KEY` | For citations | - | Tavily search API key |
| `TAVILY_SEARCH_DEPTH` | No | `advanced` | Search depth: basic/advanced |
| `BIOONTOLOGY_API_KEY` | For BioOntology | - | BioPortal API key |
| `FINDINGMODEL_DB_PATH` | No | Auto-download | Path to index database |
| `ANATOMIC_DB_PATH` | No | Auto-download | Path to anatomic database |
| `FINDINGMODEL_REMOTE_DB_URL` | With hash | - | Custom index download URL |
| `FINDINGMODEL_REMOTE_DB_HASH` | With URL | - | SHA256 hash for index |
| `ANATOMIC_REMOTE_DB_URL` | With hash | - | Custom anatomic download URL |
| `ANATOMIC_REMOTE_DB_HASH` | With URL | - | SHA256 hash for anatomic |
| `ANATOMIC_OPENAI_API_KEY` | No | Falls back to `OPENAI_API_KEY` | OpenAI key for anatomic semantic search |
| `LOGFIRE_TOKEN` | For tracing | - | Logfire.dev write token |
| `DISABLE_SEND_TO_LOGFIRE` | No | `false` | Disable cloud tracing |

## Troubleshooting

### "API key not configured"

Ensure the appropriate key is set for your model provider:
```bash
# Check which provider your model uses
echo $DEFAULT_MODEL  # e.g., "openai:gpt-5-mini" needs OPENAI_API_KEY
```

### "Ollama server not reachable"

1. Ensure Ollama is running: `ollama serve`
2. Check the URL: `curl http://localhost:11434/api/tags`
3. For remote servers, verify network access and `OLLAMA_BASE_URL`

### "Ollama model not found"

Pull the model first:
```bash
ollama pull <model-name>
```

### Database download failures

1. Check network connectivity
2. Verify custom URLs/hashes if configured
3. Check write permissions to data directory
4. See [Database Management Guide](database-management.md) for manual download

### Startup validation fails

If `validate_default_model_keys()` fails, ensure API keys are set for all three default model tiers, or override defaults to use a provider you have configured:

```bash
# Use Ollama for all tiers (no API keys needed)
DEFAULT_MODEL=ollama:llama3
DEFAULT_MODEL_FULL=ollama:llama3:70b
DEFAULT_MODEL_SMALL=ollama:llama3
```
