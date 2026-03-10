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
| Google | `google:` / `google-gla:` / `google-vertex:` | `GOOGLE_API_KEY` | Cost-effective, fast (AI Studio API) |
| Ollama | `ollama:` | None (local) | Development, air-gapped environments |
| Gateway | `gateway/*:` | `PYDANTIC_AI_GATEWAY_API_KEY` | Enterprise proxy, centralized billing |

> **Gateway fallback:** If a provider-specific key (e.g., `OPENAI_API_KEY`) is missing but `PYDANTIC_AI_GATEWAY_API_KEY` is set, requests are automatically routed through the gateway. This means you can use a single gateway key for all providers. Direct keys take priority when present.
>
> **Google prefixes:** `google:`, `google-gla:`, and `google-vertex:` are interchangeable — the system routes to Google AI Studio (GLA) when `GOOGLE_API_KEY` is set, or to Vertex AI via the gateway when only `PYDANTIC_AI_GATEWAY_API_KEY` is available.

### OpenAI Setup

OpenAI is the default provider with the best performance on medical domain tasks.

1. Get an API key from [platform.openai.com](https://platform.openai.com/api-keys)
2. Add to your `.env`:

```bash
OPENAI_API_KEY=sk-...
```

**Supported models:**
- `openai:gpt-5.4` - Flagship reasoning model (default base and full tiers)
- `openai:gpt-5-mini` - Cost-effective option (good for base tier overrides)
- `openai:gpt-5-nano` - Lightest option for high-volume simple tasks

### Anthropic Setup

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)
2. Add to your `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=anthropic:claude-sonnet-4-6
```

**Supported models:**
- `anthropic:claude-sonnet-4-6` - Balanced (recommended)
- `anthropic:claude-haiku-4-5` - Fast, low cost
- `anthropic:claude-opus-4-6` - Most capable

### Google Gemini Setup

Google Gemini offers competitive pricing and performance.

1. Get an API key from [AI Studio](https://aistudio.google.com/apikey)
2. Add to your `.env`:

```bash
GOOGLE_API_KEY=AI...
DEFAULT_MODEL=google-gla:gemini-3-flash-preview
```

**Provider prefixes** — all three are interchangeable and route automatically:
- `google:` / `google-gla:` / `google-vertex:` — The system picks the best path based on your keys:
  - `GOOGLE_API_KEY` set → routes to Google AI Studio (GLA)
  - Only `PYDANTIC_AI_GATEWAY_API_KEY` set → routes to Vertex AI via gateway
- `gateway/google:` or `gateway/google-vertex:` — Explicitly routes through the Pydantic AI Gateway (Vertex AI backend).

**Supported models:**
- `google-gla:gemini-3-flash-preview` - Fast, cost-effective (default small tier)
- `google-gla:gemini-3.1-pro-preview` - Full-capability Gemini 3.1 Pro
- `google-gla:gemini-3.1-flash-lite-preview` - Lightest Gemini option

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

## Supported Models

The following models are tested and supported. Model names must be specified exactly as shown — the API name matters and outdated names will fail at runtime.

### OpenAI

| Model | Spec | Notes |
|-------|------|-------|
| GPT-5.4 | `openai:gpt-5.4` | Flagship; supports `xhigh` reasoning; **base and full tier default** |
| GPT-5-mini | `openai:gpt-5-mini` | Fast, economical; good base-tier override |
| GPT-5-nano | `openai:gpt-5-nano` | Lightest option; high-volume simple tasks |

> **Note:** `openai:gpt-5` is a real model but not on the approved list — use `gpt-5.4` instead.

### Anthropic

| Model | Spec | Notes |
|-------|------|-------|
| Claude Sonnet 4.6 | `anthropic:claude-sonnet-4-6` | Balanced (recommended for Anthropic tier) |
| Claude Haiku 4.5 | `anthropic:claude-haiku-4-5` | Fast, low cost |
| Claude Opus 4.6 | `anthropic:claude-opus-4-6` | Most capable; use for complex editing |

> **Naming note:** Anthropic API names use hyphens, not dots: `claude-sonnet-4-6` not `claude-sonnet-4.6`.

### Google Gemini (via `google-gla:` or `gateway/google:`)

| Model | Spec | Notes |
|-------|------|-------|
| Gemini 3 Flash | `google-gla:gemini-3-flash-preview` | Fast, cost-effective; **small tier default** |
| Gemini 3.1 Flash Lite | `google-gla:gemini-3.1-flash-lite-preview` | Lightest Gemini option |
| Gemini 3.1 Pro | `google-gla:gemini-3.1-pro-preview` | Full-capability Gemini; use for complex tasks |

> **Provider note:** All Google prefixes (`google:`, `google-gla:`, `google-vertex:`) are interchangeable — the system routes to AI Studio (GLA) if `GOOGLE_API_KEY` is set, or Vertex AI via gateway if only `PYDANTIC_AI_GATEWAY_API_KEY` is available.

### Ollama (Local / Self-Hosted)

| Model | Spec | Notes |
|-------|------|-------|
| Llama 4 Maverick | `ollama:llama4:maverick` | Meta Llama 4 16x17B MoE (pull: `ollama pull llama4:maverick`) |
| Llama 3.3 | `ollama:llama3.3` | Llama 3.3 70B (pull: `ollama pull llama3.3`) |

Pull any Ollama model with `ollama pull <name>` before use. The library validates availability at startup and will list what's actually available if the model isn't found.

## Model Tier System

The library uses a three-tier model selection system to balance cost and capability.

### Tier Overview

| Tier | Environment Variable | Default | Reasoning Default | Use Case |
|------|---------------------|---------|-------------------|----------|
| `small` | `DEFAULT_MODEL_SMALL` | `google-gla:gemini-3-flash-preview` | `low` | Simple classification, query generation |
| `base` | `DEFAULT_MODEL` | `openai:gpt-5.4` | `none` | Most agent workflows |
| `full` | `DEFAULT_MODEL_FULL` | `openai:gpt-5.4` | `high` | Complex reasoning, model editing |

> **Note on API keys:** The small-tier default uses Google Gemini and requires `GOOGLE_API_KEY`. If you only have an OpenAI key, set `DEFAULT_MODEL_SMALL=openai:gpt-5-mini` in your `.env`.

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
# OpenAI-only setup (no Google key)
DEFAULT_MODEL_SMALL=openai:gpt-5-mini

# Use Claude for complex tasks
DEFAULT_MODEL_FULL=anthropic:claude-opus-4-6

# Use local model for simple tasks
DEFAULT_MODEL_SMALL=ollama:llama3
```

### Reasoning Level Configuration

Each tier has a configurable reasoning level that controls how much computational effort the model uses. Reasoning improves quality but increases cost and latency.

| Variable | Default | Options |
|----------|---------|---------|
| `DEFAULT_REASONING_SMALL` | `low` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `DEFAULT_REASONING_BASE` | `none` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `DEFAULT_REASONING_FULL` | `high` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |

```bash
# Disable reasoning for cost savings on small tasks
DEFAULT_REASONING_SMALL=none

# Use high reasoning for base tier
DEFAULT_REASONING_BASE=high

# Downgrade full-tier reasoning to save cost
DEFAULT_REASONING_FULL=low
```

Reasoning is applied using provider-native typed settings:
- **OpenAI** (`gpt-5.4`, `gpt-5-mini`, etc.): `openai_reasoning_effort` — `none` disables reasoning entirely; `xhigh` is supported on gpt-5.4 for maximum effort
- **Google Gemini**: `google_thinking_config` with `thinking_level` — valid levels are `MINIMAL`, `LOW`, `MEDIUM`, `HIGH` (no higher level exists; `xhigh` maps to `HIGH`). Thinking cannot be fully disabled on Gemini 3 models: Flash maps `none`/`minimal` → `MINIMAL`; Pro maps `none`/`minimal` → `LOW` (the minimum Pro supports)
- **Anthropic**: `anthropic_thinking` — Opus 4.6+ uses adaptive thinking with `anthropic_effort` (low/medium/high); older models (Sonnet 4.6, Haiku 4.5) use extended thinking with `budget_tokens`. `none` disables thinking on all models

## Per-Agent Model Overrides

For advanced use cases, you can override the model used by specific agents without modifying code.

### Configuration

Override agents via environment variables:

```bash
AGENT_MODEL_OVERRIDES__enrich_classify=anthropic:claude-opus-4-6
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

### Planned: Per-Agent Reasoning Overrides

A future companion to `AGENT_MODEL_OVERRIDES__*` will allow setting reasoning level per agent:

```bash
# Not yet implemented — planned
AGENT_REASONING_OVERRIDES__edit_instructions=high
AGENT_REASONING_OVERRIDES__anatomic_search=none
```

This will follow the same nested-delimiter pattern as model overrides. For now, use `DEFAULT_REASONING_SMALL/BASE/FULL` to set tier-wide defaults.

### Startup Validation

The CLI validates API keys before running any AI command. If a required key is missing, you'll see a clear error with instructions before any agent runs:

```
Error: Missing API keys for default models: default_model_small
(google-gla:gemini-3-flash-preview) requires GOOGLE_API_KEY.
Set the missing key(s) in .env or your environment,
or override the model tier (e.g., DEFAULT_MODEL_SMALL=openai:gpt-5-mini).
```

For programmatic use, call `validate_default_model_keys()` explicitly at application startup:

```python
from findingmodel_ai.config import settings

settings.validate_default_model_keys()
```

Error messages include an override hint:
```
ConfigurationError: Missing API keys for default models:
default_model_small (google-gla:gemini-3-flash-preview) requires GOOGLE_API_KEY.
Set the missing key(s) in .env or your environment,
or override the model tier (e.g., DEFAULT_MODEL_SMALL=openai:gpt-5-mini).
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

### Embedding Profile Selection for Index Search

Runtime embedding profile selection is package-specific:

- `findingmodel` supports `openai` and `local` profile DB artifacts.
- `anatomic-locations` currently supports only OpenAI-embedded DB artifacts.

```bash
# findingmodel runtime profile (default: auto)
FINDINGMODEL_EMBEDDING_PROFILE=auto
# FINDINGMODEL_EMBEDDING_PROFILE=local

# anatomic-locations runtime profile (default: openai; auto resolves to openai)
ANATOMIC_EMBEDDING_PROFILE=openai
```

Supported profiles:

- `findingmodel: auto` → `openai` / `text-embedding-3-small` / `512` when an OpenAI key is set; otherwise `fastembed` / `BAAI/bge-small-en-v1.5` / `384`
- `findingmodel: openai` → `openai` / `text-embedding-3-small` / `512`
- `findingmodel: local` → `fastembed` / `BAAI/bge-small-en-v1.5` / `384`
- `anatomic-locations: auto/openai` → `openai` / `text-embedding-3-small` / `512`

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

Example CLI usage:

```bash
findingmodel-ai ontology search "pneumothorax" --ontology SNOMEDCT --max-results 10
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
| `DEFAULT_MODEL` | No | `openai:gpt-5.4` | Base tier model |
| `DEFAULT_MODEL_FULL` | No | `openai:gpt-5.4` | Full tier model |
| `DEFAULT_MODEL_SMALL` | No | `google-gla:gemini-3-flash-preview` | Small tier model (requires `GOOGLE_API_KEY` or `PYDANTIC_AI_GATEWAY_API_KEY`) |
| `DEFAULT_REASONING_SMALL` | No | `low` | Reasoning level for small tier |
| `DEFAULT_REASONING_BASE` | No | `none` | Reasoning level for base tier |
| `DEFAULT_REASONING_FULL` | No | `high` | Reasoning level for full tier |
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
| `FINDINGMODEL_EMBEDDING_PROFILE` | No | `auto` | Runtime embedding profile for findingmodel index (`auto`, `openai`, or `local`) |
| `ANATOMIC_EMBEDDING_PROFILE` | No | `openai` | Runtime embedding profile for anatomic index (`auto` or `openai`; `local` is unsupported) |
| `ANATOMIC_OPENAI_API_KEY` | No | Falls back to `OPENAI_API_KEY` | OpenAI key for anatomic semantic search |
| `OIDM_MAINTAIN_EMBEDDING_PROVIDER` | Maintainers | `openai` | Build/publish embedding provider for `oidm-maintain` |
| `OIDM_MAINTAIN_EMBEDDING_MODEL` | Maintainers | `text-embedding-3-small` | Build/publish embedding model |
| `OIDM_MAINTAIN_EMBEDDING_DIMENSIONS` | Maintainers | `512` | Build/publish embedding dimensions |
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
