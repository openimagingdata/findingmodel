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
- `openai:gpt-5.4` - Flagship reasoning model; used for editing agents
- `openai:gpt-5.4-mini` - Fast default base-tier model
- `openai:gpt-5.4-nano` - Lightest option for high-volume simple tasks

### Anthropic Setup

1. Get an API key from [console.anthropic.com](https://console.anthropic.com/)
2. Add to your `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
AGENT_MODEL_OVERRIDES__edit_instructions=anthropic:claude-sonnet-4-6
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
AGENT_MODEL_OVERRIDES__ontology_search=google-gla:gemini-3-flash-preview
```

**Provider prefixes** — all three are interchangeable and route automatically:
- `google:` / `google-gla:` / `google-vertex:` — The system picks the best path based on your keys:
  - `GOOGLE_API_KEY` set → routes to Google AI Studio (GLA)
  - Only `PYDANTIC_AI_GATEWAY_API_KEY` set → routes to Vertex AI via gateway
- `gateway/google:` or `gateway/google-vertex:` — Explicitly routes through the Pydantic AI Gateway (Vertex AI backend).

**Supported models:**
- `google-gla:gemini-3-flash-preview` - Fast, cost-effective; fallback for generative agents
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
AGENT_MODEL_OVERRIDES__describe_finding=ollama:llama3

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
AGENT_MODEL_OVERRIDES__ontology_search=gateway/openai:gpt-5-mini
```

Gateway supports: `gateway/openai:`, `gateway/anthropic:`, `gateway/google:`

## Supported Models

The following models are tested and supported. Model names must be specified exactly as shown — the API name matters and outdated names will fail at runtime.

### OpenAI

| Model | Spec | Notes |
|-------|------|-------|
| GPT-5.4 | `openai:gpt-5.4` | Flagship; supports `xhigh` reasoning; used for editing agents |
| GPT-5.4-mini | `openai:gpt-5.4-mini` | Fast, economical; used for classification agents |
| GPT-5.4-nano | `openai:gpt-5.4-nano` | Lightest option; high-volume simple tasks |

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
| Gemini 3 Flash | `google-gla:gemini-3-flash-preview` | Fast, cost-effective; fallback for generative agents |
| Gemini 3.1 Flash Lite | `google-gla:gemini-3.1-flash-lite-preview` | Lightest Gemini option |
| Gemini 3.1 Pro | `google-gla:gemini-3.1-pro-preview` | Full-capability Gemini; use for complex tasks |

> **Provider note:** All Google prefixes (`google:`, `google-gla:`, `google-vertex:`) are interchangeable — the system routes to AI Studio (GLA) if `GOOGLE_API_KEY` is set, or Vertex AI via gateway if only `PYDANTIC_AI_GATEWAY_API_KEY` is available.

### Ollama (Local / Self-Hosted)

| Model | Spec | Notes |
|-------|------|-------|
| Llama 4 Maverick | `ollama:llama4:maverick` | Meta Llama 4 16x17B MoE (pull: `ollama pull llama4:maverick`) |
| Llama 3.3 | `ollama:llama3.3` | Llama 3.3 70B (pull: `ollama pull llama3.3`) |

Pull any Ollama model with `ollama pull <name>` before use. The library validates availability at startup and will list what's actually available if the model isn't found.

## Model Selection

Each agent has its own model + reasoning configuration defined in `supported_models.toml`. There is no global tier system — each agent declares an ordered fallback chain across providers (OpenAI, Google, Anthropic). The system works with any single API key configured.

### How It Works

When an agent needs a model, the resolver checks (in order):
1. **Env override** (`AGENT_MODEL_OVERRIDES__<tag>`) — single model, highest priority
2. **TOML fallback chain** — filters to available providers, wraps in `FallbackModel`
3. **Error** — if nothing is available, fails with a clear message

### Programmatic Access

```python
from findingmodel_ai.config import settings

# Get model for a specific agent (per-agent chain from TOML)
model = settings.get_agent_model("ontology_search")
model = settings.get_agent_model("edit_instructions")
```

### Overriding Per-Agent

Override model and/or reasoning for specific agents via environment:

```bash
# Override a specific agent's model
AGENT_MODEL_OVERRIDES__ontology_search=anthropic:claude-sonnet-4-6

# Override reasoning level
AGENT_REASONING_OVERRIDES__anatomic_select=medium

# Use local model for an agent
AGENT_MODEL_OVERRIDES__describe_finding=ollama:llama3
```

### Reasoning Level Configuration

Each agent's reasoning level is set in `supported_models.toml` alongside its model. Reasoning controls how much computational effort the model uses — it improves quality but increases cost and latency. Override per-agent via environment:

```bash
AGENT_REASONING_OVERRIDES__anatomic_select=medium
AGENT_REASONING_OVERRIDES__ontology_search=none
```

Valid levels: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. Levels are normalized per-model automatically (e.g., Gemini Flash maps `none` → `minimal` since it can't fully disable thinking).

Reasoning is applied using provider-native typed settings:
- **OpenAI** (`gpt-5.4`, `gpt-5-mini`, etc.): `openai_reasoning_effort` — `none` disables reasoning entirely; `xhigh` is supported on gpt-5.4 for maximum effort
- **Google Gemini**: `google_thinking_config` with `thinking_level` — valid levels are `MINIMAL`, `LOW`, `MEDIUM`, `HIGH` (no higher level exists; `xhigh` maps to `HIGH`). Thinking cannot be fully disabled on Gemini 3 models: Flash maps `none`/`minimal` → `MINIMAL`; Pro maps `none`/`minimal` → `LOW` (the minimum Pro supports)
- **Anthropic**: `anthropic_thinking` — Opus 4.6+ uses adaptive thinking with `anthropic_effort` (low/medium/high); older models (Sonnet 4.6, Haiku 4.5) use extended thinking with `budget_tokens`. `none` disables thinking on all models

### Per-Agent Defaults

Defaults are declared in `supported_models.toml` under `[agents.<tag>]`. Each entry lists models ordered by latency/quality preference, covering all three major providers.

#### Generative Agents (nano primary — 1-2s)

| Tag | Primary Model | Reasoning | Fallbacks |
|-----|--------------|-----------|-----------|
| `ontology_search` | `gpt-5.4-nano` | low | gemini-flash, haiku |
| `describe_finding` | `gpt-5.4-nano` | none | gemini-flash, haiku |
| `anatomic_search` | `gpt-5.4-nano` | low | gemini-flash, haiku |
| `similar_plan` | `gpt-5.4-nano` | low | gemini-flash, haiku |
| `describe_details` | `gpt-5.4-nano` | low | gemini-flash, haiku |

#### Classification Agents (mini primary — 5s)

| Tag | Primary Model | Reasoning | Fallbacks |
|-----|--------------|-----------|-----------|
| `ontology_match` | `gpt-5.4-mini` | none | gemini-3.1-pro, sonnet |
| `anatomic_select` | `gpt-5.4-mini` | none | gemini-3.1-pro, sonnet |
| `similar_select` | `gpt-5.4-mini` | low | gemini-3.1-pro, sonnet |
| `metadata_assign` | `gpt-5.4-mini` | low | gemini-3.1-pro, sonnet |

#### Complex Structured Output (gpt-5.4/opus — 6s+)

| Tag | Primary Model | Reasoning | Fallbacks |
|-----|--------------|-----------|-----------|
| `edit_instructions` | `gpt-5.4` | low | opus/medium, gemini-3.1-pro |
| `edit_markdown` | `gpt-5.4` | low | opus/medium, gemini-3.1-pro |
| `import_markdown` | `claude-opus-4-6` | medium | gpt-5.4/low, gemini-3.1-pro |

### Provider Availability & Fallback

The system works with **any single provider configured**. If you only have `OPENAI_API_KEY`, agents that prefer Google or Anthropic models will automatically use their OpenAI fallback. With all three keys, agents get full `FallbackModel` protection — if one provider has an API error, the request automatically retries on the next provider.

### Startup Validation

The CLI validates API keys before running any AI command. If a required key is missing, you'll see a clear error with instructions before any agent runs:

```
ConfigurationError: No models available for agents: ontology_search, ontology_match.
Set at least one provider API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
or PYDANTIC_AI_GATEWAY_API_KEY) in .env or your environment.
```

For programmatic use, call `validate_default_model_keys()` explicitly at application startup:

```python
from findingmodel_ai.config import settings

settings.validate_default_model_keys()
```

Error messages include remediation guidance:
```
ConfigurationError: No models available for agents: ontology_search, ontology_match.
Set at least one provider API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
or PYDANTIC_AI_GATEWAY_API_KEY) in .env or your environment.
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
FINDINGMODEL_MANIFEST_URL=https://your-host/manifest.json

ANATOMIC_REMOTE_DB_URL=https://your-host/anatomic_locations.duckdb
ANATOMIC_REMOTE_DB_HASH=sha256:def456...
ANATOMIC_MANIFEST_URL=https://your-host/manifest.json
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
| `AGENT_MODEL_OVERRIDES__<tag>` | No | - | Override model for specific agent tag (e.g., metadata_assign) |
| `AGENT_REASONING_OVERRIDES__<tag>` | No | - | Override reasoning level for specific agent tag |
| `TAVILY_API_KEY` | For citations | - | Tavily search API key |
| `TAVILY_SEARCH_DEPTH` | No | `advanced` | Search depth: basic/advanced |
| `BIOONTOLOGY_API_KEY` | For BioOntology | - | BioPortal API key |
| `FINDINGMODEL_DB_PATH` | No | Auto-download | Path to index database |
| `ANATOMIC_DB_PATH` | No | Auto-download | Path to anatomic database |
| `FINDINGMODEL_REMOTE_DB_URL` | With hash | - | Custom index download URL |
| `FINDINGMODEL_REMOTE_DB_HASH` | With URL | - | SHA256 hash for index |
| `FINDINGMODEL_MANIFEST_URL` | No | Hosted manifest | Manifest URL for findingmodel database artifacts |
| `ANATOMIC_REMOTE_DB_URL` | With hash | - | Custom anatomic download URL |
| `ANATOMIC_REMOTE_DB_HASH` | With URL | - | SHA256 hash for anatomic |
| `ANATOMIC_MANIFEST_URL` | No | Hosted manifest | Manifest URL for anatomic database artifacts |
| `FINDINGMODEL_EMBEDDING_PROFILE` | No | `auto` | Runtime embedding profile for findingmodel index (`auto`, `openai`, or `local`) |
| `ANATOMIC_EMBEDDING_PROFILE` | No | `openai` | Runtime embedding profile for anatomic index (`auto` or `openai`; `local` is unsupported) |
| `ANATOMIC_OPENAI_API_KEY` | No | Falls back to `OPENAI_API_KEY` | OpenAI key for anatomic semantic search |
| `OIDM_MAINTAIN_EMBEDDING_PROVIDER` | Maintainers | `openai` | Build/publish embedding provider for `oidm-maintain` |
| `OIDM_MAINTAIN_EMBEDDING_MODEL` | Maintainers | `text-embedding-3-small` | Build/publish embedding model |
| `OIDM_MAINTAIN_EMBEDDING_DIMENSIONS` | Maintainers | `512` | Build/publish embedding dimensions |
| `LOGFIRE_TOKEN` | For eval tracing | - | Optional Logfire write token used by eval instrumentation |

## Troubleshooting

### "API key not configured"

Ensure the appropriate key is set for your model provider:
```bash
# Check which provider your model uses
# Check your agent overrides and which keys are set
env | grep -E 'OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|AGENT_MODEL_OVERRIDES'
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

If `validate_default_model_keys()` fails, ensure at least one API key is set for a provider in the agent chains, or override specific agents to use Ollama:

```bash
# Use Ollama for specific agents (no API keys needed)
AGENT_MODEL_OVERRIDES__describe_finding=ollama:llama3
AGENT_MODEL_OVERRIDES__ontology_search=ollama:llama3
AGENT_MODEL_OVERRIDES__edit_instructions=ollama:llama3:70b
```
