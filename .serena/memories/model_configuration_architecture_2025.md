# Model Configuration Architecture (December 2025)

## Overview

Findingmodel supports multiple AI providers with tier-based model selection and per-agent configuration.

## ModelSpec Type

Validated model string format (see `src/findingmodel/config.py`):
```python
MODEL_SPEC_PATTERN = r"^(openai|anthropic|gemini|ollama|gateway/(openai|anthropic|gemini)):[\w.:-]+$"
```

Examples: `openai:gpt-5-mini`, `anthropic:claude-sonnet-4-5`, `gateway/openai:gpt-5`, `ollama:llama3.2:70b`

## Test Constants (`test/conftest.py`)

```python
TEST_OPENAI_MODEL = "openai:gpt-5-nano"      # Cheapest for tests
TEST_ANTHROPIC_MODEL = "anthropic:claude-haiku-4-5"
```

**Rule**: Never hard-code model strings in tests; always use these constants.

## Tier-Based Selection

| Tier | Default | Use Case |
|------|---------|----------|
| `small` | `openai:gpt-5-nano` | Simple classification, query generation |
| `base` | `openai:gpt-5-mini` | Most agent workflows |
| `full` | `openai:gpt-5.2` | Complex reasoning, editing |

Access via: `settings.get_model("base")` or `settings.get_model("small")`

## Provider Support

| Provider | Status | Config | Factory Method |
|----------|--------|--------|----------------|
| OpenAI | ✅ Complete | `OPENAI_API_KEY` | `_make_openai_model()` |
| Anthropic | ✅ Complete | `ANTHROPIC_API_KEY` | `_make_anthropic_model()` |
| Gateway | ✅ Complete | `PYDANTIC_AI_GATEWAY_API_KEY` | via openai/anthropic factories |
| Gemini | 🔲 Planned | `GOOGLE_API_KEY` | Phase 2 |
| Ollama | 🔲 Planned | `OLLAMA_BASE_URL` | Phase 3 |

## Implementation Phases

### Phase 1: Foundation ✅ Complete
- Added ModelSpec validated type
- Created test constants
- Updated all hard-coded models in tests
- Updated config.py defaults

### Phase 2: Gemini Support (Pending)
- Add `google_api_key` config field
- Add `_make_gemini_model()` factory
- Add routing for `gemini:` prefix
- Add `google` extra to pydantic-ai-slim dependency

### Phase 3: Ollama Support (Pending)
- Add `ollama_base_url` config field
- Add `_make_ollama_model()` factory
- Add routing for `ollama:` prefix

### Phase 4: Per-Agent Configuration (Pending)
- Add nested `AgentModels` config class with `env_nested_delimiter='__'`
- Add `get_model_for_agent()` method
- Update 15 agent factories across 8 tool files

## 15 Agents Requiring Configuration

| Agent | File | Default Tier |
|-------|------|--------------|
| enrichment | finding_enrichment.py | base |
| enrichment_unified | finding_enrichment.py | base |
| agentic_enrichment | finding_enrichment_agentic.py | base |
| anatomic_query | anatomic_location_search.py | small |
| location_selection | anatomic_location_search.py | small |
| search | similar_finding_models.py | base |
| term_generation | similar_finding_models.py | small |
| analysis | similar_finding_models.py | base |
| categorization | ontology_concept_match.py | base |
| query_generator | ontology_concept_match.py | small |
| edit | model_editor.py | base |
| markdown_edit | model_editor.py | base |
| finding_info | finding_description.py | varies |
| finding_details | finding_description.py | small |
| markdown_in | markdown_in.py | base |

## Plan File

Full implementation plan: `tasks/model-configuration-expansion.md`
