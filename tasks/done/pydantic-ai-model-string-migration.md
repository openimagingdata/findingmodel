# Pydantic AI Model String Configuration Migration

**Status**: Complete
**Created**: 2025-11-26
**Completed**: 2025-11-27
**Branch**: `dev`

## Overview

Migrate from separate provider/model-name configuration to Pydantic AI's unified `provider:model` string format. This enables:

- Using any supported provider (OpenAI, Anthropic, Gateway)
- Switching providers via environment variables without code changes
- Future-proof configuration for new providers

## Decisions Made

1. **Remove `provider` parameter** from functions like `create_info_from_name()` - not used externally
2. **Update tests** to check string content instead of `isinstance(model, OpenAIModel)`
3. **No backward compatibility** for old env vars - document in CHANGELOG, bump minor version
4. **Embeddings unchanged** - still use direct OpenAI client (separate concern)
5. **Support three providers initially** - `gateway/*`, `openai`, `anthropic`

## Implementation Steps

### Step 1: Update `config.py` - Core Changes ✅

**File**: `src/findingmodel/config.py`

- [x] Add new model string fields:
  - `default_model: str = "openai:gpt-5-mini"`
  - `default_model_full: str = "openai:gpt-5"`
  - `default_model_small: str = "openai:gpt-5-nano"`
  - `pydantic_ai_gateway_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))`

- [x] Add `get_model()` method to `FindingModelConfig` class that returns a `pydantic_ai.models.Model` object (not a string)

- [x] Add `_create_model_from_string()` private method that:
  - Parses "provider:model_name" format
  - Returns `OpenAIResponsesModel` for OpenAI (uses Responses API, not Chat Completions)
  - Returns `AnthropicModel` for Anthropic
  - Returns string for gateway (let Pydantic AI handle it)
  - Raises `ConfigurationError` for unknown providers

- [x] Remove old fields:
  - `openai_default_model`, `openai_default_model_full`, `openai_default_model_small`
  - `anthropic_default_model`, `anthropic_default_model_full`, `anthropic_default_model_small`
  - `model_provider`

- [x] Remove methods:
  - `check_ready_for_openai()`
  - `check_ready_for_anthropic()`

- [x] Update `__all__` to remove `ModelProvider`

- [x] Keep: `openai_api_key`, `anthropic_api_key`, `check_ready_for_tavily()`, embedding settings

### Step 2: Update `tools/common.py` ✅

**File**: `src/findingmodel/tools/common.py`

- [x] Remove `get_model()` function
- [x] Remove `get_openai_model()` deprecated function
- [x] Remove imports: `OpenAIModel`, `AnthropicModel`, `OpenAIProvider`, `AnthropicProvider`
- [x] Remove import of `ModelProvider` from config
- [x] Keep: `get_async_tavily_client()`, all embedding functions, `get_markdown_text_from_path_or_text()`

### Step 3: Update Tool Files ✅

Each tool file needs import updates. Change from:
```python
from findingmodel.tools.common import get_model
# ...
model=get_model(model_tier)
```
To:
```python
from findingmodel.config import settings
# ...
model=settings.get_model(model_tier)
```

**Files updated**:
- [x] `src/findingmodel/tools/model_editor.py`
- [x] `src/findingmodel/tools/finding_description.py` (also remove `provider` param)
- [x] `src/findingmodel/tools/markdown_in.py`
- [x] `src/findingmodel/tools/anatomic_location_search.py`
- [x] `src/findingmodel/tools/ontology_concept_match.py`
- [x] `src/findingmodel/tools/similar_finding_models.py`

### Step 4: Special Handling for `finding_description.py` ✅

**File**: `src/findingmodel/tools/finding_description.py`

- [x] Remove `provider` parameter from `create_info_from_name()`
- [x] Remove `provider` parameter from `_create_finding_info_agent()`
- [x] Remove `ModelProvider` import
- [x] Update deprecated functions that reference old settings

### Step 5: Update Tests ✅

**Files**: `test/test_tools.py`, `test/test_anatomic_locations.py`, `test/test_ontology_search.py`, `test/test_model_editor.py`

- [x] Remove tests for old `get_model()` function (test `settings.get_model()` instead)
- [x] Remove tests for `get_openai_model()` deprecated function
- [x] Remove tests using `provider=` parameter
- [x] Update monkeypatches that use `provider=None`
- [x] Add new tests for `settings.get_model()` with different model strings
- [x] Add tests for API key validation with pattern matching
- [x] Update tests to use `patch.object(FindingModelConfig, "get_model", ...)` pattern

### Step 6: Update Configuration Files ✅

**File**: `.env.sample`

- [x] Replace old model config vars with new format
- [x] Add `PYDANTIC_AI_GATEWAY_API_KEY` (commented)
- [x] Remove `MODEL_PROVIDER`

### Step 7: Run Tests and Verify ✅

- [x] Run `task check` (format, lint, mypy) - All passed
- [x] Run `task test` (unit tests) - 421 passed
- [x] Run `task test-full` (integration tests with API keys) - 431 passed, 1 unrelated failure
- [x] Verify no regressions

### Step 8: Documentation ✅

- [x] Update CHANGELOG.md with breaking changes
- [x] Update README.md, CLAUDE.md, copilot-instructions.md, evals/README.md
- [x] Version bump to 0.7.0 (breaking change)

## New Environment Variable Format

```bash
# Old format (removed)
OPENAI_DEFAULT_MODEL=gpt-5-mini
ANTHROPIC_DEFAULT_MODEL=claude-sonnet-4-5
MODEL_PROVIDER=openai

# New format
DEFAULT_MODEL=openai:gpt-5-mini
DEFAULT_MODEL_FULL=openai:gpt-5
DEFAULT_MODEL_SMALL=openai:gpt-5-nano

# Alternative configurations:
# DEFAULT_MODEL=anthropic:claude-sonnet-4-5
# DEFAULT_MODEL=gateway/openai:gpt-5-mini
# DEFAULT_MODEL=gateway/anthropic:claude-sonnet-4-5
```

## API Key Requirements by Provider

| Model String Prefix | Required Environment Variable | Returned Model Class |
|---------------------|-------------------------------|---------------------|
| `openai:*` | `OPENAI_API_KEY` | `OpenAIResponsesModel` |
| `anthropic:*` | `ANTHROPIC_API_KEY` | `AnthropicModel` |
| `gateway/openai:*` | `PYDANTIC_AI_GATEWAY_API_KEY` | `OpenAIResponsesModel` via `gateway_provider` |
| `gateway/anthropic:*` | `PYDANTIC_AI_GATEWAY_API_KEY` | `AnthropicModel` via `gateway_provider` |
| Other | Raises `ConfigurationError` | N/A |

**Note**: OpenAI uses `OpenAIResponsesModel` (Responses API) instead of `OpenAIChatModel` (Chat Completions API) for better structured output support.

## Files Modified

- `src/findingmodel/config.py`
- `src/findingmodel/tools/common.py`
- `src/findingmodel/tools/model_editor.py`
- `src/findingmodel/tools/finding_description.py`
- `src/findingmodel/tools/markdown_in.py`
- `src/findingmodel/tools/anatomic_location_search.py`
- `src/findingmodel/tools/ontology_concept_match.py`
- `src/findingmodel/tools/similar_finding_models.py`
- `test/test_tools.py`
- `.env.sample`
- `CHANGELOG.md`
