# Agent Refactoring Plan

## Overview

This document outlines the migration plan for three major refactoring efforts:
1. Replace `instructor` library with Pydantic AI - ✅ **COMPLETE**
2. Replace Perplexity search API with Tavily - ✅ **COMPLETE**
3. Add Anthropic model support alongside OpenAI - ✅ **COMPLETE**

**Status:** All three plans are complete.

---

## Plan 1: Replace instructor with Pydantic AI ✅

**Status:** COMPLETE

**Objective:** Remove the `instructor` dependency and use Pydantic AI's native Agent pattern for structured outputs.

### Task 1.1: Refactor `create_model_from_markdown()` to use Pydantic AI Agent ✅

**Files:**
- `src/findingmodel/tools/markdown_in.py`

**Implementation:**
1. Replace lines 40-45 (instructor client usage) with Pydantic AI Agent pattern
2. Create Agent with:
   - `model=get_openai_model(openai_model)`
   - `output_type=FindingModelBase`
   - Instructions from rendered prompt template
3. Convert messages to single user prompt (extract user sections)
4. Call `await agent.run(user_prompt)` and return `result.output`
5. Keep existing error handling for non-FindingModelBase results

**Example Pattern:**
```python
# Replace this:
client = get_async_instructor_client()
result = await client.chat.completions.create(
    messages=messages,
    response_model=FindingModelBase,
    model=openai_model,
)

# With this:
instructions, user_prompt = _extract_prompt_sections(messages)
agent = Agent[None, FindingModelBase](
    model=get_openai_model(openai_model),
    output_type=FindingModelBase,
    instructions=instructions,
)
result = await agent.run(user_prompt)
return result.output
```

**Helper Function Needed:**
Add `_extract_prompt_sections(messages)` to extract system instructions and user prompt from message list (similar to `finding_description.py:18-34`).

**Testing:**
- Unit test with `TestModel` override (add to `test/test_tools.py`)
- Integration test with `@pytest.mark.callout` for live API
- Verify backward compatibility with existing callers

**Acceptance Criteria:**
- `create_model_from_markdown()` returns same structure as before
- All existing tests pass
- No instructor imports remain in file

---

### Task 1.1.1: Refactor finding_description.py to use render_agent_prompt() ✅

**Files:**
- `src/findingmodel/tools/finding_description.py`

**Context:**
During Task 1.1, we added `render_agent_prompt()` to `prompt_template.py` to eliminate the OpenAI message-based pattern. The `finding_description.py` file still uses `create_prompt_messages()` and has its own `_render_finding_description_prompt()` helper that does the same extraction pattern.

**Implementation:**
1. **Update imports** (line 13):
   - Remove: `from .prompt_template import create_prompt_messages, load_prompt_template`
   - Add: `from .prompt_template import load_prompt_template, render_agent_prompt`

2. **Replace `_render_finding_description_prompt()` function** (lines 18-34):
   - Remove the entire function
   - It's no longer needed since `render_agent_prompt()` does this work

3. **Update `create_info_from_name()` function** (lines 37-55):
   - Replace lines 45-46:
     ```python
     # Old:
     instructions, user_prompt = _render_finding_description_prompt(finding_name)

     # New:
     template = load_prompt_template(PROMPT_TEMPLATE_NAME)
     instructions, user_prompt = render_agent_prompt(template, finding_name=finding_name)
     ```

**Testing:**
- Run `task test` - all unit tests should pass
- Run `task evals:finding_description` - eval scores should not regress

**Acceptance Criteria:**
- `_render_finding_description_prompt()` helper removed
- `create_info_from_name()` uses `render_agent_prompt()` directly
- All existing tests pass
- Eval scores show no regression

---

### Task 1.2: Remove instructor dependency from common.py ✅

**Files:**
- `src/findingmodel/tools/common.py`

**Implementation:**
1. Remove line 5: `from instructor import AsyncInstructor, from_openai`
2. Remove function `get_async_instructor_client()` (lines 30-32)
3. Verify no other files import this function (should only be `markdown_in.py`)

**Testing:**
- Run `task check` to verify no import errors
- Run `task test` to verify all tests pass

**Acceptance Criteria:**
- No instructor imports in `common.py`
- `get_async_instructor_client()` function removed
- All tests pass

---

### Task 1.3: Remove instructor from dependencies ✅

**Files:**
- `pyproject.toml`

**Implementation:**
1. Remove `instructor` from dependencies list
2. Run `uv sync` to update lockfile
3. Verify no other code depends on instructor with: `grep -r "instructor" src/ test/`

**Testing:**
- `uv sync` completes without errors
- `task test-full` passes
- No grep results for "instructor" in src/ or test/ (except comments/docstrings if any)

**Acceptance Criteria:**
- `instructor` removed from `pyproject.toml`
- Lockfile updated
- All tests pass

---

## Plan 2: Replace Perplexity with Tavily ✅

**Status:** COMPLETE

**Objective:** Replace Perplexity API with Tavily for enhanced finding details and citations.

### Task 2.1: Add Tavily configuration and dependencies ✅

**Files:**
- `pyproject.toml`
- `src/findingmodel/config.py`
- `.env.sample`

**Implementation:**

1. **Add dependency** to `pyproject.toml`:
   ```toml
   dependencies = [
       # ... existing deps
       "tavily-python>=0.6.0",
   ]
   ```

2. **Update `config.py`** - Replace Perplexity config (lines 42-45):
   ```python
   # Remove these lines:
   perplexity_base_url: HttpUrl = Field(default=HttpUrl("https://api.perplexity.ai"))
   perplexity_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
   perplexity_default_model: str = Field(default="sonar-pro")

   # Add these:
   tavily_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
   tavily_search_depth: str = Field(
       default="advanced",
       description="Tavily search depth: 'basic' or 'advanced'"
   )
   ```

3. **Replace validation method** (line 130-132):
   ```python
   # Remove check_ready_for_perplexity
   # Add:
   def check_ready_for_tavily(self) -> Literal[True]:
       if not self.tavily_api_key.get_secret_value():
           raise ConfigurationError("Tavily API key is not set")
       return True
   ```

4. **Update `.env.sample`**:
   ```bash
   # Remove: PERPLEXITY_API_KEY, PERPLEXITY_BASE_URL, PERPLEXITY_DEFAULT_MODEL
   # Add:
   TAVILY_API_KEY=your-tavily-api-key-here
   TAVILY_SEARCH_DEPTH=advanced
   ```

**Testing:**
- Settings load without errors
- `check_ready_for_tavily()` raises ConfigurationError when key missing
- `check_ready_for_tavily()` returns True when key present

**Acceptance Criteria:**
- `tavily-python` added to dependencies
- Tavily config replaces Perplexity in `config.py`
- `.env.sample` updated
- Settings validation tests pass

---

### Task 2.2: Create Tavily client helper in common.py ✅

**Files:**
- `src/findingmodel/tools/common.py`

**Implementation:**

1. **Add import** at top of file:
   ```python
   from tavily import AsyncTavilyClient
   ```

2. **Remove** `get_async_perplexity_client()` function (lines 36-39)

3. **Add new helper**:
   ```python
   def get_async_tavily_client() -> AsyncTavilyClient:
       """Get configured async Tavily search client."""
       settings.check_ready_for_tavily()
       return AsyncTavilyClient(api_key=settings.tavily_api_key.get_secret_value())
   ```

**Testing:**
- Mock settings and verify client creation
- Verify ConfigurationError raised when API key missing
- Verify client returned when API key present

**Acceptance Criteria:**
- `get_async_perplexity_client()` removed
- `get_async_tavily_client()` added and tested
- No Perplexity imports remain

---

### Task 2.3: Refactor `add_details_to_info()` to use Tavily ✅

**Files:**
- `src/findingmodel/tools/finding_description.py`

**Implementation:**

1. **Update imports** (line 12):
   ```python
   # Remove: from .common import get_async_perplexity_client
   # Add: from .common import get_async_tavily_client
   ```

2. **Update function signature** (lines 101-103):
   ```python
   async def add_details_to_info(
       finding: FindingInfo,
       search_depth: str = settings.tavily_search_depth  # Changed from model_name
   ) -> FindingInfo | None:
   ```

3. **Replace implementation** (lines 110-134):
   ```python
   # Build search query from finding
   query = f"{finding.name} medical imaging radiology"
   if finding.synonyms:
       query += f" ({', '.join(finding.synonyms[:3])})"

   # Tavily async search
   client = get_async_tavily_client()
   response = await client.search(
       query=query,
       search_depth=search_depth,
       max_results=5,
       include_answer=True,
   )

   if not response or 'answer' not in response:
       return None

   # Build FindingInfo with Tavily results
   detail = response['answer']
   citations = [result['url'] for result in response.get('results', [])]

   return FindingInfo(
       name=finding.name,
       synonyms=finding.synonyms,
       description=finding.description,
       detail=detail,
       citations=citations if citations else None,
   )
   ```

4. **Update deprecated function signatures** (lines 150-151, 179-180) to match new signature

**Testing:**
- Unit test with mocked `AsyncTavilyClient`
- Mock return structure: `{'answer': 'text', 'results': [{'url': 'http://...', 'content': '...'}]}`
- Integration test with `@pytest.mark.callout` and real async API
- Verify citations are properly extracted
- Verify async/await pattern works correctly

**Acceptance Criteria:**
- Function uses Tavily instead of Perplexity
- Returns FindingInfo with detail and citations
- Handles empty/missing responses gracefully
- Tests pass with mocked and real API

---

### Task 2.4: Update tests for Tavily migration ✅

**Files:**
- `test/test_tools.py`
- `evals/finding_description.py`

**Implementation:**

1. **Update `test/test_tools.py`**:
   - Remove `HAS_PERPLEXITY_API_KEY` check (line 17)
   - Add `HAS_TAVILY_API_KEY = bool(settings.tavily_api_key.get_secret_value())`
   - Update integration test for `add_details_to_info()` (around line 290)
   - Replace Perplexity API key check with Tavily check
   - Update mock structure to match Tavily response format

2. **Update `evals/finding_description.py`**:
   - Remove `use_perplexity` field from `FindingDescriptionInput` (line 78)
   - Remove `use_perplexity` parameter from functions (lines 130, 150, 167)
   - Remove conditional Perplexity logic (line 231)
   - Remove `create_perplexity_detail_cases()` function (lines 771-795)
   - Update eval cases to always use Tavily for details (no flag needed)
   - Remove Perplexity-specific checks in evaluators (line 937)

**Testing:**
- All unit tests pass: `task test`
- All integration tests pass: `task test-full`
- Eval suite runs: `task evals:finding_description`
- No references to "perplexity" remain in test files (case-insensitive grep)

**Acceptance Criteria:**
- Tests updated for Tavily
- Eval suite updated
- All tests pass
- No Perplexity references in test code

---

### Task 2.5: Update documentation and memories ✅

**Status:** COMPLETE

**Files:**
- Serena memory: `api_integration`
- `.env.sample` (already updated in Task 2.1)
- `README.md` (if it mentions Perplexity)

**Implementation:**

1. **Update `api_integration` memory**:
   - Replace Perplexity section with Tavily section
   - Document Tavily API key requirement
   - Document search depth options ("basic" vs "advanced")
   - Update example of how detailed findings work
   - Keep free tier info: 1,000 searches/month

2. **Check and update README.md**:
   - Search for any Perplexity mentions
   - Replace with Tavily references
   - Update API key setup instructions

**Memory Content Example:**
```markdown
### Tavily API
- **Environment Variable**: `TAVILY_API_KEY`
- **Search Depth**: `advanced` (configurable via `TAVILY_SEARCH_DEPTH`)
- **Used For**:
  - `add_details_to_info()` - Enhanced descriptions with citations
  - Research-grade medical information with source URLs
- **Free Tier**: 1,000 searches/month
```

**Testing:**
- Review memory content for accuracy
- Verify no Perplexity references remain in documentation

**Acceptance Criteria:**
- `api_integration` memory updated
- README.md updated (if needed)
- Documentation is accurate and complete

---

## Plan 3: Add Anthropic Model Support

**Objective:** Enable Anthropic Claude models as an alternative to OpenAI across all Pydantic AI agents.

**Current State:** Only OpenAI models supported via `get_openai_model()` helper.

**Dependencies:** Complete Plan 1 first (uses Pydantic AI infrastructure).

### Task 3.1: Add Anthropic configuration and dependencies

**Files:**
- `pyproject.toml`
- `src/findingmodel/config.py`
- `.env.sample`

**Implementation:**

1. **Update `pyproject.toml`**:
   ```toml
   # Option A: Upgrade to full pydantic-ai (includes anthropic)
   dependencies = [
       "pydantic-ai>=0.0.14",  # Replace pydantic-ai-slim
   ]

   # Option B: Keep slim and add anthropic optional
   dependencies = [
       "pydantic-ai-slim[anthropic]>=0.0.14",
   ]
   ```
   Choose Option A (simpler) unless bundle size is critical.

2. **Add Anthropic config to `config.py`** after OpenAI settings (after line 40):
   ```python
   # Anthropic API (optional alternative to OpenAI)
   anthropic_api_key: QuoteStrippedSecretStr = Field(default=SecretStr(""))
   anthropic_default_model: str = Field(default="claude-sonnet-4-5")
   anthropic_default_model_full: str = Field(default="claude-opus-4-1")
   anthropic_default_model_small: str = Field(default="claude-haiku-4-5")

   # Model provider selection
   model_provider: str = Field(
       default="openai",
       description="AI model provider: 'openai' or 'anthropic'"
   )
   ```

3. **Add validation method** after `check_ready_for_openai()` (after line 128):
   ```python
   def check_ready_for_anthropic(self) -> Literal[True]:
       if not self.anthropic_api_key.get_secret_value():
           raise ConfigurationError("Anthropic API key is not set")
       return True
   ```

4. **Update `.env.sample`**:
   ```bash
   # Add after OpenAI section:
   # Anthropic API (optional alternative to OpenAI)
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   ANTHROPIC_DEFAULT_MODEL=claude-sonnet-4-5
   ANTHROPIC_DEFAULT_MODEL_FULL=claude-opus-4-1
   ANTHROPIC_DEFAULT_MODEL_SMALL=claude-haiku-4-5
   MODEL_PROVIDER=openai
   ```

**Testing:**
- Settings load with new fields
- Default provider is "openai" (backward compatible)
- `check_ready_for_anthropic()` works correctly
- Invalid provider value raises validation error (add test)

**Acceptance Criteria:**
- Anthropic dependency added
- Configuration fields added
- Validation method added
- `.env.sample` updated
- Backward compatible (defaults to OpenAI)

---

### Task 3.2: Create model provider factory in common.py

**Files:**
- `src/findingmodel/tools/common.py`

**Implementation:**

1. **Add imports** at top:
   ```python
   from pydantic_ai.models.openai import OpenAIModel
   from pydantic_ai.models.anthropic import AnthropicModel
   from pydantic_ai import Model
   ```

2. **Add type aliases and model factory function**:
   ```python
   ModelProvider = Literal["openai", "anthropic"]
   ModelTier = Literal["base", "small", "full"]

   def get_model(
       model_tier: ModelTier = "base",
       *,
       provider: ModelProvider | None = None,
   ) -> Model:
       """Get AI model from configured provider.

       Args:
           model_tier: Model capability tier - "small" (fast/cheap), "base" (default),
                      or "full" (most capable)
           provider: "openai" or "anthropic" (defaults to settings.model_provider)

       Returns:
           Configured Model instance

       Raises:
           ConfigurationError: If provider unsupported or API key missing
       """
       provider = provider or settings.model_provider

       if provider == "anthropic":
           settings.check_ready_for_anthropic()
           tier_map = {
               "base": settings.anthropic_default_model,
               "full": settings.anthropic_default_model_full,
               "small": settings.anthropic_default_model_small,
           }
           model_name = tier_map[model_tier]
           anthropic_provider = AnthropicProvider(api_key=settings.anthropic_api_key.get_secret_value())
           return AnthropicModel(model_name, provider=anthropic_provider)

       elif provider == "openai":
           settings.check_ready_for_openai()
           tier_map = {
               "base": settings.openai_default_model,
               "full": settings.openai_default_model_full,
               "small": settings.openai_default_model_small,
           }
           model_name = tier_map[model_tier]
           openai_provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
           return OpenAIModel(model_name, provider=openai_provider)

       else:
           raise ConfigurationError(f"Unsupported model provider: {provider}")
   ```

   **Note:** Uses "base" (not "default") for clarity. No `model_name` parameter - tier-based selection enforces provider portability.

3. **Deprecate `get_openai_model()`** - keep for backward compatibility but issue warning:
   ```python
   def get_openai_model(model_name: str = settings.openai_default_model) -> Model:
       """Get OpenAI model (deprecated - use get_model() instead)."""
       import warnings
       warnings.warn(
           "get_openai_model() is deprecated, use get_model() instead",
           DeprecationWarning,
           stacklevel=2
       )
       # Implementation kept unchanged for compatibility
       openai_provider = OpenAIProvider(api_key=settings.openai_api_key.get_secret_value())
       return OpenAIModel(model_name, provider=openai_provider)
   ```

**Testing:**
- Test `get_model()` with both providers
- Test model tier selection ("base", "small", "full")
- Test error handling for missing API keys
- Test deprecation warning on `get_openai_model()`

**Acceptance Criteria:**
- `get_model()` function created
- Supports both OpenAI and Anthropic
- Model tier selection works
- `get_openai_model()` deprecated but functional
- Comprehensive unit tests added

---

### Task 3.3: Update tool functions to use new model factory

**Files:**
- `src/findingmodel/tools/finding_description.py`
- `src/findingmodel/tools/anatomic_location_search.py`
- `src/findingmodel/tools/ontology_concept_match.py`
- `src/findingmodel/tools/model_editor.py`

**Implementation:**

For each file, update tool function signatures to accept `model_tier` and `provider` parameters:

1. **`finding_description.py`**:
   ```python
   async def create_info_from_name(
       finding_name: str,
       model_tier: ModelTier = "small",
       provider: ModelProvider | None = None,
   ) -> FindingInfo:
   ```
   - All internal agent creation uses `get_model(model_tier, provider=provider)`

2. **`anatomic_location_search.py`**:
   - Update `find_anatomic_locations()` to accept `model_tier` and `provider` parameters
   - Replace `get_openai_model()` calls with `get_model()`

3. **`ontology_concept_match.py`**:
   - Update public functions to accept `model_tier` and `provider` parameters
   - Replace `get_openai_model()` calls with `get_model()`

4. **`model_editor.py`**:
   - Update editing functions to accept `model_tier` and `provider` parameters
   - Replace `get_openai_model()` calls with `get_model()`

**Testing:**
- All tools work with OpenAI (default)
- All tools work with Anthropic (when configured)
- Integration tests with both providers
- No regression in existing functionality

**Acceptance Criteria:**
- All tool files updated to use `get_model()`
- No direct `get_openai_model()` calls in tool code
- All tests pass with both providers
- Backward compatible (defaults to OpenAI)

---

### Task 3.4: Add integration tests for Anthropic models

**Files:**
- `test/test_tools.py`
- New file: `test/test_anthropic_integration.py` (optional)

**Implementation:**

1. **Add Anthropic test helper** in `test/test_tools.py`:
   ```python
   HAS_ANTHROPIC_API_KEY = bool(settings.anthropic_api_key.get_secret_value())

   @pytest.mark.callout
   @pytest.mark.skipif(not HAS_ANTHROPIC_API_KEY, reason="Anthropic API key not set")
   async def test_create_info_from_name_anthropic():
       """Test finding info creation with Anthropic model."""
       result = await create_info_from_name(
           "pneumothorax",
           provider="anthropic",
       )
       assert isinstance(result, FindingInfo)
       assert result.name
       assert result.description
   ```

2. **Add provider switching test**:
   ```python
   @pytest.mark.callout
   async def test_model_provider_switching():
       """Test switching between OpenAI and Anthropic."""
       # Requires both API keys set
       if not (HAS_OPENAI_API_KEY and HAS_ANTHROPIC_API_KEY):
           pytest.skip("Both API keys required")

       # Test OpenAI
       result_openai = await create_info_from_name("pneumonia", provider="openai")
       assert isinstance(result_openai, FindingInfo)

       # Test Anthropic
       result_anthropic = await create_info_from_name("pneumonia", provider="anthropic")
       assert isinstance(result_anthropic, FindingInfo)
   ```

3. **Add model tier selection test**:
   ```python
   def test_model_tier_selection():
       """Test that model tiers select correct models."""
       from findingmodel.tools.common import get_model

       # OpenAI tiers
       model = get_model("small", provider="openai")
       assert isinstance(model, OpenAIModel)

       # Anthropic tiers
       if HAS_ANTHROPIC_API_KEY:
           model = get_model("small", provider="anthropic")
           assert isinstance(model, AnthropicModel)
   ```

**Testing:**
- Run tests with OpenAI only (should pass)
- Run tests with Anthropic only (should pass)
- Run tests with both (should pass all combinations)

**Acceptance Criteria:**
- Integration tests added for Anthropic
- Provider switching tested
- All tests pass when respective API keys available
- Tests skip gracefully when keys missing

---

### Task 3.5: Update documentation and memories

**Files:**
- Serena memories: `api_integration`, `pydantic_ai_best_practices_2025_09`
- `CLAUDE.md`
- `.github/copilot-instructions.md`

**Implementation:**

1. **Update `api_integration` memory**:
   - Add Anthropic API section after OpenAI section
   - Document model provider selection
   - Document model tier mappings
   - Add example usage with provider parameter

2. **Update `pydantic_ai_best_practices_2025_09` memory**:
   - Note multi-provider support
   - Document `get_model()` usage patterns
   - Add examples with both providers

3. **Update `CLAUDE.md`**:
   - Add Anthropic to tech stack description (section 1)
   - Update "API Integration" notes with provider selection
   - Add Anthropic to Quick Reference section

4. **Update `.github/copilot-instructions.md`**:
   - Add note about model provider options
   - Keep brief (detailed info in memories)

**Memory Content Example for `api_integration`:**
```markdown
### Anthropic API
- **Environment Variable**: `ANTHROPIC_API_KEY`
- **Default Models**:
  - Main: `claude-sonnet-4-5`
  - Full: `claude-opus-4-1`
  - Small: `claude-haiku-4-5`
- **Provider Selection**: Set `MODEL_PROVIDER=anthropic` in `.env`
- **Used For**: Alternative to OpenAI for all AI-powered tools
- **Usage**: `get_model(provider="anthropic")` or set as default provider
```

**Testing:**
- Review all updated documentation for accuracy
- Ensure consistency across all files
- Verify no conflicting information

**Acceptance Criteria:**
- All memories updated
- CLAUDE.md updated
- Copilot instructions updated
- Documentation is accurate and consistent

---

## Testing Checklist (All Plans)

After completing all tasks:

- [ ] `task check` passes (formatting, linting, type checking)
- [ ] `task test` passes (unit tests, no API calls)
- [ ] `task test-full` passes (integration tests with APIs)
- [ ] `task evals` passes (eval suites show no regression)
- [ ] All deprecated functions have warnings
- [ ] No orphaned imports or dead code
- [ ] Configuration backward compatible
- [ ] Documentation complete and accurate

---

## Notes for Sub-Agents

**Before Starting:**
- Read relevant Serena memories: `project_overview`, `code_style_conventions`, `pydantic_ai_best_practices_2025_09`
- Check task dependencies - some tasks must be completed in order
- Run `task check` before committing any changes

**During Implementation:**
- Follow existing code patterns in the file you're editing
- Keep functions pure where possible (no side effects in validators)
- Add proper type hints to all new code
- Use async for I/O operations, remove if no awaits needed

**Testing:**
- Write unit tests with mocked dependencies first
- Use `TestModel` for Pydantic AI agent testing
- Add integration tests with `@pytest.mark.callout`
- Run full test suite before marking task complete

**Documentation:**
- Update docstrings for modified functions
- Update Serena memories for architectural changes
- Keep CLAUDE.md and copilot instructions in sync
- Add migration notes to memories if helpful for future reference
