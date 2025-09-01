# Implementation Plan for Anatomic Location Search Tool

## Current Status
**Last Updated:** 2025-08-29
- ✅ Phase 1: Completed - Common utilities refactored
- ✅ Phase 2: Completed - Ontology search module created
- ✅ Phase 3: Completed - LanceDB configuration added
- ✅ Phase 4: Completed - Anatomic location search implemented
- 🔄 Phase 5: In Progress - Testing
- ✅ Phase 6: Partially Complete - Exports done, documentation pending

## Phase 1: Refactor Common Utilities ✅ COMPLETED
**Step 1.1: Move `get_openai_model()` to common.py**
- ✅ Extract `_get_openai_model()` from `similar_finding_models.py`
- ✅ Add it to `common.py` as `get_openai_model()`
- ✅ Update `similar_finding_models.py` to import from common

## Phase 2: Create Ontology Search Module ✅ COMPLETED
**Step 2.1: Create ontology_search.py file**
- ✅ Create `src/findingmodel/tools/ontology_search.py` file directly (no sub-module)

**Step 2.2: Implement OntologySearchResult model**
- ✅ Define `OntologySearchResult` Pydantic model
- ✅ Fields: concept_id, concept_text, score, table_name
- ✅ Add `as_index_code()` method that returns IndexCode instance
- ✅ Use TABLE_TO_INDEX_CODE_SYSTEM mapping for system field
- ✅ Added regex pattern to strip parenthetical info from concept_text

**Step 2.3: Implement OntologySearchClient**
- ✅ Define `ONTOLOGY_TABLES` constant with ["anatomic_locations", "radlex", "snomedct"]
- ✅ Define `TABLE_TO_INDEX_CODE_SYSTEM` mapping
- ✅ Implement `__init__()` with LanceDB URI and API key handling (no side effects)
- ✅ Add `connected` property to check connection status
- ✅ Implement `connect()` method:
  - ✅ Use `lancedb.connect_async()` with URI and API key
  - ✅ Cache table references like in search_engine.py
  - ✅ Handle connection errors gracefully
- ✅ Implement `disconnect()` method (synchronous, not async):
  - ✅ Check if connected before closing
  - ✅ Clear cached table references
- ✅ Implement `search_tables()` method:
  - ✅ Use `query_type="hybrid"` for search
  - ✅ Search each specified table (or all if none specified)
  - ✅ Collect results with scores
  - ✅ Handle missing tables gracefully
  - ✅ Return dict mapping table names to lists of OntologySearchResult
- ✅ Use pattern from `scripts/search_engine.py` but production-ready

## Phase 3: Add LanceDB Configuration ✅ COMPLETED
**Step 3.1: Update config.py**
- ✅ Add `lancedb_uri: str | None` field
- ✅ Add `lancedb_api_key: SecretStr | None` field
- ✅ Ensure proper defaults and .env loading
- ✅ Added OpenAI API key environment setup for LanceDB hybrid search

## Phase 4: Implement Anatomic Location Search ✅ COMPLETED
**Step 4.1: Create anatomic_location_search.py**
- ✅ Import dependencies (get_openai_model, ontology_search components)
- ✅ Define `RawSearchResults` model for search agent output with search_terms_used and search_results fields
- ✅ Define `LocationSearchResponse` model for matching agent output with primary_location and alternate_locations as OntologySearchResult objects
- ✅ Define `SearchContext` dataclass for dependency injection with OntologySearchClient

**Step 4.2: Implement search tool**
- ✅ Create `ontology_search_tool()` async function
- ✅ Takes SearchContext, query, and limit
- ✅ Returns dict of search results (not JSON string - corrected for Pydantic AI)
- ✅ **Decision:** Limited to "anatomic_locations" table only (by design)

**Step 4.3: Create search agent**
- ✅ Implement `create_search_agent()` function
- ✅ Uses small model by default
- ✅ Has access to ontology_search_tool
- ✅ Returns RawSearchResults with terms used and results found

**Step 4.4: Create matching agent**
- ✅ Implement `create_matching_agent()` function
- ✅ Uses main model by default
- ✅ No tools needed (just analyzes provided data)
- ✅ Returns LocationSearchResponse

**Step 4.5: Implement main API function**
- ✅ Create `find_anatomic_locations()` async function
- ✅ Parameters: finding_name, description, search_model, matching_model
- ✅ Two-step workflow:
  1. ✅ Search agent generates queries and collects results
  2. ✅ Matching agent picks best primary and alternate locations
- ✅ Proper error handling and client cleanup
- ✅ Uses loguru logging pattern (not standard logging)

## Phase 5: Testing 🔄 IN PROGRESS - MAJOR ISSUES IDENTIFIED

### Critical Issues Found (2025-08-29):

1. **❌ Script Collection Problem**: `scripts/test_anatomic_search.py` is being collected by pytest during `task test` even though it contains integration tests that make real API calls. This defeats the purpose of excluding callout tests.
   - **Solution**: Move to `notebooks/demo_anatomic_location_search.py` following project convention for demo/proving scripts

2. **❌ Testing Philosophy Wrong**: Current tests are testing library behavior (Pydantic AI, Pydantic models) rather than our code's logic
   - Tests verify that agents have a `run` method (testing Pydantic AI)
   - Tests verify model validation (testing Pydantic)
   - Over-mocked workflow tests that don't test actual behavior

3. **❌ Not Following Pydantic AI Best Practices**: Should use `TestModel` and `FunctionModel` instead of mocking everything
   - Missing `models.ALLOW_MODEL_REQUESTS = False` to prevent accidental API calls
   - Not using Pydantic AI's testing utilities

**Step 5.1: Unit tests status**
- ✅ Created `test/test_ontology_search.py` for ontology search module (adequate)
- ✅ Created `test/test_anatomic_location_search.py` BUT needs major refactoring:
  - ❌ `TestAgentCreation` tests library not our code
  - ❌ `TestModels` tests Pydantic validation not business logic  
  - ❌ `TestFindAnatomicLocations` over-mocks everything
  - ✅ `TestOntologySearchTool` has decent structure
  - ✅ Integration test properly marked with `@pytest.mark.callout`

**Step 5.2: Manual testing script**
- ✅ Created `scripts/test_anatomic_search.py` BUT:
  - ❌ Name causes pytest to collect it as a test file
  - ❌ Only runs 1 of 6 defined test functions in main()
  - ❌ Wrong location - should be in `notebooks/` per project convention

## Phase 6: Documentation and Export ✅ PARTIALLY COMPLETE
**Step 6.1: Update tools __init__.py**
- ✅ Add import for `find_anatomic_locations`
- ✅ Export in `__all__` list

**Step 6.2: Add docstrings**
- ✅ Basic docstrings added to all functions
- ⏳ Could enhance with more detailed usage examples
- ⏳ Consider adding module-level docstring examples

## Implementation Completed:
1. ✅ **Phase 1** - Quick refactor to share get_openai_model()
2. ✅ **Phase 3** - Add config (needed by Phase 2)
3. ✅ **Phase 2** - Build reusable ontology search module
4. ✅ **Phase 4** - Implement main functionality
5. 🔄 **Phase 5** - Add tests (unit tests for ontology_search done, need anatomic_location_search tests)
6. ✅ **Phase 6** - Polish and export (exports done, could enhance docs)

## Files Modified:
- ✅ `src/findingmodel/tools/common.py` (added get_openai_model function)
- ✅ `src/findingmodel/tools/similar_finding_models.py` (updated to use common import)
- ✅ `src/findingmodel/config.py` (added LanceDB settings and OpenAI env setup)
- ✅ `src/findingmodel/tools/__init__.py` (added find_anatomic_locations export)

## Files Created:
- ✅ `src/findingmodel/tools/ontology_search.py`
- ✅ `src/findingmodel/tools/anatomic_location_search.py`
- ✅ `test/test_ontology_search.py` (should be consolidated into test_anatomic_location_search.py)
- ✅ `test/test_anatomic_location_search.py` (created but needs refactoring)
- ✅ `scripts/test_anatomic_search.py` (needs moving to notebooks/demo_anatomic_location_search.py)

## Implementation Decisions and Notes:

### Completed Decisions:
- ✅ **Search Scope**: Limited to "anatomic_locations" table only (intentional design decision)
- ✅ **OpenAI API Key**: Handled in config.py after settings instantiation for LanceDB hybrid search
- ✅ **Logging**: Using loguru (project standard) not standard Python logging
- ✅ **Tool Return Types**: Tools return dicts, not JSON strings, for Pydantic AI agents
- ✅ **IndexCode Integration**: `as_index_code()` strips parenthetical info from concept_text
- ✅ **Error Handling**: Connection errors logged and re-raised with context
- ✅ **Connection Management**: Create new connection per request (MVP approach)

### Required Fixes (Priority Order):

#### 1. **URGENT: Fix Script Collection Issue**
   - Move `scripts/test_anatomic_search.py` → `notebooks/demo_anatomic_location_search.py`
   - Update demo script to run all test scenarios or accept CLI args
   - Consider adding `testpaths = ["test"]` to pytest config to exclude scripts/
   - Follow project convention: demo/proving scripts go in `notebooks/` with `demo_*.py` naming

#### 2. **Consolidate Test Files**
   - Merge `test/test_ontology_search.py` into `test/test_anatomic_location_search.py`
   - OntologySearch is a supporting component, not a standalone tool
   - Reduces test file fragmentation
   - Keep all related tests together

#### 3. **Refactor Tests to Follow Pydantic AI Best Practices**
   - Add `from pydantic_ai import models; models.ALLOW_MODEL_REQUESTS = False` to test files
   - Use `TestModel` for deterministic agent testing
   - Use `FunctionModel` for controlled behavior testing
   - Remove complete mocking of agents

#### 4. **Rewrite Tests to Test Our Code, Not Libraries**
   
   **TestAgentCreation** → **TestAgentConfiguration**:
   ```python
   from pydantic_ai.models.test import TestModel
   
   async def test_search_agent_configuration():
       agent = create_search_agent("gpt-4")
       with agent.override(model=TestModel()):
           # Test that agent has correct tools, prompts, output type
           assert ontology_search_tool in agent.tools
           assert "medical terminology" in agent.system_prompt
   ```

   **TestModels** → Remove or minimize (Pydantic handles validation)

   **TestFindAnatomicLocations** → Use `FunctionModel`:
   ```python
   from pydantic_ai.models.function import FunctionModel
   
   def controlled_search_behavior(messages, info):
       # Return controlled responses to test workflow
       ...
   
   with search_agent.override(model=FunctionModel(controlled_search_behavior)):
       # Test actual workflow logic
   ```

#### 5. **Add Missing Test Coverage**
   - Context passing between agents
   - Error propagation through two-agent pipeline
   - Tool transformation logic (not just mocked calls)
   - Agent prompt effectiveness

### Testing Strategy Documentation to Add to CLAUDE.md:

```markdown
## Testing Pydantic AI Agents

Following Pydantic AI best practices:
- Use `TestModel` for simple deterministic tests
- Use `FunctionModel` for complex controlled behavior
- Set `models.ALLOW_MODEL_REQUESTS = False` to prevent real API calls
- Test behavior, not implementation details

See test/test_utils/pydantic_ai_helpers.py for utilities.
```

### Remaining Work Summary:
1. **Fix immediate issues** (blocking correct test execution)
   - Move demo script to `notebooks/demo_anatomic_location_search.py`
   - Consolidate `test_ontology_search.py` into `test_anatomic_location_search.py`
   - Add models.ALLOW_MODEL_REQUESTS = False
   - Configure pytest to only look in test/ directory

2. **Refactor existing tests** (improve test quality)
   - Adopt Pydantic AI testing patterns (TestModel, FunctionModel)
   - Focus on behavior not implementation
   - Reduce mocking, increase actual testing

3. **Future Enhancements** (not part of MVP):
   - Connection pooling for better performance
   - Retry logic for transient failures
   - Support for additional ontology tables beyond anatomic_locations
   - Caching of frequent searches
   - Property-based testing for models

## Next Steps:
1. **IMMEDIATE**: Move `scripts/test_anatomic_search.py` → `notebooks/demo_anatomic_location_search.py`
2. **IMMEDIATE**: Consolidate test files - merge `test_ontology_search.py` into `test_anatomic_location_search.py`
3. **HIGH PRIORITY**: Add `models.ALLOW_MODEL_REQUESTS = False` to test files
4. **MEDIUM PRIORITY**: Refactor tests to use Pydantic AI testing utilities
5. **LOW PRIORITY**: Add comprehensive behavior tests

## Project Conventions Established:
- **Demo/Proving Scripts**: Place in `notebooks/` with naming pattern `demo_<function_name>.py`
- **Test Consolidation**: Keep related component tests together (ontology_search is part of anatomic_location_search)
- **Test Philosophy**: Test behavior, not libraries - use Pydantic AI's TestModel/FunctionModel