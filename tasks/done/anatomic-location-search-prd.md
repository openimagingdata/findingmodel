# PRD: Anatomic Location Search Tool for FindingModel

## Overview
Create a straightforward Pydantic AI agent tool that helps associate anatomic locations with FindingModel objects by searching across multiple ontology terminology databases in LanceDB.

## Core Architecture

### 1. Modular File Structure
```
src/findingmodel/tools/
├── ontology_search.py
└── anatomic_location_search.py # Main anatomic location search tool
```

### 2. Core Components

#### A. Reusable ontology Search Module (src/findingmodel/tools/ontology_search.py)

```python
# src/findingmodel/tools/ontology_search.py
from typing import Any

import lancedb
from pydantic import BaseModel

from findingmodel import IndexCode
from findingmodel.config import settings

# Table constants
ONTOLOGY_TABLES = ["anatomic_locations", "radlex", "snomedct"]
TABLE_TO_INDEX_CODE_SYSTEM = {
    "anatomic_locations": "ANATOMICLOCATIONS",
    "radlex": "RADLEX",
    "snomedct": "SNOMEDCT"
}


class OntologySearchResult(BaseModel):
    """Search result from LanceDB"""
    concept_id: str
    concept_text: str
    score: float
    table_name: str

    def as_index_code(self) -> IndexCode:
        return IndexCode(
            system = TABLE_TO_INDEX_CODE_SYSTEM.get(self.table_name, self.table_name),
            code = self.concept_id,
            display = self.concept_text
        )

class OntologySearchClient:
    """Production-ready LanceDB client for medical terminology search"""
    
    def __init__(self, lancedb_uri: str | None = None, api_key: str | None = None):
        self._db_conn: lancedb.AsyncConnection | None = None
        self._tables: dict[str, lancedb.AsyncTable] = {}
        self._uri = lancedb_uri or settings.lancedb_uri
        self._api_key = api_key or settings.lancedb_api_key
    
    @property
    def connected(self) -> bool:
        """Check if connected to LanceDB."""
        return self._db_conn is not None
        
    async def connect(self) -> None:
        """Connect to LanceDB and cache table references."""
        # Implementation similar to scripts/search_engine.py
        
    def disconnect(self) -> None:
        """Disconnect from LanceDB (synchronous)."""
        # Note: LanceDB close() is synchronous
        
    async def search_tables(
        self,
        query: str,
        tables: list[str] | None = None,
        limit_per_table: int = 10
    ) -> dict[str, list[OntologySearchResult]]:
        """Search across specified tables using hybrid search."""
        # Implementation will use query_type="hybrid" like search_engine.py
```

#### B. Updated Common Module
```python
# src/findingmodel/tools/common.py (add this function)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

def get_openai_model(model_name: str) -> OpenAIModel:
    """Helper function to get OpenAI model instance - moved from similar_finding_models.py"""
    return OpenAIModel(
        model_name=model_name,
        provider=OpenAIProvider(api_key=settings.openai_api_key.get_secret_value()),
    )
```

#### C. Two-Agent Architecture (anatomic_location_search.py)
```python
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from findingmodel.tools.common import get_openai_model
from findingmodel.tools.ontology_search import OntologySearchResult, OntologySearchClient

class RawSearchResults(BaseModel):
    """Output from search agent"""
    search_terms_used: list[str]
    search_results: list[OntologySearchResult]

class LocationSearchResponse(BaseModel):
    """Output from matching agent"""
    primary_location: OntologySearchResult  # (ID, text)
    alternate_locations: list[OntologySearchResult]
    reasoning: str

@dataclass
class SearchContext:
    """Context for dependency injection"""
    search_client: OntologySearchClient

async def ontology_search_tool(
    ctx: RunContext[SearchContext], 
    query: str, 
    limit: int = 10
) -> dict[str, Any]:
    """Tool for searching medical terminology databases"""
    results = await ctx.deps.search_client.search_tables(
        query, 
        limit_per_table=limit
    )
    # Return dict for agent to process (not JSON string)
    return {
        table: [r.model_dump() for r in results_list]
        for table, results_list in results.items()
    }

def create_search_agent(model_name: str) -> Agent[SearchContext, RawSearchResults]:
    """Search agent: generates queries and gathers results"""
    return Agent[SearchContext, RawSearchResults](
        model=get_openai_model(model_name),
        result_type=RawSearchResults,
        deps_type=SearchContext,
        tools=[ontology_search_tool],
        system_prompt="""You are a medical terminology search specialist. Your role is to generate 
effective search queries for finding anatomic locations in medical ontology databases.

Given a finding name and optional description, you should:
1. Generate 2-4 diverse search queries that might find relevant anatomic locations
2. Consider anatomical terms, body regions, organ systems
3. Try both specific and general terms
4. Use the search tool to gather results from the ontology databases

Return all unique results found across your searches."""
    )

def create_matching_agent(model_name: str) -> Agent[None, LocationSearchResponse]:
    """Matching agent: picks best locations from search results"""
    return Agent[None, LocationSearchResponse](
        model=get_openai_model(model_name),
        result_type=LocationSearchResponse,
        system_prompt="""You are a medical imaging specialist who selects appropriate anatomic 
locations for imaging findings. Given search results from medical ontology databases, you must 
select the best primary anatomic location and 2-3 good alternates.

Selection criteria:
- Find the "sweet spot" of specificity - specific enough to be accurate but general enough 
  to encompass all locations where the finding can occur
- Prefer established ontology terms over very granular subdivisions
- Consider clinical relevance and common usage
- Provide clear reasoning for your selections

Always explain why you selected each location."""
    )
```

### 3. Main API Function (two-agent workflow)

```python
async def find_anatomic_locations(
    finding_name: str,
    description: str | None = None,
    search_model: str | None = None,
    matching_model: str | None = None
) -> LocationSearchResponse:
    """
    Find relevant anatomic locations for a finding using two-agent workflow.
    
    Args:
        finding_name: Name of the finding (e.g., "PCL tear")
        description: Optional detailed description
        search_model: Model for search agent (defaults to small model)
        matching_model: Model for matching agent (defaults to main model)
        
    Returns:
        LocationSearchResponse with selected locations and reasoning
    """
    # Set default models
    if search_model is None:
        search_model = settings.openai_default_model_small
    if matching_model is None:
        matching_model = settings.openai_default_model
        
    # Create search client
    search_client = OntologySearchClient()
    await search_client.connect()
    
    try:
        # Step 1: Search agent generates queries and gathers results
        search_agent = create_search_agent(search_model)
        search_context = SearchContext(search_client=search_client)
        
        prompt = f"Finding: {finding_name}"
        if description:
            prompt += f"\nDescription: {description}"
        
        search_result = await search_agent.run(prompt, deps=search_context)
        search_output = search_result.output
        
        # Step 2: Matching agent picks best locations from results
        matching_agent = create_matching_agent(matching_model)
        
        matching_prompt = f"""
        Finding: {finding_name}
        Description: {description or ""}
        
        Search Results Found ({len(search_output.search_results)} total):
        {json.dumps([r.model_dump() for r in search_output.search_results], indent=2)}
        
        Select the best primary anatomic location and 2-3 good alternates. 
        
        The goal is to find the "sweet spot" where it's as specific as possible, 
        but still encompassing the locations where the finding can occur. For example,
        for the finding of "adrenal nodule", you would prefer "adrenal gland" to 
        "abdomen" (too broad) and "anterior limb of adrenal" (too specific).
        """
        
        matching_result = await matching_agent.run(matching_prompt)
        return matching_result.output
        
    finally:
        if search_client.connected:
            search_client.disconnect()  # Note: disconnect is synchronous
```

### 4. Configuration

Add to `src/findingmodel/config.py`:
```python
# LanceDB settings (optional, with sensible defaults from .env)
lancedb_uri: str | None = Field(default=None, description="LanceDB connection URI")
lancedb_api_key: SecretStr | None = Field(default=None, description="LanceDB API key for cloud")
```

### 5. Testing Strategy

```python
@pytest.mark.callout  # Requires LanceDB connection
async def test_find_anatomic_locations():
    result = await find_anatomic_locations(
        "PCL tear",
        "Tear of the posterior cruciate ligament"
    )
    assert result.primary_location.concept_id  # Has concept ID
    assert result.primary_location.concept_text  # Has concept text
    assert "knee" in result.primary_location.concept_text.lower() or "cruciate" in result.primary_location.concept_text.lower()
```

### 6. Why This Modular Architecture?

**Reusable Components**:
- `OntologySearchClient` can be used by other tools (not just anatomic location search)
- `OntologySearchResult` model standardizes ontology search responses
- `get_openai_model()` helper moved to common utilities

**Clear Separation of Concerns**:
- Search agent: generates queries and collects results (small model)
- Matching agent: analyzes results and picks best matches (main model)
- Search client: handles LanceDB connectivity and search logic

**Follows Existing Patterns**:
- Same dependency injection with `SearchContext`
- Same error handling and async patterns
- Uses existing model settings (small vs main)
- Similar to `similar_finding_models.py` but with better separation

## Implementation Steps

1. **Move `_get_openai_model()` to common.py** and update `similar_finding_models.py` to use it
2. **Create ontology_search module** with reusable `OntologySearchClient` and `OntologySearchResult`
3. **Implement two agents** with proper search/matching separation
4. **Write main API function** with two-agent workflow
5. **Add configuration** for LanceDB settings
6. **Add integration test** with proper mocking
7. **Consider evals** once we have real usage patterns

This architecture provides the reusability you wanted while maintaining clear separation between search logic and matching logic.