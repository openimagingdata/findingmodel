# Product Requirements Document: Comprehensive Ontology Concept Search Tool

## Overview

### Purpose
Create a Pydantic AI-powered tool that searches across all medical ontology tables (RadLex, SNOMED-CT, and anatomic locations) to find and categorize relevant concepts for a given finding model. Unlike the anatomic location search tool which focuses solely on anatomy, this tool will identify all relevant medical concepts and intelligently categorize them by relevance.

### Key Differentiators from Anatomic Location Search
- **Broader Scope**: Searches all ontology tables for any relevant concepts
- **Exclusion of Anatomy**: Filters out anatomical concepts (handled by existing tool)
- **Multi-Category Output**: Returns results organized into three relevance tiers
- **Wider Search Strategy**: Casts a broader net initially to ensure comprehensive coverage

## Functional Requirements

### 1. Search Scope
- **Tables to Search**: 
  - `radlex` - RadLex radiology terminology
  - `snomedct` - SNOMED Clinical Terms
  - `anatomic_locations` - BUT exclude anatomical concepts
- **Concept Types to Include**:
  - Pathological processes and diseases
  - Clinical findings and observations
  - Morphological abnormalities
  - Etiological agents
  - Procedures and imaging techniques
  - Qualifiers and modifiers
- **Concept Types to Exclude**:
  - Pure anatomical structures (handled by anatomic location tool)
  - Body parts without pathological context

### 2. Multi-Agent Architecture

#### Search Agent
- **Purpose**: Generate diverse search queries to cast a wide net
- **Behavior**:
  - Generate 8-12 search queries (vs 3-5 for anatomic search)
  - Include various perspectives: clinical, pathological, radiological, morphological
  - Use both specific and general terms
  - Include synonyms and related terminology

#### Categorization Agent  
- **Purpose**: Analyze search results and categorize by relevance
- **Input**: 30-50 top search results (vs 10-15 for anatomic search)
- **Output**: Categorized concept lists with rationale
- **Categories**:
  1. **Exact Match**: Concepts that directly represent the finding model
  2. **Should Include**: Related concepts highly relevant to the finding
  3. **Marginal/Consider**: Peripherally related concepts that might be relevant

### 3. Output Format

```python
from pydantic import BaseModel, Field

class CategorizedOntologyConcepts(BaseModel):
    """Categorized ontology concepts for a finding model"""
    
    exact_matches: list[OntologySearchResult] = Field(
        description="Concepts that directly represent what the finding model describes"
    )
    
    should_include: list[OntologySearchResult] = Field(
        description="Related concepts that should be included as relevant"
    )
    
    marginal_concepts: list[OntologySearchResult] = Field(
        description="Peripherally related concepts to consider"
    )
    
    search_summary: str = Field(
        description="Summary of search strategy and categorization rationale"
    )
    
    excluded_anatomical: list[str] = Field(
        default_factory=list,
        description="List of anatomical concepts that were filtered out"
    )
```

### 4. Search Strategy

#### Phase 1: Broad Query Generation
- Generate 8-12 diverse search queries
- Include multiple perspectives:
  - Direct finding name
  - Pathological process terms
  - Clinical presentation terms
  - Imaging appearance descriptors
  - Differential diagnosis terms
  - Associated conditions

#### Phase 2: Comprehensive Search
- Execute searches across all tables
- Retrieve top 15-20 results per query
- Apply anatomical filtering
- Deduplicate while preserving scores
- Compile 30-50 candidates for categorization

#### Phase 3: Intelligent Categorization
- Analyze semantic similarity to finding model
- Consider clinical relationships
- Evaluate specificity vs generality
- Apply medical knowledge for relevance scoring
- Organize into three tiers with justification

## Pydantic AI Agent Architecture

### Agent Definitions

```python
from pydantic_ai import Agent
from pydantic import BaseModel, Field

# Search Agent - generates queries and collects results
search_agent = Agent(
    'openai:gpt-4o-mini',  # Use smaller model for search
    output_type=RawSearchResults,
    system_prompt="You are a medical terminology expert..."
)

# Categorization Agent - analyzes and categorizes results  
categorization_agent = Agent(
    'openai:gpt-4o',  # Use larger model for nuanced categorization
    output_type=CategorizedOntologyConcepts,
    system_prompt="You are a medical ontology expert..."
)
```

### Intermediate Models

```python
class RawSearchResults(BaseModel):
    """Output from search agent"""
    
    search_terms_used: list[str] = Field(
        description="List of search terms that were actually used"
    )
    all_results: list[OntologySearchResult] = Field(
        description="All unique search results found across tables"
    )
    anatomical_filtered: list[str] = Field(
        default_factory=list,
        description="Anatomical concepts that were filtered out"
    )
```

### Agent Context and Dependencies

```python
from pydantic_ai import RunContext

class SearchContext(BaseModel):
    """Context for search operations"""
    client: OntologySearchClient
    finding_name: str
    finding_description: str | None
    exclude_anatomical: bool
    
@search_agent.tool
async def search_ontology_table(
    ctx: RunContext[SearchContext],
    query: str,
    table: str
) -> list[OntologySearchResult]:
    """Execute search in specific ontology table"""
    return await ctx.data.client.search(
        query=query,
        table=table,
        limit=20
    )
```

## Technical Requirements

### 1. Extending OntologySearchClient for Multi-Table Operations

The existing `OntologySearchClient` already supports multi-table searches via `search_tables()`. We'll extend it with additional capabilities:

#### Current Capabilities (to reuse)
- **Multi-table search**: `search_tables()` already searches across multiple tables
- **Connection management**: Async connect/disconnect with LanceDB
- **Result formatting**: Returns `OntologySearchResult` objects with scores

#### New Methods to Add

```python
class ExtendedOntologySearchClient(OntologySearchClient):
    """Extended client with multi-table operations and filtering"""
    
    async def search_all_non_anatomical(
        self, 
        query: str, 
        limit_per_table: int = 20
    ) -> list[OntologySearchResult]:
        """
        Search all tables but filter out anatomical concepts.
        
        Args:
            query: Search query
            limit_per_table: Max results per table before filtering
            
        Returns:
            Filtered and merged results from all tables
        """
        # Search all tables
        all_results = await self.search_tables(
            query=query,
            tables=None,  # Search all tables
            limit_per_table=limit_per_table
        )
        
        # Merge and filter results
        merged = []
        for table_name, results in all_results.items():
            for result in results:
                if not self._is_anatomical_concept(result):
                    merged.append(result)
        
        # Sort by score and deduplicate
        return self._deduplicate_results(
            sorted(merged, key=lambda x: x.score, reverse=True)
        )
    
    async def parallel_search_queries(
        self,
        queries: list[str],
        tables: list[str] | None = None,
        limit_per_query: int = 15
    ) -> list[OntologySearchResult]:
        """
        Execute multiple queries in parallel across tables.
        
        Args:
            queries: List of search queries to execute
            tables: Tables to search (None for all)
            limit_per_query: Max results per individual query
            
        Returns:
            Deduplicated merged results from all queries
        """
        tasks = [
            self.search_tables(query, tables, limit_per_query) 
            for query in queries
        ]
        
        # Execute all searches in parallel
        all_results = await asyncio.gather(*tasks)
        
        # Merge and deduplicate
        seen_ids = set()
        merged = []
        
        for query_results in all_results:
            for table_results in query_results.values():
                for result in table_results:
                    key = (result.concept_id, result.table_name)
                    if key not in seen_ids:
                        seen_ids.add(key)
                        merged.append(result)
        
        return sorted(merged, key=lambda x: x.score, reverse=True)
    
    def _is_anatomical_concept(self, result: OntologySearchResult) -> bool:
        """
        Determine if a concept is purely anatomical.
        
        Uses multiple strategies:
        1. SNOMED-CT semantic tags (body structure)
        2. RadLex hierarchy patterns
        3. Keyword detection
        4. Table-specific rules
        """
        text_lower = result.concept_text.lower()
        
        # Check SNOMED-CT semantic tags
        if result.table_name == "snomedct":
            if "(body structure)" in text_lower:
                # But preserve pathological anatomy
                pathological_keywords = [
                    "metastasis", "tumor", "cancer", "lesion", 
                    "abnormal", "malignant", "benign", "cyst"
                ]
                if not any(kw in text_lower for kw in pathological_keywords):
                    return True
        
        # Check if from anatomic_locations table without pathology
        if result.table_name == "anatomic_locations":
            anatomical_only_patterns = [
                r"^(left|right)\s+\w+$",  # Simple laterality
                r"^\w+\s+(region|structure|part)$",  # Basic anatomy
                r"^(upper|lower|middle)\s+\w+$",  # Position descriptors
            ]
            if any(re.match(p, text_lower) for p in anatomical_only_patterns):
                if not any(kw in text_lower for kw in ["abnormal", "lesion"]):
                    return True
        
        return False
    
    def _deduplicate_results(
        self, 
        results: list[OntologySearchResult]
    ) -> list[OntologySearchResult]:
        """
        Deduplicate results while preserving best scores.
        
        Strategy:
        1. Group by concept similarity (not just ID)
        2. Keep highest scoring version
        3. Preserve results from different ontologies if semantically different
        """
        seen_concepts = {}
        deduped = []
        
        for result in results:
            # Create a key based on normalized concept text
            normalized = self._normalize_concept(result.concept_text)
            key = (normalized, result.table_name)
            
            if key not in seen_concepts:
                seen_concepts[key] = result
                deduped.append(result)
            elif result.score > seen_concepts[key].score:
                # Replace with higher scoring version
                idx = deduped.index(seen_concepts[key])
                deduped[idx] = result
                seen_concepts[key] = result
        
        return deduped
    
    def _normalize_concept(self, text: str) -> str:
        """Normalize concept text for deduplication."""
        # Remove semantic tags, parenthetical content
        text = re.sub(r'\([^)]+\)$', '', text).strip()
        # Lowercase and remove extra whitespace
        return ' '.join(text.lower().split())
```

#### Configuration for Multi-Table Search

```python
@dataclass
class MultiTableSearchConfig:
    """Configuration for multi-table ontology searches"""
    
    # Table-specific search weights
    table_weights: dict[str, float] = field(default_factory=lambda: {
        "snomedct": 1.0,      # Full weight for SNOMED-CT
        "radlex": 1.1,        # Slight preference for radiology terms
        "anatomic_locations": 0.5  # Lower weight for anatomy
    })
    
    # Anatomical filtering rules
    anatomical_exclusion_patterns: list[str] = field(default_factory=lambda: [
        r".*\(body structure\)$",
        r"^(left|right|bilateral)\s+",
        r".*\s+(region|area|zone)$"
    ])
    
    # Pathological inclusion overrides
    pathological_keywords: list[str] = field(default_factory=lambda: [
        "metastasis", "metastatic", "tumor", "neoplasm",
        "cancer", "carcinoma", "lesion", "mass",
        "abnormal", "abnormality", "disease", "disorder",
        "syndrome", "malignant", "benign", "cyst"
    ])
    
    # Performance settings
    parallel_batch_size: int = 5  # Max concurrent searches
    cache_ttl_seconds: int = 300  # Cache results for 5 minutes
```

### 2. Reuse Existing Components
- Leverage `OntologySearchResult` model as-is
- Use connection management patterns from anatomic search
- Apply same async/await patterns for performance

### 2. Anatomical Filtering
- Implement intelligent filtering to exclude pure anatomy
- Preserve pathological anatomical concepts (e.g., "hepatic metastasis")
- Use pattern matching and SNOMED-CT semantic tags
- Consider RadLex hierarchy for anatomical identification

### 3. Performance Optimization
- Parallel search execution across tables
- Batch processing of results
- Efficient deduplication algorithms
- Smart caching of common searches

### 4. Error Handling
- Graceful degradation if tables unavailable
- Partial results on timeout
- Clear error messages for debugging
- Fallback strategies for failed searches

## User Stories

### Story 1: Comprehensive Concept Discovery
**As a** medical knowledge engineer  
**I want to** find all relevant ontology concepts for a finding model  
**So that** I can ensure complete semantic coverage in the model

**Acceptance Criteria**:
- Returns concepts from multiple ontologies
- Categorizes by relevance level
- Excludes pure anatomical concepts
- Provides rationale for categorization

### Story 2: Pathological Process Identification
**As a** radiologist creating finding models  
**I want to** identify all pathological processes related to a finding  
**So that** I can properly code differential diagnoses

**Acceptance Criteria**:
- Identifies disease processes across ontologies
- Distinguishes primary from related pathologies
- Links morphological abnormalities to conditions

### Story 3: Semantic Completeness Validation
**As a** finding model curator  
**I want to** validate that all relevant concepts are captured  
**So that** the model is semantically complete

**Acceptance Criteria**:
- Casts wide initial search net
- Identifies marginal concepts for consideration
- Provides comprehensive coverage report

## API Design

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class OntologyConceptSearch:
    """Tool for comprehensive ontology concept search"""
    
    async def search_ontology_concepts(
        self,
        finding_name: str,
        finding_description: str | None = None,
        exclude_anatomical: bool = True,
        max_exact_matches: int = 5,
        max_should_include: int = 10,
        max_marginal: int = 10,
        search_timeout: float = 30.0
    ) -> CategorizedOntologyConcepts:
        """
        Search for relevant ontology concepts across all tables.
        
        Args:
            finding_name: Name of the finding model
            finding_description: Optional detailed description
            exclude_anatomical: Whether to filter out anatomical concepts
            max_exact_matches: Maximum exact match concepts to return
            max_should_include: Maximum should-include concepts
            max_marginal: Maximum marginal concepts to consider
            search_timeout: Timeout for search operations
            
        Returns:
            Categorized ontology concepts with rationale
        """
```

## Implementation Phases

### Phase 1: Core Search Infrastructure
- Extend `OntologySearchClient` for multi-table operations
- Implement anatomical concept filtering
- Create deduplication logic
- Set up parallel search execution

### Phase 2: Two-Agent System
- Implement search query generation agent
  - Define agent with `output_type=RawSearchResults` 
  - Use structured output validation
  - Include self-correction for failed outputs
- Build categorization agent with medical reasoning
  - Define agent with `output_type=CategorizedOntologyConcepts`
  - Inject dependencies via `RunContext`
  - Use reflection for categorization validation
- Create result aggregation pipeline
- Add relevance scoring logic

### Phase 3: Integration & Optimization
- Integrate with existing finding model tools
- Add caching layer
- Optimize performance
- Create comprehensive tests

### Phase 4: Enhancement
- Add configuration for category thresholds
- Implement learning from user feedback
- Add semantic relationship mapping
- Create visualization of concept relationships

## Success Metrics

1. **Coverage**: Identifies 90%+ of relevant concepts from manual curation
2. **Precision**: 80%+ agreement with expert categorization
3. **Performance**: Returns results within 10 seconds for typical queries
4. **Exclusion Accuracy**: 95%+ accuracy in filtering anatomical concepts
5. **User Satisfaction**: Reduces manual concept search time by 75%

## Example Usage

```python
# Example: Finding concepts for "hepatic metastasis"
tool = OntologyConceptSearch()
results = await tool.search_ontology_concepts(
    finding_name="hepatic metastasis",
    finding_description="Secondary malignant neoplasm in liver"
)

# Results would include:
# Exact matches: 
#   - "Metastatic malignant neoplasm to liver" (SNOMED-CT)
#   - "Hepatic metastasis" (RadLex)
#
# Should include:
#   - "Secondary malignant neoplasm" (SNOMED-CT)
#   - "Liver mass" (RadLex)
#   - "Malignant neoplasm of liver" (SNOMED-CT)
#   - "Portal vein invasion" (RadLex)
#
# Marginal:
#   - "Hepatomegaly" (SNOMED-CT)
#   - "Liver lesion" (RadLex)
#   - "Metastatic disease" (SNOMED-CT)
#
# Excluded anatomical:
#   - "Liver structure", "Hepatic parenchyma", "Portal vein"
```

## Dependencies

- Existing `OntologySearchClient` and database infrastructure
- Pydantic AI agents framework
- LanceDB vector search capabilities
- Medical ontology tables (RadLex, SNOMED-CT)

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Over-filtering anatomical concepts | Missing relevant pathological anatomy | Implement smart filtering that preserves pathological context |
| Too many marginal results | Information overload | Configurable thresholds and relevance scoring |
| Performance with broad searches | Slow response times | Parallel execution, caching, query optimization |
| Inconsistent categorization | Poor user experience | Clear categorization criteria, examples in prompts |

## Future Enhancements

1. **Relationship Mapping**: Show semantic relationships between concepts
2. **Learning System**: Improve categorization based on user feedback
3. **Custom Categories**: Allow users to define additional relevance tiers
4. **Batch Processing**: Support multiple finding models in one call
5. **Explanation System**: Detailed reasoning for each categorization decision
6. **Integration with Standard Codes Tool**: Auto-populate finding model codes