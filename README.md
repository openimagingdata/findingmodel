# `findingmodel` Package

A Python library for managing Open Imaging Finding Models - structured data models used to describe medical imaging findings in radiology reports. The library provides tools for creating, converting, and managing these finding models with AI-powered features for medical ontology integration.

## Features

- **Finding Model Management**: Create and manage structured medical finding models with attributes
- **AI-Powered Tools**: Generate finding descriptions, synonyms, and detailed information using OpenAI and Perplexity
- **Medical Ontology Integration**: Search and match concepts across multiple backends:
  - **BioOntology API**: Access to 800+ medical ontologies including SNOMED-CT, ICD-10, LOINC
  - **DuckDB Search**: High-performance vector and full-text search with HNSW indexing
- **Protocol-Based Architecture**: Flexible backend support with automatic parallel execution
- **Finding Model Index**: Fast lookup and search across finding model definitions
- **Anatomic Location Discovery**: Two-agent AI system for finding relevant anatomic locations

## Installation

```bash
pip install findingmodel
```

## Configuration

Configure the library by creating a `.env` file in your project root. See `.env.sample` for all available options.

### API Keys (Required/Optional by Feature)

Different features require different API keys:

| Feature | Required Key | Purpose |
|---------|--------------|---------|
| **Core AI Features** | `OPENAI_API_KEY` | Generate descriptions, synonyms, create models from markdown |
| **Detailed Finding Info** | `PERPLEXITY_API_KEY` | Add citations and detailed descriptions (requires OpenAI key too) |
| **800+ Medical Ontologies** | `BIOONTOLOGY_API_KEY` | Access BioOntology.org for SNOMED-CT, ICD-10, LOINC, etc. |

```bash
# Required for most features
OPENAI_API_KEY=your_key_here

# Optional - only needed for add_details_to_info()
PERPLEXITY_API_KEY=your_key_here

# Optional - only needed for BioOntology backend in ontology searches
BIOONTOLOGY_API_KEY=your_key_here
```

**Note:** The Index and anatomic location search work without any API keys (DuckDB backend). OpenAI is only needed when using AI-powered tools.

### Local Database Configuration

By default, the up-to-date finding models index database from the [GitHub repository](https://github.com/openimagingdata/findingmodels) is automatically downloaded to a data directory based on an online manifest. To use a pre-downloaded version (e.g., in production/Docker deployments), you can specify its path:

```bash
# Production: use pre-mounted files
DUCKDB_INDEX_PATH=/mnt/data/finding_models.duckdb
```

Alternatively, you can also lock to a specific version of the index database by specifying a download URL and its hash.

**Configuration Priority:**
1. If file exists and no URL/hash specified → uses file directly (no download)
2. If file exists with URL/hash → verifies hash, re-downloads if mismatch
3. If file doesn't exist with URL/hash → downloads from URL
4. If nothing specified → downloads from manifest.json (default)

The anatomic locations database for ontologic lookups works similarly. See `.env.sample` for more configuration options including custom download URLs and relative paths.

## CLI

The package provides CLI commands for model conversion and database management:

```shell
$ python -m findingmodel --help
```

**Available commands:**
- `fm-to-markdown` / `markdown-to-fm`: Convert between JSON and Markdown formats
- `make-info`: Generate finding descriptions and synonyms
- `make-stub-model`: Create basic finding model templates
- `config`: View current configuration
- `index`: Manage finding model index (build, update, stats)
- `anatomic`: Manage anatomic location database (build, validate, stats)

For database maintainers, see [Database Management Guide](docs/database-management.md) for detailed information on building and updating databases.

> **Note**: The AI-powered model editing functionality (`edit_model_natural_language`, `edit_model_markdown`) is available through the Python API. See an interactive demo at `scripts/edit_finding_model.py`.

## Models

### `FindingModelBase`

Basics of a finding model, including name, description, and attributes.

**Properties:**

* `name`: The name of the finding.
* `description`: A brief description of the finding. *Optional*.
* `synonyms`: Alternative names or abbreviations for the finding. *Optional*.
* `tags`: Keywords or categories associated with the finding. *Optional*.
* `attributes`: A collection of attributes objects associated with the finding.

**Methods:**

* `as_markdown()`: Generates a markdown representation of the finding model.

### `FindingModelFull`

Uses `FindingModelBase`, but adds contains more detailed metadata:

* Requiring IDs on models and attributes (with enumerated codes for values on choice attributes)
* Allows index codes on multiple levels (model, attribute, value)
* Allows contributors (people and organization)

### `FindingInfo`

Information on a finding, including description and synonyms, can add detailed description and citations.

**Properties:**

* `name`: The name of the finding.
* `synonyms`: Alternative names or abbreviations for the finding. *Optional*.
* `description`: A brief description of the finding. *Optional*.
* `detail`: A more detailed description of the finding. *Optional*.
* `citations`: A list of citations or references related to the finding. *Optional*.

## Index

The `Index` class provides fast lookup and search across finding model definitions. The index contains metadata about finding models, including their names, descriptions, synonyms, tags, and contributor information.

**Database auto-downloads on first use** - no manual setup required. For database maintenance, see the [Database Management Guide](docs/database-management.md).

### Searching and Lookup

```python
import asyncio
from findingmodel import Index

async def main():
    async with Index() as index:
        # Get count of indexed models
        count = await index.count()
        print(f"Total models indexed: {count}")

        # Lookup by ID, name, or synonym
        metadata = await index.get("abdominal aortic aneurysm")
        if metadata:
            print(f"Found: {metadata.name} ({metadata.oifm_id})")
            print(f"Description: {metadata.description}")
            print(f"Synonyms: {metadata.synonyms}")

        # Search for models (returns list of IndexEntry objects)
        results = await index.search("abdominal", limit=5)
        for result in results:
            print(f"- {result.name}: {result.oifm_id}")

        # Check if a model exists
        exists = await index.contains("pneumothorax")
        print(f"Pneumothorax exists: {exists}")

asyncio.run(main())
```

### Listing and Filtering

```python
async def browse_models():
    async with Index() as index:
        # Get all models with pagination
        models, total = await index.all(limit=20, offset=0, order_by="name", order_dir="asc")
        print(f"Showing {len(models)} of {total} total models:")
        for model in models:
            print(f"  - {model.name} ({model.oifm_id})")

        # Search by slug name pattern (exact match)
        results, count = await index.search_by_slug("pneumothorax", match_type="exact")
        print(f"\nExact matches: {count}")

        # Search by slug name pattern (prefix match - starts with)
        results, count = await index.search_by_slug("aortic", match_type="prefix", limit=10)
        print(f"\nModels starting with 'aortic': {count}")
        for result in results:
            print(f"  - {result.name}")

        # Search by slug name pattern (contains - default)
        results, count = await index.search_by_slug("abscess", limit=10)
        print(f"\nModels containing 'abscess': {count}")

        # Count models matching a pattern
        exact_count = await index.count_search("lung_nodule", match_type="exact")
        contains_count = await index.count_search("lung", match_type="contains")
        print(f"\nExact 'lung_nodule': {exact_count}")
        print(f"Contains 'lung': {contains_count}")

asyncio.run(browse_models())
```

**Available methods:**
- `all(limit, offset, order_by, order_dir)` - Get paginated list of all models with sorting
- `search_by_slug(pattern, match_type, limit, offset)` - Search by slug name with exact/prefix/contains matching
- `count_search(pattern, match_type)` - Count models matching a slug name pattern

### Working with Contributors

```python
async def get_contributors():
    async with Index() as index:
        # Get a person by GitHub username
        person = await index.get_person("talkasab")
        if person:
            print(f"Name: {person.name}, Email: {person.email}")

        # Get an organization by code
        org = await index.get_organization("MSFT")
        if org:
            print(f"Organization: {org.name}")

        # Get all people (sorted by name)
        people = await index.get_people()
        print(f"Found {len(people)} people:")
        for person in people[:5]:  # Show first 5
            print(f"  - {person.name} (@{person.github_username})")

        # Get all organizations (sorted by name)
        organizations = await index.get_organizations()
        print(f"Found {len(organizations)} organizations:")
        for org in organizations[:5]:  # Show first 5
            print(f"  - {org.name} ({org.code})")

        # Count contributors
        people_count = await index.count_people()
        org_count = await index.count_organizations()
        print(f"People: {people_count}, Organizations: {org_count}")

asyncio.run(get_contributors())
```

See [example usage in notebook](notebooks/findingmodel_index.ipynb) and the [Database Management Guide](docs/database-management.md) for information on updating the index.

## Tools

All tools are available through `findingmodel.tools`. Import them like:

```python
from findingmodel.tools import create_info_from_name, add_details_to_info
# Or import the entire tools module
import findingmodel.tools as tools
```

> **Note**: Previous function names (e.g., `describe_finding_name`, `create_finding_model_from_markdown`) are still available but deprecated. They will show deprecation warnings and point to the new names.

### `create_info_from_name()`

Takes a finding name and generates a usable description and possibly synonyms (`FindingInfo`) using OpenAI models (requires `OPENAI_API_KEY` to be set to a valid value).

```python
import asyncio
from findingmodel.tools import create_info_from_name

async def describe_finding():
    # Generate basic finding information
    info = await create_info_from_name("Pneumothorax")
    print(f"Name: {info.name}")
    print(f"Synonyms: {info.synonyms}")
    print(f"Description: {info.description[:100]}...")
    return info

info = asyncio.run(describe_finding())
# Output:
# Name: pneumothorax
# Synonyms: ['PTX']
# Description: Pneumothorax is the presence of air in the pleural space...
```

### `add_details_to_info()`

Takes a described finding as above and uses Perplexity to get a lot of possible reference information, possibly including citations (requires `PERPLEXITY_API_KEY` to be set to a valid value).

```python
import asyncio
from findingmodel.tools import add_details_to_info
from findingmodel import FindingInfo

async def enhance_finding():
    # Start with basic finding info
    finding = FindingInfo(
        name="pneumothorax", 
        synonyms=['PTX'],
        description='Pneumothorax is the presence of air in the pleural space'
    )
    
    # Add detailed information and citations
    enhanced = await add_details_to_info(finding)
    
    print(f"Detail length: {len(enhanced.detail)} characters")
    print(f"Citations found: {len(enhanced.citations)}")
    
    # Show first few citations
    for i, citation in enumerate(enhanced.citations[:3], 1):
        print(f"  {i}. {citation}")
    
    return enhanced

enhanced_info = asyncio.run(enhance_finding())
# Output:
# Detail length: 2547 characters  
# Citations found: 8
#   1. https://pubs.rsna.org/doi/full/10.1148/rg.2020200020
#   2. https://ajronline.org/doi/full/10.2214/AJR.17.18721
#   3. https://radiopaedia.org/articles/pneumothorax
```

### `create_model_from_markdown()`

Creates a `FindingModel` from a markdown file or text using OpenAI API.

```python
import asyncio
from pathlib import Path
from findingmodel.tools import create_model_from_markdown, create_info_from_name

async def create_from_markdown():
    # First create basic info about the finding
    finding_info = await create_info_from_name("pneumothorax")
    
    # Option 1: Create from markdown text
    markdown_outline = """
    # Pneumothorax Attributes
    - Size: small (<2cm), moderate (2-4cm), large (>4cm)
    - Location: apical, basilar, lateral, complete
    - Tension: present, absent, indeterminate
    - Cause: spontaneous, traumatic, iatrogenic
    """
    
    model = await create_model_from_markdown(
        finding_info, 
        markdown_text=markdown_outline
    )
    print(f"Created model with {len(model.attributes)} attributes")
    
    # Option 2: Create from markdown file
    # Save markdown to file first
    Path("pneumothorax.md").write_text(markdown_outline)
    
    model_from_file = await create_model_from_markdown(
        finding_info,
        markdown_path="pneumothorax.md"
    )
    
    # Display the attributes
    for attr in model.attributes:
        print(f"- {attr.name}: {attr.type}")
        if hasattr(attr, 'values'):
            print(f"  Values: {[v.name for v in attr.values]}")
    
    return model

model = asyncio.run(create_from_markdown())
# Output:
# Created model with 4 attributes
# - size: choice
#   Values: ['small (<2cm)', 'moderate (2-4cm)', 'large (>4cm)']
# - location: choice  
#   Values: ['apical', 'basilar', 'lateral', 'complete']
# - tension: choice
#   Values: ['present', 'absent', 'indeterminate']
# - cause: choice
#   Values: ['spontaneous', 'traumatic', 'iatrogenic']
```

### `create_model_stub_from_info()`

Given even a basic `FindingInfo`, turn it into a `FindingModelBase` object with at least two attributes:

* **presence**: Whether the finding is seen  
(present, absent, indeterminate, unknown)
* **change from prior**: How the finding has changed from prior exams  
(unchanged, stable, increased, decreased, new, resolved, no prior)

```python
import asyncio
from findingmodel.tools import create_info_from_name, create_model_stub_from_info

async def create_stub():
    # Create finding info
    finding_info = await create_info_from_name("pneumothorax")
    
    # Create a basic model stub with standard presence/change attributes
    stub_model = create_model_stub_from_info(finding_info)
    
    print(f"Model name: {stub_model.name}")
    print(f"Created model with {len(stub_model.attributes)} attributes:")
    
    for attr in stub_model.attributes:
        print(f"\n- {attr.name} ({attr.type}):")
        if hasattr(attr, 'values'):
            for value in attr.values:
                print(f"  • {value.name}")
    
    # You can also add tags
    stub_with_tags = create_model_stub_from_info(
        finding_info, 
        tags=["chest", "emergency", "trauma"]
    )
    print(f"\nTags: {stub_with_tags.tags}")
    
    return stub_model

stub = asyncio.run(create_stub())
# Output:
# Model name: pneumothorax
# Created model with 2 attributes:
# 
# - presence (choice):
#   • present
#   • absent  
#   • indeterminate
#   • unknown
# 
# - change from prior (choice):
#   • unchanged
#   • stable
#   • increased
#   • decreased
#   • new
#   • resolved
#   • no prior
# 
# Tags: ['chest', 'emergency', 'trauma']
```

### `add_ids_to_model()`

Generates and adds OIFM IDs to a `FindingModelBase` object and returns it as a `FindingModelFull` object. Note that the `source` parameter refers to the source component of the OIFM ID, which describes the originating organization of the model (e.g., `MGB` for Mass General Brigham and `MSFT` for Microsoft).

```python
import asyncio
from findingmodel.tools import (
    add_ids_to_model, 
    create_model_stub_from_info,
    create_info_from_name
)

async def add_identifiers():
    # Create a basic model (without IDs)
    finding_info = await create_info_from_name("pneumothorax")
    stub_model = create_model_stub_from_info(finding_info)
    
    # Add OIFM IDs for tracking and standardization
    # Source can be 3 or 4 letters (e.g., "MGB", "MSFT")
    full_model = add_ids_to_model(stub_model, source="MSFT")
    
    print(f"Model ID: {full_model.oifm_id}")
    print(f"Attribute IDs:")
    for attr in full_model.attributes:
        print(f"  - {attr.name}: {attr.oifma_id}")
        if hasattr(attr, 'values'):
            for value in attr.values:
                print(f"    • {value.name}: {value.oifmv_id}")
    
    return full_model

full_model = asyncio.run(add_identifiers())
# Output:
# Model ID: OIFM_MSFT_123456
# Attribute IDs:
#   - presence: OIFMA_MSFT_789012
#     • present: OIFMV_MSFT_345678
#     • absent: OIFMV_MSFT_901234
#     • indeterminate: OIFMV_MSFT_567890
#     • unknown: OIFMV_MSFT_123456
#   - change from prior: OIFMA_MSFT_789013
#     • unchanged: OIFMV_MSFT_345679
#     • stable: OIFMV_MSFT_901235
#     ...

### `assign_real_attribute_ids()`

Finalizes placeholder attribute IDs (`PLACEHOLDER_ATTRIBUTE_ID`) that were created through the editing workflows. This is used by the interactive demos before saving, but you can also call it directly when scripting bulk edits.

```python
from findingmodel.finding_model import FindingModelFull
from findingmodel.tools.add_ids import PLACEHOLDER_ATTRIBUTE_ID
from findingmodel.tools.model_editor import assign_real_attribute_ids


def finalize_ids(model_json: str) -> FindingModelFull:
    model = FindingModelFull.model_validate_json(model_json)
    # Ensure any newly added attributes receive permanent IDs and value codes
    finalized = assign_real_attribute_ids(model)
    return finalized


# Placeholder-rich model JSON from an editing session
with open("pulmonary_embolism.edited.json", "r") as fh:
    edited_json = fh.read()

model_with_ids = finalize_ids(edited_json)
assert all(attr.oifma_id != PLACEHOLDER_ATTRIBUTE_ID for attr in model_with_ids.attributes)
```
```

### `add_standard_codes_to_model()`

Edits a `FindingModelFull` in place to include some RadLex and SNOMED-CT codes that correspond to some typical situations.

```python
import asyncio
from findingmodel.tools import (
    add_standard_codes_to_model,
    add_ids_to_model,
    create_model_stub_from_info,
    create_info_from_name
)

async def add_medical_codes():
    # Create a full model with IDs
    finding_info = await create_info_from_name("pneumothorax")
    stub_model = create_model_stub_from_info(finding_info)
    full_model = add_ids_to_model(stub_model, source="MSFT")
    
    # Add standard medical vocabulary codes
    add_standard_codes_to_model(full_model)
    
    print("Added standard codes:")
    
    # Check model-level codes
    if full_model.index_codes:
        print(f"\nModel codes:")
        for code in full_model.index_codes:
            print(f"  - {code.system}: {code.code} ({code.display})")
    
    # Check attribute-level codes
    for attr in full_model.attributes:
        if attr.index_codes:
            print(f"\n{attr.name} attribute codes:")
            for code in attr.index_codes:
                print(f"  - {code.system}: {code.code}")
        
        # Check value-level codes
        if hasattr(attr, 'values'):
            for value in attr.values:
                if value.index_codes:
                    print(f"  {value.name} value codes:")
                    for code in value.index_codes:
                        print(f"    - {code.system}: {code.code}")
    
    return full_model

model_with_codes = asyncio.run(add_medical_codes())
# Output:
# Added standard codes:
# 
# Model codes:
#   - RadLex: RID5352 (pneumothorax)
#   - SNOMED-CT: 36118008 (Pneumothorax)
# 
# presence attribute codes:
#   - RadLex: RID39039
#   present value codes:
#     - RadLex: RID28472
#   absent value codes:
#     - RadLex: RID28473
# ...
```

### `find_similar_models()`

Searches for existing finding models in the database that are similar to a proposed new finding. This helps avoid creating duplicate models by identifying existing models that could be edited instead. Uses AI agents to perform intelligent search and analysis.

```python
import asyncio
from findingmodel.tools import find_similar_models
from findingmodel.index import Index

async def check_for_similar_models():
    # Initialize index (DuckDB backend)
    index = Index()
    
    # Search for models similar to a proposed finding
    analysis = await find_similar_models(
        finding_name="pneumothorax",
        description="Presence of air in the pleural space causing lung collapse",
        synonyms=["PTX", "collapsed lung"],
        index=index  # Optional, will create one if not provided
    )
    
    print(f"Recommendation: {analysis.recommendation}")
    print(f"Confidence: {analysis.confidence:.2f}")
    
    if analysis.similar_models:
        print("
Similar existing models found:")
        for model in analysis.similar_models:
            print(f"  - {model.name} (ID: {model.oifm_id})")
    
    # The recommendation will be one of:
    # - "edit_existing": Very similar model found, edit it instead
    # - "create_new": No similar models, safe to create new one
    # - "review_needed": Some similarity found, manual review recommended
    
    return analysis

result = asyncio.run(check_for_similar_models())
# Output:
# Recommendation: edit_existing
# Confidence: 0.90
# 
# Similar existing models found:
#   - pneumothorax (ID: OIFM_MSFT_123456)
```

**Key Features:**
- **Intelligent search**: Uses AI agents to search with various terms and strategies
- **Duplicate prevention**: Identifies if a model already exists for the finding
- **Smart recommendations**: Provides guidance on whether to create new or edit existing
- **Synonym matching**: Checks both names and synonyms for matches
- **Confidence scoring**: Indicates how confident the system is in its recommendation

### `find_anatomic_locations()`

Finds relevant anatomic locations for a finding using a two-agent workflow. The search agent generates diverse queries to search medical ontology databases (anatomic_locations, radlex, snomedct), while the matching agent selects the best primary and alternate locations based on clinical relevance and specificity.

```python
import asyncio
from findingmodel.tools import find_anatomic_locations

async def find_locations():
    # Find anatomic locations for a finding
    result = await find_anatomic_locations(
        finding_name="PCL tear",
        description="Tear of the posterior cruciate ligament",
        search_model="gpt-4o-mini",  # Optional, defaults to small model
        matching_model="gpt-4o"      # Optional, defaults to main model
    )
    
    print(f"Primary location: {result.primary_location.concept_text}")
    print(f"  ID: {result.primary_location.concept_id}")
    print(f"  Table: {result.primary_location.table_name}")
    
    if result.alternate_locations:
        print("\nAlternate locations:")
        for alt in result.alternate_locations:
            print(f"  - {alt.concept_text} ({alt.table_name})")
    
    print(f"\nReasoning: {result.reasoning}")
    
    # Convert to IndexCode for use in finding models
    index_code = result.primary_location.as_index_code()
    print(f"\nAs IndexCode: {index_code.system}:{index_code.code}")
    
    return result

result = asyncio.run(find_locations())
# Output:
# Primary location: Posterior cruciate ligament
#   ID: RID2905
#   Table: radlex
# 
# Alternate locations:
#   - Knee joint (anatomic_locations)
#   - Cruciate ligament structure (snomedct)
# 
# Reasoning: Selected "Posterior cruciate ligament" as the primary location because...
# 
# As IndexCode: RADLEX:RID2905
```

**Key Features:**
- **Two-agent architecture**: Search agent finds candidates, matching agent selects best options
- **Multiple ontology sources**: Searches across anatomic_locations, RadLex, and SNOMED-CT
- **Intelligent selection**: Finds the "sweet spot" of specificity - specific enough to be accurate but general enough to be useful
- **Reusable components**: `LanceDBOntologySearchClient` can be used for other ontology searches
- **Production ready**: Proper error handling, logging, and connection lifecycle management

### `match_ontology_concepts()`

High-performance search for relevant medical concepts across multiple ontology databases. Supports both LanceDB vector search and BioOntology REST API through a flexible Protocol-based architecture.

```python
import asyncio
from findingmodel.tools.ontology_concept_match import match_ontology_concepts

async def search_concepts():
    # Automatically uses all configured backends (LanceDB and/or BioOntology)
    result = await match_ontology_concepts(
        finding_name="pneumonia",
        finding_description="Inflammation of lung parenchyma",  # Optional
        exclude_anatomical=True  # Exclude anatomical structures (default: True)
    )
    
    print(f"Exact matches ({len(result.exact_matches)}):")
    for concept in result.exact_matches:
        print(f"  - {concept.code}: {concept.text}")
    
    print(f"\nShould include ({len(result.should_include)}):")
    for concept in result.should_include:
        print(f"  - {concept.code}: {concept.text}")
    
    print(f"\nMarginal relevance ({len(result.marginal)}):")
    for concept in result.marginal:
        print(f"  - {concept.code}: {concept.text}")
    
    return result

result = asyncio.run(search_concepts())
# Output:
# Exact matches (5):
#   - RID5350: pneumonia
#   - 233604007: Pneumonia
#   - RID34769: viral pneumonia
#   - 53084003: Bacterial pneumonia
#   - RID3541: pneumonitis
# 
# Should include (3):
#   - RID5351: lobar pneumonia
#   - RID5352: bronchopneumonia
#   - 233607000: Atypical pneumonia
# 
# Marginal relevance (2):
#   - RID4866: pulmonary edema
#   - RID34637: bronchitis
```

**Key Features:**
- **Multi-backend support**: Automatically uses LanceDB and/or BioOntology based on configuration
- **Protocol-based architecture**: Clean abstraction allows easy addition of new search providers
- **High performance**: ~10 second searches with parallel backend execution
- **Guaranteed exact matches**: Post-processing ensures exact name matches are never missed
- **Smart categorization**: Three tiers - exact matches, should include, marginal
- **Excludes anatomy**: Focuses on diseases/conditions (use `find_anatomic_locations()` for anatomy)

### `edit_model_natural_language()` and `edit_model_markdown()`

AI-powered editing tools for finding models with two modes: natural language commands and Markdown-based editing. Both preserve existing OIFM IDs and only allow safe additions and non-semantic text changes.

```python
import asyncio
from findingmodel import FindingModelFull
from findingmodel.tools.model_editor import (
    edit_model_natural_language,
    edit_model_markdown,
    export_model_for_editing
)

async def edit_with_natural_language():
    # Load an existing model
    with open("pneumothorax.fm.json") as f:
        model = FindingModelFull.model_validate_json(f.read())
    
    # Add a new attribute using natural language
    result = await edit_model_natural_language(
        model=model,
        command="Add severity attribute with values mild, moderate, severe"
    )
    
    # Check for any rejected changes
    if result.rejections:
        print("Some changes were rejected:")
        for rejection in result.rejections:
            print(f"  - {rejection}")
    
    # The updated model with new attribute
    updated_model = result.model
    print(f"Model now has {len(updated_model.attributes)} attributes")
    
    return result

async def edit_with_markdown():
    # Load an existing model
    with open("pneumothorax.fm.json") as f:
        model = FindingModelFull.model_validate_json(f.read())
    
    # Export to editable Markdown format
    markdown_content = export_model_for_editing(model)
    print("Current Markdown:")
    print(markdown_content)
    
    # Add new attribute section to the markdown
    edited_markdown = markdown_content + """
### severity

Severity of the pneumothorax

- mild: Small pneumothorax with minimal clinical impact
- moderate: Medium-sized pneumothorax requiring monitoring
- severe: Large pneumothorax requiring immediate intervention

"""
    
    # Apply the Markdown edits
    result = await edit_model_markdown(
        model=model,
        edited_markdown=edited_markdown
    )
    
    # Check results
    if result.rejections:
        print("Some changes were rejected:")
        for rejection in result.rejections:
            print(f"  - {rejection}")
    
    updated_model = result.model
    print(f"Model now has {len(updated_model.attributes)} attributes")
    
    return result

# Run examples
nl_result = asyncio.run(edit_with_natural_language())
md_result = asyncio.run(edit_with_markdown())
```

**Safety Features:**
- **ID preservation**: All existing OIFM IDs (model, attribute, value) are preserved
- **Safe changes only**: Only allows adding new attributes/values or editing non-semantic text
- **Rejection feedback**: Clear explanations when changes are rejected as unsafe
- **Validation**: Built-in validation ensures model integrity and proper ID generation

**Editable Markdown Format:**
```markdown
# Model Name

Model description here.

Synonyms: synonym1, synonym2

## Attributes

### attribute_name

Optional attribute description

- value1: Optional value description
- value2: Another value
- value3

### another_attribute

- option1
- option2
```

**Use Cases:**
- **Natural Language**: "Add location attribute with upper, middle, lower lobe options"
- **Markdown**: Direct editing of exported model structure with full control over formatting
- **Collaborative**: Export to Markdown, share with clinical experts, import their edits
- **Batch editing**: Multiple attribute additions in a single Markdown edit session
