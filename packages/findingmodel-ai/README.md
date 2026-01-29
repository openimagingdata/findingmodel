# findingmodel-ai

AI-powered tools for creating and editing Open Imaging Finding Models using LLMs.

## Installation

```bash
pip install findingmodel-ai
```

## Configuration

Create a `.env` file with at least one AI provider:

```bash
# Choose one or more AI providers
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Optional: For detailed finding info with citations
TAVILY_API_KEY=your_key_here

# Optional: Override default model
DEFAULT_MODEL=anthropic:claude-sonnet-4-5
```

### Supported AI Providers

| Provider | Example | API Key |
|----------|---------|---------|
| OpenAI | `openai:gpt-5-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google:gemini-3-flash-preview` | `GOOGLE_API_KEY` |
| Ollama (local) | `ollama:llama3` | None required |
| Gateway | `gateway/openai:gpt-5-mini` | `PYDANTIC_AI_GATEWAY_API_KEY` |

See [Configuration Guide](../../docs/configuration.md) for model tiers and per-agent overrides.

## CLI (`findingmodel-ai`)

```bash
# Generate finding info from a name
findingmodel-ai make-info "pneumothorax"

# Create a basic model template
findingmodel-ai make-stub-model "pneumothorax"

# Convert Markdown to finding model
findingmodel-ai markdown-to-fm outline.md
```

## Tools

### Creating Finding Info

```python
import asyncio
from findingmodel_ai.authoring import create_info_from_name, add_details_to_info

async def main():
    # Generate basic info from a finding name
    info = await create_info_from_name("Pneumothorax")
    print(f"Name: {info.name}")
    print(f"Synonyms: {info.synonyms}")
    print(f"Description: {info.description}")

    # Add detailed info with citations (requires TAVILY_API_KEY)
    enhanced = await add_details_to_info(info)
    print(f"Detail: {enhanced.detail[:200]}...")
    print(f"Citations: {len(enhanced.citations)}")

asyncio.run(main())
```

### Creating Models from Markdown

```python
import asyncio
from findingmodel_ai.authoring import create_model_from_markdown, create_info_from_name

async def main():
    info = await create_info_from_name("pneumothorax")

    markdown = """
    # Pneumothorax Attributes
    - Size: small (<2cm), moderate (2-4cm), large (>4cm)
    - Location: apical, basilar, lateral
    - Tension: present, absent
    """

    model = await create_model_from_markdown(info, markdown_text=markdown)
    print(f"Created model with {len(model.attributes)} attributes")

asyncio.run(main())
```

### AI-Powered Model Editing

```python
import asyncio
from findingmodel import FindingModelFull
from findingmodel_ai.authoring import edit_model_natural_language, edit_model_markdown

async def main():
    # Load existing model
    with open("pneumothorax.fm.json") as f:
        model = FindingModelFull.model_validate_json(f.read())

    # Edit with natural language
    result = await edit_model_natural_language(
        model=model,
        command="Add severity attribute with values mild, moderate, severe"
    )

    if result.rejections:
        print(f"Rejected changes: {result.rejections}")

    print(f"Updated model has {len(result.model.attributes)} attributes")

asyncio.run(main())
```

### Finding Anatomic Locations

```python
import asyncio
from findingmodel_ai.search import find_anatomic_locations

async def main():
    result = await find_anatomic_locations(
        finding_name="PCL tear",
        description="Tear of the posterior cruciate ligament"
    )

    print(f"Primary: {result.primary_location.concept_text}")
    print(f"  ID: {result.primary_location.concept_id}")

    for alt in result.alternate_locations:
        print(f"Alternate: {alt.concept_text}")

asyncio.run(main())
```

### Ontology Concept Matching

```python
import asyncio
from findingmodel_ai.search import match_ontology_concepts

async def main():
    result = await match_ontology_concepts(
        finding_name="pneumonia",
        finding_description="Inflammation of lung parenchyma"
    )

    print(f"Exact matches: {len(result.exact_matches)}")
    for concept in result.exact_matches:
        print(f"  - {concept.code}: {concept.text}")

asyncio.run(main())
```

### Finding Similar Models

```python
import asyncio
from findingmodel_ai.search import find_similar_models

async def main():
    analysis = await find_similar_models(
        finding_name="pneumothorax",
        description="Presence of air in the pleural space",
        synonyms=["PTX", "collapsed lung"]
    )

    print(f"Recommendation: {analysis.recommendation}")
    print(f"Confidence: {analysis.confidence:.2f}")

    if analysis.similar_models:
        for model in analysis.similar_models:
            print(f"  Similar: {model.name} ({model.oifm_id})")

asyncio.run(main())
```

## Architecture

### Two-Agent Patterns

Complex workflows use paired agents:
- **Search agent**: Generates diverse queries to find candidates
- **Matching agent**: Selects best options based on clinical relevance

### Multi-Provider Support

All tools support multiple AI providers through Pydantic AI:
- Configure globally via `DEFAULT_MODEL`
- Override per-agent via `AGENT_MODEL_OVERRIDES__<tag>`

### Structured Outputs

All agents return typed Pydantic models for reliable downstream processing.

## Related Packages

- **[findingmodel](../findingmodel/README.md)**: Core models and Index API
- **[anatomic-locations](../anatomic-locations/README.md)**: Anatomic location queries

## Documentation

- [Configuration Guide](../../docs/configuration.md)
- [Anatomic Locations Guide](../../docs/anatomic-locations.md)
