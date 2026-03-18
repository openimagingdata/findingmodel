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
DEFAULT_MODEL=anthropic:claude-sonnet-4-6
```

### Supported AI Providers

| Provider | Example | API Key |
|----------|---------|---------|
| OpenAI | `openai:gpt-5.4` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Google (direct) | `google-gla:gemini-3-flash-preview` | `GOOGLE_API_KEY` |
| Ollama (local) | `ollama:llama3` | None required |
| Gateway | `gateway/openai:gpt-5.4` | `PYDANTIC_AI_GATEWAY_API_KEY` |

See [Configuration Guide](../../docs/configuration.md) for model tiers and per-agent overrides.

## CLI (`findingmodel-ai`)

```bash
# Generate finding info from a name
findingmodel-ai make-info "pneumothorax"

# Create a basic model template
findingmodel-ai make-stub-model "pneumothorax"

# Convert a Markdown outline to a finding model
findingmodel-ai markdown-to-fm outline.md

# Assign canonical structured metadata to an existing model
findingmodel-ai assign-metadata pneumonia.fm.json --output pneumonia.updated.fm.json --review-output pneumonia.metadata-review.json

# Search BioOntology.org for ontology concepts
findingmodel-ai ontology search "pneumothorax" --ontology SNOMEDCT --max-results 10
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

`create_model_from_markdown()` is an AI-assisted outline importer. It is useful for turning human-authored notes into an initial model, but it is not intended to guarantee faithful reconstruction of a previously exported Markdown document.

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

`edit_model_markdown()` is convenience tooling for lightweight human or LLM-assisted editing. The canonical representation remains the JSON model, not the Markdown text.

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
        model=model, command="Add severity attribute with values mild, moderate, severe"
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
        finding_name="PCL tear", description="Tear of the posterior cruciate ligament"
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
        finding_name="pneumonia", finding_description="Inflammation of lung parenchyma"
    )

    print(f"Exact matches: {len(result.exact_matches)}")
    for concept in result.exact_matches:
        print(f"  - {concept.code}: {concept.text}")


asyncio.run(main())
```

### Canonical Metadata Assignment

`assign_metadata()` is the new canonical in-memory metadata-assignment entrypoint. It returns an updated `FindingModelFull` plus a separate review object for candidate provenance, confidence, warnings, timings, and optional Logfire trace correlation.

For executable paths that opt into Logfire, `findingmodel_ai.observability.ensure_logfire_configured()` instruments both Pydantic AI and outbound `httpx` calls, so BioOntology and similar non-LLM requests appear in the same trace as the agent run.

```python
import asyncio
from findingmodel import FindingModelFull
from findingmodel_ai.metadata import assign_metadata


async def main():
    with open("pneumonia.fm.json") as f:
        model = FindingModelFull.model_validate_json(f.read())

    result = await assign_metadata(model, model_tier="small")

    print(result.model.body_regions)
    print(result.model.index_codes)
    print(result.review.logfire_trace_id)


asyncio.run(main())
```

### Finding Similar Models

Uses a 5-phase pipeline: fast-path exact lookup → LLM planning (search terms + facet hypotheses) → multi-pass search (unfiltered + facet-filtered + optional `related_models()`) → LLM selection with rejection taxonomy → typed result assembly.

```python
import asyncio
from findingmodel_ai.search import find_similar_models


async def main():
    result = await find_similar_models(
        finding_name="pneumothorax",
        description="Presence of air in the pleural space",
        synonyms=["PTX", "collapsed lung"],
    )

    print(f"Recommendation: {result.recommendation}")  # "edit_existing" or "create_new"

    for match in result.matches:
        print(f"  Match: {match.entry.name} ({match.entry.oifm_id})")
        print(f"    Reasoning: {match.match_reasoning}")

    if result.closest_rejection:
        print(f"  Closest rejected: {result.closest_rejection.entry.name}")
        print(f"    Reason: {result.closest_rejection.rejection_reason.value}")

    # Search pass statistics
    print(f"  Search passes: {result.search_passes}")


asyncio.run(main())
```

## Architecture

### Pipeline Patterns

Complex workflows use structured pipelines:
- **Fast-path**: Exact lookups before any LLM involvement
- **LLM planning**: Structured output for search terms and facet hypotheses
- **Multi-pass search**: Unfiltered pass (recall protection) + facet-filtered pass + deterministic related models
- **LLM selection**: Structured pick with rejection taxonomy and asymmetric generality rule

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
