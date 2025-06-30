# `findingmodel` Package

Contains library code for managing `FindingModel` objects.

Look in the [demo notebook](notebooks/findingmodel_tools.ipynb).

## CLI

```shell
$ python -m findingmodel
Usage: python -m findingmodel [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  config           Show the currently active configuration.
  fm-to-markdown   Convert finding model JSON file to Markdown format.
  make-info        Generate description/synonyms and more...
  make-stub-model  Generate a simple finding model object (presence and...
  markdown-to-fm   Convert markdown file to finding model format.
```

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

For a directory structured with a `defs` sub-directory containing definitions files (e.g., in a clone of the [Open Imaging Finding Model repository](https://github.com/openimagingdata/findingmodels)), creates/maintains an index as a JSONL file `index.jsonl` in the base directory (alongside the `defs` directory).

```python
from findingmodel.index import Index

index = Index() # Initialize with base directory; will find existing JSONL
print(await index.count())

metadata = index.get("abdominal aortic aneurysm") # Lookup by ID, name, synonym
print(metadata.model_dump())
# > {'attributes': [{'attribute_id': 'OIFMA_MSFT_898601',
# >                  'name': 'presence',
# >                  'type': 'choice'},
# >                 {'attribute_id': 'OIFMA_MSFT_783072',
# >                  'name': 'change from prior',
# >                  'type': 'choice'}],
# >  'description': 'An abdominal aortic aneurysm (AAA) is a localized dilation of '
# >                 'the abdominal aorta, typically defined as a diameter greater '
# >                 'than 3 cm, which can lead to rupture and significant '
# >                 'morbidity or mortality.',
# >  'filename': 'abdominal_aortic_aneurysm.fm.json',
# >  'name': 'abdominal aortic aneurysm',
# >  'oifm_id': 'OIFM_MSFT_134126',
# >  'synonyms': ['AAA'],
# >  'tags': None}

results = index.search("abdominal") # Returns matching names or synonyms
```

See [example usage in notebook](notebooks/findingmodel_index.ipynb).

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
from findingmodel.tools import create_info_from_name

await create_info_from_name("Pneumothorax")

>>> FindingInfo(name="pneumothorax", synonyms=["PTX"], 
  description="Pneumothorax is the...")
```

### `add_details_to_info()`

Takes a described finding as above and uses Perplexity to get a lot of possible reference information, possibly including citations (requires `PERPLEXITY_API_KEY` to be set to a valid value).

```python
from findingmodel.tools import add_details_to_info

finding = FindingInfo(name="pneumothorax", synonyms=['PTX'],
    description='Pneumothorax is the presence...')

await add_details_to_info(finding)

>>> FindingInfo(name='pneumothorax', synonyms=['PTX'], 
 description='Pneumothorax is the...'
 detail='## Pneumothorax\n\n### Appearance on Imaging Studies\n\nA pneumothorax...',
 citations=['https://pubs.rsna.org/doi/full/10.1148/rg.2020200020', 
  'https://ajronline.org/doi/full/10.2214/AJR.17.18721', ...])
```

### `create_model_from_markdown()`

Creates a `FindingModel` from a markdown file or text using OpenAI API.

```python
from findingmodel.tools import create_model_from_markdown, create_info_from_name

# First create basic info about the finding
finding_info = await create_info_from_name("pneumothorax")

# Then create a model from markdown outline
markdown_outline = """
# Pneumothorax Attributes
- Size: small, moderate, large
- Location: apical, basilar, complete
- Tension: present, absent
"""

model = await create_model_from_markdown(
    finding_info, 
    markdown_text=markdown_outline
)
```

### `create_model_stub_from_info()`

Given even a basic `FindingInfo`, turn it into a `FindingModelBase` object with at least two attributes:

* **presence**: Whether the finding is seen  
(present, absent, indeterminate, unknown)
* **change from prior**: How the finding has changed from prior exams  
(unchanged, stable, increased, decreased, new, resolved, no prior)

```python
from findingmodel.tools import create_info_from_name, create_model_stub_from_info

# Create finding info
finding_info = await create_info_from_name("pneumothorax")

# Create a basic model stub with standard presence/change attributes
stub_model = create_model_stub_from_info(finding_info)
print(f"Created model with {len(stub_model.attributes)} attributes")
```

### `add_ids_to_model()`

Generates and adds OIFM IDs to a `FindingModelBase` object and returns it as a `FindingModelFull` object. Note that the `source` parameter refers to the source component of the OIFM ID, which describes the originating organization of the model (e.g., `MGB` for Mass General Brigham and `MSFT` for Microsoft).

```python
from findingmodel.tools import add_ids_to_model, create_model_stub_from_info

# Create a basic model (without IDs)
stub_model = create_model_stub_from_info(finding_info)

# Add OIFM IDs for tracking and standardization
full_model = add_ids_to_model(stub_model, source="MSFT")
print(f"Model ID: {full_model.oifm_id}")
```

### `add_standard_codes_to_model()`

Edits a `FindingModelFull` in place to include some Radlex and SNOMED-CT codes
that correspond to some typical situations.

```python
from findingmodel.tools import add_standard_codes_to_model

# Add standard medical vocabulary codes
add_standard_codes_to_model(full_model)
print("Added standard RadLex and SNOMED-CT codes")
```
