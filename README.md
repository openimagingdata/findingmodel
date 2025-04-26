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

Uses `FindingModelBase`, but adds contains more detailed metadata, especially requiring IDs on models and attributes and allows index codes on multiple levels.

### `FindingInfo`

Information on a finding, including description and synonyms, can add detailed description and citations.

**Properties:**

* `name`: The name of the finding.
* `synonyms`: Alternative names or abbreviations for the finding. *Optional*.
* `description`: A brief description of the finding. *Optional*.
* `detail`: A more detailed description of the finding. *Optional*.
* `citations`: A list of citations or references related to the finding. *Optional*.

## Tools

### `describe_finding_name()`

Takes a finding name and generates a usable description and possibly synonyms (`FindingInfo`) using OpenAI models (requires `OPENAI_API_KEY` to be set to a valid value).

```python
from findingmodel.tools import describe_finding_name

await describe_finding_name("Pneumothorax")

>>> FindingInfo(finding_name="pneumothorax", synonyms=["PTX"], 
  description="Pneumothorax is the...")
```

### `get_detail_on_finding()`

Takes a described finding as above and uses Perplexity to get a lot of possible reference information, possibly including citations (requires `PERPLEXITY_API_KEY` to be set to a valid value).

```python
from findingmodel.tools import get_detail_on_finding

finding = FindingInfo(finding_name="pneumothorax", synonyms=['PTX'],
    description='Pneumothorax is the presence...')

await get_detail_on_finding(finding)

>>> FindingInfo(finding_name='pneumothorax', synonyms=['PTX'], 
 description='Pneumothorax is the...'
 detail='## Pneumothorax\n\n### Appearance on Imaging Studies\n\nA pneumothorax...',
 citations=['https://pubs.rsna.org/doi/full/10.1148/rg.2020200020', 
  'https://ajronline.org/doi/full/10.2214/AJR.17.18721', ...])
```

### `create_finding_model_from_markdown()`

Creates a `FindingModel` from a markdown file or text using OpenAI API.

<!-- TODO: Insert code example  -->

### `create_finding_model_stub_from_finding_info()`

Given even a basic `FindingInfo`, turn it into a `FindingModelBase` object with at least two attributes:

* **presence**: Whether the finding is seen  
(present, absent, indeterminate, unknown)
* **change from prior**: How the finding has changed from prior exams  
(unchanged, stable, increased, decreased, new, resolved, no prior)

<!-- TODO: Insert code example -->

### `add_ids_to_finding_model()`

Generates and adds OIFM IDs to a `FindingModelBase` object and returns it as a `FindingModelFull` object. Note that the `source` parameter refers to the source component of the OIFM ID, which describes the originating organization of the model (e.g., `MGB` for Mass General Brigham and `MSFT` for Microsoft).

### `add_standard_codes_to_finding_model()`

Edits a `FindingModelFull` in place to include some Radlex and SNOMED-CT codes
that correspond to some typical situations.
