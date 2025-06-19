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

## Search Repository

Also assumes a base directory with a `defs` subdirectory.

> Requires installing with `search` optional dependencies—use `uv add findingmode[search]`

```python
import findingmodel as fm
from findingmodel.search_repository import SearchRepository
repo = fm.FindingModelRepository("data") # Model JSON files stored in defs sub-directory

aaa_model = repo.get_model("abdominal aortic aneurysm")
# Finds the file containing the AAA model, loads it, returns as FindingModelFull object

new_model = fm.FindingModelBase(...)
saved_model = repo.save_model(new_model)
# Assigns IDs to the new finding model and saves it to the directory

repo.remove_model(aaa_model.oifm_id)
# Deletes the file associated with the model and deletes the file

for summary in repo.search_summary("heart"):
  print(f"{summary.name} - {summary.id} - {summary.score:.3f}")
# Does hybrid full-text/vector search and returns sorted results
# >  Ventricular diameters - OIFM_MSFT_367670 - 0.016
# >  aortic dissection - OIFM_MSFT_573630 - 0.016
# >  pulmonary embolism - OIFM_MSFT_932618 - 0.016
```

See [example usage in notebook](notebooks/findingmodel_search_repo.ipynb).

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
