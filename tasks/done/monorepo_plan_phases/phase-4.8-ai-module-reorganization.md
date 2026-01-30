# Phase 4.8: findingmodel-ai Module Reorganization

**Status:** ✅ COMPLETE (2026-01-18)
**Depends on:** Phase 4.7 (AI Separation Cleanup)
**Priority:** Medium - improves code organization before first publish

## Problem Statement

The `findingmodel-ai` package has a structural issue: the `tools/` directory is meaninglessly named since the **entire library** is AI tools. The name adds no organizational value.

### Current Issues

1. **Redundant `tools/` layer** - The whole package is AI tools, so `tools/` is meaningless
2. **Flat structure** - 12 Python modules in one directory mixing different concerns:
   - AI-powered agents (9 files)
   - External API client (ontology_search.py)
   - Internal utilities (common.py, prompt_template.py)
   - Evaluator infrastructure (evaluators.py)
3. **Unclear module boundaries** - Related functionality is scattered

### Current Module Categories

| Module | Category | Purpose |
|--------|----------|---------|
| `finding_enrichment.py` | Enrichment | Unified enrichment pipeline (3-stage) |
| `finding_enrichment_agentic.py` | Enrichment | Alternative tool-calling approach |
| `ontology_concept_match.py` | Search | SNOMED/RadLex matching (4-stage) |
| `anatomic_location_search.py` | Search | Anatomic location finding (3-stage) |
| `similar_finding_models.py` | Search | Find similar models to edit |
| `ontology_search.py` | Search | BioOntology API client (httpx) |
| `finding_description.py` | Authoring | Generate FindingInfo from name |
| `markdown_in.py` | Authoring | Create model from markdown |
| `model_editor.py` | Authoring | Natural language/markdown editing |
| `common.py` | Internal | Embedding cache, Tavily client |
| `prompt_template.py` | Internal | Jinja2 template loading |
| `evaluators.py` | Evals | pydantic_evals evaluator classes |

---

## Current Structure

```
packages/findingmodel-ai/
├── evals/                    # Evaluation suites
│   ├── anatomic_search.py
│   ├── finding_description.py
│   ├── markdown_in.py
│   ├── model_editor.py
│   ├── ontology_match.py
│   ├── similar_models.py
│   └── utils.py
├── scripts/                  # CLI scripts
│   ├── edit_finding_model.py
│   ├── enrich_finding.py
│   ├── enrich_findings_batch.py
│   ├── ontology_concept_match.py
│   └── test_unified_enrichment.py
├── src/findingmodel_ai/
│   ├── tools/                # <-- Redundant name
│   │   ├── __init__.py
│   │   ├── anatomic_location_search.py
│   │   ├── common.py
│   │   ├── evaluators.py
│   │   ├── finding_description.py
│   │   ├── finding_enrichment.py
│   │   ├── finding_enrichment_agentic.py
│   │   ├── markdown_in.py
│   │   ├── model_editor.py
│   │   ├── ontology_concept_match.py
│   │   ├── ontology_search.py
│   │   ├── prompt_template.py
│   │   ├── prompt_templates/
│   │   └── similar_finding_models.py
│   ├── __init__.py
│   ├── cli.py
│   └── config.py
└── tests/
```

---

## Proposed Structure

Organize by functional domain, eliminate the redundant `tools/` layer:

```
src/findingmodel_ai/
├── __init__.py              # Package API exports
├── cli.py
├── config.py
│
├── enrichment/              # Finding enrichment workflows
│   ├── __init__.py          # exports: enrich_finding_unified, enrich_finding_agentic
│   ├── unified.py           # Main 3-stage pipeline
│   └── agentic.py           # Alternative tool-calling approach
│
├── search/                  # Search and matching workflows
│   ├── __init__.py          # exports: match_ontology_concepts, find_anatomic_locations, etc.
│   ├── ontology.py          # ontology_concept_match (4-stage pipeline)
│   ├── anatomic.py          # anatomic_location_search (3-stage pipeline)
│   ├── similar.py           # similar_finding_models
│   └── bioontology.py       # BioOntology API client (public)
│
├── authoring/               # Model creation and editing
│   ├── __init__.py          # exports: create_info_from_name, create_model_from_markdown, etc.
│   ├── description.py       # finding_description (name → FindingInfo)
│   ├── markdown_in.py       # markdown → FindingModel
│   └── editor.py            # model_editor (natural language + markdown editing)
│
├── _internal/               # Non-public utilities
│   ├── __init__.py
│   ├── prompts.py           # prompt_template functionality
│   └── common.py            # embedding cache, Tavily client
│
├── evaluators.py            # Keep at top-level (must be in src for installability)
└── prompt_templates/        # Move up from tools/
```

---

## Implementation Steps

### Step 1: Create new directory structure

```bash
mkdir -p packages/findingmodel-ai/src/findingmodel_ai/{enrichment,search,authoring,_internal}
```

### Step 2: Move and rename files

| From | To |
|------|-----|
| `tools/finding_enrichment.py` | `enrichment/unified.py` |
| `tools/finding_enrichment_agentic.py` | `enrichment/agentic.py` |
| `tools/ontology_concept_match.py` | `search/ontology.py` |
| `tools/anatomic_location_search.py` | `search/anatomic.py` |
| `tools/similar_finding_models.py` | `search/similar.py` |
| `tools/ontology_search.py` | `search/bioontology.py` |
| `tools/finding_description.py` | `authoring/description.py` |
| `tools/markdown_in.py` | `authoring/markdown_in.py` |
| `tools/model_editor.py` | `authoring/editor.py` |
| `tools/common.py` | `_internal/common.py` |
| `tools/prompt_template.py` | `_internal/prompts.py` |
| `tools/evaluators.py` | `evaluators.py` (top-level) |
| `tools/prompt_templates/` | `prompt_templates/` (top-level) |

### Step 3: Create subpackage `__init__.py` files

**enrichment/__init__.py:**
```python
from findingmodel_ai.enrichment.unified import enrich_finding, enrich_finding_unified
from findingmodel_ai.enrichment.agentic import enrich_finding_agentic

__all__ = ["enrich_finding", "enrich_finding_unified", "enrich_finding_agentic"]
```

**search/__init__.py:**
```python
from findingmodel_ai.search.ontology import match_ontology_concepts
from findingmodel_ai.search.anatomic import find_anatomic_locations
from findingmodel_ai.search.similar import find_similar_models
from findingmodel_ai.search.bioontology import BioOntologySearchClient

__all__ = [
    "match_ontology_concepts",
    "find_anatomic_locations",
    "find_similar_models",
    "BioOntologySearchClient",
]
```

**authoring/__init__.py:**
```python
from findingmodel_ai.authoring.description import create_info_from_name, add_details_to_info
from findingmodel_ai.authoring.markdown_in import create_model_from_markdown
from findingmodel_ai.authoring.editor import (
    EditResult,
    edit_model_natural_language,
    edit_model_markdown,
    export_model_for_editing,
    assign_real_attribute_ids,
    create_edit_agent,
    create_markdown_edit_agent,
)

__all__ = [
    "create_info_from_name",
    "add_details_to_info",
    "create_model_from_markdown",
    "EditResult",
    "edit_model_natural_language",
    "edit_model_markdown",
    "export_model_for_editing",
    "assign_real_attribute_ids",
    "create_edit_agent",
    "create_markdown_edit_agent",
]
```

**_internal/__init__.py:**
```python
# Internal utilities - not part of public API
```

### Step 4: Update main package `__init__.py`

Re-export everything for flat import access:
```python
from findingmodel_ai.enrichment import enrich_finding_unified, enrich_finding_agentic
from findingmodel_ai.search import match_ontology_concepts, find_anatomic_locations, find_similar_models
from findingmodel_ai.authoring import (
    create_info_from_name,
    add_details_to_info,
    create_model_from_markdown,
    edit_model_natural_language,
    edit_model_markdown,
)
```

### Step 5: Update internal imports in moved files

Each moved file needs import path updates for:
- Cross-module imports (e.g., `from .common import` → `from findingmodel_ai._internal.common import`)
- Prompt template paths

### Step 6: Update all consumer imports

Files to update:
- `cli.py`
- `scripts/*.py` (5 files)
- `evals/*.py` (6 files)
- `tests/*.py` (test files)

### Step 7: Delete old `tools/` directory

After all imports updated and tests pass.

---

## Files Summary

### Create (4 files):
- `enrichment/__init__.py`
- `search/__init__.py`
- `authoring/__init__.py`
- `_internal/__init__.py`

### Move/Rename (13 items):
- 12 Python modules from `tools/` to new locations
- `prompt_templates/` directory

### Update imports (~20 files):
- `cli.py`
- `scripts/edit_finding_model.py`
- `scripts/enrich_finding.py`
- `scripts/enrich_findings_batch.py`
- `scripts/ontology_concept_match.py`
- `scripts/test_unified_enrichment.py`
- `evals/anatomic_search.py`
- `evals/finding_description.py`
- `evals/markdown_in.py`
- `evals/model_editor.py`
- `evals/ontology_match.py`
- `evals/similar_models.py`
- `tests/test_*.py`
- All moved modules (internal imports)

### Delete:
- `tools/` directory (after migration complete)

---

## Acceptance Criteria

- [ ] New directory structure created (enrichment/, search/, authoring/, _internal/)
- [ ] All modules moved to new locations with updated names
- [ ] Subpackage `__init__.py` files export public APIs
- [ ] Main `__init__.py` re-exports for flat access
- [ ] All internal imports updated in moved files
- [ ] All consumer imports updated (cli, scripts, evals, tests)
- [ ] `tools/` directory deleted
- [ ] All tests pass (`task test`)
- [ ] All checks pass (`task check`)

---

## Verification

```bash
# All checks pass
task check

# All tests pass
task test

# CLI still works
uv run fm-ai --help

# New import paths work
python -c "from findingmodel_ai import enrich_finding_unified"
python -c "from findingmodel_ai.enrichment import enrich_finding_unified"
python -c "from findingmodel_ai.search import find_anatomic_locations"
python -c "from findingmodel_ai.authoring import edit_model_natural_language"
python -c "from findingmodel_ai.search import BioOntologySearchClient"
```

---

## Design Decisions

1. **BioOntology client is public** - Users can import `BioOntologySearchClient` directly for custom searches
2. **evaluators.py stays in src** - Must be installable for users to use in their own evals
3. **No backward compatibility for `tools.*`** - This is the first publish, no existing users
4. **_internal/ prefix** - Signals these modules are not public API
