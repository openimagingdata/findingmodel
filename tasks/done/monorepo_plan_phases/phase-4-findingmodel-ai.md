# Phase 4: Create findingmodel-ai

**Status:** ⏳ PENDING

**Goal:** Extract AI tools to separate package

## Directory Setup

```bash
mkdir -p packages/findingmodel-ai/src/findingmodel_ai/tools
mkdir -p packages/findingmodel-ai/tests
mkdir -p packages/findingmodel-ai/evals
```

## Code to Move

| From findingmodel | To findingmodel-ai |
|-------------------|-------------------|
| `tools/model_editor.py` | `tools/model_editor.py` |
| `tools/similar_finding_models.py` | `tools/similar_finding_models.py` |
| `tools/ontology_concept_match.py` | `tools/ontology_concept_match.py` |
| `tools/ontology_search.py` | `tools/ontology_search.py` |
| `tools/anatomic_location_search.py` | `tools/anatomic_location_search.py` |
| `tools/create_stub.py` | `tools/create_stub.py` |
| `tools/finding_description.py` | `tools/finding_description.py` |
| `tools/markdown_in.py` | `tools/markdown_in.py` |
| `tools/common.py` (AI parts) | `tools/common.py` |
| `tools/evaluators.py` | `tools/evaluators.py` |
| `tools/prompt_template.py` | `tools/prompt_template.py` |
| `evals/*` | `evals/*` |

## Verification

```bash
uv run --package findingmodel-ai pytest
# All evals still run
```
