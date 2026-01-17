# Phase 4.8: findingmodel-ai Module Reorganization

**Status:** PLANNED
**Depends on:** Phase 4.7 (AI Separation Cleanup)
**Priority:** Low - organizational cleanup, not blocking functionality

## Problem Statement

The `findingmodel-ai` package has grown organically and has some structural issues:

1. **Flat tools/ directory** - 12 Python modules in a single directory mixing different concerns:
   - AI-powered tools using pydantic-ai (9 files)
   - Pure utility code (prompt_template.py - jinja2 only)
   - External API clients (ontology_search.py - httpx/BioOntology)
   - Evaluator infrastructure (evaluators.py - pydantic_evals)

2. **Unclear module boundaries** - Some tools are logically related but scattered:
   - Enrichment: `finding_enrichment.py`, `finding_enrichment_agentic.py`
   - Search: `ontology_search.py`, `ontology_concept_match.py`, `anatomic_location_search.py`, `similar_finding_models.py`
   - Model creation: `finding_description.py`, `markdown_in.py`
   - Model editing: `model_editor.py`

3. **Evaluator placement** - `tools/evaluators.py` contains pydantic_evals evaluators but there's also a separate `evals/` directory at package root

4. **Backward compatibility debt** - Previous phases required re-exports for compatibility; should document what's deprecated

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
│   ├── tools/                # All AI tools (flat)
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

## Proposed Changes

### Option A: Minimal - Document and Clean (Recommended)

**Philosophy:** The current structure works. Focus on documentation and minor cleanup rather than major reorganization that could break imports.

#### A1. Document Module Organization

Create `packages/findingmodel-ai/src/findingmodel_ai/tools/README.md`:

```markdown
# findingmodel_ai.tools

## Module Categories

### AI-Powered Tools (pydantic-ai)
- `finding_description.py` - Generate FindingInfo from name
- `finding_enrichment.py` - Unified finding enrichment
- `finding_enrichment_agentic.py` - Agentic enrichment alternative
- `markdown_in.py` - Create FindingModel from markdown
- `model_editor.py` - Natural language model editing
- `ontology_concept_match.py` - AI categorization of ontology concepts
- `anatomic_location_search.py` - Anatomic location matching
- `similar_finding_models.py` - Find similar models

### External API Clients
- `ontology_search.py` - BioOntology API client (httpx)

### Utilities
- `common.py` - Shared utilities for AI tools
- `prompt_template.py` - Jinja2 template loading
- `evaluators.py` - pydantic_evals evaluator classes

### Template Files
- `prompt_templates/` - Jinja2 templates for AI prompts
```

#### A2. Move evaluators.py to evals/ (Optional)

The `evaluators.py` file contains `Evaluator` subclasses for pydantic_evals. These are used by the eval suites in `evals/`. Consider moving:

```
tools/evaluators.py → evals/evaluators.py
```

Update imports in:
- `evals/*.py` files
- `tests/tools/test_evaluators.py`

#### A3. Update Deprecation Notices

In `tools/__init__.py`, clearly document deprecated re-exports:

```python
# Deprecated re-exports for backward compatibility
# These functions moved to findingmodel core - import from findingmodel.create_stub instead
from findingmodel.create_stub import (
    create_finding_model_stub_from_finding_info,  # DEPRECATED
    create_model_stub_from_info,                   # Use findingmodel.create_model_stub_from_info
)
```

---

### Option B: Subpackage Reorganization (Future)

**Philosophy:** Group related functionality into subpackages for better discoverability.

**Not recommended now** because:
- Breaks many imports
- Requires extensive re-export setup
- Current flat structure is manageable

If pursued later, proposed structure:

```
tools/
├── __init__.py           # Re-exports for backward compatibility
├── enrichment/           # Finding enrichment tools
│   ├── unified.py
│   └── agentic.py
├── search/               # Search and matching tools
│   ├── ontology.py
│   ├── anatomic.py
│   └── similar.py
├── creation/             # Model creation tools
│   ├── description.py
│   └── markdown.py
├── editing/              # Model editing tools
│   └── editor.py
└── _internal/            # Internal utilities
    ├── common.py
    └── prompts.py
```

---

## Acceptance Criteria

### For Option A (Minimal):

- [ ] `tools/README.md` created documenting module organization
- [ ] Deprecation notices added to `tools/__init__.py` for re-exports
- [ ] (Optional) `evaluators.py` moved to `evals/` with import updates
- [ ] All tests pass
- [ ] All checks pass

### For Option B (Future):

- [ ] Subpackage structure created
- [ ] All public APIs re-exported from `tools/__init__.py`
- [ ] All existing imports continue to work
- [ ] Documentation updated
- [ ] All tests pass

---

## Verification

```bash
# Verify all tests pass
task test

# Verify all checks pass
task check

# Verify backward compatibility - these imports should work
python -c "from findingmodel_ai.tools import enrich_finding_unified"
python -c "from findingmodel_ai.tools import create_model_stub_from_info"
python -c "from findingmodel_ai.tools.finding_enrichment import enrich_finding"
```

---

## Files Summary

### Option A - Create:
- `packages/findingmodel-ai/src/findingmodel_ai/tools/README.md`

### Option A - Modify:
- `packages/findingmodel-ai/src/findingmodel_ai/tools/__init__.py` - add deprecation docs

### Option A - Optional Move:
- `tools/evaluators.py` → `evals/evaluators.py`
- Update imports in `evals/*.py` and `tests/tools/test_evaluators.py`

---

## Notes

This phase is **low priority** because:
1. Current structure works and tests pass
2. Import paths are documented in `tools/__init__.py`
3. No user-facing issues reported
4. Focus should be on feature work, not reorganization

Consider implementing only if:
- New developers find the structure confusing
- Adding many new tools that would benefit from grouping
- Implementing features that require clearer module boundaries
