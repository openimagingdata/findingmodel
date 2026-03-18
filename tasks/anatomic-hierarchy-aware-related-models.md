# Anatomic Hierarchy-Aware Related Models

**Status:** Planned (follow-up from Slice 5)

## Problem

`related_models()` uses flat `list_has_any()` matching on `anatomic_location_ids`. Two models must share the *exact same* RID to be considered related. This misses obvious relationships:

- Model A: "pneumonia" tagged with RID1301 (lung)
- Model B: "right lower lobe consolidation" tagged with RID1326 (right lower lobe)

These should be related — right lower lobe is *contained in* lung — but the current flat match never sees them.

`anatomic_location_ids` carries the highest scoring weight (5.0) in `RelatedModelWeights`, so this gap directly suppresses the strongest similarity signal.

## Constraint

`FindingModelIndex` lives in `findingmodel`, which has no dependency on `anatomic-locations`. The containment hierarchy (`get_containment_descendants`, `get_containment_ancestors`) is only available through `AnatomicLocationIndex`.

## Options

### A. Caller-supplied expanded IDs (recommended)

Add an optional parameter to `related_models()`:

```python
async def related_models(
    self,
    model_id: str,
    *,
    expanded_anatomic_ids: list[str] | None = None,  # NEW
    ...
) -> list[tuple[IndexEntry, float]]:
```

When provided, the prefilter query uses `expanded_anatomic_ids` instead of the source model's raw IDs. The caller (in `findingmodel-ai`) does the expansion:

```python
# In findingmodel-ai, which has access to both indexes
descendants = anatomic_index.get_containment_descendants(rid)
expanded = source_ids + [d.id for d in descendants]
results = await fm_index.related_models(model_id, expanded_anatomic_ids=expanded)
```

**Pros:** No new cross-package dependency. `findingmodel` stays pure.
**Cons:** Caller must know to expand. Easy to forget.

### B. Expansion callback

```python
async def related_models(
    self,
    model_id: str,
    *,
    expand_anatomic_ids: Callable[[list[str]], list[str]] | None = None,
    ...
)
```

`related_models()` calls the callback on the source model's anatomic IDs before querying. Default is no expansion (flat match, backward compatible).

**Pros:** Encapsulates the pattern. Hard to forget once wired.
**Cons:** Slightly more complex API.

### C. Store expanded lineage at build time

The maintainer build step expands each model's anatomic locations to include all containment ancestors, storing the full lineage in the `anatomic_location_ids` column. Then flat matching catches hierarchy relationships automatically.

**Pros:** Zero runtime cost. All consumers get hierarchy matching for free.
**Cons:** Increases DB column size. Requires rebuild of all fixtures. Expansion must be rerun whenever the anatomic ontology is updated.

### D. Do nothing in `findingmodel`, expand in `findingmodel-ai` only

The `find_similar_models()` pipeline in `similar.py` already has access to both packages. In `_phase3_search()`, before calling `related_models()`, expand the anatomic IDs and pass them through (requires option A or B to exist).

Same as option A but scoped to the pipeline rather than general-purpose callers.

## Recommendation

**Option A for now, Option C eventually.** Option A is a minimal change to `related_models()` that unblocks hierarchy-aware matching for callers that have access to the anatomic index. Option C is the right long-term answer because it makes hierarchy matching free for all consumers (MCP tools, CLI, etc.), but it requires a build pipeline change and fixture rebuild that should be its own slice.

## Scoring consideration

The scoring function `_score_related()` also needs attention: when comparing expanded IDs (e.g., source has lung + all descendants) against a candidate's raw IDs, the overlap fraction should probably be computed against the candidate's raw IDs, not the expanded set. Otherwise a model tagged with one specific lobe would get a low overlap fraction against a source that expanded to 30+ descendant RIDs. This needs thought.
