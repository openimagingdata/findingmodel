# Faceted Classification Implementation Plan

**Spec Reference:** [findingmodel-facets.md](./findingmodel-facets.md)

**Date:** 2025-11-07

---

## Executive Summary

Implement the 8-facet classification system defined in `findingmodel-facets.md` to enable better search, filtering, and semantic organization of finding models. This requires changes to:

1. Core data models (Pydantic)
2. Database schema (DuckDB)
3. Creation workflows (AI agents)
4. Search/index functionality
5. Existing model retrofitting (8 test models)

---

## Current System Architecture

### Data Models

- **FindingInfo**: Basic finding data (name, synonyms, description, detail, citations)
- **FindingModelBase**: Unidentified model with attributes but no OIFM ID
- **FindingModelFull**: Complete model with OIFM ID, contributors, anatomic_locations, index_codes

### Storage & Index

- **DuckDB backend** with 9 tables:
  - `finding_models` (main table with embeddings)
  - `people`, `organizations` (contributors)
  - `model_people`, `model_organizations` (junction tables)
  - `synonyms`, `tags`, `attributes` (denormalized)
  - `finding_model_json` (full JSON storage)
- **Search**: Hybrid BM25 FTS + HNSW semantic search with RRF fusion
- **No foreign keys**: Manual cleanup pattern before mutations
- **Drop/rebuild indexes**: For batch updates

### Creation Workflow

1. `create_info_from_name()` → generates FindingInfo via AI
2. `create_model_stub_from_info()` → converts to FindingModelBase
3. Model editor (`edit_model_markdown()`) → refines via markdown
4. Save as `.fm.json` files
5. Batch ingest via `update_from_directory()`

### Current Corpus

- Only 8 `.fm.json` files (all in `test/data/`)
- No large production corpus yet
- Easy to retrofit manually if needed

---

## Key Design Decisions

### 1. Data Structure Representation

**Decision: Flat optional fields on FindingModelBase and FindingModelFull**

```python
# Add to both classes
body_region: BodyRegion | None = None  # {primary: str, sub_region: str | None}
subspecialties: list[str] | None = Field(default=None, min_length=1)
etiologies: list[str] | None = None
entity_type: EntityType | None = None  # Enum
applicable_modalities: list[str] | None = Field(default=None, min_length=1)
expected_time_course: ExpectedTimeCourse | None = None  # {duration: str, modifiers: list[str] | None}
age_associations: list[str] | None = None
sex_specificity: SexSpecificity | None = None  # Enum
```

**Rationale:**
- Follows existing pattern (tags, synonyms, anatomic_locations are optional)
- Backward compatible with existing 8 models
- Pydantic validation handles structure/enums
- Can add warnings/recommendations later

**Alternative considered:** Nested `facets: FindingFacets | None` - rejected for being less discoverable

### 2. Database Schema

**Decision: JSON columns in main `finding_models` table**

```sql
ALTER TABLE finding_models ADD COLUMN body_region_primary VARCHAR;
ALTER TABLE finding_models ADD COLUMN body_region_sub VARCHAR;
ALTER TABLE finding_models ADD COLUMN subspecialties JSON;
ALTER TABLE finding_models ADD COLUMN etiologies JSON;
ALTER TABLE finding_models ADD COLUMN entity_type VARCHAR;
ALTER TABLE finding_models ADD COLUMN applicable_modalities JSON;
ALTER TABLE finding_models ADD COLUMN time_course_duration VARCHAR;
ALTER TABLE finding_models ADD COLUMN time_course_modifiers JSON;
ALTER TABLE finding_models ADD COLUMN age_associations JSON;
ALTER TABLE finding_models ADD COLUMN sex_specificity VARCHAR;
```

**Rationale:**
- Simple, similar to how we could have done tags/synonyms
- DuckDB JSON functions are fast for filtering
- No additional join overhead
- Can migrate to junction tables later if we need facet aggregations

**Alternative considered:** Denormalized junction tables (like synonyms/tags pattern) - deferred until we need aggregation counts

### 3. Facet Assignment Strategy

**Decision: AI-driven with three entry points**

1. **During info creation:** Extend `create_info_from_name()` to also generate facets
2. **Standalone utility:** New `assign_facets()` function for retrofitting
3. **Model editor:** Display/allow refinement of facets in markdown

**Rationale:**
- Facets available from the start for new models
- Can retrofit existing models programmatically
- Human oversight via editor
- Modular design (can use pieces independently)

### 4. Search Integration

**Decision: Start with body_region and subspecialties filtering, expand as needed**

Minimum viable search enhancement:
```python
async def search(
    query: str,
    *,
    limit: int = 10,
    tags: Sequence[str] | None = None,
    body_regions: Sequence[str] | None = None,  # NEW
    subspecialties: Sequence[str] | None = None,  # NEW
) -> list[IndexEntry]:
```

Use DuckDB JSON functions:
```sql
WHERE body_region_primary = ANY(?)
AND list_has_any(subspecialties, ?)
```

**Deferred:** Full faceted search UI with aggregation counts, all 8 facets as filters

---

## Implementation Phases

### Phase 1: Core Data Models

**File: `src/findingmodel/facets.py` (new)**

Create Pydantic models for all 8 facets per spec:

1. **BodyRegion** - Pydantic model with `primary` and optional `sub_region`
2. **EntityType** - Enum (Finding, Diagnosis, Grouping/Category, Normal Variant, Measurement, Recommendation)
3. **SexSpecificity** - Enum (Male-specific, Female-specific, Sex-neutral)
4. **ExpectedTimeCourse** - Pydantic model with `duration` enum and optional `modifiers` list
5. **Subspecialty values** - Constants or validation for RSNA subspecialties
6. **Etiology values** - Validation for hierarchical format (e.g., `inflammatory:infectious`)
7. **Modality values** - DICOM modality codes
8. **Age association values** - FDA age groups

**Validation requirements:**
- Use `Annotated` + `Field` for constraints per code_style_conventions
- Hierarchical etiology validation (colon-separated with known prefixes)
- Enum values match spec exactly
- Type hints for all fields

**File: `src/findingmodel/finding_model.py` (modify)**

Add facet fields to:
- `FindingModelBase` (lines ~301-325)
- `FindingModelFull` (lines ~344-398)

All fields optional (`| None`) for backward compatibility.

**File: `src/findingmodel/finding_model.py` (modify markdown methods)**

Update `as_markdown()` in both classes:
- Add facets section after description
- Format hierarchical facets appropriately
- Handle None values gracefully

**Testing:**
- `test/test_facets.py`: Unit tests for facet model validation
- `test/test_finding_model_facets.py`: Serialization/deserialization with facets

---

### Phase 2: Database Schema

**File: `src/findingmodel/index.py` (modify)**

1. **Update `_SCHEMA_STATEMENTS`** (lines 108-190):
   - Add 10 new columns to `finding_models` table definition
   - All columns NULL by default (backward compatible)

2. **Update `DuckDBIndex._add_finding_model()`**:
   - Extract facet data from `FindingModelFull`
   - Flatten `BodyRegion` to two columns
   - Flatten `ExpectedTimeCourse` to two columns
   - Serialize multi-select facets as JSON

3. **Update `DuckDBIndex._to_index_entry()`**:
   - Include facets in `IndexEntry` results
   - Reconstruct nested structures (BodyRegion, ExpectedTimeCourse)

4. **Consider search_text inclusion**:
   - Should facets be included in FTS `search_text`?
   - Body region, subspecialty, etiology might improve discoverability
   - Test both approaches

**Testing:**
- `test/test_index_facets.py`: Database operations with facets
- Test NULL handling for models without facets
- Test round-trip (write model with facets, read back, verify)

---

### Phase 3: Facet Assignment Agent

**File: `src/findingmodel/tools/facet_assignment.py` (new)**

Create Pydantic AI agent for assigning facets:

```python
async def assign_facets(
    finding_name: str,
    finding_description: str,
    anatomic_locations: list[str] | None = None,
    synonyms: list[str] | None = None,
) -> FindingFacets:
    """Assign all 8 facets to a finding using AI.

    Uses clinical knowledge to determine appropriate facet values.
    May reference anatomic_locations for body region validation.
    """
```

**Agent design:**
- Use `Agent` with `output_type=FindingFacets` for structured output
- System prompt includes facet definitions and examples from spec
- Consider few-shot examples for complex facets (etiology, time course)
- Use `TestModel`/`FunctionModel` for testing per `pydantic_ai_best_practices_2025_09`

**Integration points:**
- Called during `create_info_from_name()` flow
- Standalone for retrofitting
- Can be invoked from model editor

**File: `src/findingmodel/tools/finding_description.py` (modify)**

Option A: Extend existing `create_info_from_name()` to also return facets
Option B: New `create_info_with_facets_from_name()` function

Recommendation: Option B for backward compatibility, then deprecate Option A later.

**File: `src/findingmodel/tools/create_stub.py` (modify)**

Update stub creation functions:
- Accept optional facets parameter
- Include facets in generated `FindingModelBase`
- Preserve facets through info→base→full conversions

**Testing:**
- `test/test_facet_assignment.py`: Agent tests with `FunctionModel` for deterministic outputs
- Test edge cases (multisystem findings, indeterminate time course, etc.)
- Integration test with real OpenAI API (@pytest.mark.callout)

---

### Phase 4: Model Editor Integration

**File: `src/findingmodel/tools/model_editor.py` (modify)**

Update markdown export/import:

1. **`export_model_for_editing()`**:
   - Add facets section to markdown output
   - Format as YAML frontmatter or dedicated section
   - Use clear labels matching spec

2. **Markdown parsing**:
   - Parse facets section back to Pydantic models
   - Validate during import
   - Handle missing/invalid facets gracefully

3. **Editing validation**:
   - Ensure facet edits validated through `_basic_edit_validation()`
   - Preserve facet data through edit cycles

**Format decision needed:**
- YAML frontmatter (above description)?
- Dedicated "## Facets" section (after attributes)?
- Inline with description?

Recommendation: Dedicated section for clarity and editability.

**Testing:**
- `test/test_model_editor_facets.py`: Round-trip editing with facets
- Test facet validation errors
- Test facet preservation through edits

---

### Phase 5: Retrofit Existing Models

**Current corpus:** 8 models in `test/data/`:
- `aortic_dissection.fm.json`
- `breast_density.fm.json`
- `breast_malignancy_risk.fm.json`
- `ventricular_diameters.fm.json`
- `thyroid_nodule_codes.fm.json`
- (3 others)

**Approach:**

1. **Script: `scripts/assign_facets_batch.py` (new)**
   ```bash
   uv run python scripts/assign_facets_batch.py \
     --input-dir test/data/defs \
     --output-dir test/data/defs_with_facets \
     --review-report facet_review.md
   ```

2. **Process:**
   - Load each `.fm.json`
   - Call `assign_facets()` with model data
   - Write updated JSON with facets
   - Generate review report showing assignments

3. **Manual review:**
   - Review `facet_review.md`
   - Correct any obvious errors
   - Validate against spec

4. **Commit:**
   - Replace original models with faceted versions
   - Or keep separate temporarily for comparison

**Alternative:** Given small corpus, could assign facets manually using model editor

---

### Phase 6: Search Integration

**File: `src/findingmodel/index.py` (modify)**

Update `DuckDBIndex.search()` method:

1. **Add filter parameters**:
   ```python
   async def search(
       self,
       query: str,
       *,
       limit: int = 10,
       tags: Sequence[str] | None = None,
       body_regions: Sequence[str] | None = None,
       subspecialties: Sequence[str] | None = None,
       # Could add others later: modalities, entity_types, etc.
   ) -> list[IndexEntry]:
   ```

2. **Implement filtering in `_search_fts()` and `_search_semantic()`**:
   ```sql
   WHERE
     (body_region_primary = ANY(?) OR ? IS NULL)
     AND (list_has_any(subspecialties, ?) OR ? IS NULL)
     AND (tag IN (...) OR ? IS NULL)  -- existing tag logic
   ```

3. **Consider indexes**:
   - Start without, measure performance
   - Add if needed: `CREATE INDEX idx_body_region ON finding_models(body_region_primary)`
   - JSON indexes for multi-select: `CREATE INDEX idx_subspecialties ON finding_models USING GIN(subspecialties)`

**File: `src/findingmodel/cli.py` (modify)**

Add facet filter options to search command:
```bash
fm search "lung nodule" \
  --body-region "Chest/Thorax" \
  --subspecialty "Chest/Thoracic Imaging" \
  --modality CT
```

**Optional: Facet aggregations**

Add method for getting facet counts (useful for UIs):
```python
async def get_facet_counts(
    self,
    query: str | None = None,
    current_filters: dict | None = None,
) -> dict[str, dict[str, int]]:
    """Return counts per facet value, optionally filtered."""
```

**Testing:**
- `test/test_search_facets.py`: Search with various facet filters
- Test filter combinations (AND logic)
- Test empty results
- Test with/without query text

---

### Phase 7: Documentation & Polish

**Serena memory: `facet_system_2025` (new)**

Document:
- Facet model locations and structure
- Facet assignment workflow
- Search filtering capabilities
- Database schema for facets
- Examples of faceted models

**Update existing memories:**
- `project_overview`: Mention faceted classification
- `suggested_commands`: Add facet search examples
- `code_style_conventions`: If any facet-specific patterns emerge

**Notebook: `notebooks/faceted_finding_models.ipynb` (new)**

Examples:
- Creating a finding with facets
- Searching by facets
- Analyzing facet distributions
- Retrofitting a model with facets

**CLI help text:**
- Update `--help` for search command
- Document facet filter options
- Show examples

---

## Open Questions & Decisions Needed

### 1. Scope for Initial Implementation

**Question:** Implement all 8 facets at once, or phase them in?

**Options:**
- **Full implementation:** All 8 facets from the start
  - Pro: Complete solution, no migration later
  - Con: More complex, more testing, longer to ship

- **Phased approach:** Start with body_region + subspecialty + entity_type
  - Pro: Ship value sooner, learn from usage
  - Con: Additional migration steps later

**Recommendation:** Need your input on urgency vs completeness

### 2. Required vs Optional Facets

Per spec, validation rules suggest:
- **Required:** Body Region, Subspecialty, Entity Type, Applicable Modalities
- **Recommended:** Etiology, Expected Time Course, Age Association, Sex-Specificity

**Question:** Should we enforce required facets for new models from the start?

**Options:**
- Make all optional initially (backward compatible, easy adoption)
- Enforce required fields for new models only (not retrofitted ones)
- Add warnings for missing recommended fields

**Recommendation:** Start all optional, add warnings later based on usage patterns

### 3. Facet Assignment Quality Assurance

**Question:** How much human review do we want before trusting AI-assigned facets?

**Options:**
- Manual review of all AI assignments (high quality, slow)
- Sample-based review with eval suite (balanced)
- Trust AI with spot-checks (fast, riskier)

**Recommendation:** Given small corpus (8 models), manual review is feasible

### 4. Markdown Format for Facets

**Question:** How should facets appear in markdown export?

**Options:**

A. **YAML frontmatter**:
```markdown
---
body_region:
  primary: Chest/Thorax
  sub_region: Lungs
subspecialties:
  - Chest/Thoracic Imaging
entity_type: Finding
---

# Pulmonary Consolidation

...
```

B. **Dedicated section**:
```markdown
# Pulmonary Consolidation

Description...

## Attributes
...

## Facets

**Body Region:** Chest/Thorax → Lungs
**Subspecialty:** Chest/Thoracic Imaging
**Entity Type:** Finding
...
```

C. **Inline with description**:
```markdown
# Pulmonary Consolidation

**Body Region:** Chest/Thorax → Lungs | **Entity Type:** Finding

Description...
```

**Recommendation:** Need your preference - affects editability and parsing

### 5. Search Performance & Indexing Strategy

**Question:** When should we add database indexes for facets?

**Options:**
- Add indexes upfront (preemptive optimization)
- Measure first, optimize if needed (YAGNI principle)
- Add indexes only for most-used facets (body_region, subspecialty)

**Recommendation:** Start without indexes, add based on performance testing with realistic queries

---

## Success Criteria

- [ ] All 8 facets defined as Pydantic models with validation matching spec
- [ ] Facet fields added to FindingModelBase and FindingModelFull (backward compatible)
- [ ] DuckDB schema includes facet columns
- [ ] Facet assignment agent implemented and tested
- [ ] Model editor supports displaying and editing facets
- [ ] All existing 8 test models have facets assigned
- [ ] Search API supports filtering by at minimum body_region and subspecialty
- [ ] CLI supports facet filters
- [ ] Tests cover facet validation, assignment, search filtering
- [ ] Documentation updated (memories, notebooks, CLI help)
- [ ] No breaking changes to existing code/data

---

## Technical Risks & Mitigations

### Risk: AI facet assignment quality

**Impact:** Incorrect facets reduce search quality, require manual correction

**Mitigation:**
- Build eval suite with gold standard test set
- Manual review of assignments for small corpus
- Validate against known standards (DICOM codes, RSNA subspecialties)
- Allow easy correction via model editor

### Risk: Schema migration issues

**Impact:** DuckDB schema changes could break existing data/code

**Mitigation:**
- All new columns NULL by default (backward compatible)
- Test with existing 8 models before/after
- Keep schema version tracking
- Can rebuild from source JSON files if needed

### Risk: Search performance degradation

**Impact:** Multiple facet filters could slow queries

**Mitigation:**
- Start simple, measure performance
- Add indexes as needed (DuckDB is fast with small datasets)
- Consider denormalization if JSON array filtering is slow
- Profile queries with realistic facet combinations

### Risk: Inconsistent facet values

**Impact:** Typos, variations in facet values reduce filtering effectiveness

**Mitigation:**
- Pydantic validation with enums/constraints
- Reference standard terminologies where possible
- Clear documentation with examples
- Consider autocomplete/suggestions in future UI

---

## Dependencies

**Internal:**
- Pydantic AI agent framework (existing)
- DuckDB index backend (existing)
- Model editor markdown parsing (existing)
- OpenAI API for facet assignment (existing)

**External:**
- DuckDB JSON functions for filtering
- Potentially anatomic location search for body region validation

**Documentation:**
- Facet spec in `tasks/findingmodel-facets.md`
- RSNA subspecialty definitions
- DICOM modality codes
- FDA age group classifications

---

## Next Steps

To proceed, need decisions on:

1. **Scope:** All 8 facets at once, or phased implementation?
2. **Required vs optional:** Enforce required facets for new models?
3. **Markdown format:** Frontmatter, dedicated section, or inline?
4. **Starting phase:** Which phase should we begin with?

Once these are decided, can start implementation with Phase 1 (data models).
