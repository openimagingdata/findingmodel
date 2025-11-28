# Finding Enrichment System - Product Requirements Document

## Implementation Status

**Status**: Testing Complete (Phases 1-8)
**Date**: 2025-11-25

### ✅ Completed Components

- **Phase 1-5**: Core implementation in `src/findingmodel/tools/finding_enrichment.py`
  - Data models (FindingEnrichmentResult, EnrichmentContext, EnrichmentClassification)
  - Index lookup using DuckDBIndex
  - Ontology code search (SNOMED/RadLex) using match_ontology_concepts
  - Pydantic AI agent with structured output for classification tasks
  - Main enrichment orchestration with parallel execution
  - Multi-provider support (OpenAI/Anthropic)

- **Phase 6**: Script interface in `scripts/enrich_finding.py`
  - CLI with argparse (finding name/OIFM ID, --provider flag)
  - Loguru logging enabled for real-time feedback
  - Clean JSON output with model_dump_json(exclude_none=True, indent=2)

- **Phase 7**: Unit tests in `test/test_finding_enrichment.py`
  - 57 tests covering data models, helpers, agent configuration, orchestration
  - Uses TestModel for deterministic agent behavior testing
  - Comprehensive mocking of external dependencies
  - Run with: `task test -- test/test_finding_enrichment.py`

- **Phase 8**: Integration tests in `test/test_finding_enrichment.py`
  - 8 tests marked with `@pytest.mark.callout`
  - Tests pneumonia, fracture, pulmonary nodule, liver lesion, ground-glass opacity
  - Edge cases: ambiguous "mass", unknown finding
  - Performance test validates <30s completion
  - Run with: `task test-full -- test/test_finding_enrichment.py -m callout`

### 🚧 Remaining Work

- **Phase 9**: Manual validation & iteration (not started)
- **Phase 10**: Documentation & memory updates (in progress)

### Usage

```bash
# Basic usage
python scripts/enrich_finding.py "pneumonia"

# With specific provider
python scripts/enrich_finding.py "pneumonia" --provider anthropic

# With OIFM ID
python scripts/enrich_finding.py OIFM_AI_000001
```

---

## Overview

A Pydantic AI-based system to automatically enrich finding models with structured metadata including ontology codes, body regions, etiologies, imaging modalities, subspecialties, and anatomic locations.

## Goals

1. Automate the enrichment of finding models with clinical and imaging metadata
2. Leverage existing tools (ontology search, anatomic location lookup) where possible
3. Produce structured, human-reviewable output for validation and correction
4. Enable future integration into FindingModel schema and database workflows

## Non-Goals (for MVP)

- Direct database updates or FindingModel schema modifications
- Batch processing of multiple findings
- Automated validation or correction of results
- CLI integration (will remain in scripts/ initially)

## User Story

As a finding model curator, I want to provide a finding name or OIFM ID and receive comprehensive metadata about that finding so that I can review, correct, and eventually integrate this data into our finding models.

## Functional Requirements

### Input

The system shall accept either:
- **Finding name** (string): e.g., "pneumonia", "pulmonary nodule"
- **OIFM ID** (string): e.g., "OIFM_AI_000001"

For both input types, the system will look up the finding in the Index database to retrieve existing FindingModel data if available.

### Output Structure

The system shall produce a structured result containing:

```python
class FindingEnrichmentResult(BaseModel):
    finding_name: str
    oifm_id: str | None  # If found in index

    # Ontology codes (finding/disorder codes only, not anatomic)
    snomed_codes: list[IndexCode]  # Zero or more SNOMED CT codes
    radlex_codes: list[IndexCode]  # Zero or more RadLex codes

    # Body regions (from predefined list)
    body_regions: list[Literal["ALL", "Head", "Neck", "Chest", "Breast",
                               "Abdomen", "Arm", "Leg"]]

    # Etiologies (from predefined taxonomy, multiple allowed)
    etiologies: list[str]  # From the 21 defined etiology categories

    # Imaging modalities (from predefined list, multiple allowed)
    modalities: list[Literal["XR", "CT", "MR", "US", "PET", "NM",
                             "MG", "RF", "DSA"]]

    # Subspecialties (from predefined list, multiple allowed)
    subspecialties: list[Literal["AB", "BR", "CA", "CH", "ER", "GI",
                                 "GU", "HN", "IR", "MI", "MK", "NR",
                                 "OB", "OI", "PD", "VI"]]

    # Anatomic locations (multiple allowed)
    anatomic_locations: list[AnatomicLocationResult]  # From existing tool

    # Metadata
    enrichment_timestamp: datetime
    model_provider: str
    model_tier: str
```

### Processing Requirements

1. **Index Lookup**
   - Query the Index database (DuckDB) for the finding by name or OIFM ID
   - Retrieve existing FindingModel data if available (for context)

2. **Ontology Code Search**
   - Use existing `match_ontology_concepts()` function
   - Filter results to finding/disorder codes only (exclude anatomic location codes)
   - Return both SNOMED CT and RadLex codes
   - If no codes found, return empty lists (not an error)

3. **Anatomic Location Lookup**
   - Use existing `find_anatomic_location()` function
   - Support multiple locations (first scaled application of this capability)
   - If no locations found, return empty list (not an error)

4. **Classification Tasks**
   - Use AI agent with structured output for:
     - Body regions
     - Etiologies
     - Modalities
     - Subspecialties
   - Allow multiple values for all categories
   - If agent uncertain, leave empty rather than guessing

5. **Error Handling**
   - Best-effort approach: continue processing even if individual lookups fail
   - Empty results are acceptable (agent leaves blank if uncertain)
   - Fatal errors only for input validation or system failures

### Output Format

- **Primary**: Structured JSON for machine processing
- **Secondary**: Pretty-printed human-readable format (optional flag)
- Output to stdout by default
- Option to save to file

## Technical Architecture

### Components (As Implemented)

1. **Enrichment Agent** (Pydantic AI)
   - **Two-phase design**: Agent classifies → Main function assembles
   - Agent outputs `EnrichmentClassification` (body regions, etiologies, modalities, subspecialties)
   - Main `enrich_finding()` function assembles complete `FindingEnrichmentResult`
   - Uses structured output with Pydantic models
   - Integrates with existing tools via dependency injection (EnrichmentContext)

2. **Agent Tools** (decorated with @agent.tool)
   - `search_ontology_codes()`: Calls search_ontology_codes_for_finding(), returns formatted string
   - `find_anatomic_location()`: Calls find_anatomic_locations() with small tier, returns JSON string

3. **Helper Functions**
   - `search_ontology_codes_for_finding()`: Calls match_ontology_concepts with exclude_anatomical=True, separates SNOMED/RadLex
   - Direct use of `find_anatomic_locations()` from anatomic_location_search module
   - Direct use of `DuckDBIndex` for finding lookup (no wrapper needed)

4. **Script Interface**
   - Location: `scripts/enrich_finding.py`
   - CLI arguments: finding identifier (positional), --provider (optional)
   - Loguru logging explicitly enabled with logger.enable("findingmodel")
   - JSON output only (no pretty-print option implemented)

### Technology Stack

- **Pydantic AI**: Agent framework with structured output
- **DuckDB**: Index database queries
- **Existing Tools**: match_ontology_concepts, find_anatomic_location
- **Multi-provider AI**: Tier-based model selection (base tier for MVP)

### Data Flow (As Implemented)

```
Input (name or OIFM ID)
  ↓
Step 1: Index Lookup (DuckDBIndex.get() + get_full())
  → existing_model: FindingModelFull | None
  ↓
Step 2: Parallel Execution (asyncio.gather with return_exceptions=True)
  ├─ search_ontology_codes_for_finding() → (snomed_codes, radlex_codes)
  └─ find_anatomic_locations() → LocationSearchResponse
  ↓
Step 3: Create EnrichmentContext
  - finding_name, finding_description
  - existing_codes (from existing_model)
  - existing_model reference
  ↓
Step 4: Agent Classification (Pydantic AI with two tools available)
  Agent analyzes context and calls tools as needed:
  - search_ontology_codes() tool (if needed)
  - find_anatomic_location() tool (if needed)
  → EnrichmentClassification
    (body_regions, etiologies, modalities, subspecialties, reasoning)
  ↓
Step 5: Assemble FindingEnrichmentResult
  - Combine agent classifications with parallel results
  - Add metadata (timestamp, provider, tier)
  ↓
Output JSON (model_dump_json with exclude_none=True)
```

**Key Implementation Details:**
- Parallel execution in Step 2 provides results upfront (agent can still call tools if needed)
- Agent has tools available but may not need them if parallel results sufficient
- Two-phase design: Agent focuses on classification, main function assembles complete result
- Best-effort error handling: return_exceptions=True allows partial results

## Validation & Testing

### Unit Tests
- Mock all AI agents and tool calls
- Test data model validation
- Test error handling paths
- Use `ALLOW_MODEL_REQUESTS = False` pattern

### Integration Tests
- Mark with `@pytest.mark.callout`
- Test on 5-10 representative findings:
  - Simple cases: "pneumonia", "fracture"
  - Complex cases: "pulmonary nodule", "liver mass"
  - Edge cases: "mass" (ambiguous), "artifact" (not a pathology)
- Use TestModel/FunctionModel for deterministic testing where possible

### Manual Validation
- Run on 20-30 diverse findings
- Medical expert review of classifications
- Document common error patterns
- Iterate on prompts and logic

## Success Criteria

1. Successfully processes 90%+ of common finding names without errors
2. Ontology code matches are clinically appropriate (human validation)
3. Classifications (regions, etiologies, modalities, subspecialties) are ≥80% accurate
4. Anatomic location lookup works reliably with multiple locations
5. Output is well-structured and human-reviewable
6. Processing time <30 seconds per finding (acceptable for manual workflow)

## Future Enhancements (Post-MVP)

1. Batch processing mode (read from file, process multiple findings)
2. Database update capability (write back to FindingModel)
3. CLI integration (`python -m findingmodel enrich`)
4. Caching of ontology and anatomic lookups
5. Confidence scores for classifications
6. Interactive correction mode
7. Integration with FindingModel schema (add new fields)
8. Evaluation suite with ground truth dataset

## Appendices

### Etiology Taxonomy

- `inflammatory:infectious`
- `inflammatory`
- `neoplastic:benign`
- `neoplastic:malignant`
- `neoplastic:metastatic`
- `neoplastic:potential` (indeterminate lesions, incidentalomas)
- `traumatic-acute` (acute injury)
- `post-traumatic` (sequelae of prior injury)
- `iatrogenic:post-operative`
- `iatrogenic:post-radiation`
- `iatrogenic:device`
- `iatrogenic:medication-related`
- `vascular:ischemic`
- `vascular:hemorrhagic`
- `vascular:thrombotic`
- `degenerative`
- `congenital`
- `metabolic`
- `toxic`
- `mechanical` (obstruction, herniation, torsion)
- `idiopathic`
- `normal-variant`

### Modality Codes

- **XR**: Radiography (plain X-rays)
- **CT**: Computed Tomography
- **MR**: Magnetic Resonance Imaging
- **US**: Ultrasound (including Doppler)
- **PET**: Positron Emission Tomography
- **NM**: Nuclear Medicine (single-photon/SPECT)
- **MG**: Mammography
- **RF**: Fluoroscopy (real-time X-ray)
- **DSA**: Digital Subtraction Angiography

### Subspecialty Codes

- **AB**: Abdominal Radiology
- **BR**: Breast Imaging
- **CA**: Cardiac Imaging
- **CH**: Chest/Thoracic Imaging
- **ER**: Emergency Radiology
- **GI**: Gastrointestinal Radiology
- **GU**: Genitourinary Radiology
- **HN**: Head & Neck Imaging
- **IR**: Interventional Radiology
- **MI**: Molecular Imaging/Nuclear Medicine
- **MK**: Musculoskeletal Radiology
- **NR**: Neuroradiology
- **OB**: OB/Gyn Radiology
- **OI**: Oncologic Imaging
- **PD**: Pediatric Radiology
- **VI**: Vascular Imaging
