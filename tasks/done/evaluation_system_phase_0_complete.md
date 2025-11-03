# Evaluation System - Next Steps

**Status:** Phase 0 COMPLETE ✅ | Phase 1 Ready to Implement

**Created:** 2025-10-24
**Phase 0 Completed:** 2025-10-25

**Priority:** Medium (foundation complete, ready to scale to additional agents)

---

## Context

### What's Been Completed

The evaluation system foundation is in excellent shape:

✅ **Eval/Test Separation Complete** (`separate_evals_from_tests_plan.md`)
- Three-tier testing structure established (unit, integration, evals)
- `evals/` directory at project root (separate from `test/`)
- Pytest no longer discovers eval suites
- Clear documentation and task commands

✅ **Base Framework Complete** (`refactor_model_editor_evals.md` Phases 1-2)
- Reusable evaluator library (`evals/base.py`) - 5 generic evaluators
- Shared utilities (`evals/utils.py`)
- 25 unit tests for evaluators (`test/test_base_evaluators.py`)
- Hybrid scoring pattern established (strict + partial credit)

✅ **First Eval Suite Complete** (`refactor_model_editor_evals.md` Phase 2)
- model_editor evaluation suite operational (12 cases, 5 evaluators)
- Dataset.evaluate() pattern implemented
- Comprehensive documentation

✅ **Logfire Integration Added** (`refactor_model_editor_evals.md` Phase 3)
- Observability instrumentation implemented
- Configuration via .env
- Graceful degradation without token

### The Problem

**Phase 3 implementation violates best practices and creates technical debt:**

1. ❌ **Logfire configuration duplicated per module** - Should be at package level (Python logging best practice)
2. ❌ **Manual spans are redundant** - Pydantic Evals auto-instruments `Dataset.evaluate()`
3. ❌ **Copy/paste required for new eval suites** - ~80 lines of boilerplate per suite (400+ lines for 5 suites)
4. ❌ **Violates DRY principle** - Same configuration logic repeated everywhere

**Research findings:**
- Pydantic Evals docs: "All you need to do is configure Logfire via `logfire.configure`" - automatic instrumentation handles spans
- Python logging best practices: "Configure at application level" not per-module
- Official pattern: Configure once, use everywhere

**Impact:** This blocks efficient scaling to 5 remaining agents. Must fix before expanding coverage.

---

## Phase 0: Fix Logfire Observability Architecture ✅ COMPLETE

**Goal:** Eliminate code duplication, follow best practices, enable DRY eval development

**Status:** COMPLETE (2025-10-25)

**Actual Impact:**
- Removed ~160 lines from model_editor.py
- Fixed latent TestModel bug (2 test failures)
- Consolidated documentation (eliminated ~250 lines duplication)
- All 325 tests passing

---

### Task 0.1: Create Centralized Logfire Configuration

**Create:** `evals/__init__.py`

**Content:**
```python
"""Evaluation suites for findingmodel agents.

This module configures Logfire observability once for the entire package.
Individual eval modules require NO Logfire code - automatic instrumentation
is provided by:
- Pydantic Evals: Dataset.evaluate() creates root + per-case spans
- Pydantic AI: logfire.instrument_pydantic_ai() traces agent/model/tool calls

Configuration is read from .env via findingmodel.config.settings.

See: https://ai.pydantic.dev/evals/#integration-with-logfire
"""

import logfire
from logfire import ConsoleOptions
from findingmodel.config import settings

# Configure Logfire once for entire evals package
# Follows Python logging best practice: configure at package level
logfire.configure(
    token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
    send_to_logfire=False if settings.disable_send_to_logfire else "if-token-present",
    console=ConsoleOptions(
        colors="auto",
        min_log_level="debug",
    )
    if settings.logfire_verbose
    else False,
)

# Instrument Pydantic AI once for automatic agent/model/tool tracing
logfire.instrument_pydantic_ai()

__all__ = []  # No public exports - configuration only
```

**Acceptance Criteria:**
- [x] File exists at `evals/__init__.py`
- [x] Can import evals package without errors
- [x] Configuration logic identical to current model_editor.py (lines 62-74)
- [x] No runtime errors or warnings

**Status:** ✅ COMPLETE

**Dependencies:** None

**Why this works:** Python guarantees parent package `__init__.py` is executed before any child module imports. Configuration happens automatically on first import.

---

### Task 0.2: Refactor evals/model_editor.py

**Goal:** Remove all manual Logfire code, rely on automatic instrumentation

**Changes:**

1. **Remove Logfire imports** (lines 47-49)
   - Delete: `import logfire`
   - Delete: `from logfire import ConsoleOptions`

2. **Remove configuration** (lines 55, 62-74)
   - Delete: `from findingmodel.config import settings` (Logfire-related import)
   - Delete: Entire logfire.configure() call
   - Delete: logfire.instrument_pydantic_ai() call

3. **Remove metadata lookup helpers** (lines 717-731)
   - Delete: `_make_metadata_lookup_key()` function
   - Delete: `_case_metadata_map` creation (lines 833-835)
   - These were only used for manual logging

4. **Simplify run_model_editor_task()** (lines 734-800)

   **Before (~67 lines with spans/logging):**
   ```python
   async def run_model_editor_task(input_data: ModelEditorInput) -> ModelEditorActualOutput:
       lookup_key = _make_metadata_lookup_key(input_data)
       case_name, should_succeed = _case_metadata_map.get(lookup_key, ("unknown", True))

       with logfire.span("eval_case {name}", name=case_name, ...):
           if settings.logfire_verbose:
               logfire.debug(...)

           try:
               model = FindingModelFull.model_validate_json(input_data.model_json)
               with logfire.span("model_editor_execution", ...):
                   if input_data.edit_type == "natural_language":
                       result = await model_editor.edit_model_natural_language(...)
                   elif input_data.edit_type == "markdown":
                       result = await model_editor.edit_model_markdown(...)

               logfire.info("Case completed", ...)
               return ModelEditorActualOutput(...)
           except Exception as e:
               logfire.error("Case execution failed", ...)
               return ModelEditorActualOutput(..., error=str(e))
   ```

   **After (~20 lines, no spans/logging):**
   ```python
   async def run_model_editor_task(input_data: ModelEditorInput) -> ModelEditorActualOutput:
       """Execute a single model_editor evaluation case.

       Dataset.evaluate() automatically creates spans and captures inputs/outputs.
       Pydantic AI instrumentation captures agent/model/tool calls.
       No manual Logfire code needed.
       """
       try:
           model = FindingModelFull.model_validate_json(input_data.model_json)

           if input_data.edit_type == "natural_language":
               result = await model_editor.edit_model_natural_language(model, input_data.command)
           elif input_data.edit_type == "markdown":
               result = await model_editor.edit_model_markdown(model, input_data.command)
           else:
               raise ValueError(f"Unknown edit_type: {input_data.edit_type}")

           return ModelEditorActualOutput(
               model=result.model,
               changes=result.changes,
               rejections=result.rejections,
           )
       except Exception as e:
           # Return error in output for evaluation
           model = FindingModelFull.model_validate_json(input_data.model_json)
           return ModelEditorActualOutput(model=model, changes=[], rejections=[], error=str(e))
   ```

5. **Simplify run_model_editor_evals()** (lines 915-965)

   **Before (~50 lines with spans/logging):**
   ```python
   async def run_model_editor_evals() -> EvaluationReport[...]:
       with logfire.span("model_editor_eval_suite", ...):
           logfire.info("Starting model_editor evaluation suite", ...)

           report = await model_editor_dataset.evaluate(run_model_editor_task)

           all_scores = [...]
           overall_score = sum(all_scores) / len(all_scores)

           logfire.info("Evaluation suite completed", ...)
           return report
   ```

   **After (~10 lines, no spans/logging):**
   ```python
   async def run_model_editor_evals() -> EvaluationReport[
       ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput
   ]:
       """Run model_editor evaluation suite.

       Dataset.evaluate() automatically creates evaluation spans and captures
       all inputs, outputs, and scores for visualization in Logfire.
       """
       report = await model_editor_dataset.evaluate(run_model_editor_task)
       return report
   ```

6. **Update module docstring** (lines 19-42)
   - Remove detailed Logfire setup instructions
   - Replace with: "Logfire observability configured automatically in evals/__init__.py"
   - Add reference to automatic instrumentation

7. **Keep __main__ block unchanged**
   - Console output still needs overall_score calculation
   - This is for human-readable output, not Logfire

**Acceptance Criteria:**
- [x] No `import logfire` statements (except in removed sections)
- [x] No `logfire.configure()` call
- [x] No manual `logfire.span()` calls
- [x] No manual `logfire.info/debug/error()` calls
- [x] run_model_editor_task() is ~20 lines (from ~67)
- [x] run_model_editor_evals() is ~10 lines (from ~50)
- [x] Total line reduction: ~160 lines
- [x] File has no linting errors
- [x] File imports successfully

**Status:** ✅ COMPLETE

**Dependencies:** Task 0.1 complete

**Rationale:** Automatic instrumentation from Pydantic Evals + Pydantic AI captures everything manual spans were capturing. Manual spans add no value and violate DRY.

---

### Task 0.3: Test Functionality

**Goal:** Verify refactoring doesn't break eval functionality

**Test Steps:**

1. **Capture baseline (optional - if not already running):**
   ```bash
   # Before refactoring (if possible)
   task evals:model_editor > baseline_scores.txt 2>&1
   grep "OVERALL SCORE" baseline_scores.txt
   ```

2. **After refactoring:**
   ```bash
   # Should produce identical scores
   task evals:model_editor > refactored_scores.txt 2>&1
   grep "OVERALL SCORE" refactored_scores.txt

   # Should see same score (within floating point tolerance)
   # Example: OVERALL SCORE: 0.89
   ```

3. **Import test:**
   ```python
   # Verify package import triggers configuration
   import evals.model_editor
   # No errors = success
   ```

4. **Run unit tests:**
   ```bash
   # Existing tests should still pass
   task test
   ```

5. **Logfire trace verification (if token available):**
   - Set LOGFIRE_TOKEN in .env
   - Run: `task evals:model_editor`
   - Check Logfire UI for traces
   - Verify presence of:
     - Evaluation root span (automatic)
     - Per-case spans (automatic)
     - Agent run spans (automatic)
     - Model call spans (automatic)

**Acceptance Criteria:**
- [x] Overall scores identical (or within 0.01 tolerance)
- [x] No import errors
- [x] No runtime errors
- [x] Existing unit tests pass (`task test`) - All 325 tests passing
- [x] Console output format unchanged (for human readability)

**Status:** ✅ COMPLETE

**Notes:**
- Fixed latent TestModel bug during testing (tests passed Pydantic models instead of dicts to `custom_output_args`)
- Bug only manifested when Logfire instrumentation was enabled
- Fixed by adding `.model_dump()` calls in 2 test files

**Dependencies:** Tasks 0.1, 0.2 complete

---

### Task 0.4: Update Documentation

**Goal:** Ensure documentation reflects automatic instrumentation pattern

**Files to Update:**

1. **evals/CLAUDE.md**

   **Current state:** Has section "## Logfire Integration (Phase 3 - Planned)"

   **Changes:**
   - Replace with: "## Logfire Integration (Automatic)"
   - Content:
     ```markdown
     ## Logfire Integration (Automatic)

     Logfire observability is configured automatically for all evaluation suites.

     ### How It Works

     - **Configuration:** Centralized in `evals/__init__.py`
     - **Automatic instrumentation:** Pydantic Evals + Pydantic AI
     - **No code required:** Individual eval modules need ZERO Logfire code

     ### What Gets Traced Automatically

     When you run an eval suite, Logfire captures:
     - Evaluation root span (from Dataset.evaluate())
     - Per-case execution spans (from Dataset.evaluate())
     - Agent run spans (from logfire.instrument_pydantic_ai())
     - Model call spans with prompts/completions
     - Tool execution spans
     - Evaluation scores and results

     ### Setup (Optional)

     Logfire works in local-only mode by default. For cloud tracing:
     1. Create account at https://logfire.pydantic.dev/
     2. Get write token from dashboard
     3. Add to .env: `LOGFIRE_TOKEN=your_token_here`
     4. Run evals normally - traces automatically appear in UI

     ### Environment Variables

     - `LOGFIRE_TOKEN` - Write token (optional, enables cloud tracing)
     - `DISABLE_SEND_TO_LOGFIRE` - Force local-only mode (default: false)
     - `LOGFIRE_VERBOSE` - Enable console logging (default: false)

     ### For New Eval Suites

     **You need ZERO Logfire code.** Just write your eval logic:
     ```python
     # No logfire imports needed
     # No configuration needed
     # No manual spans needed

     dataset = Dataset(cases=..., evaluators=...)

     async def run_tool_name_evals():
         report = await dataset.evaluate(task_function)
         return report
     ```

     Observability happens automatically via package-level configuration.

     See: https://ai.pydantic.dev/evals/#integration-with-logfire
     ```

2. **evals/README.md**

   **Current state:** Has "## Observability with Logfire" section

   **Changes:**
   - Simplify to emphasize automatic nature
   - Remove manual setup instructions
   - Add: "Configured automatically - no setup needed per eval suite"

3. **evals/evals_guide.md**

   **Changes:**
   - Remove any manual Logfire setup examples
   - Add section: "Observability is automatic via evals/__init__.py"

4. **docs/logfire_observability_guide.md**

   **Changes:**
   - Add section: "Package-Level Configuration Pattern"
   - Update examples to show automatic instrumentation
   - Add note: "Manual spans are unnecessary - shown for reference only"
   - Emphasize Dataset.evaluate() automatic instrumentation

**Acceptance Criteria:**
- [x] evals/CLAUDE.md replaced with lightweight pointer (~52 lines, -250 lines)
- [x] evals/README.md unchanged (already correct)
- [x] evals/evals_guide.md updated with TestModel warning (+19 lines)
- [x] All documentation consistent with automatic instrumentation approach
- [x] No references to manual span creation in examples
- [x] Documentation consolidation: SSOT principles applied

**Status:** ✅ COMPLETE

**Notes:**
- Applied excellent documentation hygiene
- CLAUDE.md now points to `.serena/memories/agent_evaluation_best_practices_2025.md` as authoritative source
- Eliminated ~250 lines of duplication across 4 documents
- Added TestModel bug warning to evals_guide.md

**Dependencies:** Tasks 0.1, 0.2, 0.3 complete

---

### Task 0.5: Update Serena Memory

**Goal:** Document new pattern for future reference by AI agents

**File:** `.serena/memories/agent_evaluation_best_practices_2025.md`

**Changes:**

1. Update "Logfire Integration" section:
   ```markdown
   ## Logfire Integration (Automatic)

   Logfire observability is configured ONCE at package level in `evals/__init__.py`.
   Individual eval modules require ZERO Logfire code.

   ### What Happens Automatically

   When `evals/__init__.py` is imported (automatically on first eval module import):
   1. logfire.configure() is called with settings from .env
   2. logfire.instrument_pydantic_ai() enables automatic agent tracing

   When Dataset.evaluate() is called:
   1. Root span created for evaluation suite (automatic)
   2. Per-case spans created for each case (automatic)
   3. All Pydantic AI spans captured (agent runs, model calls, tools)
   4. Inputs, outputs, and scores recorded (automatic)

   ### For New Eval Suites

   NO Logfire code required. Just define:
   - Data models
   - Evaluators
   - Cases
   - Dataset
   - Main function calling dataset.evaluate()

   Observability happens automatically via package-level configuration.
   ```

2. Add to "Anti-Patterns to Avoid":
   ```markdown
   ❌ Adding logfire.configure() to individual eval modules (configure once in __init__.py)
   ❌ Creating manual spans in eval modules (Dataset.evaluate() does this automatically)
   ❌ Manual logging in task functions (automatic instrumentation captures everything)
   ```

3. Update "Creating New Eval Suites" template to remove all Logfire code

**Acceptance Criteria:**
- [x] Memory updated with automatic instrumentation pattern
- [x] Anti-patterns section includes manual Logfire code
- [x] Template shows zero Logfire code for new eval suites
- [x] Clear explanation of package-level configuration
- [x] All unique content from CLAUDE.md migrated (+155 lines)
- [x] TestModel bug documented in anti-patterns and lessons learned

**Status:** ✅ COMPLETE

**Notes:**
- Absorbed all unique content from CLAUDE.md (file structure, naming conventions, import patterns, etc.)
- Added TestModel bug as Lesson #8 with full explanation
- Memory is now authoritative source for AI agents
- Updated Phase 0 completion status

**Dependencies:** Tasks 0.1, 0.2, 0.3, 0.4 complete

---

### Phase 0 Success Criteria

**Must Have:**
- [x] evals/__init__.py exists with centralized configuration
- [x] evals/model_editor.py has NO manual Logfire code (~160 lines removed)
- [x] `task evals:model_editor` produces identical scores
- [x] No import or runtime errors
- [x] All existing tests pass
- [x] Documentation updated (CLAUDE.md, README.md, guide, observability guide)
- [x] Serena memory updated

**Benefits Achieved:**
- ✅ DRY principle maintained (configuration in ONE place)
- ✅ Follows Python/Pydantic best practices
- ✅ Zero Logfire code required for new eval suites
- ✅ Prevents ~400 lines of duplication across 5 future suites
- ✅ Automatic instrumentation leveraged fully
- ✅ Clean architecture for scaling

**Next:** Phase 1 - Expand to 5 remaining agents

---

### Phase 0 Completion Summary

**Completed:** 2025-10-25

**What Was Delivered:**

1. **Centralized Logfire Configuration** (`evals/__init__.py`)
   - Package-level configuration following Python best practices
   - Zero Logfire code required in individual eval modules
   - Automatic instrumentation for all evaluation suites

2. **Refactored model_editor.py**
   - Removed ~160 lines of manual Logfire code
   - Simplified run_model_editor_task() from ~67 to ~20 lines
   - Simplified run_model_editor_evals() from ~50 to ~10 lines
   - No functional changes - all 12 cases still passing

3. **Fixed TestModel Bug**
   - Discovered latent bug in test code (2 test failures)
   - Root cause: Passing Pydantic models to TestModel instead of dicts
   - Only manifested with Logfire instrumentation enabled
   - Fixed in test/test_model_editor.py and test/test_ontology_search.py
   - All 325 tests now passing

4. **Documentation Consolidation**
   - Applied SSOT (Single Source of Truth) principles
   - CLAUDE.md reduced from ~302 to ~52 lines (lightweight pointer)
   - agent_evaluation_best_practices_2025.md expanded by +155 lines (absorbed CLAUDE.md)
   - evals_guide.md updated with TestModel warning (+19 lines)
   - README.md unchanged (already correct)
   - Eliminated ~250 lines of duplication across 4 documents

5. **Updated Serena Memory**
   - Added all unique content from CLAUDE.md
   - Documented TestModel bug (anti-patterns + lessons learned)
   - Updated Phase 0 completion status
   - Memory is now authoritative reference for AI agents

**Metrics:**
- Lines removed from model_editor.py: ~160
- Lines eliminated via documentation consolidation: ~250
- Total reduction: ~410 lines
- Tests passing: 325/325 (100%)
- Documentation files updated: 4

**Benefits Achieved:**
- ✅ DRY principle maintained (configuration in ONE place)
- ✅ Follows Python/Pydantic best practices
- ✅ Zero Logfire code required for new eval suites
- ✅ Prevents ~400 lines of duplication across 5 future suites
- ✅ Automatic instrumentation leveraged fully
- ✅ Clean architecture for scaling
- ✅ Excellent documentation hygiene
- ✅ Fixed latent bug before it spread

**Unexpected Discoveries:**
- TestModel bug that only manifested with instrumentation
- Opportunity for documentation consolidation
- Importance of `.model_dump()` when using TestModel

---

## Phase 1: Expand Agent Evaluation Coverage

**Goal:** Create comprehensive evaluation suites for remaining 5 agents

**Prerequisites:** Phase 0 complete (Logfire refactoring)

**Current State:** 1 eval suite operational (model_editor with 12 cases, 5 evaluators)

**Target:** 6 total eval suites covering all AI agents

---

### Overview

Create eval suites in priority order:

1. **anatomic_location_search** - Two-agent architecture (20+ cases)
2. **ontology_concept_match** - Multi-backend matching (20+ cases)
3. **markdown_in** - Parsing accuracy (12+ cases)
4. **similar_finding_models** - Similarity/ranking (15+ cases)
5. **finding_description** - LLM-generated content quality (15+ cases)

Each suite follows the established pattern from model_editor:
- Focused evaluators (3-5 per suite)
- Hybrid scoring (strict + partial credit)
- Dataset.evaluate() pattern
- ZERO Logfire code (automatic via evals/__init__.py)
- Comprehensive documentation

---

### Agent 1: anatomic_location_search

**Priority:** #1 (tests framework flexibility with two-agent architecture)

**File:** `evals/anatomic_search.py`

**Agent Details:**
- Two-agent architecture: search agent + matching agent
- Backend support: MongoDB and DuckDB
- Handles anatomic hierarchy and relationships
- Located at: `src/findingmodel/tools/anatomic_location_search.py`

#### Test Case Categories (Minimum 20 cases)

**Success Cases (10):**
- Common anatomic terms (e.g., "heart", "lung", "liver")
- Specific locations (e.g., "left ventricle", "right upper lobe")
- Hierarchical relationships (e.g., "mitral valve" should relate to "heart")
- Synonyms and variations (e.g., "pulmonary artery" vs "PA")
- Anatomic systems (e.g., "cardiovascular system")

**Rejection/Error Cases (5):**
- Non-anatomic terms (e.g., "diabetes", "protocol")
- Ambiguous terms (e.g., "trunk" - could be body or vascular)
- Misspellings with no close matches
- Empty or very short queries
- Special characters and invalid input

**Edge Cases (5):**
- Very long anatomic names
- Terms with multiple valid interpretations
- Case sensitivity variations
- Terms with accents or special characters
- Composite anatomic locations (e.g., "left anterior descending artery")

**Backend-Specific Cases:**
- MongoDB backend available vs unavailable
- Fallback to DuckDB when MongoDB fails
- Performance with large result sets
- Handling of missing data in either backend

#### Evaluators to Create (4 focused)

1. **SearchAccuracyEvaluator** (strict)
   - Returns 1.0 if correct anatomic location found in top-K results
   - Returns 0.0 otherwise
   - Non-negotiable: must find the right location

2. **HierarchyEvaluator** (partial credit)
   - Verifies hierarchical relationships are preserved
   - Score = proportion of expected hierarchy present
   - Example: "left ventricle" should have "heart" as parent

3. **BackendFallbackEvaluator** (hybrid)
   - Strict: Fallback must occur when MongoDB unavailable (0.0 or 1.0)
   - Partial: Quality of results from fallback backend

4. **RankingQualityEvaluator** (partial credit)
   - Assesses result ranking using MRR (Mean Reciprocal Rank)
   - Score = 1/rank of first correct result
   - Most relevant result should rank first

#### Implementation

```python
"""Evaluation suite for anatomic_location_search agent.

NO Logfire configuration needed - automatic instrumentation via evals/__init__.py.
"""

from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from evals.base import ExactMatchEvaluator, KeywordMatchEvaluator
from findingmodel.tools import anatomic_location_search

# Data models
class AnatomicSearchInput(BaseModel):
    query: str
    backend: str = "mongodb"  # or "duckdb"

class AnatomicSearchExpected(BaseModel):
    should_succeed: bool
    expected_location_name: str | None = None
    expected_hierarchy: list[str] = []

class AnatomicSearchOutput(BaseModel):
    locations: list[dict]
    backend_used: str
    error: str | None = None

# Agent-specific evaluators (4 focused)
class SearchAccuracyEvaluator(Evaluator[AnatomicSearchInput, AnatomicSearchOutput, AnatomicSearchExpected]):
    """Check if correct location found in results (strict)."""
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Implementation
        ...

class HierarchyEvaluator(Evaluator[...]):
    """Verify hierarchical relationships (partial credit)."""
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Implementation
        ...

class BackendFallbackEvaluator(Evaluator[...]):
    """Test MongoDB → DuckDB fallback (hybrid)."""
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Implementation
        ...

class RankingQualityEvaluator(Evaluator[...]):
    """Assess result ranking quality using MRR (partial credit)."""
    def evaluate(self, ctx: EvaluatorContext[...]) -> float:
        # Implementation
        ...

# Cases (20+)
def create_successful_cases() -> list[Case]:
    """Success cases where search should find correct location."""
    return [
        Case(
            name="common_term_heart",
            inputs=AnatomicSearchInput(query="heart"),
            expected_output=AnatomicSearchExpected(
                should_succeed=True,
                expected_location_name="Heart",
            ),
        ),
        # ... 9 more success cases
    ]

def create_rejection_cases() -> list[Case]:
    """Cases where search should reject or return empty."""
    return [
        Case(
            name="non_anatomic_diabetes",
            inputs=AnatomicSearchInput(query="diabetes"),
            expected_output=AnatomicSearchExpected(should_succeed=False),
        ),
        # ... 4 more rejection cases
    ]

def create_edge_cases() -> list[Case]:
    """Boundary conditions and unusual inputs."""
    return [
        # ... 5 edge cases
    ]

# Dataset
all_cases = create_successful_cases() + create_rejection_cases() + create_edge_cases()
evaluators = [
    SearchAccuracyEvaluator(),
    HierarchyEvaluator(),
    BackendFallbackEvaluator(),
    RankingQualityEvaluator(),
]
anatomic_search_dataset = Dataset(cases=all_cases, evaluators=evaluators)

# Task function
async def run_anatomic_search_task(input_data: AnatomicSearchInput) -> AnatomicSearchOutput:
    """Execute anatomic location search case.

    No manual Logfire code needed - automatic instrumentation captures everything.
    """
    try:
        result = await anatomic_location_search.search(
            query=input_data.query,
            backend=input_data.backend,
        )
        return AnatomicSearchOutput(
            locations=result.locations,
            backend_used=result.backend_used,
        )
    except Exception as e:
        return AnatomicSearchOutput(locations=[], backend_used="error", error=str(e))

# Main eval function
async def run_anatomic_search_evals() -> EvaluationReport[...]:
    """Run anatomic_location_search evaluation suite.

    Dataset.evaluate() automatically creates spans and captures results.
    """
    report = await anatomic_search_dataset.evaluate(run_anatomic_search_task)
    return report

# Standalone execution
if __name__ == "__main__":
    import asyncio

    async def main():
        print("\nRunning anatomic_location_search evaluation suite...")
        print("=" * 80)

        report = await run_anatomic_search_evals()

        print("\n" + "=" * 80)
        print("ANATOMIC LOCATION SEARCH EVALUATION RESULTS")
        print("=" * 80 + "\n")
        report.print(include_input=False, include_output=True)

        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        print(f"\nOVERALL SCORE: {overall_score:.2f}")
        print("=" * 80 + "\n")

    asyncio.run(main())
```

#### Taskfile Integration

Add to `Taskfile.yml`:
```yaml
  evals:anatomic_search:
    desc: "Run anatomic_location_search evaluation suite"
    cmds:
      - echo "Running anatomic_location_search evaluations..."
      - uv run python -m evals.anatomic_search
    silent: true
```

Update `evals` task:
```yaml
  evals:
    desc: "Run all agent evaluation suites"
    cmds:
      - echo "Running all agent evaluations..."
      - uv run python -m evals.model_editor
      - uv run python -m evals.anatomic_search  # NEW
```

#### Acceptance Criteria

- [ ] evals/anatomic_search.py exists with 20+ cases
- [ ] 4 focused evaluators implemented
- [ ] ZERO Logfire code in eval module (automatic via __init__.py)
- [ ] Success rate ≥85% for common terms
- [ ] Backend fallback works correctly
- [ ] Tests both MongoDB and DuckDB backends
- [ ] Can run: `task evals:anatomic_search`
- [ ] Can run: `python -m evals.anatomic_search`
- [ ] Documentation added to evals_guide.md

---

### Agent 2: ontology_concept_match

**Priority:** #2 (similar to anatomic search but more complex)

**File:** `evals/ontology_match.py`

**Agent Details:**
- Multi-backend: BioOntology API and DuckDB
- Handles concept matching and ranking
- Supports multiple ontologies (RADLEX, SNOMEDCT, etc.)
- Located at: `src/findingmodel/tools/ontology_concept_match.py`

#### Test Case Categories (Minimum 20 cases)

**Success Cases (10):**
- Common radiological concepts (e.g., "pneumothorax", "fracture")
- Specific pathologies (e.g., "Type A aortic dissection")
- Anatomy + pathology combinations
- Modality-specific terms (e.g., "FLAIR signal")
- Different ontologies (RADLEX vs SNOMEDCT)

**Rejection/Error Cases (5):**
- Non-medical terms
- Extremely rare or obsolete terms
- Ambiguous concepts without context
- Invalid ontology names

**Edge Cases (5):**
- Synonyms and abbreviations (e.g., "MI" vs "myocardial infarction")
- Similar but distinct concepts
- Case variations
- Terms with special characters
- Very long or compound concept names

**Ranking Cases:**
- Multiple valid matches (best match should rank first)
- Partial matches vs exact matches
- Context-dependent rankings

**Backend-Specific Cases:**
- BioOntology API availability
- Fallback to DuckDB when API unavailable
- Consistency between backends
- Rate limiting handling

#### Evaluators to Create (5 focused)

1. **ConceptMatchAccuracyEvaluator** (strict)
   - Returns 1.0 if correct concept found in top-K
   - Returns 0.0 otherwise

2. **RankingQualityEvaluator** (partial credit)
   - Uses NDCG (Normalized Discounted Cumulative Gain) or MRR
   - Assesses ranking quality of results

3. **BackendConsistencyEvaluator** (partial credit)
   - Compares results across backends
   - Score = overlap between backend results

4. **SynonymHandlingEvaluator** (hybrid)
   - Strict: Must handle common synonyms
   - Partial: Quality of synonym matching

5. **LLMJudgeEvaluator** (optional, partial credit)
   - Uses LLM to judge semantic match quality
   - More subjective, use sparingly due to cost

#### Implementation Pattern

Similar to anatomic_search, with adjustments for:
- Multiple ontology support
- Ranking evaluation metrics (NDCG/MRR)
- Backend consistency checks
- Optional LLM-as-judge evaluator

#### Acceptance Criteria

- [ ] evals/ontology_match.py exists with 20+ cases
- [ ] 4-5 focused evaluators implemented
- [ ] Tests all supported backends
- [ ] Ranking quality NDCG ≥0.8
- [ ] Backend consistency >90% agreement
- [ ] Documentation with examples from different ontologies
- [ ] Can run: `task evals:ontology_match`

---

### Agent 3: markdown_in

**Priority:** #3 (good learning experience, clear success criteria)

**File:** `evals/markdown_in.py`

**Agent Details:**
- Parses markdown text into finding model structure
- Handles attributes, descriptions, metadata
- Validates structure and completeness
- Located at: `src/findingmodel/tools/markdown_in.py`

#### Test Case Categories (Minimum 12 cases)

**Success Cases (6):**
- Well-formed markdown with all sections
- Simple attributes (choice, text)
- Complex attributes (numeric with units, hierarchical)
- Multiple attributes
- Nested structures

**Error Handling Cases (3):**
- Malformed markdown (syntax errors)
- Missing required sections
- Invalid attribute types
- Inconsistent structure
- Empty sections

**Edge Cases (3):**
- Very long descriptions
- Special characters in attribute names
- Unusual formatting (extra whitespace, mixed case)
- Comments and metadata
- Markdown variations (different header levels)

#### Evaluators to Create (5 focused)

1. **StructuralValidityEvaluator** (strict)
   - Uses base evaluator from evals/base.py
   - Checks parsed model has correct structure

2. **AttributePreservationEvaluator** (partial credit)
   - Verifies all attributes parsed correctly
   - Score = proportion of attributes correctly parsed

3. **TypeCorrectnessEvaluator** (partial credit)
   - Checks attribute types match markdown specification
   - Score = proportion of correct types

4. **ErrorMessageQualityEvaluator** (hybrid)
   - Strict: Error must be raised for invalid input
   - Partial: Clarity and usefulness of error message

5. **RoundTripEvaluator** (strict)
   - Model → markdown → model should be equivalent
   - Tests bidirectional conversion

#### Acceptance Criteria

- [ ] evals/markdown_in.py exists with 12+ cases
- [ ] 5 focused evaluators implemented
- [ ] Structural validity for all success cases
- [ ] Clear error messages for all error cases
- [ ] Round-trip preservation ≥99% accuracy
- [ ] Handles all attribute types from finding_model.py
- [ ] Can run: `task evals:markdown_in`

---

### Agent 4: similar_finding_models

**Priority:** #4 (moderate complexity, clear metrics)

**File:** `evals/similar_models.py`

**Agent Details:**
- Finds similar finding models in the index
- Uses semantic similarity and text matching
- Helps identify duplicates and related models
- Located at: `src/findingmodel/tools/similar_finding_models.py`

#### Test Case Categories (Minimum 15 cases)

**Success Cases (8):**
- Finding exact duplicates
- Finding near-duplicates (minor variations)
- Finding semantically similar models
- Finding models in same domain/category
- Ranking by similarity

**Negative Cases (4):**
- Dissimilar findings that shouldn't match
- Models with similar names but different meanings
- Edge cases that look similar but aren't

**Performance Cases (3):**
- Large index (100+ models)
- Query time within reasonable bounds
- Memory usage reasonable

#### Evaluators to Create (5 focused)

1. **DuplicateDetectionEvaluator** (strict)
   - Binary: found known duplicate or not

2. **RankingQualityEvaluator** (partial credit)
   - MRR or NDCG for ranking quality

3. **PrecisionAtKEvaluator** (partial credit)
   - Precision at top-K results (K=5, 10)

4. **SemanticSimilarityEvaluator** (partial credit)
   - Checks semantic similarity scores are reasonable

5. **PerformanceEvaluator** (partial credit)
   - Tracks query time and resource usage
   - Score based on performance thresholds

#### Acceptance Criteria

- [ ] evals/similar_models.py exists with 15+ cases
- [ ] Duplicate detection accuracy ≥95%
- [ ] MRR ≥0.8 for ranking quality
- [ ] Precision@5 ≥0.9 for top results
- [ ] Performance benchmarks documented
- [ ] Can run: `task evals:similar_models`

---

### Agent 5: finding_description

**Priority:** #5 (most complex, requires medical validation)

**File:** `evals/finding_description.py`

**Agent Details:**
- Generates clinical descriptions for finding models
- Uses LLM to create human-readable text
- Requires clinical accuracy validation
- Located at: `src/findingmodel/tools/finding_description.py`

#### Test Case Categories (Minimum 15 cases)

**Success Cases (8):**
- Common findings with clear descriptions
- Complex findings requiring detail
- Findings with multiple attributes
- Edge findings requiring caveats
- Standardized vs custom descriptions

**Quality Assessment Cases (4):**
- Clinical accuracy (medical correctness)
- Appropriate level of detail
- Clarity and readability
- Consistency across similar findings
- Proper medical terminology usage

**Error Handling Cases (3):**
- Unknown or rare findings
- Incomplete finding information
- Ambiguous finding names
- Missing critical attributes

#### Evaluators to Create (5 focused)

1. **LengthAppropriatenessEvaluator** (hybrid)
   - Strict: Not too short (<20 chars) or too long (>500 chars)
   - Partial: Within optimal range

2. **TerminologyEvaluator** (partial credit)
   - Verifies proper medical terminology used
   - Keyword matching for expected terms

3. **ConsistencyEvaluator** (partial credit)
   - Compares similar findings for consistency
   - Checks for contradictions

4. **LLMJudgeQualityEvaluator** (partial credit)
   - Uses LLM to assess clinical accuracy
   - Most subjective, use carefully

5. **ReadabilityEvaluator** (partial credit)
   - Checks Flesch reading ease or similar metric
   - Ensures appropriate reading level

#### Special Considerations

- **Medical Validation**: Some cases require expert review
- **Ground Truth**: Need curated reference descriptions
- **Subjectivity**: Quality assessment has inherent subjectivity
- **Cost**: LLM-as-judge can be expensive; limit to key cases

#### Acceptance Criteria

- [ ] evals/finding_description.py exists with 15+ cases
- [ ] LLM-as-judge evaluator implemented
- [ ] At least 5 cases with medical expert validation
- [ ] Consistency evaluator for similar findings
- [ ] Quality threshold: ≥80% on LLM judge score
- [ ] Documentation including validation methodology
- [ ] Can run: `task evals:finding_description`

---

## Phase 1 Success Criteria

**Per-Agent Success:**
- [ ] Minimum test cases met (12-20 depending on agent)
- [ ] All test categories covered (success, failure, edge)
- [ ] Agent-specific evaluators implemented (3-5 per agent)
- [ ] Quality threshold met (typically 80-90%)
- [ ] ZERO Logfire code (automatic via evals/__init__.py)
- [ ] Documentation includes examples and guidance
- [ ] Can run in CI/CD pipeline (future)
- [ ] Taskfile commands added

**Overall Success:**
- [ ] All 5 agents have evaluation suites (6 total with model_editor)
- [ ] All agents reach minimum quality thresholds
- [ ] Base evaluators reused across multiple agents
- [ ] Documentation exists for each suite
- [ ] Total coverage: 80+ test cases across all agents
- [ ] Serena memory updated with learnings

**Benefits Achieved:**
- ✅ Comprehensive behavioral assessment of all agents
- ✅ Regression prevention via continuous evaluation
- ✅ Quality metrics for agent improvements
- ✅ Foundation for CI/CD integration
- ✅ Reusable evaluation patterns established

---

## Appendix: Completed Work Reference

### Separate Evals from Tests (COMPLETE)

**File:** `tasks/separate_evals_from_tests_plan.md`

**Status:** All 6 phases complete

**Key Achievements:**
- Three-tier testing structure (unit, integration, evals)
- evals/ directory at project root
- Pytest no longer discovers evals
- Task commands: `task evals`, `task evals:model_editor`
- Documentation: README.md, evals_guide.md, CLAUDE.md

### Refactor Model Editor Evals (COMPLETE - Phases 1-3)

**File:** `tasks/refactor_model_editor_evals.md`

**Status:** Phases 1, 1-A, 2, 3 complete (Phase 3 needs refactoring - Phase 0 above)

**Key Achievements:**
- Base evaluator library (evals/base.py)
- 25 unit tests for evaluators
- model_editor eval suite (12 cases, 5 evaluators)
- Hybrid scoring pattern
- Dataset.evaluate() pattern
- Logfire integration (needs refactoring)

### Current Metrics

**Test Coverage:**
- Unit tests: 84 tests (fast, no API calls)
- Integration tests: ~12 tests with @pytest.mark.callout
- Eval suites: 1 complete (model_editor with 12 cases)
- Evaluator unit tests: 25 tests

**Evaluation Framework:**
- Base evaluators: 5 reusable
- Agent-specific evaluators: 5 for model_editor
- Scoring approach: Hybrid (strict + partial credit)
- Observability: Logfire integrated

**Documentation:**
- evals/README.md - Quick-start guide
- evals/evals_guide.md - Comprehensive how-to
- evals/CLAUDE.md - AI agent reference
- docs/logfire_observability_guide.md - Observability patterns
- Serena memories updated

---

## Open Questions

### General

1. **Evaluation frequency**: How often to run full eval suites? (PR, nightly, weekly?)
2. **Cost management**: How to manage API costs for LLM-as-judge?
3. **Ground truth maintenance**: Who maintains reference data for evaluations?
4. **Failure triage**: What's the process when eval quality drops?

### Agent-Specific

1. **Anatomic Location Search**: Which anatomic taxonomy to use as ground truth?
2. **Ontology Concept Match**: How to handle ontology version updates?
3. **Finding Description**: Who provides medical expert validation?
4. **Similar Finding Models**: What similarity threshold is "good enough"?

---

## References

**Documentation:**
- evals/README.md - Quick-start guide
- evals/evals_guide.md - Comprehensive evaluation guide
- evals/CLAUDE.md - AI agent reference
- docs/logfire_observability_guide.md - Observability patterns

**Implementation:**
- evals/base.py - Reusable evaluator library
- evals/model_editor.py - Example eval suite (after Phase 0 refactoring)
- test/test_base_evaluators.py - Evaluator unit tests

**Plans:**
- tasks/separate_evals_from_tests_plan.md - COMPLETE
- tasks/refactor_model_editor_evals.md - COMPLETE (Phases 1-3)
- tasks/expand_agent_eval_coverage.md - Source for Phase 1 details

**Serena Memories:**
- agent_evaluation_best_practices_2025
- logfire_observability_2025
- pydantic_ai_testing_best_practices
- test_suite_improvements_2025

**External Resources:**
- Pydantic AI Evals: https://ai.pydantic.dev/evals/
- Pydantic Logfire: https://logfire.pydantic.dev/docs/
- Python Logging Best Practices: https://docs.python.org/3/howto/logging.html
