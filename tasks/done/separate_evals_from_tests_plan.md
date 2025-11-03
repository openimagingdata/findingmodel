# Separate Evals from Tests - Implementation Plan

**Created:** 2025-10-20

**Status:** Ready to Execute

**Related Documents:**
- Root `CLAUDE.md` - Project testing philosophy
- `tasks/refactor_model_editor_evals.md` - Phase 1 & 2 completion status
- `docs/logfire_observability_guide.md` - Observability patterns for evals
- `.serena/memories/agent_evaluation_best_practices_2025.md` - Eval best practices

---

## Vision & Context

### The Problem

Currently, evaluation suites live in `test/evals/` which creates confusion:
- Pytest discovers and runs evals during normal testing (slow, expensive)
- Unclear when to add unit test vs integration test vs eval
- Mixed concerns: testing correctness vs assessing behavioral quality

### The Solution

Establish a clear three-tier testing structure:

1. **Unit Tests** (`test/test_*.py`) - Verify logic correctness with mocked dependencies
   - Fast, no API calls
   - Run with `task test`
   - **Important:** `test_base_evaluators.py` stays in `test/` because it contains unit tests OF the evaluator library itself (verifying ExactMatchEvaluator returns 1.0 on exact match, KeywordMatchEvaluator calculates partial credit correctly, etc.). These are NOT evals of agents; they test that the evaluation framework code works correctly.

2. **Integration Tests** (`test/test_*.py` with `@pytest.mark.callout`) - Verify wiring
   - Real API calls, specific scenarios
   - Run with `task test-full`

3. **Evals** (`evals/*.py`) - Assess behavioral quality comprehensively
   - Dataset.evaluate() with focused evaluators
   - Run with `task evals` or `task evals:model_editor`
   - Manual execution, not part of CI (initially)

### Outcome

- `evals/` directory at project root (separate from `test/`)
- Pytest never discovers evals (testpaths = ["test"])
- Clear documentation for humans (`evals/README.md`, `evals/evals_guide.md`)
- Clear guidance for AI agents (`evals/CLAUDE.md`)
- Clean separation of concerns

---

## Overall Phases

| Phase | Description | Tasks |
|-------|-------------|-------|
| 1 | Create evals/ directory structure | 1.1-1.4 |
| 2 | Create documentation | 2.1-2.3 |
| 3 | Clean up test/ directory | 3.1-3.3 |
| 4 | Update Taskfile | 4.1 |
| 5 | Update existing documentation | 5.1-5.5 |
| 6 | Final cleanup | 6.1 |

---

## Task Breakdown

### Phase 1: Create evals/ Directory Structure

#### Task 1.1: Create evals/ directory and __init__.py

**Context Needed:** None

**Actions:**
1. Create `evals/` directory at project root (same level as `test/`, `src/`, `docs/`)
2. Create empty `evals/__init__.py` file

**Acceptance Criteria:**
- [ ] `evals/` directory exists at project root
- [ ] `evals/__init__.py` exists and is empty

**Dependencies:** None

---

#### Task 1.2: Move base.py to evals/

**Context Needed:**
- Read `test/evals/base.py` to understand structure

**Actions:**
1. Copy `test/evals/base.py` to `evals/base.py`
2. No code changes needed (file has no imports that need updating)
3. Keep original in test/evals/ for now (will delete in Phase 3)

**Acceptance Criteria:**
- [ ] `evals/base.py` exists
- [ ] File is identical to `test/evals/base.py`
- [ ] Can import with `from evals.base import ExactMatchEvaluator`

**Dependencies:** 1.1 complete

---

#### Task 1.3: Move utils.py to evals/

**Context Needed:**
- Read `test/evals/utils.py` to check imports

**Actions:**
1. Copy `test/evals/utils.py` to `evals/utils.py`
2. Update imports if needed (check for relative imports)
3. Keep original in test/evals/ for now (will delete in Phase 3)

**Acceptance Criteria:**
- [ ] `evals/utils.py` exists
- [ ] All imports resolve correctly
- [ ] Can import with `from evals.utils import load_fm_json`

**Dependencies:** 1.1 complete

---

#### Task 1.4: Transform test_model_editor_evals.py → model_editor.py

**Context Needed:**
- Read `test/evals/test_model_editor_evals.py`
- Read `evals/base.py` and `evals/utils.py` (for import updates)

**Actions:**
1. Copy `test/evals/test_model_editor_evals.py` to `evals/model_editor.py`
2. Remove all pytest decorators (`@pytest.mark.callout`, `@pytest.mark.asyncio`)
3. Delete the `test_single_successful_case()` function entirely (lines 816-866)
4. Rename `test_run_model_editor_evals()` → `run_model_editor_evals()` (remove `test_` prefix)
5. Update imports:
   - `from .utils import ...` → `from evals.utils import ...`
   - `from .base import ...` → `from evals.base import ...` (if present)
6. Add `__main__` block at end of file (see specification below)
7. Keep original in test/evals/ for now (will delete in Phase 3)

**`__main__` block specification:**
```python
if __name__ == "__main__":
    import asyncio

    async def main():
        print("\nRunning model_editor evaluation suite...")
        print("=" * 80)

        report = await run_model_editor_evals()

        print("\n" + "=" * 80)
        print("MODEL EDITOR EVALUATION RESULTS")
        print("=" * 80 + "\n")
        report.print(include_input=False, include_output=True)
        print(f"\nOVERALL SCORE: {report.overall_score():.2f}")
        print("=" * 80 + "\n")

        # Future Phase 3: Logfire integration via pydantic-evals
        # Future: Save report to file, compare to baseline, etc.

    asyncio.run(main())
```

**Acceptance Criteria:**
- [ ] `evals/model_editor.py` exists
- [ ] No pytest decorators present
- [ ] No `test_single_successful_case` function
- [ ] Main function named `run_model_editor_evals` (no `test_` prefix)
- [ ] All imports use `from evals.` not `from .`
- [ ] `__main__` block present
- [ ] Can run with `python -m evals.model_editor` and produces eval report
- [ ] File imports resolve (no ModuleNotFoundError)

**Dependencies:** 1.1, 1.2, 1.3 complete

---

### Phase 2: Create Documentation

#### Task 2.1: Create evals/README.md

**Context Needed:**
- Read `test/evals/README.md` for structure ideas (but don't copy directly)
- Read root `README.md` for project style

**Content Requirements:**

Quick-start guide (~100-150 lines) covering:
1. What are evals? (vs unit tests, vs integration tests)
2. How to run evals:
   - `task evals` - run all evals
   - `task evals:model_editor` - run model_editor only
   - `python -m evals.model_editor` - direct execution
3. How to read eval output (report.print() format, overall score, individual evaluators)
4. When to run evals (before major changes, when assessing agent quality)
5. Link to `evals_guide.md` for writing new evals
6. Link to `CLAUDE.md` for AI agent guidance

**Acceptance Criteria:**
- [ ] File exists at `evals/README.md`
- [ ] Answers "how do I run evals?" in 5 minutes
- [ ] Clear distinction between evals and tests
- [ ] Links to evals_guide.md and CLAUDE.md

**Dependencies:** None (can do in parallel with Phase 1)

---

#### Task 2.2: Create evals/evals_guide.md

**Context Needed:**
- Read `docs/evaluation_guide.md` to extract content
- Read `test/evals/README.md` for additional context
- Read `test/evals/IMPLEMENTATION.md` for implementation notes
- Read `test/evals/add_case_example.py` for examples

**Content Requirements:**

Comprehensive guide (merge and adapt from `docs/evaluation_guide.md`) covering:
1. Three-tier testing structure (unit → integration → evals)
2. When to add each type of test
3. What makes a good eval case?
4. Evaluator patterns:
   - Exact match (strict binary scoring)
   - Keyword matching (partial credit)
   - Hybrid scoring (strict + partial)
5. Dataset.evaluate() API usage
6. Report interpretation and analysis
7. Case design guidelines (success, rejection, edge cases)
8. Examples and templates (from add_case_example.py)
9. How to add new eval suites for other agents

**Acceptance Criteria:**
- [ ] File exists at `evals/evals_guide.md`
- [ ] Content merged from `docs/evaluation_guide.md`
- [ ] Paths updated (test/evals/ → evals/)
- [ ] Complete reference for writing evals
- [ ] Examples and templates included

**Dependencies:** None (can do in parallel with Phase 1)

---

#### Task 2.3: Create evals/CLAUDE.md

**Context Needed:**
- Read root `CLAUDE.md` for style and structure
- Read `docs/logfire_observability_guide.md` for Logfire section
- Read `.serena/memories/agent_evaluation_best_practices_2025.md` for patterns

**Content Requirements:**

Project-specific AI agent reference (~150-200 lines) covering:
1. Directory structure (flat for now, nest when complex)
2. File naming conventions:
   - `tool_name.py` not `test_tool_name.py`
   - Main function: `run_tool_name_evals()` not `test_run_tool_name_evals()`
3. When to create new eval suites vs add to existing
4. Logfire integration quick reference:
   - High-level overview (how to instrument evals)
   - Link to `docs/logfire_observability_guide.md` for details
   - Phase 3 status note
5. Link to `evals_guide.md` for evaluator patterns
6. Link to Serena memory `agent_evaluation_best_practices_2025`
7. Import patterns (`from evals.base` not `from .base`)

**Logfire Section Template:**
```markdown
## Logfire Integration (Phase 3 - Planned)

For detailed Logfire patterns and instrumentation, see `docs/logfire_observability_guide.md`.

### Quick Reference

When implementing Phase 3 Logfire integration:
- Wrap eval execution in `logfire.span('eval_suite', tool_name='model_editor')`
- Include case metadata as span attributes (case name, edit type, model ID)
- Log evaluation results as structured events
- Configure with `send_to_logfire='if-token-present'` for graceful degradation
- Use `logfire.instrument_pydantic()` for automatic model tracing

### Current Status

Phase 3 is planned but not yet implemented. Evals currently output to console only.
```

**Acceptance Criteria:**
- [ ] File exists at `evals/CLAUDE.md`
- [ ] Project-specific conventions documented
- [ ] Logfire quick reference with link to docs/
- [ ] Links to evals_guide.md and Serena memory
- [ ] Clear guidance for AI agents creating evals

**Dependencies:** None (can do in parallel with Phase 1)

---

### Phase 3: Clean Up test/ Directory

#### Task 3.1: Move test_base_evaluators.py from test/evals/ to test/

**Context Needed:**
- Read `test/evals/test_base_evaluators.py`

**Actions:**
1. Copy `test/evals/test_base_evaluators.py` to `test/test_base_evaluators.py`
2. Update imports:
   - `from .base import ...` → `from evals.base import ...`
   - `from .utils import ...` → `from evals.utils import ...` (if present)
3. Keep original in test/evals/ for now (will delete in Task 3.3)

**Acceptance Criteria:**
- [ ] `test/test_base_evaluators.py` exists
- [ ] All imports resolve (from evals.base, from evals.utils)
- [ ] Pytest discovers and runs these 25 tests
- [ ] All tests pass with `pytest test/test_base_evaluators.py`

**Dependencies:** Phase 1 complete (evals/base.py and evals/utils.py must exist)

---

#### Task 3.2: Remove 4 eval-like tests from test/test_model_editor.py

**Context Needed:**
- Read `test/test_model_editor.py`

**Actions:**

Delete these 4 test functions entirely:
1. `test_edit_model_natural_language_callout_real_api` (approx line 160-190)
2. `test_edit_model_markdown_callout_real_api` (approx line 195-225)
3. `test_forbidden_change_nl_callout_real_api` (approx line 230-260)
4. `test_forbidden_change_markdown_callout_real_api` (approx line 265-295)

**Rationale:** These tests assess behavioral quality (success/failure scenarios), which is now handled comprehensively by `evals/model_editor.py` with focused evaluators.

**Keep these 7 tests** (verify they remain):
- `test_edit_model_natural_language_add_attribute`
- `test_export_model_for_editing_roundtrip`
- `test_export_model_for_editing_structure_full`
- `test_export_model_for_editing_attributes_only`
- `test_assign_real_attribute_ids_infers_source`
- `test_assign_real_attribute_ids_uses_explicit_source`
- `test_assign_real_attribute_ids_no_placeholders_returns_same_object`

**Acceptance Criteria:**
- [ ] 4 test functions deleted (lines removed cleanly)
- [ ] 7 test functions remain
- [ ] File has no syntax errors
- [ ] All remaining tests pass with `pytest test/test_model_editor.py`

**Dependencies:** None (can do in parallel with other Phase 3 tasks)

---

#### Task 3.3: Delete test/evals/ directory

**Context Needed:**
- Verify Tasks 1.2, 1.3, 1.4, and 3.1 are complete (all files moved)

**Actions:**
1. Verify these files exist in new locations:
   - `evals/base.py` ✓
   - `evals/utils.py` ✓
   - `evals/model_editor.py` ✓
   - `test/test_base_evaluators.py` ✓
2. Delete `test/evals/` directory and ALL contents

**Note on files being deleted:**
- `IMPLEMENTATION.md` - content will be absorbed into evals_guide.md (Task 2.2)
- `add_case_example.py` - examples will go in evals_guide.md (Task 2.2)
- `README.md` - superseded by evals/README.md and evals/evals_guide.md
- `list_cases.py` - utility script for listing eval cases; **DECISION: DELETE IT**. Not worth maintaining. Users can run the evals to see what cases exist, or read the source code. The script would need import updates and adds minimal value.

**Acceptance Criteria:**
- [ ] `test/evals/` directory does not exist
- [ ] `pytest test/` still runs successfully (no broken imports)
- [ ] All evals files exist in `evals/`

**Dependencies:** 1.2, 1.3, 1.4, 3.1 complete

---

### Phase 4: Update Taskfile

#### Task 4.1: Add evals tasks to Taskfile.yml

**Context Needed:**
- Read `Taskfile.yml` to understand structure and style

**Actions:**

Add two new tasks after the `test-full` task:

```yaml
  evals:
    desc: "Run all agent evaluation suites"
    cmds:
      - echo "Running all agent evaluations..."
      - uv run python -m evals.model_editor
      # Future: add other agent evals here
    silent: true

  evals:model_editor:
    desc: "Run model_editor evaluation suite"
    cmds:
      - echo "Running model_editor evaluations..."
      - uv run python -m evals.model_editor
    silent: true
```

**Acceptance Criteria:**
- [ ] Two new tasks added to Taskfile.yml
- [ ] `task evals` executes successfully
- [ ] `task evals:model_editor` executes successfully
- [ ] Both produce eval reports

**Dependencies:** Phase 1 complete (evals/model_editor.py must exist and be runnable)

---

### Phase 5: Update Existing Documentation

#### Task 5.1: Update tasks/refactor_model_editor_evals.md

**Context Needed:**
- Read `tasks/refactor_model_editor_evals.md`

**Actions:**

Search and replace throughout the file:
1. `test/evals/` → `evals/`
2. `docs/evaluation_guide.md` → `evals/evals_guide.md`

Add to "Related Documents" section:
```markdown
- `evals/CLAUDE.md` - AI agent reference for eval development
```

Update "Evaluation Suite Structure" section (around line 450):
```markdown
## Evaluation Suite Structure

```
evals/
├── base.py                          # Reusable evaluators & base classes
├── model_editor.py                  # ✅ model_editor evaluation (Phase 2 COMPLETE)
├── utils.py                         # Shared helpers
├── README.md                        # Quick-start guide
├── evals_guide.md                   # Comprehensive how-to-write guide
└── CLAUDE.md                        # AI agent reference
```
```

**Acceptance Criteria:**
- [ ] All paths updated (test/evals/ → evals/)
- [ ] Documentation references updated
- [ ] evals/CLAUDE.md referenced
- [ ] Structure section shows new evals/ layout

**Dependencies:** Phase 2 complete (new docs exist to reference)

---

#### Task 5.2: Update root CLAUDE.md

**Context Needed:**
- Read `CLAUDE.md`

**Actions:**

Add to section "2. Architecture touchpoints" or "4. Testing + QA":

```markdown
### Three-Tier Testing Structure

- **Unit tests** (`test/test_*.py`) - Verify logic correctness with mocked dependencies
  - Run with `task test`
- **Integration tests** (`test/test_*.py` with `@pytest.mark.callout`) - Verify wiring with real APIs
  - Run with `task test-full`
- **Evals** (`evals/*.py`) - Assess behavioral quality comprehensively
  - Run with `task evals` or `task evals:model_editor`
  - See `evals/CLAUDE.md` for eval development guidance
```

**Acceptance Criteria:**
- [ ] Three-tier structure documented
- [ ] Reference to evals/CLAUDE.md added
- [ ] Clear when to use each testing approach

**Dependencies:** Phase 2 complete (evals/CLAUDE.md exists)

---

#### Task 5.3: Update Serena memory agent_evaluation_best_practices_2025

**Context Needed:**
- Read `.serena/memories/agent_evaluation_best_practices_2025.md`

**Actions:**

Update "Evaluation Suite Structure" section (around line 140):
```markdown
## Evaluation Suite Structure

```
evals/                               # Root-level directory (NOT in test/)
├── base.py                          # Reusable evaluators & base classes
├── model_editor.py                  # ✅ model_editor evaluation (Phase 2 COMPLETE)
├── utils.py                         # Shared helpers
├── README.md                        # Quick-start guide
├── evals_guide.md                   # Comprehensive how-to-write guide
├── CLAUDE.md                        # AI agent reference for eval development
└── (future: anatomic_search.py, ontology_match.py, etc.)

test/                                # Test directory (pytest discovers here)
├── test_base_evaluators.py          # Unit tests for evaluator library
└── test_*.py                        # Unit and integration tests
```
```

Search and replace throughout:
- `test/evals/` → `evals/`

Add to "Resources" section:
```markdown
- **Eval development:** `evals/CLAUDE.md` - Project-specific conventions
- **Quick start:** `evals/README.md` - How to run evals
- **Comprehensive guide:** `evals/evals_guide.md` - How to write evals
```

**Acceptance Criteria:**
- [ ] All paths updated (test/evals/ → evals/)
- [ ] Structure section shows evals/ at root level
- [ ] References to new documentation files
- [ ] Clear separation: evals/ for behavioral assessment, test/ for correctness

**Dependencies:** Phase 2 complete (new docs exist)

---

#### Task 5.4: Update tasks/expand_agent_eval_coverage.md

**Context Needed:**
- Read `tasks/expand_agent_eval_coverage.md`

**Actions:**

Search and replace throughout the file:
- `test/evals/` → `evals/`

Update file references:
- `test/evals/test_anatomic_search_evals.py` → `evals/anatomic_search.py`
- `test/evals/test_ontology_match_evals.py` → `evals/ontology_match.py`
- etc.

**Acceptance Criteria:**
- [ ] All paths updated to evals/
- [ ] All filenames follow new convention (tool_name.py not test_tool_name.py)
- [ ] References to evals_guide.md updated

**Dependencies:** None

---

#### Task 5.5: Check and update test/conftest.py if needed

**Context Needed:**
- Read `test/conftest.py`

**Actions:**
1. Search for any references to "evals" or "test/evals"
2. If found, update paths to `evals/` (unlikely but verify)
3. If no references found, no action needed

**Acceptance Criteria:**
- [ ] File checked for evals references
- [ ] Any found references updated
- [ ] Fixtures still work correctly

**Dependencies:** Phase 1 complete

---

### Phase 6: Final Cleanup

#### Task 6.1: Remove docs/evaluation_guide.md

**Context Needed:**
- Verify Task 2.2 complete (content merged into evals/evals_guide.md)

**Actions:**
1. Review `evals/evals_guide.md` to confirm all content from `docs/evaluation_guide.md` is present
2. Delete `docs/evaluation_guide.md`

**Acceptance Criteria:**
- [ ] `docs/evaluation_guide.md` does not exist
- [ ] Content preserved in `evals/evals_guide.md`
- [ ] No broken links (all references updated in Phase 5)

**Dependencies:** 2.2 complete (evals/evals_guide.md exists with merged content), Phase 5 complete (all references updated)

---

## Task Dependencies

```
Phase 1 (Directory Structure):
  1.1 → 1.2, 1.3, 1.4
  1.2, 1.3 → 1.4

Phase 2 (Documentation):
  All tasks independent, can run in parallel

Phase 3 (Cleanup):
  Phase 1 → 3.1
  1.2, 1.3, 1.4, 3.1 → 3.3
  3.2 independent

Phase 4 (Taskfile):
  Phase 1 → 4.1

Phase 5 (Doc Updates):
  Phase 2 → 5.1, 5.2, 5.3
  5.4, 5.5 independent

Phase 6 (Final Cleanup):
  2.2, Phase 5 → 6.1
```

**Critical Path:** 1.1 → 1.2/1.3 → 1.4 → 3.1 → 3.3

---

## Verification Steps

After all tasks complete, verify:

### Pytest Behavior
```bash
# Should only discover test/, not evals/
pytest --collect-only | grep -c "evals/"  # Should be 0

# Should run all unit tests including test_base_evaluators.py
pytest test/                               # Should pass

# Should run integration tests
pytest test/ -m callout                    # Should pass
```

### Evals Execution
```bash
# Should run model_editor evals
task evals                                 # Should produce report
task evals:model_editor                    # Should produce report
python -m evals.model_editor               # Should produce report
```

### Directory Structure
```bash
# Should not exist
test -d test/evals && echo "FAIL: test/evals still exists" || echo "PASS"
test -f docs/evaluation_guide.md && echo "FAIL: docs/evaluation_guide.md still exists" || echo "PASS"

# Should exist
test -d evals && echo "PASS" || echo "FAIL: evals/ missing"
test -f evals/model_editor.py && echo "PASS" || echo "FAIL: evals/model_editor.py missing"
test -f evals/README.md && echo "PASS" || echo "FAIL: evals/README.md missing"
test -f evals/evals_guide.md && echo "PASS" || echo "FAIL: evals/evals_guide.md missing"
test -f evals/CLAUDE.md && echo "PASS" || echo "FAIL: evals/CLAUDE.md missing"
```

### Import Resolution
```bash
# Should import successfully
python -c "from evals.base import ExactMatchEvaluator; print('PASS')"
python -c "from evals.utils import load_fm_json; print('PASS')"
python -c "from evals.model_editor import run_model_editor_evals; print('PASS')"
```

---

## Success Criteria

- ✅ All tasks complete with acceptance criteria met
- ✅ Pytest discovers only `test/`, not `evals/`
- ✅ `task test` runs unit tests (including test_base_evaluators.py)
- ✅ `task test-full` runs integration tests
- ✅ `task evals` and `task evals:model_editor` execute successfully
- ✅ Three-tier testing structure clear and documented
- ✅ All paths updated (test/evals/ → evals/)
- ✅ No broken imports or references
- ✅ `test/evals/` and `docs/evaluation_guide.md` deleted
- ✅ Clear documentation for humans (README, evals_guide)
- ✅ Clear documentation for AI agents (CLAUDE.md)

---

## Notes

- Tasks can be delegated to sub-agents with minimal context (each task specifies what to read)
- Phases can partially overlap (Phase 2 can start before Phase 1 completes)
- Critical path keeps original files until copies verified (safe migration)
- No time estimates included (as requested)
