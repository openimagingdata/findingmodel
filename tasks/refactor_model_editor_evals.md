# Refactor Model Editor Evaluation Suite

## Status: Phase 2 Complete - Ready for Phase 3

**Created:** 2025-10-18

**Updated:** 2025-10-18

**Priority:** Medium (Phase 3 is optional)

**Current Phase:** Phase 2 ✅ COMPLETE (all tasks including fixes)

**Related PR:** `copilot/add-eval-case-functionality`

**Related Documents:** `docs/evaluation_guide.md`, memory: `agent_evaluation_best_practices_2025`

## Overview

Refactor the model_editor evaluation suite to align with Pydantic AI Evals best practices and create reusable evaluation components that can be used for all agents in the project.

## Current State

### Implemented (in PR `copilot/add-eval-case-functionality`)

- ✅ Initial model_editor evaluation suite with 11 test cases
- ✅ Custom `evaluate_model_editor_case()` function
- ✅ Documentation and example templates
- ✅ Test data using real finding models
- ✅ Both mock and full API testing modes

### Phase 2 Improvements (NOW COMPLETE)

- ✅ Using Pydantic Evals standard `Evaluator` classes (5 focused evaluators)
- ✅ Using `Dataset.evaluate()` pattern
- ✅ Focused evaluators with single responsibilities
- ✅ Continuous scoring (0.0-1.0) with hybrid approach (strict + partial credit)
- ⏸️ Observability integration ready (structure in place for Phase 3)
- ✅ Reusable pattern established for other agents
- ✅ Framework-provided execution with `report.print()` and threshold assertions

## Goals

1. **Align with Pydantic AI Evals best practices** - Use the framework as intended
2. **Create reusable evaluation components** - Base evaluators and patterns for all agents
3. **Enable observability** - Integrate with Logfire for debugging and monitoring
4. **Establish patterns** - Make it easy to create new eval suites for other agents
5. **Maintain quality** - Keep all existing test coverage while improving structure

## Implementation Plan

### Phase 1: Foundation ✅ IMPLEMENTED (except tests)

**Status:** COMPLETE - code implemented, unit tests pending

**Priority:** High

Create reusable base components that can be used across all agent evaluation suites.

#### Task 1.1: Create Base Evaluator Library ✅ DONE (code only)

**File:** `test/evals/base.py`

Create reusable evaluator classes that work across all agents:

```python
# Evaluators to implement:
- ExactMatchEvaluator[InputT, str]
- ContainsEvaluator[InputT, str]
- KeywordMatchEvaluator[InputT, OutputT]
- StructuralValidityEvaluator[InputT, BaseModel]
- ErrorHandlingEvaluator[InputT, OutputT]
```

**Acceptance Criteria:**

- [x] All evaluators return continuous scores (0.0-1.0)
- [x] Type-safe with proper generics
- [x] Well-documented with docstrings and examples
- [ ] **Unit tests for each evaluator** ⚠️ **BLOCKING** - See Phase 1-A
- [x] Reusable across different agent types

**Implementation:** `test/evals/base.py` (lines 23-367)
- ExactMatchEvaluator
- ContainsEvaluator
- KeywordMatchEvaluator
- StructuralValidityEvaluator
- ErrorHandlingEvaluator

---

#### Task 1.2: Create Base Suite Class ✅ DONE

**File:** `test/evals/base.py`

Create abstract base class for agent evaluation suites:

```python
class AgentEvaluationSuite(ABC, Generic[InputT, ExpectedT, ActualT]):
    @abstractmethod
    def create_successful_cases(self) -> list[Case]: ...
    @abstractmethod
    def create_failure_cases(self) -> list[Case]: ...
    @abstractmethod
    def create_edge_cases(self) -> list[Case]: ...
    @abstractmethod
    async def execute_agent(self, input_data: InputT) -> ActualT: ...
```

**Acceptance Criteria:**

- [x] Abstract base with clear interface
- [x] Helper methods for building datasets
- [x] Documentation on how to subclass
- [x] Example implementation in docstring

**Implementation:** `test/evals/base.py` (lines 374-546)
- AgentEvaluationSuite[InputT, ExpectedT, ActualT]
- 4 abstract methods + 2 concrete helpers
- Comprehensive docstring with full usage example

---

#### Task 1.3: Create Shared Utilities ✅ DONE

**File:** `test/evals/utils.py`

Extract common helpers from existing evals:

```python
# Utilities to implement:
- load_fm_json(filename: str) -> str
- create_mock_result(...) -> AgentResult
- compare_models(...) -> bool
- extract_text_for_keywords(...) -> str
```

**Acceptance Criteria:**

- [x] All utilities well-documented
- [x] Type hints throughout
- [ ] Used by multiple eval suites (will be verified in Phase 2)

**Implementation:** `test/evals/utils.py`
- load_fm_json()
- create_mock_edit_result()
- compare_models()
- extract_text_for_keywords()
- get_attribute_names()

---

### Phase 1-A: Evaluator Unit Tests ✅ COMPLETE

**Status:** ✅ COMPLETE

**Priority:** CRITICAL (was blocking Phase 2)

Add unit tests for the 5 base evaluators. Task 1.1 acceptance criteria required tests, but they were not implemented.

#### Task 1-A.1: Create Test File ✅ DONE

**File:** `test/evals/test_base_evaluators.py`

Implemented 25 focused unit tests covering:
- ExactMatchEvaluator (4 tests)
- ContainsEvaluator (5 tests)
- KeywordMatchEvaluator (7 tests)
- StructuralValidityEvaluator (5 tests)
- ErrorHandlingEvaluator (4 tests)

**Acceptance Criteria:**

- [x] All 25 tests implemented and passing (0.09s execution time)
- [x] Tests verify OUR logic (scoring, edge cases, configuration)
- [x] Tests do NOT test framework functionality
- [x] Float comparisons use pytest.approx
- [x] All tests run in < 1 second (no API calls)
- [x] No linting or type checking errors

**Rationale:** Per `pydantic_ai_testing_best_practices`, we test our code behavior (scoring calculations, partial credit, edge cases), not library implementation. Without these tests, Phase 2 refactoring has no safety net.

**Status:** ✅ Tests complete - Phase 2 unblocked.

---

### Phase 2: Refactor Model Editor Evals ✅ COMPLETE

**Status:** ✅ COMPLETE - All tasks implemented, reviewed, and verified

**Priority:** High

Apply the new patterns to the existing model_editor evaluation suite.

**Prerequisites:** ✅ Phase 1-A unit tests complete

#### Task 2.1: Create Focused Evaluators ✅ DONE

**File:** `test/evals/test_model_editor_evals.py`

Replace single `evaluate_model_editor_case()` function with multiple evaluators:

```python
# Evaluators created:
- IDPreservationEvaluator (strict)
- AttributeAdditionEvaluator (partial credit)
- ChangeTrackingEvaluator (hybrid)
- RejectionAccuracyEvaluator (hybrid)
- ContentPreservationEvaluator (strict)
```

**Acceptance Criteria:**

- [x] Each evaluator tests one aspect
- [x] Return continuous scores with hybrid approach (strict + partial credit)
- [x] Inherit from base `Evaluator` class
- [x] Well-documented

**Implementation:** Lines 326-656 in `test/evals/test_model_editor_evals.py`

---

#### Task 2.2: Convert to Dataset Pattern ✅ DONE

**File:** `test/evals/test_model_editor_evals.py`

Use `Dataset.evaluate()` instead of manual loops:

**Before:**

```python
for case in all_cases:
    actual = await case._execute(...)
    result = evaluate_model_editor_case(case, actual)
```

**After:**

```python
# Evaluators passed to Dataset constructor at module level
report = await dataset.evaluate(run_model_editor_task)
report.print(include_input=False, include_output=True)
```

**Acceptance Criteria:**

- [x] Uses `Dataset.evaluate()` (note: not evaluate_async, that doesn't exist)
- [x] Proper task function wrapper
- [x] Built-in reporting with `report.print()`
- [x] Threshold assertions using `report.overall_score()`
- [x] All 12 test cases pass (11 original + 1 new)

**Implementation:** Lines 676-684 (dataset), 764-812 (test function)

---

#### Task 2.3: Update Documentation ✅ DONE

**Files:** `test/evals/README.md`, `test/evals/add_case_example.py`

Update all documentation to reflect new patterns:

**Acceptance Criteria:**

- [x] README shows new Evaluator-based pattern
- [x] Examples use Dataset.evaluate()
- [x] Instructions for adding cases updated
- [x] Hybrid scoring approach documented

**Implementation:**
- `test/evals/README.md` - Added "Evaluation Approach" section, updated examples
- `test/evals/add_case_example.py` - Updated module docstring and example comments
- Module docstring in `test_model_editor_evals.py` - Expanded with pattern explanation

---

### Phase 3: Observability

**Priority:** Medium

Add observability features for debugging and monitoring.

#### Task 3.1: Add Logfire Integration

**Files:** `test/evals/test_model_editor_evals.py`, `pyproject.toml`

Enable OpenTelemetry tracing via Logfire:

```python
import logfire

logfire.configure(
    send_to_logfire='if-token-present',
    environment='test',
    service_name='findingmodel-evals',
)
```

**Acceptance Criteria:**

- [ ] Logfire configured at module level
- [ ] Traces sent to Logfire when token present
- [ ] Works offline (gracefully degrades)
- [ ] Documentation on viewing traces in Logfire UI

---

#### Task 3.2: Add Metrics Collection

**File:** `test/evals/test_model_editor_evals.py`

Track evaluation metrics:

```python
# Metrics to track:
- Duration per case
- Token usage (if available)
- Success rate by category
- Evaluator score distributions
```

**Acceptance Criteria:**

- [ ] Metrics collected automatically
- [ ] Summary printed at end of run
- [ ] Can export to JSON/CSV for analysis
- [ ] Regression detection (compare to baseline)

---

### Phase 5: CI/CD Integration (Optional)

**Priority:** Low

Automate evaluation runs in CI/CD pipeline.

#### Task 5.1: Create Evaluation Workflow

**File:** `.github/workflows/evals.yml`

CI workflow for running evaluations:

```yaml
# Run on:
- Pull requests (mock tests only)
- Nightly (full evals with API)
- Manual trigger
```

**Acceptance Criteria:**

- [ ] Runs mock tests on all PRs
- [ ] Runs full evals nightly with API keys
- [ ] Stores results as artifacts
- [ ] Fails PR if eval quality drops

---

#### Task 5.2: Regression Tracking

**File:** `test/evals/regression.py`

Track evaluation scores over time:

**Acceptance Criteria:**

- [ ] Store baseline scores
- [ ] Compare current run to baseline
- [ ] Alert on regressions (>5% drop)
- [ ] Update baselines on acceptance

---

## Success Criteria

### Phase 1 Complete When

- [x] Base evaluator library exists and is tested
- [x] **Base evaluator tests written** (Phase 1-A) ✅ COMPLETE
- [x] Base suite class exists with documentation
- [x] Utilities extracted and reusable

### Phase 2 Complete When ✅ COMPLETE

- [x] Model editor evals refactored to use Evaluator classes
- [x] Using Dataset.evaluate() pattern
- [x] All 12 test cases pass (11 original + 1 new)
- [x] Documentation updated
- [x] Critical import bug fixed
- [x] Senior review completed

### Phase 3 Complete When

- [ ] Logfire integration working
- [ ] Metrics collected and reported
- [ ] Can debug failures via traces

### Phase 5 Complete When

- [ ] CI/CD workflow running evals
- [ ] Regression tracking in place
- [ ] Alerts on quality drops

## Dependencies

### Required

- `pydantic-evals>=0.1.0` (already added)
- `pydantic-ai` (already present)
- `pytest>=7.0` (already present)
- `pytest-asyncio` (already present)

### Optional

- `logfire` (for observability)
- `pandas` (for metrics analysis)

## Risks and Mitigations

### Risk: Breaking existing tests

**Mitigation:** Run full test suite after each phase. Keep original tests until refactored versions pass.

### Risk: Logfire costs

**Mitigation:** Use 'if-token-present' mode. Only enable for important debugging sessions.

### Risk: Agent changes breaking evals

**Mitigation:** Version eval suites with agents. Update evals when agent behavior intentionally changes.

## Open Questions

1. **LLM-as-judge implementation:** Which model to use? GPT-4? Claude? Local model?
2. **Baseline storage:** Where to store regression baselines? Git? Separate DB?
3. **Performance benchmarks:** Should we include latency/cost metrics in evaluations?

## Next Actions

- [x] ~~Review this plan with team~~
- [x] ~~Get approval for Phase 1-2 implementation~~
- [x] ~~Begin Task 1.1 (base evaluator library)~~ - DONE
- [x] ~~Begin Task 1.2 (base suite class)~~ - DONE
- [x] ~~Begin Task 1.3 (shared utilities)~~ - DONE
- [x] ~~BEGIN Phase 1-A: Write evaluator unit tests~~ - ✅ DONE
- [x] ~~Complete Phase 1-A tests (blocking for Phase 2)~~ - ✅ DONE
- [x] ~~Begin Phase 2 (refactor model_editor evals)~~ - ✅ DONE
- [x] ~~Task 2.1: Create focused evaluators~~ - ✅ DONE
- [x] ~~Task 2.2: Convert to Dataset pattern~~ - ✅ DONE
- [x] ~~Task 2.3: Update documentation~~ - ✅ DONE
- [x] ~~Fix critical import bug~~ - ✅ DONE
- [ ] **Optional: Begin Phase 3 (observability integration)** 🔲 NEXT (if desired)

## Implementation Status Summary

| Phase | Status | Files Created/Modified | Notes |
|-------|--------|----------------------|-------|
| Phase 1 | ✅ 100% COMPLETE | `test/evals/base.py`, `test/evals/utils.py` | All tasks complete including tests |
| Phase 1-A | ✅ 100% COMPLETE | `test/evals/test_base_evaluators.py` | All 25 tests passing (0.09s) |
| **Phase 2** | **✅ 100% COMPLETE** | **`test/evals/test_model_editor_evals.py`, `test/evals/README.md`, `test/evals/add_case_example.py`** | **5 evaluators, Dataset pattern, hybrid scoring, import bug fixed** |
| Phase 3 | 🔲 Optional | - | Observability integration (Logfire, metrics) |
| Phase 5 | 🔲 Optional | - | CI/CD integration |

## References

- **Full guide:** `docs/evaluation_guide.md`
- **Implementation:** `test/evals/test_model_editor_evals.py` (Phase 2 refactoring)
- **Base evaluators:** `test/evals/base.py` (reusable evaluators)
- **Unit tests:** `test/evals/test_base_evaluators.py` (25 tests for base evaluators)
- **Serena memory:** `agent_evaluation_best_practices_2025` (includes Phase 2 lessons learned)
- **Current PR:** `copilot/add-eval-case-functionality`
- **Expansion plan:** `tasks/expand_agent_eval_coverage.md`
- **Pydantic AI Evals:** <https://ai.pydantic.dev/evals/>
- **Related memories:**
  - `pydantic_ai_testing_best_practices`
  - `test_suite_improvements_2025`
  - `ai_assistant_usage_2025`
