# Refactor Model Editor Evaluation Suite

## Status: Phase 3 Complete - All Core Work Done

**Created:** 2025-10-18

**Updated:** 2025-10-24

**Priority:** Medium

**Current Phase:** Phase 3 ✅ COMPLETE (including test cleanup)

**Related PR:** `copilot/add-eval-case-functionality`

**Related Documents:**

- `evals/evals_guide.md` - Comprehensive guide to agent evaluation
- `docs/logfire_observability_guide.md` - Logfire observability patterns (Phase 3)
- **Serena Memories:**
  - `agent_evaluation_best_practices_2025` - Evaluation patterns and lessons learned
  - `logfire_observability_2025` - Logfire integration best practices

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

**File:** `evals/base.py`

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

**Implementation:** `evals/base.py` (lines 23-367)

- ExactMatchEvaluator
- ContainsEvaluator
- KeywordMatchEvaluator
- StructuralValidityEvaluator
- ErrorHandlingEvaluator

---

#### Task 1.2: Create Base Suite Class ✅ DONE

**File:** `evals/base.py`

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

**Implementation:** `evals/base.py` (lines 374-546)

- AgentEvaluationSuite[InputT, ExpectedT, ActualT]
- 4 abstract methods + 2 concrete helpers
- Comprehensive docstring with full usage example

---

#### Task 1.3: Create Shared Utilities ✅ DONE

**File:** `evals/utils.py`

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

**Implementation:** `evals/utils.py`

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

**File:** `evals/test_base_evaluators.py`

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

**File:** `evals/test_model_editor_evals.py`

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

**Implementation:** Lines 326-656 in `evals/test_model_editor_evals.py`

---

#### Task 2.2: Convert to Dataset Pattern ✅ DONE

**File:** `evals/test_model_editor_evals.py`

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

**Files:** `evals/README.md`, `evals/add_case_example.py`

Update all documentation to reflect new patterns:

**Acceptance Criteria:**

- [x] README shows new Evaluator-based pattern
- [x] Examples use Dataset.evaluate()
- [x] Instructions for adding cases updated
- [x] Hybrid scoring approach documented

**Implementation:**

- `evals/README.md` - Added "Evaluation Approach" section, updated examples
- `evals/add_case_example.py` - Updated module docstring and example comments
- Module docstring in `test_model_editor_evals.py` - Expanded with pattern explanation

---

### Phase 3: Observability with Logfire

**Status:** ✅ COMPLETE

**Priority:** Medium

**Updated:** 2025-10-24

Add observability features using Pydantic Logfire for debugging, monitoring, and understanding agent behavior during evaluations.

**Reference Documentation:** `docs/logfire_observability_guide.md`

**Key Design Decision:** Logfire integration should be **on by default** but gracefully degrade when no token is present. This ensures developers get observability benefits without requiring setup, while production users can opt-in to cloud tracing.

---

#### Task 3.0: Add Configuration Fields

**Files:**
- `src/findingmodel/config.py`
- `.env.sample`

Add Logfire configuration fields to `FindingModelConfig`:

```python
class FindingModelConfig(BaseSettings):
    """Configuration for FindingModel library."""

    # ... existing fields ...

    # Logfire configuration (observability platform)
    logfire_token: QuoteStrippedSecretStr | None = Field(
        default=None,
        description="Logfire.dev write token for cloud tracing (optional)"
    )
    disable_send_to_logfire: bool = Field(
        default=False,
        description="Disable sending data to Logfire platform (local-only mode)"
    )
    logfire_verbose: bool = Field(
        default=False,
        description="Enable verbose Logfire console logging"
    )
```

Update `.env.sample` to document Logfire configuration:

```bash
# Optional: Logfire observability (for eval tracing and debugging)
# Get token from https://logfire.pydantic.dev/ after creating an account
# LOGFIRE_TOKEN=
# DISABLE_SEND_TO_LOGFIRE=false
# LOGFIRE_VERBOSE=false
```

**Acceptance Criteria:**

- [x] `logfire_token` field added (optional, default None, uses SecretStr)
- [x] `disable_send_to_logfire` field added (default False)
- [x] `logfire_verbose` field added (default False)
- [x] Fields are read from environment variables automatically
- [x] Settings object accessible via `from findingmodel.config import settings`
- [x] `.env.sample` updated with Logfire configuration documentation

---

#### Task 3.1: Add Logfire Dependency

**Files:** `pyproject.toml`

Add Logfire to development dependencies:

```toml
[project.optional-dependencies]
dev = [
    # ... existing dependencies
    "logfire>=1.0.0",  # OpenTelemetry observability platform
]
```

**Acceptance Criteria:**

- [x] Logfire added to `dev` dependencies
- [x] Version constraint appropriate (>=1.0.0 or compatible)
- [x] Dependency installs cleanly with `uv pip install -e ".[dev]"`

---

#### Task 3.2: Configure Logfire in Model Editor Evals

**Files:** `evals/model_editor.py`

Add Logfire configuration at module level with graceful degradation:

```python
"""
Model Editor evaluation suite with Logfire observability.

This module uses Pydantic Logfire for tracing and observability.

**Logfire Integration:**
- Automatically detects LOGFIRE_TOKEN from .env
- Sends traces to cloud if token present and DISABLE_SEND_TO_LOGFIRE=false
- Can be forced to local-only mode with DISABLE_SEND_TO_LOGFIRE=true
- Gracefully becomes no-op when no token present

**Setup for Cloud Tracing:**
    1. Create account at https://logfire.pydantic.dev/
    2. Get write token from dashboard
    3. Add LOGFIRE_TOKEN=xxx to .env
    4. Run evals normally - traces appear in Logfire UI

**Environment Variables:**
- LOGFIRE_TOKEN: Write token from logfire.pydantic.dev (optional)
- DISABLE_SEND_TO_LOGFIRE: Set to true to force local-only mode (default: false)
- LOGFIRE_VERBOSE: Set to true for verbose console logging (default: false)
"""

import logfire
from findingmodel.config import settings

# Configure Logfire
# send_to_logfire logic: False if explicitly disabled, 'if-token-present' otherwise
logfire.configure(
    token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
    send_to_logfire=False if settings.disable_send_to_logfire else 'if-token-present',
    console_colors='auto',
    console_min_log_level='debug' if settings.logfire_verbose else 'info',
)

# Instrument Pydantic AI agents (PRIMARY instrumentation)
logfire.instrument_pydantic_ai()
```

**Acceptance Criteria:**

- [x] Logfire configured with conditional `send_to_logfire` based on `disable_send_to_logfire` setting
- [x] Token passed via `token=` parameter from settings
- [x] Uses project settings system (`settings.logfire_verbose`, `settings.logfire_token`, `settings.disable_send_to_logfire`)
- [x] Uses `ConsoleOptions` (actual Logfire 1.0+ API, not direct `console_*` parameters as plan specified)
- [x] Console disabled by default (only enabled with `logfire_verbose=True`)
- [x] Calls `logfire.instrument_pydantic_ai()` after configure (PRIMARY instrumentation)
- [x] No hard-coded service names or environment metadata
- [x] No conditional instrumentation in eval code (configuration handles behavior)
- [x] Spans work locally without token (local-only mode)
- [x] Can force local-only mode even with token via DISABLE_SEND_TO_LOGFIRE=true
- [x] Clear docstring explaining Logfire integration with .env-based setup
- [x] Consistent with project's API key management patterns

---

#### Task 3.3: Instrument Evaluation Execution

**Files:** `evals/model_editor.py`

Add Logfire tracing to the main evaluation function:

```python
async def run_model_editor_evals():
    """
    Run the full model_editor evaluation suite with Logfire tracing.

    Returns the evaluation Report object for programmatic access.
    """
    with logfire.span(
        'model_editor_eval_suite',
        total_cases=len(all_cases),
        evaluator_count=len(evaluators),
    ):
        logfire.info(
            'Starting model_editor evaluation suite',
            cases_total=len(all_cases),
            evaluators=[e.__class__.__name__ for e in evaluators],
        )

        report = await model_editor_dataset.evaluate(run_model_editor_task)

        # Calculate and log overall score
        all_scores = [score.value for case in report.cases for score in case.scores.values()]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        logfire.info(
            'Evaluation suite completed',
            overall_score=overall_score,
            cases_total=len(report.cases),
        )

        return report
```

**Acceptance Criteria:**

- [x] Top-level span wraps entire evaluation suite
- [x] Logs suite start with metadata
- [x] Logs suite completion with overall score
- [x] Span includes case count and evaluator count as attributes
- [x] No changes to evaluation logic or return values
- [x] Works with and without Logfire token (spans are no-ops when token absent)

---

#### Task 3.4: Instrument Task Execution Function

**Files:** `evals/model_editor.py`

Add tracing to individual case execution:

```python
async def run_model_editor_task(case: ModelEditorCase) -> ModelEditorActualOutput:
    """
    Execute a single model_editor evaluation case with Logfire tracing.

    This function is called by Dataset.evaluate() for each case.
    """
    model = FindingModelFull.model_validate_json(case.model_json)

    with logfire.span(
        'eval_case {name}',
        name=case.name,
        case_name=case.name,
        edit_type=case.edit_type,
        should_succeed=case.should_succeed,
    ):
        # Log case start (verbose mode only)
        if settings.logfire_verbose:
            logfire.debug(
                'Starting evaluation case',
                case_name=case.name,
                model_id=case.model_json[:50] + '...',
                command_length=len(case.command),
            )

        # Execute case
        with logfire.span('model_editor_execution', operation=case.edit_type):
            if case.edit_type == 'natural_language':
                result = await model_editor.edit_model(model, case.command)
            else:  # markdown
                result = await model_editor.edit_model_from_markdown(model, case.command)

        # Log result
        logfire.info(
            'Case completed',
            case_name=case.name,
            success=len(result.changes) > 0,
            changes_count=len(result.changes),
            rejections_count=len(result.rejections),
        )

    return ModelEditorActualOutput(
        model=result.model,
        changes=result.changes,
        rejections=result.rejections,
    )
```

**Acceptance Criteria:**

- [x] Each case wrapped in its own span
- [x] Span includes case metadata (name, type, expected outcome)
- [x] Logs case start in verbose mode (`settings.logfire_verbose`)
- [x] Logs case completion with results
- [x] Model editor execution in sub-span
- [x] No changes to core logic or return values
- [x] No conditional instrumentation - spans are always used (no-op when token absent)

---

#### Task 3.5: Add Logfire to Base Evaluators (Optional)

**Files:** `evals/base.py`

Add optional Logfire tracing to base evaluators for detailed evaluation debugging:

```python
class KeywordMatchEvaluator(Evaluator[InputT, OutputT]):
    """
    Base evaluator for keyword matching with optional Logfire tracing.
    """

    def __init__(
        self,
        keywords: list[str],
        text_extractor: Callable[[OutputT], str],
        case_insensitive: bool = True,
        enable_tracing: bool = False,  # Opt-in for tracing
    ):
        self.keywords = keywords
        self.text_extractor = text_extractor
        self.case_insensitive = case_insensitive
        self.enable_tracing = enable_tracing

    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        """Evaluate with optional tracing."""
        if self.enable_tracing:
            import logfire
            with logfire.span(
                'evaluate_keywords',
                evaluator='KeywordMatchEvaluator',
                case_name=getattr(ctx, 'case_name', 'unknown'),
                keywords_count=len(self.keywords),
            ):
                score = self._evaluate_impl(ctx)
                logfire.debug(
                    'Keyword evaluation',
                    score=score,
                    matched_keywords=self._get_matched_keywords(ctx),
                )
                return score
        else:
            return self._evaluate_impl(ctx)

    def _evaluate_impl(self, ctx: EvaluatorContext[InputT, OutputT]) -> float:
        """Core evaluation logic (unchanged)."""
        # ... existing implementation
```

**Note:** This task is **optional** and may add too much overhead. Consider implementing only if debugging evaluators becomes necessary.

**Acceptance Criteria:**

- [ ] Tracing is opt-in per evaluator instance
- [ ] Default is `enable_tracing=False` (no overhead)
- [ ] When enabled, logs evaluation details
- [ ] No changes to default behavior
- [ ] All existing tests pass

**Decision:** Defer this task unless debugging needs arise

---

#### Task 3.6: Update Documentation

**Files:**

- `docs/logfire_observability_guide.md` ✅ **CREATED**
- `evals/README.md`
- `evals/model_editor.py` (module docstring already updated in Task 3.2)

Update documentation to explain Logfire integration:

**README.md updates:**

Add new section after "Running the Evaluations":

````markdown
## Observability with Logfire

The evaluation suite integrates with [Pydantic Logfire](https://logfire.pydantic.dev/)
for observability and debugging.

### Quick Start

Logfire works out of the box in local-only mode (console logging). To enable cloud tracing:

```bash
# 1. Create account at https://logfire.pydantic.dev/
# 2. Get write token from dashboard
# 3. Add to .env file:
echo "LOGFIRE_TOKEN=your_token_here" >> .env

# 4. Run evaluations - traces automatically appear in Logfire UI
python -m evals.model_editor
```
````

### Viewing Traces

When configured with a token, traces appear in your Logfire dashboard at [logfire.pydantic.dev](https://logfire.pydantic.dev/).

You can see:

- Evaluation suite execution timeline
- Individual case execution spans
- Pydantic AI agent calls with prompts and completions
- Performance metrics and timing
- Error traces with full context

### Environment Variables (in .env)

- `LOGFIRE_TOKEN=xxx` - Write token from logfire.pydantic.dev (optional, enables cloud tracing)
- `DISABLE_SEND_TO_LOGFIRE=true` - Force local-only mode (default: false)
- `LOGFIRE_VERBOSE=true` - Enable verbose logging (default: false)

### More Information

See `docs/logfire_observability_guide.md` for comprehensive documentation.

**Acceptance Criteria:**

- [x] Comprehensive guide created at `docs/logfire_observability_guide.md` ✅
- [x] README.md updated with Logfire section
- [x] Module docstring updated (completed in Task 3.2)
- [x] Examples of viewing traces in Logfire UI
- [x] Environment variable documentation
- [x] Troubleshooting section for common issues

---

#### Task 3.7: Create Serena Memory

**Files:** Serena memory system

Create memory documenting Logfire best practices for the project:

**Memory Name:** `logfire_observability_2025`

**Content:** Summary of Logfire integration patterns, configuration, and best practices specific to FindingModel project.

**Acceptance Criteria:**

- [x] Memory created via `write_memory` tool
- [x] Covers configuration patterns
- [x] Documents environment variables
- [x] Links to full documentation
- [x] Includes examples from evaluation suites
- [x] Notes about graceful degradation

---

### Phase 3 Complete When

- [x] Logfire added to dependencies
- [x] Configuration fields added to FindingModelConfig (logfire_token, disable_send_to_logfire, logfire_verbose)
- [x] Module-level configuration implemented with conditional send_to_logfire
- [x] Evaluation suite instrumented (top-level span + logging)
- [x] Task execution function instrumented (per-case spans)
- [x] Documentation updated (README + comprehensive guide)
- [x] Serena memory updated
- [x] All existing tests still pass
- [x] Works with and without LOGFIRE_TOKEN
- [x] DISABLE_SEND_TO_LOGFIRE=true forces local-only mode even with token
- [x] No warnings when token absent
- [x] No conditional instrumentation in eval code (spans always used, configuration handles behavior)
- [x] Console output disabled by default for clean eval runs (only enabled with LOGFIRE_VERBOSE=true)

**Benefits:**

- Rich observability into evaluation execution
- Debugging support for failing cases
- Performance insights (case duration, bottlenecks)
- Optional cloud tracing for production monitoring
- Zero friction for local development (works without setup)
- Foundation for other agent evaluation suites

**Additional Cleanup (2025-10-24):**

After completing Phase 3, cleaned up `test/test_model_editor.py` to align with three-tier testing structure:
- Removed 3 redundant callout tests (now comprehensively covered by evals)
- Kept 1 simple integration test for basic API sanity checking
- Simplified remaining integration test to verify result structure only (no behavioral checks)
- Result: 7 unit tests + 1 minimal integration test

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

**File:** `evals/regression.py`

Track evaluation scores over time:

**Acceptance Criteria:**

- [ ] Store baseline scores
- [ ] Compare current run to baseline
- [ ] Alert on regressions (>5% drop)
- [ ] Update baselines on acceptance

---

## Success Criteria

### Phase 1 Complete When ✅ COMPLETE

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

### Phase 3 Complete Requirements ✅

- [x] Logfire added to dependencies (pyproject.toml)
- [x] Module-level configuration implemented with graceful degradation
- [x] Evaluation suite instrumented (top-level span + logging)
- [x] Task execution function instrumented (per-case spans)
- [x] Documentation updated (README + module docstrings)
- [x] All existing tests still pass
- [x] Works with and without LOGFIRE_TOKEN
- [x] No warnings when token absent
- [x] Respects DISABLE_SEND_TO_LOGFIRE environment variable
- [x] Console output disabled by default (LOGFIRE_VERBOSE required for console logging)

### Phase 5 Complete Requirements

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
- [x] ~~Phase 3 (observability integration)~~ - ✅ COMPLETE
- [x] ~~Cleanup: Remove redundant callout tests from test_model_editor.py~~ - ✅ DONE

## Implementation Status Summary

| Phase       | Status               | Files Created/Modified                                                                                | Notes                                                               |
| ----------- | -------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Phase 1     | ✅ 100% COMPLETE     | `evals/base.py`, `evals/utils.py`                                                           | All tasks complete including tests                                  |
| Phase 1-A   | ✅ 100% COMPLETE     | `evals/test_base_evaluators.py`                                                                  | All 25 tests passing (0.09s)                                        |
| Phase 2     | ✅ 100% COMPLETE     | `evals/test_model_editor_evals.py`, `evals/README.md`, `evals/add_case_example.py` | 5 evaluators, Dataset pattern, hybrid scoring, import bug fixed     |
| **Phase 3** | **✅ 100% COMPLETE** | **`evals/model_editor.py`, `src/findingmodel/config.py`, `evals/README.md`, `.env.sample`, `pyproject.toml`** | **Logfire observability, console off by default, graceful degradation** |
| Phase 5     | 🔲 Optional          | -                                                                                                     | CI/CD integration                                                   |

## References

### Documentation

- **Evaluation guide:** `evals/evals_guide.md` - Comprehensive guide to agent evaluation
- **Logfire guide:** `docs/logfire_observability_guide.md` - Observability patterns for evals (Phase 3)

### Implementation

- **Main eval suite:** `evals/test_model_editor_evals.py` - Phase 2 refactored implementation
- **Base evaluators:** `evals/base.py` - Reusable evaluator library
- **Unit tests:** `evals/test_base_evaluators.py` - 25 tests for base evaluators

### Plans and Tasks

- **Current PR:** `copilot/add-eval-case-functionality`
- **Expansion plan:** `tasks/expand_agent_eval_coverage.md` - Roadmap for other agents

### Serena Memories

- **`agent_evaluation_best_practices_2025`** - Evaluation patterns, lessons learned from Phase 2
- **`logfire_observability_2025`** - Logfire integration best practices (Phase 3)
- **`pydantic_ai_testing_best_practices`** - General AI testing conventions
- **`test_suite_improvements_2025`** - Testing improvements
- **`ai_assistant_usage_2025`** - AI assistant collaboration patterns

### External Resources

- **Pydantic AI Evals:** <https://ai.pydantic.dev/evals/>
- **Pydantic Logfire:** <https://logfire.pydantic.dev/docs/>
