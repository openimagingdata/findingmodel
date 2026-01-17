"""Unit tests for reusable evaluators.

This test suite verifies the PerformanceEvaluator scoring logic, time source handling,
and edge case behavior. Per pydantic_ai_testing_best_practices, these tests focus on
OUR code behavior (scoring calculations, time source precedence, N/A cases) rather
than framework functionality.

All tests use simple Pydantic models and execute quickly without external dependencies.
"""

from typing import Any

from findingmodel_ai.tools.evaluators import PerformanceEvaluator
from pydantic import BaseModel
from pydantic_ai import models
from pydantic_evals.evaluators import EvaluatorContext

# Prevent accidental API calls during unit tests
models.ALLOW_MODEL_REQUESTS = False


def _create_ctx(
    output: Any,
    duration: float | None = None,
    metadata: Any = None,
) -> EvaluatorContext:
    """Helper to create minimal EvaluatorContext for performance evaluator tests.

    Args:
        output: The output object (can have duration, query_time, or error attributes)
        duration: The ctx.duration value (execution time from Pydantic Evals)
        metadata: Optional metadata (None triggers N/A case)

    Returns:
        EvaluatorContext configured for testing PerformanceEvaluator
    """
    return EvaluatorContext(
        name="test",
        inputs={},
        metadata=metadata,
        expected_output=None,
        output=output,
        duration=duration,
        _span_tree=None,  # type: ignore[arg-type]
        attributes={},
        metrics={},
    )


# Test output models for different time sources
class OutputWithDuration(BaseModel):
    """Output model with duration attribute."""

    duration: float
    result: str = "success"


class OutputWithQueryTime(BaseModel):
    """Output model with query_time attribute (backward compatibility)."""

    query_time: float
    result: str = "success"


class OutputWithError(BaseModel):
    """Output model with error attribute."""

    error: str | None = None
    duration: float | None = None


class OutputWithoutTime(BaseModel):
    """Output model with no time attributes."""

    result: str = "success"


# ============================================================================
# PerformanceEvaluator - Under Time Limit Tests (3 tests)
# ============================================================================


def test_performance_under_limit_from_ctx_duration() -> None:
    """Verify execution under time limit returns 1.0 when using ctx.duration."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=2.5,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_performance_under_limit_from_output_duration() -> None:
    """Verify execution under time limit returns 1.0 when using output.duration."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithDuration(duration=3.0),
        duration=None,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_performance_under_limit_from_query_time() -> None:
    """Verify execution under time limit returns 1.0 when using output.query_time."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithQueryTime(query_time=1.5),
        duration=None,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


# ============================================================================
# PerformanceEvaluator - Over Time Limit Tests (3 tests)
# ============================================================================


def test_performance_over_limit_from_ctx_duration() -> None:
    """Verify execution over time limit returns 0.0 when using ctx.duration."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=7.5,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_performance_over_limit_from_output_duration() -> None:
    """Verify execution over time limit returns 0.0 when using output.duration."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithDuration(duration=10.0),
        duration=None,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_performance_over_limit_from_query_time() -> None:
    """Verify execution over time limit returns 0.0 when using output.query_time."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithQueryTime(query_time=15.0),
        duration=None,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


# ============================================================================
# PerformanceEvaluator - Boundary Tests (2 tests)
# ============================================================================


def test_performance_exactly_at_limit() -> None:
    """Verify execution exactly at time limit returns 1.0 (inclusive boundary)."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=5.0,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # <= means 5.0 is acceptable


def test_performance_just_over_limit() -> None:
    """Verify execution just over time limit returns 0.0."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=5.001,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


# ============================================================================
# PerformanceEvaluator - Time Source Precedence Tests (2 tests)
# ============================================================================


def test_performance_ctx_duration_takes_precedence() -> None:
    """Verify ctx.duration takes precedence over output attributes."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    # ctx.duration = 2.0 (under), output.duration = 10.0 (over)
    # Should use ctx.duration and pass
    ctx = _create_ctx(
        output=OutputWithDuration(duration=10.0),
        duration=2.0,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # Uses ctx.duration=2.0, not output.duration=10.0


def test_performance_query_time_over_output_duration() -> None:
    """Verify output.query_time takes precedence over output.duration."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    # Create output with both query_time and duration
    class OutputWithBoth(BaseModel):
        query_time: float
        duration: float

    # query_time = 2.0 (under), duration = 10.0 (over)
    # Should use query_time (backward compatibility precedence) and pass
    ctx = _create_ctx(
        output=OutputWithBoth(query_time=2.0, duration=10.0),
        duration=None,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # Uses output.query_time=2.0, not duration=10.0


# ============================================================================
# PerformanceEvaluator - N/A Cases Tests (4 tests)
# ============================================================================


def test_performance_missing_metadata_returns_na() -> None:
    """Verify missing metadata returns 1.0 (N/A case)."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithDuration(duration=10.0),
        duration=None,
        metadata=None,  # Missing metadata
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # N/A case


def test_performance_error_case_returns_na() -> None:
    """Verify error case returns 1.0 (N/A - error scored separately)."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithError(error="Something went wrong", duration=10.0),
        duration=None,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # N/A when error present


def test_performance_missing_duration_returns_na() -> None:
    """Verify missing duration from all sources returns 1.0 (N/A case)."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),  # No duration or query_time
        duration=None,  # No ctx.duration
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # N/A when no time source available


def test_performance_error_overrides_slow_execution() -> None:
    """Verify error case returns 1.0 even if execution was slow."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    # Execution took 10s (over limit) but had error
    ctx = _create_ctx(
        output=OutputWithError(error="Timeout", duration=10.0),
        duration=10.0,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0  # Error takes precedence


# ============================================================================
# PerformanceEvaluator - Various Time Values Tests (4 tests)
# ============================================================================


def test_performance_very_fast_execution() -> None:
    """Verify very fast execution (0.5s) passes 5s limit."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=0.5,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_performance_one_second_execution() -> None:
    """Verify 1.0s execution passes 5s limit."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=1.0,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


def test_performance_very_slow_execution() -> None:
    """Verify very slow execution (30s) fails 5s limit."""
    evaluator = PerformanceEvaluator(time_limit=5.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=30.0,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 0.0


def test_performance_instantaneous_execution() -> None:
    """Verify instantaneous execution (0.0s) passes any limit."""
    evaluator = PerformanceEvaluator(time_limit=1.0)

    ctx = _create_ctx(
        output=OutputWithoutTime(),
        duration=0.0,
        metadata={"test": "data"},
    )

    score = evaluator.evaluate(ctx)
    assert score == 1.0


# ============================================================================
# PerformanceEvaluator - Configuration Tests (2 tests)
# ============================================================================


def test_performance_custom_time_limit() -> None:
    """Verify custom time limit configuration works correctly."""
    evaluator = PerformanceEvaluator(time_limit=2.0)  # Custom 2s limit

    # 1.5s should pass
    ctx_pass = _create_ctx(
        output=OutputWithoutTime(),
        duration=1.5,
        metadata={"test": "data"},
    )
    assert evaluator.evaluate(ctx_pass) == 1.0

    # 3.0s should fail
    ctx_fail = _create_ctx(
        output=OutputWithoutTime(),
        duration=3.0,
        metadata={"test": "data"},
    )
    assert evaluator.evaluate(ctx_fail) == 0.0


def test_performance_default_time_limit() -> None:
    """Verify default time limit is 30.0 seconds."""
    evaluator = PerformanceEvaluator()  # Use default

    # 29.9s should pass
    ctx_pass = _create_ctx(
        output=OutputWithoutTime(),
        duration=29.9,
        metadata={"test": "data"},
    )
    assert evaluator.evaluate(ctx_pass) == 1.0

    # 30.1s should fail
    ctx_fail = _create_ctx(
        output=OutputWithoutTime(),
        duration=30.1,
        metadata={"test": "data"},
    )
    assert evaluator.evaluate(ctx_fail) == 0.0
