"""Reusable evaluators for Pydantic AI evaluation suites.

EVALUATOR PHILOSOPHY:
    This module contains ONLY truly reusable evaluators - those used across multiple eval suites
    with complex, non-trivial logic. Most evaluators should remain inline in eval scripts for
    clarity and context.

    Prefer this hierarchy:
    1. Pydantic Evals built-in evaluators (if available)
    2. Inline evaluators in eval scripts (for most cases)
    3. Evaluators in this module (only if used 2+ times and complex)

    Keep evaluators focused, composable, and well-documented. Each evaluator should have:
    - Clear docstring explaining purpose and scoring logic
    - Usage example showing how to instantiate
    - Strict type hints

CURRENTLY AVAILABLE EVALUATORS:
    - PerformanceEvaluator: Validates execution time against configurable threshold
"""

from dataclasses import dataclass
from typing import TypeVar

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

# Generic type variables for evaluator input/output/metadata
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")


@dataclass
class PerformanceEvaluator(Evaluator[InputT, OutputT, MetadataT]):
    """Evaluate execution time against a configurable threshold.

    Uses strict scoring (0.0 or 1.0) because performance is critical for user experience.
    Returns 1.0 if execution completes within time limit, 0.0 if it exceeds the threshold.

    USE CASES:
        - SLA compliance testing (e.g., "API calls must complete in <2s")
        - Performance regression detection
        - Resource consumption validation
        - Timeout enforcement testing

    REQUIREMENTS:
        The output object must have a `duration` attribute (float) containing execution time
        in seconds, OR a `query_time` attribute (for backward compatibility with existing evals).

    SCORING LOGIC:
        - Returns 1.0 if ctx.duration <= time_limit
        - Returns 0.0 if ctx.duration > time_limit
        - Returns 1.0 if execution error occurred (N/A case - error scored separately)
        - Returns 1.0 if metadata is missing (N/A case)

    Example usage:
        >>> from pydantic_evals import Dataset, Case
        >>> from findingmodel.tools.evaluators import PerformanceEvaluator
        >>>
        >>> # Create evaluator with 5-second time limit
        >>> perf_evaluator = PerformanceEvaluator(time_limit=5.0)
        >>>
        >>> # Add to dataset with other evaluators
        >>> dataset = Dataset(
        ...     cases=[...],
        ...     evaluators=[
        ...         CorrectnessEvaluator(),
        ...         PerformanceEvaluator(time_limit=10.0),  # 10s limit
        ...     ]
        ... )
        >>>
        >>> # Run evaluation
        >>> report = await dataset.evaluate(run_task)

    Attributes:
        time_limit: Maximum acceptable execution time in seconds (default: 30.0)
    """

    time_limit: float = 30.0

    def evaluate(self, ctx: EvaluatorContext[InputT, OutputT, MetadataT]) -> float:
        """Evaluate execution time performance.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata.
                 The output should have a `duration` or `query_time` attribute.

        Returns:
            1.0 if performance acceptable (time <= limit or N/A), 0.0 if too slow

        Note:
            Execution errors receive 1.0 since performance evaluation is N/A when errors occur.
            The error is captured and scored separately by other evaluators.
        """
        # Handle missing metadata - N/A case, return 1.0
        if ctx.metadata is None:
            return 1.0

        # Skip if execution error occurred - N/A for performance, return 1.0
        # (error handling is evaluated separately)
        if hasattr(ctx.output, "error") and ctx.output.error:
            return 1.0

        # Get execution time - check ctx.duration first (Pydantic Evals standard)
        # Fall back to output.query_time (backward compatibility with existing evals)
        execution_time = None
        if ctx.duration is not None:
            execution_time = ctx.duration
        elif hasattr(ctx.output, "query_time"):
            execution_time = ctx.output.query_time
        elif hasattr(ctx.output, "duration"):
            execution_time = ctx.output.duration

        # If we can't find execution time, return 1.0 (N/A case)
        if execution_time is None:
            return 1.0

        # Strict check: execution time must be under threshold
        return 1.0 if execution_time <= self.time_limit else 0.0
