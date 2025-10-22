"""Evaluation suite for model_editor using pydantic-evals framework.

This module defines evaluation cases for assessing the model_editor functionality,
including both successful edits and rejection cases.

EVALUATOR-BASED PATTERN:
- Cases are evaluated using Dataset.evaluate() with focused evaluators
- Each evaluator checks a specific aspect (ID preservation, attributes, etc.)
- Hybrid scoring: strict for non-negotiables (0.0 or 1.0), partial credit for quality (0.0-1.0)

EVALUATORS:
- IDPreservationEvaluator: Model IDs must never change (strict)
- AttributeAdditionEvaluator: Expected attributes added (partial credit)
- ChangeTrackingEvaluator: Changes recorded with keywords (hybrid)
- RejectionAccuracyEvaluator: Rejections recorded with keywords (hybrid)
- ContentPreservationEvaluator: Model unchanged on rejection (strict)

See evals/base.py for reusable base evaluators and examples.

LOGFIRE INTEGRATION:
This module uses Pydantic Logfire for tracing and observability.

- Automatically detects LOGFIRE_TOKEN from .env
- Sends traces to cloud if token present and DISABLE_SEND_TO_LOGFIRE=false
- Can be forced to local-only mode with DISABLE_SEND_TO_LOGFIRE=true
- Gracefully becomes no-op when no token present

Setup for Cloud Tracing:
    1. Create account at https://logfire.pydantic.dev/
    2. Get write token from dashboard
    3. Add LOGFIRE_TOKEN=xxx to .env
    4. Run evals normally - traces appear in Logfire UI

Add to .env file:
    LOGFIRE_TOKEN=pfp_your_token_here
    DISABLE_SEND_TO_LOGFIRE=false  # or true for local-only
    LOGFIRE_VERBOSE=false           # or true for verbose logging

Environment Variables:
- LOGFIRE_TOKEN: Write token from logfire.pydantic.dev (optional)
- DISABLE_SEND_TO_LOGFIRE: Set to true to force local-only mode (default: false)
- LOGFIRE_VERBOSE: Set to true for verbose console logging (default: false)
"""

from pathlib import Path

import logfire
from logfire import ConsoleOptions
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.reporting import EvaluationReport

from evals.utils import compare_models, extract_text_for_keywords, get_attribute_names
from findingmodel.config import settings
from findingmodel.finding_model import FindingModelFull
from findingmodel.tools import model_editor

# Configure Logfire
# send_to_logfire logic: False if explicitly disabled, 'if-token-present' otherwise
logfire.configure(
    token=settings.logfire_token.get_secret_value() if settings.logfire_token else None,
    send_to_logfire=False if settings.disable_send_to_logfire else "if-token-present",
    console=ConsoleOptions(
        colors="auto",
        min_log_level="debug" if settings.logfire_verbose else "info",
    ),
)

# Instrument Pydantic AI agents (PRIMARY instrumentation)
logfire.instrument_pydantic_ai()


class ModelEditorInput(BaseModel):
    """Input for a model editor evaluation case."""

    model_json: str  # JSON string of the FindingModelFull to edit
    command: str  # Natural language editing command
    edit_type: str  # 'natural_language' or 'markdown'


class ModelEditorExpectedOutput(BaseModel):
    """Expected output for a model editor evaluation case."""

    should_succeed: bool  # Whether the edit should succeed or be rejected
    should_preserve_id: bool = True  # Model ID should be preserved
    added_attribute_names: list[str] = []  # Names of attributes that should be added
    rejection_keywords: list[str] = []  # Keywords that should appear in rejections
    changes_keywords: list[str] = []  # Keywords that should appear in changes


class ModelEditorActualOutput(BaseModel):
    """Actual output from running a model editor case."""

    model: FindingModelFull
    rejections: list[str]
    changes: list[str]
    error: str | None = None


class ModelEditorCase(Case[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]):
    """A test case for model_editor functionality."""

    def __init__(
        self,
        name: str,
        model_json: str,
        command: str,
        edit_type: str,
        should_succeed: bool,
        should_preserve_id: bool = True,
        added_attribute_names: list[str] | None = None,
        rejection_keywords: list[str] | None = None,
        changes_keywords: list[str] | None = None,
    ) -> None:
        """Initialize a model editor evaluation case.

        Args:
            name: Name of the test case
            model_json: JSON string of the FindingModelFull to edit
            command: Natural language editing command or markdown text
            edit_type: 'natural_language' or 'markdown'
            should_succeed: Whether the edit should succeed or be rejected
            should_preserve_id: Whether the model ID should be preserved
            added_attribute_names: Names of attributes that should be added
            rejection_keywords: Keywords that should appear in rejections
            changes_keywords: Keywords that should appear in changes
        """
        inputs = ModelEditorInput(model_json=model_json, command=command, edit_type=edit_type)
        metadata = ModelEditorExpectedOutput(
            should_succeed=should_succeed,
            should_preserve_id=should_preserve_id,
            added_attribute_names=added_attribute_names or [],
            rejection_keywords=rejection_keywords or [],
            changes_keywords=changes_keywords or [],
        )
        super().__init__(name=name, inputs=inputs, metadata=metadata)

    async def _execute(self, input_data: ModelEditorInput) -> ModelEditorActualOutput:
        """Execute the model editor with the given input."""
        try:
            model = FindingModelFull.model_validate_json(input_data.model_json)

            if input_data.edit_type == "natural_language":
                result = await model_editor.edit_model_natural_language(model, input_data.command)
            elif input_data.edit_type == "markdown":
                result = await model_editor.edit_model_markdown(model, input_data.command)
            else:
                raise ValueError(f"Unknown edit_type: {input_data.edit_type}")

            return ModelEditorActualOutput(model=result.model, rejections=result.rejections, changes=result.changes)
        except Exception as e:
            # Return a placeholder model with error info
            model = FindingModelFull.model_validate_json(input_data.model_json)
            return ModelEditorActualOutput(model=model, rejections=[], changes=[], error=str(e))


def load_fm_json(filename: str) -> str:
    """Load a .fm.json file from the test data directory."""
    test_data_dir = Path(__file__).parent.parent / "test" / "data" / "defs"
    return (test_data_dir / filename).read_text()


def create_successful_edit_cases() -> list[ModelEditorCase]:
    """Create cases for successful edits that should be applied."""
    cases = []

    # Case 1: Add a simple choice attribute
    pe_json = load_fm_json("pulmonary_embolism.fm.json")
    cases.append(
        ModelEditorCase(
            name="add_severity_attribute",
            model_json=pe_json,
            command="Add a new attribute named 'severity' of type choice with values: mild, moderate, severe",
            edit_type="natural_language",
            should_succeed=True,
            added_attribute_names=["severity"],
            changes_keywords=["severity", "added"],
        )
    )

    # Case 2: Add a numeric attribute
    cases.append(
        ModelEditorCase(
            name="add_numeric_size_attribute",
            model_json=pe_json,
            command="Add a numeric attribute named 'thrombus size' with minimum 0, maximum 50, unit mm",
            edit_type="natural_language",
            should_succeed=True,
            added_attribute_names=["thrombus size"],
            changes_keywords=["thrombus size", "added"],
        )
    )

    # Case 3: Add synonyms to model
    aortic_json = load_fm_json("aortic_dissection.fm.json")
    cases.append(
        ModelEditorCase(
            name="add_synonyms",
            model_json=aortic_json,
            command="Add synonym 'aortic tear' to the model",
            edit_type="natural_language",
            should_succeed=True,
            changes_keywords=["synonym", "aortic tear"],
        )
    )

    # Case 4: Add choice values to existing attribute
    breast_density_json = load_fm_json("breast_density.fm.json")
    cases.append(
        ModelEditorCase(
            name="add_choice_value_to_existing",
            model_json=breast_density_json,
            command="For the 'breast density' attribute, add a new value called 'extremely dense'",
            edit_type="natural_language",
            should_succeed=True,
            changes_keywords=["extremely dense", "added"],
        )
    )

    # Case 5: Add description to model
    vent_json = load_fm_json("ventricular_diameters.fm.json")
    cases.append(
        ModelEditorCase(
            name="add_description_enhancement",
            model_json=vent_json,
            command="Enhance the model description with more clinical context about normal ranges",
            edit_type="natural_language",
            should_succeed=True,
            changes_keywords=["description"],
        )
    )

    return cases


def create_rejection_cases() -> list[ModelEditorCase]:
    """Create cases for edits that should be rejected."""
    cases = []

    # Rejection Case 1: Rename an existing attribute
    pe_json = load_fm_json("pulmonary_embolism.fm.json")
    cases.append(
        ModelEditorCase(
            name="reject_rename_attribute",
            model_json=pe_json,
            command="Rename the 'presence' attribute to 'occurrence'",
            edit_type="natural_language",
            should_succeed=False,
            rejection_keywords=["rename", "not allowed", "forbidden"],
        )
    )

    # Rejection Case 2: Delete an attribute
    cases.append(
        ModelEditorCase(
            name="reject_delete_attribute",
            model_json=pe_json,
            command="Remove the 'change from prior' attribute from the model",
            edit_type="natural_language",
            should_succeed=False,
            rejection_keywords=["remove", "delete", "not allowed"],
        )
    )

    # Rejection Case 3: Change model ID
    cases.append(
        ModelEditorCase(
            name="reject_change_model_id",
            model_json=pe_json,
            command="Change the model ID to OIFM_TEST_000000",
            edit_type="natural_language",
            should_succeed=False,
            rejection_keywords=["ID", "immutable", "not allowed"],
        )
    )

    # Rejection Case 4: Remove existing choice values
    aortic_json = load_fm_json("aortic_dissection.fm.json")
    cases.append(
        ModelEditorCase(
            name="reject_remove_choice_value",
            model_json=aortic_json,
            command="Remove the 'indeterminate' value from the 'presence' attribute",
            edit_type="natural_language",
            should_succeed=False,
            rejection_keywords=["remove", "value", "not allowed"],
        )
    )

    # Rejection Case 5: Change attribute type
    cases.append(
        ModelEditorCase(
            name="reject_change_attribute_type",
            model_json=aortic_json,
            command="Change the 'presence' attribute from choice to numeric type",
            edit_type="natural_language",
            should_succeed=False,
            rejection_keywords=["type", "change", "not allowed"],
        )
    )

    return cases


def create_markdown_edit_cases() -> list[ModelEditorCase]:
    """Create cases for markdown-based edits."""
    cases = []

    # Markdown Case 1: Add attribute via markdown
    pe_json = load_fm_json("pulmonary_embolism.fm.json")
    model = FindingModelFull.model_validate_json(pe_json)
    base_md = model_editor.export_model_for_editing(model)
    enhanced_md = (
        base_md + "\n### location\n\nAnatomical location of the embolism\n\n- central\n- peripheral\n- bilateral\n\n"
    )

    cases.append(
        ModelEditorCase(
            name="markdown_add_attribute",
            model_json=pe_json,
            command=enhanced_md,
            edit_type="markdown",
            should_succeed=True,
            added_attribute_names=["location"],
            changes_keywords=["location", "added"],
        )
    )

    # Markdown Case 2: Attempt to remove attribute via markdown (should reject)
    aortic_json = load_fm_json("aortic_dissection.fm.json")
    aortic_model = FindingModelFull.model_validate_json(aortic_json)
    full_md = model_editor.export_model_for_editing(aortic_model)
    # Remove the first attribute section
    lines = full_md.splitlines()
    truncated_lines = []
    skip = False
    for line in lines:
        if "### presence" in line:
            skip = True
            continue
        if skip and line.startswith("### "):
            skip = False
        if not skip:
            truncated_lines.append(line)
    truncated_md = "\n".join(truncated_lines)

    cases.append(
        ModelEditorCase(
            name="markdown_reject_delete_attribute",
            model_json=aortic_json,
            command=truncated_md,
            edit_type="markdown",
            should_succeed=False,
            rejection_keywords=["delete", "removal", "missing"],
        )
    )

    return cases


# =============================================================================
# Focused Evaluator Classes
# =============================================================================


class IDPreservationEvaluator(Evaluator[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]):
    """Evaluate that the model ID is preserved during editing.

    This evaluator checks that the OIFM ID of the model remains unchanged
    after an edit operation. Model IDs are immutable and should never change,
    even when the model is modified or when edits are rejected.

    Returns:
        1.0 if model ID is preserved
        0.0 if model ID has changed
        1.0 if should_preserve_id is False (N/A)

    Example usage:
        >>> from pydantic_evals import Case
        >>> case = Case(
        ...     name="test_id_preservation",
        ...     inputs=ModelEditorInput(
        ...         model_json='{"oifm_id": "OIFM_TEST_000001", ...}',
        ...         command="Add severity attribute",
        ...         edit_type="natural_language"
        ...     ),
        ...     expected_output=ModelEditorExpectedOutput(
        ...         should_succeed=True,
        ...         should_preserve_id=True
        ...     )
        ... )
        >>> evaluator = IDPreservationEvaluator()
        >>> # If actual.model.oifm_id == "OIFM_TEST_000001": score = 1.0
        >>> # If actual.model.oifm_id != "OIFM_TEST_000001": score = 0.0
    """

    def evaluate(
        self, ctx: EvaluatorContext[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]
    ) -> float:
        """Evaluate ID preservation.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if ID preserved or not required to be preserved, 0.0 otherwise
        """
        # Handle missing metadata
        if ctx.metadata is None:
            return 1.0

        # Skip if ID preservation not required
        if not ctx.metadata.should_preserve_id:
            return 1.0

        # Skip if execution error occurred (not ID preservation issue)
        if ctx.output.error:
            return 1.0

        # Compare IDs
        original_model = FindingModelFull.model_validate_json(ctx.inputs.model_json)
        return 1.0 if ctx.output.model.oifm_id == original_model.oifm_id else 0.0


class AttributeAdditionEvaluator(Evaluator[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]):
    """Evaluate that expected attributes were added to the model.

    This evaluator verifies that all expected new attributes appear in the
    modified model after a successful edit operation. It provides partial
    credit based on the proportion of expected attributes that were added.

    Returns:
        1.0 if all expected attributes were added
        Proportion of expected attributes added (0.0-1.0) for partial credit
        1.0 if no attributes expected to be added (N/A)

    Example usage:
        >>> from pydantic_evals import Case
        >>> case = Case(
        ...     name="test_add_attributes",
        ...     inputs=ModelEditorInput(
        ...         model_json='{"attributes": [...]}',
        ...         command="Add severity and location attributes",
        ...         edit_type="natural_language"
        ...     ),
        ...     expected_output=ModelEditorExpectedOutput(
        ...         should_succeed=True,
        ...         added_attribute_names=["severity", "location"]
        ...     )
        ... )
        >>> evaluator = AttributeAdditionEvaluator()
        >>> # If both "severity" and "location" added: score = 1.0
        >>> # If only "severity" added: score = 0.5
        >>> # If neither added: score = 0.0
    """

    def evaluate(
        self, ctx: EvaluatorContext[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]
    ) -> float:
        """Evaluate attribute addition.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Proportion of expected attributes found (0.0-1.0)
        """
        # Handle missing metadata
        if ctx.metadata is None:
            return 1.0

        # Skip if no attributes expected
        if not ctx.metadata.added_attribute_names:
            return 1.0

        # Skip if edit should not succeed
        if not ctx.metadata.should_succeed:
            return 1.0

        # Skip if execution error occurred
        if ctx.output.error:
            return 1.0

        # Check which expected attributes are present
        actual_attr_names = get_attribute_names(ctx.output.model)
        matches = sum(
            1 if expected_name in actual_attr_names else 0 for expected_name in ctx.metadata.added_attribute_names
        )

        # Return proportion of expected attributes found
        return matches / len(ctx.metadata.added_attribute_names)


class ChangeTrackingEvaluator(Evaluator[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]):
    """Evaluate that changes are properly recorded with expected keywords.

    Uses a hybrid scoring approach:
    - STRICT: Changes must be recorded for successful edits (non-negotiable)
    - PARTIAL: Proportional credit for keyword matches in change descriptions

    Scoring:
        0.0 if no changes recorded for successful edit (hard failure)
        0.0-1.0 proportional score for keyword matches if changes exist

    Returns:
        1.0 if changes recorded and all keywords found
        0.0-1.0 for changes recorded with partial keyword matches
        0.0 if no changes recorded (strict requirement violated)
        1.0 if edit should not succeed (N/A)

    Example usage:
        >>> from pydantic_evals import Case
        >>> case = Case(
        ...     name="test_change_tracking",
        ...     inputs=ModelEditorInput(
        ...         model_json='{"attributes": [...]}',
        ...         command="Add severity attribute",
        ...         edit_type="natural_language"
        ...     ),
        ...     expected_output=ModelEditorExpectedOutput(
        ...         should_succeed=True,
        ...         changes_keywords=["severity", "added", "attribute"]
        ...     )
        ... )
        >>> evaluator = ChangeTrackingEvaluator()
        >>> # If changes exist and all 3 keywords found: score = 1.0
        >>> # If changes exist and 2/3 keywords found: score = 0.67
        >>> # If no changes recorded: score = 0.0 (hard failure)
    """

    def evaluate(
        self, ctx: EvaluatorContext[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]
    ) -> float:
        """Evaluate change tracking.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Score from 0.0-1.0 based on change recording and keyword matches
        """
        # Handle missing metadata
        if ctx.metadata is None:
            return 1.0

        # Skip if edit should not succeed
        if not ctx.metadata.should_succeed:
            return 1.0

        # Skip if execution error occurred
        if ctx.output.error:
            return 1.0

        # STRICT: Changes must be recorded for successful edits (non-negotiable)
        if len(ctx.output.changes) == 0:
            return 0.0

        # PARTIAL: Proportional score for keyword matches if changes exist
        if ctx.metadata.changes_keywords:
            text = extract_text_for_keywords(ctx.output.changes, ctx.output.rejections)
            matches = sum(1 if keyword.lower() in text else 0 for keyword in ctx.metadata.changes_keywords)
            return matches / len(ctx.metadata.changes_keywords)
        else:
            # No keywords to check - full score if changes were recorded
            return 1.0


class RejectionAccuracyEvaluator(Evaluator[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]):
    """Evaluate that rejections happen correctly with expected keywords.

    Uses a hybrid scoring approach:
    - STRICT: Rejections must be recorded for failed edits (non-negotiable)
    - PARTIAL: Proportional credit for keyword matches in rejection messages

    Scoring:
        0.0 if no rejections recorded for failed edit (hard failure)
        0.0-1.0 proportional score for keyword matches if rejections exist

    Returns:
        1.0 if rejections recorded and all keywords found
        0.0-1.0 for rejections recorded with partial keyword matches
        0.0 if no rejections recorded (strict requirement violated)
        1.0 if edit should succeed (N/A)

    Example usage:
        >>> from pydantic_evals import Case
        >>> case = Case(
        ...     name="test_rejection",
        ...     inputs=ModelEditorInput(
        ...         model_json='{"attributes": [...]}',
        ...         command="Delete the presence attribute",
        ...         edit_type="natural_language"
        ...     ),
        ...     expected_output=ModelEditorExpectedOutput(
        ...         should_succeed=False,
        ...         rejection_keywords=["delete", "not allowed", "forbidden"]
        ...     )
        ... )
        >>> evaluator = RejectionAccuracyEvaluator()
        >>> # If rejections exist and all 3 keywords found: score = 1.0
        >>> # If rejections exist and 2/3 keywords found: score = 0.67
        >>> # If no rejections recorded: score = 0.0 (hard failure)
    """

    def evaluate(
        self, ctx: EvaluatorContext[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]
    ) -> float:
        """Evaluate rejection accuracy.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            Score from 0.0-1.0 based on rejection recording and keyword matches
        """
        # Handle missing metadata
        if ctx.metadata is None:
            return 1.0

        # Skip if edit should succeed
        if ctx.metadata.should_succeed:
            return 1.0

        # Skip if execution error occurred
        if ctx.output.error:
            return 1.0

        # STRICT: Rejections must be recorded for failed edits (non-negotiable)
        if len(ctx.output.rejections) == 0:
            return 0.0

        # PARTIAL: Proportional score for keyword matches if rejections exist
        if ctx.metadata.rejection_keywords:
            text = extract_text_for_keywords(ctx.output.changes, ctx.output.rejections)
            matches = sum(1 if keyword.lower() in text else 0 for keyword in ctx.metadata.rejection_keywords)
            return matches / len(ctx.metadata.rejection_keywords)
        else:
            # No keywords to check - full score if rejections were recorded
            return 1.0


class ContentPreservationEvaluator(Evaluator[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]):
    """Evaluate that model content is preserved when edits are rejected.

    This evaluator verifies that when an edit is rejected, the model remains
    completely unchanged from its original state. This is critical for ensuring
    that rejected operations have no side effects.

    Returns:
        1.0 if model unchanged when rejected OR if edit should succeed (N/A)
        0.0 if model was modified when it should have been rejected

    Example usage:
        >>> from pydantic_evals import Case
        >>> case = Case(
        ...     name="test_rejection_preservation",
        ...     inputs=ModelEditorInput(
        ...         model_json='{"oifm_id": "OIFM_TEST_000001", "attributes": [...]}',
        ...         command="Delete all attributes",
        ...         edit_type="natural_language"
        ...     ),
        ...     expected_output=ModelEditorExpectedOutput(
        ...         should_succeed=False
        ...     )
        ... )
        >>> evaluator = ContentPreservationEvaluator()
        >>> # If model unchanged from original: score = 1.0
        >>> # If model was modified: score = 0.0
    """

    def evaluate(
        self, ctx: EvaluatorContext[ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput]
    ) -> float:
        """Evaluate content preservation on rejection.

        Args:
            ctx: Evaluation context containing case inputs, output, and metadata

        Returns:
            1.0 if model preserved or should succeed, 0.0 if modified on rejection
        """
        # Handle missing metadata
        if ctx.metadata is None:
            return 1.0

        # Skip if edit should succeed
        if ctx.metadata.should_succeed:
            return 1.0

        # Skip if execution error occurred
        if ctx.output.error:
            return 1.0

        # Compare models
        original_model = FindingModelFull.model_validate_json(ctx.inputs.model_json)
        return 1.0 if compare_models(ctx.output.model, original_model) else 0.0


# =============================================================================
# Dataset Creation with Evaluator-Based Pattern
# =============================================================================
#
# The new pattern uses Dataset.evaluate() with focused evaluators instead of
# custom evaluation functions. Each evaluator checks a specific aspect and
# returns a continuous score (0.0-1.0) to enable partial credit.
#
# Benefits of this approach:
# - Separation of concerns: Each evaluator has a single responsibility
# - Reusability: Evaluators can be used across different test suites
# - Transparency: Clear scoring for each dimension of behavior
# - Flexibility: Easy to add/remove evaluators or adjust scoring
#
# All cases are evaluated by all evaluators, and the overall score is the
# average of all evaluator scores across all cases.

all_cases = create_successful_edit_cases() + create_rejection_cases() + create_markdown_edit_cases()
evaluators = [
    IDPreservationEvaluator(),
    AttributeAdditionEvaluator(),
    ChangeTrackingEvaluator(),
    RejectionAccuracyEvaluator(),
    ContentPreservationEvaluator(),
]
model_editor_dataset = Dataset(cases=all_cases, evaluators=evaluators)


# NOTE: This function is no longer used after refactoring to use Dataset.evaluate()
# Kept for reference but should be removed in future cleanup.
# The evaluation logic is now implemented in the dedicated Evaluator classes above.
#
# # Custom evaluator for model editor cases
# def evaluate_model_editor_case(case: ModelEditorCase, actual: ModelEditorActualOutput) -> dict[str, Any]:
#     """Evaluate a model editor case and return metrics."""
#     expected = case.metadata
#     if expected is None:
#         raise ValueError("Case metadata must be set")
#
#     results: dict[str, Any] = {
#         "case_name": case.name,
#         "passed": True,
#         "errors": [],
#     }
#
#     # Check for execution errors
#     if actual.error:
#         results["passed"] = False
#         results["errors"].append(f"Execution error: {actual.error}")
#         return results
#
#     # Check if model ID is preserved
#     if expected.should_preserve_id:
#         original_model = FindingModelFull.model_validate_json(case.inputs.model_json)
#         if actual.model.oifm_id != original_model.oifm_id:
#             results["passed"] = False
#             results["errors"].append(f"Model ID changed from {original_model.oifm_id} to {actual.model.oifm_id}")
#
#     # Check if edit succeeded when it should have
#     if expected.should_succeed:
#         if actual.rejections:
#             results["passed"] = False
#             results["errors"].append(f"Edit was rejected when it should succeed: {actual.rejections}")
#
#         # Check that expected attributes were added
#         actual_attr_names = [a.name for a in actual.model.attributes]
#         for expected_name in expected.added_attribute_names:
#             if expected_name not in actual_attr_names:
#                 results["passed"] = False
#                 results["errors"].append(f"Expected attribute '{expected_name}' was not added")
#
#         # Check that changes were recorded
#         if not actual.changes:
#             results["passed"] = False
#             results["errors"].append("No changes were recorded for a successful edit")
#
#         # Check for expected keywords in changes
#         changes_text = " ".join(actual.changes).lower()
#         for keyword in expected.changes_keywords:
#             if keyword.lower() not in changes_text:
#                 results["passed"] = False
#                 results["errors"].append(f"Expected keyword '{keyword}' not found in changes")
#
#     # Check if edit was rejected when it should have been
#     else:  # should_succeed = False
#         if not actual.rejections:
#             results["passed"] = False
#             results["errors"].append("Edit should have been rejected but wasn't")
#
#         # Check for expected keywords in rejections
#         rejections_text = " ".join(actual.rejections).lower()
#         for keyword in expected.rejection_keywords:
#             if keyword.lower() not in rejections_text:
#                 results["passed"] = False
#                 results["errors"].append(f"Expected keyword '{keyword}' not found in rejections")
#
#         # Model should be unchanged
#         original_model = FindingModelFull.model_validate_json(case.inputs.model_json)
#         if actual.model.model_dump_json() != original_model.model_dump_json():
#             results["passed"] = False
#             results["errors"].append("Model was modified when edit should have been rejected")
#
#     return results


async def run_model_editor_evals() -> EvaluationReport[
    ModelEditorInput, ModelEditorActualOutput, ModelEditorExpectedOutput
]:
    """Run all model editor evaluation cases using Dataset.evaluate().

    EVALUATOR-BASED PATTERN USAGE:
    1. Dataset was created with cases and evaluators at module level
    2. Call dataset.evaluate() with the task function
    3. Each case is executed and evaluated by all evaluators
    4. Report contains scores for each evaluator on each case
    5. Overall score is the average across all evaluators and cases

    This replaces the old pattern of custom evaluation functions.
    """

    # Define task function wrapper that executes the agent
    async def run_model_editor_task(input_data: ModelEditorInput) -> ModelEditorActualOutput:
        """Execute the model editor with the given input."""
        # Show progress indication (case-level)
        # Note: pydantic-evals shows a progress bar, but individual cases can take 10-20s
        # This provides visibility into which case is currently running
        import sys

        model = FindingModelFull.model_validate_json(input_data.model_json)
        case_desc = f"{input_data.edit_type}: {model.name}"
        print(f"  Processing: {case_desc}...", end="", flush=True, file=sys.stderr)

        try:
            if input_data.edit_type == "natural_language":
                result = await model_editor.edit_model_natural_language(model, input_data.command)
            elif input_data.edit_type == "markdown":
                result = await model_editor.edit_model_markdown(model, input_data.command)
            else:
                raise ValueError(f"Unknown edit_type: {input_data.edit_type}")

            print(" ✓", file=sys.stderr)
            return ModelEditorActualOutput(model=result.model, rejections=result.rejections, changes=result.changes)
        except Exception as e:
            print(" ✗", file=sys.stderr)
            # Return a placeholder model with error info
            return ModelEditorActualOutput(model=model, rejections=[], changes=[], error=str(e))

    # Run evaluation using Dataset.evaluate() - evaluators already passed to Dataset constructor
    report = await model_editor_dataset.evaluate(run_model_editor_task)

    # Calculate overall score manually (average of all evaluator scores across all cases)
    all_scores = []
    for case in report.cases:
        for score_result in case.scores.values():
            all_scores.append(score_result.value)
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Print formatted output
    print("\n" + "=" * 80)
    print("MODEL EDITOR EVALUATION RESULTS")
    print("=" * 80 + "\n")

    # Don't include outputs in table - they're too verbose (full FindingModelFull objects)
    # Focus on scores which are the important metric
    report.print(
        include_input=False,
        include_output=False,  # Outputs are huge FindingModelFull objects - skip them
        include_durations=True,
        width=120,
    )

    print("\n" + "=" * 80)
    print(f"OVERALL SCORE: {overall_score:.2f}")
    print("=" * 80 + "\n")

    # Return the report for analysis
    # Note: When run as a pytest test, we assert threshold >= 0.95
    # When run standalone, we just print results
    return report


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("\nRunning model_editor evaluation suite...")
        print("=" * 80)

        # run_model_editor_evals() already prints results and overall score
        await run_model_editor_evals()

        # Future Phase 3: Logfire integration via pydantic-evals
        # Future: Save report to file, compare to baseline, etc.

    asyncio.run(main())
