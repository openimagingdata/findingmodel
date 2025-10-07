"""Evaluation tests for model_editor using pydantic-evals framework.

This module defines evaluation cases for testing the model_editor functionality,
including both successful edits and rejection cases.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_evals import Case, Dataset

from findingmodel.finding_model import FindingModelFull
from findingmodel.tools import model_editor


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


class ModelEditorCase(Case[ModelEditorInput, ModelEditorExpectedOutput, ModelEditorActualOutput]):
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
        expected_output = ModelEditorExpectedOutput(
            should_succeed=should_succeed,
            should_preserve_id=should_preserve_id,
            added_attribute_names=added_attribute_names or [],
            rejection_keywords=rejection_keywords or [],
            changes_keywords=changes_keywords or [],
        )
        super().__init__(name=name, inputs=inputs, expected_output=expected_output)

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
    test_data_dir = Path(__file__).parent.parent / "data" / "defs"
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


# Create the dataset
all_cases = create_successful_edit_cases() + create_rejection_cases() + create_markdown_edit_cases()
model_editor_dataset = Dataset(name="model_editor_evaluation", cases=all_cases)


# Custom evaluator for model editor cases
def evaluate_model_editor_case(case: ModelEditorCase, actual: ModelEditorActualOutput) -> dict[str, Any]:  # noqa: C901
    """Evaluate a model editor case and return metrics."""
    expected = case.expected_output
    results: dict[str, Any] = {
        "case_name": case.name,
        "passed": True,
        "errors": [],
    }

    # Check for execution errors
    if actual.error:
        results["passed"] = False
        results["errors"].append(f"Execution error: {actual.error}")
        return results

    # Check if model ID is preserved
    if expected.should_preserve_id:
        original_model = FindingModelFull.model_validate_json(case.inputs.model_json)
        if actual.model.oifm_id != original_model.oifm_id:
            results["passed"] = False
            results["errors"].append(f"Model ID changed from {original_model.oifm_id} to {actual.model.oifm_id}")

    # Check if edit succeeded when it should have
    if expected.should_succeed:
        if actual.rejections:
            results["passed"] = False
            results["errors"].append(f"Edit was rejected when it should succeed: {actual.rejections}")

        # Check that expected attributes were added
        actual_attr_names = [a.name for a in actual.model.attributes]
        for expected_name in expected.added_attribute_names:
            if expected_name not in actual_attr_names:
                results["passed"] = False
                results["errors"].append(f"Expected attribute '{expected_name}' was not added")

        # Check that changes were recorded
        if not actual.changes:
            results["passed"] = False
            results["errors"].append("No changes were recorded for a successful edit")

        # Check for expected keywords in changes
        changes_text = " ".join(actual.changes).lower()
        for keyword in expected.changes_keywords:
            if keyword.lower() not in changes_text:
                results["passed"] = False
                results["errors"].append(f"Expected keyword '{keyword}' not found in changes")

    # Check if edit was rejected when it should have been
    else:  # should_succeed = False
        if not actual.rejections:
            results["passed"] = False
            results["errors"].append("Edit should have been rejected but wasn't")

        # Check for expected keywords in rejections
        rejections_text = " ".join(actual.rejections).lower()
        for keyword in expected.rejection_keywords:
            if keyword.lower() not in rejections_text:
                results["passed"] = False
                results["errors"].append(f"Expected keyword '{keyword}' not found in rejections")

        # Model should be unchanged
        original_model = FindingModelFull.model_validate_json(case.inputs.model_json)
        if actual.model.model_dump_json() != original_model.model_dump_json():
            results["passed"] = False
            results["errors"].append("Model was modified when edit should have been rejected")

    return results


@pytest.mark.callout
@pytest.mark.asyncio
async def test_run_model_editor_evals() -> None:
    """Run all model editor evaluation cases."""
    results = []

    for case in all_cases:
        actual_output = await case._execute(case.inputs)
        evaluation = evaluate_model_editor_case(case, actual_output)
        results.append(evaluation)

    # Print summary
    print("\n" + "=" * 80)
    print("MODEL EDITOR EVALUATION RESULTS")
    print("=" * 80)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for result in results:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"\n{status}: {result['case_name']}")
        if result["errors"]:
            for error in result["errors"]:
                print(f"  - {error}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{total} cases passed ({100 * passed // total}%)")
    print("=" * 80 + "\n")

    # Assert that all cases passed
    assert passed == total, f"Only {passed}/{total} evaluation cases passed"


@pytest.mark.asyncio
async def test_single_successful_case() -> None:
    """Test a single successful case without calling out to API (for quick testing)."""
    from pydantic_ai.models.test import TestModel

    pe_json = load_fm_json("pulmonary_embolism.fm.json")
    model = FindingModelFull.model_validate_json(pe_json)

    # Build a modified model with severity attribute
    from findingmodel.tools.add_ids import PLACEHOLDER_ATTRIBUTE_ID

    base_data = model.model_dump()
    base_data["attributes"].append({
        "oifma_id": PLACEHOLDER_ATTRIBUTE_ID,
        "name": "severity",
        "type": "choice",
        "values": [
            {"value_code": f"{PLACEHOLDER_ATTRIBUTE_ID}.0", "name": "mild"},
            {"value_code": f"{PLACEHOLDER_ATTRIBUTE_ID}.1", "name": "moderate"},
            {"value_code": f"{PLACEHOLDER_ATTRIBUTE_ID}.2", "name": "severe"},
        ],
        "required": False,
    })
    modified_model = FindingModelFull.model_validate(base_data)

    # Create mock result
    mock_output = model_editor.EditResult(
        model=modified_model,
        rejections=[],
        changes=["Added severity attribute with values mild, moderate, severe"],
    )

    # Create agent with mock
    agent = model_editor.create_edit_agent()
    command = "Add a new attribute named 'severity' of type choice with values: mild, moderate, severe"

    with agent.override(model=TestModel(custom_output_args=mock_output)):
        result = await model_editor.edit_model_natural_language(model, command, agent=agent)

    # Verify the result matches expectations
    assert result.model.oifm_id == model.oifm_id
    assert "severity" in [a.name for a in result.model.attributes]
    assert not result.rejections
    assert result.changes


if __name__ == "__main__":
    import asyncio

    # Run the evaluations
    asyncio.run(test_run_model_editor_evals())
