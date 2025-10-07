"""Example template for adding new evaluation cases to test_model_editor_evals.py

This file shows how to create new evaluation cases. Copy the examples below
and add them to the appropriate function in test_model_editor_evals.py.
"""


# Example 1: Add a successful edit case
def example_add_successful_case() -> object:
    """Example of adding a new successful edit case."""
    from test_model_editor_evals import ModelEditorCase, load_fm_json

    # Load a model to test with
    model_json = load_fm_json("pulmonary_embolism.fm.json")

    # Create a new case
    new_case = ModelEditorCase(
        name="add_laterality_attribute",  # Unique name for this case
        model_json=model_json,
        command="Add a choice attribute named 'laterality' with values: left, right, bilateral",
        edit_type="natural_language",
        should_succeed=True,
        added_attribute_names=["laterality"],  # Attributes we expect to be added
        changes_keywords=["laterality", "added"],  # Keywords we expect in the changes summary
    )

    # Add this case to the list in create_successful_edit_cases() function
    return new_case


# Example 2: Add a rejection case
def example_add_rejection_case() -> object:
    """Example of adding a new rejection case."""
    from test_model_editor_evals import ModelEditorCase, load_fm_json

    # Load a model to test with
    model_json = load_fm_json("aortic_dissection.fm.json")

    # Create a rejection case
    new_case = ModelEditorCase(
        name="reject_change_value_name",
        model_json=model_json,
        command="In the 'presence' attribute, rename the 'absent' value to 'not present'",
        edit_type="natural_language",
        should_succeed=False,  # This should be rejected
        rejection_keywords=["rename", "value", "not allowed"],  # Keywords we expect in rejections
    )

    # Add this case to the list in create_rejection_cases() function
    return new_case


# Example 3: Add a markdown-based edit case
def example_add_markdown_case() -> object:
    """Example of adding a markdown-based edit case."""
    from test_model_editor_evals import ModelEditorCase, load_fm_json

    from findingmodel.finding_model import FindingModelFull
    from findingmodel.tools import model_editor

    # Load and parse the model
    model_json = load_fm_json("breast_density.fm.json")
    model = FindingModelFull.model_validate_json(model_json)

    # Export to markdown and modify it
    base_md = model_editor.export_model_for_editing(model)

    # Add a new attribute section to the markdown
    enhanced_md = (
        base_md
        + """
### calcifications

Presence and pattern of calcifications

- absent: No calcifications present
- scattered: Scattered benign calcifications
- clustered: Clustered suspicious calcifications

"""
    )

    # Create the case
    new_case = ModelEditorCase(
        name="markdown_add_calcifications",
        model_json=model_json,
        command=enhanced_md,
        edit_type="markdown",
        should_succeed=True,
        added_attribute_names=["calcifications"],
        changes_keywords=["calcifications", "added"],
    )

    # Add this case to the list in create_markdown_edit_cases() function
    return new_case


# Example 4: Test adding numeric attributes with units
def example_add_numeric_with_units() -> object:
    """Example of testing numeric attribute addition with units and ranges."""
    from test_model_editor_evals import ModelEditorCase, load_fm_json

    model_json = load_fm_json("ventricular_diameters.fm.json")

    new_case = ModelEditorCase(
        name="add_wall_thickness_numeric",
        model_json=model_json,
        command="Add a numeric attribute 'wall thickness' with minimum 1, maximum 20, unit mm, not required",
        edit_type="natural_language",
        should_succeed=True,
        added_attribute_names=["wall thickness"],
        changes_keywords=["wall thickness", "numeric", "added"],
    )

    return new_case


# Example 5: Test edge case - ambiguous command should be rejected
def example_reject_ambiguous_command() -> object:
    """Example of testing that ambiguous commands are rejected."""
    from test_model_editor_evals import ModelEditorCase, load_fm_json

    model_json = load_fm_json("pulmonary_embolism.fm.json")

    new_case = ModelEditorCase(
        name="reject_ambiguous_modification",
        model_json=model_json,
        command="Modify the model appropriately",  # Too vague
        edit_type="natural_language",
        should_succeed=False,
        rejection_keywords=["ambiguous", "unclear", "specific"],
    )

    return new_case


if __name__ == "__main__":
    print("This file contains example templates for adding new evaluation cases.")
    print("Copy the examples and add them to test_model_editor_evals.py in the appropriate functions:")
    print("  - create_successful_edit_cases()")
    print("  - create_rejection_cases()")
    print("  - create_markdown_edit_cases()")
    print("\nSee the docstrings in each example function for details.")
