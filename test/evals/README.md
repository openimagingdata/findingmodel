# Model Editor Evaluations

This directory contains evaluation test cases for the `model_editor` functionality using the Pydantic Evals framework.

## Overview

The evaluation suite tests the model editor's ability to:

1. **Accept safe edits**: Adding attributes, choice values, synonyms, tags, descriptions
2. **Reject unsafe edits**: Renaming/deleting attributes, changing IDs, modifying semantics
3. **Work with both input formats**: Natural language commands and markdown-based edits

## Structure

- `test_model_editor_evals.py`: Main evaluation test file containing all test cases
- `add_case_example.py`: Template examples for adding new test cases
- `list_cases.py`: Utility script to list all available test cases
- `README.md`: This documentation file
- `IMPLEMENTATION.md`: Technical implementation summary

## Test Cases

### Successful Edit Cases

These cases verify that safe edits are properly applied:

1. **add_severity_attribute**: Add a choice attribute with multiple values
2. **add_numeric_size_attribute**: Add a numeric attribute with constraints
3. **add_synonyms**: Add synonyms to the model metadata
4. **add_choice_value_to_existing**: Extend an existing choice attribute with new values
5. **add_description_enhancement**: Enhance model description with more detail
6. **markdown_add_attribute**: Add an attribute via markdown editing

### Rejection Cases

These cases verify that unsafe edits are properly rejected:

1. **reject_rename_attribute**: Attempt to rename an existing attribute (should reject)
2. **reject_delete_attribute**: Attempt to delete an attribute (should reject)
3. **reject_change_model_id**: Attempt to change the model ID (should reject)
4. **reject_remove_choice_value**: Attempt to remove a choice value (should reject)
5. **reject_change_attribute_type**: Attempt to change attribute type (should reject)
6. **markdown_reject_delete_attribute**: Attempt to delete via markdown (should reject)

## Running the Evaluations

### Run all evaluation cases (requires OpenAI API):

```bash
# Using pytest (from repository root)
pytest test/evals/test_model_editor_evals.py::test_run_model_editor_evals -v -s

# Or with task runner
task test test/evals/test_model_editor_evals.py::test_run_model_editor_evals
```

### Run a single test case without API calls (uses TestModel):

```bash
pytest test/evals/test_model_editor_evals.py::test_single_successful_case -v
```

### Run as a standalone script:

```bash
cd test/evals
python test_model_editor_evals.py
```

### List all available test cases:

```bash
cd test/evals
python list_cases.py
# Or make it executable and run directly
./list_cases.py
```

## Adding New Test Cases

To add new evaluation cases:

1. **For successful edits**, add to `create_successful_edit_cases()`:
   ```python
   cases.append(
       ModelEditorCase(
           name="your_case_name",
           model_json=load_fm_json("your_model.fm.json"),
           command="Your editing command",
           edit_type="natural_language",  # or "markdown"
           should_succeed=True,
           added_attribute_names=["expected_attr_name"],
           changes_keywords=["keyword1", "keyword2"],
       )
   )
   ```

2. **For rejection cases**, add to `create_rejection_cases()`:
   ```python
   cases.append(
       ModelEditorCase(
           name="reject_your_case",
           model_json=load_fm_json("your_model.fm.json"),
           command="Command that should be rejected",
           edit_type="natural_language",
           should_succeed=False,
           rejection_keywords=["expected", "in", "rejection"],
       )
   )
   ```

3. **For markdown edits**, add to `create_markdown_edit_cases()`:
   ```python
   model = FindingModelFull.model_validate_json(your_json)
   base_md = model_editor.export_model_for_editing(model)
   modified_md = base_md + "\n### new_attribute\n...\n"
   
   cases.append(
       ModelEditorCase(
           name="markdown_your_case",
           model_json=your_json,
           command=modified_md,
           edit_type="markdown",
           should_succeed=True,
           added_attribute_names=["new_attribute"],
       )
   )
   ```

## Test Data

The evaluation cases use real finding model JSON files from `test/data/defs/`:

- `pulmonary_embolism.fm.json`
- `aortic_dissection.fm.json`
- `breast_density.fm.json`
- `ventricular_diameters.fm.json`
- `abdominal_aortic_aneurysm.fm.json`
- `breast_malignancy_risk.fm.json`

You can add new test cases using any of these models or add new models to the data directory.

## Evaluation Metrics

Each case is evaluated on:

- **ID Preservation**: Model and attribute IDs must remain unchanged
- **Success/Rejection**: Edits succeed or are rejected as expected
- **Attribute Addition**: Expected attributes are added with correct properties
- **Changes Tracking**: Successful edits are recorded in the changes list
- **Rejection Tracking**: Rejected edits are recorded with clear reasons
- **Content Preservation**: Original content is preserved when edits are rejected

## Dependencies

- `pydantic-evals`: Evaluation framework
- `pydantic-ai`: For agent testing (TestModel)
- `pytest`: Test runner
- `pytest-asyncio`: Async test support

Install with:

```bash
pip install pydantic-evals pydantic-ai pytest pytest-asyncio
```

## CI/CD Integration

The evaluation tests are marked with `@pytest.mark.callout` because they require API access. They should be run:

- During development when testing model_editor changes
- In CI/CD with proper API credentials
- As part of release validation

To run without API calls for quick validation, use the `test_single_successful_case` test which uses TestModel mocking.
