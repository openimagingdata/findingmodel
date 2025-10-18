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

The evaluator-based pattern makes adding cases straightforward. Just specify what you expect, and the evaluators automatically check it.

### Example: Add a successful edit case

```python
# In create_successful_edit_cases():
cases.append(
    ModelEditorCase(
        name="add_laterality_attribute",
        model_json=load_fm_json("pulmonary_embolism.fm.json"),
        command="Add a choice attribute named 'laterality' with values: left, right, bilateral",
        edit_type="natural_language",
        should_succeed=True,
        added_attribute_names=["laterality"],  # AttributeAdditionEvaluator checks this
        changes_keywords=["laterality", "added"],  # ChangeTrackingEvaluator checks these
    )
)
```

**The evaluators will automatically check:**
- ID preserved (IDPreservationEvaluator - strict)
- "laterality" attribute added (AttributeAdditionEvaluator - partial credit)
- Changes recorded with keywords (ChangeTrackingEvaluator - hybrid)

### Example: Add a rejection case

```python
# In create_rejection_cases():
cases.append(
    ModelEditorCase(
        name="reject_rename_attribute",
        model_json=load_fm_json("pulmonary_embolism.fm.json"),
        command="Rename the 'presence' attribute to 'occurrence'",
        edit_type="natural_language",
        should_succeed=False,
        rejection_keywords=["rename", "not allowed"],  # RejectionAccuracyEvaluator checks these
    )
)
```

**The evaluators will automatically check:**
- ID preserved (IDPreservationEvaluator - strict)
- Rejections recorded (RejectionAccuracyEvaluator - strict)
- Keywords in rejection message (RejectionAccuracyEvaluator - partial credit)
- Model unchanged from original (ContentPreservationEvaluator - strict)

### Example: Add a markdown edit case

```python
# In create_markdown_edit_cases():
model = FindingModelFull.model_validate_json(load_fm_json("breast_density.fm.json"))
base_md = model_editor.export_model_for_editing(model)
enhanced_md = base_md + "\n### calcifications\n\nPresence of calcifications\n\n- absent\n- present\n\n"

cases.append(
    ModelEditorCase(
        name="markdown_add_calcifications",
        model_json=model.model_dump_json(),
        command=enhanced_md,
        edit_type="markdown",
        should_succeed=True,
        added_attribute_names=["calcifications"],
        changes_keywords=["calcifications", "added"],
    )
)
```

See `add_case_example.py` for more detailed examples with explanations of what each evaluator checks.

## Test Data

The evaluation cases use real finding model JSON files from `test/data/defs/`:

- `pulmonary_embolism.fm.json`
- `aortic_dissection.fm.json`
- `breast_density.fm.json`
- `ventricular_diameters.fm.json`
- `abdominal_aortic_aneurysm.fm.json`
- `breast_malignancy_risk.fm.json`

You can add new test cases using any of these models or add new models to the data directory.

## Evaluation Approach

The evaluation suite uses **focused evaluators** from the Pydantic Evals framework. Each evaluator checks a specific aspect of model editor behavior, with **hybrid scoring**:

- **Strict requirements**: Non-negotiable checks (0.0 or 1.0)
- **Partial credit**: Quality metrics with proportional scoring (0.0-1.0)

### Evaluators

Defined in `test_model_editor_evals.py` as specialized evaluators (note: `test/evals/base.py` contains generic reusable evaluators):

1. **IDPreservationEvaluator** (strict): Model IDs must never change
2. **AttributeAdditionEvaluator** (partial): Proportional score for expected attributes added
3. **ChangeTrackingEvaluator** (hybrid):
   - Strict: Changes must be recorded for successful edits
   - Partial: Keyword matches in change descriptions
4. **RejectionAccuracyEvaluator** (hybrid):
   - Strict: Rejections must be recorded for failed edits
   - Partial: Keyword matches in rejection messages
5. **ContentPreservationEvaluator** (strict): Model unchanged when edits rejected

Each case is evaluated by all evaluators using `Dataset.evaluate()`, producing an overall score from 0.0-1.0.

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
