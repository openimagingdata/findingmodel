#!/usr/bin/env python3
"""List all available evaluation cases for model_editor.

This utility script displays all test cases defined in the evaluation suite,
organized by category.
"""

from test_model_editor_evals import (
    create_markdown_edit_cases,
    create_rejection_cases,
    create_successful_edit_cases,
)


def main() -> None:
    """Display all evaluation cases organized by category."""
    print("=" * 80)
    print("MODEL EDITOR EVALUATION CASES")
    print("=" * 80)

    # Successful edit cases
    successful_cases = create_successful_edit_cases()
    print(f"\n📊 SUCCESSFUL EDIT CASES ({len(successful_cases)} cases)")
    print("-" * 80)
    for i, case in enumerate(successful_cases, 1):
        print(f"\n{i}. {case.name}")
        print(f"   Type: {case.inputs.edit_type}")
        print(f"   Command: {case.inputs.command[:80]}...")
        if case.expected_output.added_attribute_names:
            print(f"   Expected attributes: {', '.join(case.expected_output.added_attribute_names)}")

    # Rejection cases
    rejection_cases = create_rejection_cases()
    print(f"\n\n🚫 REJECTION CASES ({len(rejection_cases)} cases)")
    print("-" * 80)
    for i, case in enumerate(rejection_cases, 1):
        print(f"\n{i}. {case.name}")
        print(f"   Type: {case.inputs.edit_type}")
        print(f"   Command: {case.inputs.command[:80]}...")
        if case.expected_output.rejection_keywords:
            print(f"   Expected keywords: {', '.join(case.expected_output.rejection_keywords)}")

    # Markdown edit cases
    markdown_cases = create_markdown_edit_cases()
    print(f"\n\n📝 MARKDOWN EDIT CASES ({len(markdown_cases)} cases)")
    print("-" * 80)
    for i, case in enumerate(markdown_cases, 1):
        print(f"\n{i}. {case.name}")
        print(f"   Type: {case.inputs.edit_type}")
        should_succeed = "should succeed" if case.expected_output.should_succeed else "should be rejected"
        print(f"   Expected: {should_succeed}")
        if case.expected_output.added_attribute_names:
            print(f"   Expected attributes: {', '.join(case.expected_output.added_attribute_names)}")

    # Summary
    total = len(successful_cases) + len(rejection_cases) + len(markdown_cases)
    print("\n" + "=" * 80)
    print(f"TOTAL: {total} evaluation cases")
    print("=" * 80)
    print("\nTo run these cases:")
    print("  pytest test/evals/test_model_editor_evals.py::test_run_model_editor_evals -v -s")
    print("\nTo add new cases:")
    print("  See test/evals/add_case_example.py for templates")
    print("  Read test/evals/README.md for detailed instructions")
    print()


if __name__ == "__main__":
    main()
