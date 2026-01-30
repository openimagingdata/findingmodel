# Tasks: AI-Based Finding Model Editor

**Feature Branch**: `001-ai-based-finding`
**Feature Directory**: `/Users/talkasab/Repos/findingmodel/specs/001-ai-based-finding/`

## Task List

### T001. [P] Project Setup and Linting ✅ COMPLETED
- Ensure all dependencies are installed (pydantic-ai, pytest, etc.)
- Confirm linting and formatting tools are configured
- Path: `/Users/talkasab/Repos/findingmodel/`

### T002. [P] Contract Test: Natural Language Editing ✅ COMPLETED
- Write unit tests for `edit_model_natural_language` as specified in `contracts/natural_language_api.md`
- Use `TestModel` to avoid real LLM calls
- Path: `src/findingmodel/tools/model_editor.py`, `test/test_model_editor.py`
- Status: Unit tests implemented with TestModel override pattern

### T003. [P] Contract Test: Markdown Editing ✅ COMPLETED
- Write unit tests for `edit_model_markdown` and `export_model_for_editing` as specified in `contracts/markdown_api.md`
- Use `TestModel` to avoid real LLM calls
- Path: `src/findingmodel/tools/model_editor.py`, `test/test_model_editor.py`
- Status: Unit tests implemented (function renamed to export_model_for_editing)

### T004. Implement: Natural Language Editing ✅ COMPLETED
- Implement `edit_model_natural_language(model: FindingModelFull, command: str) -> EditResult`
- Use Pydantic AI agent pattern
- Path: `src/findingmodel/tools/model_editor.py`
- Depends on: T002
- Status: Implemented with agent factories and output validation (source param removed)

### T005. Implement: Markdown Editing ✅ COMPLETED
- Implement `export_model_for_editing(model: FindingModelFull) -> str` and `edit_model_markdown(model: FindingModelFull, edited_markdown: str) -> EditResult`
- Path: `src/findingmodel/tools/model_editor.py`
- Depends on: T003
- Status: Implemented with agent factories and validation (source param removed, export function renamed)

### T006. Demo Script: Natural Language Editing ✅ COMPLETED
- Create `scripts/edit_finding_model.py` to demonstrate natural language editing
- Path: `scripts/edit_finding_model.py`
- Depends on: T004
- Status: Demo script created and updated to show rejections

### T007. Demo Script: Markdown Editing ✅ COMPLETED
- Create `scripts/edit_finding_model.py` to demonstrate markdown editing
- Path: `scripts/edit_finding_model.py`
- Depends on: T005
- Status: Demo script created with proper editable Markdown format and rejection display

### T008. [P] Polish: Documentation and Final Tests ✅ COMPLETED
- Update documentation and ensure all tests pass
- Path: `/Users/talkasab/Repos/findingmodel/`
- Depends on: T006, T007
- Status: Tests passing (137 passed, 22 deselected), validation framework implemented, callout tests added
- Documentation: Added model editing section to README.md with comprehensive examples and API documentation

## Parallelization Guidance
- T002 and T003 can be run in parallel ([P])
- T004 and T005 can be run in parallel after their respective contract tests
- T006 and T007 can be run in parallel after their respective implementations
- T008 can only be run after all previous tasks are complete

---

**To execute in parallel:**
- Run T002 and T003 together
- After both pass, run T004 and T005 together
- After both pass, run T006 and T007 together
- Finish with T008
