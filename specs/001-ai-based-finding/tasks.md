# Tasks: AI-Based Finding Model Editor

**Feature Branch**: `001-ai-based-finding`
**Feature Directory**: `/Users/talkasab/Repos/findingmodel/specs/001-ai-based-finding/`

## Task List

### T001. [P] Project Setup and Linting
- Ensure all dependencies are installed (pydantic-ai, pytest, etc.)
- Confirm linting and formatting tools are configured
- Path: `/Users/talkasab/Repos/findingmodel/`

### T002. [P] Contract Test: Natural Language Editing
- Write unit tests for `edit_model_natural_language` as specified in `contracts/natural_language_api.md`
- Use `TestModel` to avoid real LLM calls
- Path: `src/findingmodel/tools/model_editor.py`, `test/test_model_editor.py`

### T003. [P] Contract Test: Markdown Editing
- Write unit tests for `edit_model_markdown` and `export_model_to_markdown` as specified in `contracts/markdown_api.md`
- Use `TestModel` to avoid real LLM calls
- Path: `src/findingmodel/tools/model_editor.py`, `test/test_model_editor.py`

### T004. Implement: Natural Language Editing
- Implement `edit_model_natural_language(model: FindingModelFull, command: str, source: str) -> EditResult`
- Use Pydantic AI agent pattern
- Path: `src/findingmodel/tools/model_editor.py`
- Depends on: T002

### T005. Implement: Markdown Editing
- Implement `export_model_to_markdown(model: FindingModelFull) -> str` and `edit_model_markdown(model: FindingModelFull, edited_markdown: str, source: str) -> EditResult`
- Path: `src/findingmodel/tools/model_editor.py`
- Depends on: T003

### T006. Demo Script: Natural Language Editing
- Create `notebooks/demo_edit_model_from_command.py` to demonstrate natural language editing
- Path: `notebooks/demo_edit_model_from_command.py`
- Depends on: T004

### T007. Demo Script: Markdown Editing
- Create `notebooks/demo_edit_model_from_markdown.py` to demonstrate markdown editing
- Path: `notebooks/demo_edit_model_from_markdown.py`
- Depends on: T005

### T008. [P] Polish: Documentation and Final Tests
- Update documentation and ensure all tests pass
- Path: `/Users/talkasab/Repos/findingmodel/`
- Depends on: T006, T007

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
