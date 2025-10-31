

# Implementation Plan: AI-Based Finding Model Editor

**Branch**: `001-ai-based-finding` | **Date**: 2025-09-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-ai-based-finding/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Create a simple, maintainable finding model editor with two modes:
- **Natural language command mode**: User types a command (e.g., "add severity attribute with mild, moderate, severe options"). The tool parses and applies the change directly to the model using existing structures.
- **Markdown mode**: The tool exports a simplified, human-editable Markdown version of the model's attributes list. The user edits this Markdown and sends it back. The tool compares the edited Markdown to the original and applies the changes to the model, using only the attributes section.

All changes must:
- Preserve all existing OIFM IDs (model, attribute, value)
- Only allow additions or non-semantic description edits
- Use existing `FindingModelFull` and related classes—no new data model abstractions
- Use the existing ID manager for new IDs

## Technical Context
## Technical Context
- **Language/Version**: Python 3.11+
- **Dependencies**: Pydantic AI, OpenAI API, existing findingmodel package
- **Configuration**: All configuration (including API keys) is managed via `config.py` (using environment variables as needed), not accessed directly from environment variables in the main logic.
- **Storage**: File-based models, MongoDB index (unchanged)
- **Testing**: pytest, async support, minimal mocking
- **Target**: library and demo scripts (with explicit demo scripts for both editing modes)
- **Constraints**: No new abstractions; only use existing model/data structures

## Constitution Check
All constitutional requirements are met:
- Medical domain integrity: Only safe, clinically valid changes allowed
- Protocol-based: Uses existing Pydantic AI agent pattern
- ID immutability: All IDs preserved, new IDs via existing manager
- Async: All API calls async
- Test-driven: All logic covered by tests

specs/[###-feature]/
ios/ or android/
src/findingmodel/
tests/

## Project Structure
All code goes in the existing `src/findingmodel/tools/` directory. No new submodules unless absolutely necessary.

**Demo scripts:**
- `scripts/edit_finding_model.py`: Demonstrates both natural language and markdown editing modes on a .fm.json file.

## Implementation Approach

1. **Natural Language Editing**
   - Function: `edit_model_natural_language(model: FindingModelFull, command: str) -> EditResult`
   - Uses a Pydantic AI agent to parse the command and apply the change directly to the model.
   - Only allows: add attribute, add values, edit descriptions (non-semantic), safe renames.
   - Uses existing ID manager for new IDs.
   - See contract: `contracts/natural_language_api.md`
   - **Demo script**: `scripts/edit_finding_model.py` — Loads a .fm.json file, applies a natural language command, and saves the result.

2. **Markdown Editing**
   - Function: `export_model_for_editing(model: FindingModelFull) -> str`
   - Exports a simplified Markdown version of the model's attributes list for human editing.
   - Function: `edit_model_markdown(model: FindingModelFull, edited_markdown: str) -> EditResult`
   - Compares the edited Markdown to the original, and applies changes to the model (attributes only).
   - See contract: `contracts/markdown_api.md`
   - **Demo script**: `scripts/edit_finding_model.py` — Loads a .fm.json file, exports/imports attributes as markdown, and saves the result.

## Testing
- Use pytest for all new functions
- Mock OpenAI/AI calls for unit tests

## Progress Tracking
- All requirements and constraints are satisfied by the above approach.
- No unnecessary abstractions or features will be added.
- All implementation will be as direct and maintainable as possible.
