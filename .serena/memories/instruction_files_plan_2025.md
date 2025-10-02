# Instruction File Alignment Plan (2025-09-28)

## Goal
Align Claude and Copilot instruction files with Serena memories so every assistant shares a single source of truth.

## Structure Blueprint
- **Section 1: Always Use Serena**
  - Note mandatory use of Serena MCP for lookups (`read_memory`) and documentation (`write_memory`).
  - Include quick command examples for querying memories and creating new ones.
- **Section 2: Project Snapshot**
  - Summarize key points from `project_overview` memory (purpose, stack, layout).
  - Link to additional Serena memories for deeper dives (e.g., `protocol_based_architecture_2025`).
- **Section 3: Coding Standards & Testing**
  - Reference `code_style_conventions`, `pydantic_ai_testing_best_practices`, and `suggested_commands` memories.
  - Highlight callout markers and API key handling rules.
- **Section 4: Workflow Expectations**
  - Remind agents to update Serena memories after noteworthy changes (new modules, command updates, refactors).
  - Capture checklists or runbooks in Serena (`task_completion_checklist`).

## Update Process
1. Before editing instruction files, review relevant Serena memories (`project_state_january_2025`, `documentation_corrections_2025`).
2. After updating instructions, create or amend Serena memories to reflect the canonical guidance (this memory can be updated with the latest references).
3. Encourage reviewers to verify Serena memories are referenced in new documentation PRs.
