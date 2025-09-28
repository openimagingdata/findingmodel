# AI Assistant Usage Guidelines (2025-09-28)

## Core Principles
- Always use the Serena MCP server for repository context and note-taking. Run lookups with `find_symbol`, `search_for_pattern`, or `read_memory` before scanning files manually.
- Document new insights by writing Serena memories (`write_memory`) instead of in-editor notes to keep a centralized knowledge base.

## Instruction File Strategy
- Keep `.github/copilot-instructions.md` concise (1-2 screens) with direct references to Serena memories for deeper context.
- Maintain `CLAUDE.md` as the detailed project companion, mirroring Copilot instructions but expanding on workflows, shortcuts, and AI usage patterns.
- When instructions change, update both files and log a Serena memory summarizing the change.

## Best Practices (Sources: GitHub Copilot Custom Instructions Guidance, Claude Code Memory Docs)
- Lead with project overview, tech stack, testing commands, and coding conventions.
- Use numbered sections and short bullet lists for scannability.
- Explicitly state required tools (uv, task, pytest) and note when Taskfile commands are canonical.
- Emphasize preventing accidental external API calls in tests (`models.ALLOW_MODEL_REQUESTS = False`).

## Required Behaviors for Agents
1. Before answering architecture or standards questions, query Serena memories (`project_overview`, `code_style_conventions`, `suggested_commands`).
2. When creating new utilities or tests, cross-check naming and patterns against Serena memories and update them if conventions evolve.
3. For new features, create a Serena memory summarizing rationale, interfaces, and follow-up work so future instructions can reference it.
