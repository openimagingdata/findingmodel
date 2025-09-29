# AI Assistant Usage Guidelines (2025-10-06)

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

## Testing Guidelines (Updated 2025-10-06)
- **Unit tests**: Use `TestModel()` or `FunctionModel()` from pydantic_ai for deterministic mocked responses. Never manage global flags.
- **Integration tests**: Mark with `@pytest.mark.callout` to allow real API calls during `task test-full`.
- **Antipattern removed**: Do NOT set `models.ALLOW_MODEL_REQUESTS = False` at module level or in fixtures. This is a development safety guard, not a test isolation mechanism.

## Model Editing Feature (v0.4)
- **Dual-mode editor**: Supports both natural language commands and markdown-based editing via CLI
- **ID preservation**: All existing OIFM IDs are maintained during edits
- **Placeholder workflow**: New content uses temporary placeholder IDs; promote to stable IDs before publishing using provided utilities
- **Clinical validation**: AI-powered medical domain accuracy checking
- **Safe edits only**: Only additions and semantic-preserving modifications allowed

## Required Behaviors for Agents
1. Before answering architecture or standards questions, query Serena memories (`project_overview`, `code_style_conventions`, `suggested_commands`).
2. When creating new utilities or tests, cross-check naming and patterns against Serena memories and update them if conventions evolve.
3. For new features, create a Serena memory summarizing rationale, interfaces, and follow-up work so future instructions can reference it.
