---
name: docs-review
description: Review rubric for documentation — accuracy, discoverability, alignment across instruction files
---

# Documentation Review

Review documentation for accuracy, discoverability, and alignment with current code.

## Scope

- README.md, CHANGELOG.md
- CLAUDE.md, `.github/copilot-instructions.md`
- Serena memories
- Inline docstrings and comments
- Plan documents and task files

**Out of scope**: Code quality (use `/python-review`), test quality (use `/test-review`)

## Criteria

### 1. Accuracy
- Code examples match actual implementations?
- File paths and function names correct?
- CLI commands work as documented?
- No references to removed features or old architecture?

### 2. Discoverability
- Information placed where someone would look for it?
- No orphan documents without references?
- Cross-references between related docs?

### 3. Alignment
- CLAUDE.md and copilot-instructions.md consistent?
- Serena memories up to date with code?
- CHANGELOG entries follow existing format (concise, user-facing)?
- Instruction file hierarchy maintained: Serena → CLAUDE.md → copilot

### 4. Conciseness
- No duplicate information across documents?
- YAGNI for docs — only what's needed now?
- Brief and scannable?

## Documentation Hierarchy

1. **Serena memories** — canonical source of truth for conventions and architecture
2. **CLAUDE.md** — detailed guidance, references Serena
3. **copilot-instructions.md** — quick reference card
4. **README.md** — external users, installation, usage
5. **CHANGELOG.md** — user-facing changes per release

## Severity Guide

- **Critical**: Incorrect instructions that would break workflow, wrong CLI commands
- **Important**: Stale references, drift between instruction files, missing CHANGELOG entry
- **Minor**: Could be more concise, formatting improvement, missing cross-reference
