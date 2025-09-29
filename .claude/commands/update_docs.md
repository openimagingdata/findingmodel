# Update Documentation and Best Practices

Analyze the recent work completed in this project and update all relevant documentation. Use Serena's memory (via serena MCP commands) to understand what has changed and what knowledge should be preserved. Follow these steps:

## 1. ANALYZE RECENT CHANGES

- Review git history for commits from the last ${ARGUMENTS:-7} days
- Use git and Serena MCP tools to examine modified files and understand semantic changes
- Identify new patterns, APIs, components, or architectural changes

## 2. SCAN EXISTING DOCUMENTATION

Review all documentation in these locations:

- From project root: .github/copilot-instructions.md, README.md, CLAUDE.md, CHANGELOG.md
- Serena MCP tools: Review stored project knowledge and decisions (serena's read_memory)

## 3. UPDATE DOCUMENTATION SYSTEMATICALLY

### Update README.md

- New features or capabilities added
- Installation/setup changes
- Updated usage examples reflecting current implementation
- New dependencies or requirements

### Update CLAUDE.md (and sub-directory CLAUDE.md) and copilot-instructions.md

- New coding patterns discovered during implementation
- Updated best practices based on what worked well
- Anti-patterns to avoid based on issues encountered
- New project-specific conventions established

### Update CHANGELOG.md

- Use the standard CHANGELOG.md format as seen at https://keepachangelog.com/en/1.0.0/
- Add entry for recent changes with date--don't forget to use bash's `date`, since you don't know it
- Group by: Added, Changed, Fixed, Deprecated, Removed, Security
- Reference relevant commits

### Update API Documentation

- New endpoints, methods, or interfaces
- Changed parameters or return types
- Deprecated features with migration guides
- Updated examples using actual code from tests

## 4. SYNCHRONIZE WITH SERENA'S MEMORY

Ask Serena to:

- Store important architectural decisions in memory (serena MCP's `write_memory`)
- Update technology understanding based on new patterns
- Record rationale for significant changes
- Save learned optimizations and performance improvements

Use these Serena commands:

- `write_memory`: Save key decisions and patterns
- `read_memory`: Verify stored knowledge is accurate

## 6. VALIDATE DOCUMENTATION

For each updated document:

- Ensure code examples are valid and run
- Verify all links work
- Check that instructions are reproducible
- Confirm consistency across all docs
- Test that newcomers could follow the guides

## 7. UPDATE TEAM KNOWLEDGE

Create a summary of changes for the team:

- What's new that everyone should know
- Breaking changes or migration needs
- New tools or workflows introduced
- Performance improvements achieved
- Lessons learned worth sharing

## IMPORTANT RULES

- Use bash's `date` command to find the current date--you won't know it otherwise (it's not January 2025)
- NEVER remove existing valid documentation without explicit approval
- ALWAYS preserve historical context and rationale
- USE Serena's semantic understanding to ensure accuracy
- MAINTAIN consistent tone and formatting across all docs
- INCLUDE practical examples from actual implementation
- TEST all code snippets and commands before documenting
- CITE relevant commits or PRs when documenting changes

## SERENA INTEGRATION NOTES

- Use the Serena  memories by calling serena's `read_memory` tool
- Leverage Serena's understanding of code relationships
- Use Serena's symbol-level comprehension for accurate API docs
- Let Serena help identify what's truly important to document

Remember: Good documentation is a living artifact that evolves with the code. Use both git history and Serena's semantic memory to create documentation that captures not just WHAT changed, but WHY it changed and HOW to use it effectively.
