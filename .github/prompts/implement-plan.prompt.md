---
description: 'Implement development plan step-by-step with validation at each stage'
mode: agent
model: GPT-5-Codex (Preview) (copilot)
tools: ['edit', 'search', 'runCommands', 'pylance mcp server/pylanceDocuments', 'pylance mcp server/pylanceFileSyntaxErrors', 'pylance mcp server/pylanceImports', 'pylance mcp server/pylanceInstalledTopLevelModules', 'pylance mcp server/pylanceInvokeRefactoring', 'pylance mcp server/pylancePythonEnvironments', 'pylance mcp server/pylanceRunCodeSnippet', 'pylance mcp server/pylanceSettings', 'pylance mcp server/pylanceSyntaxErrors', 'pylance mcp server/pylanceUpdatePythonEnvironment', 'pylance mcp server/pylanceWorkspaceRoots', 'pylance mcp server/pylanceWorkspaceUserFiles', 'pylance mcp server/pylanceDocuments', 'pylance mcp server/pylanceFileSyntaxErrors', 'pylance mcp server/pylanceImports', 'pylance mcp server/pylanceInstalledTopLevelModules', 'pylance mcp server/pylanceInvokeRefactoring', 'pylance mcp server/pylancePythonEnvironments', 'pylance mcp server/pylanceRunCodeSnippet', 'pylance mcp server/pylanceSettings', 'pylance mcp server/pylanceSyntaxErrors', 'pylance mcp server/pylanceUpdatePythonEnvironment', 'pylance mcp server/pylanceWorkspaceRoots', 'pylance mcp server/pylanceWorkspaceUserFiles', 'pylance mcp server/pylanceDocuments', 'pylance mcp server/pylanceFileSyntaxErrors', 'pylance mcp server/pylanceImports', 'pylance mcp server/pylanceInstalledTopLevelModules', 'pylance mcp server/pylanceInvokeRefactoring', 'pylance mcp server/pylancePythonEnvironments', 'pylance mcp server/pylanceRunCodeSnippet', 'pylance mcp server/pylanceSettings', 'pylance mcp server/pylanceSyntaxErrors', 'pylance mcp server/pylanceUpdatePythonEnvironment', 'pylance mcp server/pylanceWorkspaceRoots', 'pylance mcp server/pylanceWorkspaceUserFiles', 'ref/*', 'perplexity/search', 'serena/*', 'pylance mcp server/*', 'usages', 'think', 'problems', 'changes', 'testFailure', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-vscode.vscode-websearchforcopilot/websearch', 'todos', 'runTests']
---

# Implement Development Plan

**Plan:** ${input:planFile:Path to plan file (e.g., tasks/my-plan.md)}

**Standards:** Automatically loaded from `.github/copilot-instructions.md` and area-specific `CLAUDE.md` files.

## Approach: One Phase at a Time

### Before Starting Phase

1. **Check memory** - Use serena to search for similar implementations
2. **Understand the phase:**
   - What does this accomplish?
   - How does it fit the larger plan?
   - What are dependencies from prior phases?
3. **Plan the approach** - Use think tool to reason about:
   - Simplest implementation (YAGNI)
   - Existing patterns to follow
   - Potential pitfalls

### During Implementation

- **Work incrementally** - implement → test → fix loop
- **Follow existing patterns** - use search/usages to find examples
- **Keep it simple** - no over-engineering
- **Use project standards:**
  - Comprehensive type hints
  - Async/await for I/O
  - Flowbite + Alpine.js only (UI)
  - 100% test pass rate

### After Completing Phase

**CRITICAL: Validate before proceeding**

Run these checks:
```
runTests → All pass?
problems → Zero errors?
```

Verify:
- [ ] All checklist items complete [x]
- [ ] No TODOs or temporary code
- [ ] Code follows standards (check CLAUDE.md files)
- [ ] Edge cases handled
- [ ] Tests cover new functionality

Document changes:
- Added: [files and summary]
- Modified: [files and summary]
- Removed: [files and reason]

Any problems, please try to resolve them or ask for help.

**Stop and ask for approval before next phase.**

## If Something Goes Wrong

- Tests fail → Use problems to debug, fix, retry
- Approach not working → Stop, use think, propose alternative
- Blocked → Explain what you tried, ask for help

## Remember

Quality matters more than speed. Implement carefully, test thoroughly, follow our standards. Take your time to get it right.