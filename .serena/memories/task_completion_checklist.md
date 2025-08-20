# Task Completion Checklist

When completing any coding task in this project, ALWAYS run these commands in order:

## 1. Code Quality Checks (REQUIRED)
Run these commands to ensure code quality:
```bash
task check
```
Or if Task is not available:
```bash
uv run ruff format        # Format code
uv run ruff check --fix   # Fix linting issues
uv run mypy src          # Type checking
```

## 2. Run Tests (REQUIRED)
Verify changes don't break existing functionality:
```bash
task test
```
Or:
```bash
uv run pytest -rs -m "not callout"
```

## 3. Test External APIs (if modified)
If you modified any tools that use OpenAI/Perplexity:
```bash
task test-full
```

## 4. Verify Specific Changes
If you modified specific functionality, run targeted tests:
```bash
task test -- test/test_<relevant_file>.py
```

## 5. Check Git Status
Review what files were changed:
```bash
git status
git diff
```

## Important Notes
- NEVER commit without running `task check` first
- Tests marked with `@pytest.mark.callout` require API keys
- If type checking fails, fix type hints before proceeding
- If linting fails after auto-fix, manual intervention needed
- All new code must have type annotations

## When to Skip Checks
Only skip these checks if:
- Making documentation-only changes
- User explicitly says to skip
- Emergency hotfix with user approval

Remember: The CLAUDE.md file states these commands should be run, making them essential for proper task completion.