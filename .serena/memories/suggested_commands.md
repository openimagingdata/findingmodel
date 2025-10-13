# Development Commands

## Primary Commands (using Task)
These are the preferred commands for development:

### Testing
- `task test` - Run tests excluding external API calls (default for development)
- `task test-full` - Run full test suite including API integration tests
- `task test -- test/test_findingmodel.py::TestClass::test_method` - Run specific test

### Code Quality
- `task check` - Run formatter, linter with auto-fix, and type checking
- `task quiet` - Run all checks quietly (minimal output)

### Building
- `task build` - Build the package distribution

## Alternative Commands (using uv directly)
If Task is not available:

### Testing
- `uv run pytest -rs -m "not callout"` - Run tests without API calls
- `uv run pytest -rs` - Run all tests

### Code Quality
- `uv run ruff format` - Format code
- `uv run ruff check --fix` - Lint with auto-fix
- `uv run mypy src` - Type checking

### Building
- `uv build` - Build package

## CLI Usage
The package provides a CLI accessible via:
- `python -m findingmodel` - Show available commands
- `python -m findingmodel config` - Show current configuration
- `python -m findingmodel make-info "finding name"` - Generate finding info
- `python -m findingmodel make-stub-model "finding name"` - Create basic model
- `python -m findingmodel markdown-to-fm input.md` - Convert markdown to model
- `python -m findingmodel fm-to-markdown model.json` - Convert model to markdown
- `python -m findingmodel edit-model model.json` - AI-powered model editor (natural language and markdown)
- `python -m findingmodel validate-model model.json` - Validate model integrity and clinical accuracy

### Anatomic Location Management (2025-10-13)
- `python -m findingmodel anatomic build` - Build anatomic location database from default URL
- `python -m findingmodel anatomic build --source FILE|URL` - Build from custom source
- `python -m findingmodel anatomic build --force` - Overwrite existing database
- `python -m findingmodel anatomic build --output PATH` - Custom output path
- `python -m findingmodel anatomic validate --source FILE|URL` - Validate data without building
- `python -m findingmodel anatomic stats` - Show database statistics
- `python -m findingmodel anatomic stats --db-path PATH` - Stats for custom database path

## Git Commands (Darwin/macOS)
- `git status` - Check repository status
- `git diff` - View unstaged changes
- `git add .` - Stage all changes
- `git commit -m "message"` - Commit staged changes
- `git push` - Push to remote repository
- `git log --oneline -10` - View recent commits

## Environment Setup
1. Copy `.env.sample` to `.env`
2. Add API keys:
   - `OPENAI_API_KEY=your_key_here`
   - `PERPLEXITY_API_KEY=your_key_here` (optional for enhanced descriptions)
3. Install dependencies: `uv sync`

## Virtual Environment
- uv manages virtual environments automatically
- No need to manually activate/deactivate