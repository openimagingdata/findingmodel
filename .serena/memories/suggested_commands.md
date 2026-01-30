# Development Commands

## Primary Commands (using Task)

### Testing
- `task test` - Run tests excluding external API calls (default)
- `task test-full` - Run full test suite including API integration tests

### Per-Package Testing
- `uv run --package oidm-common pytest packages/oidm-common`
- `uv run --package anatomic-locations pytest packages/anatomic-locations`
- `uv run --package findingmodel pytest packages/findingmodel -m "not callout"`
- `uv run --package oidm-maintenance pytest packages/oidm-maintenance`

### Code Quality
- `task check` - Run formatter, linter with auto-fix, and type checking
- `ruff check packages/` - Lint all packages
- `ruff format packages/` - Format all packages

### Workspace Management
- `uv sync --all-packages` - Sync all workspace packages

## CLI Commands

### findingmodel
```bash
uv run findingmodel --help                    # Show all commands
uv run findingmodel config                    # Show current configuration
uv run findingmodel index stats               # Show index statistics
uv run findingmodel fm-to-markdown model.json # Convert model to markdown
```

### findingmodel-ai
```bash
uv run findingmodel-ai --help                    # Show all commands
uv run findingmodel-ai make-info "finding"       # Generate finding info
uv run findingmodel-ai make-stub-model "finding" # Create basic model
uv run findingmodel-ai markdown-to-fm input.md   # Convert markdown to model
```

### anatomic-locations
```bash
uv run anatomic-locations --help                   # Show all commands
uv run anatomic-locations search "nasal"           # Search anatomic locations
uv run anatomic-locations stats                    # Show database statistics
```

### oidm-maintenance (oidm-maintain) - Maintainers Only
```bash
uv run oidm-maintain --help                        # Show all commands

# FindingModel database
uv run oidm-maintain findingmodel build --source DIR --output PATH
uv run oidm-maintain findingmodel publish --db-path PATH

# Anatomic database  
uv run oidm-maintain anatomic build --source FILE --output PATH
uv run oidm-maintain anatomic publish --db-path PATH
```

## Alternative Commands (using uv directly)

### Testing
- `uv run pytest -rs -m "not callout"` - Run tests without API calls
- `uv run pytest -rs` - Run all tests

### Code Quality
- `uv run ruff format packages/` - Format code
- `uv run ruff check --fix packages/` - Lint with auto-fix
- `uv run mypy packages/findingmodel/src` - Type checking

## Environment Setup
1. Copy `.env.sample` to `.env`
2. Add API keys:
   - `OPENAI_API_KEY=your_key_here` (required for AI features)
   - `ANTHROPIC_API_KEY=your_key_here` (optional)
   - `GOOGLE_API_KEY=your_key_here` (optional)
   - `TAVILY_API_KEY=your_key_here` (optional for enhanced search)
3. Install dependencies: `uv sync --all-packages`

## Git Commands
- `git status` - Check repository status
- `git diff` - View unstaged changes
- `git log --oneline -10` - View recent commits

## Package Publishing Order
When publishing to PyPI, packages must be published in dependency order:
1. oidm-common
2. anatomic-locations
3. findingmodel
4. (future) findingmodel-ai
