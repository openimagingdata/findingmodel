# Code Style and Conventions

## Python Version
- Python 3.11+ required
- Target version explicitly set in ruff config

## Code Style
- **Line Length**: 120 characters max
- **Formatting**: Handled by ruff format
- **Linting**: ruff with extensive rule sets (bugbear, comprehensions, isort, type annotations, simplify)
- **Type Hints**: Strict type checking with mypy
  - All functions have type annotations
  - Uses `Annotated` types for validation
  - Pydantic models for data structures
  - Custom type aliases (e.g., `NameString`, `AttributeId`)

## Code Patterns
- **Pydantic Models**: BaseModel for all data structures
- **Async/Await**: Used throughout for API calls and database operations
- **Field Validation**: Pydantic Field() with validators
- **Docstrings**: Triple quotes, brief descriptions on classes and key methods
- **Imports**: Organized with isort, type imports separated
- **Constants**: UPPER_SNAKE_CASE (e.g., `ID_LENGTH`, `ATTRIBUTES_FIELD_DESCRIPTION`)

## Testing
- pytest with asyncio support
- Test markers: `@pytest.mark.callout` for external API tests
- Test files named `test_*.py`
- Fixtures in conftest.py
- Test data in `test/data/` directory

## Error Handling
- Custom exception classes (e.g., `ConfigurationError`)
- Validation through Pydantic models
- Type checking enforced

## Naming Conventions
- **Classes**: PascalCase (e.g., `FindingModelBase`)
- **Functions/Methods**: snake_case (e.g., `create_info_from_name`)
- **Variables**: snake_case
- **Type Aliases**: PascalCase (e.g., `AttributeId`)
- **Constants**: UPPER_SNAKE_CASE

## File Organization
- One main concept per file
- Related utilities grouped in tools/ directory
- Test files mirror source structure